{
  description = "nx_tinygrad — an Elixir Nx compiler and tensor backend using tinygrad (AMD, native KFD + LLVM, no ROCm)";

  # nixos-unstable: Elixir 1.20 / OTP 29 (proper releases) via beam29Packages and
  # a ROCm-free tinygrad. We don't depend on ROCm, so there's no heavy closure to
  # cache-match against — the only thing that rebuilds is pure-Python tinygrad.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      beam = pkgs.beam29Packages;
      elixir = beam.elixir_1_20;
      erlang = beam.erlang;

      # tinygrad pinned to a post-0.13.0 master rev: the 0.13.0 release
      # miscompiles f16 convs on the AMD LLVM renderer (NaN / garbage — SD in
      # f16 rendered black); this rev is verified clean (worker_tests, mix
      # test, GPU suite, SD parity). Still ROCm-free (rocmSupport stays off).
      # nixpkgs' patches target the 0.13.0 tree, so we redo the library-path
      # pinning with substitutions matching the master layout; master also
      # honors LIBC_PATH/LLVM_PATH env overrides at runtime.
      tinygradPinned = pkgs.python3Packages.tinygrad.overrideAttrs (old: {
        version = "0.13.0-unstable-2026-07-19";
        src = pkgs.fetchFromGitHub {
          owner = "tinygrad";
          repo = "tinygrad";
          rev = "7b05caf5c5c58a54a82bf1a987a6b7ba5a3f2aa4";
          hash = "sha256-u4oRiCuK9QQS8Drc044HnDAqN2DmFtpQz9Wt6WSX2Gk=";
        };
        patches = [ ];
        postPatch = ''
          substituteInPlace tinygrad/runtime/autogen/libc.py \
            --replace-fail \
              "dll = c.DLL('libc', 'c', use_errno=True)" \
              "dll = c.DLL('libc', '${pkgs.lib.getLib pkgs.stdenv.cc.libc}/lib/libc.so.6', use_errno=True)"
          substituteInPlace tinygrad/runtime/autogen/llvm.py \
            --replace-fail \
              "else ['LLVM', 'LLVM-21'" \
              "else ['${pkgs.lib.getLib pkgs.llvmPackages.llvm}/lib/libLLVM.so', 'LLVM-21'"
          substituteInPlace tinygrad/runtime/autogen/libclang.py \
            --replace-fail \
              "else ['clang', 'clang-21'" \
              "else ['${pkgs.lib.getLib pkgs.llvmPackages.libclang}/lib/libclang.so', 'clang-21'"
          # The CPU renderer shells out to a C compiler. Its default reads
          # $CC, which dev shells commonly set to a gcc wrapper that rejects
          # clang-style --target flags — pin nix clang under our own var.
          substituteInPlace tinygrad/runtime/support/compiler_cpu.py \
            --replace-fail \
              "getenv(\"CC\", 'clang')" \
              "getenv(\"NX_TINYGRAD_CC\", '${pkgs.lib.getExe pkgs.llvmPackages.clang-unwrapped}')"
        '';
        # Master's test suite is not our gate; the flake's python-tests check
        # runs the worker suite instead. buildPythonPackage runs pytest in the
        # installCheck phase, so that is the flag to clear on an override.
        doCheck = false;
        doInstallCheck = false;
      });

      # Worker Python environment. Plain nixpkgs tinygrad derivation (with the
      # src/pin overrides above), so the closure contains NO ROCm/HIP/comgr.
      # The AMD backend runs through tinygrad's native KFD driver and compiles
      # kernels with the pinned libLLVM through DEV=KFD+AMD:LLVM.
      workerPython = pkgs.python3.withPackages (ps: [
        tinygradPinned
        ps.numpy
      ]);

      # Same interpreter plus test tooling, for `nix flake check` python tests.
      workerPythonTest = pkgs.python3.withPackages (ps: [
        tinygradPinned
        ps.numpy
        ps.pytest
      ]);

      # Packaged worker: a wrapper that runs priv/worker/main.py with the pinned
      # interpreter. main.py adds its own directory to sys.path, so no PYTHONPATH
      # juggling is required.
      nx-tinygrad-worker = pkgs.stdenv.mkDerivation {
        pname = "nx-tinygrad-worker";
        version = "0.1.0";
        src = ./priv/worker;
        nativeBuildInputs = [ pkgs.makeWrapper ];
        dontConfigure = true;
        dontBuild = true;
        installPhase = ''
          runHook preInstall
          mkdir -p "$out/lib/nx_tinygrad_worker" "$out/bin"
          cp -r ./* "$out/lib/nx_tinygrad_worker/"
          makeWrapper ${workerPython}/bin/python "$out/bin/nx_tinygrad_worker" \
            --add-flags "$out/lib/nx_tinygrad_worker/main.py" \
            --set PYTHONUNBUFFERED 1
          runHook postInstall
        '';
      };

      # Substrings that must never appear in the default worker runtime closure.
      rocmDenylist = [
        "rocm"
        "hipcc"
        "comgr"
        "libamdhip64"
        "libhsa-runtime64"
        "rocblas"
        "miopen"
      ];
      rocmPattern = builtins.concatStringsSep "|" rocmDenylist;
    in
    {
      packages.${system} = {
        default = nx-tinygrad-worker;
        inherit nx-tinygrad-worker;
        python = workerPython;
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          elixir
          erlang
          # rebar3 3.27 fails to build from source under OTP 29-rc; use the one
          # built against OTP 28 (an escript, loadable under OTP 29) to compile
          # rebar deps such as :telemetry.
          pkgs.rebar3
          # Shell `python`/`pytest` include pytest for running worker tests; the
          # actual worker interpreter (NX_TINYGRAD_PYTHON) stays lean.
          workerPythonTest
          pkgs.cargo
          pkgs.rustc
          pkgs.rustfmt
          pkgs.clippy
          pkgs.gcc
          pkgs.git
        ];

        env = {
          # The Elixir Worker spawns this interpreter running priv/worker/main.py.
          NX_TINYGRAD_PYTHON = "${workerPython}/bin/python";
          # Use nixpkgs' rebar3 rather than mix's downloaded one (which is broken
          # under Nix), so rebar deps like :telemetry compile.
          MIX_REBAR3 = "${pkgs.rebar3}/bin/rebar3";
          # Keep hex/rebar/mix state inside the project during development.
          MIX_HOME = ".nix-mix";
          HEX_HOME = ".nix-hex";
        };

        shellHook = ''
          export PATH="$MIX_HOME/bin:$HEX_HOME/bin:$PATH"
          echo "nx_tinygrad devshell"
          echo "  elixir ${elixir.version} / erlang ${erlang.version}"
          echo "  worker python: ${workerPython}/bin/python (tinygrad ${tinygradPinned.version}, no ROCm)"
        '';
      };

      checks.${system} = {
        # Fail if any ROCm/HIP/comgr path leaks into the worker runtime closure.
        no-rocm-closure =
          pkgs.runCommand "no-rocm-closure"
            {
              closure = pkgs.closureInfo { rootPaths = [ nx-tinygrad-worker ]; };
            }
            ''
              echo "Scanning worker closure for ROCm/HIP/comgr paths..."
              if grep -Ei '${rocmPattern}' "$closure/store-paths"; then
                echo "ERROR: ROCm-related paths present in the default worker closure." >&2
                exit 1
              fi
              echo "OK: no ROCm/HIP/comgr in worker closure."
              touch "$out"
            '';

        # Python worker unit tests (CPU, no network).
        python-tests =
          pkgs.runCommand "python-tests"
            {
              nativeBuildInputs = [ workerPythonTest ];
            }
            ''
              cp -r ${./priv/worker} worker
              cp -r ${./worker_tests} worker_tests
              chmod -R u+w worker worker_tests
              export PYTHONPATH="$PWD/worker"
              export HOME="$TMPDIR"
              export DEV=CPU
              echo "Running worker pytest suite (DEV=CPU)..."
              python -m pytest -q worker_tests | tee "$TMPDIR/out.txt"
              cp "$TMPDIR/out.txt" "$out"
            '';
      };

      formatter.${system} = pkgs.nixfmt;
    };
}
