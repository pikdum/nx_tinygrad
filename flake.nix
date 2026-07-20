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
      elixir = beam.elixir_1_20; # 1.20.0-rc.4
      erlang = beam.erlang; # 29.0-rc3

      # Worker Python environment. We deliberately use plain nixpkgs tinygrad
      # (rocmSupport defaults to false), so the closure contains NO ROCm/HIP/
      # comgr. The AMD backend runs through tinygrad's native KFD driver and
      # compiles kernels with libLLVM (AMD_LLVM=1), whose path nixpkgs already
      # patches into tinygrad unconditionally.
      workerPython = pkgs.python3.withPackages (ps: [
        ps.tinygrad
        ps.numpy
      ]);

      # Same interpreter plus test tooling, for `nix flake check` python tests.
      workerPythonTest = pkgs.python3.withPackages (ps: [
        ps.tinygrad
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
          echo "  worker python: ${workerPython}/bin/python (tinygrad ${pkgs.python3Packages.tinygrad.version}, no ROCm)"
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

      formatter.${system} = pkgs.nixfmt-rfc-style;
    };
}
