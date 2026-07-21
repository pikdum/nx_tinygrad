defmodule NxTinygrad.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/pikdum/nx_tinygrad"

  def project do
    [
      app: :nx_tinygrad,
      version: @version,
      elixir: "~> 1.18",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "An Elixir Nx compiler and tensor backend using tinygrad.",
      package: package(),
      name: "NxTinygrad",
      source_url: @source_url,
      docs: docs(),
      test_paths: test_paths(Mix.env())
    ]
  end

  def application do
    [
      extra_applications: [:crypto, :logger],
      mod: {NxTinygrad.Application, []}
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # GPU tests live under test/gpu and are only run explicitly.
  defp test_paths(_), do: ["test"]

  defp deps do
    [
      # Keep this loose enough to admit the nx that Bumblebee requires (0.12.x as
      # of Bumblebee 0.7). Hard-pinning to a single 0.x line (e.g. "~> 0.13.0")
      # makes the examples/*.exs Mix.install graphs unresolvable. `mix test` still
      # resolves the newest permitted nx; the Bumblebee examples pick 0.12.
      {:nx, "~> 0.12"},
      {:telemetry, "~> 1.2"},
      {:rustler, "~> 0.38.0"},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:benchee, "~> 1.3", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib priv/worker/*.py native/nx_tinygrad_ref/src native/nx_tinygrad_ref/Cargo.toml
           native/nx_tinygrad_ref/README.md Cargo.toml Cargo.lock mix.exs README.md LICENSE
           CHANGELOG.md STATUS.md SPEC.md docs examples bench)
    ]
  end

  defp docs do
    [
      main: "readme",
      source_ref: "v#{@version}",
      extras:
        [
          "README.md",
          "STATUS.md",
          "SPEC.md",
          "CHANGELOG.md",
          "LICENSE",
          {"bench/README.md", [filename: "benchmarks"]}
        ] ++
          Path.wildcard("docs/*.md")
    ]
  end
end
