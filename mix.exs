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
      description: "An Elixir Nx compiler and tensor backend using tinygrad (AMD, native KFD + LLVM).",
      package: package(),
      name: "NxTinygrad",
      source_url: @source_url,
      docs: docs(),
      test_paths: test_paths(Mix.env())
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {NxTinygrad.Application, []}
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # GPU tests live under test/gpu and are only run explicitly.
  defp test_paths(_), do: ["test"]

  defp deps do
    [
      {:nx, "~> 0.9"},
      {:telemetry, "~> 1.2"},
      {:rustler, "~> 0.36"},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:benchee, "~> 1.3", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib priv/worker native mix.exs README.md LICENSE CHANGELOG.md SPEC.md)
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md", "docs/architecture.md", "docs/protocol.md", "docs/amd-nixos.md"]
    ]
  end
end
