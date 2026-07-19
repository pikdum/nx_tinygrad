defmodule ExTinygrad do
  @moduledoc """
  An Elixir Nx compiler and tensor backend that uses [tinygrad](https://tinygrad.org)
  as the optimizing compiler and accelerator runtime.

  The whole `Nx.Defn` expression is transferred to tinygrad as a single graph,
  compiled and captured once with `TinyJit`, then replayed on subsequent calls.

  ## Usage

      predict = ExTinygrad.jit(&Model.predict/3, device: "KFD+AMD:LLVM")
      result = predict.(x, weights, bias)

  Equivalent direct Nx usage:

      predict =
        Nx.Defn.jit(&Model.predict/3,
          compiler: ExTinygrad.Compiler,
          device: "KFD+AMD:LLVM"
        )

  See `ExTinygrad.Compiler` for the compiler and `ExTinygrad.Backend` for the
  tensor backend.
  """

  @doc "Library version."
  def version, do: unquote(Mix.Project.config()[:version])

  # The public API (jit/2, compile/3, device_info/1, ...) is implemented as
  # milestones land; see ExTinygrad.Compiler.
end
