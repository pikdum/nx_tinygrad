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

  @doc """
  JIT-compile `fun` with the ExTinygrad compiler.

  Equivalent to `Nx.Defn.jit(fun, compiler: ExTinygrad.Compiler)` with the given
  options merged in.
  """
  def jit(fun, opts \\ []), do: Nx.Defn.jit(fun, put_compiler(opts))

  @doc "Like `jit/2` but immediately applies `args`."
  def jit_apply(fun, args, opts \\ []) when is_list(args) do
    Nx.Defn.jit_apply(fun, args, put_compiler(opts))
  end

  @doc "Return the worker's `device_info` map."
  def device_info(opts \\ []) do
    worker = Keyword.get(opts, :worker, :default)
    {:ok, info, []} = ExTinygrad.Worker.request(worker, "device_info", %{})
    info
  end

  @doc "Return the worker's statistics map."
  def worker_stats(opts \\ []) do
    worker = Keyword.get(opts, :worker, :default)
    {:ok, stats, []} = ExTinygrad.Worker.request(worker, "stats", %{})
    stats
  end

  @doc "Block until all queued device work on the worker completes."
  def synchronize(opts \\ []) do
    worker = Keyword.get(opts, :worker, :default)
    {:ok, %{}, []} = ExTinygrad.Worker.request(worker, "synchronize", %{})
    :ok
  end

  defp put_compiler(opts), do: Keyword.put(opts, :compiler, ExTinygrad.Compiler)
end
