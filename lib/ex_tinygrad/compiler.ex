defmodule ExTinygrad.Compiler do
  @moduledoc """
  `Nx.Defn.Compiler` that lowers an entire defn expression to the graph IR, sends
  it once to a Python worker, and executes it through tinygrad.

  For M2 this operates in host mode: inputs are shipped as blobs and outputs come
  back as `Nx.BinaryBackend` tensors, validated against `Nx.BinaryBackend`. M3
  adds caching + TinyJit capture/replay; M4 adds device-resident tensors.
  """
  @behaviour Nx.Defn.Compiler

  alias ExTinygrad.{Config, Graph, Lowering, Worker}
  alias Nx.Defn.Composite

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    __compile__(key, vars, fun, opts).(args_list)
  end

  @impl true
  def __compile__(_key, vars, fun, opts) do
    {roots, output_container} = precompile(fun, vars)
    graph = Lowering.to_graph(roots)

    worker = Keyword.get(opts, :worker, :default)
    compile_timeout = Keyword.get(opts, :compile_timeout, Config.compile_timeout())
    execute_timeout = Keyword.get(opts, :execute_timeout, Config.execute_timeout())

    {:ok, %{"executable_id" => executable_id}, []} =
      request(worker, "compile", %{"graph" => Graph.to_wire(graph)}, Graph.blobs(graph),
        timeout: compile_timeout
      )

    ctx = %{
      worker: worker,
      executable_id: executable_id,
      graph: graph,
      output_container: output_container,
      execute_timeout: execute_timeout
    }

    fn args_list -> Enum.map(args_list, &run_one(&1, ctx)) end
  end

  @impl true
  def __partitions_options__(opts), do: [opts]

  @impl true
  def __to_backend__(_opts), do: {Nx.BinaryBackend, []}

  # -- compilation --------------------------------------------------------

  # Call fun with the parameter templates; collect the (devectorized) output
  # expression leaves in order, and keep the container structure for later
  # reconstruction.
  defp precompile(fun, vars) do
    {output_container, roots_rev} =
      vars
      |> fun.()
      |> Composite.traverse([], fn tensor, acc ->
        devec = Nx.devectorize(tensor)
        {devec, [devec | acc]}
      end)

    {Enum.reverse(roots_rev), output_container}
  end

  # -- runtime ------------------------------------------------------------

  defp run_one(params, ctx) do
    {inputs, blobs} = build_inputs(params, ctx.graph)

    {:ok, %{"outputs" => output_specs}, output_blobs} =
      request(
        ctx.worker,
        "execute",
        %{"executable_id" => ctx.executable_id, "inputs" => inputs, "output" => "host"},
        blobs,
        timeout: ctx.execute_timeout
      )

    tensors = decode_outputs(output_specs, output_blobs)
    reconstruct(ctx.output_container, tensors)
  end

  defp build_inputs(params, graph) do
    {inputs, blobs, _count} =
      Enum.reduce(graph.inputs, {[], [], 0}, fn input, {inputs, blobs, k} ->
        tensor = params |> Enum.fetch!(input["index"]) |> apply([])
        binary = Nx.to_binary(tensor)

        spec = %{
          "kind" => "blob",
          "blob_index" => k,
          "shape" => input["shape"],
          "dtype" => input["dtype"]
        }

        {[spec | inputs], [binary | blobs], k + 1}
      end)

    {Enum.reverse(inputs), Enum.reverse(blobs)}
  end

  defp decode_outputs(output_specs, output_blobs) do
    Enum.zip_with(output_specs, output_blobs, fn spec, blob ->
      type = ExTinygrad.Dtype.to_nx!(spec["dtype"])

      blob
      |> Nx.from_binary(type)
      |> Nx.reshape(List.to_tuple(spec["shape"]))
    end)
  end

  # Rebuild the output container, swapping each expr leaf's data for the computed
  # tensor's data (preserving names/shape metadata from the template).
  defp reconstruct(output_container, tensors) do
    {result, []} =
      Composite.traverse(output_container, tensors, fn template, [tensor | rest] ->
        {%{template | data: tensor.data}, rest}
      end)

    result
  end

  defp request(worker, command, args, blobs, opts) do
    case Worker.request(worker, command, args, blobs, opts) do
      {:ok, _result, _blobs} = ok -> ok
      {:error, exception} -> raise exception
    end
  end
end
