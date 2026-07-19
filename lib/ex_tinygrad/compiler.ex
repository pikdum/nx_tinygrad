defmodule ExTinygrad.Compiler do
  @moduledoc """
  `Nx.Defn.Compiler` that lowers an entire defn expression to the graph IR, sends
  it once to a Python worker, and executes it through tinygrad (one execute RPC
  per invocation).

  The graph is compiled and captured once (cached by `ExTinygrad.ExecutableCache`)
  and replayed thereafter. By default outputs are device-resident
  (`ExTinygrad.Backend`); pass `output: :host` for `Nx.BinaryBackend` results.
  Device-resident inputs from the same worker are passed by handle rather than
  re-uploaded.

  Emits `[:ex_tinygrad, :compile | :execute, :start | :stop]` telemetry spans.
  """
  @behaviour Nx.Defn.Compiler
  require Logger

  alias ExTinygrad.{
    Backend,
    Config,
    Dtype,
    ExecutableCache,
    Graph,
    GraphCacheKey,
    Lowering,
    OutputContainer,
    Worker
  }

  alias Nx.Defn.Composite

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    __compile__(key, vars, fun, opts).(args_list)
  end

  @impl true
  def __compile__(_key, vars, fun, opts) do
    {roots, output_container} = precompile(fun, vars)

    if roots == [] do
      # No tensor outputs (e.g. an empty tuple): nothing to compile or execute.
      fn args_list -> Enum.map(args_list, fn _ -> output_container end) end
    else
      compile_graph(roots, output_container, opts)
    end
  end

  defp compile_graph(roots, output_container, opts) do
    graph = Lowering.to_graph(roots)

    worker = resolve_worker(opts)
    execute_timeout = Keyword.get(opts, :execute_timeout, Config.execute_timeout())
    executable_id = ensure_compiled(worker, graph, opts)

    ctx = %{
      worker: worker,
      executable_id: executable_id,
      graph: graph,
      output_container: output_container,
      execute_timeout: execute_timeout,
      output: Keyword.get(opts, :output, :device)
    }

    fn args_list -> Enum.map(args_list, &run_one(&1, ctx)) end
  end

  @impl true
  def __partitions_options__(opts), do: [opts]

  @impl true
  def __to_backend__(opts), do: {Backend, Keyword.take(opts, [:worker])}

  # An explicit `:worker` wins; otherwise `:device` routes to (and starts) a
  # worker for that device; otherwise the :default worker.
  defp resolve_worker(opts) do
    cond do
      worker = Keyword.get(opts, :worker) -> worker
      device = Keyword.get(opts, :device) -> ExTinygrad.WorkerSupervisor.worker_for_device(device)
      true -> :default
    end
  end

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

  # Compile the graph in the worker, reusing a cached executable when the graph
  # matches and the worker is still on the same generation.
  defp ensure_compiled(worker, graph, opts) do
    if Keyword.get(opts, :cache, Config.cache?()) do
      info = Worker.info(worker)
      generation = Worker.generation(worker)

      key =
        GraphCacheKey.compute(graph,
          device: info["device"],
          tinygrad_commit: info["tinygrad_version"]
        )

      case ExecutableCache.get(key) do
        %{generation: ^generation, executable_id: id} ->
          id

        _ ->
          id = compile_worker(worker, graph, opts)
          ExecutableCache.put(key, %{generation: generation, executable_id: id})
          id
      end
    else
      compile_worker(worker, graph, opts)
    end
  end

  defp compile_worker(worker, graph, opts) do
    :telemetry.span([:ex_tinygrad, :compile], %{worker: worker, node_count: length(graph.nodes)}, fn ->
      args = %{
        "graph" => Graph.to_wire(graph),
        "validate_capture" => Keyword.get(opts, :validate_capture, true)
      }

      {:ok, %{"executable_id" => id} = result, []} =
        request(worker, "compile", args, Graph.blobs(graph),
          timeout: Keyword.get(opts, :compile_timeout, Config.compile_timeout())
        )

      Logger.debug(
        "ex_tinygrad compiled executable #{id} " <>
          "(#{result["kernel_count"]} kernels, #{Float.round(result["compile_ms"] || 0.0, 2)} ms)"
      )

      {id, %{worker: worker, executable_id: id, kernel_count: result["kernel_count"]}}
    end)
  end

  # -- runtime ------------------------------------------------------------

  defp run_one(params, ctx) do
    metadata = %{worker: ctx.worker, executable_id: ctx.executable_id, output: ctx.output}

    :telemetry.span([:ex_tinygrad, :execute], metadata, fn ->
      {inputs, blobs} = build_inputs(params, ctx.graph, ctx.worker)
      output_mode = Atom.to_string(ctx.output)

      {:ok, %{"outputs" => output_specs}, output_blobs} =
        request(
          ctx.worker,
          "execute",
          %{"executable_id" => ctx.executable_id, "inputs" => inputs, "output" => output_mode},
          blobs,
          timeout: ctx.execute_timeout
        )

      tensors = decode_outputs(ctx.output, output_specs, output_blobs, ctx.worker)
      {OutputContainer.reconstruct(ctx.output_container, tensors), metadata}
    end)
  end

  # Build execute inputs. A runtime tensor already resident in this worker (same
  # generation) is passed by handle; everything else is shipped as a blob.
  defp build_inputs(params, graph, worker) do
    generation = Worker.generation(worker)

    {inputs, blobs, _k} =
      Enum.reduce(graph.inputs, {[], [], 0}, fn input, {inputs, blobs, k} ->
        tensor = params |> Enum.fetch!(input["index"]) |> apply([])

        case tensor.data do
          %Backend{worker: ^worker, generation: ^generation} = b ->
            {[%{"kind" => "handle", "id" => Backend.handle(b)} | inputs], blobs, k}

          %Backend{worker: ^worker, generation: stale} ->
            raise ExTinygrad.StaleTensorError,
              worker: worker,
              tensor_generation: stale,
              worker_generation: generation

          _ ->
            spec = %{
              "kind" => "blob",
              "blob_index" => k,
              "shape" => input["shape"],
              "dtype" => input["dtype"]
            }

            {[spec | inputs], [Nx.to_binary(tensor) | blobs], k + 1}
        end
      end)

    {Enum.reverse(inputs), Enum.reverse(blobs)}
  end

  # Device mode: outputs are worker handles wrapped as ExTinygrad.Backend tensors.
  defp decode_outputs(:device, output_specs, [], worker) do
    Enum.map(output_specs, fn spec ->
      type = Dtype.to_nx!(spec["dtype"])
      shape = List.to_tuple(spec["shape"])

      %Nx.Tensor{
        data: Backend.build(spec["id"], worker, shape, type),
        shape: shape,
        type: type,
        names: List.duplicate(nil, tuple_size(shape))
      }
    end)
  end

  # Host mode: outputs are raw blobs wrapped as Nx.BinaryBackend tensors.
  defp decode_outputs(:host, output_specs, output_blobs, _worker) do
    Enum.zip_with(output_specs, output_blobs, fn spec, blob ->
      type = Dtype.to_nx!(spec["dtype"])
      Nx.reshape(Nx.from_binary(blob, type), List.to_tuple(spec["shape"]))
    end)
  end

  defp request(worker, command, args, blobs, opts) do
    case Worker.request(worker, command, args, blobs, opts) do
      {:ok, _result, _blobs} = ok -> ok
      {:error, exception} -> raise exception
    end
  end
end
