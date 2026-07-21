defmodule NxTinygrad.Backend do
  @moduledoc """
  An `Nx.Backend` whose tensors live as buffers in a Python worker.

  Besides data movement (`from_binary`, `to_binary`, `backend_copy`,
  `backend_transfer`, `backend_deallocate`, `inspect`), most Nx operations run
  **eagerly on the device**: each call ships one graph-IR node to the worker
  (`run_node`), which applies it through the same operation table compiled
  graphs use — so eager semantics match `NxTinygrad.jit/2` exactly. This is
  what lets Bumblebee remap checkpoint params (transpose / reshape / upcast)
  on-device at load time, with weights landing resident.

  Operations that need traced functions (`reduce`, `window_reduce`, `block`)
  or host pointers still raise; hot loops should still go through
  `NxTinygrad.jit/2`, which fuses whole graphs instead of dispatching per op.

  The backend struct records the worker, its generation, and the buffer handle.
  A tensor whose generation no longer matches the worker (a restart) is stale:
  its data cannot be recovered and using it raises `NxTinygrad.StaleTensorError`.
  """
  @behaviour Nx.Backend

  defstruct [:ref, :worker, :generation, :shape, :type]

  alias NxTinygrad.{Dtype, TensorRef, Worker, WorkerIds}

  @eager """
  this operation cannot run eagerly on NxTinygrad.Backend (it needs a traced
  function or host pointers). Wrap the computation in Nx.Defn.jit/2 or
  NxTinygrad.jit/2.
  """

  # -- allocation / data movement -----------------------------------------

  @impl true
  def init(opts), do: Keyword.validate!(opts, worker: :default)

  @impl true
  def from_binary(%Nx.Tensor{shape: shape, type: type} = out, binary, opts) do
    worker = Keyword.get(opts, :worker, :default)
    dtype = Dtype.to_name!(type)

    {%{"id" => handle}, []} =
      request!(worker, "upload", %{"shape" => Tuple.to_list(shape), "dtype" => dtype}, [binary])

    :telemetry.execute([:nx_tinygrad, :transfer, :upload], %{bytes: byte_size(binary)}, %{worker: worker})
    %{out | data: build(handle, worker, shape, type)}
  end

  @impl true
  def to_binary(%Nx.Tensor{data: %__MODULE__{} = b, type: {_, bits}}, limit) do
    ensure_fresh!(b)
    {_meta, [blob]} = request!(b.worker, "download", %{"id" => handle(b)})
    :telemetry.execute([:nx_tinygrad, :transfer, :download], %{bytes: byte_size(blob)}, %{worker: b.worker})
    keep = min(byte_size(blob), limit * div(bits, 8))
    binary_part(blob, 0, keep)
  end

  @impl true
  def backend_deallocate(%Nx.Tensor{data: %__MODULE__{ref: ref, worker: worker}}) do
    # `take/1` claims the reference so a later GC does not release it again.
    case TensorRef.take(ref) do
      {_worker_id, generation, handle} ->
        if generation == Worker.generation(worker) do
          _ = Worker.request(worker, "release", %{"ids" => [handle]})
        end

        :ok

      nil ->
        :already_deallocated
    end
  end

  @impl true
  def backend_copy(%Nx.Tensor{data: %__MODULE__{worker: worker}} = tensor, __MODULE__, opts) do
    if Keyword.get(opts, :worker, :default) == worker do
      # Same worker: mint an independent handle device-side (a same-shape
      # view; worker buffers are immutable, and the underlying buffer lives
      # until every handle referencing it is released). This makes
      # preallocating already-resident params ~free instead of a full
      # download + re-upload round trip.
      eager!(tensor, "reshape", [tensor], %{"shape" => Tuple.to_list(tensor.shape)})
    else
      __MODULE__.from_binary(tensor, full_binary(tensor), opts)
    end
  end

  def backend_copy(tensor, backend, opts) do
    backend.from_binary(tensor, full_binary(tensor), opts)
  end

  @impl true
  def backend_transfer(tensor, Nx.Tensor, opts), do: backend_transfer(tensor, Nx.BinaryBackend, opts)

  def backend_transfer(tensor, backend, opts) do
    new = backend.from_binary(tensor, full_binary(tensor), opts)
    _ = backend_deallocate(tensor)
    new
  end

  @impl true
  def inspect(%Nx.Tensor{} = tensor, inspect_opts) do
    limit = inspect_opts.limit
    binary = Nx.to_binary(tensor, if(limit == :infinity, do: [], else: [limit: limit + 1]))
    Nx.Backend.inspect(tensor, binary, inspect_opts)
  rescue
    _ -> Inspect.Algebra.string("#NxTinygrad.Backend<data unavailable (worker restarted?)>")
  end

  # -- internals ----------------------------------------------------------

  @doc false
  def build(handle, worker, shape, type) do
    generation = Worker.generation(worker)
    ref = TensorRef.new(WorkerIds.id_for(worker), generation, handle)

    %__MODULE__{
      ref: ref,
      worker: worker,
      generation: generation,
      shape: shape,
      type: type
    }
  end

  @doc "The worker buffer handle behind a backend tensor."
  def handle(%__MODULE__{ref: ref}), do: TensorRef.handle(ref)

  defp full_binary(%Nx.Tensor{} = tensor), do: to_binary(tensor, Nx.size(tensor))

  defp ensure_fresh!(%__MODULE__{worker: worker, generation: gen}) do
    current = Worker.generation(worker)

    if current != gen do
      raise NxTinygrad.StaleTensorError,
        worker: worker,
        tensor_generation: gen,
        worker_generation: current
    end

    :ok
  end

  defp request!(worker, command, args, blobs \\ []) do
    case Worker.request(worker, command, args, blobs) do
      {:ok, result, blobs} -> {result, blobs}
      {:error, exception} -> raise exception
    end
  end

  # -- eager operations -----------------------------------------------------
  #
  # Each op ships one graph-IR node to the worker ("run_node"), which applies
  # it via the same operation table compiled graphs use and returns a new
  # device buffer. Attr encodings mirror NxTinygrad.Lowering exactly.

  @binary_ops [:add, :subtract, :multiply, :pow, :remainder, :divide, :atan2, :min, :max, :quotient] ++
                [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
                [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] ++
                [:logical_and, :logical_or, :logical_xor]

  @unary_ops Enum.map(Nx.Shared.unary_math_funs(), &elem(&1, 0)) ++
               [:abs, :bitwise_not, :ceil, :conjugate, :floor, :negate, :round, :sign] ++
               [:count_leading_zeros, :population_count, :real, :imag, :is_nan, :is_infinity]

  for op <- @binary_ops do
    @impl true
    def unquote(op)(out, l, r), do: eager!(out, unquote(Atom.to_string(op)), [l, r], %{})
  end

  for op <- @unary_ops do
    @impl true
    def unquote(op)(out, t), do: eager!(out, unquote(Atom.to_string(op)), [t], %{})
  end

  @impl true
  def constant(%Nx.Tensor{shape: {}} = out, scalar, opts) do
    binary = Nx.BinaryBackend.constant(out, scalar, []) |> Nx.to_binary()
    from_binary(out, binary, opts)
  end

  def constant(out, scalar, opts) do
    scalar_t = constant(%{out | shape: {}, names: []}, scalar, opts)
    eager!(out, "broadcast", [scalar_t], %{"shape" => Tuple.to_list(out.shape), "axes" => []}, opts)
  end

  @impl true
  def eye(out, opts), do: eager!(out, "eye", [], %{}, opts)

  @impl true
  def iota(out, axis, opts) do
    attrs = %{"axis" => axis && normalize_axis(axis, tuple_size(out.shape))}
    eager!(out, "iota", [], attrs, opts)
  end

  @impl true
  def as_type(out, t), do: eager!(out, "as_type", [t], %{})

  @impl true
  def bitcast(out, t), do: eager!(out, "bitcast", [t], %{})

  @impl true
  def reshape(out, t), do: eager!(out, "reshape", [t], %{"shape" => Tuple.to_list(out.shape)})

  @impl true
  def squeeze(out, t, axes),
    do: eager!(out, "squeeze", [t], %{"axes" => normalize_axes(axes, rank(t))})

  @impl true
  def broadcast(out, t, shape, axes),
    do: eager!(out, "broadcast", [t], %{"shape" => Tuple.to_list(shape), "axes" => axes})

  @impl true
  def transpose(out, t, axes), do: eager!(out, "transpose", [t], %{"axes" => axes})

  @impl true
  def reverse(out, t, axes),
    do: eager!(out, "reverse", [t], %{"axes" => normalize_axes(axes, rank(t))})

  @impl true
  def pad(out, t, pad_value, config) do
    value = encode_number(Nx.to_number(pad_value))

    if Enum.any?(config, fn {lo, hi, _i} -> lo < 0 or hi < 0 end) do
      # Negative padding crops; the worker only pads. Pad non-negatively,
      # then slice the crop off (Nx semantics).
      pos = Enum.map(config, fn {lo, hi, i} -> [max(lo, 0), max(hi, 0), i] end)

      inter_shape =
        Enum.zip_with(Tuple.to_list(t.shape), pos, fn d, [lo, hi, _i] -> d + lo + hi end)

      inter = %{out | shape: List.to_tuple(inter_shape)}
      padded = eager!(inter, "pad", [t], %{"config" => pos, "value" => value})

      lengths = Tuple.to_list(out.shape)

      eager!(out, "slice", [padded], %{
        "starts" => Enum.map(config, fn {lo, _hi, _i} -> %{"static" => max(-lo, 0)} end),
        "lengths" => lengths,
        "strides" => List.duplicate(1, length(lengths))
      })
    else
      cfg = Enum.map(config, fn {lo, hi, i} -> [lo, hi, i] end)
      eager!(out, "pad", [t], %{"config" => cfg, "value" => value})
    end
  end

  @impl true
  def dot(out, l, contract_l, batch_l, r, contract_r, batch_r) do
    eager!(out, "dot", [l, r], %{
      "contract_left" => contract_l,
      "contract_right" => contract_r,
      "batch_left" => batch_l,
      "batch_right" => batch_r
    })
  end

  @impl true
  def clip(out, t, min, max), do: eager!(out, "clip", [t, min, max], %{})

  @impl true
  def select(out, pred, on_true, on_false),
    do: eager!(out, "select", [pred, on_true, on_false], %{})

  @impl true
  def slice(out, t, starts, lengths, strides) do
    dims = Tuple.to_list(t.shape)

    specs =
      [starts, dims, lengths]
      |> Enum.zip_with(fn [s, dim, len] ->
        n = if is_integer(s), do: s, else: Nx.to_number(s)
        %{"static" => max(0, min(n, dim - len))}
      end)

    eager!(out, "slice", [t], %{"starts" => specs, "lengths" => lengths, "strides" => strides})
  end

  @impl true
  def put_slice(out, t, start_indices, slice) do
    # The worker clamps put_slice starts itself (Nx semantics).
    specs =
      Enum.map(start_indices, fn s ->
        %{"static" => if(is_integer(s), do: s, else: Nx.to_number(s))}
      end)

    eager!(out, "put_slice", [t, slice], %{"starts" => specs})
  end

  @impl true
  def gather(out, t, idx, opts),
    do: eager!(out, "gather", [t, idx], %{"axes" => normalize_opt_axes(opts[:axes], rank(t))})

  @impl true
  def concatenate(out, tensors, axis),
    do: eager!(out, "concatenate", tensors, %{"axis" => normalize_axis(axis, tuple_size(out.shape))})

  @impl true
  def stack(out, tensors, axis),
    do: eager!(out, "stack", tensors, %{"axis" => normalize_axis(axis, tuple_size(out.shape))})

  for op <- [:all, :any, :sum, :product, :reduce_max, :reduce_min] do
    @impl true
    def unquote(op)(out, t, opts) do
      attrs = %{"axes" => normalize_axes(opts[:axes], rank(t)), "keep_axes" => !!opts[:keep_axes]}
      eager!(out, unquote(Atom.to_string(op)), [t], attrs)
    end
  end

  for op <- [:argmax, :argmin] do
    @impl true
    def unquote(op)(out, t, opts) do
      attrs = %{
        "axis" => opts[:axis] && normalize_axis(opts[:axis], rank(t)),
        "keep_axis" => !!opts[:keep_axis],
        "tie_break" => Atom.to_string(opts[:tie_break] || :low)
      }

      eager!(out, unquote(Atom.to_string(op)), [t], attrs)
    end
  end

  for op <- [:sort, :argsort] do
    @impl true
    def unquote(op)(out, t, opts) do
      attrs = %{
        "axis" => normalize_axis(opts[:axis] || 0, rank(t)),
        "descending" => opts[:direction] == :desc
      }

      eager!(out, unquote(Atom.to_string(op)), [t], attrs)
    end
  end

  for op <- [:window_sum, :window_product, :window_max, :window_min] do
    @impl true
    def unquote(op)(out, t, window, opts) do
      attrs = %{
        "window" => Tuple.to_list(window),
        "strides" => opts[:strides],
        "padding" => Enum.map(opts[:padding], fn {lo, hi} -> [lo, hi] end),
        "window_dilations" => opts[:window_dilations]
      }

      eager!(out, unquote(Atom.to_string(op)), [t], attrs)
    end
  end

  for op <- [:window_scatter_max, :window_scatter_min] do
    @impl true
    def unquote(op)(out, t, source, init, window, opts) do
      attrs = %{
        "init" => encode_number(Nx.to_number(init)),
        "window" => Tuple.to_list(window),
        "strides" => opts[:strides],
        "padding" => Enum.map(opts[:padding], fn {lo, hi} -> [lo, hi] end)
      }

      eager!(out, unquote(Atom.to_string(op)), [t, source], attrs)
    end
  end

  for op <- [:indexed_add, :indexed_put] do
    @impl true
    def unquote(op)(out, t, idx, updates, opts) do
      attrs = %{"axes" => normalize_opt_axes(opts[:axes], rank(t))}
      eager!(out, unquote(Atom.to_string(op)), [t, idx, updates], attrs)
    end
  end

  @impl true
  def conv(out, t, kernel, opts) do
    attrs = %{
      "strides" => opts[:strides],
      "padding" => Enum.map(opts[:padding], fn {lo, hi} -> [lo, hi] end),
      "input_dilation" => opts[:input_dilation],
      "kernel_dilation" => opts[:kernel_dilation],
      "feature_group_size" => opts[:feature_group_size],
      "batch_group_size" => opts[:batch_group_size],
      "input_permutation" => opts[:input_permutation],
      "kernel_permutation" => opts[:kernel_permutation],
      "output_permutation" => opts[:output_permutation]
    }

    eager!(out, "conv", [t, kernel], attrs)
  end

  @impl true
  def triangular_solve(out, a, b, opts) do
    attrs = %{
      "transform_a" => Atom.to_string(opts[:transform_a] || :none),
      "left_side" => opts[:left_side] != false,
      "lower" => opts[:lower] != false
    }

    eager!(out, "triangular_solve", [a, b], attrs)
  end

  for op <- [:fft, :ifft] do
    @impl true
    def unquote(op)(out, t, opts) do
      attrs = %{"length" => opts[:length], "axis" => normalize_axis(opts[:axis], rank(t))}
      eager!(out, unquote(Atom.to_string(op)), [t], attrs)
    end
  end

  # -- eager plumbing --------------------------------------------------------

  defp eager!(out, op, inputs, attrs, opts \\ []) do
    worker = Keyword.get(opts, :worker) || eager_worker(inputs)
    {specs, blobs} = eager_inputs(inputs, worker)

    args = %{
      "op" => op,
      "attrs" => attrs,
      "shape" => Tuple.to_list(out.shape),
      "dtype" => Dtype.to_name!(out.type),
      "inputs" => specs
    }

    {%{"id" => id}, []} = request!(worker, "run_node", args, blobs)
    %{out | data: build(id, worker, out.shape, out.type)}
  end

  defp eager_worker(inputs) do
    Enum.find_value(inputs, :default, fn
      %Nx.Tensor{data: %__MODULE__{worker: worker}} -> worker
      _ -> nil
    end)
  end

  # Inputs already on this worker pass by handle; anything else (a scalar the
  # caller made on Nx.BinaryBackend, a tensor from another worker) ships as an
  # inline blob. The compute still runs on tinygrad — never a silent fallback.
  defp eager_inputs(inputs, worker) do
    {specs, {blobs, _n}} =
      Enum.map_reduce(inputs, {[], 0}, fn
        %Nx.Tensor{data: %__MODULE__{worker: ^worker} = b}, acc ->
          ensure_fresh!(b)
          {%{"kind" => "handle", "id" => handle(b)}, acc}

        %Nx.Tensor{} = t, {blobs, n} ->
          spec = %{
            "kind" => "blob",
            "blob_index" => n,
            "shape" => Tuple.to_list(t.shape),
            "dtype" => Dtype.to_name!(t.type)
          }

          {spec, {[Nx.to_binary(t) | blobs], n + 1}}
      end)

    {specs, Enum.reverse(blobs)}
  end

  defp rank(%Nx.Tensor{shape: shape}), do: tuple_size(shape)

  defp normalize_axis(axis, rank) when axis < 0, do: axis + rank
  defp normalize_axis(axis, _rank), do: axis

  defp normalize_axes(nil, 0), do: []
  defp normalize_axes(nil, rank), do: Enum.to_list(0..(rank - 1))

  defp normalize_axes(axes, rank) when is_list(axes) do
    axes |> Enum.map(&normalize_axis(&1, rank)) |> Enum.sort()
  end

  defp normalize_opt_axes(nil, rank), do: Enum.to_list(0..(rank - 1))
  defp normalize_opt_axes(axes, rank), do: Enum.map(axes, &normalize_axis(&1, rank))

  defp encode_number(%Complex{re: re, im: im}), do: %{"re" => encode_number(re), "im" => encode_number(im)}
  defp encode_number(n) when is_number(n), do: n
  defp encode_number(:infinity), do: "Infinity"
  defp encode_number(:neg_infinity), do: "-Infinity"
  defp encode_number(:nan), do: "NaN"

  # A block is a composite of primitive ops; running its default fun eagerly
  # recurses straight back into this backend's eager surface (how
  # Nx.BinaryBackend implements it too). Unlocks take/take_along_axis/tile.
  @impl true
  def block(struct, _output, args, fun), do: apply(fun, [struct | args])

  # -- unsupported eager surface (raises, no silent fallback) --------------

  @still_unsupported [
    to_batched: 3,
    from_pointer: 5,
    to_pointer: 2,
    reduce: 5,
    window_reduce: 6
  ]

  for {name, arity} <- @still_unsupported do
    args = for i <- 1..arity, do: Macro.var(:"_arg#{i}", __MODULE__)

    @impl true
    def unquote(name)(unquote_splicing(args)) do
      raise NxTinygrad.Error, message: "#{unquote(name)}/#{unquote(arity)}: " <> unquote(@eager)
    end
  end
end
