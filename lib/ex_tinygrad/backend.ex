defmodule ExTinygrad.Backend do
  @moduledoc """
  An `Nx.Backend` whose tensors live as buffers in a Python worker.

  Only data movement is supported eagerly (`from_binary`, `to_binary`,
  `backend_copy`, `backend_transfer`, `backend_deallocate`, `inspect`).
  Elementwise/reduction/etc. operations must go through `ExTinygrad.jit/2` or
  `Nx.Defn.jit/2` — calling them eagerly raises, rather than silently falling
  back to another backend.

  The backend struct records the worker, its generation, and the buffer handle.
  A tensor whose generation no longer matches the worker (a restart) is stale:
  its data cannot be recovered and using it raises `ExTinygrad.StaleTensorError`.
  """
  @behaviour Nx.Backend

  defstruct [:handle, :worker, :generation, :shape, :type]

  alias ExTinygrad.{Dtype, Worker}

  @eager """
  ExTinygrad eager operations are not supported in version 0.1.
  Wrap this computation in Nx.Defn.jit/2 or ExTinygrad.jit/2.
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

    %{out | data: build(handle, worker, shape, type)}
  end

  @impl true
  def to_binary(%Nx.Tensor{data: %__MODULE__{} = b, type: {_, bits}}, limit) do
    ensure_fresh!(b)
    {_meta, [blob]} = request!(b.worker, "download", %{"id" => b.handle})
    keep = min(byte_size(blob), limit * div(bits, 8))
    binary_part(blob, 0, keep)
  end

  @impl true
  def backend_deallocate(%Nx.Tensor{data: %__MODULE__{} = b}) do
    case Worker.request(b.worker, "release", %{"ids" => [b.handle]}) do
      {:ok, _, _} -> :ok
      {:error, _} -> :already_deallocated
    end
  end

  @impl true
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
    _ -> Inspect.Algebra.string("#ExTinygrad.Backend<data unavailable (worker restarted?)>")
  end

  # -- internals ----------------------------------------------------------

  @doc false
  def build(handle, worker, shape, type) do
    %__MODULE__{
      handle: handle,
      worker: worker,
      generation: Worker.generation(worker),
      shape: shape,
      type: type
    }
  end

  defp full_binary(%Nx.Tensor{} = tensor), do: to_binary(tensor, Nx.size(tensor))

  defp ensure_fresh!(%__MODULE__{worker: worker, generation: gen}) do
    current = Worker.generation(worker)

    if current != gen do
      raise ExTinygrad.StaleTensorError,
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

  # -- unsupported eager surface (raises, no silent fallback) --------------

  @binary_ops [:add, :subtract, :multiply, :pow, :remainder, :divide, :atan2, :min, :max, :quotient] ++
                [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
                [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] ++
                [:logical_and, :logical_or, :logical_xor]

  @unary_ops Enum.map(Nx.Shared.unary_math_funs(), &elem(&1, 0)) ++
               [:abs, :bitwise_not, :ceil, :conjugate, :floor, :negate, :round, :sign] ++
               [:count_leading_zeros, :population_count, :real, :imag, :is_nan, :is_infinity]

  @other_ops [
    eye: 2,
    iota: 3,
    to_batched: 3,
    from_pointer: 5,
    to_pointer: 2,
    constant: 3,
    as_type: 2,
    bitcast: 2,
    reshape: 2,
    squeeze: 3,
    broadcast: 4,
    transpose: 3,
    pad: 4,
    reverse: 3,
    dot: 7,
    clip: 4,
    slice: 5,
    put_slice: 4,
    gather: 4,
    concatenate: 3,
    stack: 3,
    select: 4,
    conv: 4,
    all: 3,
    any: 3,
    sum: 3,
    product: 3,
    reduce_max: 3,
    reduce_min: 3,
    argmax: 3,
    argmin: 3,
    reduce: 5,
    window_reduce: 6,
    window_sum: 4,
    window_product: 4,
    window_max: 4,
    window_min: 4,
    sort: 3,
    argsort: 3,
    window_scatter_max: 6,
    window_scatter_min: 6,
    indexed_add: 5,
    indexed_put: 5,
    triangular_solve: 4,
    fft: 3,
    ifft: 3,
    block: 4
  ]

  @all_unsupported Enum.map(@binary_ops, &{&1, 3}) ++
                     Enum.map(@unary_ops, &{&1, 2}) ++ @other_ops

  for {name, arity} <- @all_unsupported do
    args = for i <- 1..arity, do: Macro.var(:"_arg#{i}", __MODULE__)

    @impl true
    def unquote(name)(unquote_splicing(args)) do
      raise ExTinygrad.Error, message: "#{unquote(name)}/#{unquote(arity)}: " <> unquote(@eager)
    end
  end
end
