defmodule ExTinygrad.Lowering do
  @moduledoc """
  Lowers an `Nx.Defn.Expr` DAG into the deterministic `ExTinygrad.Graph` IR.

  The traversal is post-order, assigning sequential ids so children always get
  smaller ids than their parents (topological). Ids are our own — Nx expression
  ids (references) are used only to deduplicate shared subexpressions.

  Unsupported operations raise `ExTinygrad.CompileError` before anything is sent
  to Python; there is no silent host fallback.
  """
  alias ExTinygrad.{Dtype, Graph}
  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  @unary ~w(negate abs exp expm1 log log1p sqrt rsqrt tanh sigmoid sin cos floor ceil)a
  @binary ~w(add subtract multiply divide pow max min)a
  @comparison ~w(equal not_equal less less_equal greater greater_equal)a
  @reduce ~w(sum reduce_max reduce_min all any)a

  @doc "Lower a list of output expression tensors into an `ExTinygrad.Graph`."
  @spec to_graph([T.t()]) :: Graph.t()
  def to_graph(outputs) when is_list(outputs) do
    state = %{ids: %{}, counter: 0, inputs: [], constants: [], nodes: [], blobs: [], blob_count: 0}

    {output_specs, state} =
      Enum.map_reduce(outputs, state, fn root, st ->
        {id, st} = lower(root, st)
        {%{"node" => id, "shape" => shape_of(root), "dtype" => dtype_of(root)}, st}
      end)

    %Graph{
      inputs: Enum.reverse(state.inputs),
      constants: Enum.reverse(state.constants),
      nodes: Enum.reverse(state.nodes),
      outputs: output_specs,
      blobs: Enum.reverse(state.blobs)
    }
  end

  # -- traversal ----------------------------------------------------------

  defp lower(%T{data: %Expr{id: ref}} = tensor, state) do
    case Map.fetch(state.ids, ref) do
      {:ok, id} -> {id, state}
      :error -> lower_new(tensor, state)
    end
  end

  # parameter: a runtime input at flattened position `pos`.
  defp lower_new(%T{data: %Expr{op: :parameter, args: [pos], id: ref}} = t, state) do
    {id, state} = alloc(state, ref)
    input = %{"id" => id, "index" => pos, "shape" => shape_of(t), "dtype" => dtype_of(t)}
    {id, %{state | inputs: [input | state.inputs]}}
  end

  # metadata: transparent wrapper; alias this node to its inner expression.
  defp lower_new(%T{data: %Expr{op: :metadata, args: [inner, _meta], id: ref}}, state) do
    {id, state} = lower(inner, state)
    {id, %{state | ids: Map.put(state.ids, ref, id)}}
  end

  # scalar constant broadcast to this node's shape.
  defp lower_new(%T{data: %Expr{op: :constant, args: [number], id: ref}} = t, state) do
    {id, state} = alloc(state, ref)
    const = %{"id" => id, "value" => encode_number(number), "shape" => shape_of(t), "dtype" => dtype_of(t)}
    {id, %{state | constants: [const | state.constants]}}
  end

  # baked-in literal tensor: ship its bytes as an inline constant blob.
  defp lower_new(%T{data: %Expr{op: :tensor, args: [tensor], id: ref}} = t, state) do
    {id, state} = alloc(state, ref)
    index = state.blob_count
    const = %{"id" => id, "data_index" => index, "shape" => shape_of(t), "dtype" => dtype_of(t)}

    {id,
     %{
       state
       | constants: [const | state.constants],
         blobs: [Nx.to_binary(tensor) | state.blobs],
         blob_count: index + 1
     }}
  end

  defp lower_new(%T{data: %Expr{op: op, args: [a]}} = t, state) when op in @unary do
    {[aid], state} = lower_children([a], state)
    add_node(state, t, Atom.to_string(op), [aid], %{})
  end

  defp lower_new(%T{data: %Expr{op: op, args: [a, b]}} = t, state)
       when op in @binary or op in @comparison do
    {ids, state} = lower_children([a, b], state)
    add_node(state, t, Atom.to_string(op), ids, %{})
  end

  defp lower_new(%T{data: %Expr{op: :select, args: [p, on_true, on_false]}} = t, state) do
    {ids, state} = lower_children([p, on_true, on_false], state)
    add_node(state, t, "select", ids, %{})
  end

  defp lower_new(%T{data: %Expr{op: op, args: [a, opts]}} = t, state) when op in @reduce do
    {[aid], state} = lower_children([a], state)
    axes = normalize_axes(opts[:axes], rank(a))
    add_node(state, t, Atom.to_string(op), [aid], %{"axes" => axes, "keep_axes" => !!opts[:keep_axes]})
  end

  defp lower_new(%T{data: %Expr{op: :reshape, args: [a]}} = t, state) do
    {[aid], state} = lower_children([a], state)
    add_node(state, t, "reshape", [aid], %{"shape" => shape_of(t)})
  end

  defp lower_new(%T{data: %Expr{op: :squeeze, args: [a, axes]}} = t, state) do
    {[aid], state} = lower_children([a], state)
    add_node(state, t, "squeeze", [aid], %{"axes" => normalize_axes(axes, rank(a))})
  end

  defp lower_new(%T{data: %Expr{op: :broadcast, args: [a, shape, axes]}} = t, state) do
    {[aid], state} = lower_children([a], state)
    add_node(state, t, "broadcast", [aid], %{"shape" => Tuple.to_list(shape), "axes" => axes})
  end

  defp lower_new(%T{data: %Expr{op: :transpose, args: [a, axes]}} = t, state) do
    {[aid], state} = lower_children([a], state)
    add_node(state, t, "transpose", [aid], %{"axes" => axes})
  end

  defp lower_new(%T{data: %Expr{op: :concatenate, args: [list, axis]}} = t, state) do
    {ids, state} = lower_children(list, state)
    add_node(state, t, "concatenate", ids, %{"axis" => normalize_axis(axis, rank(t))})
  end

  defp lower_new(%T{data: %Expr{op: :slice, args: [a, starts, lengths, strides]}} = t, state) do
    {[aid], state} = lower_children([a], state)
    add_node(state, t, "slice", [aid], %{"starts" => starts, "lengths" => lengths, "strides" => strides})
  end

  defp lower_new(%T{data: %Expr{op: :as_type, args: [a]}} = t, state) do
    {[aid], state} = lower_children([a], state)
    add_node(state, t, "as_type", [aid], %{})
  end

  defp lower_new(%T{data: %Expr{op: :dot, args: [a, ca, ba, b, cb, bb]}} = t, state) do
    {[aid, bid], state} = lower_children([a, b], state)

    attrs = %{
      "contract_left" => ca,
      "contract_right" => cb,
      "batch_left" => ba,
      "batch_right" => bb
    }

    add_node(state, t, "dot", [aid, bid], attrs)
  end

  defp lower_new(%T{data: %Expr{op: op}} = t, _state) do
    raise ExTinygrad.CompileError,
      message: "unsupported Nx operation: #{op}",
      operation: op,
      output_spec: %{shape: shape_of(t), dtype: safe_dtype(t)},
      hint: "operation #{op} is not yet lowered by ExTinygrad in v0.1"
  end

  # -- helpers ------------------------------------------------------------

  defp lower_children(tensors, state) do
    {ids, state} =
      Enum.reduce(tensors, {[], state}, fn tensor, {ids, st} ->
        {id, st} = lower(tensor, st)
        {[id | ids], st}
      end)

    {Enum.reverse(ids), state}
  end

  defp add_node(state, %T{data: %Expr{id: ref}} = t, op, input_ids, attrs) do
    {id, state} = alloc(state, ref)

    node = %{
      "id" => id,
      "op" => op,
      "inputs" => input_ids,
      "attrs" => attrs,
      "shape" => shape_of(t),
      "dtype" => dtype_of(t)
    }

    {id, %{state | nodes: [node | state.nodes]}}
  end

  defp alloc(state, ref) do
    id = state.counter
    {id, %{state | counter: id + 1, ids: Map.put(state.ids, ref, id)}}
  end

  defp shape_of(%T{shape: shape}), do: Tuple.to_list(shape)
  defp dtype_of(%T{type: type}), do: Dtype.to_name!(type)

  defp safe_dtype(%T{type: type}),
    do: with({:ok, n} <- Dtype.to_name(type), do: n, else: (_ -> inspect(type)))

  defp rank(%T{shape: shape}), do: tuple_size(shape)

  defp normalize_axis(axis, rank) when axis < 0, do: axis + rank
  defp normalize_axis(axis, _rank), do: axis

  defp normalize_axes(nil, 0), do: []
  defp normalize_axes(nil, rank), do: Enum.to_list(0..(rank - 1))

  defp normalize_axes(axes, rank) when is_list(axes) do
    axes |> Enum.map(&normalize_axis(&1, rank)) |> Enum.sort()
  end

  # Encode numeric constants; map non-finite atoms to strings the worker understands.
  defp encode_number(n) when is_number(n), do: n
  defp encode_number(:infinity), do: "Infinity"
  defp encode_number(:neg_infinity), do: "-Infinity"
  defp encode_number(:nan), do: "NaN"
end
