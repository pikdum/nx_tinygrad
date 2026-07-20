defmodule NxTinygrad.Lowering do
  @moduledoc """
  Lowers an `Nx.Defn.Expr` DAG into the deterministic `NxTinygrad.Graph` IR.

  The traversal is post-order, assigning sequential ids so children always get
  smaller ids than their parents (topological). Ids are our own — Nx expression
  ids (references) are used only to deduplicate shared subexpressions.

  Unsupported operations raise `NxTinygrad.CompileError` before anything is sent
  to Python; there is no silent host fallback.
  """
  alias NxTinygrad.{Dtype, Graph}
  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  @unary ~w(negate abs exp expm1 log log1p sqrt rsqrt tanh sigmoid sin cos
            tan asin acos atan sinh cosh asinh acosh atanh erf erfc erf_inv cbrt sign round
            conjugate count_leading_zeros population_count
            is_nan is_infinity bitwise_not floor ceil)a
  @binary ~w(add subtract multiply divide pow max min remainder quotient atan2
             bitwise_and bitwise_or bitwise_xor left_shift right_shift
             logical_and logical_or logical_xor)a
  @comparison ~w(equal not_equal less less_equal greater greater_equal)a
  @reduce ~w(sum product reduce_max reduce_min all any)a

  @doc "Lower a list of output expression tensors into an `NxTinygrad.Graph`."
  @spec to_graph([T.t()]) :: Graph.t()
  def to_graph(outputs) when is_list(outputs) do
    state = %{
      ids: %{},
      counter: 0,
      inputs: [],
      constants: [],
      nodes: [],
      blobs: [],
      blob_count: 0,
      param_bind: nil,
      tuples: %{}
    }

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

  # parameter: normally a runtime input at flattened position `pos`. Inside a
  # block's default expression, `pos` instead indexes the block's inputs, which
  # are already lowered — alias to that id rather than declaring a graph input.
  defp lower_new(%T{data: %Expr{op: :parameter, args: [pos], id: ref}} = t, state) do
    case state.param_bind do
      %{} = bind when is_map_key(bind, pos) ->
        id = Map.fetch!(bind, pos)
        {id, %{state | ids: Map.put(state.ids, ref, id)}}

      _ ->
        {id, state} = alloc(state, ref)
        input = %{"id" => id, "index" => pos, "shape" => shape_of(t), "dtype" => dtype_of(t)}
        {id, %{state | inputs: [input | state.inputs]}}
    end
  end

  # block: an optional Nx operation carrying a pre-traced *pure* default
  # expression (arg 2) as a function of its inputs (arg 1). We never invoke the
  # impure callback (arg 3); we bind the inputs to the default's parameters and
  # lower that expression, so composites like cumulative_* reduce to primitives.
  defp lower_new(%T{data: %Expr{op: :block, args: [_marker, inputs, default, _fun], id: ref}} = t, state)
       when is_list(inputs) do
    cond do
      match?(%T{data: %Expr{}}, default) ->
        {[root_id], state} = lower_block(inputs, [default], state)
        {root_id, %{state | ids: Map.put(state.ids, ref, root_id)}}

      is_tuple(default) ->
        # Tuple-valued block; canonically reached via `elem`, but handle a direct
        # reference too by caching the element ids and returning the first.
        {ids, state} = lower_tuple(t, state)
        {hd(ids), state}

      true ->
        raise NxTinygrad.CompileError,
          message: "unsupported Nx.Block with impure default",
          operation: :block,
          output_spec: %{shape: shape_of(t), dtype: safe_dtype(t)},
          hint: "this block does not expose a pure default expression graph"
    end
  end

  # elem: project one tensor out of a tuple-valued source (e.g. a tuple block
  # from top_k or the QR/LU/SVD linalg composites).
  defp lower_new(%T{data: %Expr{op: :elem, args: [src, index], id: ref}}, state) do
    {ids, state} = lower_tuple(src, state)
    id = Enum.at(ids, index)
    {id, %{state | ids: Map.put(state.ids, ref, id)}}
  end

  # cond: pure clauses; lower to a right-folded chain of predicated selects
  # (all branches are side-effect-free defn expressions of the same shape).
  defp lower_new(%T{data: %Expr{op: :cond, args: [clauses, default], id: ref}} = t, state) do
    {default_id, state} = lower(default, state)
    shape = shape_of(t)
    dtype = dtype_of(t)

    {result_id, state} =
      clauses
      |> Enum.reverse()
      |> Enum.reduce({default_id, state}, fn {pred, expr}, {else_id, st} ->
        {pred_id, st} = lower(pred, st)
        {expr_id, st} = lower(expr, st)
        add_raw_node(st, "select", [pred_id, expr_id, else_id], %{}, shape, dtype)
      end)

    {result_id, %{state | ids: Map.put(state.ids, ref, result_id)}}
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

  defp lower_new(%T{data: %Expr{op: :clip, args: [a, min, max]}} = t, state) do
    {ids, state} = lower_children([a, min, max], state)
    add_node(state, t, "clip", ids, %{})
  end

  defp lower_new(%T{data: %Expr{op: :stack, args: [list, axis]}} = t, state) do
    {ids, state} = lower_children(list, state)
    add_node(state, t, "stack", ids, %{"axis" => normalize_axis(axis, rank(t))})
  end

  defp lower_new(%T{data: %Expr{op: :eye}} = t, state) do
    add_node(state, t, "eye", [], %{})
  end

  defp lower_new(%T{data: %Expr{op: op, args: [a, opts]}} = t, state) when op in @reduce do
    {[aid], state} = lower_children([a], state)
    axes = normalize_axes(opts[:axes], rank(a))
    add_node(state, t, Atom.to_string(op), [aid], %{"axes" => axes, "keep_axes" => !!opts[:keep_axes]})
  end

  defp lower_new(%T{data: %Expr{op: op, args: [a, opts]}} = t, state) when op in [:argmax, :argmin] do
    {[aid], state} = lower_children([a], state)
    axis = opts[:axis]

    attrs = %{
      "axis" => axis && normalize_axis(axis, rank(a)),
      "keep_axis" => !!opts[:keep_axis],
      "tie_break" => Atom.to_string(opts[:tie_break] || :low)
    }

    add_node(state, t, Atom.to_string(op), [aid], attrs)
  end

  defp lower_new(%T{data: %Expr{op: op, args: [a, window, opts]}} = t, state)
       when op in [:window_sum, :window_max, :window_min, :window_product] do
    {[aid], state} = lower_children([a], state)

    attrs = %{
      "window" => Tuple.to_list(window),
      "strides" => opts[:strides],
      "padding" => Enum.map(opts[:padding], fn {lo, hi} -> [lo, hi] end),
      "window_dilations" => opts[:window_dilations]
    }

    add_node(state, t, Atom.to_string(op), [aid], attrs)
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

  defp lower_new(%T{data: %Expr{op: :reverse, args: [a, axes]}} = t, state) do
    {[aid], state} = lower_children([a], state)
    add_node(state, t, "reverse", [aid], %{"axes" => normalize_axes(axes, rank(a))})
  end

  defp lower_new(%T{data: %Expr{op: :concatenate, args: [list, axis]}} = t, state) do
    {ids, state} = lower_children(list, state)
    add_node(state, t, "concatenate", ids, %{"axis" => normalize_axis(axis, rank(t))})
  end

  # slice: start indices may be static integers or dynamic scalar tensors
  # (e.g. slicing at a loop counter). Dynamic starts become extra node inputs the
  # worker reads and clamps at runtime.
  defp lower_new(%T{data: %Expr{op: :slice, args: [a, starts, lengths, strides]}} = t, state) do
    {aid, state} = lower(a, state)

    {start_specs, dyn_exprs} =
      Enum.reduce(starts, {[], []}, fn
        s, {specs, dyn} when is_integer(s) -> {[%{"static" => s} | specs], dyn}
        s, {specs, dyn} -> {[%{"input" => length(dyn)} | specs], [s | dyn]}
      end)

    {dyn_ids, state} = lower_children(Enum.reverse(dyn_exprs), state)

    add_node(state, t, "slice", [aid | dyn_ids], %{
      "starts" => Enum.reverse(start_specs),
      "lengths" => lengths,
      "strides" => strides
    })
  end

  defp lower_new(%T{data: %Expr{op: :pad, args: [a, value, config]}} = t, state) do
    {[aid], state} = lower_children([a], state)
    pad_value = scalar_constant!(value, :pad)
    cfg = Enum.map(config, fn {low, high, interior} -> [low, high, interior] end)
    add_node(state, t, "pad", [aid], %{"config" => cfg, "value" => encode_number(pad_value)})
  end

  defp lower_new(%T{data: %Expr{op: op, args: [a, opts]}} = t, state) when op in [:sort, :argsort] do
    {[aid], state} = lower_children([a], state)
    axis = normalize_axis(opts[:axis] || 0, rank(a))

    add_node(state, t, Atom.to_string(op), [aid], %{"axis" => axis, "descending" => opts[:direction] == :desc})
  end

  # gather: coordinate gather over `axes`; the last dim of `idx` enumerates a
  # coordinate over those axes (order matters, so we preserve it).
  defp lower_new(%T{data: %Expr{op: :gather, args: [a, idx, opts]}} = t, state) do
    {ids, state} = lower_children([a, idx], state)
    axes = (opts[:axes] || Enum.to_list(0..(rank(a) - 1))) |> Enum.map(&normalize_axis(&1, rank(a)))
    add_node(state, t, "gather", ids, %{"axes" => axes})
  end

  # iota: index counter along `axis` (nil = flattened), no runtime inputs.
  defp lower_new(%T{data: %Expr{op: :iota, args: [axis]}} = t, state) do
    add_node(state, t, "iota", [], %{"axis" => axis && normalize_axis(axis, rank(t))})
  end

  # put_slice: overwrite a contiguous block starting at compile-time offsets.
  defp lower_new(%T{data: %Expr{op: :put_slice, args: [a, starts, slice]}} = t, state) do
    {ids, state} = lower_children([a, slice], state)
    start_vals = Enum.map(starts, &scalar_constant!(&1, :put_slice))
    add_node(state, t, "put_slice", ids, %{"starts" => start_vals})
  end

  # indexed_add / indexed_put: scatter (accumulate / overwrite) at coordinates
  # over `axes`, mirroring gather's coordinate layout.
  defp lower_new(%T{data: %Expr{op: op, args: [a, idx, updates, opts]}} = t, state)
       when op in [:indexed_add, :indexed_put] do
    {ids, state} = lower_children([a, idx, updates], state)
    axes = (opts[:axes] || Enum.to_list(0..(rank(a) - 1))) |> Enum.map(&normalize_axis(&1, rank(a)))
    add_node(state, t, Atom.to_string(op), ids, %{"axes" => axes})
  end

  defp lower_new(%T{data: %Expr{op: :as_type, args: [a]}} = t, state) do
    {[aid], state} = lower_children([a], state)
    add_node(state, t, "as_type", [aid], %{})
  end

  defp lower_new(%T{data: %Expr{op: :bitcast, args: [a]}} = t, state) do
    {[aid], state} = lower_children([a], state)
    add_node(state, t, "bitcast", [aid], %{})
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

  defp lower_new(%T{data: %Expr{op: :conv, args: [inp, ker, opts]}} = t, state) do
    {ids, state} = lower_children([inp, ker], state)

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

    add_node(state, t, "conv", ids, attrs)
  end

  defp lower_new(%T{data: %Expr{op: op}} = t, _state) do
    raise NxTinygrad.CompileError,
      message: "unsupported Nx operation: #{op}",
      operation: op,
      output_spec: %{shape: shape_of(t), dtype: safe_dtype(t)},
      hint: "operation #{op} is not yet lowered by NxTinygrad in v0.1"
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

  # Lower a tuple-valued source into its list of element ids, cached by ref so
  # multiple `elem` projections of the same block share one lowering.
  defp lower_tuple(%T{data: %Expr{op: :block, args: [_m, inputs, default, _fun], id: ref}}, state)
       when is_list(inputs) and is_tuple(default) do
    case Map.fetch(state.tuples, ref) do
      {:ok, ids} ->
        {ids, state}

      :error ->
        {ids, state} = lower_block(inputs, Tuple.to_list(default), state)
        {ids, %{state | tuples: Map.put(state.tuples, ref, ids)}}
    end
  end

  # while: a data-dependent loop. We lower the initial loop vars in the outer
  # graph, lower the condition and body as isolated sub-graphs (their loop-var
  # parameters become sub-graph inputs), and emit a multi-output `while` node the
  # worker runs as an eager Python loop. The tuple of final loop-var ids is what
  # `elem` projects.
  defp lower_tuple(%T{data: %Expr{op: :while, args: [init, _params, cond, body], id: ref}}, state) do
    case Map.fetch(state.tuples, ref) do
      {:ok, ids} ->
        {ids, state}

      :error ->
        init_list = Tuple.to_list(init)
        body_list = Tuple.to_list(body)

        {init_ids, state} = lower_children(init_list, state)
        {cond_sub, state} = lower_isolated([cond], state)
        {body_sub, state} = lower_isolated(body_list, state)

        {output_specs, state} =
          Enum.map_reduce(body_list, state, fn expr, st ->
            id = st.counter
            {%{"id" => id, "shape" => shape_of(expr), "dtype" => dtype_of(expr)}, %{st | counter: id + 1}}
          end)

        output_ids = Enum.map(output_specs, & &1["id"])

        node = %{
          "id" => hd(output_ids),
          "op" => "while",
          "inputs" => init_ids,
          "attrs" => %{"cond" => cond_sub, "body" => body_sub},
          "outputs" => output_specs
        }

        {output_ids, %{state | nodes: [node | state.nodes], tuples: Map.put(state.tuples, ref, output_ids)}}
    end
  end

  # metadata is a transparent wrapper; project through it to the inner tuple.
  defp lower_tuple(%T{data: %Expr{op: :metadata, args: [inner, _meta]}}, state) do
    lower_tuple(inner, state)
  end

  defp lower_tuple(%T{data: %Expr{op: op}} = t, _state) do
    raise NxTinygrad.CompileError,
      message: "elem projection from unsupported tuple source: #{op}",
      operation: op,
      output_spec: %{shape: shape_of(t), dtype: safe_dtype(t)},
      hint: "tuple projection supports tuple-valued blocks and while loops"
  end

  defp lower_tuple(other, _state) do
    raise NxTinygrad.CompileError,
      message: "elem projection from unsupported tuple source: #{inspect(other, limit: 2)}",
      operation: :elem,
      output_spec: %{shape: [], dtype: "n/a"},
      hint: "tuple projection supports tuple-valued blocks and while loops"
  end

  # Bind a block's inputs to its default expression's parameters, then lower each
  # of the given default outputs (single- or multi-output) under that binding.
  defp lower_block(inputs, outputs, state) do
    {input_ids, state} = lower_children(inputs, state)
    bind = input_ids |> Enum.with_index() |> Map.new(fn {id, idx} -> {idx, id} end)
    saved = state.param_bind
    {ids, state} = lower_children(outputs, %{state | param_bind: bind})
    {ids, %{state | param_bind: saved}}
  end

  # Lower a set of output expressions into a self-contained sub-graph with its own
  # id space. Loop-var parameters become sub-graph inputs (indexed by position).
  # Inline-literal blobs are shared with the parent's flat blob list.
  defp lower_isolated(outputs, state) do
    sub = %{
      ids: %{},
      counter: 0,
      inputs: [],
      constants: [],
      nodes: [],
      blobs: state.blobs,
      blob_count: state.blob_count,
      param_bind: nil,
      tuples: %{}
    }

    {out_specs, sub} =
      Enum.map_reduce(outputs, sub, fn expr, s ->
        {id, s} = lower(expr, s)
        {%{"node" => id, "shape" => shape_of(expr), "dtype" => dtype_of(expr)}, s}
      end)

    subgraph = %{
      "inputs" => Enum.reverse(sub.inputs),
      "constants" => Enum.reverse(sub.constants),
      "nodes" => Enum.reverse(sub.nodes),
      "outputs" => out_specs
    }

    {subgraph, %{state | blobs: sub.blobs, blob_count: sub.blob_count}}
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

  # Emit a node not tied to an Nx expression ref (for synthetic nodes such as the
  # selects a `cond` expands into).
  defp add_raw_node(state, op, input_ids, attrs, shape, dtype) do
    id = state.counter

    node = %{
      "id" => id,
      "op" => op,
      "inputs" => input_ids,
      "attrs" => attrs,
      "shape" => shape,
      "dtype" => dtype
    }

    {id, %{state | counter: id + 1, nodes: [node | state.nodes]}}
  end

  defp shape_of(%T{shape: shape}), do: Tuple.to_list(shape)
  defp dtype_of(%T{type: type}), do: Dtype.to_name!(type)

  defp safe_dtype(%T{type: type}),
    do: with({:ok, n} <- Dtype.to_name(type), do: n, else: (_ -> inspect(type)))

  defp rank(%T{shape: shape}), do: tuple_size(shape)

  # Extract a compile-time scalar number from a constant/tensor expression (e.g.
  # a pad fill value). Anything else raises, since the worker needs a literal.
  defp scalar_constant!(%T{data: %Expr{op: :metadata, args: [inner, _meta]}}, op),
    do: scalar_constant!(inner, op)

  defp scalar_constant!(%T{data: %Expr{op: :constant, args: [number]}}, _op), do: number

  defp scalar_constant!(%T{data: %Expr{op: :tensor, args: [tensor]}}, _op), do: Nx.to_number(tensor)

  defp scalar_constant!(%T{} = other, op) do
    raise NxTinygrad.CompileError,
      message: "#{op} requires a compile-time scalar constant value",
      operation: op,
      output_spec: %{shape: shape_of(other), dtype: safe_dtype(other)},
      hint: "only scalar constant fill values are supported"
  end

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
