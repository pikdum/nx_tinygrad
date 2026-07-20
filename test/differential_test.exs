defmodule NxTinygrad.CondGraphs do
  @moduledoc false
  import Nx.Defn

  defn tri(t) do
    cond do
      Nx.all(Nx.greater(t, 0)) -> Nx.negate(t)
      Nx.all(Nx.less(t, 0)) -> Nx.multiply(t, 10)
      true -> Nx.multiply(t, 2)
    end
  end
end

defmodule NxTinygrad.WhileGraphs do
  @moduledoc false
  import Nx.Defn

  defn count_up(t) do
    {r, _i} =
      while {acc = t, i = 0}, i < 5 do
        {acc + 1.0, i + 1}
      end

    r
  end

  defn grow_until(t) do
    {r, _} =
      while {acc = t, _c = 0}, Nx.less(Nx.sum(acc), 100.0) do
        {acc * 1.5, 0}
      end

    r
  end

  defn dynamic_slice(t, i) do
    Nx.slice(t, [i, 0], [1, 3])
  end
end

defmodule NxTinygrad.DifferentialTest do
  @moduledoc "Broad differential coverage against Nx.BinaryBackend."
  use ExUnit.Case, async: false

  import NxTinygrad.TestGraphs, only: [assert_close: 2, assert_close: 3]

  test "advertised unary operations match across representative domains" do
    x = Nx.tensor([0.01, 0.1, 0.5, 1.0, 2.0], type: :f32)

    fun = fn t ->
      {
        Nx.negate(t),
        Nx.abs(t),
        Nx.exp(t),
        Nx.expm1(t),
        Nx.log(t),
        Nx.log1p(t),
        Nx.sqrt(t),
        Nx.rsqrt(t),
        Nx.tanh(t),
        Nx.sigmoid(t),
        Nx.sin(t),
        Nx.cos(t),
        Nx.floor(t),
        Nx.ceil(t)
      }
    end

    assert_close(NxTinygrad.jit(fun).(x), fun.(x), atol: 1.0e-6, rtol: 1.0e-5)
  end

  test "binary broadcasting, comparisons, and select match" do
    x = Nx.tensor([[-2.0], [0.5], [3.0]])
    y = Nx.tensor([[0.25, 1.0, 2.0, 4.0]])

    fun = fn a, b ->
      {
        Nx.add(a, b),
        Nx.subtract(a, b),
        Nx.multiply(a, b),
        Nx.divide(a, b),
        Nx.pow(Nx.add(Nx.abs(a), 0.5), b),
        Nx.max(a, b),
        Nx.min(a, b),
        Nx.equal(a, b),
        Nx.not_equal(a, b),
        Nx.less(a, b),
        Nx.less_equal(a, b),
        Nx.greater(a, b),
        Nx.greater_equal(a, b),
        Nx.select(Nx.greater(a, b), a, b)
      }
    end

    assert_close(NxTinygrad.jit(fun).(x, y), fun.(x, y), atol: 1.0e-5, rtol: 1.0e-4)
  end

  test "shape, slicing, concatenation, and reduction operations match" do
    x = Nx.iota({2, 3, 4}, type: :f32) |> Nx.subtract(6.0)

    fun = fn t ->
      reshaped = Nx.reshape(t, {4, 6})

      {
        Nx.transpose(t, axes: [2, 0, 1]),
        reshaped,
        Nx.squeeze(Nx.reshape(t, {1, 2, 3, 4}), axes: [0]),
        Nx.broadcast(Nx.tensor(2.0), {2, 3}),
        Nx.concatenate([reshaped, reshaped], axis: -1),
        Nx.slice(t, [0, 0, 0], [2, 2, 2], strides: [1, 1, 2]),
        Nx.sum(t, axes: [-1]),
        Nx.reduce_max(t, axes: [0], keep_axes: true),
        Nx.reduce_min(t, axes: [1]),
        Nx.all(t),
        Nx.any(t)
      }
    end

    assert_close(NxTinygrad.jit(fun).(x), fun.(x))
  end

  test "all supported wire dtypes execute on CPU" do
    for name <- NxTinygrad.Dtype.supported_names() do
      type = NxTinygrad.Dtype.to_nx!(name)
      x = Nx.iota({2, 3}, type: type)
      fun = fn t -> Nx.add(Nx.multiply(t, 2), 1) end

      assert_close(NxTinygrad.jit(fun).(x), fun.(x), atol: 1.0e-3, rtol: 1.0e-3)
    end
  end

  test "deterministic random elementwise graphs match over varied shapes" do
    {_, _key} =
      Enum.reduce([{1}, {2, 3}, {2, 1, 4}, {4, 3, 2}], {0, Nx.Random.key(42)}, fn shape, {index, key} ->
        {x, key} = Nx.Random.normal(key, shape: shape, type: :f32)
        {y, key} = Nx.Random.uniform(key, shape: shape, type: :f32)

        fun = fn a, b ->
          a
          |> Nx.multiply(b)
          |> Nx.add(0.25)
          |> Nx.tanh()
          |> Nx.subtract(Nx.sin(a))
          |> Nx.sigmoid()
        end

        assert_close(NxTinygrad.jit(fun).(x, y), fun.(x, y), atol: 1.0e-5, rtol: 1.0e-4)
        {index + 1, key}
      end)
  end

  test "scalar and batched dot configurations match" do
    scalar = Nx.tensor(3.0)
    vector = Nx.tensor([1.0, 2.0, 3.0])
    a = Nx.iota({2, 3, 4}, type: :f32)
    b = Nx.iota({2, 4, 5}, type: :f32)

    scalar_fun = fn x, y -> Nx.multiply(x, y) end
    batched_fun = fn x, y -> Nx.dot(x, [2], [0], y, [1], [0]) end

    assert_close(NxTinygrad.jit(scalar_fun).(scalar, Nx.sum(vector)), scalar_fun.(scalar, Nx.sum(vector)))
    assert_close(NxTinygrad.jit(batched_fun).(a, b), batched_fun.(a, b), atol: 1.0e-4, rtol: 1.0e-4)
  end

  test "NaN, infinity, and signed-zero behavior matches for basic unary ops" do
    x = Nx.tensor([:neg_infinity, -1.0, -0.0, 0.0, 1.0, :infinity, :nan], type: :f32)
    fun = fn t -> {Nx.negate(t), Nx.abs(t), Nx.exp(t)} end
    actual = NxTinygrad.jit(fun, output: :host).(x)
    expected = fun.(x)

    for {a, e} <- Enum.zip(Tuple.to_list(actual), Tuple.to_list(expected)) do
      assert Nx.to_flat_list(a) == Nx.to_flat_list(e)
    end

    assert Nx.to_binary(elem(actual, 1)) == Nx.to_binary(elem(expected, 1))
  end

  test "integer overflow follows Nx output dtype semantics" do
    x = Nx.tensor([127, -128, 126, -127], type: :s8)
    y = Nx.tensor([1, -1, 2, -2], type: :s8)
    fun = fn a, b -> {Nx.add(a, b), Nx.subtract(a, b), Nx.multiply(a, b)} end

    assert_close(NxTinygrad.jit(fun).(x, y), fun.(x, y))
  end

  test "extended transcendental and sign unary ops match on valid domains" do
    broad = Nx.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], type: :f32)
    unit = Nx.tensor([-0.9, -0.5, 0.0, 0.5, 0.9], type: :f32)
    ge1 = Nx.tensor([1.0, 1.5, 2.0, 5.0], type: :f32)

    broad_fun = fn t ->
      {Nx.sinh(t), Nx.cosh(t), Nx.asinh(t), Nx.atan(t), Nx.tan(t), Nx.erf(t), Nx.cbrt(t), Nx.sign(t)}
    end

    unit_fun = fn t -> {Nx.asin(t), Nx.acos(t), Nx.atanh(t)} end
    ge1_fun = fn t -> Nx.acosh(t) end

    assert_close(NxTinygrad.jit(broad_fun).(broad), broad_fun.(broad), atol: 1.0e-5, rtol: 1.0e-4)
    assert_close(NxTinygrad.jit(unit_fun).(unit), unit_fun.(unit), atol: 1.0e-5, rtol: 1.0e-4)
    assert_close(NxTinygrad.jit(ge1_fun).(ge1), ge1_fun.(ge1), atol: 1.0e-5, rtol: 1.0e-4)
  end

  test "is_nan and is_infinity predicates match exactly" do
    x = Nx.tensor([:neg_infinity, -1.0, -0.0, 0.0, 1.0, :infinity, :nan], type: :f32)
    fun = fn t -> {Nx.is_nan(t), Nx.is_infinity(t)} end

    assert_close(NxTinygrad.jit(fun).(x), fun.(x))
  end

  test "bitwise and shift ops match on signed integers" do
    a = Nx.tensor([0, 1, 5, -1, -8, 255], type: :s32)
    b = Nx.tensor([3, 2, 6, 1, 1, 128], type: :s32)
    shift = Nx.tensor([0, 1, 2, 0, 1, 3], type: :s32)

    fun = fn a, b, s ->
      {
        Nx.bitwise_not(a),
        Nx.bitwise_and(a, b),
        Nx.bitwise_or(a, b),
        Nx.bitwise_xor(a, b),
        Nx.left_shift(a, s),
        Nx.right_shift(a, s)
      }
    end

    assert_close(NxTinygrad.jit(fun).(a, b, shift), fun.(a, b, shift))
  end

  test "logical ops treat nonzero as true and yield u8" do
    a = Nx.tensor([0, 1, 0, 5, -2], type: :s32)
    b = Nx.tensor([0, 0, 3, 2, 7], type: :s32)
    fun = fn a, b -> {Nx.logical_and(a, b), Nx.logical_or(a, b), Nx.logical_xor(a, b)} end

    assert_close(NxTinygrad.jit(fun).(a, b), fun.(a, b))
  end

  test "product reduction matches over axes and full tensor" do
    x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 0.5, 2.0]], type: :f32)

    fun = fn t ->
      {Nx.product(t), Nx.product(t, axes: [1]), Nx.product(t, axes: [0], keep_axes: true)}
    end

    assert_close(NxTinygrad.jit(fun).(x), fun.(x), atol: 1.0e-5, rtol: 1.0e-4)
  end

  test "remainder and quotient match Nx truncated-division semantics" do
    fa = Nx.tensor([5.3, -5.3, 5.3, -5.3, 7.0], type: :f32)
    fb = Nx.tensor([2.0, 2.0, -2.0, -2.0, 3.0], type: :f32)
    ia = Nx.tensor([7, -7, 7, -7, 9], type: :s32)
    ib = Nx.tensor([2, 2, -2, -2, 4], type: :s32)

    float_fun = fn a, b -> Nx.remainder(a, b) end
    int_fun = fn a, b -> {Nx.remainder(a, b), Nx.quotient(a, b)} end

    assert_close(NxTinygrad.jit(float_fun).(fa, fb), float_fun.(fa, fb))
    assert_close(NxTinygrad.jit(int_fun).(ia, ib), int_fun.(ia, ib))
  end

  test "atan2 matches across all quadrants and axes" do
    y = Nx.tensor([1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0], type: :f32)
    x = Nx.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0], type: :f32)
    fun = fn a, b -> Nx.atan2(a, b) end

    assert_close(NxTinygrad.jit(fun).(y, x), fun.(y, x), atol: 1.0e-5, rtol: 1.0e-4)
  end

  test "reverse matches over selected and all axes" do
    x = Nx.iota({2, 3}, type: :f32)
    fun = fn t -> {Nx.reverse(t), Nx.reverse(t, axes: [1]), Nx.reverse(t, axes: [0])} end

    assert_close(NxTinygrad.jit(fun).(x), fun.(x))
  end

  test "argmax and argmin match across axis, flatten, keep_axis, and tie-break" do
    x = Nx.tensor([[1.0, 5.0, 2.0, 5.0], [9.0, 0.0, 9.0, 3.0]], type: :f32)

    fun = fn t ->
      {
        Nx.argmax(t, axis: 1),
        Nx.argmin(t, axis: 1),
        Nx.argmax(t),
        Nx.argmax(t, axis: 0, keep_axis: true),
        Nx.argmax(t, axis: 1, tie_break: :high),
        Nx.argmax(t, tie_break: :high)
      }
    end

    assert_close(NxTinygrad.jit(fun).(x), fun.(x))
  end

  test "cumulative ops lower through their pure default expression" do
    x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], type: :f32)

    fun = fn t ->
      {
        Nx.cumulative_sum(t, axis: 1),
        Nx.cumulative_sum(t, axis: 0),
        Nx.cumulative_product(t, axis: 1),
        Nx.cumulative_max(t, axis: 1),
        Nx.cumulative_min(t, axis: 1),
        Nx.cumulative_sum(t, axis: 1, reverse: true)
      }
    end

    assert_close(NxTinygrad.jit(fun).(x), fun.(x), atol: 1.0e-5, rtol: 1.0e-4)
  end

  test "pad matches with edge padding and finite/-infinity fill" do
    x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)

    fun = fn t ->
      {
        Nx.pad(t, 0.0, [{1, 1, 0}, {2, 0, 0}]),
        Nx.pad(t, 7.5, [{0, 0, 0}, {1, 1, 0}]),
        Nx.pad(t, :neg_infinity, [{1, 0, 0}, {0, 0, 0}])
      }
    end

    actual = NxTinygrad.jit(fun, output: :host).(x)
    expected = fun.(x)

    for {a, e} <- Enum.zip(Tuple.to_list(actual), Tuple.to_list(expected)) do
      assert Nx.to_flat_list(a) == Nx.to_flat_list(e)
    end
  end

  test "sort and argsort match ascending and descending" do
    x = Nx.tensor([[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]], type: :f32)

    fun = fn t ->
      {
        Nx.sort(t, axis: 1),
        Nx.sort(t, axis: 1, direction: :desc),
        Nx.sort(t, axis: 0),
        Nx.argsort(t, axis: 1),
        Nx.argsort(t, axis: 1, direction: :desc)
      }
    end

    assert_close(NxTinygrad.jit(fun).(x), fun.(x))
  end

  test "gather, take, and take_along_axis match Nx indexing semantics" do
    t = Nx.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]], type: :f32)
    coords = Nx.tensor([[0, 0], [2, 1], [1, 2]], type: :s64)
    row_idx = Nx.tensor([[0], [2]], type: :s64)
    take_idx = Nx.tensor([2, 0, 1, 2], type: :s64)
    along_idx = Nx.tensor([[0, 2, 1], [2, 2, 0], [1, 0, 0]], type: :s64)

    fun = fn t, coords, row_idx, take_idx, along_idx ->
      {
        Nx.gather(t, coords),
        Nx.gather(t, row_idx, axes: [0]),
        Nx.take(t, take_idx, axis: 0),
        Nx.take(t, take_idx, axis: 1),
        Nx.take_along_axis(t, along_idx, axis: 1)
      }
    end

    args = [t, coords, row_idx, take_idx, along_idx]
    assert_close(apply(NxTinygrad.jit(fun), args), apply(fun, args))
  end

  test "embedding lookup via take matches (2-D index into rows)" do
    embeddings = Nx.iota({6, 4}, type: :f32)
    token_ids = Nx.tensor([[0, 3, 5], [2, 2, 1]], type: :s64)

    fun = fn table, ids -> Nx.take(table, ids, axis: 0) end

    assert_close(
      NxTinygrad.jit(fun).(embeddings, token_ids),
      fun.(embeddings, token_ids)
    )
  end

  test "clip, stack, tile, round, and eye match" do
    x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
    r = Nx.tensor([-2.5, -1.6, -0.5, 0.5, 1.5, 2.5, 3.4], type: :f32)

    xfun = fn t ->
      {
        Nx.clip(t, 2.0, 5.0),
        Nx.stack([t, Nx.multiply(t, 2.0)]),
        Nx.stack([t, Nx.multiply(t, 2.0)], axis: 1),
        Nx.tile(t, [2, 1])
      }
    end

    rfun = fn t -> Nx.round(t) end
    eye_fun = fn _ -> {Nx.eye(3), Nx.eye({2, 4}), Nx.eye({2, 3, 3})} end

    assert_close(NxTinygrad.jit(xfun).(x), xfun.(x))
    assert_close(NxTinygrad.jit(rfun).(r), rfun.(r))
    assert_close(NxTinygrad.jit(eye_fun).(x), eye_fun.(x))
  end

  test "erfc matches over a moderate domain" do
    x = Nx.tensor([-2.0, -1.0, -0.3, 0.0, 0.3, 1.0, 2.0], type: :f32)
    fun = fn t -> Nx.erfc(t) end

    assert_close(NxTinygrad.jit(fun).(x), fun.(x), atol: 1.0e-5, rtol: 1.0e-3)
  end

  test "window pooling reductions match with stride, padding, and dilation" do
    x = Nx.iota({1, 1, 5, 5}, type: :f32) |> Nx.add(1.0)
    small = Nx.iota({1, 1, 4, 4}, type: :f32) |> Nx.remainder(3.0) |> Nx.add(1.0)

    fun = fn t, s ->
      {
        Nx.window_max(t, {1, 1, 2, 2}, strides: [1, 1, 2, 2]),
        Nx.window_sum(t, {1, 1, 2, 2}, strides: [1, 1, 1, 1]),
        Nx.window_min(t, {1, 1, 3, 3}, strides: [1, 1, 2, 2]),
        Nx.window_max(t, {1, 1, 2, 2}, strides: [1, 1, 1, 1], padding: [{0, 0}, {0, 0}, {1, 0}, {1, 0}]),
        Nx.window_max(t, {1, 1, 2, 2}, strides: [1, 1, 1, 1], window_dilations: [1, 1, 2, 2]),
        Nx.window_product(s, {1, 1, 2, 2}, strides: [1, 1, 2, 2])
      }
    end

    assert_close(NxTinygrad.jit(fun).(x, small), fun.(x, small), atol: 1.0e-4, rtol: 1.0e-4)
  end

  test "top_k, determinant, and integer bit ops match" do
    v = Nx.tensor([3.0, 1.0, 4.0, 1.5, 5.0, 9.0], type: :f32)
    ints = Nx.tensor([0, 1, 8, 255, 1024, -1], type: :s32)
    spd = Nx.tensor([[4.0, 1.0], [1.0, 3.0]], type: :f32)

    fun = fn v, ints, spd ->
      {tv, ti} = Nx.top_k(v, k: 3)
      {tv, ti, Nx.count_leading_zeros(ints), Nx.population_count(ints), Nx.LinAlg.determinant(spd)}
    end

    args = [v, ints, spd]
    assert_close(apply(NxTinygrad.jit(fun), args), apply(fun, args))
  end

  test "erf_inv matches within approximation tolerance" do
    x = Nx.tensor([-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9], type: :f32)
    fun = fn t -> Nx.erf_inv(t) end

    assert_close(NxTinygrad.jit(fun).(x), fun.(x), atol: 1.0e-4, rtol: 1.0e-3)
  end

  test "while loops (fixed count and data-dependent) match Nx" do
    x = Nx.tensor([10.0, 20.0, 30.0], type: :f32)
    assert_close(NxTinygrad.jit(&NxTinygrad.WhileGraphs.count_up/1).(x), NxTinygrad.WhileGraphs.count_up(x))

    assert_close(
      NxTinygrad.jit(&NxTinygrad.WhileGraphs.grow_until/1).(x),
      NxTinygrad.WhileGraphs.grow_until(x)
    )
  end

  test "window_scatter (select-and-scatter) matches Nx incl. overlap and ties" do
    {x, _} = Nx.Random.normal(Nx.Random.key(1), shape: {1, 1, 4, 4}, type: :f32)
    s2 = Nx.iota({1, 1, 2, 2}, type: :f32) |> Nx.add(1.0)
    s3 = Nx.iota({1, 1, 3, 3}, type: :f32) |> Nx.add(1.0)

    fun = fn t, s2, s3 ->
      {
        Nx.window_scatter_max(t, s2, 0.0, {1, 1, 2, 2}, strides: [1, 1, 2, 2]),
        Nx.window_scatter_max(t, s3, 0.0, {1, 1, 2, 2}, strides: [1, 1, 1, 1]),
        Nx.window_scatter_min(t, s2, 0.0, {1, 1, 2, 2}, strides: [1, 1, 2, 2])
      }
    end

    args = [x, s2, s3]
    assert_close(apply(NxTinygrad.jit(fun), args), apply(fun, args))
  end

  test "max-pool backward gradient matches Nx" do
    {img, _} = Nx.Random.normal(Nx.Random.key(5), shape: {2, 3, 4, 4}, type: :f32)

    fun = fn t ->
      Nx.Defn.value_and_grad(t, fn t -> Nx.sum(Nx.window_max(t, {1, 1, 2, 2}, strides: [1, 1, 2, 2])) end)
    end

    assert_close(apply(NxTinygrad.jit(fun), [img]), apply(fun, [img]), atol: 1.0e-4, rtol: 1.0e-4)
  end

  test "cholesky (iterative linalg via while) matches Nx" do
    spd = Nx.tensor([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]], type: :f32)
    fun = fn t -> Nx.LinAlg.cholesky(t) end

    assert_close(NxTinygrad.jit(fun).(spd), fun.(spd), atol: 1.0e-4, rtol: 1.0e-3)
  end

  test "dynamic slice with a runtime start index matches Nx" do
    t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], type: :f32)

    for i <- [0, 2, 3] do
      idx = Nx.tensor(i, type: :s32)

      assert_close(
        NxTinygrad.jit(&NxTinygrad.WhileGraphs.dynamic_slice/2).(t, idx),
        NxTinygrad.WhileGraphs.dynamic_slice(t, idx)
      )
    end
  end

  test "cond lowers to predicated selects across all branches" do
    for input <- [Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([-1.0, -2.0, -3.0]), Nx.tensor([-1.0, 2.0, -3.0])] do
      assert_close(NxTinygrad.jit(&NxTinygrad.CondGraphs.tri/1).(input), NxTinygrad.CondGraphs.tri(input))
    end
  end

  test "high-level composites (mean, variance, std, logsumexp) match" do
    x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], type: :f32)

    fun = fn t ->
      {
        Nx.mean(t, axes: [1]),
        Nx.variance(t, axes: [1]),
        Nx.standard_deviation(t),
        Nx.logsumexp(t, axes: [1])
      }
    end

    assert_close(NxTinygrad.jit(fun).(x), fun.(x), atol: 1.0e-5, rtol: 1.0e-4)
  end

  test "conv gradients (wrt kernel and input) match Nx" do
    {input, _} = Nx.Random.normal(Nx.Random.key(7), shape: {2, 3, 7, 7}, type: :f32)
    {kernel, _} = Nx.Random.normal(Nx.Random.key(8), shape: {4, 3, 3, 3}, type: :f32)

    gk = fn i, k ->
      Nx.Defn.value_and_grad(k, fn k -> Nx.sum(Nx.conv(i, k, strides: [2, 2], padding: [{1, 1}, {1, 1}])) end)
    end

    gi = fn i, k ->
      Nx.Defn.value_and_grad(i, fn i -> Nx.sum(Nx.conv(i, k, strides: [2, 2], padding: [{1, 1}, {1, 1}])) end)
    end

    args = [input, kernel]
    assert_close(apply(NxTinygrad.jit(gk), args), apply(gk, args), atol: 1.0e-3, rtol: 1.0e-3)
    assert_close(apply(NxTinygrad.jit(gi), args), apply(gi, args), atol: 1.0e-3, rtol: 1.0e-3)
  end

  test "bitcast reinterprets bits without value conversion" do
    x = Nx.tensor([1.0, -2.0, 3.5, 0.0], type: :f32)
    fun = fn t -> {Nx.bitcast(t, :s32), Nx.bitcast(Nx.bitcast(t, :s32), :f32)} end

    actual = NxTinygrad.jit(fun, output: :host).(x)
    expected = fun.(x)

    for {a, e} <- Enum.zip(Tuple.to_list(actual), Tuple.to_list(expected)) do
      assert Nx.to_flat_list(a) == Nx.to_flat_list(e)
    end
  end

  test "put_slice, indexed_put, and indexed_add match (incl. duplicate indices)" do
    t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], type: :f32)
    block = Nx.tensor([[90.0], [91.0]], type: :f32)
    coords = Nx.tensor([[0, 0], [2, 1], [0, 0]], type: :s64)
    updates = Nx.tensor([10.0, 20.0, 100.0], type: :f32)
    put_coords = Nx.tensor([[1, 1], [0, 2]], type: :s64)
    put_vals = Nx.tensor([55.0, 66.0], type: :f32)

    fun = fn t, block, coords, updates, put_coords, put_vals ->
      {
        Nx.put_slice(t, [0, 1], block),
        Nx.indexed_add(t, coords, updates),
        Nx.indexed_put(t, put_coords, put_vals)
      }
    end

    args = [t, block, coords, updates, put_coords, put_vals]
    assert_close(apply(NxTinygrad.jit(fun), args), apply(fun, args))
  end

  test "conv matches Nx across stride, padding, dilation, and groups" do
    {input, _k} = Nx.Random.normal(Nx.Random.key(7), shape: {2, 4, 8, 8}, type: :f32)
    {kernel, _k} = Nx.Random.normal(Nx.Random.key(8), shape: {6, 4, 3, 3}, type: :f32)
    {grouped_kernel, _k} = Nx.Random.normal(Nx.Random.key(9), shape: {6, 2, 3, 3}, type: :f32)

    fun = fn i, k, gk ->
      {
        Nx.conv(i, k, strides: [1, 1], padding: [{1, 1}, {1, 1}]),
        Nx.conv(i, k, strides: [2, 2], padding: :valid),
        Nx.conv(i, k, strides: [1, 1], padding: [{2, 1}, {0, 2}]),
        Nx.conv(i, k, strides: [1, 1], kernel_dilation: [2, 2]),
        Nx.conv(i, gk, strides: [1, 1], feature_group_size: 2)
      }
    end

    args = [input, kernel, grouped_kernel]
    assert_close(apply(NxTinygrad.jit(fun), args), apply(fun, args), atol: 1.0e-4, rtol: 1.0e-3)
  end
end
