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
end
