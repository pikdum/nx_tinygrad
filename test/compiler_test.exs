defmodule NxTinygrad.CompilerTest do
  @moduledoc "End-to-end CPU parity against Nx.BinaryBackend through the full worker path."
  use ExUnit.Case, async: false

  import NxTinygrad.TestGraphs
  alias NxTinygrad.TestGraphs, as: G

  setup do
    assert NxTinygrad.Worker.whereis(:default) != nil
    :ok
  end

  test "elementwise chain" do
    x = Nx.tensor([[1.0, -2.0], [3.0, 0.5]])
    y = Nx.tensor([[0.5, 4.0], [-1.0, 2.0]])
    assert_close(NxTinygrad.jit(&G.elementwise/2).(x, y), G.elementwise(x, y))
  end

  test "broadcasting" do
    x = Nx.iota({2, 3}, type: :f32)
    b = Nx.tensor([10.0, 20.0, 30.0])
    assert_close(NxTinygrad.jit(&G.broadcasting/2).(x, b), G.broadcasting(x, b))
  end

  test "reduction (with and without keep_axes)" do
    x = Nx.iota({3, 4}, type: :f32)
    assert_close(NxTinygrad.jit(&G.reduction/1).(x), G.reduction(x))
    assert_close(NxTinygrad.jit(&G.reduce_keep/1).(x), G.reduce_keep(x))
  end

  test "matrix multiplication" do
    x = Nx.iota({2, 3}, type: :f32)
    w = Nx.iota({3, 4}, type: :f32)
    b = Nx.tensor([1.0, 2.0, 3.0, 4.0])
    assert_close(NxTinygrad.jit(&G.matmul/3).(x, w, b), G.matmul(x, w, b))
  end

  test "comparison + select" do
    x = Nx.tensor([[1.0, 5.0], [3.0, 2.0]])
    y = Nx.tensor([[4.0, 4.0], [4.0, 4.0]])
    assert_close(NxTinygrad.jit(&G.comparison/2).(x, y), G.comparison(x, y))
  end

  test "shape ops (transpose/reshape)" do
    x = Nx.iota({2, 3}, type: :f32)
    assert_close(NxTinygrad.jit(&G.shapes/1).(x), G.shapes(x))
  end

  test "softmax (reduce_max, exp, sub, sum, divide)" do
    x = Nx.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    assert_close(NxTinygrad.jit(&G.softmax/1).(x), G.softmax(x))
  end

  test "expm1 and log1p preserve small values" do
    x = Nx.tensor([1.0e-8, -1.0e-8, 1.0e-6, -1.0e-6, 1.0e-3], type: :f32)

    assert_close(NxTinygrad.jit(&Nx.expm1/1).(x), Nx.expm1(x), atol: 1.0e-12, rtol: 1.0e-6)
    assert_close(NxTinygrad.jit(&Nx.log1p/1).(x), Nx.log1p(x), atol: 1.0e-12, rtol: 1.0e-6)
  end

  test "multiple outputs (tuple)" do
    x = Nx.iota({2, 3}, type: :f32)
    {a, b} = NxTinygrad.jit(&G.multi_output/1).(x)
    {ea, eb} = G.multi_output(x)
    assert_close(a, ea)
    assert_close(b, eb)
  end

  test "nested output container (map + tuple + list)" do
    x = Nx.iota({2, 2}, type: :f32)
    y = Nx.iota({2, 2}, type: :f32) |> Nx.add(1.0)
    got = NxTinygrad.jit(&G.nested_container/2).(x, y)
    exp = G.nested_container(x, y)
    assert_close(got.sum, exp.sum)
    {gt, {gl}} = got.parts
    {et, {el}} = exp.parts
    assert_close(gt, et)
    assert_close(gl, el)
  end

  test "repeated input tensor passed to both operands" do
    x = Nx.tensor([1.0, 2.0, 3.0])
    assert_close(NxTinygrad.jit(&G.repeated_input/1).(x), G.repeated_input(x))
  end

  test "device: option routes to a worker for that device" do
    x = Nx.iota({2, 3}, type: :f32)
    result = NxTinygrad.jit(&G.reduction/1, device: "CPU").(x)
    assert_close(result, G.reduction(x))
  end

  test "__to_backend__ resolves the same worker execution uses (weight residency)" do
    # Regression: __to_backend__ used to drop :device and always leave the
    # worker defaulted, so weights preallocated via Nx.backend_copy /
    # Bumblebee's `preallocate_params` landed on the wrong worker for a
    # non-default device. They were then re-uploaded as one multi-GB blob every
    # call, overflowing the transport frame (this is what blocked Stable
    # Diffusion). It must route the same way the executable does.
    assert {NxTinygrad.Backend, be_opts} = NxTinygrad.Compiler.__to_backend__([])
    assert Keyword.fetch!(be_opts, :worker) == :default

    assert {NxTinygrad.Backend, [worker: :some_worker]} =
             NxTinygrad.Compiler.__to_backend__(worker: :some_worker)

    device = NxTinygrad.Config.device()
    assert {NxTinygrad.Backend, be_opts} = NxTinygrad.Compiler.__to_backend__(device: device)
    assert Keyword.fetch!(be_opts, :worker) == NxTinygrad.WorkerSupervisor.worker_for_device(device)
  end

  test "a weight preallocated to the compiler backend is reused resident across a jit call" do
    # Mirror Bumblebee's preallocate_params flow: copy inputs to the compiler's
    # backend once, then run a compiled function on the same device. The
    # resident buffer must be passed by handle and give correct results — the
    # mechanism that makes multi-GB models (Stable Diffusion) runnable.
    x = Nx.iota({2, 3}, type: :f32)
    {backend, backend_opts} = NxTinygrad.Compiler.__to_backend__(device: "CPU")
    resident = Nx.backend_copy(x, {backend, backend_opts})

    assert %NxTinygrad.Backend{} = resident.data
    assert_close(NxTinygrad.jit(&Nx.negate/1, device: "CPU").(resident), Nx.negate(x))
  end

  test "compiling twice yields equal results (idempotent)" do
    x = Nx.iota({2, 3}, type: :f32)
    r1 = NxTinygrad.jit(&G.reduction/1).(x)
    r2 = NxTinygrad.jit(&G.reduction/1).(x)
    assert_close(r1, r2)
  end

  test "invalid output mode is rejected before worker execution" do
    before = NxTinygrad.worker_stats()["execute_count"]

    assert_raise ArgumentError, ~r/expected :device or :host/, fn ->
      NxTinygrad.jit(&G.reduction/1, output: :elsewhere).(Nx.iota({2, 3}, type: :f32))
    end

    assert NxTinygrad.worker_stats()["execute_count"] == before
  end
end
