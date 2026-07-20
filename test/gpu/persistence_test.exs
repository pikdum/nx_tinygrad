defmodule NxTinygrad.GPU.PersistenceTest do
  @moduledoc "Device-resident tensors persist and stay immutable across executions."
  use ExUnit.Case, async: false
  @moduletag :gpu

  alias NxTinygrad.Backend

  setup_all do
    NxTinygrad.GPUHelpers.ensure_amd_worker()
    :ok
  end

  test "outputs stay resident on device and can be reused as inputs" do
    x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]]) |> Nx.backend_transfer({Backend, worker: :amd})
    double = NxTinygrad.jit(fn t -> Nx.multiply(t, 2.0) end, worker: :amd, output: :device)

    r1 = double.(x)
    assert %Backend{worker: :amd} = r1.data

    # r1 is a device tensor; feed it back into another compiled call.
    r2 = double.(r1)
    assert Nx.to_flat_list(Nx.backend_transfer(r2)) == [4.0, 8.0, 12.0, 16.0]
  end

  test "a retained output is unchanged after later executions (immutability)" do
    f = NxTinygrad.jit(fn t -> Nx.add(t, 1.0) end, worker: :amd, output: :device)

    a = f.(Nx.tensor([10.0, 20.0, 30.0]))
    snapshot = Nx.to_flat_list(Nx.backend_copy(a, Nx.BinaryBackend))

    # Run several more times; `a` must not change.
    Enum.each(1..5, fn i -> f.(Nx.tensor([i * 1.0, i * 2.0, i * 3.0])) end)

    assert Nx.to_flat_list(Nx.backend_transfer(a)) == snapshot
    assert snapshot == [11.0, 21.0, 31.0]
  end
end
