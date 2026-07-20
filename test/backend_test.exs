defmodule NxTinygrad.BackendTest do
  @moduledoc "Device-resident tensor backend."
  use ExUnit.Case, async: false

  alias NxTinygrad.Backend

  defp to_device(tensor), do: Nx.backend_transfer(tensor, {Backend, worker: :default})

  test "from_binary/to_binary round-trips through the worker" do
    t = Nx.tensor([1.0, 2.0, 3.0, 4.0]) |> to_device()
    assert %Backend{} = t.data

    assert Nx.to_binary(t) ==
             <<1.0::float-little-32, 2.0::float-little-32, 3.0::float-little-32, 4.0::float-little-32>>
  end

  test "backend_transfer moves data back to the host" do
    t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]]) |> to_device()
    host = Nx.backend_transfer(t)
    assert Nx.type(host) == {:f, 32}
    assert Nx.to_flat_list(host) == [1.0, 2.0, 3.0, 4.0]
  end

  test "backend_copy leaves the source usable" do
    t = Nx.tensor([1.0, 2.0, 3.0]) |> to_device()
    copy = Nx.backend_copy(t, Nx.BinaryBackend)
    assert Nx.to_flat_list(copy) == [1.0, 2.0, 3.0]
    # original still readable
    assert Nx.to_flat_list(Nx.backend_transfer(t)) == [1.0, 2.0, 3.0]
  end

  test "inspect downloads a bounded representation" do
    t = Nx.tensor([1.0, 2.0, 3.0]) |> to_device()
    rendered = inspect(t)
    assert rendered =~ "1.0"
    assert rendered =~ "3.0"
  end

  test "eager operations raise instead of falling back" do
    t = Nx.tensor([1.0, 2.0, 3.0]) |> to_device()

    assert_raise NxTinygrad.Error, ~r/eager operations are not supported/, fn ->
      Nx.add(t, t)
    end

    assert_raise NxTinygrad.Error, ~r/eager operations are not supported/, fn ->
      Nx.exp(t)
    end
  end

  test "deallocate releases the handle" do
    t = Nx.tensor([1.0, 2.0]) |> to_device()
    assert Nx.backend_deallocate(t) == :ok
  end

  test "jit with output: :device returns device tensors" do
    before = NxTinygrad.worker_stats()["immutable_copy_fallback"]
    x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = NxTinygrad.jit(fn t -> Nx.multiply(t, 2.0) end, output: :device).(x)
    assert %Backend{} = result.data
    assert Nx.to_flat_list(Nx.backend_transfer(result)) == [2.0, 4.0, 6.0, 8.0]
    assert NxTinygrad.worker_stats()["immutable_copy_fallback"] > before
  end

  test "jit with output: :host returns BinaryBackend tensors" do
    x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = NxTinygrad.jit(fn t -> Nx.multiply(t, 2.0) end, output: :host).(x)
    assert result.data.__struct__ == Nx.BinaryBackend
    assert Nx.to_flat_list(result) == [2.0, 4.0, 6.0, 8.0]
  end

  test "device-resident input is reused as a handle (no re-upload)" do
    x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]]) |> to_device()
    double = NxTinygrad.jit(fn t -> Nx.multiply(t, 2.0) end, output: :device)
    r = double.(x)
    assert Nx.to_flat_list(Nx.backend_transfer(r)) == [2.0, 4.0, 6.0, 8.0]
  end
end
