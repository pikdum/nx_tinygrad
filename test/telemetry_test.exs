defmodule NxTinygrad.TelemetryTest do
  use ExUnit.Case, async: false
  import Nx.Defn

  defn(double(x), do: Nx.multiply(x, 2.0))

  test "compile, execute, and transfer emit telemetry events" do
    test_pid = self()

    events = [
      [:nx_tinygrad, :compile, :stop],
      [:nx_tinygrad, :execute, :stop],
      [:nx_tinygrad, :transfer, :download]
    ]

    :telemetry.attach_many(
      "nx-tinygrad-telemetry-test",
      events,
      fn event, measurements, metadata, _ -> send(test_pid, {:telemetry, event, measurements, metadata}) end,
      nil
    )

    on_exit(fn -> :telemetry.detach("nx-tinygrad-telemetry-test") end)

    NxTinygrad.ExecutableCache.clear()

    result = NxTinygrad.jit(&double/1).(Nx.tensor([1.0, 2.0, 3.0]))
    _host = Nx.backend_transfer(result)

    assert_receive {:telemetry, [:nx_tinygrad, :compile, :stop], %{duration: _}, %{kernel_count: _}}
    assert_receive {:telemetry, [:nx_tinygrad, :execute, :stop], %{duration: _}, %{output: :device}}
    assert_receive {:telemetry, [:nx_tinygrad, :transfer, :download], %{bytes: bytes}, _}
    assert bytes == 12
  end
end
