defmodule ExTinygrad.TelemetryTest do
  use ExUnit.Case, async: false
  import Nx.Defn

  defn(double(x), do: Nx.multiply(x, 2.0))

  test "compile, execute, and transfer emit telemetry events" do
    test_pid = self()

    events = [
      [:ex_tinygrad, :compile, :stop],
      [:ex_tinygrad, :execute, :stop],
      [:ex_tinygrad, :transfer, :download]
    ]

    :telemetry.attach_many(
      "ex-tinygrad-telemetry-test",
      events,
      fn event, measurements, metadata, _ -> send(test_pid, {:telemetry, event, measurements, metadata}) end,
      nil
    )

    on_exit(fn -> :telemetry.detach("ex-tinygrad-telemetry-test") end)

    ExTinygrad.ExecutableCache.clear()

    result = ExTinygrad.jit(&double/1).(Nx.tensor([1.0, 2.0, 3.0]))
    _host = Nx.backend_transfer(result)

    assert_receive {:telemetry, [:ex_tinygrad, :compile, :stop], %{duration: _}, %{kernel_count: _}}
    assert_receive {:telemetry, [:ex_tinygrad, :execute, :stop], %{duration: _}, %{output: :device}}
    assert_receive {:telemetry, [:ex_tinygrad, :transfer, :download], %{bytes: bytes}, _}
    assert bytes == 12
  end
end
