defmodule ExTinygrad.WorkerTest do
  @moduledoc "Integration tests that exercise the full Elixir <-> Python path on CPU."
  use ExUnit.Case, async: false

  alias ExTinygrad.Worker

  setup do
    # The application boots a :default CPU worker; make sure it's up.
    assert Worker.whereis(:default) != nil, "default worker not running"
    :ok
  end

  test "hello handshake reports protocol version, device, and generation" do
    hello = Worker.info(:default)
    assert hello["protocol_version"] == 1
    assert hello["device"] == "CPU"
    assert hello["generation"] >= 1
    assert is_binary(hello["tinygrad_version"])
  end

  test "device_info runs the smoke computation and loads no ROCm" do
    {:ok, info, []} = Worker.request(:default, "device_info", %{})
    assert info["usable"] == true
    assert info["selected"] == "CPU"
    assert Enum.all?(Map.values(info["rocm_libraries_loaded"]), &(&1 == false))
  end

  test "upload -> download round-trips f32 bytes exactly" do
    data = <<1.0::float-little-32, 2.0::float-little-32, 3.0::float-little-32>>
    {:ok, %{"id" => id}, []} = Worker.request(:default, "upload", %{"shape" => [3], "dtype" => "f32"}, [data])
    assert is_integer(id)

    {:ok, meta, [out]} = Worker.request(:default, "download", %{"id" => id})
    assert meta["shape"] == [3]
    assert meta["dtype"] == "f32"
    assert out == data
  end

  test "stats reflect buffers and byte counters" do
    data = <<0::float-little-32, 0::float-little-32>>
    {:ok, %{"id" => id}, []} = Worker.request(:default, "upload", %{"shape" => [2], "dtype" => "f32"}, [data])

    {:ok, stats, []} = Worker.request(:default, "stats", %{})
    assert stats["buffer_count"] >= 1
    assert stats["upload_bytes"] >= 8

    {:ok, %{"released" => n}, []} = Worker.request(:default, "release", %{"ids" => [id]})
    assert n == 1

    # release is idempotent
    {:ok, %{"released" => 0}, []} = Worker.request(:default, "release", %{"ids" => [id]})
  end

  test "synchronize succeeds" do
    assert {:ok, %{}, []} = Worker.request(:default, "synchronize", %{})
  end

  test "unknown command yields a structured protocol error" do
    assert {:error, %ExTinygrad.WorkerError{class: "ProtocolError"}} =
             Worker.request(:default, "bogus_command", %{})
  end

  test "downloading a released/unknown buffer raises StaleReference" do
    assert {:error, %ExTinygrad.WorkerError{class: "StaleReference"}} =
             Worker.request(:default, "download", %{"id" => 999_999})
  end
end
