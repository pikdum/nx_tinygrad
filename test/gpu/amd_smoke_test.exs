defmodule ExTinygrad.GPU.AmdSmokeTest do
  @moduledoc "AMD RX 7900 XT via native KFD + LLVM, no ROCm."
  use ExUnit.Case, async: false
  @moduletag :gpu

  import ExTinygrad.TestGraphs
  alias ExTinygrad.TestGraphs, as: G

  setup_all do
    info = ExTinygrad.GPUHelpers.ensure_amd_worker()
    {:ok, info: info}
  end

  test "device_info reports a real AMD GPU through KFD+AMD:LLVM", %{info: info} do
    assert info["usable"] == true
    assert info["interface"] == "KFD"
    assert info["renderer"] == "LLVM"
    assert info["architecture"] =~ ~r/^gfx/
    assert info["selected"] == "AMD"
  end

  test "no ROCm/HIP/comgr libraries are loaded in the worker", %{info: info} do
    loaded = info["rocm_libraries_loaded"]
    assert Enum.all?(Map.values(loaded), &(&1 == false)), "ROCm libraries loaded: #{inspect(loaded)}"
  end

  test "f32 elementwise parity" do
    x = Nx.tensor([[1.0, -2.0], [3.0, 0.5]])
    y = Nx.tensor([[0.5, 4.0], [-1.0, 2.0]])
    assert_close(ExTinygrad.jit(&G.elementwise/2, worker: :amd).(x, y), G.elementwise(x, y))
  end

  test "f32 matmul parity" do
    x = Nx.iota({8, 16}, type: :f32) |> Nx.divide(16.0)
    w = Nx.iota({16, 4}, type: :f32) |> Nx.divide(16.0)
    b = Nx.tensor([0.1, 0.2, 0.3, 0.4])
    assert_close(ExTinygrad.jit(&G.matmul/3, worker: :amd).(x, w, b), G.matmul(x, w, b))
  end

  test "softmax parity (reduce_max/exp/sub/sum/divide)" do
    x = Nx.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    assert_close(ExTinygrad.jit(&G.softmax/1, worker: :amd).(x), G.softmax(x))
  end

  test "MLP value_and_grad parity on GPU" do
    w = Nx.tensor([[0.1, -0.2], [0.3, 0.05], [-0.1, 0.2]])
    x = Nx.tensor([[0.5, -0.3, 0.8], [0.1, 0.2, -0.4]])
    t = Nx.tensor([[0.3, -0.2], [0.1, 0.4]])

    {v, g} = ExTinygrad.jit(&G.linear_value_and_grad/3, worker: :amd).(w, x, t)
    {ev, eg} = G.linear_value_and_grad(w, x, t)
    assert_close(v, ev, atol: 1.0e-4, rtol: 1.0e-3)
    assert_close(g, eg, atol: 1.0e-4, rtol: 1.0e-3)
  end
end
