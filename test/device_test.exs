defmodule ExTinygrad.DeviceTest do
  use ExUnit.Case, async: true

  alias ExTinygrad.Device

  test "KFD+AMD:LLVM maps to tinygrad AMD with KFD interface and LLVM renderer" do
    parsed = Device.parse("KFD+AMD:LLVM")
    assert parsed.tinygrad_device == "AMD"
    assert parsed.interface == "KFD"
    assert parsed.renderer == "LLVM"
    assert parsed.env == %{"AMD_IFACE" => "KFD", "AMD_LLVM" => "1"}
  end

  test "bare AMD defaults to KFD + LLVM (never PCI)" do
    parsed = Device.parse("AMD")
    assert parsed.interface == "KFD"
    assert parsed.env["AMD_IFACE"] == "KFD"
    assert parsed.env["AMD_LLVM"] == "1"
  end

  test "CPU has no special environment" do
    parsed = Device.parse("CPU")
    assert parsed.tinygrad_device == "CPU"
    assert parsed.env == %{}
  end

  test "nil and blank normalize to CPU" do
    assert Device.parse(nil).spec == "CPU"
    assert Device.parse("   ").spec == "CPU"
  end

  test "is case-insensitive" do
    assert Device.parse("kfd+amd:llvm").tinygrad_device == "AMD"
  end
end
