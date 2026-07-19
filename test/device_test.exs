defmodule ExTinygrad.DeviceTest do
  use ExUnit.Case, async: true

  alias ExTinygrad.Device

  test "KFD+AMD:LLVM is a native tinygrad DEV string; backend is AMD" do
    parsed = Device.parse("KFD+AMD:LLVM")
    assert parsed.tinygrad_device == "AMD"
    assert parsed.interface == "KFD"
    assert parsed.renderer == "LLVM"
    assert parsed.dev == "KFD+AMD:LLVM"
    assert parsed.env == %{}
  end

  test "bare AMD defaults to KFD + LLVM (never PCI)" do
    parsed = Device.parse("AMD")
    assert parsed.interface == "KFD"
    assert parsed.dev == "KFD+AMD:LLVM"
  end

  test "CPU maps to DEV=CPU with no extra environment" do
    parsed = Device.parse("CPU")
    assert parsed.tinygrad_device == "CPU"
    assert parsed.dev == "CPU"
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
