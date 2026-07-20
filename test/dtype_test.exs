defmodule NxTinygrad.DtypeTest do
  use ExUnit.Case, async: true

  alias NxTinygrad.Dtype

  test "maps the v0.1 required types" do
    assert Dtype.to_name!({:f, 32}) == "f32"
    assert Dtype.to_name!({:s, 32}) == "s32"
    assert Dtype.to_name!({:u, 8}) == "u8"
  end

  test "round-trips names and Nx types" do
    for name <- Dtype.supported_names() do
      type = Dtype.to_nx!(name)
      assert Dtype.to_name!(type) == name
    end
  end

  test "unsupported Nx type raises a compile error" do
    assert_raise NxTinygrad.CompileError, fn -> Dtype.to_name!({:c, 64}) end
  end

  test "unknown name returns an error tuple" do
    assert {:error, _} = Dtype.to_nx("nope")
  end
end
