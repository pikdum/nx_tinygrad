defmodule NxTinygrad.ProtocolTest do
  use ExUnit.Case, async: true

  alias NxTinygrad.Protocol

  defp roundtrip(req_id, meta, blobs) do
    {:ok, decoded} = meta |> encode(req_id, blobs) |> Protocol.decode()
    decoded
  end

  defp encode(meta, req_id, blobs), do: IO.iodata_to_binary(Protocol.encode(req_id, meta, blobs))

  test "round-trips metadata with no blobs" do
    meta = %{"command" => "hello", "args" => %{"protocol_version" => 1}}
    assert {42, ^meta, []} = roundtrip(42, meta, [])
  end

  test "round-trips multiple blobs preserving order and bytes" do
    blobs = [<<1, 2, 3>>, <<>>, :binary.copy(<<0xAB>>, 1000)]
    meta = %{"command" => "execute"}
    assert {7, ^meta, ^blobs} = roundtrip(7, meta, blobs)
  end

  test "round-trips large request ids (u64)" do
    big = 0xFFFF_FFFF_FFFF_FFFF
    assert {^big, _, _} = roundtrip(big, %{"command" => "x"}, [])
  end

  test "preserves binary tensor payloads exactly" do
    data = for i <- 0..255, into: <<>>, do: <<i>>
    assert {1, _, [^data]} = roundtrip(1, %{"command" => "upload"}, [data])
  end

  test "rejects a frame with bad magic" do
    assert {:error, :bad_magic} = Protocol.decode(<<"NOPE", 0::64, 0::32, 0::16, 0::16>>)
  end

  test "rejects a truncated frame" do
    <<head::binary-size(10), _::binary>> = encode(%{"command" => "x"}, 1, [])
    assert {:error, _} = Protocol.decode(head)
  end

  test "metadata is encoded as compact JSON" do
    bin = encode(%{"b" => 2, "a" => 1}, 1, [])
    assert String.contains?(bin, ~s({"))
    refute String.contains?(bin, ", ")
  end
end
