defmodule ExTinygrad.Protocol do
  @moduledoc """
  Encode/decode of the XTG1 wire frame carried inside each Port `packet: 4`
  message. Mirrors `priv/worker/protocol.py`.

  Frame layout (all integers big-endian):

      4  bytes  magic "XTG1"
      8  bytes  request id (u64)
      4  bytes  JSON metadata length (u32)
      2  bytes  blob count (u16)
      2  bytes  reserved
      8*N       blob lengths (u64)
      M         UTF-8 JSON metadata
      ...       concatenated blob bytes
  """

  @magic "XTG1"

  @doc """
  Encode a frame. `meta` is a JSON-encodable map, `blobs` a list of binaries.
  Returns iodata suitable for `Port.command/2`.
  """
  @spec encode(non_neg_integer(), map(), [binary()]) :: iodata()
  def encode(req_id, meta, blobs \\ []) do
    json = JSON.encode!(meta)
    blob_count = length(blobs)

    blob_lengths =
      for blob <- blobs, into: <<>>, do: <<byte_size(blob)::unsigned-big-64>>

    header =
      <<@magic, req_id::unsigned-big-64, byte_size(json)::unsigned-big-32, blob_count::unsigned-big-16,
        0::unsigned-big-16>>

    [header, blob_lengths, json | blobs]
  end

  @doc "Decode a frame binary into `{:ok, {req_id, meta, blobs}}` or `{:error, reason}`."
  @spec decode(binary()) :: {:ok, {non_neg_integer(), map(), [binary()]}} | {:error, term()}
  def decode(
        <<@magic, req_id::unsigned-big-64, json_len::unsigned-big-32, blob_count::unsigned-big-16,
          _reserved::unsigned-big-16, rest::binary>>
      ) do
    with {:ok, blob_lengths, rest} <- take_lengths(rest, blob_count, []),
         <<json::binary-size(^json_len), rest::binary>> <- rest,
         {:ok, meta} <- decode_json(json),
         {:ok, blobs} <- take_blobs(rest, blob_lengths, []) do
      {:ok, {req_id, meta, blobs}}
    else
      {:error, _} = err -> err
      _ -> {:error, :truncated_frame}
    end
  end

  def decode(_), do: {:error, :bad_magic}

  defp take_lengths(rest, 0, acc), do: {:ok, Enum.reverse(acc), rest}

  defp take_lengths(<<len::unsigned-big-64, rest::binary>>, n, acc) when n > 0,
    do: take_lengths(rest, n - 1, [len | acc])

  defp take_lengths(_, _, _), do: {:error, :truncated_blob_lengths}

  defp take_blobs(rest, [], acc) do
    if rest == <<>>, do: {:ok, Enum.reverse(acc)}, else: {:error, :trailing_bytes}
  end

  defp take_blobs(bin, [len | rest_lens], acc) do
    case bin do
      <<blob::binary-size(^len), rest::binary>> -> take_blobs(rest, rest_lens, [blob | acc])
      _ -> {:error, :truncated_blob}
    end
  end

  defp decode_json(json) do
    {:ok, JSON.decode!(json)}
  rescue
    e -> {:error, {:invalid_json, e}}
  end
end
