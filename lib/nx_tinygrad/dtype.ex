defmodule NxTinygrad.Dtype do
  @moduledoc """
  Mapping between Nx numeric types and nx_tinygrad's stable dtype names.

  The stable names are the wire format shared with the Python worker. Keeping
  the mapping centralized on both sides avoids drift.

  v0.1 required types: `{:f, 32}` <-> `"f32"`, `{:s, 32}` <-> `"s32"`,
  `{:u, 8}` <-> `"u8"`. A few more are mapped for convenience and forward
  compatibility.
  """

  @nx_to_name %{
    {:f, 16} => "f16",
    {:f, 32} => "f32",
    {:f, 64} => "f64",
    {:bf, 16} => "bf16",
    {:s, 8} => "s8",
    {:s, 16} => "s16",
    {:s, 32} => "s32",
    {:s, 64} => "s64",
    {:u, 8} => "u8",
    {:u, 16} => "u16",
    {:u, 32} => "u32",
    {:u, 64} => "u64"
  }

  @name_to_nx Map.new(@nx_to_name, fn {k, v} -> {v, k} end)

  # Types the worker handles today (bf16 rides a uint16 transport carrier and is
  # bitcast to tinygrad bfloat16 in the worker).
  @supported_names ~w(f16 f32 f64 bf16 s8 s16 s32 s64 u8 u16 u32 u64)

  @doc "Stable wire name for an Nx type, or `{:error, reason}`."
  @spec to_name(Nx.Type.t()) :: {:ok, String.t()} | {:error, String.t()}
  def to_name(type) do
    case Map.fetch(@nx_to_name, type) do
      {:ok, name} -> {:ok, name}
      :error -> {:error, "unsupported Nx type: #{inspect(type)}"}
    end
  end

  @doc "Stable wire name for an Nx type, raising `NxTinygrad.CompileError` on failure."
  @spec to_name!(Nx.Type.t()) :: String.t()
  def to_name!(type) do
    case to_name(type) do
      {:ok, name} ->
        name

      {:error, reason} ->
        raise NxTinygrad.CompileError,
          message: reason,
          hint: "supported dtypes: #{Enum.join(@supported_names, ", ")}"
    end
  end

  @doc "Nx type for a stable wire name."
  @spec to_nx(String.t()) :: {:ok, Nx.Type.t()} | {:error, String.t()}
  def to_nx(name) do
    case Map.fetch(@name_to_nx, name) do
      {:ok, type} -> {:ok, type}
      :error -> {:error, "unknown dtype name: #{inspect(name)}"}
    end
  end

  @spec to_nx!(String.t()) :: Nx.Type.t()
  def to_nx!(name) do
    case to_nx(name) do
      {:ok, type} -> type
      {:error, reason} -> raise NxTinygrad.Error, message: reason
    end
  end

  @doc "Whether the stable name is supported by the worker runtime today."
  def supported?(name), do: name in @supported_names

  def supported_names, do: @supported_names
end
