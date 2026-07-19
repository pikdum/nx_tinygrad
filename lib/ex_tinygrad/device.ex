defmodule ExTinygrad.Device do
  @moduledoc """
  Parsing of logical device strings into a concrete tinygrad configuration.

  Logical strings follow the spec form `[<IFACE>+]<BACKEND>[:<RENDERER>]`, e.g.
  `"KFD+AMD:LLVM"`, `"CPU"`, `"AMD"`.

  tinygrad 0.12.0's `Device[...]` does not accept `"KFD+AMD:LLVM"` literally, so
  we translate it into a concrete tinygrad device name plus environment variables
  here. Those variables (notably `AMD_LLVM`/`AMD_IFACE`) are read by tinygrad at
  import time, so they are passed to the worker Port's environment and must be
  set before the worker imports tinygrad.

  This mirrors `priv/worker/device.py`; keep the two in sync.
  """

  @type t :: %{
          spec: String.t(),
          backend: String.t(),
          interface: String.t() | nil,
          renderer: String.t() | nil,
          tinygrad_device: String.t(),
          env: %{String.t() => String.t()}
        }

  @doc """
  Parse a logical device string.

  ## Examples

      iex> ExTinygrad.Device.parse("KFD+AMD:LLVM").env
      %{"AMD_IFACE" => "KFD", "AMD_LLVM" => "1"}

      iex> ExTinygrad.Device.parse("CPU").tinygrad_device
      "CPU"
  """
  @spec parse(String.t() | nil) :: t()
  def parse(spec) do
    spec = normalize(spec)

    {iface, rest} =
      case String.split(spec, "+", parts: 2) do
        [iface, rest] -> {String.upcase(iface), rest}
        [rest] -> {nil, rest}
      end

    {backend, renderer} =
      case String.split(rest, ":", parts: 2) do
        [backend, renderer] -> {String.upcase(backend), String.upcase(renderer)}
        [backend] -> {String.upcase(backend), nil}
      end

    {interface, renderer, env} = configure(backend, iface, renderer)

    %{
      spec: spec,
      backend: backend,
      interface: interface,
      renderer: renderer,
      tinygrad_device: backend,
      env: env
    }
  end

  # AMD: never default to PCI/USB (they can unbind the amdgpu driver). Force KFD
  # and the LLVM renderer unless the caller asked for something specific.
  defp configure("AMD", iface, renderer) do
    interface = iface || "KFD"
    renderer = renderer || "LLVM"

    env = %{"AMD_IFACE" => interface}
    env = if renderer == "LLVM", do: Map.put(env, "AMD_LLVM", "1"), else: env

    {interface, renderer, env}
  end

  defp configure(_backend, iface, renderer), do: {iface, renderer, %{}}

  defp normalize(nil), do: "CPU"

  defp normalize(spec) when is_binary(spec) do
    case String.trim(spec) do
      "" -> "CPU"
      other -> other
    end
  end

  @doc "The tinygrad device name for a logical device string."
  @spec tinygrad_device(String.t() | nil) :: String.t()
  def tinygrad_device(spec), do: parse(spec).tinygrad_device

  @doc "The environment variables required to run the given logical device string."
  @spec env(String.t() | nil) :: %{String.t() => String.t()}
  def env(spec), do: parse(spec).env
end
