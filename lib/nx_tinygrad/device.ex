defmodule NxTinygrad.Device do
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
          dev: String.t(),
          env: %{String.t() => String.t()}
        }

  @doc """
  Parse a logical device string.

  On tinygrad 0.13 the interface prefix and renderer suffix are part of the `DEV`
  string itself, so `"KFD+AMD:LLVM"` is passed through as `DEV` verbatim; the
  backend (`"AMD"`) is what tensors are created on.

  ## Examples

      iex> NxTinygrad.Device.parse("KFD+AMD:LLVM").dev
      "KFD+AMD:LLVM"

      iex> NxTinygrad.Device.parse("AMD").dev
      "KFD+AMD:LLVM"

      iex> NxTinygrad.Device.parse("CPU").tinygrad_device
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

    {interface, renderer, dev} = configure(backend, iface, renderer)

    %{
      spec: spec,
      backend: backend,
      interface: interface,
      renderer: renderer,
      tinygrad_device: backend,
      dev: dev,
      env: %{}
    }
  end

  # AMD: default to the KFD interface (never PCI/USB, which can unbind amdgpu) and
  # the LLVM renderer. The whole thing is the DEV string on tinygrad 0.13.
  defp configure("AMD", iface, renderer) do
    interface = iface || "KFD"
    renderer = renderer || "LLVM"
    {interface, renderer, "#{interface}+AMD:#{renderer}"}
  end

  defp configure(backend, iface, renderer) do
    dev =
      [iface && "#{iface}+", backend, renderer && ":#{renderer}"]
      |> Enum.reject(&is_nil/1)
      |> Enum.join()

    {iface, renderer, dev}
  end

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
