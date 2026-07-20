defmodule NxTinygrad.Error do
  @moduledoc """
  Base exception for nx_tinygrad. Specific failures use the more precise
  exceptions defined in this file.
  """
  defexception [:message, :details]

  @impl true
  def exception(opts) when is_list(opts) do
    %__MODULE__{message: Keyword.fetch!(opts, :message), details: Keyword.get(opts, :details, %{})}
  end

  def exception(message) when is_binary(message) do
    %__MODULE__{message: message, details: %{}}
  end
end

defmodule NxTinygrad.CompileError do
  @moduledoc "Raised when an Nx expression cannot be lowered to the nx_tinygrad graph IR."
  defexception [:message, :operation, :path, :input_specs, :output_spec, :hint]

  @impl true
  def message(%__MODULE__{message: message}), do: message
end

defmodule NxTinygrad.ProtocolError do
  @moduledoc "Raised when a worker frame violates the wire protocol."
  defexception [:message]
end

defmodule NxTinygrad.WorkerError do
  @moduledoc """
  A structured error returned by the Python worker. Preserves the worker's
  error class, command, generation, device and structured details.
  """
  defexception [:message, :class, :command, :generation, :device, :details, :python_traceback]

  @impl true
  def message(%__MODULE__{message: message, class: class}) do
    "[#{class}] #{message}"
  end
end

defmodule NxTinygrad.WorkerCrashedError do
  @moduledoc "Raised when the Python worker process crashes or exits unexpectedly."
  defexception [:message, :exit_status, :generation]

  @impl true
  def message(%__MODULE__{message: nil, exit_status: status}), do: "worker crashed (exit #{status})"
  def message(%__MODULE__{message: message}), do: message
end

defmodule NxTinygrad.StaleTensorError do
  @moduledoc """
  Raised when a backend tensor references a worker generation that no longer
  exists (the worker restarted). Its GPU data cannot be recovered.
  """
  defexception [:message, :worker, :tensor_generation, :worker_generation]

  @impl true
  def message(%__MODULE__{message: nil} = e) do
    "stale tensor: created by worker #{inspect(e.worker)} generation #{e.tensor_generation}, " <>
      "but the worker is now at generation #{e.worker_generation}. Its data cannot be recovered."
  end

  def message(%__MODULE__{message: message}), do: message
end

defmodule NxTinygrad.UnsupportedOperationError do
  @moduledoc "Raised when a graph references an operation the worker does not support."
  defexception [:message, :operation]

  @impl true
  def message(%__MODULE__{message: message}), do: message
end

defmodule NxTinygrad.DeviceUnavailableError do
  @moduledoc "Raised when the requested tinygrad device cannot be opened."
  defexception [:message, :device]

  @impl true
  def message(%__MODULE__{message: message}), do: message
end
