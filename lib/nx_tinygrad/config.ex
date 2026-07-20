defmodule NxTinygrad.Config do
  @moduledoc "Access to application configuration with sensible defaults."

  @defaults %{
    device: "CPU",
    start_default_worker: false,
    debug: 0,
    compile_timeout: 120_000,
    execute_timeout: 60_000,
    cache: true,
    executable_cache_size: 256,
    python_executable: nil,
    worker_executable: nil
  }

  @spec get(atom()) :: term()
  def get(key) when is_map_key(@defaults, key) do
    Application.get_env(:nx_tinygrad, key, Map.fetch!(@defaults, key))
  end

  def device, do: get(:device)
  def start_default_worker?, do: get(:start_default_worker)
  def debug, do: get(:debug)
  def compile_timeout, do: get(:compile_timeout)
  def execute_timeout, do: get(:execute_timeout)
  def cache?, do: get(:cache)
  def executable_cache_size, do: get(:executable_cache_size)

  @doc """
  The Python interpreter used to run the worker.

  Set through application config or `NX_TINYGRAD_PYTHON`; falls back to
  `python3` on PATH. Raises a clear error when none is available.
  """
  @spec python_executable() :: String.t()
  def python_executable do
    configured = get(:python_executable) || System.get_env("NX_TINYGRAD_PYTHON") || "python3"
    resolve_executable!(configured, "Python interpreter")
  end

  @doc "Executable and arguments used to start the worker Port."
  @spec worker_command() :: {String.t(), [String.t()]}
  def worker_command do
    case get(:worker_executable) || System.get_env("NX_TINYGRAD_WORKER") do
      nil -> {python_executable(), [worker_main()]}
      executable -> {resolve_executable!(executable, "nx_tinygrad worker"), []}
    end
  end

  @doc "Absolute path to the worker's `main.py`."
  @spec worker_main() :: String.t()
  def worker_main do
    Path.join(:code.priv_dir(:nx_tinygrad), "worker/main.py")
  end

  defp resolve_executable!(executable, label) do
    System.find_executable(executable) ||
      raise NxTinygrad.Error,
        message:
          "#{label} #{inspect(executable)} was not found or is not executable; " <>
            "set NX_TINYGRAD_WORKER or NX_TINYGRAD_PYTHON"
  end
end
