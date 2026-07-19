defmodule ExTinygrad.Config do
  @moduledoc "Access to application configuration with sensible defaults."

  @defaults %{
    device: "CPU",
    start_default_worker: true,
    debug: 0,
    compile_timeout: 120_000,
    execute_timeout: 60_000,
    cache: true
  }

  @spec get(atom()) :: term()
  def get(key) when is_map_key(@defaults, key) do
    Application.get_env(:ex_tinygrad, key, Map.fetch!(@defaults, key))
  end

  def device, do: get(:device)
  def start_default_worker?, do: get(:start_default_worker)
  def debug, do: get(:debug)
  def compile_timeout, do: get(:compile_timeout)
  def execute_timeout, do: get(:execute_timeout)
  def cache?, do: get(:cache)

  @doc """
  The Python interpreter used to run the worker.

  Set by the Nix devshell via `EX_TINYGRAD_PYTHON`; falls back to `python3` on
  PATH otherwise.
  """
  @spec python_executable() :: String.t()
  def python_executable do
    System.get_env("EX_TINYGRAD_PYTHON") || System.find_executable("python3") || "python3"
  end

  @doc "Absolute path to the worker's `main.py`."
  @spec worker_main() :: String.t()
  def worker_main do
    Path.join(:code.priv_dir(:ex_tinygrad), "worker/main.py")
  end
end
