defmodule NxTinygrad.ConfigTest do
  use ExUnit.Case, async: false

  alias NxTinygrad.Config

  test "worker command prefers a packaged executable" do
    previous = System.get_env("NX_TINYGRAD_WORKER")
    executable = System.find_executable("true")
    System.put_env("NX_TINYGRAD_WORKER", executable)

    on_exit(fn -> restore_env("NX_TINYGRAD_WORKER", previous) end)

    assert Config.worker_command() == {executable, []}
  end

  test "missing Python produces an actionable error" do
    previous_python = System.get_env("NX_TINYGRAD_PYTHON")
    previous_config = Application.get_env(:nx_tinygrad, :python_executable)
    System.delete_env("NX_TINYGRAD_PYTHON")
    Application.put_env(:nx_tinygrad, :python_executable, "definitely-not-a-python-executable")

    on_exit(fn ->
      restore_env("NX_TINYGRAD_PYTHON", previous_python)
      restore_config(:python_executable, previous_config)
    end)

    assert_raise NxTinygrad.Error, ~r/NX_TINYGRAD_WORKER or NX_TINYGRAD_PYTHON/, fn ->
      Config.python_executable()
    end
  end

  defp restore_env(key, nil), do: System.delete_env(key)
  defp restore_env(key, value), do: System.put_env(key, value)

  defp restore_config(key, nil), do: Application.delete_env(:nx_tinygrad, key)
  defp restore_config(key, value), do: Application.put_env(:nx_tinygrad, key, value)
end
