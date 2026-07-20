defmodule NxTinygrad.TensorRef do
  @moduledoc """
  Rustler NIF resource holding worker-buffer reference metadata (worker id,
  generation, handle).

  The resource carries no GPU state. When it is garbage-collected, its Rust
  `Drop` pushes a release onto a native queue that `NxTinygrad.ReleaseReaper`
  drains. `take/1` claims a reference for explicit release so a later GC does not
  release it twice.
  """
  use Rustler, otp_app: :nx_tinygrad, crate: :nx_tinygrad_ref

  @doc "Create a reference resource for `{worker_id, generation, handle}`."
  def new(_worker_id, _generation, _handle), do: err()

  @doc "Claim the reference: returns `{worker_id, generation, handle}` once, then `nil`."
  def take(_ref), do: err()

  @doc "The buffer handle."
  def handle(_ref), do: err()

  @doc "The generation the reference was created in."
  def generation(_ref), do: err()

  @doc "Drain all queued releases from dropped references."
  def drain_releases, do: err()

  @doc "Create a GC-owned reference to a compiled worker executable."
  def new_executable(_worker_id, _generation, _handle), do: err()

  @doc "The executable handle stored in an executable reference."
  def executable_handle(_ref), do: err()

  @doc "Drain queued releases from dropped executable references."
  def drain_executable_releases, do: err()

  defp err, do: :erlang.nif_error(:nif_not_loaded)
end
