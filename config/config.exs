import Config

config :nx_tinygrad,
  # Default tinygrad device for the :default worker.
  device: "CPU",
  # Start the :default worker when the application boots. Disabled for library
  # consumers; the first default-worker request starts it lazily.
  start_default_worker: false,
  # tinygrad DEBUG level forwarded to the worker.
  debug: 0,
  # Timeouts (ms).
  compile_timeout: 120_000,
  execute_timeout: 60_000,
  # In-memory graph/executable cache.
  cache: true,
  executable_cache_size: 256

if config_env() == :test do
  import_config "test.exs"
end
