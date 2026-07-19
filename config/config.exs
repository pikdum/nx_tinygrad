import Config

config :ex_tinygrad,
  # Default tinygrad device for the :default worker.
  device: "CPU",
  # Start the :default worker when the application boots.
  start_default_worker: true,
  # tinygrad DEBUG level forwarded to the worker.
  debug: 0,
  # Timeouts (ms).
  compile_timeout: 120_000,
  execute_timeout: 60_000,
  # In-memory graph/executable cache.
  cache: true

if config_env() == :test do
  import_config "test.exs"
end
