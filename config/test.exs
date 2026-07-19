import Config

# Integration tests exercise the full Elixir <-> Python path on CPU.
config :ex_tinygrad,
  device: "CPU",
  start_default_worker: true,
  debug: 0

config :logger, level: :warning
