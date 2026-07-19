ExUnit.start()

# GPU tests are excluded unless EX_TINYGRAD_GPU_TESTS=1.
gpu? = System.get_env("EX_TINYGRAD_GPU_TESTS") == "1"
ExUnit.configure(exclude: if(gpu?, do: [], else: [:gpu]))
