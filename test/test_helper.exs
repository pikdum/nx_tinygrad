ExUnit.start()

# GPU tests are excluded unless NX_TINYGRAD_GPU_TESTS=1.
gpu? = System.get_env("NX_TINYGRAD_GPU_TESTS") == "1"
ExUnit.configure(exclude: if(gpu?, do: [], else: [:gpu]))
