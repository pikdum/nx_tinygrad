# Stable Diffusion v1.4 text-to-image, with the CLIP text encoder, the UNet
# denoiser, and the VAE decoder all compiled by nx_tinygrad.
#
#   elixir examples/stable_diffusion.exs "a prompt"                                    # CPU
#   NX_TINYGRAD_DEVICE="KFD+AMD:LLVM" elixir examples/stable_diffusion.exs "a prompt"  # AMD GPU
#
# Env knobs: SD_NUM_STEPS (default 20), SD_NUM_IMAGES (default 1), SD_RUNS
# (default 1 — extra runs reuse the compiled graphs and show the warm cost),
# SD_SAFETY=0 to skip the safety checker (an extra CLIP-vision pass per image —
# cheap once warm, but it does add one-time kernel-compile cost on run 1).
#
# Downloads ~5 GB from Hugging Face on first run. The weights (~4 GB, mostly the
# UNet) load DIRECTLY onto the execution worker: checkpoint bytes upload once
# and every param remap (transpose / reshape / f16->f32 upcast) runs device-side
# through the backend's eager ops — an order of magnitude faster than remapping
# on Nx.BinaryBackend — and the params land device-resident, passed by handle
# every call instead of being re-shipped as a multi-GB frame (which would
# overflow the worker's transport frame). `preallocate_params: true` is kept as
# a safety net; for already-resident params it is a ~free handle-level copy.
#
# SD on CPU is slow — the first UNet execute JIT-compiles the whole kernel graph
# in tinygrad, which is why the timeouts below are generous. The AMD GPU is far
# faster. Writes each result to sd_out_<i>.png.

Mix.install([
  {:nx_tinygrad, path: Path.expand("..", __DIR__)},
  {:bumblebee, "~> 0.6"},
  {:stb_image, "~> 0.6"}
])

# Tokenization, the scheduler loop, and serving glue run on the binary backend;
# only the model defns are compiled by nx_tinygrad.
Nx.global_default_backend(Nx.BinaryBackend)

device = System.get_env("NX_TINYGRAD_DEVICE", "CPU")

# Plain defn calls (not wrapped in Nx.Defn.compile) also go through nx_tinygrad.
# This matters for the safety checker's featurizer: its NxImage.resize would
# otherwise run under the default Nx.Defn.Evaluator on BinaryBackend, which
# costs ~113 s for one 512x512 bicubic resize (~0.5 s compiled).
Nx.Defn.global_default_options(
  compiler: NxTinygrad.Compiler,
  device: device,
  output: :host,
  execute_timeout: 1_200_000,
  compile_timeout: 1_200_000
)

num_steps = String.to_integer(System.get_env("SD_NUM_STEPS", "20"))
num_images = String.to_integer(System.get_env("SD_NUM_IMAGES", "1"))
num_runs = String.to_integer(System.get_env("SD_RUNS", "1"))
safety? = System.get_env("SD_SAFETY", "1") == "1"

# SD_TYPE=f16|bf16 casts the clip/unet/vae params device-side at load and runs
# their compute in that type (Axon mixed-precision policy). Measured on the
# RX 7900 XT (warm, 20 steps): f32 ~19 s (default, correct); f16 ~16 s but the
# image comes out black (a model-level overflow — tinygrad's f16 matmul/softmax
# /layernorm all probe clean); bf16 ~34 s, correct but slow (LLVM emulates
# scalar bf16, only WMMA is native). The safety checker stays f32 — it is
# ~free warm and gates is_safe.
model_type =
  case System.get_env("SD_TYPE", "f32") do
    "f32" -> []
    t when t in ["f16", "bf16"] -> [type: String.to_existing_atom(t)]
  end

prompt =
  case System.argv() do
    [p | _] -> p
    [] -> "numbat in forest, detailed, digital art"
  end

repo = "CompVis/stable-diffusion-v1-4"

# Load params straight onto the worker the compiled graphs will execute on
# (the same resolution NxTinygrad.Compiler uses for `device:`).
worker = NxTinygrad.WorkerSupervisor.worker_for_device(device)
param_backend = {NxTinygrad.Backend, worker: worker}

load_t0 = System.monotonic_time(:millisecond)

timed = fn label, fun ->
  t0 = System.monotonic_time(:millisecond)
  result = fun.()
  IO.puts("  load #{label}: #{System.monotonic_time(:millisecond) - t0} ms")
  result
end

# SD v1.4's text_encoder/ ships no tokenizer of its own; use CLIP's (which has a
# Rust-compatible tokenizer.json), exactly as Bumblebee's own example does.
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-large-patch14"})

{:ok, clip} =
  timed.("text_encoder", fn ->
    Bumblebee.load_model(
      {:hf, repo, subdir: "text_encoder"},
      [backend: param_backend] ++ model_type
    )
  end)

{:ok, unet} =
  timed.("unet", fn ->
    Bumblebee.load_model({:hf, repo, subdir: "unet"}, [backend: param_backend] ++ model_type)
  end)

{:ok, vae} =
  timed.("vae", fn ->
    Bumblebee.load_model(
      {:hf, repo, subdir: "vae"},
      [architecture: :decoder, backend: param_backend] ++ model_type
    )
  end)

{:ok, scheduler} = Bumblebee.load_scheduler({:hf, repo, subdir: "scheduler"})

safety_opts =
  if safety? do
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, repo, subdir: "feature_extractor"})

    {:ok, safety_checker} =
      timed.("safety_checker", fn ->
        Bumblebee.load_model({:hf, repo, subdir: "safety_checker"}, backend: param_backend)
      end)

    [safety_checker: safety_checker, safety_checker_featurizer: featurizer]
  else
    []
  end

serving =
  Bumblebee.Diffusion.StableDiffusion.text_to_image(
    clip,
    unet,
    vae,
    tokenizer,
    scheduler,
    [
      num_steps: num_steps,
      num_images_per_prompt: num_images,
      preallocate_params: true,
      compile: [batch_size: 1, sequence_length: 60],
      defn_options: [
        compiler: NxTinygrad.Compiler,
        device: device,
        output: :host,
        execute_timeout: 1_200_000,
        compile_timeout: 1_200_000
      ]
    ] ++ safety_opts
  )

IO.puts("model load: #{System.monotonic_time(:millisecond) - load_t0} ms (device-resident)")

IO.puts(
  "Generating #{num_images} image(s) on device=#{device}, #{num_steps} steps\nprompt: #{inspect(prompt)}\n"
)

results =
  Enum.reduce(1..num_runs, nil, fn i, _ ->
    run_t0 = System.monotonic_time(:millisecond)
    %{results: results} = Nx.Serving.run(serving, prompt)
    IO.puts("generate (run #{i}): #{System.monotonic_time(:millisecond) - run_t0} ms")
    results
  end)

stats = NxTinygrad.worker_stats(worker: worker)

IO.puts(
  "denoise steps: #{stats["while_steps_symbolic"]} symbolic-JIT, " <>
    "#{stats["while_steps_jit"]} static-JIT, #{stats["while_steps_interpreted"]} interpreted, " <>
    "#{stats["while_jit_fallbacks"]} fallbacks"
)

results
|> Enum.with_index()
|> Enum.each(fn {%{image: image} = result, i} ->
  is_safe = Map.get(result, :is_safe, "n/a")
  path = "sd_out_#{i}.png"
  image |> StbImage.from_nx() |> StbImage.write_file!(path)
  IO.puts("wrote #{path}  (#{inspect(image.shape)}, is_safe=#{is_safe})")
end)

IO.puts("\nPASS ✅  Stable Diffusion ran through nx_tinygrad on device=#{device}")
