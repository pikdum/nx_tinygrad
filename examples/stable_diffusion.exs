# Stable Diffusion v1.4 text-to-image, with the CLIP text encoder, the UNet
# denoiser, and the VAE decoder all compiled by nx_tinygrad.
#
#   elixir examples/stable_diffusion.exs "a prompt"                                    # CPU
#   NX_TINYGRAD_DEVICE="KFD+AMD:LLVM" elixir examples/stable_diffusion.exs "a prompt"  # AMD GPU
#
# Env knobs: SD_NUM_STEPS (default 20), SD_NUM_IMAGES (default 1).
#
# Downloads ~5 GB from Hugging Face on first run. The weights (~4 GB, mostly the
# UNet) are made device-resident via `preallocate_params: true`: they upload to
# the worker ONCE and are passed by handle on every denoise step, instead of
# being re-shipped as a multi-GB frame each call (which overflows the worker's
# transport frame). This is what makes a model this large runnable at all.
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
num_steps = String.to_integer(System.get_env("SD_NUM_STEPS", "20"))
num_images = String.to_integer(System.get_env("SD_NUM_IMAGES", "1"))

prompt =
  case System.argv() do
    [p | _] -> p
    [] -> "numbat in forest, detailed, digital art"
  end

repo = "CompVis/stable-diffusion-v1-4"

# SD v1.4's text_encoder/ ships no tokenizer of its own; use CLIP's (which has a
# Rust-compatible tokenizer.json), exactly as Bumblebee's own example does.
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-large-patch14"})
{:ok, clip} = Bumblebee.load_model({:hf, repo, subdir: "text_encoder"})
{:ok, unet} = Bumblebee.load_model({:hf, repo, subdir: "unet"})
{:ok, vae} = Bumblebee.load_model({:hf, repo, subdir: "vae"}, architecture: :decoder)
{:ok, scheduler} = Bumblebee.load_scheduler({:hf, repo, subdir: "scheduler"})
{:ok, featurizer} = Bumblebee.load_featurizer({:hf, repo, subdir: "feature_extractor"})
{:ok, safety_checker} = Bumblebee.load_model({:hf, repo, subdir: "safety_checker"})

serving =
  Bumblebee.Diffusion.StableDiffusion.text_to_image(clip, unet, vae, tokenizer, scheduler,
    num_steps: num_steps,
    num_images_per_prompt: num_images,
    safety_checker: safety_checker,
    safety_checker_featurizer: featurizer,
    preallocate_params: true,
    compile: [batch_size: 1, sequence_length: 60],
    defn_options: [
      compiler: NxTinygrad.Compiler,
      device: device,
      output: :host,
      execute_timeout: 1_200_000,
      compile_timeout: 1_200_000
    ]
  )

IO.puts(
  "Generating #{num_images} image(s) on device=#{device}, #{num_steps} steps\nprompt: #{inspect(prompt)}\n"
)

%{results: results} = Nx.Serving.run(serving, prompt)

results
|> Enum.with_index()
|> Enum.each(fn {%{image: image, is_safe: is_safe}, i} ->
  path = "sd_out_#{i}.png"
  image |> StbImage.from_nx() |> StbImage.write_file!(path)
  IO.puts("wrote #{path}  (#{inspect(image.shape)}, is_safe=#{is_safe})")
end)

IO.puts("\nPASS ✅  Stable Diffusion ran through nx_tinygrad on device=#{device}")
