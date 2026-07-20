# One-shot integration test: run a real Bumblebee image-classification model
# (ResNet-50) with the forward pass compiled by nx_tinygrad.
#
#   elixir examples/bumblebee_image_classification.exs path/to/image.jpg
#   NX_TINYGRAD_DEVICE="KFD+AMD:LLVM" elixir examples/bumblebee_image_classification.exs img.jpg
#
# Exercises the conv-net inference path: conv (strides/padding), ReLU, window
# pooling, and the final dense classifier. Downloads the model on first run.
# With no image argument, a synthetic one is used just to prove the graph runs.

Mix.install([
  {:nx_tinygrad, path: Path.expand("..", __DIR__)},
  {:bumblebee, "~> 0.6"},
  {:stb_image, "~> 0.6"}
])

Nx.global_default_backend(Nx.BinaryBackend)
device = System.get_env("NX_TINYGRAD_DEVICE", "CPU")

repo = "microsoft/resnet-50"
{:ok, model} = Bumblebee.load_model({:hf, repo})
{:ok, featurizer} = Bumblebee.load_featurizer({:hf, repo})

serving =
  Bumblebee.Vision.image_classification(model, featurizer,
    compile: [batch_size: 1],
    defn_options: [compiler: NxTinygrad.Compiler, device: device]
  )

image =
  case System.argv() do
    [path | _] ->
      %{shape: {h, w, c}} = img = StbImage.read_file!(path) |> StbImage.to_nx()
      IO.puts("Loaded #{path} (#{h}x#{w}x#{c})")
      img

    [] ->
      IO.puts("No image given; using a synthetic 224x224 image to prove the graph runs.")
      Nx.iota({224, 224, 3}, type: :u8) |> Nx.remainder(256) |> Nx.as_type(:u8)
  end

%{predictions: preds} = Nx.Serving.run(serving, image)

IO.puts("\nTop predictions:")

for %{label: label, score: score} <- Enum.take(preds, 5) do
  IO.puts("  #{Float.round(score, 3)}  #{label}")
end

IO.puts("\nPASS ✅  ResNet-50 conv-net ran through nx_tinygrad on device=#{device}")
