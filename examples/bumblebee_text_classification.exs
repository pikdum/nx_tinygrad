# One-shot integration test: run a real Bumblebee text-classification model
# (a BERT-family encoder) with the forward pass compiled by nx_tinygrad.
#
#   elixir examples/bumblebee_text_classification.exs
#   NX_TINYGRAD_DEVICE="KFD+AMD:LLVM" elixir examples/bumblebee_text_classification.exs
#
# Exercises the encoder path that already works today: token/position embeddings
# (gather), multi-head attention (batched dot + softmax), layer norm, GELU (erf).
# Downloads the model from Hugging Face on first run.

Mix.install([
  {:nx_tinygrad, path: Path.expand("..", __DIR__)},
  {:bumblebee, "~> 0.6"}
])

# Tokenization and serving glue run on the binary backend; the model's defn is
# compiled by nx_tinygrad.
Nx.global_default_backend(Nx.BinaryBackend)
device = System.get_env("NX_TINYGRAD_DEVICE", "CPU")

repo = "finiteautomata/bertweet-base-sentiment-analysis"
{:ok, model} = Bumblebee.load_model({:hf, repo})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, repo})

serving =
  Bumblebee.Text.text_classification(model, tokenizer,
    compile: [batch_size: 1, sequence_length: 64],
    defn_options: [compiler: NxTinygrad.Compiler, device: device, output: :host]
  )

for text <- [
      "I absolutely loved this, best thing all year!",
      "This was a complete waste of time.",
      "It was fine, nothing special."
    ] do
  %{predictions: [%{label: label, score: score} | _]} = Nx.Serving.run(serving, text)
  IO.puts("#{Float.round(score, 3)}  #{label}\t| #{text}")
end

IO.puts("\nPASS ✅  BERT encoder ran through nx_tinygrad on device=#{device}")
