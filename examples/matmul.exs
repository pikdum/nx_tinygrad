# Matrix multiplication example.
#
#   mix run examples/matmul.exs

n = 256
a = Nx.iota({n, n}, type: :f32) |> Nx.divide(n * n)
b = Nx.iota({n, n}, type: :f32) |> Nx.divide(n * n)

matmul = ExTinygrad.jit(fn x, y -> Nx.dot(x, y) end)

# First call compiles + captures; subsequent calls replay.
result = matmul.(a, b) |> Nx.backend_transfer()
reference = Nx.dot(a, b)

close = Nx.all_close(result, reference, atol: 1.0e-4)
IO.puts("#{n}x#{n} matmul close to BinaryBackend? #{Nx.to_number(close) == 1}")
IO.inspect(result[0][0..3], label: "first row (first 4)")
