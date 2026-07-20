# Operation coverage (v0.1)

The compiler lowers the operations below. Anything else raises
`NxTinygrad.CompileError` at compile time (no silent host fallback), naming the
operation and its shapes.

## Syntax nodes

`parameter`, `constant`, `tensor` (baked-in literal), `metadata` (transparent).
Multiple outputs and nested `Nx.Container`s (tuples, maps, structs) are
reconstructed on the Elixir side.

## Elementwise

- **Unary:** `negate`, `abs`, `exp`, `expm1`, `log`, `log1p`, `sqrt`, `rsqrt`,
  `tanh`, `sigmoid`, `sin`, `cos`, `floor`, `ceil`
- **Binary:** `add`, `subtract`, `multiply`, `divide`, `pow`, `max`, `min`
  (numpy-style broadcasting)

## Comparison & selection

`equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` (produce
`u8`), and `select`.

## Shape

`reshape`, `squeeze`, `broadcast`, `transpose`, `concatenate`, `slice`,
`as_type`.

## Reductions

`sum`, `reduce_max`, `reduce_min`, `all`, `any` (with `axes` and `keep_axes`).

## Linear algebra

`dot` — implemented generally via `einsum`, covering 2-D matmul, vector·matrix,
matrix·vector, and batched matmul. Contraction/batch axes come straight from Nx.

## dtypes

Wire names map to Nx types. v0.1 required: `f32`, `s32`, `u8`. Also mapped:
`f16`, `f64`, `s8`, `s16`, `s64`, `u16`, `u32`, `u64`. `f64` is functional
but not performance-optimized on the tested AMD device. Not supported: `bf16`,
complex, packed, or quantized types. Nx determines output types; tinygrad results
are cast to satisfy the serialized output spec.

## Autograd

`Nx.Defn.value_and_grad` works without tinygrad autograd: Nx rewrites the
gradient into forward operations before the compiler runs, so the compiler only
needs the forward op set above (which covers the linear/MLP gradient graphs).

## Not in v0.1

`conv`, `pad`, `put_slice`, `indexed_add`, `indexed_put`, `gather`, `take`,
`take_along_axis`, `argmax`, `argmin`, `sort`, `argsort`, dynamic `while`/`cond`,
`Nx.Block` with impure
defaults. These raise a detailed compile error.
