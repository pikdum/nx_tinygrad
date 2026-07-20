# Operation coverage

The compiler lowers the operations below; the set grows as the coverage march
toward a general-purpose `Nx.Backend` proceeds. Anything else raises
`NxTinygrad.CompileError` at compile time (no silent host fallback), naming the
operation and its shapes. Every listed op is verified against `Nx.BinaryBackend`
in `test/differential_test.exs`.

## Syntax nodes

`parameter`, `constant`, `tensor` (baked-in literal), `metadata` (transparent).
Multiple outputs and nested `Nx.Container`s (tuples, maps, structs) are
reconstructed on the Elixir side.

## Elementwise

- **Unary (math):** `negate`, `abs`, `sign`, `exp`, `expm1`, `log`, `log1p`,
  `sqrt`, `rsqrt`, `cbrt`, `pow`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`,
  `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `sigmoid`, `erf`, `erfc`,
  `round`, `floor`, `ceil` (`cbrt` composed as magnitude-root + sign, `round`
  composed to Nx half-away-from-zero, `erfc` as `1 - erf`; the rest map to
  tinygrad primitives)
- **Unary (special):** `erf_inv` (Giles rational approximation), `conjugate`
  (identity on real inputs)
- **Unary (predicate / bitwise):** `is_nan`, `is_infinity` (produce `u8`),
  `bitwise_not`, `count_leading_zeros`, `population_count`
- **Binary:** `add`, `subtract`, `multiply`, `divide`, `pow`, `max`, `min`,
  `remainder`, `quotient`, `atan2`, `bitwise_and`, `bitwise_or`, `bitwise_xor`,
  `left_shift`, `right_shift` (numpy-style broadcasting; `remainder`/`quotient`
  match Nx truncated division, `atan2` is composed from `atan`)
- **Logical:** `logical_and`, `logical_or`, `logical_xor` (nonzero is true,
  produce `u8`)

## Comparison & selection

`equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` (produce
`u8`), `select`, and `clip`.

## Shape

`reshape`, `squeeze`, `broadcast`, `transpose`, `reverse`, `concatenate`,
`slice`, `stack`, `tile`, `eye`, `as_type`, `iota`, `pad` (edge padding with a
scalar constant fill, including ±infinity; interior/negative padding not yet
supported), `sort`, `argsort`.

## Indexing

`gather` (Nx coordinate-gather over `:axes`). `take` and `take_along_axis` are
covered via the block path (they decompose to `gather` + `iota`). `put_slice`
(static or dynamic start offsets), `indexed_add`, `indexed_put` (coordinate
scatter over `:axes`; `indexed_add` accumulates duplicate indices).
`window_scatter_max`/`window_scatter_min` (select-and-scatter — max/min-pool
backward).

## Reductions

`sum`, `product`, `reduce_max`, `reduce_min`, `all`, `any` (with `axes` and
`keep_axes`); `argmax`, `argmin` (with `axis`, `keep_axis`, and `:low`/`:high`
tie-breaks); windowed `window_sum`, `window_max`, `window_min`,
`window_product` (with strides, padding, and window dilation).

## Composite / control flow

Optional Nx ops that carry a pre-traced *pure* default expression are lowered by
binding their inputs to that expression and lowering it into primitives — the
impure callback is never run. This covers `cumulative_sum/product/max/min`,
`top_k`, `Nx.LinAlg.determinant`, `Nx.LinAlg.cholesky` (iterates through the
`while` path), and `triangular_solve` (unrolled substitution). Tuple-valued
sources are projected with `elem`.

`cond` lowers to a chain of predicated `select`s (pure branches). `while` lowers
to a multi-output node whose condition and body are self-contained sub-graphs;
the worker runs it as an eager loop (reading the scalar condition each
iteration), so data-dependent trip counts work. Graphs containing a `while` (or
a dynamic `slice`) run eagerly rather than via TinyJit capture.

## Linear algebra

`dot` — implemented generally via `einsum`, covering 2-D matmul, vector·matrix,
matrix·vector, and batched matmul. Contraction/batch axes come straight from Nx.

## Convolution

`conv` — Nx default-layout convolution onto tinygrad `conv2d` (general over
spatial rank). Supports `strides`, per-edge asymmetric `padding`,
`kernel_dilation`, and `feature_group_size`. Not yet supported (raise a compile
error): `input_dilation` (transposed conv), non-identity input/kernel/output
permutations, and `batch_group_size` > 1.

## dtypes

Wire names map to Nx types: `f16`, `f32`, `f64`, `bf16`, `c64`, `c128`, `s8`,
`s16`, `s32`, `s64`, `u8`, `u16`, `u32`, `u64`. `bf16` rides a uint16 transport
carrier (bitcast to tinygrad bfloat16); `c64`/`c128` are carried as paired reals
(see Complex numbers below). `f64` is functional but not performance-optimized on
the tested AMD device. Not supported: packed or quantized types. Nx determines
output types; tinygrad results are cast to satisfy the serialized output spec.

## Autograd

`Nx.Defn.value_and_grad` works without tinygrad autograd: Nx rewrites the
gradient into forward operations before the compiler runs, so the compiler only
needs the forward op set above (which covers the linear/MLP gradient graphs).

## Complex numbers

tinygrad has no complex dtype, so a complex tensor of logical shape `S` is held
internally as a real float tensor of shape `S + [2]` (last axis `[real, imag]`).
Supported: `c64`/`c128` dtypes, complex `add`/`subtract`/`multiply`/`divide`/
`negate`, `conjugate`, `real`, `imag`, `abs`, `exp`, `as_type`, the shape ops,
`sum`, `dot`, `select`, and `fft`/`ifft` (a real cos/sin DFT matmul).

## Linear algebra (composites)

`Nx.LinAlg` composites lower through the primitive set and control flow:
`cholesky`, `qr`, `lu`, `svd`, `eigh`, `triangular_solve`, `determinant`. (svd
singular values / eigh eigenvalues match Nx; U/V are sign-ambiguous as usual.)

`reduce` and `window_reduce` with an arbitrary accumulator function lower by
folding the traced reducer body over the (statically unrolled) axes/windows.

## Not yet supported

Only these edges remain, all outside the core defn op surface:

- Eager (non-`defn`) backend ops that raise by design: `to_batched` on a device
  tensor (batch on the host first), and pointer interop (`from_pointer` /
  `to_pointer`).
- `conv` with `batch_group_size` > 1, and `window_reduce` with non-zero padding.
