# Operator fusion v0

## Status and scope

The first fusion pass is a conservative C99 optimization over typed IR. It
combines eligible single-use tensor temporaries into one row-major output loop.
The pass preserves the original expression tree and does not reassociate
operations.

The initial eligible subset is limited to:

- Pure, straight-line functions containing local `let` bindings and a tensor
  return.
- `f32` or `f64` tensors with the same static shape throughout a fused chain.
- Unary `+`/`-` and elementwise `+`, `-`, and `*`, including scalar
  broadcasting already accepted by the type checker.
- Intermediate tensor bindings with exactly one use.

The pass leaves multi-use temporaries intact. It does not fuse reductions,
division, conversion calls, indexing, effectful functions, functions targeted
by `gradient`, or any expression outside this subset. The interpreter retains
the unfused IR behavior.

## Lowering and numerical behavior

An eligible chain is represented by a `fused_tensor` IR expression. C99 emits
one loop over the statically known element count and evaluates the original
expression tree at each index. Intermediate `f32` operations are explicitly
cast back to `float`; expression order is unchanged, and the backend keeps
floating-point contraction disabled. No reassociation or algebraic
simplification is performed.

Fusion is applied by default when emitting C99, including per-source project
units. `emit_c99_module(..., optimize=False)` remains available for baseline
equivalence tests. This opt-out is a testing/diagnostic hook, not a second
language mode.

## Validation and remaining work

Tests check deterministic transformed output, conservative rejection of
multi-use/reduction/autodiff-target cases, fewer emitted loops, and native
equivalence with both unoptimized C99 and the interpreter under GCC and Clang.
The equivalence cases include scalar broadcasting, an `f32` rounding-sensitive
input, and rank-two/rank-three tensors. Dedicated `f32` and `f64` cancellation
cases check that fusion preserves non-associative expression order.
Project-unit emission is also checked to apply the pass and compile
successfully with both native compilers.

Run the repeatable native comparison with:

```powershell
.\venv\Scripts\python.exe -m src.fusion_benchmark --compiler gcc
```

By default, the benchmark compiles and measures three distinct four-operation
chains (`scale_shift_restore`, `offset_cancel`, and `negated_scale`) in fused
and unfused forms at `-O2`. Each workload checks matching checksums and reports
medians over alternating samples. Select one chain with `--workload`; tune the
shape and sampling with `--elements`, `--loops`, and `--samples`. Results are
hardware/load-dependent, and checksum comparison is not a substitute for the
interpreter/C99 differential tests.

The earlier single-chain baseline on the current Windows/RTX 3060 development
machine measured 24 ms unfused versus 4 ms fused with GCC, and 10 ms versus
3 ms with Clang (4096 elements, 2000 calls, 7 samples). This is not a
multi-workload performance claim; the expanded suite must be re-measured
separately on supported release targets before making broader claims.

One expanded 7-sample run on that development machine produced these medians
(unfused/fused milliseconds, 4096 elements, 2000 calls per sample):

| Workload | GCC | Clang |
|---|---:|---:|
| `scale_shift_restore` | 17 / 4 | 8 / 3 |
| `offset_cancel` | 15 / 3 | 5 / 2 |
| `negated_scale` | 17 / 4 | 7 / 3 |

All three runs reported matching fused/unfused checksums. These coarse local
measurements show benefit for these selected chains only and do not establish
performance on other machines or workloads.

Broader operator coverage, cost-based decisions, fusion across function
boundaries, reductions, and CUDA fusion remain future work.
