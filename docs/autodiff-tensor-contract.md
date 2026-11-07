# Tensor autodiff v0 contract

**Status:** implemented in the Python interpreter and experimental C99 backend
for the bounded subset below. This is a prototype contract, not a production
memory-safety or numerical guarantee.

This document records the deliberately small first tensor-gradient increment
and its current implementation boundary.

## Proposed API and function shape

The existing intrinsic spelling is retained:

```aether
gradient(objective, input)
```

For tensor autodiff v0, `objective` is a named, unary, pure function with the
conceptual signature:

```text
Tensor<T, S, U_in> -> Scalar<T, U_out>
```

where `T` is exactly `f32` or `f64`, and `S` is a non-empty, statically known
shape. The objective returns one scalar, not a tensor. Input and output units
may be linear, affine-temperature, or angular units when their ratio has a
valid unit representation. Effects, control flow, nested user-function calls,
and implicit conversions are excluded.

The input expression must have exactly the objective's tensor element type and
shape and unit. The result has the same row-major layout, shape, and element
type as the input; its unit is `unit(objective result) / unit(input)`. The
numeric scale must reflect the units in which the forward function computes.
This overload does not change the existing scalar `gradient(f, x)` contract.
Scalar and tensor gradients share expression tracing for the operations each
overload accepts; scalar type and unit restrictions are unchanged.

The gradient result has the same row-major shape and element type as the input.
Its unit dimensions, rational scale, and symbolic pi power are the quotient
`unit(output) / unit(input)`; affine offsets do not contribute to the
derivative. Explicit `convert` operations, including affine-temperature and
angular conversions, are supported and differentiated by their multiplier
including any power of pi.

`Scalar<T, U_out>` and the tensor unit argument above are explanatory
notation, not literal Aether type constructors. Numeric scalar types and the
third tensor type argument accept unit annotations. The
[unit-prefix and unit-expression implementation](./unit-prefix-design.md)
provides their exact scale and dimension metadata.

## Proposed differentiable subset

The objective body may contain local bindings followed by one final return.
Expressions are limited to numeric literals, names, grouping, unary `+`/`-`,
and the following operations that are already admitted by the tensor type
contract:

- Scalar `+`, `-`, `*`, `/`.
- Tensor `+` and `-` for equal shapes, plus the currently supported scalar
  broadcasting forms.
- Tensor/scalar `*` and tensor/scalar `/`, plus scalar/tensor `*`.
- `sum(tensor)` and `mean(tensor)`, each reducing all elements to one scalar.
- Explicit `convert(value, "unit")`, including affine offsets and symbolic pi
  factors.
- Built-in `scale(tensor, scalar)` and `subtract(tensor, tensor)`, including
  their pipeline forms.

Tensor/tensor multiplication and division remain unsupported. Indexing and
all other calls are rejected in v0. In particular, `abs`, `min`, `max`, and
`max_abs` have no derivative convention in this proposal.

## Reverse-mode rules

The pass accumulates adjoints in reverse evaluation order. Contributions to a
shared value are added, not overwritten.

For tensor-valued intermediate `t`, scalar `s`, and tensor adjoint `g`:

| Forward operation | Tensor contribution | Scalar contribution |
|---|---|---|
| `t + u` | `g` to each tensor operand | — |
| `t - u` | `g` to `t`, `-g` to `u` | — |
| `t + s`, `s + t` | `g` to `t` | `sum(g)` to `s` |
| `t - s` | `g` to `t` | `-sum(g)` to `s` |
| `s - t` | `-g` to `t` | `sum(g)` to `s` |
| `t * s`, `s * t` | `g * s` to `t` | `sum(g * t)` to `s` |
| `t / s` | `g / s` to `t` | `sum(-g * t / (s * s))` to `s` |
| `sum(t)` | `upstream` broadcast to every element of `t` | — |
| `mean(t)` | `upstream / element_count` broadcast to every element of `t` | — |

Unary `+t` forwards the upstream gradient unchanged; `-t` negates it.

These are mathematical rules; the exact floating-point evaluation order and
bitwise reproducibility are not promised by this prototype.
Forward `f64` `sum` and `mean` reductions accumulate sequentially in row-major
element order, with `f64` precision after each addition. `mean` divides the
completed sum once by the element count. This avoids Python-version-dependent
compensated summation and matches generated C99. Interpreter and native tests
also compare `f64` gradients for `sum`, `mean`, and their shared product on
cancellation-sensitive inputs in more than one element order.

For `f32` objectives, forward values and local derivative contributions are
rounded to `f32` at operation boundaries in both the interpreter and C99
backend. Reverse adjoints accumulate in double precision, then the returned
gradient is rounded to its declared `f32` type. With the default IEEE-754
round-to-nearest mode, an exact halfway result between `1.0f` and its next
representable value rounds to `1.0f` (ties-to-even); a result just above that
halfway point rounds to the next `f32`. Another exact halfway result whose
lower neighbor is odd rounds to the upper even neighbor; its negative
counterpart rounds symmetrically. Scalar and `sum`-based tensor tests check
both tie outcomes and positive/negative gradients in the interpreter and
generated C99 under GCC and Clang. The local derivative factor for
`mean(tensor)` is rounded to the scalar result type before it contributes to
the reverse adjoints; a cancellation-sensitive three-element `f32` case checks
this rule in the interpreter and compiled backend.
Subnormal boundary tests also check that half the least positive `f32`
subnormal rounds to signed zero, while the tie at 1.5 subnormal units rounds
to the second subnormal. At 2.5 subnormal units, tests verify ties-to-even
rounds to the second subnormal, values immediately below and above round to
the second and third subnormals respectively, and negative gradients are
symmetric. At half a subnormal, values immediately below and at the tie round
to signed zero, while the immediate value above rounds to the least subnormal.
Scalar and `mean`-based tensor gradients cover both signs around this boundary
under GCC and Clang.
This selected behavior does not promise bitwise reproducibility for all
operations, rounding modes, or compiler targets. Differential tests
cover cancellation-sensitive scalar and tensor adjoint accumulation, including
retaining a unit residual between cancelling contributions of magnitude
`2^24`, exact full cancellation, and graphs with multiple cancellation paths
in different orders. `f64` quotient-gradient tests cover positive, negative,
and small nonzero denominators, comparing analytic results across the
interpreter, GCC, and Clang. Scalar and tensor `f64` gradients also verify
that division by zero raises in the interpreter and traps in native C99.
Rank-three `f64` gradients check scalar broadcasting through a shared `mean`
value for addition, both subtraction operand orders, multiplication, and
division. Multiplication is tested with both tensor/scalar operand orders;
unary negation also checks the sign of a broadcast reduction adjoint. The
cases include reduction and sign of broadcast adjoints and
cancellation-sensitive row-major inputs. `[2,2,2]` and `[2,3,2]` cases
complement `[2,1,2]` to check rank-three tensors without singleton dimensions
and with an uneven inner dimension. Results are compared with analytic results
in the interpreter and native C99. A corresponding `f32` `[2,3,2]` gradient
case checks that both implementations preserve the expected `13.0` result
across the non-singleton shape. The uneven-shape cases also batch direct
`sum` and `mean` gradients to verify the unit and reciprocal-element-count
adjoints for both precisions. A rectangular `f64` `[2,3]` case compares the
interpreter and C99 gradients for a shared mean reduction and checks that the
returned nested result preserves both dimensions.
A rectangular `f32` `[2,3]` shared-mean product and an `f64` `[2,1,2,2]`
shared-mean division cover rank/type combinations missing from the prior
matrix; both are checked against analytic results in the interpreter and
compiled C99.
Other tests cover tensor and scalar division gradients, a shared branch near the
`f32` integer precision boundary, subnormal scalar and tensor gradients, and
gradients that overflow to positive infinity. Coverage includes a shared
scaled tensor value used by both `sum` and `mean` and explicit linear unit
conversion. These selected cases are checked against analytic results under
GCC and Clang. Conversion tests also cover affine-temperature and angular
conversion derivatives in the interpreter and native C99. Other `f32` inputs
may still differ in least-significant bits across operations or compiler
targets.

## Runtime and diagnostic behavior

- A statically unsupported target or operation is rejected during compilation;
  no fallback derivative is synthesized.
- A tensor input with the wrong runtime shape or element type is rejected
  consistently with the existing function-boundary checks.
- Unit mismatches or unit ratios that cannot be represented are rejected; no
  unit is silently discarded.
- The objective's scalar result must have an explicit `f32` or `f64`
  annotation. Its unit is included in the derivative type.
- Division by zero follows the current tensor runtime contract: interpreter
  error and native C99 trap.
- NaN and infinity propagation follows the underlying floating-point
  operations. No cross-toolchain bitwise equality is promised.
- Zero-sized dimensions are already rejected by the tensor type contract.
- New diagnostic identifiers and exact messages should be decided with the
  implementation tests, rather than treated as fixed by this draft.

## Memory and ownership boundary

Reverse mode retains forward values until the reverse pass consumes them. With
`N` elements per input tensor and `M` tensor-valued graph nodes, a direct tape
may require memory proportional to `M * N`, in addition to adjoints and scalar
nodes. Static shape alone does not make stack allocation safe or define an
acceptable maximum.

The interpreter keeps the tape in invocation-local Python storage. Generated
C99 uses one checked heap allocation per gradient call for the scalar tape,
traps on size overflow or allocation failure using the backend's fail-fast
runtime policy, copies the adjoints into the returned tensor value, and frees
the tape before returning. Process-aborting traps do not unwind; this narrow
policy does not settle general frame, heap, ARC, FFI, or GPU ownership. See
[memory-ownership-design.md](./memory-ownership-design.md).

## Example

```aether
fn objective(values: Tensor<f32, [2]>) -> f32 {
    return sum(values * values);
}

fn derivative(values: Tensor<f32, [2]>) -> Tensor<f32, [2]> {
    return gradient(objective, values);
}
```

For finite values, the intended result is element-wise `2 * values`.

## Explicitly deferred

Tensor-valued objectives and Jacobians, indexing, nonsmooth operations,
control flow, dynamic shapes, higher-order differentiation, fusion, and CUDA
are outside v0. Linear units and explicit affine-temperature and angular
conversions are supported for the implemented subset using the separate
unit-expression system. No production memory-safety or allocation guarantee
follows from this prototype.
