# Aether roadmap and implementation status

This roadmap consolidates the work recorded in the repository and development
checkpoints. Statuses describe the current prototype, not production readiness
or completion of the full Aether vision.

## Status key

- **Done (prototype):** implemented and covered by tests for the documented
  subset.
- **Partial:** useful subset exists, but important language/runtime guarantees
  or target coverage remain.
- **In progress:** the milestone has begun; listed unfinished parts are still
  future work.
- **Not started:** no implementation is claimed.
- **Design in progress:** a design draft exists, but decisions and
  implementation are not complete.
- **Unverified:** implementation exists, but the stated environment or
  integration has not been validated.
- **Design baseline proposed:** a concrete design draft exists, but review,
  acceptance, and implementation are still required.
- **Accepted:** semantics have been approved; implementation may still be
  partial or not started.

## Framework plan

| # | Milestone | Status | Current scope and remaining work |
|---|---|---|---|
| 1 | `.hl` source, CST, and lossless roundtrip | **Done (prototype)** | `.hl` is the editable source; the CST retains whitespace/comments and supports exact source reconstruction. The grammar remains a documented subset, not a complete language specification. |
| 2 | Tri-View: `.ai` view, `.map`, and BPE benchmark | **Done (prototype)** | Deterministic views, topology extraction, profile-based aliasing, and token-count comparisons exist. Qwen/Llama exact counts require locally available tokenizer packages/data; the Claude profile is explicitly a proxy. |
| 3 | AST frontend and diagnostics | **Partial** | The syntax-aware frontend parses and checks the supported language subset with source locations. General-purpose control flow, a complete grammar, and a production-grade recovery strategy are not claimed. |
| 4 | Offline LSP and developer measurements | **Done (prototype)** | Stdio LSP, incremental document updates, diagnostics, Tri-View requests, and source/view mappings are implemented. The server uses deterministic offline alias fallback; benchmark figures are environment-specific. |
| 5 | Static types, units, and effects | **Partial** | Bounded inference, static tensor shapes, effect checking, exact SI-prefix scales, and compound unit expressions in numeric/tensor types exist. The unit registry and type/effect analysis remain bounded; this is not a complete type/effect system or proof of memory safety. |
| 6 | `Result`, `match`, and `?` | **Done (bounded subset)** | Restricted `Ok`/`Err` matching, propagation, and checks for supported result use are implemented. General enums, unrestricted pattern matching, and full control-flow analysis remain out of scope. |
| 7 | Typed IR and reference interpreter | **Partial** | The typed IR retains types, spans, effects, and unit-conversion metadata. The interpreter executes the documented subset and requires injected handlers for external effects; it is a reference implementation, not a production runtime. |
| 8 | C ABI FFI | **Done (restricted subset)** | Scalar `extern "C"` declarations, effect checking, explicit link-library mappings, and native host linking are implemented. Pointers, aggregates, callbacks, strings, tensor ABI, and ownership rules are not supported. |
| 9 | Multi-file modules and project compilation | **Done (restricted subset)** | Relative imports, aliases, cycle checks, project compilation, and per-module C99 translation units are implemented under documented path and visibility rules. Workspace/LSP import resolution and a general package system remain future work. |
| 10 | Deterministic C99 CPU backend | **Partial** | Checked scalar lowering and the first static floating-tensor subset are implemented. C99 builds disable FP contraction; supported conversion constants are emitted as single hexadecimal literals. `f64` sum/mean use sequential row-major accumulation in the interpreter and C99 backend, covered by order-sensitive tests. Interpreter/C99 parity is covered for selected operations, not promised for every input or target. Unsupported constructs are diagnosed rather than substituted. |
| 11 | Host builds and artifact manifests | **Done (restricted workflow)** | Object/executable builds, explicit FFI library maps, multi-object workflows, and source/object fingerprint checks exist. Object builds are tested for repeatability across temporary directories; source hashes normalize line endings while parsing preserves original offsets. The manifest detects staleness/tampering but is not a signed security boundary; incremental caching is not implemented. |
| 12 | Toolchain and CI reproducibility | **Partial / unverified remotely** | All 243 tests pass locally with GCC and Clang using strict C99 flags. The latest GitHub Actions rerun could not allocate runners because the account is locked due to a billing issue; remote CI success must not be inferred from local results. |
| 13 | Memory and ownership model | **Accepted / bounded ARC implemented** | The accepted baseline defines value semantics, rebinding-only `var`, frame/tape lifetimes, acyclic ARC, optional COW, fail-fast implicit allocation, cleanup on `?`, and explicit transfer for future ownership-bearing FFI. The implemented ARC subset supports retain/release and owned returns for immutable `FrameBuffer` storage, borrowed parameters, temporary owners, and LIFO cleanup in the interpreter and C99 backend. General ARC, COW, heap tensor storage, arbitrary destructors, and ownership-bearing FFI remain unimplemented. |
| 14 | Tensors, autodiff, fusion, and CUDA | **In progress** | Static shapes, the interpreter tensor contract, C99 CPU tensors, scalar and bounded tensor reverse-mode autodiff v0, and unit-prefix/compound-expression prerequisites are implemented. The C99 `f32` mean-gradient factor is now rounded consistently with the interpreter and covered by GCC/Clang regression tests; broader tensor-backend conformance hardening remains. Operator fusion, CUDA execution, and GPU validation are future work. |

### Tensor milestone detail

Completed parts of milestone 14:

1. Define and test static tensor shapes, supported operations, unit behavior,
   reductions, leading-axis indexing, and runtime index checks in the semantic
   analyzer and reference interpreter.
2. Reject tensor comparisons and zero-rank shapes until their result/type
   contracts are defined.
3. Emit deterministic C99 row-major structs for static `f32`/`f64` tensors and
   implement the documented initial CPU operation subset, with diagnostics for
   unsupported tensor element types and operations.
4. Compile and run tensor C99 tests under strict flags with GCC and Clang.

Still open in milestone 14:

5. Continue hardening tensor backend conformance: differential tests now cover
   f64 elementwise/broadcast operations and rank-three indexing, representative
   reductions with NaN, infinities, and signed zero, plus linear,
   affine-temperature, and angular conversions. Multidimensional indexing,
   `min`/`max` NaN ordering, and tensor division-by-zero behavior also have
   native coverage. The interpreter now rounds `f32` inputs, operation results,
   reductions, conversions, and autodiff forward values/local derivatives to
   single precision, matching the C99 backend; selected scalar/tensor cases
   compare exactly, including overflow and subnormal results. Scalar
   broadcasting and same-shape tensor arithmetic have differential tests for
   representative `f32` and `f64` tensors of ranks one through three; semantic
   tests reject shape-expanding broadcasts and unsupported tensor/scalar
   operator directions. `f32` autodiff now has differential and analytic
   coverage for a shared scaled value used by `sum` and `mean` and for explicit
   linear unit conversion. Additional differential tests cover a shared branch
   near the `f32` integer precision boundary, subnormal scalar/tensor gradients,
   gradients overflowing to positive infinity, and tensor/scalar division
   gradients with positive, negative, and small nonzero denominators. New
   cancellation tests confirm that double-precision adjoint accumulation
   preserves a unit residual between many cancelling contributions of
   magnitude `2^24` across different branch orders, including for tensor
   inputs, with GCC and Clang. Broader numerical coverage and validation on
   additional targets remain open. Scalar and `sum`-based tensor gradients
   also test final `f32` rounding at an exact ties-to-even midpoint and just
   above it under GCC and Clang, covering both midpoint parity outcomes and
   negative gradients. Scalar and `mean`-based tensor gradients also verify
   subnormal ties at half and 1.5 times the least positive subnormal, including
   signed zero and the second subnormal. Around 2.5 subnormal units, tie-even
   and immediately neighboring values verify rounding to the second or third
   subnormal, with symmetric negative cases. Immediately below and above half
   a subnormal are also tested to establish the zero-to-subnormal transition
   under GCC and Clang.
6. Specify the scalar `gradient(f, x)` v0 contract: **Done** for unitless
   `f32`/`f64`, pure unary targets, the supported arithmetic expression subset,
   result type, and division-by-zero behavior. The bounded tensor-gradient
   contract is also **Done (v0)** as recorded in item 9 below.
7. Implement deterministic scalar autodiff in typed IR and validate it in the
   reference runtime: **Done (v0)** with reverse-mode accumulation and explicit
   diagnostics for unsupported targets.
8. Lower scalar autodiff to CPU C99 and compare native results with the
   reference implementation and analytic cases: **Done (v0)** for the
   documented subset, including GCC/Clang and imported project targets.
9. Finalize and implement the bounded static tensor-to-scalar gradient
   contract in the interpreter and C99 backend: **Done (v0)** for `f32`/`f64`,
   supported linear units, explicit linear conversion, arithmetic,
   broadcasting, `sum`, and `mean`. The separate ownership baseline is
   accepted, with bounded `FrameBuffer` ARC implemented; general ARC remains
   out of scope.
10. Design operator fusion as a separate IR optimization with deterministic
   output and equivalence tests; do not make fusion a prerequisite for correct
   autodiff semantics.
11. Add CUDA code generation/runtime only after the tensor and autodiff
    contracts are stable; validate CPU/GPU parity on supported hardware.

## Current position

The project is at **milestone 14**. The first CPU tensor subset, scalar
reverse-mode autodiff v0, and composable SI-prefix/unit-expression support are
implemented in the prototype. Tensor conformance hardening remains partial:
differential tests compare C99 and interpreter behavior for representative
reductions, exceptional floating values, tensor division-by-zero, and
linear/affine/angular conversions. A bounded tensor reverse-mode gradient is
now implemented in the interpreter and C99 backend, including exact unit
quotients, explicit linear conversions, `sum`/`mean`, broadcasting, and
invocation-local tape cleanup. The planned bounded conformance increment is
complete and passes locally under GCC and Clang. Broader precision, additional
broadcast/rank combinations, and target-specific behavior remain open. The
current differential suite covers representative scalar-broadcast and
same-shape cases across ranks one through three, but does not exhaust all
shape/operator/type combinations. Fusion and CUDA remain separate later
milestones.

The current local verification is **234 tests passing with GCC and 234 with
Clang**, plus `compileall` and `git diff --check`. This does not substitute for
remote CI or production qualification.

Milestone 13's baseline is accepted. The bounded `FrameBuffer` ARC increment
is implemented for the documented prototype boundary:

- `frame_buffer(size)` creates immutable shared storage with checked positive
  size, exactly one reference per owning binding, and fail-fast allocation and
  refcount overflow behavior.
- The analyzer permits local copies, rebinding, borrowed parameters, owned
  returns, and `Result<FrameBuffer, E>` within the subset; resource-bearing
  aggregates and FFI remain rejected.
- Typed IR records borrowed/owned values, temporary owners, retain/release/
  transfer actions, and frame cleanup order. Interpreter and C99 execution
  retain before releasing on rebinding and clean temporary owners after each
  full expression.
- Tests instrument reference counts and allocator hooks for aliases, self
  rebinding, parameter returns, `Ok`/`?`, nested calls, LIFO cleanup,
  allocation failure, and size/refcount overflow.
- A process-aborting trap still does not unwind or promise cleanup.

This increment does not implement general ARC, COW/mutation, arbitrary
destructors, or heap-backed tensors. Those remain separate future gates in
[memory-ownership-design.md](./memory-ownership-design.md), not blockers for the
bounded `FrameBuffer` contract.

The tensor-autodiff contract and its bounded conformance increment are
implemented; further broadening is a later milestone-14 task, not the next
dependency for ownership cleanup.

## Deferred vision work

The optional AI/LLM layer and Aether Studio are intentionally separate from the
deterministic compiler. They are not prerequisites for language semantics,
native CPU correctness, autodiff, or CUDA. Grammar-constrained model assistance
may be explored later, but it must remain optional and must not alter
deterministic builds.
