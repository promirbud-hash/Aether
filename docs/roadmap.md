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
| 3 | AST frontend and diagnostics | **Partial** | The syntax-aware frontend parses and checks the supported language subset with source locations. Structured `if`/`else`, `while`, half-open integer `for` ranges, lexical scopes, `break`/`continue`, all-path return checking, named struct initializers/field access/limited mutable-field assignment, bitwise operators, and hexadecimal/binary integer literals are implemented across frontend and semantics; a complete grammar and production-grade recovery strategy are not claimed. |
| 4 | Offline LSP and developer measurements | **Done (prototype)** | Stdio LSP, incremental document updates, diagnostics, Tri-View requests, and source/view mappings are implemented. The server uses deterministic offline alias fallback; benchmark figures are environment-specific. |
| 5 | Static types, units, and effects | **Partial** | Bounded inference, static tensor shapes, effect checking, exact SI-prefix scales, and compound unit expressions in numeric/tensor types exist. The unit registry and type/effect analysis remain bounded; this is not a complete type/effect system or proof of memory safety. |
| 6 | `Result`, `match`, and `?` | **Done (bounded subset)** | Restricted `Ok`/`Err` matching, propagation, and checks for supported result use are implemented. General enums, unrestricted pattern matching, and full control-flow analysis remain out of scope. |
| 7 | Typed IR and reference interpreter | **Partial** | The typed IR retains types, spans, effects, and unit-conversion metadata, including structured control-flow blocks and struct declarations/expressions. The interpreter executes the documented subset with struct value-copy semantics, provides real console/file I/O and process arguments with injectable streams/root/handlers, and remains a reference implementation rather than a production runtime. Network effects still require handlers. `Result` bindings and `FrameBuffer` locals inside control-flow blocks are explicitly rejected pending path-sensitive ownership analysis. |
| 8 | C ABI FFI | **Partial, expanded subset implemented** | Scalar `extern "C"` declarations plus parameter-only `Slice<u8>`/`SliceMut<u8>` for one-dimensional static byte tensors are implemented; byte slices lower to pointer/length pairs and mutable slices require a `var` binding. Imported extern declarations preserve their C symbol and support explicit library mapping in project executable builds. The interpreter uses injected handlers, including `library:symbol` keys. `@C` exports unitless C scalar functions, and APIs emit source/project headers. General pointers, aggregates, callbacks, strings, returned/stored slices, and ownership-bearing FFI remain unsupported. |
| 9 | Multi-file modules and project compilation | **Done (restricted subset)** | Relative imports, aliases, cycle checks, project compilation, and per-module C99 translation units are implemented under documented path and visibility rules. Workspace/LSP import resolution and a general package system remain future work. |
| 10 | Deterministic C99 CPU backend | **Partial** | Checked scalar lowering, structured control flow, by-value user structs with named initialization/member access/direct mutable-local field assignment, checked `i8/i16/i32` tensor operations/reductions/indexing, `u8` FFI buffers, matching-type integer bitwise operations and `0x`/`0b` literals, a freestanding `@target(mcu)` profile, restricted `@C` exports/header emission, and the host runtime for UTF-8 strings, console I/O, project-relative files, and process arguments are implemented. Struct fields with undefined aggregate ownership (`str`, `Result`, `FrameBuffer`) and recursive by-value layouts are rejected. C99 currently diagnoses `**` as unsupported although the parser/interpreter accept it. C99 integer arithmetic overflow traps; interpreter operations use unbounded Python integer intermediates and check declared types at function boundaries, so overflow within an expression can differ from C99 even if a later operation brings the value back into range. The C runtime uses process working directory as its project root; multi-module runtime state is shared across generated translation units. C99 builds disable FP contraction; supported conversion constants are emitted as single hexadecimal literals. `f64` sum/mean use sequential row-major accumulation in the interpreter and C99 backend, covered by order-sensitive tests. Representative scalar/tensor autodiff parity passes under GCC and Clang at `-O0` and `-O2`; interpreter/C99 parity is covered for selected operations, not promised for every input or target. The MCU profile omits host runtime/stdlib support and dynamic allocation. Standalone generated freestanding code has been cross-compiled with LLVM Clang 21.1.7 for ARM Cortex-M0+/M4, RISC-V32 `rv32imac`/`ilp32`, and AVR ATmega328P; ARM GCC and Pico SDK firmware linking remain unverified. `AETHER_TRAP()` overrides must not return. Unsupported constructs and built-in types without defined representations are diagnosed rather than substituted. |
| 11 | Host builds and artifact manifests | **Done (restricted workflow)** | Object/executable builds, explicit FFI library maps, multi-object workflows, and source/object fingerprint checks exist. Object builds are tested for repeatability across temporary directories; source hashes normalize line endings while parsing preserves original offsets. The manifest detects staleness/tampering but is not a signed security boundary; incremental caching is not implemented. |
| 12 | Toolchain and CI reproducibility | **Partial / CI verified** | The latest full pytest run passed 344 tests with GCC (5 skipped) and 345 with Clang (4 skipped), with 267 subtests passing in each run and the compiler required by the test configuration. Four skips in each run require an active MSVC x64 environment; the GCC run additionally skipped one native test blocked by Windows Defender. `cl.exe` is not available in the current environment. NVCC is available: four CUDA tests pass, while three CUDA tests requiring MSVC x64 are skipped. Python `compileall` and `git diff --check` pass. The previous Visual Studio 2022 x64 validation passed all 306 then-existing tests, including direct generated-C99 compilation/execution with `cl.exe` at `/Od` and `/O2`, and all seven CUDA tests using NVCC and the local GPU. The Pico SDK cross-build remains unverified: `PICO_SDK_PATH`, CMake, Ninja, and `arm-none-eabi-gcc` are unavailable here. GitHub Actions passed both GCC/Clang jobs and the Python package workflow on commit `cbfdb61` (runs [42405045830](https://github.com/promirbud-hash/Aether/actions/runs/42405045830) and [42405045834](https://github.com/promirbud-hash/Aether/actions/runs/42405045834)); the newer local changes still need a remote CI run. |
| 13 | Memory and ownership model | **Accepted / bounded ARC implemented** | The accepted baseline defines value semantics, rebinding-only `var`, frame/tape lifetimes, acyclic ARC, optional COW, fail-fast implicit allocation, cleanup on `?`, and explicit transfer for future ownership-bearing FFI. The implemented ARC subset supports retain/release and owned returns for immutable `FrameBuffer` storage, borrowed parameters, temporary owners, and LIFO cleanup in the interpreter and C99 backend. General ARC, COW, heap tensor storage, arbitrary destructors, and ownership-bearing FFI remain unimplemented. |
| 14 | Tensors, autodiff, fusion, and CUDA | **In progress** | Static shapes, the interpreter tensor contract, C99 CPU tensors, scalar and bounded tensor reverse-mode autodiff v0, and unit-prefix/compound-expression prerequisites are implemented. Recent differential tests fill selected rank/type gaps (`f32` rank 2 and `f64` rank 4); representative scalar/tensor gradients agree across the interpreter, GCC, Clang, and direct MSVC-generated C99 at tested optimization levels. The first conservative C99 fusion pass combines eligible single-use elementwise tensor temporaries; non-associative f32/f64 behavior is tested, and three representative arithmetic-chain benchmarks showed local improvement with GCC and Clang. An experimental CUDA v0 emits a host wrapper and kernel for a narrow unitless f32 elementwise subset; all seven local CUDA tests pass with NVCC, active MSVC x64, and the available GPU. Reductions, division, f64, autodiff, and broader fusion remain out of scope for CUDA. |

## Priority plan: finish the foundations and prepare the public project

The beta goal is a practical general-purpose language; “Python 1.0” denotes
the desired maturity/usefulness, not Python syntax or runtime compatibility.
The collection contract is now accepted and documented in
[aether-subset.md](./aether-subset.md): generic fixed arrays, owned slices,
and copy-on-write dynamic lists, with checked `Result` APIs, value semantics,
and an initial plain-value element subset. The implementation is still open.

The immediate focus is **not expanding CUDA**. Keep the existing CUDA v0
experimental and bounded as documented in
[cuda-backend-design.md](./cuda-backend-design.md); do not treat local GPU test
success as broad device or toolchain support. Work first on the language
foundations and on making the repository straightforward to set up, understand,
and validate. The stages below are ordered work areas, not a claim that every
listed feature is required before publishing the current prototype.

### Stage 0: reproducible development and public repository setup

**Status: onboarding and the canonical test matrix are documented; remaining
public-release preparation is still open.**

- `README.md` documents Python 3.10+, the base and development dependency
  files, canonical `unittest` commands, and required GCC/Clang runs.
- Optional `pytest`/`flake8`, tokenizer backends, MSVC, CUDA/NVCC, and GPU
  checks are identified separately; skips are explicitly not counted as
  validation for those targets.
- The GCC/Clang workflow uses the same canonical unittest command and requires
  the selected native compiler, so missing compilers fail CI instead of
  silently skipping C99 coverage.
- Review the repository's public-use essentials, especially the project
  license and contribution guidance. No root license file is currently tracked;
  select and add an appropriate license before inviting external use or
  contributions. Do not imply production readiness or unsupported backend
  coverage.
- Keep validating the Python 3.10 package workflow and Python 3.13 GCC/Clang
  matrix; MSVC/CUDA validation remains environment-dependent.

### Stage 1: close the language core (frontend, semantics, IR)

**Status: partial; this is the primary language work after onboarding.**

1. **Control flow and ranges — implemented for the current subset.** AST forms,
   semantic analysis, typed-IR lowering, interpreter behavior, and C99
   generation now cover `if`/`else`, `for`, `while`, half-open `..` ranges,
   `break`, and `continue`. Conditions require `bool`; range bounds must be
   same-type unitless integers, are evaluated once, and iterate upward; blocks
   have lexical scopes; non-void functions require a return on every path.
   Loops do not imply termination or guaranteed return. Path-sensitive
   `Result` consumption and `FrameBuffer` locals inside control-flow blocks
   remain explicitly rejected and need a later ownership-analysis extension.
   Compound assignments `+=`, `-=`, `*=`, `/=` are implemented across the
   frontend, semantic analysis, typed IR, interpreter, and C99 for numeric
   scalar `var` bindings, with differential tests. Fixed-width bitwise
   operators and `0x`/`0b` integer literals are also implemented end to end;
   mismatched operand types are rejected, invalid shifts trap in C99, and
   signed right shift is arithmetic.
2. **Broaden struct support.** The bounded end-to-end subset now supports
   named initializers, field reads, direct field assignment on a mutable local,
   independent value copies, C99 layouts, and qualified imported types.
   Remaining work includes ownership-safe string/resource fields, richer
   nested mutation, layout/API stability guarantees, and broader cross-backend
   edge-case coverage.
3. **Strings and runtime I/O — first subset implemented.** The contract now
   defines UTF-8 strings, C99 pointer/length representation and returned-text
   lifetime, stable `IoError`/`FsError`/`ArgsError` payloads, project-root
   confinement, console/file operations, and `args_len()`/`args_get(index)`.
   Interpreter and C99 have native runtimes; injected interpreter handlers
   remain available. Full `Slice`/collection representations, stronger
   cross-platform path guarantees, and network operations remain open. Network
   calls are handler-only in the interpreter and unsupported in C99.
4. **Collection contract — accepted; implementation in progress.** Preserve
   existing `Tensor[i]` behavior. The parser, semantic checks, and reference
   interpreter now support the initial `Array<T, [N]>` and dynamic
   `List<T>` with plain scalar/struct elements, owned half-open range slices,
   COW list value semantics, `var`-only mutation, and checked `get`/`set`/
   `append`/`pop`/`insert`/`remove` operations returning `Result` with stable
   `CollectionError` codes. C99 lowering includes reference-counted cleanup for
   owned lists/slices across bindings, calls, returns, assignments, and
   control-flow scopes, with differential native tests for mutation, copies,
   returns, and loop-local collections. Native execution of the expanded test
   pass under Clang; the full GCC/Clang test matrix passes, although the GCC
   native executable for one case was blocked by Windows Defender.
   Allocator-failure parity remains unverified. Keep
   FFI byte slices as temporary borrowed views; reject dynamic collections in
   the freestanding MCU profile. Complete cross-backend success/error testing
   before marking the milestone done.

### Stage 2: make Tri-View edits bidirectional

**Status: reverse alias expansion and versioned LSP edit proposals are
implemented for the supported `.hl` grammar; editor-client application remains
to be validated end to end.**

1. `.ai` output always includes a profile-tagged `@a(profile;alias=source,...)`
   header, including when there are no aliases. The reverse converter rejects
   missing/mismatched profiles, malformed or duplicate declarations, stale
   alias choices, and source that fails frontend/semantic validation; accepted
   text is rendered as canonical `.hl`.
2. `aether/applyAIViewEdits` applies ranged UTF-16 edits to the exact current
   view, checks the source version and target version, and returns a versioned
   full-document `WorkspaceEdit`, candidate diagnostics, and refreshed AI/source
   mappings. The server does not mutate an open document on the client's behalf:
   after applying the edit, the client must send the normal `didChange`. Verify
   this roundtrip with an actual editor client before marking the stage complete.

### Stage 3: finish the explicitly supported C99 v1 subset

**Status: deterministic scalar and selected tensor/autodiff paths exist; the
backend remains a diagnosed subset, not a complete implementation of the
interpreter.**

1. **Preserve the bounded C99 contract with regression tests.** `match Result`
   remains supported for its documented payload subset. Regression tests cover
   fixed-width overflow traps, invalid shift counts, signed right shifts,
   wrapping left-shift boundaries, and the stable `E_C99_OPERATOR` diagnostic
   for `**` (supported by the parser/interpreter, not by C99). Matching-type
   bitwise operators and base-prefixed integer literals are lowered through
   the interpreter and C99. The interpreter computes unbounded integer
   intermediates and checks declared types at function boundaries; it is not
   an oracle for C99 overflow traps, including overflows later reduced by
   another operation.
2. Continue interpreter/C99 autodiff parity only against explicit uncovered
   contract cases. Existing tests compare selected scalar/tensor gradients
   across GCC and Clang at multiple optimization levels. They do not cover
   control flow, nested user-function calls, indexing, nonsmooth
   `abs`/`min`/`max`/`max_abs`, or all shape/operator/type combinations; these
   remain outside the v0 differentiable subset rather than unverified claims
   of parity. Run the conditional MSVC check where `cl.exe` is available.
3. **Imported FFI project integration implemented.** Imported `extern "C"`
   declarations preserve their native C symbol despite project-qualified
   internal names. Project builds require explicit `@link` library mappings;
   the reference interpreter supports injected handlers keyed by
   `library:symbol`. Regression coverage exercises handler dispatch, effect
   propagation, explicit mapping, and native linking.

4. **Embedded subset implemented; target integration remains open.** `@target(mcu)`
   generates freestanding C99 with an overridable `AETHER_TRAP()` and rejects
   host runtime/dynamic allocation. C99 supports checked `i8/i16/i32` tensor
   operations, byte-buffer `Slice<u8>`/`SliceMut<u8>` extern parameters, and
   `@C` scalar exports with generated headers. LLVM Clang 21.1.7 successfully
   compiled generated freestanding C99 objects, including integer tensor
   operations and borrowed byte FFI, for ARM Cortex-M0+/M4, RISC-V32
   `rv32imac`/`ilp32`, and AVR ATmega328P; host GCC and Clang also compile the
   profile with `-ffreestanding`. These are object-only compile checks, not
   execution or linked firmware validation. ARM GCC and a full Pico SDK build
   remain unverified because `arm-none-eabi-gcc`, `PICO_SDK_PATH`, CMake, and
   Ninja are unavailable in the current environment. `AETHER_TRAP()` overrides
   are invoked by checked failures and must not return. Generated freestanding
   code alone does not provide peripheral drivers, startup code, or a linker
   script.

### CUDA: frozen experimental scope; future handoff

**Status: experimental CUDA v0 implemented and locally tested; no expansion is
planned in the current foundation phase.** Its present contract is a narrow,
unitless `f32` elementwise subset with explicit host wrappers/statuses and no
CPU fallback. In particular, it does not currently support reductions,
`f64`, division, conversions, indexing, user calls, autodiff, effects, or local
scalar temporaries. Broader architecture/toolchain coverage and CUDA CI are
also absent.

Future CUDA work should be a separately designed and tested project phase:

1. Extend fusion to the specifically supported single-`let` function shape
   only after defining which typed-IR forms are safe to fuse.
2. Define parallel reduction contracts and implement `sum`, `mean`, and
   `max_abs`, including empty/shape rules, floating-point order, exceptional
   values, and shared-memory/resource limits.
3. Define and implement automatic kernel-call generation/integration from the
   C99 entry module, including ownership, errors, device availability, and
   fallback policy. The current wrapper is explicitly called by its host
   caller; automatic dispatch and CPU fallback are not present.
4. Add architecture/toolchain validation and CUDA CI before claiming portable
   support. The local RTX 3060/NVCC result is only evidence for that tested
   setup.

Until that phase is intentionally reopened, treat these items as handoff notes
for an experienced future team, not as current implementation tasks or
publication blockers.

### Stage 4: provide one developer CLI

**Status: no unified `aether` command yet; Python APIs and module entry points
are used directly.**

Create a single `aether` entry point with `build`, `run`, `check`, `view`,
`lsp`, and `bench` subcommands backed by the stable APIs in `src.api`. Define
exit codes, diagnostics/output conventions, project-root and configuration
discovery, and behavior for unsupported backend features. Keep the CLI as a
thin interface over tested APIs rather than duplicating compiler logic.

### Completion gates for each stage

- Write down the supported behavior and unsupported cases before implementing
  a new language/backend feature.
- Add semantic rejection tests and interpreter/native differential tests for
  accepted behavior; exercise imported-module paths where relevant.
- Run the smallest targeted tests first, then the documented full matrix for
  affected targets. Record skipped optional-toolchain tests separately from
  successful runtime validation.
- Update this roadmap and the relevant contract documentation in the same
  change so the public status remains accurate.

### Tensor milestone detail

Completed parts of milestone 14:

1. Define and test static tensor shapes, supported operations, unit behavior,
   reductions, leading-axis indexing, and runtime index checks in the semantic
   analyzer and reference interpreter.
2. Reject tensor comparisons and zero-rank shapes until their result/type
   contracts are defined.
3. Emit deterministic C99 row-major structs for static `f32`/`f64` and
   selected integer tensors (`i8/i16/i32`), including checked arithmetic,
   reductions, indexing, and diagnostics for unsupported operations.
4. Compile and run tensor C99 tests under strict flags with GCC and Clang.
   The supported type boundary now also rejects user-defined names that shadow
   built-ins and rejects `char`, `Pool`, and `Id` until their runtime
   representations are specified. `Slice<u8>`/`SliceMut<u8>` are supported
   only as extern parameters, with `u8` tensors serving as static byte buffers.
   Tests cover function parameters and returns, nested types, and imported
   project modules.

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
   `f64` autodiff tests now check `sum`, `mean`, and their shared product on
   cancellation-sensitive inputs in both row-major orders.
   `f64` quotient-gradient tests cover positive, negative, and small nonzero
   denominators, comparing analytic, interpreter, GCC, and Clang results.
   The interpreter and native C99 backend also verify that an `f64` tensor
   and scalar gradient traps on division by zero. A rank-three `f64` case
   checks scalar broadcast through a shared `mean` value, including
   cancellation-sensitive row-major reduction; a division-gradient case
   extends that check to scalar denominator propagation, while an addition
   case checks that broadcast adjoints are reduced back to the shared scalar.
   Both subtraction operand orders also verify the opposite signs for scalar
   and tensor adjoint propagation. Multiplication checks both scalar/tensor
   operand orders, while unary negation checks the sign of a broadcast
   reduction adjoint. The `[2,2,2]` case complements `[2,1,2]` with three
   non-singleton axes, and `[2,3,2]` checks an uneven inner extent and a
   twelve-element row-major reduction. A matching `f32` case verifies
   interpreter/C99 parity for this uneven rank-three shape. Both `[2,3,2]`
   precision cases batch direct `sum`, `mean`, and shared-product gradients.
   Native rank-four `f32` and `f64` gradients and an `f32` rank-two gradient
   extend selected cross-rank/type coverage; they do not establish exhaustive
   conformance.
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
   autodiff semantics. **Initial bounded pass implemented** for pure, single-
   use f32/f64 elementwise chains with identical static shapes. GCC/Clang tests
   cover non-associative f32/f64 cases to verify expression-order preservation.
   A representative native benchmark shows a local improvement, but is not a
   general performance guarantee; broader operator coverage remains gated on
   conformance and representative measurements.
11. Start CUDA code generation/runtime only after the tensor and autodiff
    contracts are stable. **A narrow experimental v0 is now implemented**:
    `f32` elementwise kernels, explicit host wrappers/statuses, no CPU fallback,
    and a conditional CPU/GPU differential test. NVCC and one RTX 3060 device
    were verified locally. Broader semantics, device/toolchain coverage, and
    CUDA CI remain future work.

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
shape/operator/type combinations. The first bounded fusion increment is now
implemented separately, and an initial CUDA f32 elementwise emitter has passed
a local GPU smoke test. Broader fusion and CUDA conformance remain future work.

The latest full local suites pass with **344 passed, 5 skipped under GCC** and
**345 passed, 4 skipped under Clang**; 267 subtests pass under each compiler.
Four skips per run require an active MSVC x64 environment; the GCC run also
skips one native test blocked by Windows Defender. NVCC is available and four
CUDA tests pass, while three CUDA tests requiring MSVC x64 are skipped. The
current environment has no `cl.exe`, and the embedded Pico SDK cross-build
prerequisites are absent. The previous MSVC x64
verification passed all 306 tests then present with no skips. Generated
scalar/tensor C99 autodiff was compiled directly by `cl.exe` under `/Od` and
`/O2`, with `/W4 /WX` and only C4127 suppressed for compile-time overflow
guards; those native results matched the interpreter and analytic values.
All seven CUDA backend tests also passed previously using NVCC and the local
GPU. In the current full suite, a representative scalar/tensor autodiff
comparison passes at `-O0` and `-O2` with GCC and Clang. `compileall` and
`git diff --check` pass. Remote GitHub Actions was rechecked on commit
`cbfdb61`; both the GCC/Clang test matrix and Python package workflow passed.

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

The first operator-fusion increment is implemented as a separate typed-IR
pass. It preserves expression order and f32 operation boundaries, skips
gradient targets and unsupported/effectful cases, and is checked against
unoptimized C99 plus the interpreter. Three representative native benchmark
chains have been measured with GCC and Clang; see
[operator-fusion-design.md](./operator-fusion-design.md) for its boundaries.

The experimental CUDA v0 emitter and host wrapper are defined in
[cuda-backend-design.md](./cuda-backend-design.md). The native GPU test is
conditional on CUDA, host compiler, and device availability; a successful
local run does not establish support across GPU architectures or toolchains.

## Deferred vision work

The optional AI/LLM layer and Aether Studio are intentionally separate from the
deterministic compiler. They are not prerequisites for language semantics,
native CPU correctness, autodiff, or CUDA. Grammar-constrained model assistance
may be explored later, but it must remain optional and must not alter
deterministic builds.
