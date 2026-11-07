# Aether roadmap v2: from prototype to a real project

## Purpose

This roadmap turns the current research prototype into a maintainable,
usable, and supportable software project. It is a sequence of evidence-based
gates, not a promise that every part of the original language vision will ship.
The first production target must be deliberately smaller than the full vision.

The existing [roadmap](./roadmap.md) remains the detailed record of prototype
milestones and implementation status. This document focuses on the work still
needed for users to rely on Aether.

## What "real project" means

For this roadmap, Aether is ready for an initial stable release when:

- The project has a named target audience and a small set of real workloads
  that it commits to support.
- Users can install a versioned release, write programs within a versioned
  language subset, build them on declared platforms, and understand failures
  from actionable diagnostics.
- Documented semantics are enforced consistently by the frontend, interpreter,
  and supported native backends; unsupported behavior fails explicitly.
- Automated tests, CI, release artifacts, and upgrade guidance provide
  repeatable evidence for each supported configuration.
- Security and operational limitations are explicit. In particular, the
  current interpreter and generated C are not sandbox boundaries.

Passing the current local suite is useful prototype evidence, but is not by
itself a production-readiness claim.

## Current baseline

- The repository is a Python prototype with a documented `.hl` syntax subset,
  CST/AST frontend, typed IR, interpreter, offline LSP, project imports, and an
  experimental Safe-C99 backend.
- Selected static `f32`/`f64` tensors, scalar and bounded tensor autodiff,
  units, a restricted `Result` subset, a bounded `FrameBuffer` ownership
  increment, and conservative C99 tensor fusion are implemented.
- The latest recorded local validation is 286 tests run with GCC and 286 with
  Clang; all non-skipped tests pass, plus `compileall` and `git diff --check`.
  All six CUDA backend tests also pass with the MSVC environment initialized,
  including interpreter/C99/GPU comparisons across block-boundary sizes and
  host-runtime error paths.
- Several guarantees remain bounded or unverified: the language is a subset,
  memory ownership is not general, tensor/backend conformance is representative
  rather than exhaustive, remote CI is blocked by the account billing lock,
  and there is no stable release/install/support process.
- A local benchmark of three representative arithmetic-chain patterns showed
  faster fused medians under GCC and Clang for these selected cases. All
  measurements are machine-specific evidence, not a general performance
  claim. An
  NVIDIA RTX 3060 and CUDA Toolkit 12.9 are present, and a CUDA runtime probe
  successfully enumerated one device. A bounded experimental CUDA v0 emitter
  and host wrapper now compile and run across representative grid-stride
  boundary sizes; broader device and operation coverage is still absent.

See [the subset specification](./aether-subset.md),
[the detailed roadmap](./roadmap.md), and the focused
[ownership](./memory-ownership-design.md),
[tensor autodiff](./autodiff-tensor-contract.md), and
[fusion](./operator-fusion-design.md) contracts for exact boundaries.

## Phases and exit gates

| Phase | Goal | Current progress | Work required to exit |
|---|---|---|---|
| 0. Product definition | Decide what the first dependable Aether release is for | **Not started as a formal project decision.** The repository demonstrates compiler, AI-view, LSP, units, tensor, and native-build experiments, but does not state one committed first-release audience or workload. | Name the primary user and use case; choose the initial target platforms and compiler toolchains; define non-goals and support expectations; select a small representative program suite. Record the choices in a product brief and use them to bound later phases. |
| 1. Language contract | Make the supported language behavior stable and reviewable | **Prototype subset documented.** Syntax and several feature contracts exist, but documentation is distributed and some semantics are explicitly partial. | Publish a versioned language specification for the chosen release subset; resolve gaps such as control-flow/return checking, integer overflow, error behavior, effect execution, unit edge cases, and FFI boundaries; establish compatibility and deprecation rules; ensure unsupported constructs have stable diagnostics. |
| 2. Compiler foundation | Make compilation and diagnostics dependable for ordinary projects | **Core exists; production behavior incomplete.** Frontend, typed IR, imports, and deterministic C99 emission work for bounded cases. | Define public API boundaries; improve parser recovery and diagnostic consistency; validate all source/module paths and limits; add source maps and actionable errors across project builds; check recursion, size arithmetic, and resource exhaustion; prevent internal exceptions from leaking as user-facing success or opaque crashes. |
| 3. Runtime and semantic safety | Define trustworthy execution and memory behavior | **Partial, with accepted bounded ownership baseline.** Interpreter, tensor autodiff tape cleanup, and `FrameBuffer` ARC cover specific paths; this is not general memory safety. | Choose whether the initial product is compiler-only, interpreter-enabled, or both; finish ownership/escape rules for every shipped type; define allocation failure and trap behavior; close normal and recoverable-error cleanup paths; decide whether execution of untrusted source is unsupported or requires a separately designed sandbox. Never imply sandboxing from current checks. |
| 4. Backend conformance and performance | Make supported native output correct and useful on declared targets | **Experimental C99 backend; representative conformance.** GCC and Clang pass the local suite; fusion has f32/f64 non-associativity tests and three benchmarked chains. CUDA v0 now compiles and runs a narrow f32 elementwise kernel on the local RTX 3060; tensor autodiff and both backends remain restricted. | Freeze an explicit operation/type/shape support matrix; expand differential and boundary tests for every shipped feature; test multiple optimization levels and target environments; validate generated C with supported compilers; measure representative compile/runtime costs; retain fusion only where equivalence is proven and benefit measured. Expand CUDA only after CPU conformance and workload needs justify it. |
| 5. Tooling and developer experience | Make the project approachable and diagnosable | **Prototype tools exist.** LSP and deterministic views are implemented, but workspace imports and a complete editor workflow are missing. | Create an installation and quick-start path; provide editor configuration and examples for supported workflows; improve LSP project/workspace behavior and error recovery; specify formatting, linting, and generated-file policies; document debugging and native compiler setup; validate the workflow from a clean environment. |
| 6. Build, packaging, and supply chain | Deliver reproducible, verifiable artifacts | **Build primitives and manifests exist; release packaging absent.** Object and executable builds and source/object fingerprint checks are implemented for the documented workflow. | Define versioned package and artifact formats; make builds reproducible from clean inputs; pin and audit dependencies; validate manifests and toolchain identity; test installation/uninstallation and upgrades; define signing/provenance policy if required by the target users; publish checksums and platform support. |
| 7. Quality engineering and CI | Make every change continuously verifiable | **Local testing is broad for the prototype; remote CI is blocked.** 286 tests pass locally with both GCC and Clang; GitHub confirms the account is locked due to a billing issue, so jobs receive no runner and execute no steps. | Restore usable remote CI by resolving the account billing lock; rerun tests on clean supported OS/toolchain combinations; separate unit, integration, conformance, and release tests; add deterministic property/fuzz tests for parser, analyzer, IR, and code generation; add regression tests for every fixed defect; publish CI results and test the release process end to end. |
| 8. Stable release and maintenance | Support users after initial adoption | **Not started.** No stable release policy or support lifecycle is claimed. | Define semantic versioning and compatibility guarantees; publish release notes and migration guides; set issue triage/security reporting and support policy; create a release checklist and rollback plan; establish a cadence for patch/security releases; gather feedback from pilot users before declaring the first stable release. |

## Recommended sequencing

1. Complete Phase 0 before expanding language scope. Product choices determine
   which semantics, platforms, and tests are actually required.
2. Freeze a small release subset in Phase 1. Do not equate the current parser's
   accepted syntax with a stable language contract.
3. Advance Phases 2-4 against the selected workloads. Treat correctness,
   resource handling, and diagnostics as release gates, not optional polish.
4. Build the clean-install and editor workflows in Phases 5-6 while CI coverage
   grows in Phase 7. These tracks can proceed in parallel once the release
   subset and supported platforms are decided.
5. Run a pilot release before Phase 8's stable-release decision. Expand only
   in response to validated user needs and measured gaps.

## Explicitly defer unless a target workload requires them

- CUDA and other new backends.
- Full general-purpose language features, unrestricted control flow, and a
  complete general type/effect system.
- General ARC/COW, arbitrary destructors, owning FFI aggregates, and mutable
  heap-backed tensors.
- Cross-function/operator fusion, cost-based global optimization, and
  performance claims without repeatable benchmarks.
- Network/model-dependent AI features in the compiler's deterministic build
  path.

Deferral is not a claim that these features are unimportant; it keeps the first
reliable release within a testable and supportable boundary.

## Decisions needed before a stable-release plan

The roadmap intentionally does not assume answers to these product-level
questions:

1. Who is the primary first-release user, and what real task should Aether make
   measurably easier?
2. Is the first deliverable a compiler/toolchain, an embeddable Python
   prototype, or both?
3. Which operating systems, CPU architectures, and C compilers must be
   supported at launch?
4. Is execution of untrusted Aether code a requirement? If yes, what threat
   model and isolation boundary are expected?
5. What compatibility and support lifetime should a stable release promise?

Until these are answered, phases describe the engineering gates, not a fixed
release date or a commitment to the entire language vision.
