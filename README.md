# Aether

**A small, statically checked programming language prototype with a Python
reference interpreter and a C99 backend.**

Aether is being developed toward a practical, portable language whose programs
can be checked and executed consistently in an interpreter and compiled to
native C. The repository also explores lossless `.hl` editing, compact AI
views, code topology, and tokenizer-based size comparisons.

> **Project status:** experimental prototype, not a 1.0 release and not a
> Python-compatible implementation. The language and backend deliberately
> support a documented subset; they diagnose unsupported features rather than
> promising general-purpose behavior.

## Goals and design assumptions

- **A usable language core first.** Grow syntax, types, control flow,
  structures, collections, and runtime features as coherent end-to-end slices.
- **Comparable execution paths.** The interpreter is the reference behavior;
  supported C99 constructs should be checked against it with tests.
- **Explicit limits and failures.** Document ownership, effects, and backend
  boundaries. Reject unsupported constructs with diagnostics instead of
  emitting plausible but incorrect code.
- **Portable output.** Use C99 as a practical compilation target, with an
  embedded/freestanding profile as a separate, explicitly validated subset.
- **Self-compilation is a future milestone.** It is a goal to measure with
  bootstrap stages, not a capability claimed by this repository today.
- **AI views are developer tooling.** `.ai` views and topology maps are
  deterministic representations of source; they do not change program
  semantics or replace `.hl` as the source of truth.

The project is intentionally not trying to implement all of Python, nor to
promise GUI frameworks, concurrency, or a broad standard library before real
programs establish their requirements.

## Current state at a glance

| Area | Current status |
|---|---|
| Language | Functions, static types, units, effects, `Result`, control flow, bounded structs, tensors, and collections are implemented to varying documented limits. |
| Runtime | The interpreter supports console and project-root file I/O, arguments, injectable handlers, and a bounded ownership model. |
| C99 | A useful but incomplete subset has native builds, typed diagnostics, integer tensors, autodiff, FFI, and host runtime support. It is not feature-identical to the interpreter. |
| Embedded | Freestanding C99 objects have been cross-compiled with LLVM Clang for ARM Cortex-M0+/M4, RISC-V32, and AVR. A full board SDK firmware link/run is not yet claimed. |
| AI views and LSP | Lossless source/CST, Tri-View support, offline LSP, and versioned `.ai` edit proposals are implemented as prototype tooling. |
| Self-compile | Not started. There is no Aether-written compiler or bootstrap test suite yet. |

Collection allocation-failure parity, broader backend conformance, a unified
end-user CLI, and a fuller standard library remain open work. See the
[roadmap](docs/roadmap.md) for status and priorities.

## A small example

The Python API can compile and execute a function in the supported subset:

```python
from src.api import run_source

program = """
fn sum_to(limit: i32) -> i32 {
    var total = 0;
    for index in 0..limit {
        total += index;
    }
    return total;
}
"""

print(run_source(program, "sum_to", 5))  # 10
```

`emit_c99(source)` emits C99 for supported constructs. `run_source` is useful
for prototyping and testing; it is not a security sandbox or a production
runtime. See the [language subset](docs/aether-subset.md) for exact semantics.

## Setup and tests

The project supports Python 3.10 and newer. CI runs the GCC/Clang test matrix
on Python 3.13 and the Python package/lint workflow on Python 3.10. Python
3.10+ is required for the language tooling and tests; `requirements.txt`
contains `tiktoken`, needed by the default exact-tokenizer benchmark. Install
the development dependencies as well if you want to run the optional pytest
and lint workflows:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The standard-library `unittest` suite is the canonical test command. Native
C99 tests require the selected compiler to be on `PATH`; these commands make
the compiler mandatory so a missing compiler fails instead of silently
skipping native tests:

```powershell
$env:AETHER_REQUIRE_C99_COMPILER = "1"
$env:AETHER_C99_COMPILER = "gcc"
.\venv\Scripts\python.exe -m unittest discover -s tests -v

$env:AETHER_C99_COMPILER = "clang"
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run Python syntax and whitespace checks separately when validating a change:

```powershell
.\venv\Scripts\python.exe -m compileall -q src tests
git diff --check
```

`pytest` and `flake8` are development-only tools used by the secondary package
workflow; they are not needed to run the canonical unittest suite.
`transformers` is optional and is needed only for Qwen/Llama tokenizer profiles;
those profiles use locally cached models and do not download them. MSVC,
CUDA/NVCC, and a CUDA-capable GPU are optional: tests depending on unavailable
toolchains or hardware are skipped, which is not evidence that those targets
passed validation.

## Generate views and benchmark

```powershell
.\venv\Scripts\python.exe -m src.benchmark
```

The command writes `examples/sensor.map` and one profile-specific AI view for
each available tokenizer, then reports `.hl`, `.ai`, and graph-slice token
counts separately for every profile. The default profile measures the exact
GPT-4o `o200k_base` tokenizer. Select profiles with repeatable `--profile`
options, for example:

```powershell
.\venv\Scripts\python.exe -m src.benchmark --profile qwen2.5-coder
```

Profiles are JSON files in `profiles/`; they define tokenizer backend/model,
operator and idiom spellings, and semantic alias candidates. Alias choices are
measured against the full rendered view (including the expansion header) with
that profile's encoder. No tokenizer or model is silently substituted.

The Qwen and Llama profiles use Hugging Face tokenizers only when the tokenizer
is already cached locally. They never download files; install `transformers`
and cache the model tokenizer separately if exact local measurements are
desired. Llama model access may require accepting its license. The Claude
profile is explicitly an `o200k_base` proxy, not Claude's tokenizer; provide a
real encoder to `benchmark_profiles` for exact Claude measurements.

Use `--source`, `--python`, `--active`, and `--profile-dir` to select input files,
the graph node, or an alternate profile directory. Active nodes use ids such as
`f:normalize_voltage` and `t:SensorFrame`.

## LSP and source mapping

Start the standard-I/O language server with:

```powershell
.\venv\Scripts\python.exe -m src.lsp
```

The server supports UTF-16 LSP ranges, incremental `didChange`, push and pull
diagnostics, and document-scoped cache invalidation. Custom requests are
`aether/getViews`, `aether/sourcePosition`, `aether/viewPosition`, and
`aether/applyAIViewEdits`. The view request can choose a tokenizer profile and
active map node; returned mappings relate each emitted token (including
aliases) to source and view ranges. `applyAIViewEdits` validates edits against
the current `.ai` view and document version, expands aliases, and returns a
versioned full-document `WorkspaceEdit` plus candidate diagnostics and mappings.
The client applies the edit and reports it through the normal `didChange`
notification.
LSP generation is deliberately offline: it uses the profile's stable fallback
alias ordering and does not initialize or download tokenizer/model packages.
For tokenizer-optimized output use the benchmark/generator API with an explicit
encoder.

## Latency and memory measurement

```powershell
.\venv\Scripts\python.exe -m src.dev_benchmark --iterations 100
```

This reports median/P95/max frontend-plus-Tri-View time and peak traced Python
allocation for the selected source. Measurements describe the current
environment and are not performance guarantees. It defaults to the offline,
deterministic phraser path; add `--tokenizer-optimized` to separately measure
the slower tokenizer-guided alias search.

## Language and runtime

The lossless CST preserves the exact `.hl` source. The syntax-aware frontend,
semantic checks, and typed IR cover the documented subset, including
`if`/`else`, loops, ranges, scoped bindings, `Result`, units and effects,
value-semantic structs, static tensors, and a bounded collection API.
Unsupported syntax and backend operations produce diagnostics. This is not a
complete type system or a claim that every feature behaves identically on all
targets.

The main Python API is in `src.api`:

- `compile_source(source)` returns typed IR or raises `CompilationError`.
- `run_source(source, function_name, *arguments)` executes a function with the
  reference interpreter.
- `compile_project(entry_path)` and `run_project(...)` handle restricted
  relative imports and qualified names such as `math::sum`.

The interpreter provides console I/O, project-root-confined file operations,
and process arguments, with injectable streams and handlers for testing and
embedding. Network calls remain handler-only. Strings use UTF-8. Runtime
behavior and ownership guarantees are bounded by the contracts in
[`docs/aether-subset.md`](docs/aether-subset.md) and
[`docs/memory-ownership-design.md`](docs/memory-ownership-design.md).

## C99 and embedded targets

`emit_c99(source)` generates deterministic C99 for the supported subset.
`build_object` and `build_executable` compile and link host programs;
multi-module equivalents and manifest-checked object workflows are also
available. FFI libraries require explicit `@link` mappings. The C99 backend
does not silently replace unsupported language operations.

The separate `@target(mcu)` profile emits freestanding code, rejects hosted
runtime calls and dynamic collections, and supports a bounded set of integer
tensors and borrowed byte-buffer FFI parameters. Generated objects have been
cross-compiled with LLVM Clang for several embedded target triples. This does
not mean Aether supplies board startup, drivers, linker scripts, or a complete
SDK build. See the [Pico SDK example](examples/embedded/pico/README.md) and
[roadmap](docs/roadmap.md).

Scalar and bounded tensor reverse-mode autodiff are implemented for the
documented subset. A separate experimental CUDA emitter supports only a narrow
unitless `f32` elementwise subset and requires a CUDA device; it has no CPU
fallback. See the [autodiff contract](docs/autodiff-tensor-contract.md) and
[CUDA design](docs/cuda-backend-design.md).

## Further reading

- [Language subset and runtime contract](docs/aether-subset.md)
- [Development roadmap and status](docs/roadmap.md)
- [Prototype-to-release roadmap](docs/roadmap_v2.md)
- [Collections and ownership](docs/memory-ownership-design.md)
- [Tensor autodiff contract](docs/autodiff-tensor-contract.md)
- [Units](docs/unit-prefix-design.md)
- [CUDA backend](docs/cuda-backend-design.md)

There is not yet a unified end-user CLI, a broad standard library, or an
Aether-written self-hosting compiler. Those are future work, not implied by
the current prototype.
