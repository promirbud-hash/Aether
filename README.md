# Aether prototype

The initial Python prototype explores lossless `.hl` parsing, compact AI views,
code topology, and tokenizer-based size comparisons.

## Setup and tests

```powershell
py -3.10 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

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
`aether/getViews`, `aether/sourcePosition`, and `aether/viewPosition`. The view
request can choose a tokenizer profile and active map node; returned mappings
relate each emitted token (including aliases) to source and view ranges.
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

The CST preserves exact source text, and the deterministic frontend builds
syntax-aware declaration nodes plus an AST for the documented v0 subset.
`docs/aether-subset.md` records that grammar and its current semantic checks.
The analyzer reports syntax errors, duplicate declarations/bindings, unknown
local names/calls, type and call mismatches, invalid tensor dimensions, unit
incompatibilities, and undeclared transitive effects. Its bounded inference
covers the documented expression subset; it is not a complete compiler type
system. The map extractor uses AST declarations, effects, recognized units,
calls, and function-to-type references.
The stable Python entry points are `compile_source` and `run_source` in
`src.api`; compilation fails with diagnostics rather than executing an invalid
module. The reference interpreter validates function-boundary primitive,
struct, tensor, and `Result` payloads for the supported subset.
For multi-file projects, `compile_project(entry_path)` loads relative `.hl`
imports and returns a combined typed IR; `run_project(entry_path, function_name,
...)` runs an entry-module function. `emit_c99_project`,
`build_project_object`, and `build_project_executable` provide the supported
Safe-C99 build path for projects. Imports use qualified names (`alias::function`
and `alias::Type`) and reject cycles. Workspace/LSP import resolution is not
yet supported. Import paths resolve from the importing file and must remain
inside the entry file's directory; the same module may be imported through
multiple distinct aliases.
Unit conversions are explicit (`convert(value, "m")`) and are represented as
rational scale/offset operations (plus a symbolic pi power for angular units)
in semantic results and executed by the reference interpreter. Its typed IR
preserves source spans, inferred expression types, effects, and conversion
metadata.
Absolute-temperature arithmetic is rejected rather than implicitly
approximated. Effect annotations currently describe the effect contract
checked against effects inferred from the built-ins `io_write`, `io_read_line`,
`fs_read_text`, `fs_write_text`, `fs_exists`, `net_get`, and `net_post`, then
propagated through calls. These built-ins return typed `Result<T, Error>`
values. The reference interpreter executes `Result` matching and `?`, but
those operations are inert unless an embedding caller injects a handler that
returns `ResultValue`. No real file, network, or console access occurs by
default.
`Result<T, E>` has a restricted exhaustive `match` expression over `Ok` and
`Err`, plus postfix `?` for propagating the same error type from a function
returning `Result`. The reference interpreter supports this subset along with
scalar/tensor arithmetic and unit conversion. Its role is executable
prototyping, not a production runtime or memory-safety boundary. See
`docs/aether-subset.md` for the supported operations and limitations.
The prototype also parses typed `extern "C"` declarations with `@link` and
checks the `@ffi` effect. Interpreter tests can provide explicit
`external_handlers`; no shared library is loaded and no linker is run. The
supported ABI types and current limitations are documented in
`docs/aether-subset.md`.

The experimental Safe-C99 backend is available as `emit_c99(source)` from
`src.api`. It emits deterministic C99 for a checked scalar subset and rejects
unsupported features with diagnostics. `build_object(source, output_path)`
compiles that subset to a host object file. `build_executable(source,
output_path)` links a program whose entry point is `fn main() -> i32` or
`fn main() -> void`. Native C ABI calls require explicit `libraries` mappings
from each `@link` name to a linker-compatible file. The build APIs do not
dynamically load libraries or run generated programs. See
`docs/aether-subset.md` for the supported operations and limitations.
