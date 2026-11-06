# Aether syntax subset v0

This document defines the syntax accepted by the prototype frontend. It is an
initial, versioned subset, not a complete language specification.

## Python API and reference runtime

The public entry points are `compile_source` and `run_source` in `src.api`.
Compilation parses and analyzes source, then returns typed IR; any syntax,
semantic, or lowering error raises `CompilationError` with source diagnostics.
`run_source` compiles first and executes one named function:

```python
from src.api import run_source
from src.interpreter import ResultValue

program = '''
@io
fn read() -> Result<str, IoError> { return io_read_line(); }
'''
result = run_source(
    program,
    "read",
    effect_handlers={"io_read_line": lambda: ResultValue.ok("ready")},
)
```

Handlers are keyed by the exact built-in operation name and must return
`ResultValue`. Successful handler payloads are checked against the declared
result type; error payloads remain opaque. With no handler, an effectful call
raises `AetherRuntimeError` without performing an external action.

At function boundaries, the interpreter validates primitive argument and
return types, exact struct field sets and field types, rectangular tensor
shapes, and tensor element types. An annotated non-`void` function that reaches
the end without returning raises `AetherRuntimeError`; complete-return-path
checking is not yet a static diagnostic. Struct values are represented by mappings.
Unitful scalar/tensor values are represented by `Quantity`; raw numeric/tensor
inputs are interpreted in the declared input unit, and unitful outputs are
returned as `Quantity`. Integer values use Python's unbounded integer
representation; the interpreter does not define target-width overflow or C99
behavior. Calls have a configurable recursion-depth limit. These checks are
for prototyping and testing, not a sandbox or production runtime.

## Lexical rules

- Identifiers begin with a Unicode letter or `_`; subsequent characters may
  also contain digits.
- Integer and decimal literals, quoted strings, and `true`, `false`, `none`
  are supported.
- `//`, `#`, and `/* ... */` comments and all whitespace are preserved by CST
  parsing. Comments and whitespace are ignored by syntax parsing.
- Delimiters must be balanced. Each syntax/semantic diagnostic uses a 1-based
  line and column and a zero-based, half-open character offset range.

## Grammar (informal EBNF)

```ebnf
module       = { import | attribute* declaration } ;
import       = "import" string "as" identifier ";" ;
declaration  = function | extern_function | type_decl ;
attribute    = "@" identifier [ "(" [ expression_list ] ")" ] ;
function     = ("fn" | "func") identifier "(" [ parameters ] ")"
               [ "->" type ] block ;
extern_function = 'extern "C" fn' identifier "(" [ parameters ] ")"
                  "->" type ";" ;
parameters   = parameter { "," parameter } [ "," ] ;
parameter    = identifier ":" type ;
type_decl    = ("type" | "struct") identifier "{" { field [ "," ] } "}" ;
field        = identifier ":" type ;
type         = [ identifier "::" ] identifier
               [ "<" type { "," type } ">" ]
             | "[" [ integer { "," integer } ] "]" ;
block        = "{" { statement [ ";" ] } "}" ;
statement    = ("let" | "var" | "const") identifier [ ":" type ]
               "=" expression
             | "return" [ expression ]
             | expression ;
expression   = prefix { postfix | infix expression } ;
prefix       = literal | identifier | "(" expression ")" | unary expression
             | match_expression ;
postfix      = "(" [ expression_list ] ")" | "." identifier
             | "?." identifier | "[" expression "]" | "?"
             | "::" identifier ;
match_expression = "match" expression "{"
                   result_arm "," result_arm [ "," ] "}" ;
result_arm   = ("Ok" | "Err") "(" ( identifier | "_" ) ")" "=>" expression ;
infix        = "|>" | "??" | "||" | "&&" | comparison
             | "+" | "-" | "*" | "/" | "%" | "**" ;
```

## Multi-file projects v0

Projects can be compiled with `compile_project(entry_path)` and run with
`run_project(entry_path, function_name, *arguments)`. Imports use a quoted
relative `.hl` path, resolved from the file containing the import:

```aether
import "math/vector.hl" as vector;

fn main(value: i32) -> i32 {
    return vector::double(value);
}
```

The entry file's containing directory is the project root. Import paths may use
`.` and `..` as long as their resolved targets remain inside that root; absolute
paths, paths that escape the root, non-`.hl` paths, and paths containing control
characters are rejected. Re-importing one file through different aliases is
allowed and loads the module only once.

Imported functions and custom types must be referenced through their alias
using `alias::name`; unqualified imports, wildcard imports, and re-exports are
not supported. All declarations in an imported file are currently visible to
its importer through that alias. An alias may have the same name as a local
function because qualified and unqualified names are distinct. Duplicate aliases
within one module, import cycles, missing files, and unknown aliases or symbols
produce diagnostics with the originating file and source location.
`compile_source` remains the single-file API and reports an error if imports or
qualified names are used there.

Project compilation combines modules for semantic analysis, the reference
interpreter, and the supported C99 subset. Imported native `extern "C"`
declarations are rejected; native declarations in the entry module can still
use the explicit `@link` mapping described below. Workspace/LSP import
resolution is not part of this version.

Infix precedence from lowest to highest is `|>`, `??`, `||`, `&&`,
comparisons, `+ -`, `* / %`, and `**`. `**` is right-associative; the other
operators are left-associative. Prefix `!`, `-`, and `+` bind more tightly
than infix operators. Calls, member access, optional member access, and index
access bind most tightly.
Although `??` and `?.` are tokenized and parsed, they are rejected by semantic
analysis: an option type and its propagation/coalescing rules are not defined.
Postfix `?` is currently reserved for `Result` propagation only.

The prototype accepts comma-separated type arguments and bracketed dimensions,
so types such as `Tensor<f32, [4, 4], V>` are represented. Tensor dimensions
must be positive integer constants. The semantic analyzer infers types for the
supported expression subset, checks function arguments and return values,
tracks tensor shapes and unit dimensions through arithmetic, and rejects
incompatible units. Units carry rational scale and offset metadata; degree
conversions carry a separate symbolic power of pi.
Addition, subtraction, and comparisons require the exact same unit; there is no
implicit conversion. Use `convert(value, "target_unit")` to request an explicit
conversion. The semantic result records its exact rational multiplier and
addend (and symbolic pi power where needed) for the typed IR and reference
interpreter.
Absolute-temperature arithmetic is rejected until temperature-delta semantics
are specified. Tensor element-wise operations require matching element types
and shapes; scalar broadcasting is supported for addition/subtraction when the
unit and element type match. The reference interpreter uses truncation toward zero
for integer division and integer remainder; division uses floating-point
semantics when either operand is floating point.
Decimal literals default to `f32`; integer literals default to `i32`. Numeric
suffixes such as `100_mm` are not yet parsed as typed quantities.

## Effectful built-in operations v0.1

The initial effectful API uses unqualified, reserved built-in names. Calls are
synchronous and return `Result<T, E>`; they do not throw or silently substitute
an empty value. Error payloads are currently opaque. `Result` supports only the
two type arguments shown here. The supported syntax includes the restricted
`match` expression and `?` propagation described below.

| Operation | Signature | Required effect | Contract |
|---|---|---|---|
| `io_write(text)` | `str -> Result<void, IoError>` | `@io` | Writes the exact UTF-8 text to standard output; adds no newline. |
| `io_read_line()` | `() -> Result<str, IoError>` | `@io` | Reads one UTF-8 line from standard input without its line terminator; EOF is an error. |
| `fs_read_text(path)` | `str -> Result<str, FsError>` | `@fs` | Reads a UTF-8 text file. |
| `fs_write_text(path, text)` | `(str, str) -> Result<void, FsError>` | `@fs` | Creates or replaces a UTF-8 text file; does not create missing parent directories. |
| `fs_exists(path)` | `str -> Result<bool, FsError>` | `@fs` | Returns `Ok(false)` only when the path does not exist; other failures are errors. |
| `net_get(url)` | `str -> Result<str, NetError>` | `@net` | Performs HTTP GET and returns the UTF-8 response body. |
| `net_post(url, body)` | `(str, str) -> Result<str, NetError>` | `@net` | Performs HTTP POST with a UTF-8 text body and returns the UTF-8 response body. |

Filesystem paths are interpreted relative to the project root; absolute paths
and paths that escape the root are errors. Network operations accept only
`http`/`https` URLs, validate TLS certificates, and do not follow redirects
implicitly. These are the intended host-operation contracts; the reference
interpreter requires injected handlers and does not implement these operations.
The initial signatures intentionally omit custom headers,
timeouts, binary data, and asynchronous operations.

Effect attributes declare an upper bound on a function's effects. The analyzer
infers effects from built-in operations and propagates them transitively across
user-function calls and extern declarations; an inferred effect not declared
on a function is an `E_EFFECT_VIOLATION`. An unused effect annotation does not
itself force callers to declare that effect. Built-in operation names are
reserved for these signatures.

## Foreign function interface v1

The compiler accepts explicit declarations for scalar functions exposed
through the C ABI:

```aether
@ffi
@link("sensor")
extern "C" fn read_channel(channel: u32) -> i32;

@ffi
fn sample(channel: u32) -> i32 {
    return read_channel(channel);
}
```

An extern declaration must have one `@link("library")`, use `extern "C"`,
declare `@ffi`, and specify a return type (`void` for no return value).
Function parameters and return values are restricted to `i8/i16/i32/i64`,
`u8/u16/u32/u64`, `f32`, `f64`, and `void` as a return type. In particular,
strings, tensors, `Result`, `usize`, pointers, aggregates, variadic arguments,
callbacks, and non-C ABIs are not supported. The library name is a simple
linker name, not a path.

The `@ffi` effect is required by each extern declaration and propagates to
callers, which must also declare `@ffi`. It may also be used on an ordinary
function as an effect annotation. An extern function's Aether name is also its
C linker symbol; the backend rejects names that are not portable C
identifiers. The IR retains the symbol, ABI, library, signature, and effect
metadata.

The C99 backend emits matching C prototypes and direct calls for supported
extern declarations. `build_executable` requires an explicit mapping from each
`@link` name to a linker-compatible object or library file, for example
`libraries={"sensor": "build/sensor.o"}`. No library is searched for or
downloaded implicitly; paths are passed directly to the compiler driver
without a shell. Linking errors are reported and do not replace an existing
output executable. This stage does not load libraries dynamically and does
not define native ownership or error conventions.

The reference interpreter remains independent of native code: tests may
provide an explicit `external_handlers` mapping keyed by Aether symbol name.
Arguments and scalar results are checked against the declared types; a
missing handler raises `AetherRuntimeError`. C++, Rust, and other-language
libraries must expose a compatible C ABI through their own wrapper. Python
embedding is a separate, optional runtime integration and is not part of this
FFI.

## Safe-C99 backend v0

`emit_c99(source)` in `src.api` emits deterministic C99 for a deliberately
small scalar subset. `emit_c99_project(entry_path)` does the same for a
multi-file project after resolving imports. Both accept functions with explicit
scalar return types, scalar parameters and locals, literals, user-function
calls, pipelines to user functions, scalar C ABI declarations and calls,
boolean logic, comparisons, and basic arithmetic. Generated
function and local identifiers use a stable UTF-8-hex encoding with the
`aether_fn_` and `aether_v_` prefixes. This avoids C keyword collisions; the
output is a translation unit, not an executable and has no generated `main`.
The generated translation unit is checked by the test suite with
`-std=c99 -pedantic -Wall -Wextra -Werror`; compiler-backed tests also execute
harnesses and compare supported results with `run_source`.

Supported types are `bool`, `i8/i16/i32/i64`, `u8/u16/u32/u64`, `f32`, `f64`,
and `void` as a function return type. Fixed-width C types must be available on
the target implementation. Integer addition, subtraction, multiplication,
division, remainder, and signed negation are guarded against overflow and
invalid divisors; a failed check calls `abort()`. Unsigned arithmetic is also
checked rather than wrapping. Integer division truncates toward zero and
remainder follows the corresponding C quotient. Floating division by zero is
guarded; other floating-point exceptional behavior follows the target C
implementation.

The backend reports `CompilationError` diagnostics for unsupported constructs
instead of silently emitting a substitute. Tensors, units, structs, strings,
`Result`, exponentiation, built-in operations, effects other than `@ffi`, and
non-C extern declarations are not supported. The emitter itself does not
invoke a C compiler or linker or dynamically load libraries.
Integer boundary checks and invalid-operation traps are exercised by compiled
harnesses. The interpreter uses Python integers and therefore does not trap on
fixed-width overflow; compare only in-range reference executions. Locally,
compiler-backed tests skip if no `cc`, `gcc`, `clang`, or `zig` (with its `cc`
subcommand) is installed. CI sets
`AETHER_REQUIRE_C99_COMPILER=1`, making a missing compiler a test failure
instead of a skip.

## Host C99 object and executable builds

`build_object(source, output_path, compiler=None)` in `src.api` runs the host
C compiler on the generated translation unit and writes an object file. The
default compiler search order is `cc`, `gcc`, `clang`, then `zig cc`; callers
may select a compiler explicitly, including a command sequence such as
`("zig", "cc")`. Compiler arguments are passed without a shell. The driver
uses strict C99 warnings-as-errors and compile-only mode, captures diagnostics,
and replaces the requested output atomically only after successful compilation.
It does not select a cross-compilation target or resolve FFI symbols; unresolved
symbols may remain in the object for a later link step.

`build_executable(source, output_path, compiler=None, libraries=None)` compiles
and links a host executable. Executable source must define `fn main() -> i32`
or `fn main() -> void` with no parameters; the function may declare `@ffi` when
it calls native functions. The backend adds the standard C `int main(void)`
wrapper; an Aether `i32` result becomes the process exit code, while a `void`
result becomes exit code zero. Each `@link` name must have exactly one explicit
entry in `libraries`, mapping it to an existing linker-compatible file. No
libraries or linker options are inferred or fetched. The output is replaced
only after successful linking, and the API does not execute the program. The
host compiler and its target ABI define the executable format.

`build_project_object(entry_path, output_path, compiler=None)` and
`build_project_executable(entry_path, output_path, compiler=None,
libraries=None)` provide the corresponding object and executable build paths
for projects. The root module must define `main` for an executable. Imported
function names are encoded deterministically into internal C identifiers;
calls across module boundaries compile into the same translation unit.
Imported C ABI declarations are currently rejected rather than silently
omitted. In the entry module, FFI libraries still require the exact explicit
mapping for every declared `@link`.

## Result handling v0.2

`Result<T, E>` has exactly two variants: `Ok(T)` and `Err(E)`. A `match`
expression must include exactly one arm for each variant:

```aether
@io
fn read_or_default() -> str {
    let result = io_read_line();
    return match result {
        Ok(text) => text,
        Err(error) => "input unavailable",
    };
}
```

Arms bind one payload name or `_` to ignore the payload. This is an expression:
both arms must infer compatible result types. Nested block arms, arbitrary
patterns, guards, and matching custom enums are not part of this subset.

The postfix `?` operator accepts only `Result<T, E>` and is allowed only in a
function declared to return `Result<U, E>`. It extracts `T` on success and
propagates the same `E` as an error; mismatched error types and non-`Result`
contexts are compile-time diagnostics. For example:

```aether
@io
fn read_required() -> Result<str, IoError> {
    return io_read_line()?;
}
```

The reference interpreter represents these values with `ResultValue`, executes
matches, and propagates errors through `?`. An effectful built-in is inert
unless the embedding program supplies a handler for that exact operation; the
handler must return `ResultValue`. There is no default filesystem, network, or
console access. A discarded `Result` or a local not definitely handled,
propagated, transferred to another binding, or returned is diagnosed as
`E_UNHANDLED_RESULT`; reuse after consumption is diagnosed as
`E_RESULT_REUSE`. This is a bounded expression-flow check, not a general
ownership or control-flow analysis.

## Semantic checks in v0.2

The analyzer checks duplicate top-level declarations, duplicate
fields/parameters/local bindings, unknown named types, references to unknown
local names and calls to unknown unqualified functions. Functions with no
explicit effect attribute are pure. Combining `@pure` with an effect attribute
is diagnosed. Tensor types
support static shapes and the units listed in the frontend; arithmetic checks
unit dimensions and propagates units and scales through scalar/tensor
multiplication and division. Invalid built-in arity/types, unknown units,
tensor element types, and unsupported unit exponentiation are diagnosed.
`mean`, `max_abs`, and `sum` currently accept tensors; `mean` returns the input
floating element type or `f32` for integer tensors. `len` accepts a tensor and
returns its leading dimension.
This is an initial type/effect subset, not a complete inference,
memory-safety, or contract solver. The typed IR retains expression types,
source spans, effects, and explicit unit-conversion metadata. The reference
interpreter executes the supported scalar/tensor operations, user-function
calls, `Result`, `match`, `?`, and rational/affine unit conversions. Unit
conversion to radians evaluates the symbolic pi factor using Python's
floating-point `pi`.

The interpreter is a development reference, not a production runtime or
security boundary. It does not implement native I/O, files, networking, CUDA,
memory ownership, contracts, or target lowering. Embedders may explicitly
inject effect handlers; the interpreter never performs those external effects
by itself. Runtime checks are limited to the documented subset and do not
replace a complete compiler/runtime type system.

Built-in calls currently recognized are `mean`, `max_abs`, `subtract`,
`scale`, `abs`, `min`, `max`, `sum`, `len`, `convert`, and the effectful
operations above. Known attributes are `io`, `net`, `fs`, `ffi`, `link`,
`persist`, `contract`, `C`, and `target`. This list is provisional and does not imply that contracts,
persistence, or target lowering are implemented.

`parse_source` returns a partial AST and diagnostics when the supported
grammar encounters an error. Tokenization errors (for example an unterminated
string) remain fatal `ParseError`s, consistent with the lossless CST API.
