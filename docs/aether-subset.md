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

User-defined type names cannot shadow built-in type names. Although the
frontend recognizes `char`, `Slice`, `Pool`, and `Id`, their runtime
representations are not defined in this subset, so semantic analysis rejects
them with `E_UNSUPPORTED_TYPE`.

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
             | identifier "=" expression
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

`let` and `const` bindings are immutable. `var` bindings may be rebound with
`identifier = expression`; rebinding a parameter or immutable binding is a
semantic error. Assignment changes only the binding and never mutates a tensor,
aggregate, or another alias. Assignment currently rejects `Result` values.

The prototype accepts comma-separated type arguments and bracketed dimensions,
so types such as `Tensor<f32, [4, 4], V>` are represented. A tensor must have at
least one dimension; every dimension is a positive integer constant. Tensor
elements must use a supported numeric type. The semantic analyzer tracks
element types, exact static shapes, and unit dimensions through the supported
operations. Units carry rational scale and offset metadata; degree conversions
carry a separate symbolic power of pi.

Numeric types and the third tensor type argument accept compound unit
expressions, for example `f32<V/mm>` and `Tensor<f32, [4], V/mm>`. Expressions
support unit symbols, `1` for dimensionless units, parentheses, `*`, `/`, and
integer powers. Multiplication binds more tightly than division; repeated
division is left-associative. `convert(value, "V/mm")` accepts the same unit
expression syntax. Resolution keeps exact rational scales and dimensions.

SI prefixes are case-sensitive and include the decimal prefixes from `q`
through `Q`; exact unit symbols take precedence, followed by longest-prefix
resolution. Prefixes apply to all currently registered linear units, including
linear derived and already-prefixed units. `µ` is an alias for `u`, which is
the normalized spelling. `kg` is not prefixable; `g` is. Absolute-temperature
units and angle units are not prefixable, and absolute temperatures cannot be
composed or exponentiated. The finite unit registry still limits which physical
dimensions are recognized; tensor autodiff accepts supported input/output
unit ratios, including affine-temperature and angular conversions, as
specified in [the tensor-gradient contract](./autodiff-tensor-contract.md).

The current tensor operation contract is:

| Operation | Supported tensor operands | Result |
|---|---|---|
| `+`, `-` | Two tensors with identical element types, shapes, and exact units; or one tensor and one scalar of the same element type and exact unit, in either order | Element-wise tensor; scalar broadcasting does not change shape |
| `*` | Tensor/scalar or scalar/tensor with matching element type; tensor/tensor is not supported | Element-wise tensor; units multiply |
| `/` | Tensor/scalar with matching element type; scalar/tensor and tensor/tensor are not supported | Element-wise tensor; units divide |
| `abs` | One numeric tensor | Same tensor shape, element type, and unit |
| `min`, `max` | Two tensors with identical element types, shapes, and exact units | Element-wise tensor |
| `mean`, `sum`, `max_abs` | One non-empty tensor | Reduction over all elements; `mean` returns the floating element type or `f32` for integer tensors; `sum` and `max_abs` retain the element type |
| `len` | One tensor | `i64` containing the leading dimension |
| Index `tensor[i]` | One integer index into the leading dimension | Removes the leading dimension; indexing the final dimension returns one scalar element |

Tensor addition, subtraction, comparisons, `min`, and `max` require exact
matching units; there is no implicit unit conversion. Scalar addition and
subtraction also require an exact unit match. Multiplication and division
combine unit dimensions and scales. Use `convert(value, "target_unit")` for
explicit conversion; it preserves tensor shape and applies to each element.
Scalar-broadcasting differential tests cover representative `f32` and `f64`
tensors of ranks one through three. Tensor operands with distinct shapes are
rejected even when their dimensions could be expanded under broader
broadcasting conventions; tensor/tensor multiplication and division and
scalar/tensor division are also rejected.
Tensor comparisons are rejected until boolean tensor types and their
operations are defined, rather than producing a scalar type for an
element-wise runtime result. The semantic result records exact rational
conversion multipliers and addends (and symbolic pi powers where needed) for
the typed IR and reference interpreter.
The C99 backend forms the complete conversion multiplier using the same
binary64 pi power as the interpreter and emits one hexadecimal floating-point
literal, avoiding intermediate pi-rounding changes.

Index bounds are checked at runtime; negative indices are rejected rather than
using Python's wraparound indexing. Rectangularity, dimensions, and numeric
element values are checked at function boundaries. Reductions operate over all
elements and reject empty runtime data. The reference interpreter uses
truncation toward zero for integer division and integer remainder; division
uses floating-point semantics when either operand is floating point.
Absolute-temperature arithmetic is rejected until temperature-delta semantics
are specified.

Tensor backend conformance tests compare the reference interpreter with strict
C99 builds for representative `f32` reductions and linear, affine-temperature,
and angular unit conversions. They also compare `f32` tensor arithmetic with
inputs rounded to the declared element type. `f32` inputs, value-producing
arithmetic operations, conversions, and results are rounded to single
precision. `f32` reductions accumulate in element order with single-precision
rounding after each addition. The reverse-mode forward tape and local
derivatives use the same `f32` rounding; adjoints accumulate in double
precision, as in the C99 backend. The interpreter stores numbers as Python
floats but explicitly rounds them at these `f32` boundaries.
`f64` reductions also accumulate sequentially in row-major element order,
rounding each addition to `f64`; `mean` divides the completed sum once by the
element count. The interpreter uses an explicit loop rather than Python's
version-dependent compensated floating-point `sum`, keeping its order and
results aligned with generated C99.

Conformance tests include loss of precision, overflow to infinity, subnormal
underflow, NaN, infinities, and signed zero for `max_abs`. Tensor division by
zero is an interpreter runtime error and a native C trap. Exact agreement is
tested for selected `f32` cases and is not promised for all floating-point
inputs or exceptional-value behavior across every C implementation.
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
output executable. This stage does not load libraries dynamically. Supported scalar FFI values use C
by-value representations and are not ARC-managed. Ownership-bearing FFI
values remain unsupported; any future value crossing that boundary must use
an explicit transfer contract rather than automatic ARC.

The reference interpreter remains independent of native code: tests may
provide an explicit `external_handlers` mapping keyed by Aether symbol name.
Arguments and scalar results are checked against the declared types; a
missing handler raises `AetherRuntimeError`. C++, Rust, and other-language
libraries must expose a compatible C ABI through their own wrapper. Python
embedding is a separate, optional runtime integration and is not part of this
FFI.

## Safe-C99 backend v0

`emit_c99(source)` in `src.api` emits deterministic C99 for a deliberately
small scalar and static-tensor subset. `emit_c99_project(entry_path)` does the
same for a multi-file project after resolving imports. Both accept functions
with explicit scalar or tensor return types, scalar or tensor parameters and
locals, literals, user-function calls, pipelines to user functions, scalar C
ABI declarations and calls, boolean logic, comparisons, and basic arithmetic.
Generated
function and local identifiers use a stable UTF-8-hex encoding with the
`aether_fn_` and `aether_v_` prefixes. This avoids C keyword collisions; the
output is a translation unit, not an executable and has no generated `main`.
The generated translation unit is checked by the test suite with
`-std=c99 -pedantic -Wall -Wextra -Werror`; compiler-backed tests also execute
harnesses and compare supported results with `run_source`.

Supported scalar types are `bool`, `i8/i16/i32/i64`, `u8/u16/u32/u64`, `f32`,
`f64`, and `void` as a function return type. Native tensors are fixed-size
row-major C structs and currently support only `f32` and `f64` elements.
The backend supports element-wise tensor `+` and `-` with tensor/tensor or
tensor/scalar operands, `*` with tensor/scalar operands, tensor/scalar `/`,
`abs`, `min`, `max`, `sum`, `mean`, `max_abs`, leading-dimension `len`, and
leading-axis indexing. Explicit `convert` is supported for unitful floating
scalars and tensors, applying the typed IR's rational scale/addend and symbolic
power of pi. Tensor division by zero and negative/out-of-range indexes trap.
Integer tensors are rejected until checked integer tensor arithmetic is
specified. Fixed-width C types must be available on the target implementation.
Integer addition, subtraction, multiplication,
division, remainder, and signed negation are guarded against overflow and
invalid divisors; a failed check calls `abort()`. Unsigned arithmetic is also
checked rather than wrapping. Integer division truncates toward zero and
remainder follows the corresponding C quotient. Floating division by zero is
guarded; other floating-point exceptional behavior follows the target C
implementation.

The backend reports `CompilationError` diagnostics for unsupported constructs
instead of silently emitting a substitute. Structs, strings, exponentiation,
unsupported built-in forms, effects other than `@ffi`, and
non-C extern declarations are not supported. A limited inline scalar `Result`
subset is described below. The emitter itself does not
invoke a C compiler or linker or dynamically load libraries.
Integer boundary checks and invalid-operation traps are exercised by compiled
harnesses. The interpreter uses Python integers and therefore does not trap on
fixed-width overflow; compare only in-range reference executions. Locally,
compiler-backed tests skip if no `cc`, `gcc`, `clang`, or `zig` (with its `cc`
subcommand) is installed. CI sets
`AETHER_REQUIRE_C99_COMPILER=1`, making a missing compiler a test failure
instead of a skip.

### Reverse-mode autodiff v0

The compiler-recognized `gradient(function_name, input)` intrinsic computes
the reverse-mode derivative of a named function with respect to its one
scalar or static-tensor input. The scalar overload retains the original
unitless `f32`/`f64` contract. The tensor overload requires a unary, pure
objective with matching `f32`/`f64` tensor input shape and a scalar result of
the same element type. The objective result type must be explicitly annotated.
The gradient result has the input shape and element type, with unit
`unit(result) / unit(input)` and its exact rational scale.

The target body may contain local bindings followed by one final return.
Supported expressions are numeric literals, names, unary `+`/`-`, scalar and
tensor `+`, `-`, `*`, `/` within the existing tensor-operation contract,
`sum(tensor)`, `mean(tensor)`, explicit `convert(value, "unit")`, and the
linear `scale`/`subtract` built-ins, including pipeline forms. Unit conversions
may be affine-temperature or angular; the conversion offset has zero derivative
and the multiplier includes any symbolic pi factor. Broadcasting follows the
tensor arithmetic rules; tensor/tensor multiplication and division are
unsupported. Derivatives accumulate in reverse order and combine contributions
to shared values.

The tensor overload rejects effects, control flow, indexing, nested
user-function calls, implicit conversions, and operations without a derivative
convention (`abs`, `min`, `max`, and `max_abs`). Division by zero retains the
normal runtime failure behavior (an interpreter runtime error or a native C
trap). Generated C99 allocates an invocation-local tape on the heap, checks
size overflow and allocation failure, copies the gradient to the returned
tensor value, and frees the tape. Allocation failure traps; no general ownership or
memory-safety guarantee follows. Floating-point overflow, NaN propagation,
and cross-toolchain bitwise equality are not guaranteed by this prototype.
For `f32` autodiff, forward values and local derivative contributions are
rounded at operation boundaries in the interpreter and C99 backend; reverse
adjoints accumulate in double precision. Differential tests cover cancellation
of many large positive and negative adjoint contributions in different branch
orders while preserving a small residual, and exact cancellation. They also
confirm final `f32` gradient conversion uses ties-to-even at an exact halfway
boundary (including both parity choices and negative gradients), and rounds
upward just above it, for scalar gradients and tensor gradients through `sum`,
under GCC and Clang. This selected case does not promise bitwise reproducibility
for all operations or targets. Tests also
check subnormal ties: half the least positive subnormal rounds to signed zero,
and 1.5 subnormal units round to the second subnormal for scalar and
`mean`-based tensor gradients under GCC and Clang. At 2.5 subnormal units,
the exact tie rounds to the second subnormal, with immediately lower/higher
values rounding to the second/third; negative cases are symmetric. Around half
a subnormal, tests also cover values immediately below and above the tie,
including signed-zero behavior. They also
cover scalar and tensor division gradients with positive, negative, and small
nonzero denominators; a shared branch near the `f32` integer precision
boundary; subnormal scalar and tensor gradients; gradients overflowing to
positive infinity; a shared scaled value used by `sum` and `mean`; and
explicit linear unit conversion. These selected cases are checked against
analytic results with GCC and Clang.

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

`build_project_object(entry_path, output_path, compiler=None)` emits the
resolved project as one C99 translation unit and compiles it to an object.
`build_project_objects(entry_path, output_directory, compiler=None)` instead
compiles one object per source module and returns a mapping from project-relative
`.hl` paths to object paths in the output directory. Output names are stable
hashes of source paths, avoiding collisions for equal basenames. The output
directory must not already exist; all objects are built in a staging directory
and published together only after every module compiles successfully.
The directory includes `aether-objects.json`, which records the format version,
entry source, source hashes, generated-C hashes, object names, object hashes,
and C toolchain metadata. Version 2 requires normalized project-relative source
paths, unique source and object entries, single-file object names, lowercase
64-character SHA-256 digests, and a compiler fingerprint. The fingerprint
records the resolved compiler executable, a hash of the full compiler command,
the compiler's `--version` output, its `-dumpmachine` target, and the C99
compile and link flags. Compilers that do not provide both identity probes are
rejected for object-set builds. Version 1 manifests are not compatible with
version 2. Duplicate JSON keys and malformed or unsupported manifest content
are rejected. This fingerprint helps prevent accidental toolchain mixing; it
does not prove ABI compatibility and is not a signed security boundary.
Source hashes normalize CRLF and CR line endings to LF so Git checkout
conventions do not invalidate otherwise identical project objects; parsing
still uses the original text so CST and LSP offsets remain relative to the
exact source received. C99 compilation disables FP contraction and uses a
fixed random seed, and translation units are passed to the compiler by their
stable basename to avoid temporary-directory paths in object metadata.
`link_project_objects(entry_path, object_paths, output_path, compiler=None,
libraries=None)` links that object mapping after checking that it contains
exactly one object for each current project source, that the current toolchain
matches the recorded fingerprint, and that sources, generated C, and objects
match the manifest. FFI libraries continue to require an exact explicit `@link`
mapping. A failed link preserves any existing output executable. Missing or
unreadable manifests, missing object files, mismatched toolchains, and invalid
paths or hashes are reported as `C99LinkConfigurationError` before linking.
`build_project_executable(entry_path, output_path, compiler=None,
libraries=None)` instead emits and compiles one C99 translation unit per
source file, then links the resulting objects and explicitly mapped FFI
libraries. Project function declarations are available as prototypes in each
unit, and their encoded C names are deterministic. Only the entry module
defines the generated process `main`; the Aether entry function must be
`fn main() -> i32` or `fn main() -> void` with no parameters. Failed compilation
or linking does not replace an existing executable. Imported C ABI declarations
are currently rejected rather than silently omitted. In the entry module, FFI
libraries still require the exact explicit mapping for every declared `@link`.

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

The pure scalar subset additionally permits `Ok(value)` and `Err(ErrorTag)` in
functions with an explicit `Result<T, E>` return annotation. `ErrorTag` is one
of `IoError`, `FsError`, or `NetError`; it is a category marker, not a
user-defined error payload. The C99 backend supports `Result<T, E>` only when
`T` is a supported unitless scalar (`bool`, fixed-width integer, `f32`, or
`f64`) or the bounded `FrameBuffer` resource, and only for internal Aether
functions. It lowers the value to an inline tagged C struct; `?` branches to a
common function cleanup epilogue, preserving the error tag or wrapping the
successful value in the enclosing `Result`. Exhaustive `Ok`/`Err` matches are
lowered for supported payload and result types. No effects, FFI crossing,
strings, or tensor payloads are included in this increment.

The accepted ownership baseline specifies value semantics, `var` rebinding,
frame/tape lifetimes, acyclic ARC, optional COW, fail-fast implicit allocation,
and explicit transfer for future ownership-bearing FFI. General ARC, heap
tensor storage, and ownership-bearing FFI remain unimplemented. This prototype
now has one deliberately narrow frame resource, documented below; it does not
provide a general ownership guarantee. The tested FFI surface remains
scalar-only. See [the ownership baseline](./memory-ownership-design.md).

## Frame-local byte buffers and ownership v0.2

The internal `frame_buffer(size)` operation creates immutable byte storage
with shared backing. Each owning local binding owns exactly one reference;
`let`/`var` copies retain the backing. `frame_buffer_len(buffer)` is the only
operation that reads its metadata. Internal Aether functions may accept a
borrowed `FrameBuffer` parameter and return an owned one:

```aether
fn measure(size: u64) -> u64 {
    let bytes = frame_buffer(size);
    return frame_buffer_len(bytes);
}

fn identity(bytes: FrameBuffer) -> FrameBuffer {
    return bytes;
}
```

Parameters are borrowed: calls do not retain or release the caller's reference.
Returning a parameter or copying it to a local retains a reference. Returning a
local or `Ok(buffer)` establishes an owned reference for the caller before
frame cleanup. Function results and temporary arguments own their returned
reference and release it at the end of the full expression unless transferred
to a local or caller. Rebinding `var` evaluates and retains the replacement
before releasing the previous value.

The buffer cannot be stored in a user aggregate or cross FFI. Its size must be
positive and fit the target's `size_t`; zero, size/refcount overflow, or
allocation failure triggers a fail-fast trap in generated C99. The reference
interpreter models the same reference transitions and reports allocation-limit
or invalid-state failures as `AetherRuntimeError`.

The typed IR records resource-owning local bindings in construction order.
Both normal returns and `?` propagation release local owners in reverse
order. The backing is freed exactly once when its count reaches zero. A nested
call cleans its own frame before returning to its caller. The generated
C99 defaults to `malloc`/`free`; tests can override the
`AETHER_FRAME_ALLOC`, `AETHER_FRAME_FREE`, and `AETHER_FRAME_SIZE_MAX` macros
to verify allocation, cleanup, and narrow-target size overflow deterministically.
Cleanup is not promised after a process-aborting trap. This bounded probe does
not implement general-purpose ARC, COW/mutation, ownership-bearing aggregates,
general destructors, or a public allocator API.

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
