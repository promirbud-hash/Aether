# Unit prefixes and unit expressions

**Status:** implemented in the Python prototype for the documented numeric and
tensor type subset. This is not a complete physical-units system or a
production compiler guarantee.

## Motivation

The compiler resolves SI prefixes algorithmically while retaining its finite
registry of supported unit definitions. For frequency, the exact conversion
relation is:

```text
1 MHz = 1000 kHz = 1,000,000 Hz
```

## Prefix resolution

- Unit and prefix symbols are case-sensitive (`m` is milli; `M` is mega).
- An exact registered unit symbol is resolved before attempting prefix
  decomposition. Otherwise, try recognized prefixes in longest-symbol-first
  order and resolve the remaining suffix as a supported linear unit.
- The prefix scale multiplies the underlying unit scale and does not change
  its dimensions.
- Initially support the official decimal SI prefixes from `q` (10^-30) through
  `Q` (10^30): `q, r, y, z, a, f, p, n, u, m, c, d, da, h, k, M, G, T, P, E,
  Z, Y, R, Q`.
- Preserve ASCII `u` as the canonical micro spelling because it is already
  used by `um` and `uV`; accept Unicode `µ` as an alias and normalize it to
  `u`.
- Prefixing is allowed for every currently registered linear unit, including
  already-prefixed linear spellings and derived units. This is a deliberate
  prototype boundary, not a claim that every resulting spelling is
  conventional SI notation.
- Keep `kg` as the SI base unit for mass; apply generated mass submultiples to
  `g` (for example `mg`) rather than prefixing `kg` again.
- Do not apply prefixes to affine absolute-temperature units. Angle units are
  also excluded from the initial unitful-autodiff scope.

The finite unit table remains the allowlist of supported physical dimensions.
Prefix decomposition cannot introduce a unit whose suffix is absent from that
table.

## Unit expressions required for autodiff

Scalar numeric and tensor type annotations accept unit expressions. A linear
unitful tensor input and scalar output can therefore express a derivative
unit:

```text
unit(output) / unit(input)
```

Differentiating a voltage-valued scalar objective with respect to
millimeter-valued tensor elements produces a derivative with dimension
voltage per length. The scale preserves the units of the numeric values used
by the function; dimensions alone are insufficient.

The implemented syntax allows a unit expression in numeric types and as the
third tensor type argument:

```aether
fn voltage_slope(value: Tensor<f32, [4], mm>) -> f32<V/mm> { ... }
fn frequency(value: f64<1/ms>) -> f64<kHz> { ... }
```

The parser uses this grammar:

```ebnf
unit_expr   = unit_product { "/" unit_product } ;
unit_product = unit_factor { "*" unit_factor } ;
unit_factor = unit_atom [ "^" signed_integer ] ;
unit_atom   = unit_symbol | "(" unit_expr ")" ;
```

Multiplication binds more tightly than division; repeated `/` is
left-associative; exponentiation accepts integer exponents. Parentheses can
group expressions. Semantic resolution normalizes dimensions, exact rational
scale, affine metadata, and the symbolic power of pi used by existing angle
conversions. `convert(value, "unit-expression")` accepts the same expression
syntax. The rendered type expression removes whitespace and normalizes `µ` to
`u`; equivalent expressions may retain different spellings.

Both scalar and tensor annotations can carry expressions. In particular,
signatures can declare `U_out / U_in` for a future unitful gradient. This
syntax does not itself implement unitful autodiff.

## Compatibility and validation gates

1. Keep all currently recognized unit spellings and conversion behavior.
2. Test direct prefix composition and exact scale/dimension equivalence, for
   example `MHz`, `kHz`, and `Hz`.
3. Test ambiguous spellings with exact-symbol-first and longest-prefix
   resolution, including `m`, `mm`, `ms`, and `kPa`.
4. Reject unsupported prefix/unit combinations with an explicit diagnostic.
5. Test compound-unit equality and conversion before using derived unit types
   as gradient results.
6. Preserve exact rational scales; do not round prefix factors during
   semantic analysis.
7. Keep affine temperatures and unit prefixes on them out of the initial
   linear-unit gradient contract.

## Selected requirements

- Generate prefixes for all currently recognized linear units, including
  linear derived units, while retaining the explicit exclusions for `kg`
  prefix composition, affine temperatures, and angle units.
- Preserve `u` and accept `µ` as an alias; canonical display uses `u`.
- Permit compound unit expressions in source type annotations, not only in
  inferred types.
- Use `1 MHz = 1000 kHz = 1,000,000 Hz`.

## Remaining design details

- Decide whether to narrow the current all-linear-symbol prefix policy to
  conventional SI combinations before stabilizing the language specification.
- Define canonical mathematical display for equivalent expressions; current
  rendering is stable and parseable but preserves the expression's grouping
  and symbol choices.
- Expand the tests if the unit registry gains new base or derived units.

This implementation is a prerequisite for unitful tensor autodiff, not its
implementation. The unit table remains finite; affine-temperature composition
and angle differentiation remain outside the selected initial autodiff scope.
