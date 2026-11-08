"""Initial Aether type, tensor-shape, unit, and effect analysis."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re

from src.frontend import (
    AssignmentStmt,
    Diagnostic,
    Expr,
    ExprStmt,
    ForStmt,
    FunctionDecl,
    IfStmt,
    LetStmt,
    LoopControlStmt,
    Module,
    ReturnStmt,
    Statement,
    SourceSpan,
    TypeDecl,
    TypeExpr,
    UnitExpr,
    WhileStmt,
    _BUILTIN_CALLS,
    _BUILTIN_TYPES,
    parse_unit_expression_text,
)


_NUMERIC_TYPES = {
    "i8", "i16", "i32", "i64", "i128",
    "u8", "u16", "u32", "u64", "u128",
    "f16", "bf16", "f32", "f64", "fp8", "i4", "t2",
}
_INTEGER_TYPES = {
    "i8", "i16", "i32", "i64", "i128",
    "u8", "u16", "u32", "u64", "u128", "usize",
}
def _is_range_integer(type_: AetherType | None) -> bool:
    return type_ is not None and type_.name in _INTEGER_TYPES and not (
        type_.unit
        or type_.unit_symbol
        or type_.unit_scale != 1
        or type_.unit_offset
        or type_.unit_pi_power
    )


_FLOAT_TYPES = {"f16", "bf16", "f32", "f64", "fp8"}
_UNSUPPORTED_BUILTIN_TYPES = {"char", "Pool", "Id"}
_FFI_INTEGER_TYPES = {
    "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64",
}
_FFI_TYPES = _FFI_INTEGER_TYPES | {"f32", "f64"}
@dataclass(frozen=True)
class UnitDefinition:
    dimensions: tuple[tuple[str, int], ...]
    scale: Fraction = Fraction(1)
    offset: Fraction = Fraction(0)
    kind: str = ""
    pi_power: int = 0


_UNITS: dict[str, UnitDefinition] = {
    "mm": UnitDefinition((("length", 1),), Fraction(1, 1000)),
    "cm": UnitDefinition((("length", 1),), Fraction(1, 100)),
    "m": UnitDefinition((("length", 1),)),
    "km": UnitDefinition((("length", 1),), Fraction(1000)),
    "um": UnitDefinition((("length", 1),), Fraction(1, 1_000_000)),
    "nm": UnitDefinition((("length", 1),), Fraction(1, 1_000_000_000)),
    "V": UnitDefinition((("voltage", 1),)),
    "mV": UnitDefinition((("voltage", 1),), Fraction(1, 1000)),
    "uV": UnitDefinition((("voltage", 1),), Fraction(1, 1_000_000)),
    "A": UnitDefinition((("current", 1),)),
    "mA": UnitDefinition((("current", 1),), Fraction(1, 1000)),
    "s": UnitDefinition((("time", 1),)),
    "ms": UnitDefinition((("time", 1),), Fraction(1, 1000)),
    "us": UnitDefinition((("time", 1),), Fraction(1, 1_000_000)),
    "ns": UnitDefinition((("time", 1),), Fraction(1, 1_000_000_000)),
    "Hz": UnitDefinition((("time", -1),)),
    "kHz": UnitDefinition((("time", -1),), Fraction(1000)),
    "kg": UnitDefinition((("mass", 1),)),
    "g": UnitDefinition((("mass", 1),), Fraction(1, 1000)),
    "K": UnitDefinition((("temperature", 1),), kind="absolute_temperature"),
    "degC": UnitDefinition(
        (("temperature", 1),),
        offset=Fraction(5463, 20),
        kind="absolute_temperature",
    ),
    "degF": UnitDefinition(
        (("temperature", 1),),
        Fraction(5, 9),
        Fraction(45967, 180),
        "absolute_temperature",
    ),
    "N": UnitDefinition((("length", 1), ("mass", 1), ("time", -2))),
    "Pa": UnitDefinition((("length", -1), ("mass", 1), ("time", -2))),
    "kPa": UnitDefinition(
        (("length", -1), ("mass", 1), ("time", -2)),
        Fraction(1000),
    ),
    "J": UnitDefinition((("length", 2), ("mass", 1), ("time", -2))),
    "W": UnitDefinition((("length", 2), ("mass", 1), ("time", -3))),
    "rad": UnitDefinition((("angle", 1),)),
    "deg": UnitDefinition((("angle", 1),), Fraction(1, 180), pi_power=1),
}
_SI_PREFIXES: dict[str, int] = {
    "q": -30,
    "r": -27,
    "y": -24,
    "z": -21,
    "a": -18,
    "f": -15,
    "p": -12,
    "n": -9,
    "u": -6,
    "m": -3,
    "c": -2,
    "d": -1,
    "da": 1,
    "h": 2,
    "k": 3,
    "M": 6,
    "G": 9,
    "T": 12,
    "P": 15,
    "E": 18,
    "Z": 21,
    "Y": 24,
    "R": 27,
    "Q": 30,
}
_PREFIXABLE_UNITS = frozenset(
    symbol
    for symbol, definition in _UNITS.items()
    if symbol not in {"kg", "rad", "deg"}
    and definition.kind != "absolute_temperature"
)
_EFFECTS = {"io", "net", "fs", "ffi"}


@dataclass(frozen=True)
class AetherType:
    """Resolved type with optional static shape and canonical unit dimensions."""

    name: str
    arguments: tuple[AetherType, ...] = ()
    dimensions: tuple[int, ...] = ()
    unit: tuple[tuple[str, int], ...] = ()
    unit_scale: Fraction = Fraction(1)
    unit_offset: Fraction = Fraction(0)
    unit_symbol: str = ""
    unit_kind: str = ""
    unit_pi_power: int = 0

    @property
    def is_numeric(self) -> bool:
        return self.name in _NUMERIC_TYPES

    @property
    def element_type(self) -> AetherType | None:
        if self.name == "Tensor" and self.arguments:
            return self.arguments[0]
        return None

    def display(self) -> str:
        if self.name == "Tensor" and self.arguments:
            dimensions = ",".join(str(value) for value in self.dimensions)
            unit = self.unit_symbol or _unit_display(
                self.unit,
                self.unit_scale,
                self.unit_pi_power,
            )
            suffix = f",{unit}" if unit else ""
            return f"Tensor<{self.arguments[0].display()},[{dimensions}]{suffix}>"
        unit = self.unit_symbol or _unit_display(
            self.unit,
            self.unit_scale,
            self.unit_pi_power,
        )
        if unit and self.is_numeric:
            return f"{self.name}<{unit}>"
        if self.arguments:
            return f"{self.name}<" + ",".join(
                argument.display() for argument in self.arguments
            ) + ">"
        return self.name


@dataclass(frozen=True)
class EffectSummary:
    function: str
    declared: tuple[str, ...]
    required: tuple[str, ...]


@dataclass(frozen=True)
class UnitConversion:
    span: SourceSpan
    multiplier: Fraction
    addend: Fraction
    pi_power: int
    target_unit: str


@dataclass(frozen=True)
class SemanticResult:
    expression_types: tuple[tuple[SourceSpan, AetherType], ...]
    effects: tuple[EffectSummary, ...]
    conversions: tuple[UnitConversion, ...]
    diagnostics: tuple[Diagnostic, ...]


@dataclass(frozen=True)
class BuiltinOperation:
    parameters: tuple[AetherType, ...]
    result: AetherType
    effect: str


def _result_type(value: AetherType, error: str) -> AetherType:
    return AetherType("Result", (value, AetherType(error)))


_BUILTIN_OPERATIONS = {
    "io_write": BuiltinOperation(
        (AetherType("str"),),
        _result_type(AetherType("void"), "IoError"),
        "io",
    ),
    "io_read_line": BuiltinOperation(
        (),
        _result_type(AetherType("str"), "IoError"),
        "io",
    ),
    "fs_read_text": BuiltinOperation(
        (AetherType("str"),),
        _result_type(AetherType("str"), "FsError"),
        "fs",
    ),
    "fs_write_text": BuiltinOperation(
        (AetherType("str"), AetherType("str")),
        _result_type(AetherType("void"), "FsError"),
        "fs",
    ),
    "fs_exists": BuiltinOperation(
        (AetherType("str"),),
        _result_type(AetherType("bool"), "FsError"),
        "fs",
    ),
    "args_len": BuiltinOperation((), AetherType("i64"), ""),
    "args_get": BuiltinOperation(
        (AetherType("i64"),),
        _result_type(AetherType("str"), "ArgsError"),
        "",
    ),
    "net_get": BuiltinOperation(
        (AetherType("str"),),
        _result_type(AetherType("str"), "NetError"),
        "net",
    ),
    "net_post": BuiltinOperation(
        (AetherType("str"), AetherType("str")),
        _result_type(AetherType("str"), "NetError"),
        "net",
    ),
}


def _unit_display(
    unit: tuple[tuple[str, int], ...],
    scale: Fraction = Fraction(1),
    pi_power: int = 0,
) -> str:
    if not unit and scale == 1 and pi_power == 0:
        return ""
    preferred = {
        (("length", 1),): ("m", Fraction(1)),
        (("mass", 1),): ("kg", Fraction(1)),
        (("temperature", 1),): ("K", Fraction(1)),
        (("time", 1),): ("s", Fraction(1)),
        (("time", -1),): ("Hz", Fraction(1)),
        (("voltage", 1),): ("V", Fraction(1)),
        (("current", 1),): ("A", Fraction(1)),
        (("angle", 1),): ("rad", Fraction(1)),
    }
    if unit in preferred and scale == preferred[unit][1] and pi_power == 0:
        return preferred[unit][0]
    dimensions = "*".join(
        dimension if exponent == 1 else f"{dimension}^{exponent}"
        for dimension, exponent in unit
    ) or "1"
    factor = ""
    if scale != 1:
        factor = str(scale)
    if pi_power:
        pi_factor = "pi" if pi_power == 1 else f"pi^{pi_power}"
        factor = f"{factor}*{pi_factor}" if factor else pi_factor
    return dimensions if not factor else f"{dimensions}*{factor}"


def _combine_units(
    left: tuple[tuple[str, int], ...],
    right: tuple[tuple[str, int], ...],
    sign: int,
) -> tuple[tuple[str, int], ...]:
    result = dict(left)
    for dimension, exponent in right:
        result[dimension] = result.get(dimension, 0) + sign * exponent
    return tuple(sorted((dimension, exponent) for dimension, exponent in result.items() if exponent))


def _resolve_unit_symbol(symbol: str) -> UnitDefinition | None:
    normalized = symbol.replace("µ", "u")
    exact = _UNITS.get(normalized)
    if exact is not None:
        return exact
    for prefix in sorted(_SI_PREFIXES, key=len, reverse=True):
        if not normalized.startswith(prefix):
            continue
        base_name = normalized[len(prefix):]
        if base_name not in _PREFIXABLE_UNITS:
            continue
        base = _UNITS[base_name]
        exponent = _SI_PREFIXES[prefix]
        scale = Fraction(10**exponent) if exponent >= 0 else Fraction(1, 10 ** -exponent)
        return UnitDefinition(
            base.dimensions,
            base.scale * scale,
            pi_power=base.pi_power,
        )
    return None


def _unit_expression_definition(expression: UnitExpr) -> UnitDefinition:
    if expression.kind == "dimensionless":
        return UnitDefinition(())
    if expression.kind == "symbol":
        definition = _resolve_unit_symbol(expression.value)
        if definition is None:
            raise ValueError(f"Unknown or unsupported unit {expression.value!r}")
        return definition
    if expression.kind == "group":
        return _unit_expression_definition(expression.children[0])
    if expression.kind == "power":
        base = _unit_expression_definition(expression.children[0])
        if base.kind == "absolute_temperature" and expression.exponent != 1:
            raise ValueError("Absolute-temperature units cannot be exponentiated")
        if base.offset and expression.exponent != 1:
            raise ValueError("Affine units cannot be exponentiated")
        if expression.exponent == 0:
            return UnitDefinition(())
        return UnitDefinition(
            tuple(
                (dimension, exponent * expression.exponent)
                for dimension, exponent in base.dimensions
                if exponent * expression.exponent
            ),
            base.scale ** expression.exponent,
            base.offset if expression.exponent == 1 else Fraction(0),
            base.kind if expression.exponent == 1 else "",
            base.pi_power * expression.exponent,
        )
    if expression.kind in {"multiply", "divide"}:
        left = _unit_expression_definition(expression.children[0])
        right = _unit_expression_definition(expression.children[1])
        if left.kind == "absolute_temperature" or right.kind == "absolute_temperature":
            raise ValueError("Absolute-temperature units cannot be composed")
        sign = 1 if expression.kind == "multiply" else -1
        scale = left.scale * (right.scale if sign > 0 else 1 / right.scale)
        return UnitDefinition(
            _combine_units(left.dimensions, right.dimensions, sign),
            scale,
            pi_power=left.pi_power + sign * right.pi_power,
        )
    raise ValueError(f"Unsupported unit expression kind {expression.kind!r}")


def _unit_expression_symbol(expression: UnitExpr) -> str:
    if expression.kind == "symbol":
        return expression.value.replace("µ", "u")
    if expression.kind == "dimensionless":
        return "1"
    if expression.kind == "group":
        return f"({_unit_expression_symbol(expression.children[0])})"
    if expression.kind == "power":
        return f"{_unit_expression_symbol(expression.children[0])}^{expression.exponent}"
    if expression.kind == "multiply":
        return "*".join(_unit_expression_symbol(child) for child in expression.children)
    if expression.kind == "divide":
        return "/".join(_unit_expression_symbol(child) for child in expression.children)
    return ""


def _resolve_type(type_expr: TypeExpr) -> AetherType:
    if type_expr.name == "[]":
        return AetherType(
            "Shape",
            dimensions=tuple(
                int(value) for value in type_expr.dimensions if value.isdecimal()
            ),
        )
    arguments = tuple(_resolve_type(argument) for argument in type_expr.arguments)
    if type_expr.name == "Tensor" and len(type_expr.arguments) >= 2:
        shape_expr = type_expr.arguments[1]
        dimensions = tuple(
            int(value) for value in shape_expr.dimensions if value.isdecimal()
        ) if shape_expr.name == "[]" else ()
        unit = UnitDefinition(())
        unit_symbol = ""
        if type_expr.unit_expression is not None:
            try:
                unit = _unit_expression_definition(type_expr.unit_expression)
                unit_symbol = _unit_expression_symbol(type_expr.unit_expression)
            except ValueError:
                pass
        elif len(type_expr.arguments) >= 3:
            unit_symbol = type_expr.arguments[2].name
            unit = _resolve_unit_symbol(unit_symbol) or UnitDefinition(())
        return AetherType(
            "Tensor",
            (arguments[0],),
            dimensions,
            unit.dimensions,
            unit.scale,
            unit.offset,
            unit_symbol,
            unit.kind,
            unit.pi_power,
        )
    if type_expr.name == "Array" and len(type_expr.arguments) == 2:
        shape_expr = type_expr.arguments[1]
        dimensions = tuple(
            int(value) for value in shape_expr.dimensions if value.isdecimal()
        ) if shape_expr.name == "[]" else ()
        return AetherType("Array", (arguments[0],), dimensions)
    if type_expr.unit_expression is not None:
        try:
            unit = _unit_expression_definition(type_expr.unit_expression)
        except ValueError:
            unit = UnitDefinition(())
        return AetherType(
            type_expr.name,
            arguments,
            unit=unit.dimensions,
            unit_scale=unit.scale,
            unit_offset=unit.offset,
            unit_symbol=_unit_expression_symbol(type_expr.unit_expression),
            unit_kind=unit.kind,
            unit_pi_power=unit.pi_power,
        )
    return AetherType(type_expr.name, arguments)


def _same_shape(left: AetherType, right: AetherType) -> bool:
    return left.name == right.name == "Tensor" and left.dimensions == right.dimensions


def _compatible(left: AetherType, right: AetherType) -> bool:
    if left.name == "Tensor" or right.name == "Tensor":
        return (
            _same_shape(left, right)
            and left.arguments == right.arguments
            and _same_unit(left, right)
        )
    if left.is_numeric and right.is_numeric:
        return left.name == right.name and _same_unit(left, right)
    return left == right


def _same_unit(left: AetherType, right: AetherType) -> bool:
    return (
        left.unit == right.unit
        and left.unit_scale == right.unit_scale
        and left.unit_offset == right.unit_offset
        and left.unit_kind == right.unit_kind
        and left.unit_pi_power == right.unit_pi_power
    )


def _with_unit(
    value: AetherType,
    unit_symbol: str,
    definition: UnitDefinition,
) -> AetherType:
    normalized_symbol = unit_symbol.replace("µ", "u")
    return AetherType(
        value.name,
        value.arguments,
        value.dimensions,
        definition.dimensions,
        definition.scale,
        definition.offset,
        normalized_symbol,
        definition.kind,
        definition.pi_power,
    )


def _combine_unit_types(
    left: AetherType,
    right: AetherType,
    sign: int,
) -> tuple[tuple[tuple[str, int], ...], Fraction, int]:
    return (
        _combine_units(left.unit, right.unit, sign),
        left.unit_scale * (right.unit_scale if sign > 0 else 1 / right.unit_scale),
        left.unit_pi_power + sign * right.unit_pi_power,
    )


class _Analyzer:
    def __init__(self, module: Module) -> None:
        self.module = module
        self.functions = {
            declaration.name: declaration
            for declaration in module.declarations
            if isinstance(declaration, FunctionDecl)
        }
        self.types = {
            declaration.name: declaration
            for declaration in module.declarations
            if isinstance(declaration, TypeDecl)
        }
        self.diagnostics: list[Diagnostic] = []
        self.expression_types: list[tuple[SourceSpan, AetherType]] = []
        self.conversions: list[UnitConversion] = []
        self.current_return_type: AetherType | None = None
        self.effect_summaries: dict[str, EffectSummary] = {}
        self.current_mutable_names: set[str] = set()

    def error(self, code: str, message: str, span: SourceSpan) -> None:
        self.diagnostics.append(Diagnostic(code, message, span))

    def _is_plain_collection_element(
        self,
        type_: AetherType,
        visiting: frozenset[str] = frozenset(),
    ) -> bool:
        if type_.name in _NUMERIC_TYPES | _INTEGER_TYPES | {"bool"}:
            return not (
                type_.unit
                or type_.unit_scale != 1
                or type_.unit_offset
                or type_.unit_symbol
                or type_.unit_kind
                or type_.unit_pi_power
            )
        declaration = self.types.get(type_.name)
        if declaration is None or type_.name in visiting:
            return False
        next_visiting = visiting | {type_.name}
        return all(
            self._is_plain_collection_element(
                _resolve_type(field.type),
                next_visiting,
            )
            for field in declaration.fields
        )

    def analyze(self) -> SemanticResult:
        summaries = self._analyze_effects()
        self.effect_summaries = {
            summary.function: summary for summary in summaries
        }
        for function in self.functions.values():
            if function.name in _BUILTIN_OPERATIONS or function.name in _BUILTIN_CALLS:
                self.error(
                    "E_RESERVED_NAME",
                    f"Function name {function.name!r} is reserved for a built-in operation",
                    function.span,
                )
            if function.external_abi is not None:
                self._validate_extern_function(function)
            for attribute in function.attributes:
                if attribute.name == "C" and (
                    not function.name.isascii()
                    or not function.name.isidentifier()
                    or function.name.startswith("_")
                    or function.name in {
                        "auto", "break", "case", "char", "const", "continue",
                        "default", "do", "double", "else", "enum", "extern",
                        "float", "for", "goto", "if", "inline", "int", "long",
                        "register", "restrict", "return", "short", "signed",
                        "sizeof", "static", "struct", "switch", "typedef",
                        "union", "unsigned", "void", "volatile", "while",
                        "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                        "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                        "_Thread_local",
                    }
                ):
                    self.error(
                        "E_C_EXPORT_NAME",
                        f"Function name {function.name!r} is not a portable C identifier",
                        function.span,
                    )
        for declaration in self.types.values():
            if declaration.name in _BUILTIN_TYPES and declaration.name != "FrameBuffer":
                self.error(
                    "E_RESERVED_TYPE",
                    f"Type name {declaration.name!r} is reserved for a built-in type",
                    declaration.span,
                )
            elif declaration.name == "FrameBuffer":
                self.error(
                    "E_RESERVED_TYPE",
                    "Type name 'FrameBuffer' is reserved for frame-managed storage",
                    declaration.span,
                )
            for field in declaration.fields:
                self._validate_type(field.type)
                if field.type.name == "FrameBuffer" or any(
                    argument.name == "FrameBuffer"
                    for argument in field.type.arguments
                ):
                    self.error(
                        "E_FRAME_BUFFER_ESCAPE",
                        "FrameBuffer ownership cannot be stored in user aggregates",
                        field.type.span,
                    )
        self._validate_struct_cycles()
        for function in self.functions.values():
            self._analyze_function(function)
        return SemanticResult(
            tuple(self.expression_types),
            summaries,
            tuple(self.conversions),
            tuple(self.diagnostics),
        )

    def _validate_struct_cycles(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()
        reported: set[tuple[str, str]] = set()

        def referenced_types(type_expr: TypeExpr) -> tuple[str, ...]:
            names = (type_expr.name,) if type_expr.name in self.types else ()
            return names + tuple(
                name
                for argument in type_expr.arguments
                for name in referenced_types(argument)
            )

        def visit(name: str) -> None:
            if name in visited:
                return
            visiting.add(name)
            declaration = self.types[name]
            for field in declaration.fields:
                for target in referenced_types(field.type):
                    if target in visiting:
                        edge = (name, target)
                        if edge not in reported:
                            reported.add(edge)
                            self.error(
                                "E_RECURSIVE_TYPE",
                                "Struct values cannot contain themselves recursively",
                                field.span,
                            )
                    else:
                        visit(target)
            visiting.remove(name)
            visited.add(name)

        for name in self.types:
            visit(name)

    def _validate_extern_function(self, function: FunctionDecl) -> None:
        if function.external_abi != "C":
            self.error(
                "E_EXTERN_ABI",
                f"Unsupported foreign ABI {function.external_abi!r}",
                function.span,
            )
        link_attributes = [
            attribute for attribute in function.attributes
            if attribute.name == "link"
        ]
        if len(link_attributes) != 1 or function.external_library is None:
            self.error(
                "E_EXTERN_LINK",
                'An extern "C" declaration requires exactly one @link("library")',
                function.span,
            )
        elif not re.fullmatch(r"[A-Za-z0-9_.+-]+", function.external_library):
            self.error(
                "E_EXTERN_LINK",
                "Library name must be a simple linker name, not a path",
                link_attributes[0].span,
            )
        if function.return_type is None:
            self.error(
                "E_EXTERN_RETURN_TYPE",
                "Extern declarations require an explicit return type (use void for no value)",
                function.span,
            )
        for parameter in function.parameters:
            if not self._is_ffi_type(parameter.type, allow_void=False):
                self.error(
                    "E_FFI_TYPE",
                    f"Type {parameter.type.name!r} is not supported in a C ABI parameter",
                    parameter.type.span,
                )
        if function.return_type is not None and not self._is_ffi_type(
            function.return_type,
            allow_void=True,
        ):
            self.error(
                "E_FFI_TYPE",
                f"Type {function.return_type.name!r} is not supported as a C ABI return type",
                function.return_type.span,
            )

    @staticmethod
    def _is_ffi_type(type_expr: TypeExpr, *, allow_void: bool) -> bool:
        if type_expr.name in {"Slice", "SliceMut"}:
            return (
                not allow_void
                and len(type_expr.arguments) == 1
                and type_expr.arguments[0].name == "u8"
                and not type_expr.arguments[0].arguments
                and not type_expr.dimensions
                and type_expr.unit_expression is None
            )
        if type_expr.arguments or type_expr.dimensions or type_expr.unit_expression:
            return False
        if type_expr.name == "void":
            return allow_void
        return type_expr.name in _FFI_TYPES

    def _validate_type(self, type_expr: TypeExpr) -> None:
        for argument in type_expr.arguments:
            self._validate_type(argument)
        if type_expr.name in _UNSUPPORTED_BUILTIN_TYPES:
            self.error(
                "E_UNSUPPORTED_TYPE",
                f"Built-in type {type_expr.name!r} is recognized but not supported",
                type_expr.span,
            )
        if type_expr.unit_expression is not None:
            if type_expr.name != "Tensor" and type_expr.name not in _NUMERIC_TYPES:
                self.error(
                    "E_UNIT_TYPE",
                    "Unit annotations are supported only on numeric and Tensor types",
                    type_expr.span,
                )
            try:
                _unit_expression_definition(type_expr.unit_expression)
            except ValueError as error:
                message = str(error)
                code = (
                    "E_UNKNOWN_UNIT"
                    if message.startswith("Unknown or unsupported unit")
                    else "E_UNIT_EXPRESSION"
                )
                self.error(code, message, type_expr.unit_expression.span)
        if type_expr.name == "FrameBuffer":
            return
        if type_expr.name == "Result":
            if len(type_expr.arguments) != 2:
                self.error(
                    "E_RESULT_TYPE",
                    "Result type requires a value type and an error type",
                    type_expr.span,
                )
            elif type_expr.arguments[1].name not in {
                "IoError",
                "FsError",
                "NetError",
                "ArgsError",
                "CollectionError",
            }:
                self.error(
                    "E_RESULT_ERROR_TYPE",
                    "Result error type must be a supported error category",
                    type_expr.arguments[1].span,
                )
            return
        if type_expr.name == "Array":
            if len(type_expr.arguments) != 2:
                self.error(
                    "E_ARRAY_TYPE",
                    "Array type requires an element type and a static shape",
                    type_expr.span,
                )
                return
            element, shape = type_expr.arguments
            if shape.name != "[]" or len(shape.dimensions) != 1:
                self.error(
                    "E_ARRAY_SHAPE",
                    "Array shape must contain exactly one static dimension",
                    shape.span,
                )
            elif not shape.dimensions[0].isdecimal() or int(shape.dimensions[0]) <= 0:
                self.error(
                    "E_ARRAY_DIMENSION",
                    "Array length must be a positive integer",
                    shape.span,
                )
            if not self._is_plain_collection_element(_resolve_type(element)):
                self.error(
                    "E_COLLECTION_ELEMENT",
                    "Array element type must be a supported plain value type",
                    element.span,
                )
            return
        if type_expr.name == "List":
            if len(type_expr.arguments) != 1:
                self.error(
                    "E_LIST_TYPE",
                    "List type requires exactly one element type",
                    type_expr.span,
                )
            elif not self._is_plain_collection_element(
                _resolve_type(type_expr.arguments[0])
            ):
                self.error(
                    "E_COLLECTION_ELEMENT",
                    "List element type must be a supported plain value type",
                    type_expr.arguments[0].span,
                )
            return
        if type_expr.name == "Slice":
            if len(type_expr.arguments) != 1 or not self._is_plain_collection_element(
                _resolve_type(type_expr.arguments[0])
            ):
                self.error(
                    "E_SLICE_TYPE",
                    "Owned Slice requires one supported plain element type",
                    type_expr.span,
                )
            return
        if type_expr.name == "SliceMut":
            self.error(
                "E_UNSUPPORTED_TYPE",
                "SliceMut is only supported as a borrowed extern C parameter",
                type_expr.span,
            )
            return
        if type_expr.name != "Tensor":
            return
        if len(type_expr.arguments) < 2:
            self.error(
                "E_TENSOR_SHAPE",
                "Tensor type requires an element type and a static shape",
                type_expr.span,
            )
            return
        if len(type_expr.arguments) > 3:
            self.error(
                "E_TENSOR_TYPE",
                "Tensor type accepts an element type, shape, and optional unit",
                type_expr.span,
            )
        element_type = type_expr.arguments[0]
        if element_type.name not in _NUMERIC_TYPES:
            self.error(
                "E_TENSOR_ELEMENT_TYPE",
                "Tensor element type must be a supported numeric type",
                element_type.span,
            )
        shape = type_expr.arguments[1]
        if shape.name != "[]":
            self.error(
                "E_TENSOR_SHAPE",
                "Tensor shape must use bracketed dimensions",
                shape.span,
            )
            return
        if not shape.dimensions:
            self.error(
                "E_TENSOR_SHAPE",
                "Tensor shape must contain at least one static dimension",
                shape.span,
            )
        for dimension in shape.dimensions:
            if not dimension.isdecimal() or int(dimension) <= 0:
                self.error(
                    "E_TENSOR_DIMENSION",
                    f"Tensor dimension {dimension!r} must be a positive integer",
                    shape.span,
                )
        if len(type_expr.arguments) == 3:
            unit = type_expr.arguments[2]
            if _resolve_unit_symbol(unit.name) is None or unit.arguments:
                self.error(
                    "E_UNKNOWN_UNIT",
                    f"Unknown or unsupported unit {unit.name!r}",
                    unit.span,
                )

    def _analyze_effects(self) -> tuple[EffectSummary, ...]:
        declared = {
            name: {
                attribute.name
                for attribute in function.attributes
                if attribute.name in _EFFECTS
            }
            for name, function in self.functions.items()
        }
        calls: dict[str, list[tuple[str, SourceSpan]]] = {
            name: [] for name in self.functions
        }
        required = {name: set() for name in self.functions}
        for name, function in self.functions.items():
            if function.external_abi is not None:
                required[name].add("ffi")
            def visit_statements(statements: tuple[Statement, ...]) -> None:
                for statement in statements:
                    if isinstance(statement, (IfStmt, WhileStmt)):
                        expressions = (statement.condition,)
                        visit_statements(statement.body)
                        if isinstance(statement, IfStmt):
                            visit_statements(statement.else_body)
                    elif isinstance(statement, ForStmt):
                        expressions = (statement.start, statement.end)
                        visit_statements(statement.body)
                    else:
                        expression = _statement_expression(statement)
                        expressions = (expression,) if expression is not None else ()
                    for expression in expressions:
                        for node in _walk(expression):
                            if node.kind == "call" and node.children[0].kind == "identifier":
                                callee = node.children[0].value
                                if callee in self.functions:
                                    calls[name].append((callee, node.span))
                                elif callee in _BUILTIN_OPERATIONS:
                                    effect = _BUILTIN_OPERATIONS[callee].effect
                                    if effect:
                                        required[name].add(effect)

            visit_statements(function.body)

        changed = True
        while changed:
            changed = False
            for caller, callees in calls.items():
                for callee, _ in callees:
                    new_effects = required[caller] | required[callee]
                    if new_effects != required[caller]:
                        required[caller] = new_effects
                        changed = True

        for function_name in self.functions:
            missing = required[function_name] - declared[function_name]
            if missing:
                self.error(
                    "E_EFFECT_VIOLATION",
                    f"Function {function_name!r} must declare effect(s): "
                    + ", ".join(f"@{effect}" for effect in sorted(missing)),
                    self.functions[function_name].span,
                )
        return tuple(
            EffectSummary(
                name,
                tuple(sorted(declared[name])),
                tuple(sorted(required[name])),
            )
            for name in sorted(self.functions)
        )

    def _analyze_function(self, function: FunctionDecl) -> None:
        if function.external_abi is not None:
            return

        def contains_frame_buffer(type_: AetherType | None) -> bool:
            return type_ is not None and (
                type_.name == "FrameBuffer"
                or any(contains_frame_buffer(argument) for argument in type_.arguments)
            )

        for parameter in function.parameters:
            self._validate_type(parameter.type)
        environment = {
            parameter.name: _resolve_type(parameter.type)
            for parameter in function.parameters
        }
        expected_return = (
            _resolve_type(function.return_type)
            if function.return_type is not None
            else None
        )
        self.current_return_type = expected_return
        if function.return_type is not None:
            self._validate_type(function.return_type)
        result_bindings: dict[str, SourceSpan] = {}
        handled_results: set[str] = set()
        mutable_bindings: set[str] = set()

        def field_type_for_assignment(
            target: AetherType,
            field_name: str,
            span: SourceSpan,
        ) -> AetherType | None:
            declaration = self.types.get(target.name)
            if declaration is None:
                self.error(
                    "E_ASSIGN_FIELD_TARGET",
                    f"Cannot assign a field on non-struct value of type {target.display()}",
                    span,
                )
                return None
            field = next(
                (item for item in declaration.fields if item.name == field_name),
                None,
            )
            if field is None:
                self.error(
                    "E_UNKNOWN_FIELD",
                    f"Type {target.name!r} has no field {field_name!r}",
                    span,
                )
                return None
            return _resolve_type(field.type)

        def block_returns(statements: tuple[Statement, ...]) -> bool:
            for statement in statements:
                if isinstance(statement, ReturnStmt):
                    return True
                if isinstance(statement, IfStmt) and statement.else_body:
                    if block_returns(statement.body) and block_returns(statement.else_body):
                        return True
            return False

        def analyze_nested(
            statements: tuple[Statement, ...],
            parent_environment: dict[str, AetherType],
            parent_mutable: set[str],
            loop_depth: int,
        ) -> bool:
            environment = dict(parent_environment)
            mutable = set(parent_mutable)
            for nested in statements:
                self.current_mutable_names = mutable
                if isinstance(nested, IfStmt):
                    condition = self._infer(nested.condition, environment)
                    if condition is not None and condition.name != "bool":
                        self.error(
                            "E_CONDITION_TYPE",
                            "if condition must have type bool",
                            nested.condition.span,
                        )
                    then_returns = analyze_nested(
                        nested.body,
                        environment,
                        mutable,
                        loop_depth,
                    )
                    else_returns = analyze_nested(
                        nested.else_body,
                        environment,
                        mutable,
                        loop_depth,
                    )
                    if then_returns and else_returns:
                        return True
                elif isinstance(nested, WhileStmt):
                    condition = self._infer(nested.condition, environment)
                    if condition is not None and condition.name != "bool":
                        self.error(
                            "E_CONDITION_TYPE",
                            "while condition must have type bool",
                            nested.condition.span,
                        )
                    analyze_nested(nested.body, environment, mutable, loop_depth + 1)
                elif isinstance(nested, ForStmt):
                    start_type = self._infer(nested.start, environment)
                    end_type = self._infer(nested.end, environment)
                    if start_type is not None and not _is_range_integer(start_type):
                        self.error(
                            "E_RANGE_TYPE",
                            "for range bounds must have an integer type",
                            nested.start.span,
                        )
                    if end_type is not None and not _is_range_integer(end_type):
                        self.error(
                            "E_RANGE_TYPE",
                            "for range bounds must have an integer type",
                            nested.end.span,
                        )
                    if (
                        start_type is not None
                        and end_type is not None
                        and start_type.name != end_type.name
                    ):
                        self.error(
                            "E_RANGE_TYPE",
                            "for range bounds must have the same integer type",
                            nested.span,
                        )
                    loop_environment = dict(environment)
                    if start_type is not None:
                        loop_environment[nested.name] = start_type
                    analyze_nested(
                        nested.body,
                        loop_environment,
                        mutable,
                        loop_depth + 1,
                    )
                elif isinstance(nested, LoopControlStmt):
                    if loop_depth == 0:
                        self.error(
                            "E_LOOP_CONTROL_CONTEXT",
                            f"'{nested.kind}' is only valid inside a loop",
                            nested.span,
                        )
                elif isinstance(nested, LetStmt):
                    annotation = (
                        _resolve_type(nested.annotation)
                        if nested.annotation is not None
                        else None
                    )
                    inferred = self._infer(
                        nested.value,
                        environment,
                        expected_type=annotation,
                    )
                    if annotation is not None:
                        self._validate_type(nested.annotation)
                    if (
                        annotation is not None
                        and inferred is not None
                        and not _compatible(annotation, inferred)
                    ):
                        self.error(
                            "E_TYPE_MISMATCH",
                            f"Binding {nested.name!r} declares {annotation.display()} "
                            f"but receives {inferred.display()}",
                            nested.span,
                        )
                    if contains_frame_buffer(inferred) or contains_frame_buffer(annotation):
                        self.error(
                            "E_CONTROL_RESOURCE_UNSUPPORTED",
                            "FrameBuffer locals inside control-flow blocks are not yet supported",
                            nested.span,
                        )
                    environment[nested.name] = (
                        annotation or inferred or AetherType("Unknown")
                    )
                    if nested.binding_kind == "var":
                        mutable.add(nested.name)
                    if inferred is not None and inferred.name == "Result":
                        self.error(
                            "E_CONTROL_RESULT_UNSUPPORTED",
                            "Result bindings inside control-flow blocks are not yet supported",
                            nested.span,
                        )
                elif isinstance(nested, AssignmentStmt):
                    target = environment.get(nested.name)
                    inferred = self._infer(nested.value, environment)
                    if contains_frame_buffer(target) or contains_frame_buffer(inferred):
                        self.error(
                            "E_CONTROL_RESOURCE_UNSUPPORTED",
                            "FrameBuffer rebinding inside control-flow blocks is not yet supported",
                            nested.span,
                        )
                    if target is None:
                        self.error(
                            "E_UNKNOWN_NAME",
                            f"Unknown assignment target {nested.name!r}",
                            nested.span,
                        )
                    elif nested.name not in mutable:
                        self.error(
                            "E_ASSIGN_IMMUTABLE",
                            f"Cannot rebind immutable value {nested.name!r}; declare it with 'var'",
                            nested.span,
                        )
                    elif nested.field_name is not None:
                        assigned_field_type = field_type_for_assignment(
                            target,
                            nested.field_name,
                            nested.span,
                        )
                        if (
                            inferred is not None
                            and assigned_field_type is not None
                            and not _compatible(assigned_field_type, inferred)
                        ):
                            self.error(
                                _mismatch_code(assigned_field_type, inferred),
                                f"Cannot assign {inferred.display()} to field "
                                f"{nested.name}.{nested.field_name} of type "
                                f"{assigned_field_type.display()}",
                                nested.span,
                            )
                    elif inferred is not None and not _compatible(target, inferred):
                        self.error(
                            _mismatch_code(target, inferred),
                            f"Cannot rebind {nested.name!r} of type {target.display()} "
                            f"with {inferred.display()}",
                            nested.span,
                        )
                    if nested.operator != "=" and target is not None and not target.is_numeric:
                        self.error(
                            "E_COMPOUND_ASSIGN_TYPE",
                            "Compound assignment currently requires a numeric scalar",
                            nested.span,
                        )
                elif isinstance(nested, ReturnStmt):
                    actual = (
                        self._infer(
                            nested.value,
                            environment,
                            expected_type=expected_return,
                        )
                        if nested.value is not None
                        else AetherType("void")
                    )
                    if (
                        expected_return is not None
                        and actual is not None
                        and not _compatible(expected_return, actual)
                    ):
                        self.error(
                            "E_RETURN_TYPE",
                            f"Function {function.name!r} returns {actual.display()}, "
                            f"expected {expected_return.display()}",
                            nested.span,
                        )
                    return True
                elif isinstance(nested, ExprStmt):
                    actual = self._infer(nested.expression, environment)
                    if contains_frame_buffer(actual):
                        self.error(
                            "E_CONTROL_RESOURCE_UNSUPPORTED",
                            "FrameBuffer expression statements inside control-flow blocks are not yet supported",
                            nested.span,
                        )
                    if actual is not None and actual.name == "Result":
                        self.error(
                            "E_UNHANDLED_RESULT",
                            "Result value must be handled, propagated, or returned",
                            nested.span,
                        )
            return False

        reachable = True
        for statement in function.body:
            self.current_mutable_names = mutable_bindings
            if isinstance(statement, (IfStmt, WhileStmt, ForStmt, LoopControlStmt)):
                analyze_nested((statement,), environment, mutable_bindings, 0)
                continue
            was_reachable = reachable
            statement_expression = _statement_expression(statement)
            if was_reachable and statement_expression is not None:
                referenced = {
                    node.value
                    for node in _walk(statement_expression)
                    if node.kind == "identifier" and node.value in handled_results
                }
                for name in sorted(referenced):
                    self.error(
                        "E_RESULT_REUSE",
                        f"Result binding {name!r} was already consumed",
                        statement.span,
                    )
            if isinstance(statement, LetStmt):
                if statement.annotation is not None:
                    self._validate_type(statement.annotation)
                annotation = (
                    _resolve_type(statement.annotation)
                    if statement.annotation is not None
                    else None
                )
                inferred = self._infer(
                    statement.value,
                    environment,
                    expected_type=annotation,
                )
                if (
                    contains_frame_buffer(inferred)
                    or contains_frame_buffer(annotation)
                ) and (
                    statement.binding_kind == "const"
                    or annotation is not None
                ):
                    self.error(
                        "E_FRAME_BUFFER_OWNERSHIP",
                        "FrameBuffer-containing locals must use unannotated let or var bindings",
                        statement.span,
                    )
                if annotation is not None and inferred is not None and not _compatible(annotation, inferred):
                    self.error(
                        "E_TYPE_MISMATCH",
                        f"Binding {statement.name!r} declares {annotation.display()} "
                        f"but receives {inferred.display()}",
                        statement.span,
                    )
                environment[statement.name] = annotation or inferred or AetherType("Unknown")
                self._validate_frame_buffer_usage(statement.value, environment)
                if statement.binding_kind == "var":
                    mutable_bindings.add(statement.name)
                if reachable and inferred is not None and inferred.name == "Result":
                    result_bindings[statement.name] = statement.span
                elif reachable and statement.value.kind == "identifier":
                    source_name = statement.value.value
                    if source_name in result_bindings:
                        result_bindings.pop(source_name)
                        if inferred is not None and inferred.name == "Result":
                            result_bindings[statement.name] = statement.span
            elif isinstance(statement, AssignmentStmt):
                target_type = environment.get(statement.name)
                assignment_value = statement.value
                if statement.operator != "=" and target_type is not None:
                    assignment_value = Expr(
                        "binary",
                        statement.span,
                        statement.operator[:-1],
                        (
                            Expr(
                                "identifier",
                                statement.span,
                                statement.name,
                            ),
                            statement.value,
                        ),
                    )
                inferred = self._infer(assignment_value, environment)
                self._validate_frame_buffer_usage(statement.value, environment)
                if target_type is None:
                    self.error(
                        "E_UNKNOWN_NAME",
                        f"Unknown assignment target {statement.name!r}",
                        statement.span,
                    )
                elif statement.name not in mutable_bindings:
                    self.error(
                        "E_ASSIGN_IMMUTABLE",
                        f"Cannot rebind immutable value {statement.name!r}; "
                        "declare it with 'var'",
                        statement.span,
                    )
                elif statement.field_name is not None:
                    assigned_field_type = field_type_for_assignment(
                        target_type,
                        statement.field_name,
                        statement.span,
                    )
                    if (
                        inferred is not None
                        and assigned_field_type is not None
                        and not _compatible(assigned_field_type, inferred)
                    ):
                        self.error(
                            _mismatch_code(assigned_field_type, inferred),
                            f"Cannot assign {inferred.display()} to field "
                            f"{statement.name}.{statement.field_name} of type "
                            f"{assigned_field_type.display()}",
                            statement.span,
                        )
                elif inferred is not None and not _compatible(target_type, inferred):
                    self.error(
                        _mismatch_code(target_type, inferred),
                        f"Cannot rebind {statement.name!r} of type "
                        f"{target_type.display()} with {inferred.display()}",
                        statement.span,
                    )
                if (
                    statement.operator != "="
                    and target_type is not None
                    and not target_type.is_numeric
                ):
                    self.error(
                        "E_COMPOUND_ASSIGN_TYPE",
                        "Compound assignment currently requires a numeric scalar",
                        statement.span,
                    )
                if statement.field_name is not None and statement.operator != "=":
                    self.error(
                        "E_STRUCT_FIELD_ASSIGNMENT",
                        "Compound assignment to struct fields is not supported",
                        statement.span,
                    )
                if target_type is not None and (
                    target_type.name == "Result"
                    or (inferred is not None and inferred.name == "Result")
                ):
                    self.error(
                        "E_ASSIGN_RESULT_UNSUPPORTED",
                        "Rebinding Result values is not supported",
                        statement.span,
                    )
                if (
                    target_type is not None
                    and target_type.name == "FrameBuffer"
                    and inferred is not None
                    and inferred.name != "FrameBuffer"
                ):
                    self.error(
                        "E_FRAME_BUFFER_OWNERSHIP",
                        "FrameBuffer rebinding requires another FrameBuffer value",
                        statement.span,
                    )
            elif isinstance(statement, ReturnStmt):
                actual = (
                    self._infer(
                        statement.value,
                        environment,
                        expected_type=expected_return,
                    )
                    if statement.value is not None
                    else AetherType("void")
                )
                if statement.value is not None:
                    self._validate_frame_buffer_usage(statement.value, environment)
                propagates_result = (
                    expected_return is not None
                    and expected_return.name == "Result"
                    and len(expected_return.arguments) == 2
                    and actual == expected_return.arguments[0]
                    and statement.value is not None
                    and any(
                        node.kind == "try"
                        for node in _walk(statement.value)
                    )
                )
                if (
                    function.external_abi is None
                    and expected_return is not None
                    and actual is not None
                    and not _compatible(expected_return, actual)
                    and not propagates_result
                ):
                    self.error(
                        "E_RETURN_TYPE",
                        f"Function {function.name!r} returns {actual.display()}, "
                        f"expected {expected_return.display()}",
                        statement.span,
                    )
                if statement.value is not None:
                    consumed = _consumed_result_names(statement.value, "return")
                    if was_reachable:
                        for name in consumed:
                            if name in result_bindings:
                                result_bindings.pop(name)
                                handled_results.add(name)
                reachable = False
            elif isinstance(statement, ExprStmt):
                actual = self._infer(statement.expression, environment)
                self._validate_frame_buffer_usage(statement.expression, environment)
                if actual is not None and actual.name == "Result":
                    self.error(
                        "E_UNHANDLED_RESULT",
                        "Result value must be returned, matched, propagated with '?', or bound and used",
                        statement.span,
                    )
            if isinstance(statement, LetStmt):
                consumed = _consumed_result_names(statement.value, "bind")
            elif isinstance(statement, AssignmentStmt):
                consumed = _consumed_result_names(statement.value, "bind")
            elif isinstance(statement, ExprStmt):
                consumed = _consumed_result_names(statement.expression, "expression")
            else:
                consumed = set()
            for name in (consumed if was_reachable else ()):
                if name in result_bindings:
                    result_bindings.pop(name)
                    handled_results.add(name)
                    if (
                        isinstance(statement, LetStmt)
                        and statement.value.kind == "identifier"
                        and statement.value.value == name
                        and environment.get(statement.name, AetherType("Unknown")).name
                        == "Result"
                    ):
                        result_bindings[statement.name] = statement.span

        for name, span in result_bindings.items():
            self.error(
                "E_UNHANDLED_RESULT",
                f"Result binding {name!r} is never handled, propagated, or returned",
                span,
            )
        if (
            function.external_abi is None
            and expected_return is not None
            and expected_return.name not in {"void", "Unknown"}
            and not block_returns(function.body)
        ):
            self.error(
                "E_RETURN_PATH",
                f"Function {function.name!r} must return a value on every path",
                function.span,
            )
        self.current_return_type = None

    def _validate_frame_buffer_usage(
        self,
        expression: Expr,
        environment: dict[str, AetherType],
    ) -> None:
        def contains_resource(type_: AetherType) -> bool:
            return type_.name == "FrameBuffer" or any(
                contains_resource(argument) for argument in type_.arguments
            )

        def inferred_type(node: Expr) -> AetherType:
            if node.kind == "identifier":
                return environment.get(node.value, self._type_at(node.span))
            return self._type_at(node.span)

        def visit(node: Expr) -> None:
            type_ = inferred_type(node)
            if contains_resource(type_):
                if node.kind in {"identifier", "try", "group"}:
                    if node.kind == "try" and node.children:
                        visit(node.children[0])
                    elif node.kind == "group" and node.children:
                        visit(node.children[0])
                    return
                if node.kind == "call" and node.children:
                    callee = node.children[0]
                    name = callee.value if callee.kind == "identifier" else ""
                    if name == "frame_buffer":
                        for argument in node.children[1:]:
                            visit(argument)
                        return
                    if name == "frame_buffer_len":
                        if len(node.children) == 2 and not contains_resource(
                            inferred_type(node.children[1])
                        ):
                            self.error(
                                "E_FRAME_BUFFER_OWNERSHIP",
                                "frame_buffer_len requires a FrameBuffer value",
                                node.children[1].span,
                            )
                        for argument in node.children[1:]:
                            visit(argument)
                        return
                    if name in {"Ok", "Err"}:
                        for argument in node.children[1:]:
                            visit(argument)
                        return
                    target = self.functions.get(name)
                    if target is not None:
                        for argument, parameter in zip(
                            node.children[1:], target.parameters
                        ):
                            visit(argument)
                            if (
                                contains_resource(_resolve_type(parameter.type))
                                and not contains_resource(inferred_type(argument))
                            ):
                                self.error(
                                    "E_FRAME_BUFFER_OWNERSHIP",
                                    "FrameBuffer parameter requires an owned or borrowed FrameBuffer value",
                                    argument.span,
                                )
                        return
                self.error(
                    "E_FRAME_BUFFER_OWNERSHIP",
                    "FrameBuffer values may only be copied, passed, or returned as values",
                    node.span,
                )
                return

            if node.kind == "call" and node.children:
                callee = node.children[0]
                name = callee.value if callee.kind == "identifier" else ""
                target = self.functions.get(name)
                if target is not None:
                    for argument, parameter in zip(
                        node.children[1:], target.parameters
                    ):
                        visit(argument)
                        if (
                            contains_resource(_resolve_type(parameter.type))
                            and not contains_resource(inferred_type(argument))
                        ):
                            self.error(
                                "E_FRAME_BUFFER_OWNERSHIP",
                                "FrameBuffer parameter requires a FrameBuffer value",
                                argument.span,
                            )
                    return
            for child in node.children:
                visit(child)
            for arm in node.arms:
                visit(arm.value)

        visit(expression)

    def _type_at(self, span: SourceSpan) -> AetherType:
        for expression_span, type_ in reversed(self.expression_types):
            if expression_span == span:
                return type_
        return AetherType("Unknown")

    def _record(self, expression: Expr, value: AetherType | None) -> AetherType | None:
        if value is not None:
            self.expression_types.append((expression.span, value))
        return value

    def _infer(
        self,
        expression: Expr | None,
        environment: dict[str, AetherType],
        pipeline_input: AetherType | None = None,
        *,
        expected_type: AetherType | None = None,
    ) -> AetherType | None:
        if expression is None:
            return None
        if expression.kind == "literal":
            value = expression.value
            if value in {"true", "false"}:
                return self._record(expression, AetherType("bool"))
            if value == "none":
                return self._record(expression, AetherType("None"))
            if value.startswith(("'", '"')):
                return self._record(expression, AetherType("str"))
            return self._record(
                expression,
                AetherType(
                    "f32"
                    if not value.lower().startswith(("0x", "0b"))
                    and any(char in value for char in ".eE")
                    else "i32"
                ),
            )
        if expression.kind == "identifier":
            return self._record(expression, environment.get(expression.value))
        if expression.kind == "group":
            value = self._infer(
                expression.children[0],
                environment,
                expected_type=expected_type,
            )
            return self._record(expression, value)
        if expression.kind == "array_literal":
            element_type = (
                expected_type.arguments[0]
                if expected_type is not None
                and expected_type.name in {"Array", "List", "Slice"}
                and expected_type.arguments
                else None
            )
            inferred_types = [
                self._infer(child, environment, expected_type=element_type)
                for child in expression.children
            ]
            if expected_type is not None:
                if expected_type.name not in {"Array", "List", "Slice"}:
                    self.error(
                        "E_COLLECTION_LITERAL_TYPE",
                        "Collection literal requires an Array, List, or Slice annotation",
                        expression.span,
                    )
                    return None
                if expected_type.name == "Array":
                    if len(expected_type.dimensions) != 1 or (
                        expected_type.dimensions[0] != len(expression.children)
                    ):
                        self.error(
                            "E_ARRAY_LITERAL_LENGTH",
                            f"Array literal has {len(expression.children)} elements, "
                            f"expected {expected_type.dimensions[0] if expected_type.dimensions else 0}",
                            expression.span,
                        )
                for child_type in inferred_types:
                    if (
                        child_type is not None
                        and expected_type.arguments
                        and not _compatible(expected_type.arguments[0], child_type)
                    ):
                        self.error(
                            "E_COLLECTION_ELEMENT_TYPE",
                            f"Collection element expects "
                            f"{expected_type.arguments[0].display()}, got "
                            f"{child_type.display()}",
                            expression.span,
                        )
                return self._record(expression, expected_type)
            if not inferred_types:
                self.error(
                    "E_COLLECTION_LITERAL_INFERENCE",
                    "Empty collection literal requires an explicit collection type",
                    expression.span,
                )
                return None
            inferred_element = inferred_types[0]
            if inferred_element is None:
                return None
            for child_type in inferred_types[1:]:
                if child_type is not None and not _compatible(
                    inferred_element,
                    child_type,
                ):
                    self.error(
                        "E_COLLECTION_ELEMENT_TYPE",
                        "Collection literal elements must have one matching type",
                        expression.span,
                    )
            return self._record(
                expression,
                AetherType("Array", (inferred_element,), (len(inferred_types),)),
            )
        if expression.kind == "try":
            result_type = self._infer(expression.children[0], environment)
            if result_type is None:
                return None
            if result_type.name != "Result" or len(result_type.arguments) != 2:
                self.error(
                    "E_TRY_TYPE",
                    "The '?' operator requires a Result<T, E> value",
                    expression.span,
                )
                return None
            if (
                self.current_return_type is None
                or self.current_return_type.name != "Result"
                or len(self.current_return_type.arguments) != 2
            ):
                self.error(
                    "E_TRY_CONTEXT",
                    "The '?' operator is only valid in a function returning Result<T, E>",
                    expression.span,
                )
                return None
            source_error = result_type.arguments[1]
            return_error = self.current_return_type.arguments[1]
            if source_error != return_error:
                self.error(
                    "E_TRY_ERROR_TYPE",
                    f"Cannot propagate {source_error.display()} from a function returning "
                    f"{return_error.display()}",
                    expression.span,
                )
                return None
            return self._record(expression, result_type.arguments[0])
        if expression.kind == "match":
            result_type = self._infer(expression.children[0], environment)
            if result_type is None:
                return None
            if result_type.name != "Result" or len(result_type.arguments) != 2:
                self.error(
                    "E_MATCH_TYPE",
                    "Result match requires a Result<T, E> value",
                    expression.span,
                )
                return None
            seen: set[str] = set()
            branch_types: list[AetherType] = []
            for arm in expression.arms:
                if arm.variant in seen:
                    self.error(
                        "E_MATCH_DUPLICATE_ARM",
                        f"Duplicate {arm.variant} arm",
                        arm.span,
                    )
                seen.add(arm.variant)
                arm_environment = dict(environment)
                if arm.binding is not None:
                    payload_index = 0 if arm.variant == "Ok" else 1
                    arm_environment[arm.binding] = result_type.arguments[payload_index]
                inferred_arm = self._infer(arm.value, arm_environment)
                if inferred_arm is not None:
                    branch_types.append(inferred_arm)
            if seen != {"Ok", "Err"}:
                missing = {"Ok", "Err"} - seen
                self.error(
                    "E_MATCH_NON_EXHAUSTIVE",
                    "Result match must include " + " and ".join(sorted(missing)),
                    expression.span,
                )
            if not branch_types:
                return None
            result = branch_types[0]
            for branch_type in branch_types[1:]:
                if not _compatible(result, branch_type):
                    self.error(
                        _mismatch_code(result, branch_type),
                        "Result match arms must produce compatible types",
                        expression.span,
                    )
                    return None
            return self._record(expression, result)
        if expression.kind == "struct_literal":
            declaration = self.types.get(expression.value)
            if declaration is None:
                self.error(
                    "E_UNKNOWN_TYPE",
                    f"Unknown struct type {expression.value!r}",
                    expression.span,
                )
                for child in expression.children:
                    self._infer(child, environment)
                return self._record(expression, None)
            declared_fields = {field.name: field for field in declaration.fields}
            supplied: set[str] = set()
            for name, value_expression in zip(
                expression.field_names,
                expression.children,
            ):
                if name in supplied:
                    self.error(
                        "E_STRUCT_FIELD_DUPLICATE",
                        f"Field {name!r} is initialized more than once",
                        value_expression.span,
                    )
                supplied.add(name)
                field = declared_fields.get(name)
                value_type = self._infer(value_expression, environment)
                if field is None:
                    self.error(
                        "E_STRUCT_FIELD_UNKNOWN",
                        f"Type {declaration.name!r} has no field {name!r}",
                        value_expression.span,
                    )
                elif (
                    value_type is not None
                    and not _compatible(_resolve_type(field.type), value_type)
                ):
                    expected = _resolve_type(field.type)
                    self.error(
                        "E_STRUCT_FIELD_TYPE",
                        f"Field {name!r} expects {expected.display()}, "
                        f"got {value_type.display()}",
                        value_expression.span,
                    )
            missing = set(declared_fields) - supplied
            for name in sorted(missing):
                self.error(
                    "E_STRUCT_FIELD_MISSING",
                    f"Missing initializer for field {name!r} of {declaration.name!r}",
                    expression.span,
                )
            return self._record(expression, AetherType(declaration.name))
        if expression.kind == "member":
            base = self._infer(expression.children[0], environment)
            if expression.value == "?.":
                self.error(
                    "E_OPTIONAL_MEMBER_UNSUPPORTED",
                    "Optional member access is not supported until an option type is defined",
                    expression.span,
                )
                return None
            member = expression.children[1].value
            declaration = self.types.get(base.name) if base is not None else None
            if declaration is None:
                return self._record(expression, None)
            field = next((field for field in declaration.fields if field.name == member), None)
            if field is None:
                self.error(
                    "E_UNKNOWN_FIELD",
                    f"Type {base.name!r} has no field {member!r}",
                    expression.children[1].span,
                )
                return None
            return self._record(expression, _resolve_type(field.type))
        if expression.kind == "index":
            base = self._infer(expression.children[0], environment)
            index_type = self._infer(expression.children[1], environment)
            if index_type is not None and index_type.name not in _INTEGER_TYPES:
                self.error(
                    "E_INDEX_TYPE",
                    "Collection index must be an integer",
                    expression.children[1].span,
                )
            if base is not None and base.name == "Tensor" and base.arguments:
                dimensions = base.dimensions[1:] if len(base.dimensions) > 1 else ()
                element = base.arguments[0]
                result = (
                    AetherType(
                        "Tensor",
                        (element,),
                        dimensions,
                        base.unit,
                        base.unit_scale,
                        base.unit_offset,
                        base.unit_symbol,
                        base.unit_kind,
                        base.unit_pi_power,
                    )
                    if dimensions
                    else AetherType(
                        element.name,
                        element.arguments,
                        unit=base.unit,
                        unit_scale=base.unit_scale,
                        unit_offset=base.unit_offset,
                        unit_symbol=base.unit_symbol,
                        unit_kind=base.unit_kind,
                        unit_pi_power=base.unit_pi_power,
                    )
                )
                return self._record(expression, result)
            if base is not None and base.name in {"Array", "List", "Slice"}:
                self.error(
                    "E_COLLECTION_INDEX",
                    "Use get(collection, index) to receive a checked Result",
                    expression.span,
                )
            return self._record(expression, None)
        if expression.kind == "unary":
            operand = self._infer(expression.children[0], environment)
            is_numeric_tensor = (
                operand is not None
                and operand.name == "Tensor"
                and operand.element_type is not None
                and operand.element_type.is_numeric
            )
            if (
                operand is not None
                and expression.value in {"+", "-"}
                and not operand.is_numeric
                and not is_numeric_tensor
            ):
                self.error("E_TYPE_OPERATOR", f"Unary {expression.value} requires a numeric value", expression.span)
                return None
            if operand is not None and expression.value == "!" and operand.name != "bool":
                self.error("E_TYPE_OPERATOR", "Unary ! requires bool", expression.span)
                return None
            if (
                operand is not None
                and expression.value == "~"
                and not _is_range_integer(operand)
            ):
                self.error(
                    "E_BITWISE_TYPE",
                    "Unary ~ requires a unitless integer",
                    expression.span,
                )
                return None
            return self._record(expression, AetherType("bool") if expression.value == "!" else operand)
        if expression.kind == "binary":
            if expression.value == "|>":
                left = self._infer(expression.children[0], environment)
                if left is None:
                    return self._record(expression, None)
                right_expression = expression.children[1]
                if right_expression.kind == "call":
                    value = self._infer_call(
                        right_expression,
                        environment,
                        pipeline_input=left,
                    )
                elif right_expression.kind == "identifier":
                    call = Expr(
                        "call",
                        right_expression.span,
                        children=(right_expression,),
                    )
                    value = self._infer_call(call, environment, pipeline_input=left)
                else:
                    value = self._infer(right_expression, environment, left)
                return self._record(expression, value)
            left = self._infer(expression.children[0], environment)
            right = self._infer(expression.children[1], environment)
            return self._record(
                expression,
                self._infer_binary(expression, left, right),
            )
        if expression.kind == "call":
            return self._record(
                expression,
                self._infer_call(
                    expression,
                    environment,
                    pipeline_input,
                    expected_type=expected_type,
                ),
            )
        return None

    def _infer_binary(
        self,
        expression: Expr,
        left: AetherType | None,
        right: AetherType | None,
    ) -> AetherType | None:
        if left is None or right is None:
            return None
        operator = expression.value
        if operator in {"&", "|", "^", "<<", ">>"}:
            if (
                not _is_range_integer(left)
                or not _is_range_integer(right)
                or left.name != right.name
            ):
                self.error(
                    "E_BITWISE_TYPE",
                    f"Operator {operator} requires matching unitless integer operands",
                    expression.span,
                )
                return None
            return left
        if operator in {"&&", "||"}:
            if left.name != "bool" or right.name != "bool":
                self.error(
                    "E_TYPE_OPERATOR",
                    f"Operator {operator} requires bool operands",
                    expression.span,
                )
                return None
            return AetherType("bool")
        if operator in {"==", "!=", "<", ">", "<=", ">="}:
            if left.name == "Tensor" or right.name == "Tensor":
                self.error(
                    "E_TENSOR_COMPARISON",
                    "Tensor comparisons are not supported until boolean tensor types are defined",
                    expression.span,
                )
                return None
            if not _compatible(left, right):
                self.error(
                    _mismatch_code(left, right),
                    f"Cannot compare {left.display()} with {right.display()}",
                    expression.span,
                )
            return AetherType("bool")
        if operator in {"+", "-"}:
            if left.unit_kind == "absolute_temperature" or right.unit_kind == "absolute_temperature":
                self.error(
                    "E_TEMPERATURE_ARITHMETIC",
                    "Arithmetic on absolute temperatures requires an explicit conversion or delta type",
                    expression.span,
                )
                return None
            valid_left = left.is_numeric or left.name == "Tensor"
            valid_right = right.is_numeric or right.name == "Tensor"
            if not valid_left or not valid_right:
                self.error(
                    "E_TYPE_OPERATOR",
                    f"Operator {operator} requires numeric or tensor operands",
                    expression.span,
                )
                return None
            if left.name == "Tensor" and right.is_numeric:
                if (
                    left.arguments[0].name == right.name
                    and _same_unit(left, right)
                ):
                    return left
            if right.name == "Tensor" and left.is_numeric:
                if (
                    right.arguments[0].name == left.name
                    and _same_unit(right, left)
                ):
                    return right
            if not _compatible(left, right):
                self.error(
                    _mismatch_code(left, right),
                    f"Operator {operator} cannot combine {left.display()} and {right.display()}",
                    expression.span,
                )
                return None
            if left.name == "Tensor":
                return left
            return _numeric_result(left, right)
        if operator in {"*", "/"}:
            if (
                left.unit_kind == "absolute_temperature"
                or right.unit_kind == "absolute_temperature"
            ):
                self.error(
                    "E_TEMPERATURE_ARITHMETIC",
                    "Absolute temperatures cannot be multiplied or divided",
                    expression.span,
                )
                return None
            if left.name == "Tensor" and right.is_numeric:
                if left.arguments[0].name != right.name:
                    self.error(
                        "E_TYPE_OPERATOR",
                        "Tensor and scalar element types must match; convert explicitly",
                        expression.span,
                    )
                    return None
                unit, scale, pi_power = _combine_unit_types(
                    left,
                    right,
                    1 if operator == "*" else -1,
                )
                return AetherType(
                    "Tensor",
                    left.arguments,
                    left.dimensions,
                    unit,
                    scale,
                    unit_pi_power=pi_power,
                )
            if right.name == "Tensor" and left.is_numeric and operator == "*":
                if right.arguments[0].name != left.name:
                    self.error(
                        "E_TYPE_OPERATOR",
                        "Tensor and scalar element types must match; convert explicitly",
                        expression.span,
                    )
                    return None
                unit, scale, pi_power = _combine_unit_types(right, left, 1)
                return AetherType(
                    "Tensor",
                    right.arguments,
                    right.dimensions,
                    unit,
                    scale,
                    unit_pi_power=pi_power,
                )
            if left.is_numeric and right.is_numeric:
                if left.name != right.name:
                    self.error(
                        "E_TYPE_OPERATOR",
                        f"Operator {operator} requires matching numeric types, got "
                        f"{left.name} and {right.name}",
                        expression.span,
                    )
                    return None
                unit, scale, pi_power = _combine_unit_types(
                    left,
                    right,
                    1 if operator == "*" else -1,
                )
                return AetherType(
                    left.name,
                    unit=unit,
                    unit_scale=scale,
                    unit_pi_power=pi_power,
                )
        if operator == "%" and left.is_numeric and right.is_numeric:
            if not _compatible(left, right):
                self.error(
                    _mismatch_code(left, right),
                    "Modulo operands must have identical types and units",
                    expression.span,
                )
                return None
            if left.unit_kind == "absolute_temperature":
                self.error(
                    "E_TEMPERATURE_ARITHMETIC",
                    "Modulo is not defined for absolute temperatures",
                    expression.span,
                )
                return None
            return left
        if operator == "**" and left.is_numeric and right.is_numeric:
            if right.unit or right.unit_kind or left.unit or left.unit_kind:
                self.error(
                    "E_UNIT_POWER",
                    "Exponentiation of unitful values is not yet supported",
                    expression.span,
                )
                return None
            if left.name != right.name:
                self.error(
                    "E_TYPE_OPERATOR",
                    "Exponentiation operands must have matching numeric types",
                    expression.span,
                )
                return None
            return left
        self.error(
            "E_TYPE_OPERATOR",
            f"Operator {operator} is not defined for {left.display()} and {right.display()}",
            expression.span,
        )
        return None

    def _infer_call(
        self,
        expression: Expr,
        environment: dict[str, AetherType],
        pipeline_input: AetherType | None = None,
        *,
        expected_type: AetherType | None = None,
    ) -> AetherType | None:
        callee = expression.children[0]
        if callee.kind != "identifier":
            self._infer(callee, environment)
            return None
        name = callee.value
        if name == "gradient":
            return self._infer_gradient(expression, environment, pipeline_input)
        if name == "List":
            if pipeline_input is not None or len(expression.children) not in {1, 2}:
                self.error(
                    "E_LIST_CONSTRUCTOR",
                    "List expects no arguments or one collection literal",
                    expression.span,
                )
                return None
            if len(expression.children) == 1:
                if expected_type is None or expected_type.name != "List":
                    self.error(
                        "E_LIST_INFERENCE",
                        "Empty List() requires an explicit List<T> annotation",
                        expression.span,
                    )
                    return None
                return self._record(expression, expected_type)
            source_type = self._infer(expression.children[1], environment)
            if source_type is None:
                return None
            if source_type.name not in {"Array", "List", "Slice"}:
                return self._builtin_type_error(
                    name,
                    "expects a collection literal or collection value",
                    expression.children[1].span,
                )
            result = AetherType("List", source_type.arguments)
            if expected_type is not None and not _compatible(expected_type, result):
                self.error(
                    "E_LIST_CONSTRUCTOR_TYPE",
                    f"List constructor produces {result.display()}, expected "
                    f"{expected_type.display()}",
                    expression.span,
                )
            return self._record(expression, result)
        if name in {"get", "set", "slice", "append", "pop", "insert", "remove"}:
            return self._infer_collection_call(
                name,
                expression,
                environment,
                pipeline_input,
            )
        if name == "frame_buffer":
            if pipeline_input is not None or len(expression.children) != 2:
                self.error(
                    "E_BUILTIN_ARITY",
                    "frame_buffer expects exactly one byte count",
                    expression.span,
                )
                return None
            size_type = self._infer(expression.children[1], environment)
            if size_type is not None and size_type.name != "u64":
                self.error(
                    "E_BUILTIN_TYPE",
                    "frame_buffer byte count must have type u64",
                    expression.children[1].span,
                )
                return None
            if size_type is None:
                return None
            return self._record(expression, AetherType("FrameBuffer"))
        if name == "frame_buffer_len":
            if pipeline_input is not None or len(expression.children) != 2:
                self.error(
                    "E_BUILTIN_ARITY",
                    "frame_buffer_len expects exactly one local FrameBuffer",
                    expression.span,
                )
                return None
            buffer_type = self._infer(expression.children[1], environment)
            if buffer_type is not None and buffer_type.name != "FrameBuffer":
                return self._builtin_type_error(
                    name,
                    "expects a local FrameBuffer",
                    expression.children[1].span,
                )
            if buffer_type is None:
                return None
            return self._record(expression, AetherType("u64"))
        if name in {"Ok", "Err"}:
            if pipeline_input is not None or len(expression.children) != 2:
                self.error(
                    "E_RESULT_CONSTRUCTOR_ARITY",
                    f"{name} expects exactly one payload",
                    expression.span,
                )
                return None
            expected = self.current_return_type
            if expected is None or expected.name != "Result" or len(expected.arguments) != 2:
                self.error(
                    "E_RESULT_CONSTRUCTOR_CONTEXT",
                    f"{name} must be returned from a function with an annotated Result type",
                    expression.span,
                )
                return None
            if name == "Err":
                marker = expression.children[1]
                if marker.kind != "identifier" or marker.value not in {
                    "IoError",
                    "FsError",
                    "NetError",
                    "ArgsError",
                    "CollectionError",
                }:
                    self.error(
                        "E_RESULT_CONSTRUCTOR_TYPE",
                        "Err payload must be an error category marker",
                        marker.span,
                    )
                    return None
                payload = self._record(marker, AetherType(marker.value))
            else:
                payload = self._infer(expression.children[1], environment)
            if payload is None:
                return None
            if name == "Ok":
                if not _compatible(expected.arguments[0], payload):
                    self.error(
                        "E_RESULT_CONSTRUCTOR_TYPE",
                        f"Ok payload must have type {expected.arguments[0].display()}",
                        expression.span,
                    )
                    return None
            elif payload != expected.arguments[1]:
                self.error(
                    "E_RESULT_CONSTRUCTOR_TYPE",
                    f"Err payload must be {expected.arguments[1].display()}",
                    expression.span,
                )
                return None
            return self._record(expression, expected)
        if name == "convert":
            raw_arguments = list(expression.children[1:])
            expected_arity = 2 if pipeline_input is None else 1
            if len(raw_arguments) != expected_arity:
                self.error(
                    "E_BUILTIN_ARITY",
                    "convert expects a value and unit string"
                    if pipeline_input is None
                    else "Pipeline convert expects one unit string",
                    expression.span,
                )
                return None
            if pipeline_input is not None:
                value_type = pipeline_input
                unit_expression = raw_arguments[0]
            else:
                value_type = self._infer(raw_arguments[0], environment)
                unit_expression = raw_arguments[1]
            if value_type is None or unit_expression is None:
                return None
            if unit_expression.kind != "literal" or not unit_expression.value.startswith(("'", '"')):
                self.error(
                    "E_BUILTIN_TYPE",
                    "convert unit must be a quoted unit name",
                    unit_expression.span,
                )
                return None
            target_name = unit_expression.value[1:-1]
            try:
                target_expression = parse_unit_expression_text(target_name)
                target = _unit_expression_definition(target_expression)
            except ValueError as error:
                self.error(
                    "E_UNKNOWN_UNIT"
                    if "Unknown or unsupported unit" in str(error)
                    else "E_UNIT_EXPRESSION",
                    str(error),
                    unit_expression.span,
                )
                return None
            if value_type.name != "Tensor" and not value_type.is_numeric:
                self.error(
                    "E_BUILTIN_TYPE",
                    "convert requires a numeric scalar or tensor",
                    expression.span,
                )
                return None
            if value_type.unit != target.dimensions:
                self.error(
                    "E_UNIT_MISMATCH",
                    f"Cannot convert {value_type.display()} to {target_name}",
                    expression.span,
                )
                return None
            if (
                value_type.unit_kind != target.kind
                and (value_type.unit_kind or target.kind)
            ):
                self.error(
                    "E_UNIT_MISMATCH",
                    "Conversion between absolute and non-absolute units is not supported",
                    expression.span,
                )
                return None
            self.conversions.append(
                UnitConversion(
                    expression.span,
                    value_type.unit_scale / target.scale,
                    (value_type.unit_offset - target.offset) / target.scale,
                    value_type.unit_pi_power - target.pi_power,
                    target_name,
                )
            )
            return _with_unit(value_type, target_name, target)

        arguments = [
            self._infer(argument, environment)
            for argument in expression.children[1:]
        ]
        if pipeline_input is not None:
            arguments.insert(0, pipeline_input)
        if any(argument is None for argument in arguments):
            return None
        concrete = [argument for argument in arguments if argument is not None]

        function = self.functions.get(name)
        if function is not None:
            if len(concrete) != len(function.parameters):
                self.error(
                    "E_CALL_ARITY",
                    f"Function {name!r} expects {len(function.parameters)} arguments, got {len(concrete)}",
                    expression.span,
                )
            for parameter_index, (parameter, argument) in enumerate(
                zip(function.parameters, concrete)
            ):
                expected = _resolve_type(parameter.type)
                ffi_slice = (
                    function.external_abi is not None
                    and expected.name in {"Slice", "SliceMut"}
                )
                compatible = (
                    self._ffi_slice_compatible(expected, argument)
                    if ffi_slice
                    else _compatible(expected, argument)
                )
                if not compatible:
                    self.error(
                        "E_ARGUMENT_TYPE",
                        f"Argument {parameter.name!r} expects {expected.display()}, got {argument.display()}",
                        expression.span,
                    )
                elif function.external_abi is not None and expected.name == "SliceMut":
                    argument_expression = (
                        expression.children[parameter_index + 1]
                        if pipeline_input is None
                        and len(expression.children) > parameter_index + 1
                        else None
                    )
                    if (
                        argument_expression is None
                        or argument_expression.kind != "identifier"
                        or argument_expression.value
                        not in self.current_mutable_names
                    ):
                        self.error(
                            "E_FFI_MUTABILITY",
                            "SliceMut<u8> requires a mutable local tensor binding",
                            expression.span,
                        )
            return _resolve_type(function.return_type) if function.return_type else None

        operation = _BUILTIN_OPERATIONS.get(name)
        if operation is not None:
            if len(concrete) != len(operation.parameters):
                return self._builtin_arity_error(
                    name,
                    len(operation.parameters),
                    expression.span,
                )
            for index, (expected, actual) in enumerate(
                zip(operation.parameters, concrete),
                start=1,
            ):
                if not _compatible(expected, actual):
                    self.error(
                        "E_ARGUMENT_TYPE",
                        f"Argument {index} to {name} expects "
                        f"{expected.display()}, got {actual.display()}",
                        expression.span,
                    )
            return operation.result

        if name in {"mean", "max_abs", "sum"}:
            if len(concrete) != 1:
                return self._builtin_arity_error(name, 1, expression.span)
            argument = concrete[0]
            if argument.name != "Tensor" or not argument.arguments:
                return self._builtin_type_error(
                    name,
                    "expects a tensor argument",
                    expression.span,
                )

            element = argument.arguments[0]
            if name == "mean" and element.name not in _FLOAT_TYPES:
                element = AetherType("f32")
            return AetherType(
                element.name,
                element.arguments,
                unit=argument.unit,
                unit_scale=argument.unit_scale,
                unit_offset=argument.unit_offset,
                unit_symbol=argument.unit_symbol,
                unit_kind=argument.unit_kind,
                unit_pi_power=argument.unit_pi_power,
            )
        if name in {"scale", "subtract"}:
            if len(concrete) != 2:
                return self._builtin_arity_error(name, 2, expression.span)
            return self._infer_binary(
                Expr("binary", expression.span, "*" if name == "scale" else "-", ()),
                concrete[0],
                concrete[1],
            )
        if name == "abs":
            if len(concrete) != 1:
                return self._builtin_arity_error(name, 1, expression.span)
            if concrete[0].name != "Tensor" and not concrete[0].is_numeric:
                return self._builtin_type_error(
                    name,
                    "expects a numeric scalar or tensor",
                    expression.span,
                )
            return concrete[0]
        if name in {"min", "max"}:
            if len(concrete) != 2:
                return self._builtin_arity_error(name, 2, expression.span)
            if not _compatible(concrete[0], concrete[1]):
                self.error(
                    _mismatch_code(concrete[0], concrete[1]),
                    f"{name} arguments must have matching types, shapes, and units",
                    expression.span,
                )
                return None

            if concrete[0].name != "Tensor" and not concrete[0].is_numeric:
                return self._builtin_type_error(
                    name,
                    "expects numeric or tensor arguments",
                    expression.span,
                )
            return concrete[0]
        if name == "len":
            if len(concrete) != 1:
                return self._builtin_arity_error(name, 1, expression.span)
            if concrete[0].name not in {"Tensor", "Array", "List", "Slice"}:
                return self._builtin_type_error(
                    name,
                    "expects a Tensor, Array, List, or Slice",
                    expression.span,
                )
            return AetherType("i64")
        return None

    @staticmethod
    def _ffi_slice_compatible(expected: AetherType, actual: AetherType) -> bool:
        return (
            expected.name in {"Slice", "SliceMut"}
            and len(expected.arguments) == 1
            and expected.arguments[0].name == "u8"
            and actual.name == "Tensor"
            and actual.element_type is not None
            and actual.element_type.name == "u8"
            and len(actual.dimensions) == 1
            and not (
                actual.unit
                or actual.unit_scale != 1
                or actual.unit_offset
                or actual.unit_symbol
                or actual.unit_kind
                or actual.unit_pi_power
            )
        )

    def _infer_collection_call(
        self,
        name: str,
        expression: Expr,
        environment: dict[str, AetherType],
        pipeline_input: AetherType | None,
    ) -> AetherType | None:
        if pipeline_input is not None:
            self.error(
                "E_COLLECTION_PIPELINE",
                "Collection operations must be called with explicit arguments",
                expression.span,
            )
            return None
        raw_arguments = list(expression.children[1:])
        expected_arity = {
            "get": 2,
            "set": 3,
            "slice": 3,
            "append": 2,
            "pop": 1,
            "insert": 3,
            "remove": 2,
        }[name]
        if len(raw_arguments) != expected_arity:
            return self._builtin_arity_error(
                name,
                expected_arity,
                expression.span,
            )
        argument_types = [
            self._infer(argument, environment)
            for argument in raw_arguments
        ]
        if any(value is None for value in argument_types):
            return None
        concrete = [value for value in argument_types if value is not None]
        collection_type = concrete[0]
        if collection_type.name not in {"Array", "List", "Slice"} or not collection_type.arguments:
            return self._builtin_type_error(
                name,
                "expects an Array, List, or Slice as its first argument",
                raw_arguments[0].span,
            )
        element_type = collection_type.arguments[0]
        index_positions = {
            "get": (1,),
            "set": (1,),
            "slice": (1, 2),
            "insert": (1,),
            "remove": (1,),
        }.get(name, ())
        for position in index_positions:
            if concrete[position].name not in _INTEGER_TYPES:
                self.error(
                    "E_COLLECTION_INDEX_TYPE",
                    f"{name} index and bounds must be integers",
                    raw_arguments[position].span,
                )
                return None
        if name in {"set", "append", "pop", "insert", "remove"}:
            if collection_type.name != "List" and name != "set":
                return self._builtin_type_error(
                    name,
                    "mutates only a List",
                    raw_arguments[0].span,
                )
            target = raw_arguments[0]
            if target.kind != "identifier" or target.value not in self.current_mutable_names:
                self.error(
                    "E_COLLECTION_MUTABILITY",
                    f"{name} requires a mutable local collection declared with 'var'",
                    target.span,
                )
                return None
        if name in {"set", "append"}:
            value_position = 2 if name == "set" else 1
            if not _compatible(element_type, concrete[value_position]):
                self.error(
                    "E_COLLECTION_ELEMENT_TYPE",
                    f"{name} value expects {element_type.display()}, got "
                    f"{concrete[value_position].display()}",
                    raw_arguments[value_position].span,
                )
                return None
        elif name == "insert":
            if not _compatible(element_type, concrete[2]):
                self.error(
                    "E_COLLECTION_ELEMENT_TYPE",
                    f"insert value expects {element_type.display()}, got "
                    f"{concrete[2].display()}",
                    raw_arguments[2].span,
                )
                return None
        if name == "slice":
            result = AetherType("Slice", (element_type,))
        elif name in {"get", "pop", "remove"}:
            result = element_type
        else:
            result = AetherType("void")
        return self._record(
            expression,
            _result_type(result, "CollectionError"),
        )

    def _infer_gradient(
        self,
        expression: Expr,
        environment: dict[str, AetherType],
        pipeline_input: AetherType | None,
    ) -> AetherType | None:
        if pipeline_input is not None:
            self.error(
                "E_GRADIENT_ARITY",
                "gradient must be called as gradient(function, value)",
                expression.span,
            )
            return None
        arguments = expression.children[1:]
        if len(arguments) != 2:
            self.error(
                "E_GRADIENT_ARITY",
                "gradient expects a function name and one scalar input",
                expression.span,
            )
            return None
        target_expression, input_expression = arguments
        if target_expression.kind != "identifier":
            self.error(
                "E_GRADIENT_TARGET",
                "gradient target must be a named function",
                target_expression.span,
            )
            return None
        target = self.functions.get(target_expression.value)
        if target is None:
            self.error(
                "E_GRADIENT_TARGET",
                f"Unknown gradient target {target_expression.value!r}",
                target_expression.span,
            )
            return None
        input_type = self._infer(input_expression, environment)
        if input_type is None:
            return None
        if target.external_abi is not None:
            self.error(
                "E_GRADIENT_EFFECT",
                "Extern functions cannot be differentiated",
                target.span,
            )
            return None
        if len(target.parameters) != 1:
            self.error(
                "E_GRADIENT_SIGNATURE",
                "Autodiff v0 requires a unary target function",
                target.span,
            )
            return None
        parameter_type = _resolve_type(target.parameters[0].type)
        return_type = (
            _resolve_type(target.return_type)
            if target.return_type is not None
            else None
        )
        is_scalar_gradient = (
            _is_unitless_float(parameter_type)
            and _is_unitless_float(input_type)
            and parameter_type == input_type
        )
        is_tensor_gradient = (
            parameter_type.name == "Tensor"
            and input_type.name == "Tensor"
            and parameter_type.arguments == input_type.arguments
            and parameter_type.dimensions == input_type.dimensions
            and _same_unit(parameter_type, input_type)
            and parameter_type.element_type is not None
            and parameter_type.element_type.name in {"f32", "f64"}
        )
        if not (is_scalar_gradient or is_tensor_gradient):
            self.error(
                "E_GRADIENT_TYPE",
                "Autodiff requires an input matching the target's f32/f64 "
                "scalar or static tensor parameter type",
                expression.span,
            )
            return None
        if return_type is None or return_type.name not in {"f32", "f64"}:
            self.error(
                "E_GRADIENT_TYPE",
                "Autodiff requires an explicitly annotated f32 or f64 scalar result",
                target.span,
            )
            return None
        if is_scalar_gradient and not _is_unitless_float(return_type):
            self.error(
                "E_GRADIENT_TYPE",
                "Scalar autodiff v0 requires a unitless result",
                target.span,
            )
            return None
        if is_tensor_gradient and return_type.name != parameter_type.element_type.name:
            self.error(
                "E_GRADIENT_UNIT",
                "Tensor autodiff requires matching f32/f64 element types and "
                "compatible differentiable input and output units",
                target.span,
            )
            return None
        summary = self.effect_summaries.get(target.name)
        if summary is None or summary.required or summary.declared:
            self.error(
                "E_GRADIENT_EFFECT",
                "Autodiff v0 supports only pure functions without declared effects",
                target.span,
            )
            return None
        if (
            len(target.body) < 1
            or not isinstance(target.body[-1], ReturnStmt)
            or target.body[-1].value is None
            or any(not isinstance(statement, LetStmt) for statement in target.body[:-1])
        ):
            self.error(
                "E_GRADIENT_CONTROL_FLOW",
                "Autodiff v0 supports only local bindings followed by one final return",
                target.span,
            )
            return None
        for statement in target.body:
            value = (
                statement.value
                if isinstance(statement, (LetStmt, ReturnStmt))
                else None
            )
            if value is not None and not _is_differentiable_expression(value):
                self.error(
                    "E_GRADIENT_OPERATION",
                    "Autodiff supports literals, names, unary +/- and binary "
                    "+, -, *, /; tensor objectives also support sum and mean",
                    value.span,
                )
                return None
        if is_scalar_gradient:
            return self._record(expression, parameter_type)
        assert parameter_type.element_type is not None
        unit = _combine_units(return_type.unit, parameter_type.unit, -1)
        scale = return_type.unit_scale / parameter_type.unit_scale
        pi_power = return_type.unit_pi_power - parameter_type.unit_pi_power
        output_symbol = return_type.unit_symbol or "1"
        input_symbol = parameter_type.unit_symbol or "1"
        unit_symbol = (
            ""
            if scale == 1 and pi_power == 0 and not unit
            else f"{output_symbol}/{input_symbol}"
        )
        return self._record(
            expression,
            AetherType(
                "Tensor",
                (parameter_type.element_type,),
                parameter_type.dimensions,
                unit,
                scale,
                unit_symbol=unit_symbol,
                unit_pi_power=pi_power,
            ),
        )

    def _builtin_arity_error(
        self,
        name: str,
        expected: int,
        span: SourceSpan,
    ) -> None:
        self.error(
            "E_BUILTIN_ARITY",
            f"{name} expects {expected} argument(s)",
            span,
        )
        return None

    def _builtin_type_error(
        self,
        name: str,
        message: str,
        span: SourceSpan,
    ) -> None:
        self.error("E_BUILTIN_TYPE", f"{name} {message}", span)
        return None


def _numeric_result(left: AetherType, right: AetherType) -> AetherType:
    order = {"i4": 0, "t2": 0, "i8": 1, "u8": 1, "i16": 2, "u16": 2,
             "f16": 3, "bf16": 3, "i32": 4, "u32": 4, "f32": 5,
             "i64": 6, "u64": 6, "f64": 7, "i128": 8, "u128": 8, "fp8": 3}
    name = max((left.name, right.name), key=lambda item: order.get(item, 0))
    return AetherType(
        name,
        unit=left.unit,
        unit_scale=left.unit_scale,
        unit_offset=left.unit_offset,
        unit_symbol=left.unit_symbol,
        unit_kind=left.unit_kind,
        unit_pi_power=left.unit_pi_power,
    )


def _is_unitless_float(type_: AetherType) -> bool:
    return type_.name in {"f32", "f64"} and not (
        type_.unit
        or type_.unit_scale != 1
        or type_.unit_offset
        or type_.unit_symbol
        or type_.unit_kind
        or type_.unit_pi_power
    )


def _is_differentiable_expression(expression: Expr) -> bool:
    if expression.kind == "literal":
        return bool(
            re.fullmatch(
                r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
                expression.value,
            )
        )
    if expression.kind == "identifier":
        return True
    if expression.kind == "group":
        return len(expression.children) == 1 and _is_differentiable_expression(
            expression.children[0]
        )
    if expression.kind == "unary":
        return (
            expression.value in {"+", "-"}
            and len(expression.children) == 1
            and _is_differentiable_expression(expression.children[0])
        )
    if expression.kind == "binary":
        if expression.value == "|>" and len(expression.children) == 2:
            left, right = expression.children
            if right.kind == "call" and right.children:
                callee = right.children[0]
                arguments = (left, *right.children[1:])
            else:
                callee = right
                arguments = (left,)
            if callee.kind != "identifier" or callee.value not in {
                "convert",
                "mean",
                "sum",
                "scale",
                "subtract",
            }:
                return False
            return all(
                _is_differentiable_expression(argument)
                for index, argument in enumerate(arguments)
                if not (
                    callee.value == "convert"
                    and index == len(arguments) - 1
                    and argument.kind == "literal"
                    and argument.value.startswith(("'", '"'))
                )
            )
        return (
            expression.value in {"+", "-", "*", "/"}
            and len(expression.children) == 2
            and all(_is_differentiable_expression(child) for child in expression.children)
        )
    if expression.kind == "call":
        if not expression.children or expression.children[0].kind != "identifier":
            return False
        callee = expression.children[0].value
        if callee in {"sum", "mean"}:
            return (
                len(expression.children) == 2
                and _is_differentiable_expression(expression.children[1])
            )
        if callee in {"scale", "subtract"}:
            return len(expression.children) == 3 and all(
                _is_differentiable_expression(argument)
                for argument in expression.children[1:]
            )
        return (
            callee == "convert"
            and len(expression.children) == 3
            and _is_differentiable_expression(expression.children[1])
            and expression.children[2].kind == "literal"
            and expression.children[2].value.startswith(("'", '"'))
        )
    return False


def _consumed_result_names(expression: Expr, context: str) -> set[str]:
    """Return only bindings consumed in supported Result-handling positions."""
    def direct_name(value: Expr) -> str | None:
        while value.kind == "group" and value.children:
            value = value.children[0]
        return value.value if value.kind == "identifier" else None

    def guaranteed(value: Expr) -> set[str]:
        consumed: set[str] = set()
        if value.kind == "try" and value.children:
            name = direct_name(value.children[0])
            if name is not None:
                consumed.add(name)
            consumed.update(guaranteed(value.children[0]))
            return consumed
        if value.kind == "match" and value.children:
            name = direct_name(value.children[0])
            if name is not None:
                consumed.add(name)
            consumed.update(guaranteed(value.children[0]))
            if value.arms:
                arm_consumption = [guaranteed(arm.value) for arm in value.arms]
                consumed.update(set.intersection(*arm_consumption))
            return consumed
        if value.kind == "binary" and value.value in {"&&", "||"}:
            return guaranteed(value.children[0])
        for child in value.children:
            consumed.update(guaranteed(child))
        return consumed

    consumed = guaranteed(expression)
    if context in {"return", "bind", "expression"}:
        if context in {"return", "bind"}:
            name = direct_name(expression)
            if name is not None:
                consumed.add(name)
    return consumed


def _mismatch_code(left: AetherType, right: AetherType) -> str:
    if left.name == "Tensor" and right.name == "Tensor":
        if left.dimensions != right.dimensions:
            return "E_SHAPE_MISMATCH"
        if left.arguments != right.arguments:
            return "E_TYPE_MISMATCH"
    if left.unit != right.unit or not _same_unit(left, right):
        return "E_UNIT_MISMATCH"
    return "E_TYPE_MISMATCH"


def _statement_expression(statement):
    if isinstance(statement, (LetStmt, AssignmentStmt)):
        return statement.value
    if isinstance(statement, ReturnStmt):
        return statement.value
    if isinstance(statement, ExprStmt):
        return statement.expression
    return None


def _walk(expression: Expr):
    yield expression
    for child in expression.children:
        yield from _walk(child)
    for arm in expression.arms:
        yield from _walk(arm.value)


def analyze(module: Module) -> SemanticResult:
    """Analyze the supported subset without executing code or loading AI models."""
    return _Analyzer(module).analyze()


def resolve_type(type_expr: TypeExpr) -> AetherType:
    """Resolve a syntax-level type using the prototype's built-in unit table."""
    return _resolve_type(type_expr)
