"""Deterministic C99 emission for bounded scalar and floating tensor subsets."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math
from pathlib import Path

from src.frontend import Diagnostic, SourceSpan
from src.ir import IRExpression, IRFunction, IRModule
from src.semantic import AetherType


_C_TYPES = {
    "void": "void",
    "bool": "_Bool",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "f32": "float",
    "f64": "double",
}
_INTEGER_TYPES = {
    "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64",
}
_SIGNED_TYPES = {"i8", "i16", "i32", "i64"}
_FLOAT_LIMITS = {
    "f32": Decimal("3.40282346638528859811704183484516925440e38"),
    "f64": Decimal("1.797693134862315708145274237317043567981e308"),
}
_INTEGER_LIMITS = {
    "i8": (-(1 << 7), (1 << 7) - 1),
    "i16": (-(1 << 15), (1 << 15) - 1),
    "i32": (-(1 << 31), (1 << 31) - 1),
    "i64": (-(1 << 63), (1 << 63) - 1),
    "u8": (0, (1 << 8) - 1),
    "u16": (0, (1 << 16) - 1),
    "u32": (0, (1 << 32) - 1),
    "u64": (0, (1 << 64) - 1),
}
_BINARY_OPERATORS = {
    "+", "-", "*", "/", "%", "==", "!=", "<", ">", "<=", ">=",
}
_C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "inline", "int", "long", "register", "restrict", "return", "short",
    "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while", "_Alignas", "_Alignof",
    "_Atomic", "_Bool", "_Complex", "_Generic", "_Imaginary", "_Noreturn",
    "_Static_assert", "_Thread_local",
}


class C99LoweringError(ValueError):
    """Raised when an IR module uses features outside the safe C99 subset."""

    def __init__(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        summary = "; ".join(
            f"{item.code}: {item.message}" for item in diagnostics
        )
        super().__init__(summary or "C99 lowering failed")


@dataclass(frozen=True)
class _CValue:
    prelude: tuple[str, ...]
    code: str
    type: AetherType
    ownership: str = "plain"
    cleanup: tuple[str, ...] = ()
    temporary: bool = False


def _encoded_identifier(prefix: str, name: str) -> str:
    return prefix + name.encode("utf-8").hex()


def _has_unit(type_: AetherType) -> bool:
    return bool(
        type_.unit
        or type_.unit_scale != 1
        or type_.unit_offset
        or type_.unit_symbol
        or type_.unit_kind
        or type_.unit_pi_power
    )


class _C99Emitter:
    def __init__(
        self,
        module: IRModule,
        *,
        include_entry_point: bool = False,
        defined_functions: tuple[IRFunction, ...] | None = None,
    ) -> None:
        self.module = module
        self.include_entry_point = include_entry_point
        self.functions = {function.name: function for function in module.functions}
        self.defined_functions = (
            module.functions if defined_functions is None else defined_functions
        )
        self.defined_function_names = {
            function.name for function in self.defined_functions
        }
        self.diagnostics: list[Diagnostic] = []
        self.helpers: set[tuple[str, str]] = set()
        self.needs_trap = False
        self.needs_frame_buffer = False
        self.temp_index = 0
        self.gradient_targets: set[str] = set()
        self.tensor_types: dict[str, AetherType] = {}
        self.result_types: dict[str, AetherType] = {}
        self.current_function: IRFunction | None = None
        self.frame_return_name = ""
        self.frame_cleanup_label = ""
        self.active_expression_cleanups: tuple[str, ...] = ()
        for function in module.functions:
            self._collect_type(function.return_type)
            self._collect_tensor_type(function.return_type)
            for parameter in function.parameters:
                self._collect_type(parameter.type)
                self._collect_tensor_type(parameter.type)
            for statement in function.body:
                if statement.expression is not None:
                    self._collect_expression_types(statement.expression)
                    self._collect_expression_tensor_types(statement.expression)

    def error(self, code: str, message: str, span: SourceSpan) -> None:
        self.diagnostics.append(Diagnostic(code, message, span))

    @staticmethod
    def _tensor_c_name(type_: AetherType) -> str:
        dimensions = "x".join(str(dimension) for dimension in type_.dimensions)
        element = type_.element_type
        if element is None:
            return "aether_tensor_invalid"
        return f"aether_tensor_{element.name}_{dimensions}"

    def _collect_tensor_type(self, type_: AetherType) -> None:
        if type_.name == "Tensor":
            self.tensor_types.setdefault(self._tensor_c_name(type_), type_)
        for argument in type_.arguments:
            self._collect_tensor_type(argument)

    @staticmethod
    def _result_c_name(type_: AetherType) -> str:
        if type_.name != "Result" or len(type_.arguments) != 2:
            return "aether_result_invalid"
        value_type, error_type = type_.arguments
        return f"aether_result_{value_type.name}_{error_type.name}"

    def _collect_type(self, type_: AetherType) -> None:
        if type_.name == "Result" and len(type_.arguments) == 2:
            self.result_types.setdefault(self._result_c_name(type_), type_)
        if type_.name == "FrameBuffer":
            self.needs_frame_buffer = True
            self.needs_trap = True
        for argument in type_.arguments:
            self._collect_type(argument)

    def _collect_expression_types(self, expression: IRExpression) -> None:
        self._collect_type(expression.type)
        for child in expression.children:
            self._collect_expression_types(child)
        for arm in expression.arms:
            self._collect_expression_types(arm.value)

    def _collect_expression_tensor_types(self, expression: IRExpression) -> None:
        self._collect_tensor_type(expression.type)
        for child in expression.children:
            self._collect_expression_tensor_types(child)
        for arm in expression.arms:
            self._collect_expression_tensor_types(arm.value)

    def c_type(self, type_: AetherType) -> str:
        if type_.name == "Tensor":
            return self._tensor_c_name(type_)
        if type_.name == "Result":
            return self._result_c_name(type_)
        if type_.name == "FrameBuffer":
            return "aether_frame_buffer"
        return _C_TYPES.get(type_.name, "void")

    @staticmethod
    def _contains_frame_buffer(type_: AetherType) -> bool:
        return type_.name == "FrameBuffer" or any(
            _C99Emitter._contains_frame_buffer(argument)
            for argument in type_.arguments
        )

    def _retain_code(self, name: str, type_: AetherType) -> str:
        if type_.name == "FrameBuffer":
            return f"aether_frame_buffer_retain(&{name});"
        if (
            type_.name == "Result"
            and len(type_.arguments) == 2
            and type_.arguments[0].name == "FrameBuffer"
        ):
            return (
                f"if ({name}.is_ok) "
                f"aether_frame_buffer_retain(&{name}.value);"
            )
        return ""

    def _release_code(
        self,
        name: str,
        type_: AetherType,
        *,
        skip_empty: bool = False,
    ) -> str:
        if type_.name == "FrameBuffer":
            release = f"aether_frame_buffer_release(&{name});"
            return (
                f"if ({name}.storage != NULL) {release}"
                if skip_empty
                else release
            )
        if (
            type_.name == "Result"
            and len(type_.arguments) == 2
            and type_.arguments[0].name == "FrameBuffer"
        ):
            return (
                f"if ({name}.is_ok) "
                f"aether_frame_buffer_release(&{name}.value);"
            )
        return ""

    def validate_type(self, type_: AetherType, span: SourceSpan, role: str) -> None:
        supported = (
            type_.name in _C_TYPES
            or (
                type_.name == "FrameBuffer"
                and role in {"expression", "parameter", "return"}
            )
        ) and not (
            type_.name == "void"
            and role not in {"return", "expression", "FFI return"}
        )
        if type_.name == "Result":
            supported = (
                len(type_.arguments) == 2
                and (
                    type_.arguments[0].name in _C_TYPES
                    or type_.arguments[0].name == "FrameBuffer"
                )
                and type_.arguments[0].name != "void"
                and not _has_unit(type_.arguments[0])
                and type_.arguments[1].name in {"IoError", "FsError", "NetError"}
                and not role.startswith("FFI")
            )
        if type_.name == "Tensor":
            element = type_.element_type
            supported = (
                element is not None
                and element.name in {"f32", "f64"}
                and bool(type_.dimensions)
                and all(dimension > 0 for dimension in type_.dimensions)
            )
        elif _has_unit(type_):
            supported = type_.name in {"f32", "f64"}
        if role.startswith("FFI") and (
            type_.name == "Tensor" or _has_unit(type_)
        ):
            supported = False
        if not supported:
            self.error(
                "E_C99_TYPE",
                f"C99 backend does not support {role} type {type_.display()}"
                + (
                    "; tensor backend v0 supports only f32/f64 elements"
                    if type_.name == "Tensor"
                    else ""
                ),
                span,
            )

    def validate_expression(
        self,
        expression: IRExpression,
        *,
        allow_signed_min_magnitude: bool = False,
    ) -> None:
        kind = expression.kind
        if expression.type.name not in {"IoError", "FsError", "NetError"}:
            self.validate_type(expression.type, expression.span, "expression")
        if kind == "literal":
            self._validate_literal(expression, allow_signed_min_magnitude)
            return
        if kind == "identifier":
            if expression.value in {"IoError", "FsError", "NetError"}:
                return
            return
        if kind == "group":
            self.validate_expression(expression.children[0])
            return
        if kind == "try":
            source_type = expression.children[0].type
            function_return = (
                self.current_function.return_type
                if self.current_function is not None
                else AetherType("Unknown")
            )
            if (
                source_type.name != "Result"
                or len(source_type.arguments) != 2
                or function_return.name != "Result"
                or len(function_return.arguments) != 2
                or source_type.arguments[1] != function_return.arguments[1]
                or source_type.arguments[0] != expression.type
            ):
                self.error(
                    "E_C99_TRY",
                    "C99 '?' requires a Result value with the enclosing function's error type",
                    expression.span,
                )
            self.validate_expression(expression.children[0])
            return
        if kind == "unary":
            operand = expression.children[0]
            if expression.value not in {"!", "+", "-"}:
                self.error(
                    "E_C99_OPERATOR",
                    f"Unary operator {expression.value!r} is not supported",
                    expression.span,
                )
            elif expression.value == "!" and expression.type.name != "bool":
                self.error(
                    "E_C99_OPERATOR",
                    "Unary ! is supported only for bool",
                    expression.span,
                )
            elif operand.type.name == "Tensor" and expression.value == "-":
                pass
            elif expression.value == "-" and operand.type.name in _C_TYPES:
                if operand.type.name.startswith("u"):
                    self.error(
                        "E_C99_OPERATOR",
                        "Unary negation of unsigned integers is not supported",
                        expression.span,
                    )
                elif operand.type.name in _SIGNED_TYPES:
                    is_minimum_literal = False
                    if operand.kind == "literal" and operand.value.isdecimal():
                        is_minimum_literal = (
                            int(operand.value, 10)
                            == _INTEGER_LIMITS[operand.type.name][1] + 1
                        )
                    if not is_minimum_literal:
                        self.helpers.add((operand.type.name, "neg"))
                        self.needs_trap = True
            self.validate_expression(
                operand,
                allow_signed_min_magnitude=(
                    expression.value == "-"
                    and operand.kind == "literal"
                    and operand.type.name in _SIGNED_TYPES
                ),
            )
            return
        if kind == "binary":
            operator = expression.value
            if operator == "|>":
                self._validate_pipeline(expression)
                return
            if operator == "**":
                self.error(
                    "E_C99_OPERATOR",
                    "Exponentiation is not yet supported by the C99 backend",
                    expression.span,
                )
            elif operator not in _BINARY_OPERATORS | {"&&", "||"}:
                self.error(
                    "E_C99_OPERATOR",
                    f"Binary operator {operator!r} is not supported",
                    expression.span,
                )
            elif operator in {"/", "%"} and expression.type.name != "Tensor":
                operand_type = expression.children[0].type
                if operand_type.name in _INTEGER_TYPES:
                    self.helpers.add(
                        (operand_type.name, "div" if operator == "/" else "rem")
                    )
                    self.needs_trap = True
                elif operator == "/" and operand_type.name in {"f32", "f64"}:
                    self.helpers.add((operand_type.name, "div"))
                    self.needs_trap = True
                else:
                    self.error(
                        "E_C99_OPERATOR",
                        f"Operator {operator!r} is unsupported for "
                        f"{operand_type.display()}",
                        expression.span,
                    )
            elif operator == "/" and expression.type.name == "Tensor":
                self.needs_trap = True
            elif operator in {"+", "-", "*"} and expression.type.name != "Tensor":
                operand_type = expression.type.name
                if operand_type in _INTEGER_TYPES:
                    operation = {"+": "add", "-": "sub", "*": "mul"}[operator]
                    self.helpers.add((operand_type, operation))
                    self.needs_trap = True
            self.validate_expression(expression.children[0])
            self.validate_expression(expression.children[1])
            return
        if kind == "index":
            base, index = expression.children
            if base.type.name != "Tensor":
                self.error(
                    "E_C99_INDEX",
                    "C99 backend supports indexing statically shaped tensors only",
                    expression.span,
                )
            elif base.type.element_type is None or base.type.element_type.name not in {
                "f32",
                "f64",
            }:
                self.error(
                    "E_C99_TYPE",
                    f"C99 tensor indexing does not support {base.type.display()}",
                    expression.span,
                )
            self.needs_trap = True
            self.validate_expression(base)
            self.validate_expression(index)
            return
        if kind == "gradient":
            target = self.functions.get(expression.value)
            parameter_type = (
                target.parameters[0].type
                if target is not None and target.parameters
                else AetherType("Unknown")
            )
            input_type = expression.children[0].type
            valid_target_type = (
                parameter_type.name in {"f32", "f64"}
                or (
                    parameter_type.name == "Tensor"
                    and parameter_type.element_type is not None
                    and parameter_type.element_type.name in {"f32", "f64"}
                )
            )
            if (
                target is None
                or target.external_abi is not None
                or len(target.parameters) != 1
                or not valid_target_type
                or input_type != parameter_type
                or expression.type.name != parameter_type.name
            ):
                self.error(
                    "E_C99_GRADIENT",
                    "C99 autodiff requires a matching unary f32/f64 scalar or "
                    "static tensor function target",
                    expression.span,
                )
            else:
                self.gradient_targets.add(expression.value)
                self.needs_trap = True
            self.validate_expression(expression.children[0])
            return
        if kind == "call":
            self._validate_call(expression)
            return
        self.error(
            "E_C99_EXPRESSION",
            f"Expression kind {kind!r} is not supported by the C99 backend",
            expression.span,
        )

    def _validate_pipeline(self, expression: IRExpression) -> None:
        left, right = expression.children
        self.validate_expression(left)
        if right.kind == "call":
            self._validate_call(right, pipeline_input=left)
        elif right.kind == "identifier":
            if right.value in {
                "abs", "min", "max", "mean", "max_abs", "sum", "len",
                "scale", "subtract", "convert",
            }:
                builtin_call = IRExpression(
                    "call",
                    right.span,
                    expression.type,
                    children=(right,),
                )
                self._validate_builtin_call(builtin_call, left)
            else:
                self._validate_call_target(right.value, right.span, (left.type,))
        else:
            self.error(
                "E_C99_PIPELINE",
                "C99 pipelines must target a user-defined function",
                right.span,
            )

    def _validate_call(
        self,
        expression: IRExpression,
        pipeline_input: IRExpression | None = None,
    ) -> None:
        callee = expression.children[0]
        if callee.kind != "identifier":
            self.error(
                "E_C99_CALL",
                "C99 backend supports calls to named user functions only",
                expression.span,
            )
            for argument in expression.children[1:]:
                self.validate_expression(argument)
            return
        if callee.value in {"Ok", "Err"}:
            if (
                len(expression.children) != 2
                or expression.type.name != "Result"
                or len(expression.type.arguments) != 2
            ):
                self.error(
                    "E_C99_RESULT",
                    f"{callee.value} requires a supported Result return context",
                    expression.span,
                )
            for argument in expression.children[1:]:
                self.validate_expression(argument)
            return
        if callee.value in {"frame_buffer", "frame_buffer_len"}:
            expected_result = (
                "FrameBuffer" if callee.value == "frame_buffer" else "u64"
            )
            valid_arguments = (
                len(expression.children) == 2
                and (
                    expression.children[1].type.name
                    == ("u64" if callee.value == "frame_buffer" else "FrameBuffer")
                )
            )
            if (
                pipeline_input is not None
                or not valid_arguments
                or expression.type.name != expected_result
            ):
                self.error(
                    "E_C99_FRAME_BUFFER",
                    f"{callee.value} has an invalid frame-resource signature",
                    expression.span,
                )
            if callee.value == "frame_buffer":
                self.needs_trap = True
            for argument in expression.children[1:]:
                self.validate_expression(argument)
            return
        argument_types = (
            (pipeline_input.type,) if pipeline_input is not None else ()
        ) + tuple(argument.type for argument in expression.children[1:])
        if callee.value in {
            "abs",
            "min",
            "max",
            "mean",
            "max_abs",
            "sum",
            "len",
            "scale",
            "subtract",
            "convert",
        }:
            self._validate_builtin_call(expression, pipeline_input)
        else:
            self._validate_call_target(callee.value, expression.span, argument_types)
        for index, argument in enumerate(expression.children[1:], start=1):
            if callee.value == "convert" and index == len(expression.children) - 1:
                continue
            self.validate_expression(argument)

    def _validate_builtin_call(
        self,
        expression: IRExpression,
        pipeline_input: IRExpression | None,
    ) -> None:
        callee = expression.children[0]
        name = callee.value
        arguments = (
            ([pipeline_input] if pipeline_input is not None else [])
            + list(expression.children[1:])
        )
        tensor_arguments = [argument for argument in arguments if argument.type.name == "Tensor"]
        allowed = False
        if name in {"sum", "mean", "max_abs", "len"}:
            allowed = len(arguments) == 1 and len(tensor_arguments) == 1
        elif name == "abs":
            allowed = (
                len(arguments) == 1
                and arguments[0].type.name in {"f32", "f64", "Tensor"}
            )
        elif name in {"min", "max"}:
            allowed = (
                len(arguments) == 2
                and len(tensor_arguments) == 2
                and arguments[0].type == arguments[1].type
            )
        elif name in {"scale", "subtract"}:
            allowed = len(arguments) == 2 and expression.type.name == "Tensor"
        elif name == "convert":
            allowed = len(arguments) == 2 and (
                arguments[0].type.name == "Tensor"
                or arguments[0].type.name in {"f32", "f64"}
            )
        if not allowed:
            self.error(
                "E_C99_BUILTIN",
                f"C99 tensor backend does not support this use of {name!r}",
                expression.span,
            )
        if name == "len" and expression.type.name != "i64":
            self.error(
                "E_C99_BUILTIN",
                "Tensor len must produce i64",
                expression.span,
            )
        if name in {"sum", "mean", "max_abs"} and len(arguments) == 1:
            if arguments[0].type.name != "Tensor":
                self.error(
                    "E_C99_BUILTIN",
                    f"{name} requires a tensor",
                    expression.span,
                )
        if name in {"scale", "subtract"} and len(arguments) == 2:
            if not any(argument.type.name == "Tensor" for argument in arguments):
                self.error(
                    "E_C99_BUILTIN",
                    f"{name} requires at least one tensor argument",
                    expression.span,
                )
        if name == "convert" and len(arguments) == 2:
            if expression.conversion is None:
                self.error(
                    "E_C99_BUILTIN",
                    "C99 backend requires resolved unit conversion metadata",
                    expression.span,
                )
            unit_expression = (
                expression.children[1]
                if pipeline_input is not None
                else expression.children[2]
            )
            if unit_expression.kind != "literal" or not unit_expression.value.startswith(
                ("'", '"')
            ):
                self.error(
                    "E_C99_BUILTIN",
                    "C99 tensor conversion requires a constant unit name",
                    unit_expression.span,
                )
        if name == "mean" and arguments and arguments[0].type.name == "Tensor":
            if arguments[0].type.element_type not in (
                AetherType("f32"),
                AetherType("f64"),
            ):
                self.error(
                    "E_C99_TYPE",
                    "C99 tensor mean supports only f32/f64 elements",
                    expression.span,
                )

    def _validate_call_target(
        self,
        name: str,
        span: SourceSpan,
        argument_types: tuple[AetherType, ...],
    ) -> None:
        target = self.functions.get(name)
        if target is None:
            self.error(
                "E_C99_CALL",
                f"C99 backend does not support built-in or unknown call {name!r}",
                span,
            )
            return
        if target.external_abi is not None:
            if target.external_abi != "C":
                self.error(
                    "E_C99_FFI",
                    f"Foreign ABI {target.external_abi!r} is not supported",
                    span,
                )
            if (
                not target.name.isascii()
                or not target.name.isidentifier()
                or target.name.startswith("_")
                or target.name in _C_KEYWORDS
            ):
                self.error(
                    "E_C99_FFI_SYMBOL",
                    f"Extern function name {target.name!r} is not a portable C symbol",
                    span,
                )
            if target.external_library is None:
                self.error(
                    "E_C99_FFI_LIBRARY",
                    f"Extern function {name!r} has no @link library",
                    span,
                )
            for parameter in target.parameters:
                self.validate_type(parameter.type, span, "FFI parameter")
            self.validate_type(target.return_type, span, "FFI return")
        if len(argument_types) != len(target.parameters):
            return
        for actual, expected in zip(argument_types, target.parameters):
            if actual != expected.type:
                self.error(
                    "E_C99_ARGUMENT",
                    f"Call to {name!r} has an argument of type "
                    f"{actual.display()}, expected {expected.type.display()}",
                    span,
                )

    def _validate_literal(
        self,
        expression: IRExpression,
        allow_signed_min_magnitude: bool,
    ) -> None:
        value = expression.value
        type_name = expression.type.name
        if value in {"true", "false"}:
            if type_name != "bool":
                self.error("E_C99_LITERAL", "Boolean literal has an invalid type", expression.span)
            return
        if value == "none" or value.startswith(("'", '"')):
            self.error(
                "E_C99_LITERAL",
                f"Literal {value!r} is not supported by the scalar C99 backend",
                expression.span,
            )
            return
        if type_name in _INTEGER_TYPES:
            try:
                numeric_value = int(value, 10)
            except ValueError:
                self.error("E_C99_LITERAL", f"Invalid integer literal {value!r}", expression.span)
                return
            minimum, maximum = _INTEGER_LIMITS[type_name]
            if allow_signed_min_magnitude and type_name in _SIGNED_TYPES:
                minimum_magnitude = maximum + 1
                if numeric_value == minimum_magnitude:
                    return
            if not minimum <= numeric_value <= maximum:
                self.error(
                    "E_C99_LITERAL_RANGE",
                    f"Integer literal does not fit in {type_name}",
                    expression.span,
                )
        elif type_name in {"f32", "f64"}:
            try:
                numeric_value = Decimal(value)
            except InvalidOperation:
                self.error("E_C99_LITERAL", f"Invalid floating literal {value!r}", expression.span)
                return
            if not numeric_value.is_finite() or abs(numeric_value) > _FLOAT_LIMITS[type_name]:
                self.error(
                    "E_C99_LITERAL_RANGE",
                    f"Floating literal is outside the finite range of {type_name}",
                    expression.span,
                )
        else:
            self.error(
                "E_C99_LITERAL",
                f"Literal type {type_name!r} is not supported",
                expression.span,
            )

    def validate(self) -> None:
        if self.include_entry_point:
            entry = self.functions.get("main")
            if entry is None:
                span = (
                    self.module.functions[0].span
                    if self.module.functions
                    else SourceSpan(0, 0, 1, 1)
                )
                self.error(
                    "E_C99_ENTRY_POINT",
                    "Executable output requires fn main() -> i32 or fn main() -> void",
                    span,
                )
            elif entry.parameters or entry.return_type.name not in {"i32", "void"}:
                self.error(
                    "E_C99_ENTRY_POINT",
                    "Entry function main must take no parameters and return i32 or void",
                    entry.span,
                )
        for declaration in self.module.types:
            self.error(
                "E_C99_TYPE_DECL",
                f"Type declaration {declaration.name!r} is not supported by C99 v0",
                declaration.span,
            )
        for function in self.module.functions:
            if function.external_abi is not None:
                if function.external_abi != "C":
                    self.error(
                        "E_C99_FFI",
                        f"Foreign ABI {function.external_abi!r} is not supported",
                        function.span,
                    )
                if (
                    not function.name.isascii()
                    or not function.name.isidentifier()
                    or function.name.startswith("_")
                    or function.name in _C_KEYWORDS
                ):
                    self.error(
                        "E_C99_FFI_SYMBOL",
                        f"Extern function name {function.name!r} is not a portable C symbol",
                        function.span,
                    )
                if function.external_library is None:
                    self.error(
                        "E_C99_FFI_LIBRARY",
                        f"Extern function {function.name!r} has no @link library",
                        function.span,
                    )
                if set(function.declared_effects) != {"ffi"}:
                    self.error(
                        "E_C99_FFI_EFFECT",
                        f"Extern function {function.name!r} may declare only @ffi",
                        function.span,
                    )
                self.validate_type(function.return_type, function.span, "FFI return")
                for parameter in function.parameters:
                    self.validate_type(parameter.type, function.span, "FFI parameter")
                if set(function.required_effects) != {"ffi"}:
                    self.error(
                        "E_C99_FFI_EFFECT",
                        f"Extern function {function.name!r} must require @ffi",
                        function.span,
                    )
                continue
            self.validate_type(function.return_type, function.span, "return")
            for parameter in function.parameters:
                self.validate_type(parameter.type, function.span, "parameter")
            if (
                set(function.declared_effects) - {"ffi"}
                or set(function.required_effects) - {"ffi"}
            ):
                self.error(
                    "E_C99_EFFECT",
                    f"Function {function.name!r} uses effects unsupported by C99 v0",
                    function.span,
                )
            if function.name not in self.defined_function_names:
                continue
            self.current_function = function
            mutable_locals: set[str] = set()
            for statement in function.body:
                if statement.expression is not None:
                    if statement.kind == "let" and statement.expression.type.name == "void":
                        self.error(
                            "E_C99_BINDING",
                            "A void function call cannot initialize a local binding",
                            statement.span,
                        )
                    self.validate_expression(statement.expression)
                if statement.kind == "assignment":
                    if statement.name not in mutable_locals:
                        self.error(
                            "E_C99_OWNERSHIP",
                            f"Assignment target {statement.name!r} is not a mutable local",
                            statement.span,
                        )
                elif statement.kind == "let" and statement.binding_kind == "var":
                    mutable_locals.add(statement.name)
            if function.return_type.name != "void" and (
                not function.body
                or function.body[-1].kind != "return"
                or function.body[-1].expression is None
            ):
                self.error(
                    "E_C99_RETURN",
                    f"Function {function.name!r} must end with a value return",
                    function.span,
                )
            self.current_function = None

    def _new_temp(self, type_: AetherType) -> str:
        name = f"aether_tmp_{self.temp_index}"
        self.temp_index += 1
        return f"{self.c_type(type_)} {name}"

    def _materialize(
        self,
        value: _CValue,
    ) -> tuple[list[str], str, list[str]]:
        if self._contains_frame_buffer(value.type) and value.ownership == "borrowed":
            return [*value.prelude], value.code, list(value.cleanup)
        declaration = self._new_temp(value.type)
        name = declaration.rsplit(" ", 1)[1]
        cleanup = list(value.cleanup)
        if self._contains_frame_buffer(value.type) and value.ownership == "owned":
            release = self._release_code(name, value.type)
            if release:
                cleanup.append(release)
        return [*value.prelude, f"{declaration} = {value.code};"], name, cleanup

    def expression(self, expression: IRExpression) -> _CValue:
        kind = expression.kind
        if kind == "literal":
            return _CValue((), self._literal_code(expression), expression.type)
        if kind == "identifier":
            if expression.value in {"IoError", "FsError", "NetError"}:
                error_codes = {"IoError": 1, "FsError": 2, "NetError": 3}
                return _CValue((), str(error_codes[expression.value]), expression.type)
            return _CValue(
                (),
                _encoded_identifier("aether_v_", expression.value),
                expression.type,
                expression.ownership,
            )
        if kind == "group":
            child = self.expression(expression.children[0])
            return _CValue(
                child.prelude,
                f"({child.code})",
                expression.type,
                child.ownership,
                child.cleanup,
                child.temporary,
            )
        if kind == "try":
            operand = self.expression(expression.children[0])
            prelude, name, cleanup = self._materialize(operand)
            failure_cleanup = tuple(
                reversed((*self.active_expression_cleanups, *cleanup))
            )
            prelude.extend(
                [
                    f"if (!{name}.is_ok) {{",
                    *[f"    {line}" for line in failure_cleanup],
                    f"    {self.frame_return_name}.is_ok = 0;",
                    f"    {self.frame_return_name}.value = "
                    f"({self.c_type(self.current_function.return_type.arguments[0])})"
                    "{0};",
                    f"    {self.frame_return_name}.error = {name}.error;",
                    f"    goto {self.frame_cleanup_label};",
                    "}",
                ]
            )
            if expression.type.name == "FrameBuffer":
                prelude.append(
                    f"aether_frame_buffer_retain(&{name}.value);"
                )
                return _CValue(
                    tuple(prelude),
                    f"{name}.value",
                    expression.type,
                    "owned",
                    tuple(cleanup),
                )
            return _CValue(
                tuple(prelude),
                f"{name}.value",
                expression.type,
                cleanup=tuple(cleanup),
            )
        if kind == "unary":
            operand = self.expression(expression.children[0])
            if expression.type.name == "Tensor":
                if expression.value == "+":
                    return operand
                prelude, name, cleanup = self._materialize(operand)
                declaration = self._new_temp(expression.type)
                result = declaration.rsplit(" ", 1)[1]
                count = self._tensor_size(expression.type)
                index = f"aether_i_{self.temp_index}"
                self.temp_index += 1
                prelude.append(f"{declaration} = {{0}};")
                prelude.append(
                    f"for (size_t {index} = 0; {index} < {count}; ++{index}) {{"
                )
                prelude.append(
                    f"    {result}.data[{index}] = -{name}.data[{index}];"
                )
                prelude.append("}")
                prelude.extend(cleanup)
                return _CValue(tuple(prelude), result, expression.type)
            if (
                expression.value == "-"
                and expression.children[0].kind == "literal"
                and expression.children[0].type.name in _SIGNED_TYPES
            ):
                name = expression.children[0].type.name
                value = int(expression.children[0].value, 10)
                if value == _INTEGER_LIMITS[name][1] + 1:
                    width = name[1:]
                    return _CValue((), f"INT{width}_MIN", expression.type)
            if expression.value == "-" and expression.type.name in _SIGNED_TYPES:
                code = f"aether_{expression.type.name}_neg({operand.code})"
            else:
                code = f"({expression.value}{operand.code})"
            return _CValue(
                operand.prelude,
                code,
                expression.type,
                cleanup=operand.cleanup,
            )
        if kind == "binary":
            if expression.value == "|>":
                return self._pipeline_expression(expression)
            if expression.value in {"&&", "||"}:
                return self._logical_expression(expression)
            left = self.expression(expression.children[0])
            prelude, left_name, left_cleanup = self._materialize(left)
            outer_cleanups = self.active_expression_cleanups
            self.active_expression_cleanups = (*outer_cleanups, *left_cleanup)
            try:
                right = self.expression(expression.children[1])
                right_prelude, right_name, right_cleanup = self._materialize(right)
            finally:
                self.active_expression_cleanups = outer_cleanups
            prelude.extend(right_prelude)
            operator = expression.value
            type_name = expression.type.name
            if type_name == "Tensor":
                return self._tensor_binary(
                    expression,
                    left_name,
                    right_name,
                    prelude,
                    (*left_cleanup, *right_cleanup),
                )
            if operator in {"+", "-", "*"} and type_name in _INTEGER_TYPES:
                code = f"aether_{type_name}_{self._operation_name(operator)}({left_name}, {right_name})"
            elif operator in {"/", "%"} and type_name in _INTEGER_TYPES:
                code = f"aether_{type_name}_{self._operation_name(operator)}({left_name}, {right_name})"
            elif operator == "/" and type_name in {"f32", "f64"}:
                code = f"aether_{type_name}_div({left_name}, {right_name})"
            else:
                code = f"({left_name} {operator} {right_name})"
            return _CValue(
                tuple(prelude),
                code,
                expression.type,
                cleanup=(*left_cleanup, *right_cleanup),
            )
        if kind == "call":
            return self._call_expression(expression)
        if kind == "gradient":
            argument = self.expression(expression.children[0])
            prelude, name, cleanup = self._materialize(argument)
            target = _encoded_identifier("aether_grad_", expression.value)
            return _CValue(
                tuple(prelude),
                f"{target}({name})",
                expression.type,
                cleanup=tuple(cleanup),
            )
        if kind == "index":
            return self._index_expression(expression)
        raise C99LoweringError(
            (Diagnostic("E_C99_EXPRESSION", f"Unsupported IR expression {kind!r}", expression.span),)
        )

    def _logical_expression(self, expression: IRExpression) -> _CValue:
        left = self.expression(expression.children[0])
        prelude, name, left_cleanup = self._materialize(left)
        outer_cleanups = self.active_expression_cleanups
        self.active_expression_cleanups = (*outer_cleanups, *left_cleanup)
        try:
            right = self.expression(expression.children[1])
        finally:
            self.active_expression_cleanups = outer_cleanups
        condition = name if expression.value == "&&" else f"!{name}"
        right_lines, right_name, right_cleanup = self._materialize(right)
        prelude.append(f"if ({condition}) {{")
        prelude.extend("  " + line for line in right_lines)
        prelude.append(f"  {name} = {right_name};")
        prelude.extend("  " + line for line in right_cleanup)
        prelude.append("}")
        return _CValue(
            tuple(prelude),
            name,
            expression.type,
            cleanup=tuple(left_cleanup),
        )

    def _call_expression(self, expression: IRExpression) -> _CValue:
        callee = expression.children[0]
        if callee.kind == "identifier" and callee.value in {"Ok", "Err"}:
            argument = self.expression(expression.children[1])
            if callee.value == "Ok":
                prelude = list(argument.prelude)
                if (
                    self._contains_frame_buffer(argument.type)
                    and argument.ownership == "borrowed"
                ):
                    prelude.append(
                        f"aether_frame_buffer_retain(&{argument.code});"
                    )
                code = (
                    f"({self.c_type(expression.type)})"
                    f"{{1, {argument.code}, 0}}"
                )
            else:
                prelude = list(argument.prelude)
                code = f"({self.c_type(expression.type)}){{0, 0, {argument.code}}}"
            return _CValue(
                tuple(prelude),
                code,
                expression.type,
                "owned" if self._contains_frame_buffer(expression.type) else "plain",
                argument.cleanup,
                self._contains_frame_buffer(expression.type),
            )
        if callee.kind == "identifier" and callee.value in {
            "frame_buffer",
            "frame_buffer_len",
        }:
            return self._builtin_expression(expression)
        if callee.kind == "identifier" and callee.value in {
            "abs", "min", "max", "mean", "max_abs", "sum", "len",
            "scale", "subtract", "convert",
        }:
            return self._builtin_expression(expression)
        prelude: list[str] = []
        names: list[str] = []
        cleanup: list[str] = []
        target = self.functions[callee.value]
        outer_cleanups = self.active_expression_cleanups
        try:
            for argument_expression in expression.children[1:]:
                self.active_expression_cleanups = (
                    *outer_cleanups,
                    *cleanup,
                )
                argument = self.expression(argument_expression)
                argument_lines, name, argument_cleanup = self._materialize(argument)
                prelude.extend(argument_lines)
                names.append(name)
                cleanup.extend(argument_cleanup)
        finally:
            self.active_expression_cleanups = outer_cleanups
        return _CValue(
            tuple(prelude),
            f"{self._call_target(callee.value)}({', '.join(names)})",
            target.return_type,
            "owned" if self._contains_frame_buffer(target.return_type) else "plain",
            tuple(cleanup),
            self._contains_frame_buffer(target.return_type),
        )

    def _pipeline_expression(self, expression: IRExpression) -> _CValue:
        left, right = expression.children
        left_value = self.expression(left)
        if right.kind == "identifier":
            if right.value in {
                "abs", "min", "max", "mean", "max_abs", "sum", "len",
                "scale", "subtract", "convert",
            }:
                builtin_call = IRExpression(
                    "call",
                    right.span,
                    expression.type,
                    children=(right,),
                    conversion=right.conversion,
                )
                return self._builtin_expression(
                    builtin_call,
                    pipeline_input=left_value,
                )
            prelude, left_name, left_cleanup = self._materialize(left_value)
            target_name = right.value
        else:
            target_name = right.children[0].value
            if target_name in {
                "abs", "min", "max", "mean", "max_abs", "sum", "len",
                "scale", "subtract", "convert",
            }:
                return self._builtin_expression(right, pipeline_input=left_value)
            prelude, left_name, left_cleanup = self._materialize(left_value)
            names = [left_name]
            cleanup = list(left_cleanup)
            outer_cleanups = self.active_expression_cleanups
            try:
                for argument_expression in right.children[1:]:
                    self.active_expression_cleanups = (
                        *outer_cleanups,
                        *cleanup,
                    )
                    argument = self.expression(argument_expression)
                    argument_lines, name, argument_cleanup = self._materialize(argument)
                    prelude.extend(argument_lines)
                    names.append(name)
                    cleanup.extend(argument_cleanup)
            finally:
                self.active_expression_cleanups = outer_cleanups
        if right.kind == "identifier":
            names = [left_name]
            cleanup = list(left_cleanup)
        target = self.functions[target_name]
        return _CValue(
            tuple(prelude),
            f"{self._call_target(target_name)}({', '.join(names)})",
            target.return_type,
            "owned" if self._contains_frame_buffer(target.return_type) else "plain",
            tuple(cleanup),
            self._contains_frame_buffer(target.return_type),
        )

    @staticmethod
    def _tensor_size(type_: AetherType) -> int:
        size = 1
        for dimension in type_.dimensions:
            size *= dimension
        return size

    def _tensor_binary(
        self,
        expression: IRExpression,
        left_name: str,
        right_name: str,
        prelude: list[str],
        cleanup: tuple[str, ...] = (),
    ) -> _CValue:
        result_type = expression.type
        result_declaration = self._new_temp(result_type)
        result_name = result_declaration.rsplit(" ", 1)[1]
        length = self._tensor_size(result_type)
        index_name = f"aether_i_{self.temp_index}"
        self.temp_index += 1
        left_type, right_type = (child.type for child in expression.children)
        operator = expression.value
        if operator == "/" and (
            left_type.name == "Tensor" or right_type.name == "Tensor"
        ):
            self.needs_trap = True
        left_code = (
            f"{left_name}.data[{index_name}]"
            if left_type.name == "Tensor"
            else left_name
        )
        right_code = (
            f"{right_name}.data[{index_name}]"
            if right_type.name == "Tensor"
            else right_name
        )
        if operator == "/":
            prelude.append(f"{result_declaration};")
            prelude.append(
                f"for (size_t {index_name} = 0; {index_name} < {length}; "
                f"++{index_name}) {{"
            )
            prelude.append(f"    if ({right_code} == 0.0) aether_trap();")
            prelude.append(
                f"    {result_name}.data[{index_name}] = {left_code} / {right_code};"
            )
            prelude.append("}")
        else:
            prelude.append(f"{result_declaration} = {{0}};")
            prelude.append(
                f"for (size_t {index_name} = 0; {index_name} < {length}; "
                f"++{index_name}) {{"
            )
            prelude.append(
                f"    {result_name}.data[{index_name}] = "
                f"{left_code} {operator} {right_code};"
            )
            prelude.append("}")
        return _CValue(
            tuple(prelude),
            result_name,
            result_type,
            cleanup=cleanup,
        )

    def _index_expression(self, expression: IRExpression) -> _CValue:
        base = self.expression(expression.children[0])
        index = self.expression(expression.children[1])
        prelude, base_name, base_cleanup = self._materialize(base)
        index_lines, index_name, index_cleanup = self._materialize(index)
        prelude.extend(index_lines)
        first_dimension = base.type.dimensions[0]
        if expression.children[1].type.name in _SIGNED_TYPES:
            prelude.append(
                f"if ({index_name} < 0 || (uint64_t){index_name} >= "
                f"UINT64_C({first_dimension})) aether_trap();"
            )
        else:
            prelude.append(
                f"if ((uint64_t){index_name} >= UINT64_C({first_dimension})) "
                "aether_trap();"
            )
        stride = self._tensor_size(
            AetherType(
                "Tensor",
                base.type.arguments,
                base.type.dimensions[1:],
                base.type.unit,
                base.type.unit_scale,
                base.type.unit_offset,
                base.type.unit_symbol,
                base.type.unit_kind,
                base.type.unit_pi_power,
            )
        ) if len(base.type.dimensions) > 1 else 1
        offset = f"((size_t){index_name} * {stride})"
        if expression.type.name != "Tensor":
            return _CValue(
                tuple(prelude),
                f"{base_name}.data[{offset}]",
                expression.type,
                cleanup=(*base_cleanup, *index_cleanup),
            )
        declaration = self._new_temp(expression.type)
        result_name = declaration.rsplit(" ", 1)[1]
        prelude.append(f"{declaration} = {{0}};")
        loop_index = f"aether_i_{self.temp_index}"
        self.temp_index += 1
        count = self._tensor_size(expression.type)
        prelude.append(
            f"for (size_t {loop_index} = 0; {loop_index} < {count}; "
            f"++{loop_index}) {{"
        )
        prelude.append(
            f"    {result_name}.data[{loop_index}] = "
            f"{base_name}.data[{offset} + {loop_index}];"
        )
        prelude.append("}")
        return _CValue(
        tuple(prelude),
        result_name,
        expression.type,
        cleanup=(*base_cleanup, *index_cleanup),
        )

    def _conversion_factor(self, expression: IRExpression) -> tuple[str, str]:
        conversion = expression.conversion
        if conversion is None:
            return "1.0", "0.0"

        def fraction_code(value: Fraction) -> str:
            number = float(value)
            return number.hex()

        multiplier = float(conversion.multiplier) * math.pi ** conversion.pi_power
        return multiplier.hex(), fraction_code(conversion.addend)

    def _gradient_tape_expression(
        self,
        expression: IRExpression,
        names: dict[str, int | tuple[int, ...]],
        lines: list[str],
        next_node: list[int],
    ) -> int | tuple[int, ...]:
        if expression.kind == "literal":
            return self._append_gradient_node(
                lines,
                next_node,
                self._literal_code(expression),
                expression.type,
            )
        if expression.kind == "identifier":
            try:
                return names[expression.value]
            except KeyError as error:
                raise ValueError(
                    f"Unknown differentiable local {expression.value!r}"
                ) from error
        if expression.kind == "group":
            return self._gradient_tape_expression(
                expression.children[0], names, lines, next_node
            )
        if expression.kind == "unary":
            operand = self._gradient_tape_expression(
                expression.children[0], names, lines, next_node
            )
            if expression.value == "+":
                return operand
            operand_nodes = (
                operand if isinstance(operand, tuple) else (operand,)
            )
            result_nodes = []
            element_type = self._gradient_element_type(expression.type)
            for parent in operand_nodes:
                result_nodes.append(
                    self._append_gradient_node(
                        lines,
                        next_node,
                        f"-tape[{parent}].value",
                        element_type,
                        parent_left=parent,
                        partial_left="-1.0",
                    )
                )
            return tuple(result_nodes) if expression.type.name == "Tensor" else result_nodes[0]
        if expression.kind == "binary":
            left = self._gradient_tape_expression(
                expression.children[0], names, lines, next_node
            )
            right = self._gradient_tape_expression(
                expression.children[1], names, lines, next_node
            )
            left_nodes = left if isinstance(left, tuple) else (left,)
            right_nodes = right if isinstance(right, tuple) else (right,)
            left_is_tensor = expression.children[0].type.name == "Tensor"
            right_is_tensor = expression.children[1].type.name == "Tensor"
            count = (
                len(left_nodes) if left_is_tensor
                else len(right_nodes) if right_is_tensor
                else 1
            )
            if left_is_tensor and right_is_tensor and len(left_nodes) != len(right_nodes):
                raise ValueError("Differentiable tensor operands have incompatible shapes")
            element_type = self._gradient_element_type(expression.type)
            result_nodes = []
            operator = expression.value
            for index in range(count):
                parent_left = left_nodes[index] if left_is_tensor else left_nodes[0]
                parent_right = right_nodes[index] if right_is_tensor else right_nodes[0]
                a = f"tape[{parent_left}].value"
                b = f"tape[{parent_right}].value"
                if operator == "+":
                    value_code, partial_left, partial_right = f"({a} + {b})", "1.0", "1.0"
                elif operator == "-":
                    value_code, partial_left, partial_right = f"({a} - {b})", "1.0", "-1.0"
                elif operator == "*":
                    value_code, partial_left, partial_right = f"({a} * {b})", b, a
                elif operator == "/":
                    lines.append(f"if ({b} == 0.0) aether_trap();")
                    value_code = f"({a} / {b})"
                    partial_left = f"(1.0 / {b})"
                    partial_right = f"(-{a} / ({b} * {b}))"
                else:
                    raise ValueError(f"Unsupported autodiff operator {operator!r}")
                result_nodes.append(
                    self._append_gradient_node(
                        lines,
                        next_node,
                        value_code,
                        element_type,
                        parent_left=parent_left,
                        partial_left=self._gradient_cast(
                            element_type, partial_left
                        ),
                        parent_right=parent_right,
                        partial_right=self._gradient_cast(
                            element_type, partial_right
                        ),
                    )
                )
            if expression.type.name == "Tensor":
                return tuple(result_nodes)
            return result_nodes[0]
        if expression.kind == "call":
            callee = expression.children[0]
            if callee.kind != "identifier":
                raise ValueError("Differentiable calls must use named built-ins")
            name = callee.value
            if name == "convert":
                operand = self._gradient_tape_expression(
                    expression.children[1], names, lines, next_node
                )
                operand_nodes = (
                    operand if isinstance(operand, tuple) else (operand,)
                )
                multiplier, addend = self._conversion_factor(expression)
                result_nodes = []
                element_type = self._gradient_element_type(expression.type)
                for parent in operand_nodes:
                    value_code = (
                        f"(tape[{parent}].value * ({multiplier}) + ({addend}))"
                    )
                    result_nodes.append(
                        self._append_gradient_node(
                            lines,
                            next_node,
                            value_code,
                            element_type,
                            parent_left=parent,
                            partial_left=self._gradient_cast(
                                element_type, multiplier
                            ),
                        )
                    )
                return (
                    tuple(result_nodes)
                    if expression.type.name == "Tensor"
                    else result_nodes[0]
                )
            if name not in {"sum", "mean"}:
                raise ValueError(f"Unsupported autodiff call {name!r}")
            operand = self._gradient_tape_expression(
                expression.children[1], names, lines, next_node
            )
            if not isinstance(operand, tuple) or not operand:
                raise ValueError("Autodiff reductions require a non-empty tensor")
            scalar_type = expression.type
            accumulator = self._append_gradient_node(
                lines,
                next_node,
                "0.0",
                scalar_type,
            )
            for parent in operand:
                accumulator = self._append_gradient_node(
                    lines,
                    next_node,
                    f"(tape[{accumulator}].value + tape[{parent}].value)",
                    scalar_type,
                    parent_left=accumulator,
                    partial_left="1.0",
                    parent_right=parent,
                    partial_right="1.0",
                )
            if name == "mean":
                count = len(operand)
                accumulator = self._append_gradient_node(
                    lines,
                    next_node,
                    f"(tape[{accumulator}].value / ({self.c_type(scalar_type)}){count})",
                    scalar_type,
                    parent_left=accumulator,
                    partial_left=self._gradient_cast(
                        scalar_type,
                        f"(1.0 / {count}.0)",
                    ),
                )
            return accumulator
        raise ValueError(
            f"Unsupported autodiff expression {expression.kind!r}"
        )

    def _gradient_element_type(self, type_: AetherType) -> AetherType:
        return type_.element_type if type_.name == "Tensor" else type_

    def _gradient_cast(self, type_: AetherType, value: str) -> str:
        return f"(double)(({self.c_type(type_)})({value}))"

    def _append_gradient_node(
        self,
        lines: list[str],
        next_node: list[int],
        value: str,
        value_type: AetherType,
        *,
        parent_left: int | None = None,
        partial_left: str = "0.0",
        parent_right: int | None = None,
        partial_right: str = "0.0",
    ) -> int:
        node = next_node[0]
        next_node[0] += 1
        lines.extend(
            [
                f"tape[{node}].value = {self._gradient_cast(value_type, value)};",
                f"tape[{node}].has_parent_left = "
                f"{'true' if parent_left is not None else 'false'};",
                f"tape[{node}].parent_left = "
                f"{parent_left if parent_left is not None else '0'};",
                f"tape[{node}].partial_left = "
                f"{partial_left if parent_left is not None else '0.0'};",
                f"tape[{node}].has_parent_right = "
                f"{'true' if parent_right is not None else 'false'};",
                f"tape[{node}].parent_right = "
                f"{parent_right if parent_right is not None else '0'};",
                f"tape[{node}].partial_right = "
                f"{partial_right if parent_right is not None else '0.0'};",
            ]
        )
        return node

    def _emit_gradient_helper(self, function: IRFunction) -> list[str]:
        if len(function.parameters) != 1 or not function.body:
            raise ValueError(f"Invalid autodiff target {function.name!r}")
        parameter = function.parameters[0]
        ctype = self.c_type(parameter.type)
        lines: list[str] = []
        names: dict[str, int | tuple[int, ...]]
        next_node = [0]
        if parameter.type.name == "Tensor":
            element_type = parameter.type.element_type
            if element_type is None:
                raise ValueError("Autodiff tensor parameter has no element type")
            input_nodes = []
            for index in range(self._tensor_size(parameter.type)):
                input_nodes.append(
                    self._append_gradient_node(
                        lines,
                        next_node,
                        f"aether_x.data[{index}]",
                        element_type,
                    )
                )
            names = {parameter.name: tuple(input_nodes)}
        else:
            input_node = self._append_gradient_node(
                lines,
                next_node,
                "aether_x",
                parameter.type,
            )
            names = {parameter.name: input_node}
        return_node: int | None = None
        for statement in function.body:
            if statement.expression is None:
                continue
            node = self._gradient_tape_expression(
                statement.expression,
                names,
                lines,
                next_node,
            )
            if statement.kind == "let":
                names[statement.name] = node
            elif statement.kind == "return":
                if isinstance(node, tuple):
                    raise ValueError("Autodiff objective must return a scalar")
                return_node = node
                break
            else:
                raise ValueError(
                    f"Unsupported autodiff statement {statement.kind!r}"
                )
        if return_node is None:
            raise ValueError(f"Autodiff target {function.name!r} has no return")
        count = next_node[0]
        helper_name = _encoded_identifier("aether_grad_", function.name)
        if count > 2**31 - 1:
            raise ValueError("Autodiff tape exceeds supported node count")
        result = [
            f"static {ctype} {helper_name}({ctype} aether_x) {{",
            "    aether_ad_node *tape;",
            f"    if (SIZE_MAX / sizeof(*tape) < (uintmax_t){count}) "
            "aether_trap();",
            f"    tape = (aether_ad_node *)malloc((size_t){count} * sizeof(*tape));",
            "    if (tape == NULL) aether_trap();",
            *("    " + line for line in lines),
            f"    for (size_t aether_i = 0; aether_i < (size_t){count}; "
            "++aether_i) tape[aether_i].adjoint = 0.0;",
            f"    tape[{return_node}].adjoint = 1.0;",
            f"    for (size_t aether_i = {count}; aether_i > 0; --aether_i) {{",
            "        size_t aether_node = aether_i - 1;",
            "        if (tape[aether_node].has_parent_left) {",
            "            size_t aether_parent = tape[aether_node].parent_left;",
            "            tape[aether_parent].adjoint += "
            "tape[aether_node].adjoint * tape[aether_node].partial_left;",
            "        }",
            "        if (tape[aether_node].has_parent_right) {",
            "            size_t aether_parent = tape[aether_node].parent_right;",
            "            tape[aether_parent].adjoint += "
            "tape[aether_node].adjoint * tape[aether_node].partial_right;",
            "        }",
            "    }",
        ]
        if parameter.type.name == "Tensor":
            result.append(f"    {ctype} aether_result = {{0}};")
            for index in range(self._tensor_size(parameter.type)):
                result.append(
                    f"    aether_result.data[{index}] = "
                    f"({self.c_type(parameter.type.element_type)})"
                    f"tape[{index}].adjoint;"
                )
            result.extend(
                [
                    "    free(tape);",
                    "    return aether_result;",
                    "}",
                ]
            )
        else:
            result.extend(
                [
                    "    " + f"{ctype} aether_result = "
                    f"({ctype})tape[0].adjoint;",
                    "    free(tape);",
                    "    return aether_result;",
                    "}",
                ]
            )
        return result

    def _builtin_expression(
        self,
        expression: IRExpression,
        *,
        pipeline_input: _CValue | None = None,
    ) -> _CValue:
        name = expression.children[0].value
        argument_expressions = list(expression.children[1:])
        if name == "convert" and argument_expressions:
            argument_expressions = argument_expressions[:-1]
        prelude: list[str] = []
        names: list[str] = []
        cleanup: list[str] = []
        values: list[_CValue] = []
        outer_cleanups = self.active_expression_cleanups
        try:
            if pipeline_input is not None:
                values.append(pipeline_input)
                lines, temp, value_cleanup = self._materialize(pipeline_input)
                prelude.extend(lines)
                names.append(temp)
                cleanup.extend(value_cleanup)
            for value_expression in argument_expressions:
                self.active_expression_cleanups = (
                    *outer_cleanups,
                    *cleanup,
                )
                value = self.expression(value_expression)
                values.append(value)
                lines, temp, value_cleanup = self._materialize(value)
                prelude.extend(lines)
                names.append(temp)
                cleanup.extend(value_cleanup)
        finally:
            self.active_expression_cleanups = outer_cleanups

        if name == "frame_buffer":
            return _CValue(
                tuple(prelude),
                f"aether_frame_buffer_create({names[0]})",
                expression.type,
                "owned",
                tuple(cleanup),
                True,
            )

        if name == "frame_buffer_len":
            return _CValue(
                tuple(prelude),
                f"aether_frame_buffer_len({names[0]})",
                expression.type,
                cleanup=tuple(cleanup),
            )

        if name == "len":
            prelude.append(f"(void){names[0]};")
            return _CValue(
                tuple(prelude),
                f"((int64_t)INT64_C({values[0].type.dimensions[0]}))",
                expression.type,
                cleanup=tuple(cleanup),
            )

        if name in {"sum", "mean", "max_abs"}:
            source_type = values[0].type
            count = self._tensor_size(source_type)
            scalar_type = self.c_type(expression.type)
            declaration = self._new_temp(expression.type)
            result = declaration.rsplit(" ", 1)[1]
            i = f"aether_i_{self.temp_index}"
            self.temp_index += 1
            initial = (
                self._absolute_code(f"{names[0]}.data[0]")
                if name == "max_abs"
                else "0.0"
            )
            prelude.append(f"{declaration} = ({scalar_type})({initial});")
            if name == "sum":
                prelude[-1] = f"{declaration} = 0.0;"
            prelude.append(
                f"for (size_t {i} = 0; {i} < {count}; ++{i}) {{"
            )
            item = f"{names[0]}.data[{i}]"
            if name == "sum":
                prelude.append(f"    {result} += {item};")
            elif name == "mean":
                prelude.append(f"    {result} += {item};")
            else:
                absolute = self._absolute_code(item)
                prelude.append(
                    f"    if ({absolute} > {result}) {result} = {absolute};"
                )
            prelude.append("}")
            if name == "mean":
                prelude.append(f"{result} /= ({scalar_type}){count};")
            return _CValue(
                tuple(prelude),
                result,
                expression.type,
                cleanup=tuple(cleanup),
            )

        if name == "convert":
            multiplier, addend = self._conversion_factor(expression)
            source_type = values[0].type
            if source_type.name == "Tensor":
                declaration = self._new_temp(expression.type)
                result = declaration.rsplit(" ", 1)[1]
                count = self._tensor_size(source_type)
                i = f"aether_i_{self.temp_index}"
                self.temp_index += 1
                prelude.append(f"{declaration} = {{0}};")
                prelude.append(
                    f"for (size_t {i} = 0; {i} < {count}; ++{i}) {{"
                )
                element_type = source_type.element_type
                assert element_type is not None
                prelude.append(
                    f"    {result}.data[{i}] = ({self.c_type(element_type)})"
                    f"({names[0]}.data[{i}] * ({multiplier}) + ({addend}));"
                )
                prelude.append("}")
            else:
                result = (
                    f"(({self.c_type(expression.type)})"
                    f"({names[0]} * ({multiplier}) + ({addend})))"
                )
            return _CValue(
                tuple(prelude),
                result,
                expression.type,
                cleanup=tuple(cleanup),
            )

        if name == "abs" and values[0].type.name != "Tensor":
            return _CValue(
                tuple(prelude),
                self._absolute_code(names[0]),
                expression.type,
                cleanup=tuple(cleanup),
            )

        if name in {"scale", "subtract"}:
            operator = "*" if name == "scale" else "-"
            synthetic = IRExpression(
                "binary", expression.span, expression.type, operator,
                children=tuple(
                    IRExpression("identifier", expression.span, value.type)
                    for value in values
                ),
            )
            left_name, right_name = names
            return self._tensor_binary(
                synthetic,
                left_name,
                right_name,
                prelude,
                tuple(cleanup),
            )

        if name in {"abs", "min", "max"}:
            result_type = expression.type
            declaration = self._new_temp(result_type)
            result = declaration.rsplit(" ", 1)[1]
            count = self._tensor_size(result_type)
            i = f"aether_i_{self.temp_index}"
            self.temp_index += 1
            prelude.append(f"{declaration} = {{0}};")
            prelude.append(f"for (size_t {i} = 0; {i} < {count}; ++{i}) {{")
            if name == "abs":
                item = f"{names[0]}.data[{i}]"
                value = self._absolute_code(item)
            else:
                left, right = f"{names[0]}.data[{i}]", f"{names[1]}.data[{i}]"
                comparison = "<" if name == "min" else ">"
                value = f"({right} {comparison} {left} ? {right} : {left})"
            prelude.append(f"    {result}.data[{i}] = {value};")
            prelude.append("}")
            return _CValue(
                tuple(prelude),
                result,
                result_type,
                cleanup=tuple(cleanup),
            )

        raise AssertionError(f"unhandled C99 builtin {name}")

    @staticmethod
    def _absolute_code(value: str) -> str:
        return f"({value} == 0.0 ? 0.0 : ({value} < 0.0 ? -{value} : {value}))"

    @staticmethod
    def _operation_name(operator: str) -> str:
        return {"+": "add", "-": "sub", "*": "mul", "/": "div", "%": "rem"}[operator]

    def _literal_code(self, expression: IRExpression) -> str:
        value = expression.value
        type_name = expression.type.name
        if type_name == "bool":
            return "1" if value == "true" else "0"
        if type_name in _INTEGER_TYPES:
            width = type_name[1:]
            macro = ("UINT" if type_name.startswith("u") else "INT") + f"{width}_C"
            return f"(({_C_TYPES[type_name]}){macro}({value}))"
        if type_name == "f32":
            return value + ("" if value[-1:] in "fF" else "f")
        return value

    def emit(self) -> str:
        self.validate()
        if self.diagnostics:
            raise C99LoweringError(tuple(self.diagnostics))

        output = [
            "/* Generated deterministically by the Aether C99 backend. */",
            "#if defined(__clang__)",
            "#pragma STDC FP_CONTRACT OFF",
            "#endif",
            "#include <stdint.h>",
            "#include <stdbool.h>",
            "#include <limits.h>",
            "#include <stddef.h>",
        ]
        if self.tensor_types:
            output.append("")
            for name, type_ in sorted(self.tensor_types.items()):
                element = type_.element_type
                assert element is not None
                output.extend(
                    [
                        f"typedef struct {{ {self.c_type(element)} "
                        f"data[{self._tensor_size(type_)}]; }} {name};",
                    ]
                )
            output.append("")
        if self.needs_frame_buffer:
            output.extend(
                [
                    "typedef struct {",
                    "    size_t refcount;",
                    "    size_t length;",
                    "    unsigned char data[];",
                    "} aether_frame_storage;",
                    "typedef struct {",
                    "    aether_frame_storage *storage;",
                    "} aether_frame_buffer;",
                    "",
                ]
            )
        if self.result_types:
            for name, type_ in sorted(self.result_types.items()):
                value_type, _ = type_.arguments
                output.extend(
                    [
                        f"typedef struct {{",
                        "    int is_ok;",
                        f"    {self.c_type(value_type)} value;",
                        "    int error;",
                        f"}} {name};",
                        "",
                    ]
                )
        if self.needs_trap:
            output.extend(("#include <stdlib.h>", ""))
            if self.needs_frame_buffer:
                output.extend(
                    [
                        "#ifndef AETHER_FRAME_ALLOC",
                        "#define AETHER_FRAME_ALLOC malloc",
                        "#endif",
                        "#ifndef AETHER_FRAME_FREE",
                        "#define AETHER_FRAME_FREE free",
                        "#endif",
                        "#ifndef AETHER_FRAME_SIZE_MAX",
                        "#define AETHER_FRAME_SIZE_MAX SIZE_MAX",
                        "#endif",
                        "",
                    ]
                )
            output.extend(self._emit_helpers())
        else:
            output.append("")
        if self.gradient_targets:
            output.extend(
                [
                    "typedef struct {",
                    "    double value;",
                    "    size_t parent_left;",
                    "    size_t parent_right;",
                    "    double partial_left;",
                    "    double partial_right;",
                    "    double adjoint;",
                    "    bool has_parent_left;",
                    "    bool has_parent_right;",
                    "} aether_ad_node;",
                    "",
                ]
            )
        for function in self.module.functions:
            output.append(
                self._extern_signature(function)
                if function.external_abi is not None
                else self._function_signature(function) + ";"
            )
        if self.module.functions:
            output.append("")
        for target_name in sorted(self.gradient_targets):
            output.extend(
                self._emit_gradient_helper(self.functions[target_name])
            )
            output.append("")
        for function in self.defined_functions:
            if function.external_abi is not None:
                continue
            output.extend(self._emit_function(function))
            output.append("")
        if self.include_entry_point:
            output.extend(
                [
                    "int main(void) {",
                    "    " + (
                        "return aether_fn_6d61696e();"
                        if self.functions["main"].return_type.name == "i32"
                        else "aether_fn_6d61696e();\n    return 0;"
                    ),
                    "}",
                    "",
                ]
            )
        return "\n".join(output).rstrip() + "\n"

    def _function_signature(self, function: IRFunction) -> str:
        return_type = self.c_type(function.return_type)
        parameters = ", ".join(
            f"{self.c_type(parameter.type)} "
            f"{_encoded_identifier('aether_v_', parameter.name)}"
            for parameter in function.parameters
        ) or "void"
        name = _encoded_identifier("aether_fn_", function.name)
        return f"{return_type} {name}({parameters})"

    def _extern_signature(self, function: IRFunction) -> str:
        return_type = self.c_type(function.return_type)
        parameters = ", ".join(
            f"{self.c_type(parameter.type)} "
            f"{_encoded_identifier('aether_v_', parameter.name)}"
            for parameter in function.parameters
        ) or "void"
        return f"extern {return_type} {function.name}({parameters});"

    def _call_target(self, name: str) -> str:
        function = self.functions[name]
        if function.external_abi is not None:
            return function.name
        return _encoded_identifier("aether_fn_", name)

    def _emit_function(self, function: IRFunction) -> list[str]:
        self.temp_index = 0
        self.current_function = function
        has_try = any(
            statement.expression is not None
            and self._expression_contains_try(statement.expression)
            for statement in function.body
        )
        self.frame_return_name = "aether_frame_return"
        self.frame_cleanup_label = "aether_frame_cleanup"
        has_cleanup = (
            has_try
            or bool(function.frame_resources)
            or self._contains_frame_buffer(function.return_type)
        )
        lines = [self._function_signature(function) + " {"]
        if self.needs_frame_buffer:
            lines.extend(
                [
                    "    (void)&aether_frame_buffer_create;",
                    "    (void)&aether_frame_buffer_retain;",
                    "    (void)&aether_frame_buffer_release;",
                    "    (void)&aether_frame_buffer_len;",
                ]
            )
        if has_cleanup and function.return_type.name != "void":
            lines.append(
                f"    {self.c_type(function.return_type)} "
                f"{self.frame_return_name} = {{0}};"
            )
        resource_types = dict(function.frame_resource_types)
        for resource_name in function.frame_resources:
            lines.append(
                f"    {self.c_type(resource_types[resource_name])} "
                f"{_encoded_identifier('aether_v_', resource_name)} = {{0}};"
            )
        for parameter in function.parameters:
            name = _encoded_identifier("aether_v_", parameter.name)
            lines.append(f"    (void)({name});")
        for statement in function.body:
            expression = statement.expression
            if expression is None:
                if statement.kind == "return":
                    lines.append(
                        f"    goto {self.frame_cleanup_label};"
                        if has_cleanup
                        else "    return;"
                    )
                continue
            value = self.expression(expression)
            lines.extend("    " + item for item in value.prelude)
            if statement.kind == "let":
                name = _encoded_identifier("aether_v_", statement.name)
                if self._contains_frame_buffer(expression.type):
                    lines.append(
                        f"    {self.c_type(expression.type)} "
                        f"aether_owner_{self.temp_index} = {value.code};"
                    )
                    owner_temp = f"aether_owner_{self.temp_index}"
                    self.temp_index += 1
                    if value.ownership == "borrowed":
                        retain = self._retain_code(owner_temp, expression.type)
                        if retain:
                            lines.append(f"    {retain}")
                    lines.append(f"    {name} = {owner_temp};")
                    lines.extend("    " + item for item in value.cleanup)
                else:
                    qualifier = "const " if statement.binding_kind in {"let", "const"} else ""
                    lines.append(
                        f"    {qualifier}{self.c_type(expression.type)} "
                        f"{name} = {value.code};"
                    )
                    lines.extend("    " + item for item in value.cleanup)
            elif statement.kind == "return":
                if function.return_type.name == "void":
                    lines.append(f"    (void)({value.code});")
                    lines.extend("    " + item for item in value.cleanup)
                    lines.append(
                        f"    goto {self.frame_cleanup_label};"
                        if has_cleanup
                        else "    return;"
                    )
                elif has_cleanup:
                    if (
                        function.return_type.name == "Result"
                        and value.type.name != "Result"
                    ):
                        lines.append(
                            f"    {self.frame_return_name} = "
                            f"({self.c_type(function.return_type)})"
                            f"{{1, {value.code}, 0}};"
                        )
                    else:
                        lines.append(
                            f"    {self.frame_return_name} = {value.code};"
                        )
                    if (
                        self._contains_frame_buffer(value.type)
                        and value.ownership == "borrowed"
                    ):
                        retain = self._retain_code(
                            self.frame_return_name,
                            function.return_type,
                        )
                        if retain:
                            lines.append(f"    {retain}")
                    lines.extend("    " + item for item in value.cleanup)
                    lines.append(f"    goto {self.frame_cleanup_label};")
                else:
                    if value.cleanup:
                        result_temp = self._new_temp(value.type)
                        result_name = result_temp.rsplit(" ", 1)[1]
                        lines.append(
                            f"    {result_temp} = {value.code};"
                        )
                        lines.extend("    " + item for item in value.cleanup)
                        lines.append(f"    return {result_name};")
                    else:
                        lines.append(f"    return {value.code};")
            elif statement.kind == "assignment":
                name = _encoded_identifier("aether_v_", statement.name)
                if self._contains_frame_buffer(expression.type):
                    temporary = f"aether_owner_{self.temp_index}"
                    self.temp_index += 1
                    lines.append(
                        f"    {self.c_type(expression.type)} {temporary} = "
                        f"{value.code};"
                    )
                    lines.extend("    " + item for item in value.cleanup)
                    if value.ownership == "borrowed":
                        retain = self._retain_code(temporary, expression.type)
                        if retain:
                            lines.append(f"    {retain}")
                    release = self._release_code(name, expression.type)
                    if release:
                        lines.append(f"    {release}")
                    lines.append(f"    {name} = {temporary};")
                else:
                    lines.append(f"    {name} = {value.code};")
                    lines.extend("    " + item for item in value.cleanup)
            elif statement.kind == "expression":
                if self._contains_frame_buffer(value.type) and value.ownership == "owned":
                    temporary = f"aether_owner_{self.temp_index}"
                    self.temp_index += 1
                    lines.append(
                        f"    {self.c_type(value.type)} {temporary} = {value.code};"
                    )
                    release = self._release_code(temporary, value.type)
                    if release:
                        lines.append(f"    {release}")
                else:
                    lines.append(f"    (void)({value.code});")
                lines.extend("    " + item for item in value.cleanup)
        if function.return_type.name == "void" and (
            not function.body or function.body[-1].kind != "return"
        ):
            lines.append(
                f"    goto {self.frame_cleanup_label};"
                if has_cleanup
                else "    return;"
            )
        if has_cleanup:
            lines.extend([f"{self.frame_cleanup_label}:", "    ;"])
            for resource_name in reversed(function.frame_resources):
                name = _encoded_identifier("aether_v_", resource_name)
                resource_type = resource_types[resource_name]
                release = self._release_code(
                    name,
                    resource_type,
                    skip_empty=True,
                )
                if release:
                    lines.append(f"    {release}")
            lines.append(
                "    return;"
                if function.return_type.name == "void"
                else f"    return {self.frame_return_name};"
            )
        lines.append("}")
        self.current_function = None
        return lines

    @classmethod
    def _expression_contains_try(cls, expression: IRExpression) -> bool:
        return expression.kind == "try" or any(
            cls._expression_contains_try(child) for child in expression.children
        ) or any(
            cls._expression_contains_try(arm.value) for arm in expression.arms
        )

    def _emit_helpers(self) -> list[str]:
        lines = ["static void aether_trap(void) { abort(); }"]
        if self.needs_frame_buffer:
            lines.extend(
                [
                    "",
                    "static aether_frame_buffer aether_frame_buffer_create(",
                    "    uint64_t requested",
                    ") {",
                    "    size_t size = (size_t)requested;",
                    "    uint64_t maximum = (uint64_t)AETHER_FRAME_SIZE_MAX;",
                    "    if (requested == 0 || requested > maximum ||",
                    "        (uint64_t)size != requested) {",
                    "        aether_trap();",
                    "    }",
                    "    if (size > SIZE_MAX - sizeof(aether_frame_storage)) {",
                    "        aether_trap();",
                    "    }",
                    "    aether_frame_storage *storage =",
                    "        (aether_frame_storage *)AETHER_FRAME_ALLOC(",
                    "            sizeof(aether_frame_storage) + size",
                    "        );",
                    "    if (storage == NULL) {",
                    "        aether_trap();",
                    "    }",
                    "    storage->refcount = 1;",
                    "    storage->length = size;",
                    "    aether_frame_buffer result = {storage};",
                    "    return result;",
                    "}",
                    "",
                    "static inline void aether_frame_buffer_retain(",
                    "    aether_frame_buffer *buffer",
                    ") {",
                    "    if (buffer == NULL || buffer->storage == NULL ||",
                    "        buffer->storage->refcount == 0 ||",
                    "        buffer->storage->refcount == SIZE_MAX) {",
                    "        aether_trap();",
                    "    }",
                    "    ++buffer->storage->refcount;",
                    "}",
                    "",
                    "static inline void aether_frame_buffer_release(",
                    "    aether_frame_buffer *buffer",
                    ") {",
                    "    if (buffer == NULL || buffer->storage == NULL ||",
                    "        buffer->storage->refcount == 0) {",
                    "        aether_trap();",
                    "    }",
                    "    aether_frame_storage *storage = buffer->storage;",
                    "    buffer->storage = NULL;",
                    "    --storage->refcount;",
                    "    if (storage->refcount == 0) {",
                    "        AETHER_FRAME_FREE(storage);",
                    "    }",
                    "}",
                    "",
                    "static inline uint64_t aether_frame_buffer_len(",
                    "    aether_frame_buffer buffer",
                    ") {",
                    "    if (buffer.storage == NULL ||",
                    "        buffer.storage->refcount == 0) {",
                    "        aether_trap();",
                    "    }",
                    "    return (uint64_t)buffer.storage->length;",
                    "}",
                ]
            )
        for type_name, operation in sorted(self.helpers):
            ctype = _C_TYPES[type_name]
            fn = f"aether_{type_name}_{operation}"
            if type_name in {"f32", "f64"} and operation == "div":
                continue
            if operation == "div" and type_name in _INTEGER_TYPES:
                guard = "right == 0"
                if type_name in _SIGNED_TYPES:
                    width = type_name[1:]
                    guard += f" || (left == INT{width}_MIN && right == -1)"
                lines.extend(
                    [
                        f"static {ctype} {fn}({ctype} left, {ctype} right) {{",
                        f"    if ({guard}) aether_trap();",
                        "    return left / right;",
                        "}",
                    ]
                )
                continue
            if operation == "rem":
                guard = "right == 0"
                if type_name in _SIGNED_TYPES:
                    width = type_name[1:]
                    guard += f" || (left == INT{width}_MIN && right == -1)"
                lines.extend(
                    [
                        f"static {ctype} {fn}({ctype} left, {ctype} right) {{",
                        f"    if ({guard}) aether_trap();",
                        "    return left % right;",
                        "}",
                    ]
                )
                continue
            if operation == "neg":
                width = type_name[1:]
                lines.extend(
                    [
                        f"static {ctype} {fn}({ctype} value) {{",
                        f"    if (value == INT{width}_MIN) aether_trap();",
                        f"    return ({ctype})(-value);",
                        "}",
                    ]
                )
                continue
            if type_name in _SIGNED_TYPES:
                width = type_name[1:]
                if operation == "add":
                    guard = (
                        f"(right > 0 && left > INT{width}_MAX - right) || "
                        f"(right < 0 && left < INT{width}_MIN - right)"
                    )
                    op = "+"
                elif operation == "sub":
                    guard = (
                        f"(right < 0 && left > INT{width}_MAX + right) || "
                        f"(right > 0 && left < INT{width}_MIN + right)"
                    )
                    op = "-"
                else:
                    guard = (
                        f"(left > 0 && ((right > 0 && left > INT{width}_MAX / right) || "
                        f"(right < 0 && right < INT{width}_MIN / left))) || "
                        f"(left < 0 && ((right > 0 && left < INT{width}_MIN / right) || "
                        f"(right < 0 && left < INT{width}_MAX / right)))"
                    )
                    op = "*"
            else:
                width = type_name[1:]
                if operation == "add":
                    guard, op = f"left > UINT{width}_MAX - right", "+"
                elif operation == "sub":
                    guard, op = "left < right", "-"
                else:
                    guard = f"right != 0 && left > UINT{width}_MAX / right"
                    op = "*"
            lines.extend(
                [
                    f"static {ctype} {fn}({ctype} left, {ctype} right) {{",
                    f"    if ({guard}) aether_trap();",
                    f"    return ({ctype})(left {op} right);",
                    "}",
                ]
            )
        float_helpers = [
            type_name for type_name, operation in sorted(self.helpers)
            if operation == "div" and type_name in {"f32", "f64"}
        ]
        for type_name in float_helpers:
            ctype = _C_TYPES[type_name]
            lines.extend(
                [
                    f"static {ctype} aether_{type_name}_div({ctype} left, {ctype} right) {{",
                    "    if (right == 0.0) aether_trap();",
                    "    return left / right;",
                    "}",
                ]
            )
        lines.append("")
        return lines


def emit_c99_module(
    module: IRModule,
    *,
    include_entry_point: bool = False,
) -> str:
    """Emit deterministic C99 for the supported checked-scalar subset."""
    return _C99Emitter(
        module,
        include_entry_point=include_entry_point,
    ).emit()


def emit_c99_module_units(
    module: IRModule,
    entry_source_path: str,
) -> tuple[tuple[str, str], ...]:
    """Emit one C99 translation unit per source file in a project."""
    entry_path = Path(entry_source_path).expanduser().resolve()
    functions_by_path: dict[Path, list[IRFunction]] = {
        Path(source_path).expanduser().resolve(): []
        for source_path in module.source_paths
    }
    for function in module.functions:
        if function.span.source_path is None:
            raise C99LoweringError(
                (
                    Diagnostic(
                        "E_C99_SOURCE",
                        "Project function is missing its source file path",
                        function.span,
                    ),
                )
            )
        source_path = Path(function.span.source_path).expanduser().resolve()
        functions_by_path.setdefault(source_path, []).append(function)

    if entry_path not in functions_by_path:
        span = (
            module.functions[0].span
            if module.functions
            else SourceSpan(0, 0, 1, 1, str(entry_path))
        )
        raise C99LoweringError(
            (
                Diagnostic(
                    "E_C99_SOURCE",
                    "Project entry file has no functions to emit",
                    span,
                ),
            )
        )

    ordered_paths = sorted(
        functions_by_path,
        key=lambda path: (path != entry_path, path.as_posix()),
    )
    units = []
    for source_path in ordered_paths:
        try:
            relative_path = source_path.relative_to(entry_path.parent).as_posix()
        except ValueError as error:
            span = (
                functions_by_path[source_path][0].span
                if functions_by_path[source_path]
                else SourceSpan(0, 0, 1, 1, str(source_path))
            )
            raise C99LoweringError(
                (
                    Diagnostic(
                        "E_C99_SOURCE",
                        "Project module is outside the entry source directory",
                        span,
                    ),
                )
            ) from error
        emitted = _C99Emitter(
            module,
            include_entry_point=source_path == entry_path,
            defined_functions=tuple(functions_by_path[source_path]),
        ).emit()
        units.append((relative_path, emitted))
    return tuple(units)
