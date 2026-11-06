"""Deterministic C99 emission for a bounded, checked scalar subset."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

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
    ) -> None:
        self.module = module
        self.include_entry_point = include_entry_point
        self.functions = {function.name: function for function in module.functions}
        self.diagnostics: list[Diagnostic] = []
        self.helpers: set[tuple[str, str]] = set()
        self.needs_trap = False
        self.temp_index = 0

    def error(self, code: str, message: str, span: SourceSpan) -> None:
        self.diagnostics.append(Diagnostic(code, message, span))

    def validate_type(self, type_: AetherType, span: SourceSpan, role: str) -> None:
        if (
            (
                type_.name not in _C_TYPES
                or (
                    type_.name == "void"
                    and role not in {
                        "return",
                        "expression",
                        "FFI return",
                    }
                )
            )
            or _has_unit(type_)
        ):
            self.error(
                "E_C99_TYPE",
                f"C99 backend does not support {role} type {type_.display()}",
                span,
            )

    def validate_expression(
        self,
        expression: IRExpression,
        *,
        allow_signed_min_magnitude: bool = False,
    ) -> None:
        kind = expression.kind
        self.validate_type(expression.type, expression.span, "expression")
        if kind == "literal":
            self._validate_literal(expression, allow_signed_min_magnitude)
            return
        if kind == "identifier":
            return
        if kind == "group":
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
            elif operator in {"/", "%"}:
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
            elif operator in {"+", "-", "*"}:
                operand_type = expression.type.name
                if operand_type in _INTEGER_TYPES:
                    operation = {"+": "add", "-": "sub", "*": "mul"}[operator]
                    self.helpers.add((operand_type, operation))
                    self.needs_trap = True
            self.validate_expression(expression.children[0])
            self.validate_expression(expression.children[1])
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
        argument_types = (
            (pipeline_input.type,) if pipeline_input is not None else ()
        ) + tuple(argument.type for argument in expression.children[1:])
        self._validate_call_target(callee.value, expression.span, argument_types)
        for argument in expression.children[1:]:
            self.validate_expression(argument)

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
            for statement in function.body:
                if statement.expression is not None:
                    if statement.kind == "let" and statement.expression.type.name == "void":
                        self.error(
                            "E_C99_BINDING",
                            "A void function call cannot initialize a local binding",
                            statement.span,
                        )
                    self.validate_expression(statement.expression)
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

    def _new_temp(self, type_: AetherType) -> str:
        name = f"aether_tmp_{self.temp_index}"
        self.temp_index += 1
        return f"{_C_TYPES[type_.name]} {name}"

    def _materialize(self, value: _CValue) -> tuple[list[str], str]:
        declaration = self._new_temp(value.type)
        name = declaration.rsplit(" ", 1)[1]
        return [*value.prelude, f"{declaration} = {value.code};"], name

    def expression(self, expression: IRExpression) -> _CValue:
        kind = expression.kind
        if kind == "literal":
            return _CValue((), self._literal_code(expression), expression.type)
        if kind == "identifier":
            return _CValue(
                (),
                _encoded_identifier("aether_v_", expression.value),
                expression.type,
            )
        if kind == "group":
            child = self.expression(expression.children[0])
            return _CValue(child.prelude, f"({child.code})", expression.type)
        if kind == "unary":
            operand = self.expression(expression.children[0])
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
            return _CValue(operand.prelude, code, expression.type)
        if kind == "binary":
            if expression.value == "|>":
                return self._pipeline_expression(expression)
            if expression.value in {"&&", "||"}:
                return self._logical_expression(expression)
            left = self.expression(expression.children[0])
            right = self.expression(expression.children[1])
            prelude, left_name = self._materialize(left)
            right_prelude, right_name = self._materialize(right)
            prelude.extend(right_prelude)
            operator = expression.value
            type_name = expression.type.name
            if operator in {"+", "-", "*"} and type_name in _INTEGER_TYPES:
                code = f"aether_{type_name}_{self._operation_name(operator)}({left_name}, {right_name})"
            elif operator in {"/", "%"} and type_name in _INTEGER_TYPES:
                code = f"aether_{type_name}_{self._operation_name(operator)}({left_name}, {right_name})"
            elif operator == "/" and type_name in {"f32", "f64"}:
                code = f"aether_{type_name}_div({left_name}, {right_name})"
            else:
                code = f"({left_name} {operator} {right_name})"
            return _CValue(tuple(prelude), code, expression.type)
        if kind == "call":
            return self._call_expression(expression)
        raise C99LoweringError(
            (Diagnostic("E_C99_EXPRESSION", f"Unsupported IR expression {kind!r}", expression.span),)
        )

    def _logical_expression(self, expression: IRExpression) -> _CValue:
        left = self.expression(expression.children[0])
        prelude, name = self._materialize(left)
        right = self.expression(expression.children[1])
        condition = name if expression.value == "&&" else f"!{name}"
        right_lines, right_name = self._materialize(right)
        prelude.append(f"if ({condition}) {{")
        prelude.extend("  " + line for line in right_lines)
        prelude.append(f"  {name} = {right_name};")
        prelude.append("}")
        return _CValue(tuple(prelude), name, expression.type)

    def _call_expression(self, expression: IRExpression) -> _CValue:
        callee = expression.children[0]
        arguments = [self.expression(item) for item in expression.children[1:]]
        prelude: list[str] = []
        names: list[str] = []
        target = self.functions[callee.value]
        for argument in arguments:
            argument_lines, name = self._materialize(argument)
            prelude.extend(argument_lines)
            names.append(name)
        return _CValue(
            tuple(prelude),
            f"{self._call_target(callee.value)}({', '.join(names)})",
            target.return_type,
        )

    def _pipeline_expression(self, expression: IRExpression) -> _CValue:
        left, right = expression.children
        left_value = self.expression(left)
        prelude, left_name = self._materialize(left_value)
        if right.kind == "identifier":
            target_name = right.value
            arguments: list[_CValue] = []
        else:
            target_name = right.children[0].value
            arguments = [self.expression(item) for item in right.children[1:]]
        names = [left_name]
        for argument in arguments:
            argument_lines, name = self._materialize(argument)
            prelude.extend(argument_lines)
            names.append(name)
        target = self.functions[target_name]
        return _CValue(
            tuple(prelude),
            f"{self._call_target(target_name)}({', '.join(names)})",
            target.return_type,
        )

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
            "#include <stdint.h>",
            "#include <stdbool.h>",
            "#include <limits.h>",
        ]
        if self.needs_trap:
            output.extend(("#include <stdlib.h>", ""))
            output.extend(self._emit_helpers())
        else:
            output.append("")
        for function in self.module.functions:
            output.append(
                self._extern_signature(function)
                if function.external_abi is not None
                else self._function_signature(function) + ";"
            )
        if self.module.functions:
            output.append("")
        for function in self.module.functions:
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
        return_type = _C_TYPES[function.return_type.name]
        parameters = ", ".join(
            f"{_C_TYPES[parameter.type.name]} "
            f"{_encoded_identifier('aether_v_', parameter.name)}"
            for parameter in function.parameters
        ) or "void"
        name = _encoded_identifier("aether_fn_", function.name)
        return f"{return_type} {name}({parameters})"

    def _extern_signature(self, function: IRFunction) -> str:
        return_type = _C_TYPES[function.return_type.name]
        parameters = ", ".join(
            f"{_C_TYPES[parameter.type.name]} "
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
        lines = [self._function_signature(function) + " {"]
        for parameter in function.parameters:
            name = _encoded_identifier("aether_v_", parameter.name)
            lines.append(f"    (void)({name});")
        for statement in function.body:
            expression = statement.expression
            if expression is None:
                if statement.kind == "return":
                    lines.append("    return;")
                continue
            value = self.expression(expression)
            lines.extend("    " + item for item in value.prelude)
            if statement.kind == "let":
                qualifier = "const " if statement.binding_kind in {"let", "const"} else ""
                name = _encoded_identifier("aether_v_", statement.name)
                lines.append(
                    f"    {qualifier}{_C_TYPES[expression.type.name]} "
                    f"{name} = {value.code};"
                )
            elif statement.kind == "return":
                if function.return_type.name == "void":
                    lines.append(f"    (void)({value.code});")
                    lines.append("    return;")
                else:
                    lines.append(f"    return {value.code};")
            elif statement.kind == "expression":
                lines.append(f"    (void)({value.code});")
        if function.return_type.name == "void" and (
            not function.body or function.body[-1].kind != "return"
        ):
            lines.append("    return;")
        lines.append("}")
        return lines

    def _emit_helpers(self) -> list[str]:
        lines = ["static void aether_trap(void) { abort(); }"]
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
