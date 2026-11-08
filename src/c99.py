"""Deterministic C99 emission for bounded scalar and floating tensor subsets."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math
import json
import re
from pathlib import Path

from src.frontend import Diagnostic, SourceSpan
from src.fusion import fuse_elementwise_tensors
from src.ir import IRExpression, IRFunction, IRModule, IRStatement, IRTypeDeclaration
from src.semantic import AetherType
from src.runtime import ArgsErrorCode, FsErrorCode, IoErrorCode


_C_TYPES = {
    "void": "void",
    "str": "aether_string",
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
    "&", "|", "^", "<<", ">>",
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


def _walk_ir_statements(
    statements: tuple[IRStatement, ...],
) -> Iterator[IRStatement]:
    for statement in statements:
        yield statement
        yield from _walk_ir_statements(statement.body)
        yield from _walk_ir_statements(statement.else_body)


def _statement_expressions(statement: IRStatement) -> tuple[IRExpression, ...]:
    expressions = list(statement.expressions)
    if statement.expression is not None:
        expressions.append(statement.expression)
    return tuple(expressions)


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
        self.type_declarations = {
            declaration.name: declaration for declaration in module.types
        }
        self.include_entry_point = include_entry_point
        self.shared_runtime = defined_functions is not None
        self.target_mcu = module.target == "mcu"
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
        self.collection_types: dict[str, AetherType] = {}
        self.result_types: dict[str, AetherType] = {}
        self.current_function: IRFunction | None = None
        self.frame_return_name = ""
        self.frame_cleanup_label = ""
        self.active_expression_cleanups: tuple[str, ...] = ()
        self.collection_slots_name = ""
        self.collection_slot_count = 0
        self.collection_slot_next = 0
        self.collection_bindings: dict[str, list[int]] = {}
        self.collection_scope_stack: list[list[str]] = []
        self.collection_scope_slots: list[list[int]] = []
        self.collection_loop_scope_depths: list[int] = []
        self.needs_string = False
        self.needs_host_io = False
        self.needs_host_fs = False
        self.needs_process_args = False
        self.runtime_calls: set[str] = set()
        self.module_runtime_calls: set[str] = set()
        for function in module.functions:
            self._collect_type(function.return_type)
            self._collect_tensor_type(function.return_type)
            for parameter in function.parameters:
                if (
                    function.external_library is not None
                    and parameter.type.name in {"Slice", "SliceMut"}
                ):
                    continue
                self._collect_type(parameter.type)
                self._collect_tensor_type(parameter.type)
            for statement in _walk_ir_statements(function.body):
                for expression in _statement_expressions(statement):
                    self._collect_expression_types(expression)
                    self._collect_expression_tensor_types(expression)
                    self._collect_runtime_call_metadata(expression)
        for declaration in module.types:
            for _, field_type in declaration.fields:
                self._collect_type(field_type)
                self._collect_tensor_type(field_type)

    def _collect_runtime_call_metadata(self, expression: IRExpression) -> None:
        if expression.kind == "call" and expression.children:
            callee = expression.children[0]
            if callee.kind in {"identifier", "name"} and callee.value in {
                "io_write",
                "io_read_line",
                "fs_read_text",
                "fs_write_text",
                "fs_exists",
                "args_len",
                "args_get",
            }:
                self.module_runtime_calls.add(callee.value)
                if callee.value.startswith("io_"):
                    self.needs_host_io = True
                elif callee.value.startswith("fs_"):
                    self.needs_host_fs = True
                else:
                    self.needs_process_args = True
        for child in expression.children:
            self._collect_runtime_call_metadata(child)
        for arm in expression.arms:
            self._collect_runtime_call_metadata(arm.value)

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

    def _type_token(self, type_: AetherType) -> str:
        if type_.name in _C_TYPES:
            return type_.name
        if type_.name == "Array":
            return self._collection_c_name(type_).removeprefix("aether_array_")
        if type_.name == "List":
            return self._collection_c_name(type_).removeprefix("aether_list_")
        if type_.name == "Slice":
            return self._collection_c_name(type_).removeprefix("aether_slice_")
        if type_.name in self.type_declarations:
            return type_.name.encode("utf-8").hex()
        return type_.name

    def _result_c_name(self, type_: AetherType) -> str:
        if type_.name != "Result" or len(type_.arguments) != 2:
            return "aether_result_invalid"
        value_type, error_type = type_.arguments
        value_token = self._type_token(value_type)
        error_token = self._type_token(error_type)
        return f"aether_result_{value_token}_{error_token}"

    def _result_value_c_type(self, type_: AetherType) -> str:
        return "uint8_t" if type_.name == "void" else self.c_type(type_)

    def _collection_c_name(self, type_: AetherType) -> str:
        if not type_.arguments:
            return "aether_collection_invalid"
        element = self.c_type(type_.arguments[0])
        token = (
            element.removeprefix("aether_type_")
            .replace("aether_tensor_", "tensor_")
            .replace("aether_array_", "array_")
            .replace("aether_list_", "list_")
            .replace("aether_slice_", "slice_")
        )
        if type_.name == "Array":
            dimensions = "_".join(str(value) for value in type_.dimensions)
            return f"aether_array_{token}_{dimensions}"
        if type_.name == "List":
            return f"aether_list_{token}"
        return f"aether_slice_{token}"

    def _collect_type(self, type_: AetherType) -> None:
        if type_.name == "str":
            self.needs_string = True
        if type_.name in {"Array", "List", "Slice"}:
            self.collection_types.setdefault(
                self._collection_c_name(type_),
                type_,
            )
            if type_.name in {"List", "Slice"}:
                self.needs_trap = True
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
        if type_.name in {"Array", "List", "Slice"}:
            return self._collection_c_name(type_)
        if type_.name in self.type_declarations:
            return _encoded_identifier("aether_type_", type_.name)
        return _C_TYPES.get(type_.name, "void")

    @staticmethod
    def _contains_frame_buffer(type_: AetherType) -> bool:
        return type_.name == "FrameBuffer" or any(
            _C99Emitter._contains_frame_buffer(argument)
            for argument in type_.arguments
        )

    @staticmethod
    def _contains_owned_collection(type_: AetherType) -> bool:
        return type_.name in {"List", "Slice"} or (
            type_.name == "Result"
            and bool(type_.arguments)
            and type_.arguments[0].name in {"List", "Slice"}
        )

    def _collection_storage_access(self, name: str, type_: AetherType) -> str:
        if type_.name in {"List", "Slice"}:
            return f"{name}.storage"
        if (
            type_.name == "Result"
            and len(type_.arguments) == 2
            and type_.arguments[0].name in {"List", "Slice"}
        ):
            return f"({name}.is_ok ? {name}.value.storage : NULL)"
        raise ValueError(f"{type_.display()} does not own collection storage")

    def _enter_collection_scope(self) -> None:
        self.collection_scope_stack.append([])
        self.collection_scope_slots.append([])

    def _bind_collection_slot(self, name: str) -> int:
        slot = self.collection_slot_next
        self.collection_slot_next += 1
        self.collection_bindings.setdefault(name, []).append(slot)
        if not self.collection_scope_stack:
            raise RuntimeError("collection slot requires an active lexical scope")
        self.collection_scope_stack[-1].append(name)
        self.collection_scope_slots[-1].append(slot)
        return slot

    def _lookup_collection_slot(self, name: str) -> int | None:
        slots = self.collection_bindings.get(name)
        return slots[-1] if slots else None

    def _leave_collection_scope(self, indentation: str) -> list[str]:
        if not self.collection_scope_stack:
            raise RuntimeError("collection scope stack is empty")
        names = self.collection_scope_stack.pop()
        slots = self.collection_scope_slots.pop()
        lines: list[str] = []
        for name, slot in zip(reversed(names), reversed(slots)):
            lines.append(
                indentation
                + f"aether_owned_slot_release(&{self.collection_slots_name}[{slot}]);"
            )
            self.collection_bindings[name].pop()
            if not self.collection_bindings[name]:
                del self.collection_bindings[name]
        return lines

    def _collection_slot_set_lines(
        self,
        slot: int,
        name: str,
        type_: AetherType,
        indentation: str,
    ) -> list[str]:
        collection_type = (
            type_.arguments[0]
            if type_.name == "Result"
            and type_.arguments
            and type_.arguments[0].name in {"List", "Slice"}
            else type_
        )
        return [
            f"{indentation}{self.collection_slots_name}[{slot}].storage = "
            f"{self._collection_storage_access(name, type_)};",
            f"{indentation}{self.collection_slots_name}[{slot}].release = "
            f"aether_collection_release_{self._collection_c_name(collection_type)};",
        ]

    def _collection_scope_release_lines(
        self,
        indentation: str,
        start_depth: int = 0,
    ) -> list[str]:
        lines: list[str] = []
        for scope in reversed(self.collection_scope_slots[start_depth:]):
            for slot in reversed(scope):
                lines.append(
                    f"{indentation}aether_owned_slot_release("
                    f"&{self.collection_slots_name}[{slot}]);"
                )
        return lines

    def _retain_code(self, name: str, type_: AetherType) -> str:
        if type_.name == "FrameBuffer":
            return f"aether_frame_buffer_retain(&{name});"
        if type_.name in {"List", "Slice"}:
            return (
                f"if ({name}.storage != NULL) {{ "
                f"if ({name}.storage->references == SIZE_MAX) aether_trap(); "
                f"++{name}.storage->references; }}"
            )
        if (
            type_.name == "Result"
            and len(type_.arguments) == 2
            and type_.arguments[0].name == "FrameBuffer"
        ):
            return (
                f"if ({name}.is_ok) "
                f"aether_frame_buffer_retain(&{name}.value);"
            )
        collection_type = (
            type_.arguments[0]
            if type_.name == "Result"
            and len(type_.arguments) == 2
            and type_.arguments[0].name in {"List", "Slice"}
            else type_
            if type_.name in {"List", "Slice"}
            else None
        )
        if collection_type is not None:
            value = f"{name}.value" if type_.name == "Result" else name
            condition = (
                f"{name}.is_ok && {value}.storage != NULL"
                if type_.name == "Result"
                else f"{name}.storage != NULL"
            )
            return (
                f"if ({condition}) {{ "
                f"if ({value}.storage->references == SIZE_MAX) aether_trap(); "
                f"++{value}.storage->references; }}"
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
        if type_.name in {"List", "Slice"}:
            return (
                f"if ({name}.storage != NULL) "
                f"aether_collection_release_{self._collection_c_name(type_)}"
                f"({name}.storage);"
            )
        if (
            type_.name == "Result"
            and len(type_.arguments) == 2
            and type_.arguments[0].name in {"List", "Slice"}
        ):
            collection_type = type_.arguments[0]
            return (
                f"if ({name}.is_ok && {name}.value.storage != NULL) "
                f"aether_collection_release_{self._collection_c_name(collection_type)}"
                f"({name}.value.storage);"
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
        if self.target_mcu and type_.name == "FrameBuffer":
            self.error(
                "E_C99_TARGET_ALLOCATION",
                "FrameBuffer requires dynamic allocation and is unavailable for @target(mcu)",
                span,
            )
        if self.target_mcu and type_.name in {"List", "Slice"} and not (
            type_.name == "Slice" and role == "FFI parameter"
        ):
            self.error(
                "E_C99_TARGET_ALLOCATION",
                f"{type_.name} requires dynamic allocation and is unavailable for @target(mcu)",
                span,
            )
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
        if type_.name == "Slice":
            supported = (
                len(type_.arguments) == 1
                and self._plain_collection_element(type_.arguments[0])
                and not type_.dimensions
                and not _has_unit(type_)
            )
        elif type_.name == "SliceMut":
            supported = (
                role == "FFI parameter"
                and len(type_.arguments) == 1
                and type_.arguments[0].name == "u8"
                and not type_.arguments[0].arguments
                and not type_.dimensions
                and not _has_unit(type_)
            )
        if type_.name == "Result":
            supported = (
                len(type_.arguments) == 2
                and (
                    type_.arguments[0].name in _C_TYPES
                    or type_.arguments[0].name == "FrameBuffer"
                    or type_.arguments[0].name in {
                        "Array",
                        "List",
                        "Slice",
                    }
                    or type_.arguments[0].name in self.type_declarations
                )
                and not _has_unit(type_.arguments[0])
                and type_.arguments[1].name
                in {
                    "IoError",
                    "FsError",
                    "NetError",
                    "ArgsError",
                    "CollectionError",
                }
                and not role.startswith("FFI")
            )
        if type_.name == "Tensor":
            element = type_.element_type
            supported = (
                element is not None
                and element.name in {"u8", "i8", "i16", "i32", "f32", "f64"}
                and bool(type_.dimensions)
                and all(dimension > 0 for dimension in type_.dimensions)
            )
        if type_.name == "Array":
            supported = (
                len(type_.arguments) == 1
                and len(type_.dimensions) == 1
                and type_.dimensions[0] > 0
                and self._plain_collection_element(type_.arguments[0])
            )
        if type_.name in {"List", "Slice"}:
            supported = (
                len(type_.arguments) == 1
                and self._plain_collection_element(type_.arguments[0])
            )
        if type_.name in self.type_declarations:
            supported = not role.startswith("FFI") and not role.startswith("C export")
        elif type_.name != "Tensor" and _has_unit(type_):
            supported = type_.name in {"f32", "f64"}
        if role.startswith("FFI") and (
            type_.name in {"Tensor", "str"} or _has_unit(type_)
        ):
            supported = False
        if not supported:
            self.error(
                "E_C99_TYPE",
                f"C99 backend does not support {role} type {type_.display()}"
                + (
                    "; tensor backend supports u8/i8/i16/i32/f32/f64 elements"
                    if type_.name == "Tensor"
                    else ""
                ),
                span,
            )

    def _plain_collection_element(self, type_: AetherType) -> bool:
        if type_.name in _INTEGER_TYPES | {"bool", "f32", "f64"}:
            return not _has_unit(type_)
        declaration = self.type_declarations.get(type_.name)
        if declaration is None:
            return False
        return all(
            self._plain_collection_element(field_type)
            for _, field_type in declaration.fields
        )

    def validate_expression(
        self,
        expression: IRExpression,
        *,
        allow_signed_min_magnitude: bool = False,
    ) -> None:
        kind = expression.kind
        if expression.type.name not in {
            "IoError",
            "FsError",
            "NetError",
            "ArgsError",
            "CollectionError",
        }:
            self.validate_type(expression.type, expression.span, "expression")
        if kind == "fused_tensor":
            self.validate_expression(expression.children[0])
            return
        if kind == "match":
            self._validate_match(expression)
            return
        if kind == "literal":
            self._validate_literal(expression, allow_signed_min_magnitude)
            return
        if kind == "identifier":
            if expression.value in {
                "IoError",
                "FsError",
                "NetError",
                "ArgsError",
                "CollectionError",
            }:
                return
            return
        if kind == "struct_literal":
            declaration = self.type_declarations.get(expression.type.name)
            if declaration is None:
                self.error(
                    "E_C99_STRUCT",
                    f"Unknown struct type {expression.type.name!r}",
                    expression.span,
                )
            elif set(expression.field_names) != {
                name for name, _ in declaration.fields
            }:
                self.error(
                    "E_C99_STRUCT_FIELDS",
                    "Struct initializer fields do not match its declaration",
                    expression.span,
                )
            for child in expression.children:
                self.validate_expression(child)
            return
        if kind == "array_literal":
            if (
                expression.type.name not in {"Array", "List", "Slice"}
                or not expression.type.arguments
                or (
                    expression.type.name == "Array"
                    and (
                        len(expression.type.dimensions) != 1
                        or expression.type.dimensions[0]
                        != len(expression.children)
                    )
                )
            ):
                self.error(
                    "E_C99_COLLECTION",
                    "Collection literal has an invalid type or length",
                    expression.span,
                )
            if expression.type.name in {"List", "Slice"}:
                self.needs_trap = True
            element_type = (
                expression.type.arguments[0]
                if expression.type.arguments
                else AetherType("Unknown")
            )
            for child in expression.children:
                if child.type != element_type:
                    self.error(
                        "E_C99_COLLECTION",
                        "Collection literal elements must have one matching type",
                        child.span,
                    )
                self.validate_expression(child)
            return
        if kind == "member":
            base = expression.children[0]
            declaration = self.type_declarations.get(base.type.name)
            member = expression.children[1].value
            if declaration is None or member not in {
                name for name, _ in declaration.fields
            }:
                self.error(
                    "E_C99_MEMBER",
                    f"Unknown struct field {member!r}",
                    expression.span,
                )
            self.validate_expression(base)
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
            if expression.value not in {"!", "+", "-", "~"}:
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
            elif expression.value == "~":
                if operand.type.name not in _INTEGER_TYPES:
                    self.error(
                        "E_C99_OPERATOR",
                        "Unary ~ requires a supported integer type",
                        expression.span,
                    )
                else:
                    self.helpers.add((operand.type.name, "bnot"))
            elif operand.type.name == "Tensor" and expression.value == "-":
                element_type = operand.type.element_type
                if element_type is not None and element_type.name in _SIGNED_TYPES:
                    self.helpers.add((element_type.name, "neg"))
                    self.needs_trap = True
            elif expression.value == "-" and operand.type.name in _C_TYPES:
                if operand.type.name.startswith("u"):
                    self.error(
                        "E_C99_OPERATOR",
                        "Unary negation of unsigned integers is not supported",
                        expression.span,
                    )
                elif operand.type.name in _SIGNED_TYPES:
                    is_minimum_literal = False
                    if operand.kind == "literal":
                        try:
                            literal_value = int(
                                operand.value.replace("_", ""),
                                0
                                if operand.value.lower().startswith(("0x", "0b"))
                                else 10,
                            )
                        except ValueError:
                            literal_value = -1
                        is_minimum_literal = (
                            literal_value
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
            elif operator in {"&", "|", "^", "<<", ">>"}:
                operand_type = expression.children[0].type
                if (
                    expression.type.name not in _INTEGER_TYPES
                    or expression.children[1].type.name != operand_type.name
                ):
                    self.error(
                        "E_C99_OPERATOR",
                        f"Operator {operator!r} requires matching integer operands",
                        expression.span,
                    )
                else:
                    self.helpers.add(
                        (operand_type.name, self._operation_name(operator))
                    )
                    if operator in {"<<", ">>"}:
                        self.needs_trap = True
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
                tensor_type = next(
                    (
                        child.type
                        for child in expression.children
                        if child.type.name == "Tensor"
                    ),
                    expression.type,
                )
                element_type = tensor_type.element_type
                if element_type is not None and element_type.name in _INTEGER_TYPES:
                    self.helpers.add((element_type.name, "div"))
                self.needs_trap = True
            elif operator in {"+", "-", "*"} and expression.type.name != "Tensor":
                operand_type = expression.type.name
                if operand_type in _INTEGER_TYPES:
                    operation = {"+": "add", "-": "sub", "*": "mul"}[operator]
                    self.helpers.add((operand_type, operation))
                    self.needs_trap = True
            elif operator in {"+", "-", "*"} and expression.type.name == "Tensor":
                element_type = expression.type.element_type
                if element_type is not None and element_type.name in _INTEGER_TYPES:
                    operation = {"+": "add", "-": "sub", "*": "mul"}[operator]
                    self.helpers.add((element_type.name, operation))
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
                "i8",
                "i16",
                "i32",
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

    def _validate_match(self, expression: IRExpression) -> None:
        subject = expression.children[0] if expression.children else None
        if (
            subject is None
            or subject.type.name != "Result"
            or len(subject.type.arguments) != 2
            or len(expression.arms) != 2
            or {arm.variant for arm in expression.arms} != {"Ok", "Err"}
        ):
            self.error(
                "E_C99_MATCH",
                "C99 match supports exhaustive Ok/Err matches on Result values",
                expression.span,
            )
        if subject is not None:
            self.validate_expression(subject)
        for arm in expression.arms:
            self.validate_expression(arm.value)

    def _validate_gradient_target(self, function: IRFunction) -> None:
        if (
            not function.body
            or function.body[-1].kind != "return"
            or function.body[-1].expression is None
            or any(statement.kind != "let" for statement in function.body[:-1])
        ):
            self.error(
                "E_C99_GRADIENT_BODY",
                "C99 autodiff targets support local let bindings followed by one return",
                function.span,
            )
        for statement in function.body:
            if statement.expression is not None:
                self._validate_gradient_expression(statement.expression)

    def _validate_gradient_expression(self, expression: IRExpression) -> None:
        kind = expression.kind
        if kind in {"literal", "identifier"}:
            if kind == "literal" and expression.type.name not in {
                "f32",
                "f64",
                *_INTEGER_TYPES,
            }:
                self.error(
                    "E_C99_GRADIENT_EXPRESSION",
                    "C99 autodiff supports numeric literals and local values only",
                    expression.span,
                )
            return
        if kind == "group":
            self._validate_gradient_expression(expression.children[0])
            return
        if kind == "unary":
            if expression.value not in {"+", "-"}:
                self.error(
                    "E_C99_GRADIENT_EXPRESSION",
                    f"Unary operator {expression.value!r} is not supported by C99 autodiff",
                    expression.span,
                )
            for child in expression.children:
                self._validate_gradient_expression(child)
            return
        if kind == "binary":
            if expression.value == "|>":
                _, target = expression.children
                callee = (
                    target.children[0]
                    if target.kind == "call" and target.children
                    else target
                )
                if (
                    callee.kind != "identifier"
                    or callee.value not in {"convert", "mean", "sum", "scale", "subtract"}
                ):
                    self.error(
                        "E_C99_GRADIENT_EXPRESSION",
                        "C99 autodiff pipelines support only convert, sum, mean, scale, and subtract",
                        expression.span,
                    )
                self._validate_gradient_expression(expression.children[0])
                if target.kind == "call":
                    for argument in target.children[1:]:
                        if not (
                            callee.value == "convert"
                            and argument.type.name == "str"
                        ):
                            self._validate_gradient_expression(argument)
                return
            if expression.value not in {"+", "-", "*", "/"}:
                self.error(
                    "E_C99_GRADIENT_EXPRESSION",
                    f"Binary operator {expression.value!r} is not supported by C99 autodiff",
                    expression.span,
                )
            for child in expression.children:
                self._validate_gradient_expression(child)
            return
        if kind == "call":
            callee = expression.children[0] if expression.children else None
            if (
                callee is None
                or callee.kind != "identifier"
                or callee.value not in {"convert", "mean", "sum", "scale", "subtract"}
            ):
                self.error(
                    "E_C99_GRADIENT_EXPRESSION",
                    "C99 autodiff supports only convert, sum, mean, scale, and subtract calls",
                    expression.span,
                )
                return
            for index, argument in enumerate(expression.children[1:], start=1):
                if callee.value == "convert" and index == len(expression.children) - 1:
                    continue
                self._validate_gradient_expression(argument)
            return
        self.error(
            "E_C99_GRADIENT_EXPRESSION",
            f"Expression kind {kind!r} is not supported by C99 autodiff",
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
        if callee.value in {
            "List",
            "get",
            "set",
            "slice",
            "append",
            "pop",
            "insert",
            "remove",
        }:
            self._validate_collection_call(expression)
            for argument in expression.children[1:]:
                self.validate_expression(argument)
            return
        runtime_signatures = {
            "io_write": ((AetherType("str"),), "IoError", "io"),
            "io_read_line": ((), "IoError", "io"),
            "fs_read_text": ((AetherType("str"),), "FsError", "fs"),
            "fs_write_text": (
                (AetherType("str"), AetherType("str")),
                "FsError",
                "fs",
            ),
            "fs_exists": ((AetherType("str"),), "FsError", "fs"),
            "args_len": ((), None, None),
            "args_get": ((AetherType("i64"),), "ArgsError", None),
        }
        if callee.value in runtime_signatures:
            parameters, error_type, effect = runtime_signatures[callee.value]
            actual_arguments = expression.children[1:]
            if pipeline_input is not None or len(actual_arguments) != len(parameters):
                self.error(
                    "E_C99_RUNTIME_ARITY",
                    f"{callee.value} has an invalid runtime signature",
                    expression.span,
                )
            for argument, expected in zip(actual_arguments, parameters):
                if argument.type.name != expected.name:
                    self.error(
                        "E_C99_RUNTIME_TYPE",
                        f"{callee.value} expects {expected.name} arguments",
                        argument.span,
                    )
                self.validate_expression(argument)
            if error_type is not None:
                valid_result = (
                    expression.type.name == "Result"
                    and len(expression.type.arguments) == 2
                    and expression.type.arguments[0].name
                    == (
                        "void"
                        if callee.value in {"io_write", "fs_write_text"}
                        else "bool"
                        if callee.value == "fs_exists"
                        else "str"
                    )
                    and expression.type.arguments[1].name == error_type
                )
            else:
                valid_result = expression.type.name == "i64"
            if not valid_result:
                self.error(
                    "E_C99_RUNTIME_TYPE",
                    f"{callee.value} has an invalid result type",
                    expression.span,
                )
            if effect == "io":
                self.needs_host_io = True
            elif effect == "fs":
                self.needs_host_fs = True
            elif callee.value.startswith("args_"):
                self.needs_process_args = True
            self.runtime_calls.add(callee.value)
            self.needs_string = self.needs_string or any(
                argument.name == "str" for argument in parameters
            ) or (error_type is not None and error_type != "ArgsError")
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

    def _validate_collection_call(self, expression: IRExpression) -> None:
        name = expression.children[0].value
        arguments = expression.children[1:]
        if name == "List":
            valid = (
                expression.type.name == "List"
                and len(expression.type.arguments) == 1
                and len(arguments) <= 1
                and (
                    not arguments
                    or arguments[0].type.name
                    in {"Array", "List", "Slice"}
                )
            )
            if not valid:
                self.error(
                    "E_C99_COLLECTION",
                    "List constructor has an unsupported signature",
                    expression.span,
                )
            self.needs_trap = True
            return
        expected_arity = {
            "get": 2,
            "set": 3,
            "slice": 3,
            "append": 2,
            "pop": 1,
            "insert": 3,
            "remove": 2,
        }[name]
        if (
            len(arguments) != expected_arity
            or arguments[0].type.name not in {"Array", "List", "Slice"}
        ):
            self.error(
                "E_C99_COLLECTION",
                f"{name} has an unsupported collection signature",
                expression.span,
            )
            return
        collection_type = arguments[0].type
        element_type = collection_type.arguments[0]
        result_value = (
            AetherType("Slice", (element_type,))
            if name == "slice"
            else element_type
            if name in {"get", "pop", "remove"}
            else AetherType("void")
        )
        if (
            expression.type.name != "Result"
            or expression.type.arguments
            != (result_value, AetherType("CollectionError"))
        ):
            self.error(
                "E_C99_COLLECTION",
                f"{name} has an invalid Result type",
                expression.span,
            )
        if name in {"append", "pop", "insert", "remove"} and (
            collection_type.name != "List"
        ):
            self.error(
                "E_C99_COLLECTION",
                f"{name} can mutate only a List",
                expression.span,
            )
        if name in {"set", "append", "insert", "remove"}:
            target = arguments[0]
            if target.kind != "identifier":
                self.error(
                    "E_C99_COLLECTION",
                    "List mutation requires a local variable",
                    target.span,
                )
        if name in {"set", "append"}:
            position = 2 if name == "set" else 1
            if arguments[position].type != element_type:
                self.error(
                    "E_C99_COLLECTION",
                    f"{name} value has an incompatible element type",
                    arguments[position].span,
                )
        if name == "insert" and arguments[2].type != element_type:
            self.error(
                "E_C99_COLLECTION",
                "insert value has an incompatible element type",
                arguments[2].span,
            )
        if name == "slice":
            self.needs_trap = False

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
        if name in {"sum", "mean", "max_abs"}:
            allowed = len(arguments) == 1 and len(tensor_arguments) == 1
        elif name == "len":
            allowed = (
                len(arguments) == 1
                and arguments[0].type.name
                in {"Tensor", "Array", "List", "Slice"}
            )
        elif name == "abs":
            allowed = (
                len(arguments) == 1
                and (
                    arguments[0].type.name in _C_TYPES
                    or arguments[0].type.name == "Tensor"
                )
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
                "Collection len must produce i64",
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
        if (
            name in {"sum", "mean", "max_abs", "abs"}
            and arguments
            and arguments[0].type.name == "Tensor"
        ):
            element = arguments[0].type.element_type
            if element is not None and element.name in {"i8", "i16", "i32"}:
                operation = (
                    "add" if name == "sum"
                    else "abs" if name in {"max_abs", "abs"}
                    else ""
                )
                if operation:
                    self.helpers.add((element.name, operation))
                    self.needs_trap = True
        if (
            name == "abs"
            and arguments
            and arguments[0].type.name in _SIGNED_TYPES
        ):
            self.helpers.add((arguments[0].type.name, "abs"))
            self.needs_trap = True

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
            symbol = target.external_symbol or target.name
            if target.external_abi != "C":
                self.error(
                    "E_C99_FFI",
                    f"Foreign ABI {target.external_abi!r} is not supported",
                    span,
                )
            if (
                not symbol.isascii()
                or not symbol.isidentifier()
                or symbol.startswith("_")
                or symbol in _C_KEYWORDS
            ):
                self.error(
                    "E_C99_FFI_SYMBOL",
                    f"Extern function name {symbol!r} is not a portable C symbol",
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
            for parameter, argument_type in zip(target.parameters, argument_types):
                if parameter.type.name in {"Slice", "SliceMut"} and not (
                    argument_type.name == "Tensor"
                    and argument_type.element_type is not None
                    and argument_type.element_type.name == "u8"
                    and len(argument_type.dimensions) == 1
                    and not _has_unit(argument_type)
                ):
                    self.error(
                        "E_C99_FFI_SLICE",
                        "Slice<u8> FFI parameters require a one-dimensional Tensor<u8>",
                        span,
                    )
            self.validate_type(target.return_type, span, "FFI return")
        if len(argument_types) != len(target.parameters):
            return
        for actual, expected in zip(argument_types, target.parameters):
            if expected.type.name in {"Slice", "SliceMut"}:
                continue
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
            if value.startswith(("'", '"')) and type_name == "str":
                try:
                    self._decode_string_literal(value)
                except (ValueError, json.JSONDecodeError):
                    self.error(
                        "E_C99_LITERAL",
                        f"Invalid string literal {value!r}",
                        expression.span,
                    )
                return
            self.error(
                "E_C99_LITERAL",
                f"Literal {value!r} is not supported by the scalar C99 backend",
                expression.span,
            )
            return
        if type_name in _INTEGER_TYPES:
            try:
                numeric_value = int(
                    value.replace("_", ""),
                    0 if value.lower().startswith(("0x", "0b")) else 10,
                )
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

    def _validate_statement_block(
        self,
        statements: tuple[IRStatement, ...],
        mutable_locals: set[str],
        loop_depth: int = 0,
    ) -> None:
        for statement in statements:
            if statement.kind in {"if", "while"}:
                expression = statement.expression
                if expression is not None:
                    self.validate_expression(expression)
                    if expression.type.name != "bool":
                        self.error(
                            "E_C99_CONDITION",
                            "Control-flow condition must have type bool",
                            expression.span,
                        )
                self._validate_statement_block(
                    statement.body,
                    set(mutable_locals),
                    loop_depth + int(statement.kind == "while"),
                )
                if statement.kind == "if":
                    self._validate_statement_block(
                        statement.else_body,
                        set(mutable_locals),
                        loop_depth,
                    )
            elif statement.kind == "for":
                for expression in statement.expressions:
                    self.validate_expression(expression)
                    if expression.type.name not in _INTEGER_TYPES:
                        self.error(
                            "E_C99_RANGE_TYPE",
                            "for range bounds must have an integer type",
                            expression.span,
                        )
                if (
                    len(statement.expressions) == 2
                    and statement.expressions[0].type.name
                    != statement.expressions[1].type.name
                ):
                    self.error(
                        "E_C99_RANGE_TYPE",
                        "for range bounds must have the same integer type",
                        statement.span,
                    )
                self._validate_statement_block(
                    statement.body,
                    set(mutable_locals),
                    loop_depth + 1,
                )
            elif statement.kind in {"break", "continue"}:
                if loop_depth == 0:
                    self.error(
                        "E_C99_LOOP_CONTROL",
                        f"{statement.kind} is only valid inside a loop",
                        statement.span,
                    )
            else:
                expression = statement.expression
                if (
                    statement.kind == "assignment"
                    and statement.operator != "="
                    and expression is not None
                ):
                    expression = IRExpression(
                        "binary",
                        statement.span,
                        expression.type,
                        statement.operator[:-1],
                        (
                            IRExpression(
                                "identifier",
                                statement.span,
                                expression.type,
                                statement.name,
                            ),
                            expression,
                        ),
                    )
                if expression is not None:
                    self.validate_expression(expression)
                    if statement.kind == "let" and expression.type.name == "void":
                        self.error(
                            "E_C99_BINDING",
                            "A void function call cannot initialize a local binding",
                            statement.span,
                        )
                if statement.kind == "assignment":
                    if statement.name not in mutable_locals:
                        self.error(
                            "E_C99_OWNERSHIP",
                            f"Assignment target {statement.name!r} is not a mutable local",
                            statement.span,
                        )
                elif statement.kind == "let" and statement.binding_kind == "var":
                    mutable_locals.add(statement.name)

    @staticmethod
    def _block_returns(statements: tuple[IRStatement, ...]) -> bool:
        for statement in statements:
            if statement.kind == "return":
                return True
            if (
                statement.kind == "if"
                and statement.else_body
                and _C99Emitter._block_returns(statement.body)
                and _C99Emitter._block_returns(statement.else_body)
            ):
                return True
        return False

    def _ordered_struct_declarations(self) -> tuple[IRTypeDeclaration, ...]:
        ordered: list[IRTypeDeclaration] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def dependencies(type_: AetherType) -> set[str]:
            names = (
                {type_.name}
                if type_.name in self.type_declarations
                else set()
            )
            for argument in type_.arguments:
                names.update(dependencies(argument))
            return names

        def visit(name: str) -> None:
            if name in visited:
                return
            declaration = self.type_declarations[name]
            visiting.add(name)
            for _, field_type in declaration.fields:
                for dependency in sorted(dependencies(field_type)):
                    if dependency in visiting:
                        self.error(
                            "E_C99_RECURSIVE_TYPE",
                            "C99 cannot lay out recursive struct values",
                            declaration.span,
                        )
                    else:
                        visit(dependency)
            visiting.remove(name)
            visited.add(name)
            ordered.append(declaration)

        for name in sorted(self.type_declarations):
            visit(name)
        return tuple(ordered)

    def validate(self) -> None:
        if self.target_mcu and self.include_entry_point:
            span = (
                self.functions["main"].span
                if "main" in self.functions
                else self.module.functions[0].span
                if self.module.functions
                else SourceSpan(0, 0, 1, 1)
            )
            self.error(
                "E_C99_TARGET_ENTRY",
                "@target(mcu) emits a freestanding object and cannot generate a hosted main",
                span,
            )
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
            for field_name, field_type in declaration.fields:
                self.validate_type(
                    field_type,
                    declaration.span,
                    "struct field",
                )
                if field_type.name in {"void", "str", "Result", "FrameBuffer"}:
                    self.error(
                        "E_C99_STRUCT_FIELD",
                        f"Struct field {field_name!r} cannot use {field_type.display()} "
                        "because its value ownership is not defined",
                        declaration.span,
                    )
        self._ordered_struct_declarations()
        for function in self.module.functions:
            if function.c_exported:
                if (
                    function.external_abi is not None
                    or function.name.startswith("aether_")
                    or function.name == "main"
                ):
                    self.error(
                        "E_C99_EXPORT_SYMBOL",
                        f"Exported C function name {function.name!r} is reserved by the C99 backend",
                        function.span,
                    )
                for parameter in function.parameters:
                    self.validate_type(parameter.type, function.span, "C export parameter")
                if function.return_type.name == "void":
                    pass
                else:
                    self.validate_type(
                        function.return_type,
                        function.span,
                        "C export return",
                    )
                if any(
                    parameter.type.name in {"Tensor", "Result", "FrameBuffer", "str"}
                    or _has_unit(parameter.type)
                    for parameter in function.parameters
                ) or function.return_type.name in {
                    "Tensor", "Result", "FrameBuffer", "str"
                } or _has_unit(function.return_type):
                    self.error(
                        "E_C99_EXPORT_TYPE",
                        "@C exports require unitless C scalar parameters and return types",
                        function.span,
                    )
            if function.external_abi is not None:
                symbol = function.external_symbol or function.name
                if function.external_abi != "C":
                    self.error(
                        "E_C99_FFI",
                        f"Foreign ABI {function.external_abi!r} is not supported",
                        function.span,
                    )
                if (
                    not symbol.isascii()
                    or not symbol.isidentifier()
                    or symbol.startswith("_")
                    or symbol in _C_KEYWORDS
                ):
                    self.error(
                        "E_C99_FFI_SYMBOL",
                        f"Extern function name {symbol!r} is not a portable C symbol",
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
                set(function.declared_effects) - {"ffi", "io", "fs"}
                or set(function.required_effects) - {"ffi", "io", "fs"}
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
            self._validate_statement_block(function.body, mutable_locals)
            if (
                function.return_type.name not in {"void", "Unknown"}
                and not self._block_returns(function.body)
            ):
                self.error(
                    "E_C99_RETURN",
                    f"Function {function.name!r} must end with a value return",
                    function.span,
                )
            self.current_function = None
        for target_name in sorted(self.gradient_targets):
            target = self.functions.get(target_name)
            if target is not None and target.external_abi is None:
                self._validate_gradient_target(target)
        if self.target_mcu:
            if self.runtime_calls:
                function = next(
                    (
                        item
                        for item in self.module.functions
                        if item.name in self.defined_function_names
                    ),
                    self.module.functions[0]
                    if self.module.functions
                    else None,
                )
                if function is not None:
                    self.error(
                        "E_C99_TARGET_RUNTIME",
                        "Host I/O, filesystem, and process-argument runtime calls are unavailable for @target(mcu)",
                        function.span,
                    )
            if self.gradient_targets:
                target = self.functions.get(sorted(self.gradient_targets)[0])
                if target is not None:
                    self.error(
                        "E_C99_TARGET_ALLOCATION",
                        "Autodiff requires dynamic allocation and is unavailable for @target(mcu)",
                        target.span,
                    )

    def _new_temp(self, type_: AetherType) -> str:
        name = f"aether_tmp_{self.temp_index}"
        self.temp_index += 1
        return f"{self.c_type(type_)} {name}"

    def _materialize(
        self,
        value: _CValue,
    ) -> tuple[list[str], str, list[str]]:
        if (
            self._contains_frame_buffer(value.type)
            or self._contains_owned_collection(value.type)
        ) and value.ownership == "borrowed":
            return [*value.prelude], value.code, list(value.cleanup)
        declaration = self._new_temp(value.type)
        name = declaration.rsplit(" ", 1)[1]
        cleanup = list(value.cleanup)
        if (
            self._contains_frame_buffer(value.type)
            or self._contains_owned_collection(value.type)
        ) and value.ownership == "owned":
            release = self._release_code(name, value.type)
            if release:
                cleanup.append(release)
        return [*value.prelude, f"{declaration} = {value.code};"], name, cleanup

    def expression(self, expression: IRExpression) -> _CValue:
        kind = expression.kind
        if kind == "fused_tensor":
            return self._fused_tensor_expression(expression)
        if kind == "literal":
            return _CValue((), self._literal_code(expression), expression.type)
        if kind == "identifier":
            if expression.value in {
                "IoError",
                "FsError",
                "NetError",
                "ArgsError",
            }:
                error_codes = {
                    "IoError": 1,
                    "FsError": 2,
                    "NetError": 3,
                    "ArgsError": 1,
                }
                return _CValue((), str(error_codes[expression.value]), expression.type)
            return _CValue(
                (),
                _encoded_identifier("aether_v_", expression.value),
                expression.type,
                (
                    "borrowed"
                    if self._contains_owned_collection(expression.type)
                    else expression.ownership
                ),
            )
        if kind == "struct_literal":
            prelude: list[str] = []
            cleanup: list[str] = []
            field_values: dict[str, str] = {}
            outer_cleanups = self.active_expression_cleanups
            try:
                for field_name, child in zip(
                    expression.field_names,
                    expression.children,
                ):
                    self.active_expression_cleanups = (
                        *outer_cleanups,
                        *cleanup,
                    )
                    value = self.expression(child)
                    lines, name, value_cleanup = self._materialize(value)
                    prelude.extend(lines)
                    field_values[field_name] = name
                    cleanup.extend(value_cleanup)
            finally:
                self.active_expression_cleanups = outer_cleanups
            initializer = ", ".join(
                f".{_encoded_identifier('aether_field_', name)} = {field_values[name]}"
                for name in expression.field_names
            )
            return _CValue(
                tuple(prelude),
                f"({self.c_type(expression.type)}){{{initializer}}}",
                expression.type,
                cleanup=tuple(cleanup),
            )
        if kind == "array_literal":
            collection_type = expression.type
            element_type = collection_type.arguments[0]
            prelude: list[str] = []
            values: list[str] = []
            cleanup: list[str] = []
            outer_cleanups = self.active_expression_cleanups
            try:
                for child in expression.children:
                    self.active_expression_cleanups = (
                        *outer_cleanups,
                        *cleanup,
                    )
                    value = self.expression(child)
                    lines, name, value_cleanup = self._materialize(value)
                    prelude.extend(lines)
                    values.append(name)
                    cleanup.extend(value_cleanup)
            finally:
                self.active_expression_cleanups = outer_cleanups
            if collection_type.name == "Array":
                declaration = self._new_temp(collection_type)
                result = declaration.rsplit(" ", 1)[1]
                prelude.append(f"{declaration} = {{0}};")
                prelude.extend(
                    f"{result}.data[{index}] = {value};"
                    for index, value in enumerate(values)
                )
                return _CValue(
                    tuple(prelude),
                    result,
                    collection_type,
                    cleanup=tuple(cleanup),
                )
            declaration = self._new_temp(collection_type)
            result = declaration.rsplit(" ", 1)[1]
            storage = self._collection_c_name(collection_type) + "_storage"
            element = self.c_type(element_type)
            count = len(values)
            prelude.append(f"{declaration} = {{0}};")
            prelude.extend(
                [
                    f"if ((size_t){count} > "
                    f"(SIZE_MAX - sizeof({storage})) / sizeof({element})) "
                    "{ aether_trap(); }",
                    f"{result}.storage = ({storage} *)malloc("
                    f"sizeof({storage}) + (size_t){count} * sizeof({element}));",
                    f"if ({result}.storage == NULL) aether_trap();",
                    f"{result}.storage->references = 1;",
                    f"{result}.storage->capacity = (size_t){count};",
                    f"{result}.length = (size_t){count};",
                ]
            )
            prelude.extend(
                f"{result}.storage->values[{index}] = {value};"
                for index, value in enumerate(values)
            )
            return _CValue(
                tuple(prelude),
                result,
                collection_type,
                "owned",
                tuple(cleanup),
            )
        if kind == "member":
            base = self.expression(expression.children[0])
            field_name = expression.children[1].value
            field = _encoded_identifier("aether_field_", field_name)
            return _CValue(
                base.prelude,
                f"({base.code}).{field}",
                expression.type,
                cleanup=base.cleanup,
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
        if kind == "match":
            return self._match_expression(expression)
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
            if self._contains_owned_collection(expression.type):
                retain = self._retain_code(f"{name}.value", expression.type)
                if retain:
                    prelude.append(retain)
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
                element_type = expression.type.element_type
                item = f"{name}.data[{index}]"
                value = (
                    f"aether_{element_type.name}_neg({item})"
                    if expression.value == "-"
                    and element_type is not None
                    and element_type.name in _SIGNED_TYPES
                    else f"-{item}"
                )
                prelude.append(
                    f"    {result}.data[{index}] = {value};"
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
            elif expression.value == "~":
                code = f"aether_{expression.type.name}_bnot({operand.code})"
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
            elif operator in {"&", "|", "^", "<<", ">>"}:
                code = (
                    f"aether_{type_name}_{self._operation_name(operator)}"
                    f"({left_name}, {right_name})"
                )
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

    def _match_expression(self, expression: IRExpression) -> _CValue:
        subject_expression = expression.children[0]
        subject = self.expression(subject_expression)
        prelude, subject_name, subject_cleanup = self._materialize(subject)
        declaration = self._new_temp(expression.type)
        result_name = declaration.rsplit(" ", 1)[1]
        prelude.append(f"{declaration};")
        all_borrowed = all(
            arm.value.ownership == "borrowed"
            or (
                self._contains_owned_collection(arm.value.type)
                and arm.value.kind == "identifier"
            )
            for arm in expression.arms
        )
        for index, arm in enumerate(expression.arms):
            condition = (
                f"{subject_name}.is_ok"
                if arm.variant == "Ok"
                else f"!{subject_name}.is_ok"
            )
            prelude.append(
                f"if ({condition}) {{"
                if index == 0
                else "else {"
            )
            if arm.binding is not None and arm.binding != "_":
                binding_name = _encoded_identifier("aether_v_", arm.binding)
                arm_payload_type = subject_expression.type.arguments[
                    0 if arm.variant == "Ok" else 1
                ]
                prelude.append(
                    f"    {self.c_type(arm_payload_type)} {binding_name} = "
                    f"{subject_name}.value;"
                )
            outer_cleanups = self.active_expression_cleanups
            self.active_expression_cleanups = (*outer_cleanups, *subject_cleanup)
            try:
                value = self.expression(arm.value)
            finally:
                self.active_expression_cleanups = outer_cleanups
            prelude.extend("    " + line for line in value.prelude)
            prelude.append(f"    {result_name} = {value.code};")
            if (
                (
                    self._contains_frame_buffer(expression.type)
                    or self._contains_owned_collection(expression.type)
                )
                and value.ownership == "borrowed"
                and not all_borrowed
            ):
                retain = self._retain_code(result_name, expression.type)
                if retain:
                    prelude.append(f"    {retain}")
            prelude.extend("    " + line for line in value.cleanup)
            prelude.append("}")
        owns_result = (
            (
                self._contains_frame_buffer(expression.type)
                or self._contains_owned_collection(expression.type)
            )
            and not all_borrowed
        )
        return _CValue(
            tuple(prelude),
            result_name,
            expression.type,
            (
                "owned"
                if owns_result
                else "borrowed"
                if (
                    self._contains_frame_buffer(expression.type)
                    or self._contains_owned_collection(expression.type)
                )
                else "plain"
            ),
            tuple(subject_cleanup),
        )

    def _collection_expression(self, expression: IRExpression) -> _CValue:
        name = expression.children[0].value
        arguments = expression.children[1:]
        result_type = expression.type
        if name == "List" and not arguments:
            declaration = self._new_temp(result_type)
            result = declaration.rsplit(" ", 1)[1]
            storage = self._collection_c_name(result_type) + "_storage"
            return _CValue(
                (
                    f"{declaration} = {{0}};",
                    f"{result}.storage = ({storage} *)malloc(sizeof({storage}));",
                    f"if ({result}.storage == NULL) aether_trap();",
                    f"{result}.storage->references = 1;",
                    f"{result}.storage->capacity = 0;",
                ),
                result,
                result_type,
                "owned",
            )

        prelude: list[str] = []
        names: list[str] = []
        argument_cleanups: list[str] = []
        outer_cleanups = self.active_expression_cleanups
        try:
            for index, argument_expression in enumerate(arguments):
                self.active_expression_cleanups = (
                    *outer_cleanups,
                    *argument_cleanups,
                )
                argument = self.expression(argument_expression)
                is_mutation_target = (
                    index == 0
                    and name in {"set", "append", "pop", "insert", "remove"}
                    and argument_expression.kind == "identifier"
                )
                if is_mutation_target:
                    value_name = argument.code
                    prelude.extend(argument.prelude)
                else:
                    lines, value_name, cleanup = self._materialize(argument)
                    prelude.extend(lines)
                    argument_cleanups.extend(cleanup)
                names.append(value_name)
        finally:
            self.active_expression_cleanups = outer_cleanups

        if name == "List":
            source_type = arguments[0].type
            source = names[0]
            if source_type.name == "List":
                prelude.append(
                    f"if ({source}.storage != NULL) {{ "
                    f"if ({source}.storage->references == SIZE_MAX) "
                    "{ aether_trap(); } "
                    f"++{source}.storage->references; }}"
                )
                return _CValue(
                    tuple(prelude),
                    source,
                    result_type,
                    "owned",
                    tuple(argument_cleanups),
                )
            target = self._new_temp(result_type)
            result = target.rsplit(" ", 1)[1]
            storage = self._collection_c_name(result_type) + "_storage"
            source_values = (
                f"{source}.data"
                if source_type.name == "Array"
                else f"({source}.storage->values + {source}.start)"
                if source_type.name == "Slice"
                else "NULL"
            )
            count = (
                str(source_type.dimensions[0])
                if source_type.name == "Array"
                else f"{source}.length"
            )
            element = self.c_type(result_type.arguments[0])
            prelude.extend(
                [
                    f"{target} = {{0}};",
                    f"if ((size_t)({count}) > "
                    f"(SIZE_MAX - sizeof({storage})) / sizeof({element})) "
                    "{ aether_trap(); }",
                    f"{result}.storage = ({storage} *)malloc("
                    f"sizeof({storage}) + (size_t)({count}) * sizeof({element}));",
                    f"if ({result}.storage == NULL) aether_trap();",
                    f"{result}.storage->references = 1;",
                    f"{result}.storage->capacity = (size_t)({count});",
                    f"{result}.length = (size_t)({count});",
                    f"for (size_t aether_copy_i_{self.temp_index} = 0; "
                    f"aether_copy_i_{self.temp_index} < {result}.length; "
                    f"++aether_copy_i_{self.temp_index}) {{",
                    f"    {result}.storage->values[aether_copy_i_{self.temp_index}] = "
                    f"{source_values}[aether_copy_i_{self.temp_index}];",
                    "}",
                ]
            )
            self.temp_index += 1
            return _CValue(
                tuple(prelude),
                result,
                result_type,
                "owned",
                tuple(argument_cleanups),
            )

        collection_type = arguments[0].type
        collection_name = names[0]
        element_type = collection_type.arguments[0]
        element_c = self.c_type(element_type)

        def value_at(index: str) -> str:
            if collection_type.name == "Array":
                return f"{collection_name}.data[{index}]"
            if collection_type.name in {"List", "Slice"}:
                return f"{collection_name}.storage->values[{index}]"
            return (
                f"{collection_name}.storage->values["
                f"{collection_name}.start + ({index})]"
            )

        def bounds_test(index: str, allow_end: bool = False) -> str:
            index_type = arguments[1].type
            parts = []
            if index_type.name in _SIGNED_TYPES:
                parts.append(f"({index}) < 0")
            comparison = ">" if allow_end else ">="
            collection_length = (
                str(collection_type.dimensions[0])
                if collection_type.name == "Array"
                else f"{collection_name}.length"
            )
            parts.append(
                f"(uint64_t)({index}) {comparison} "
                f"(uint64_t)({collection_length})"
            )
            return " || ".join(parts)

        result_name = self._new_temp(result_type).rsplit(" ", 1)[1]
        prelude.append(f"{self.c_type(result_type)} {result_name} = {{0}};")
        if name == "get":
            index = names[1]
            prelude.extend(
                [
                    f"if ({bounds_test(index)}) {{",
                    f"    {result_name}.error = 1;",
                    "} else {",
                    f"    {result_name}.is_ok = 1;",
                    f"    {result_name}.value = {value_at(index)};",
                    "}",
                ]
            )
        elif name == "set":
            index, value = names[1:]
            prelude.extend(
                [
                    f"if ({bounds_test(index)}) {{",
                    f"    {result_name}.error = 1;",
                    "} else {",
                ]
            )
            if collection_type.name == "List":
                self._list_make_unique(
                    collection_type,
                    collection_name,
                    prelude,
                    "    ",
                )
            prelude.extend(
                [
                    f"    if ({result_name}.error == 0) {{",
                    f"        {value_at(index)} = {value};",
                    f"        {result_name}.is_ok = 1;",
                    "    }",
                    "}",
                ]
            )
        elif name == "slice":
            start, end = names[1:]
            start_negative = (
                f"({start}) < 0 || "
                if arguments[1].type.name in _SIGNED_TYPES
                else ""
            )
            end_negative = (
                f"({end}) < 0 || "
                if arguments[2].type.name in _SIGNED_TYPES
                else ""
            )
            slice_type = result_type.arguments[0]
            slice_storage = self._collection_c_name(slice_type) + "_storage"
            slice_result = f"{result_name}.value"
            count = f"(uint64_t)({end}) - (uint64_t)({start})"
            collection_length = (
                str(collection_type.dimensions[0])
                if collection_type.name == "Array"
                else f"{collection_name}.length"
            )
            prelude.extend(
                [
                    f"if ({start_negative}{end_negative}"
                    f"(uint64_t)({start}) > (uint64_t)({end}) || "
                    f"(uint64_t)({end}) > (uint64_t)({collection_length})) {{",
                    f"    {result_name}.error = 1;",
                    "} else {",
                    f"    if ({count} > (uint64_t)((SIZE_MAX - "
                    f"sizeof({slice_storage})) / sizeof({element_c}))) {{",
                    f"        {result_name}.error = 4;",
                    "    } else {",
                    f"        {slice_result}.storage = ({slice_storage} *)malloc("
                    f"sizeof({slice_storage}) + (size_t)({count}) * sizeof({element_c}));",
                    f"        if ({slice_result}.storage == NULL) {{",
                    f"            {result_name}.error = 3;",
                    "        } else {",
                    f"            {slice_result}.storage->references = 1;",
                    f"            {slice_result}.storage->capacity = (size_t)({count});",
                    f"            {slice_result}.start = 0;",
                    f"            {slice_result}.length = (size_t)({count});",
                    f"            for (size_t aether_slice_i_{self.temp_index} = 0; "
                    f"aether_slice_i_{self.temp_index} < {slice_result}.length; "
                    f"++aether_slice_i_{self.temp_index}) {{",
                    f"                {slice_result}.storage->values[aether_slice_i_{self.temp_index}] = "
                    f"{value_at(f'(uint64_t)({start}) + aether_slice_i_{self.temp_index}')};",
                    "            }",
                    f"            {result_name}.is_ok = 1;",
                    "        }",
                    "    }",
                    "}",
                ]
            )
            self.temp_index += 1
        else:
            list_type = collection_type
            storage = self._collection_c_name(list_type) + "_storage"
            is_pop = name == "pop"
            index_name = names[1] if name in {"insert", "remove"} else None
            value_name = names[-1] if name in {"append", "insert"} else None
            length = f"{collection_name}.length"
            prelude.append(
                f"if ({collection_name}.storage == NULL && {length} != 0) "
                "{ aether_trap(); }"
            )
            if is_pop:
                prelude.append(f"if ({length} == 0) {{")
                prelude.append(f"    {result_name}.error = 2;")
                prelude.append("} else {")
                self._list_make_unique(
                    list_type, collection_name, prelude, "    ", result_name
                )
                prelude.extend(
                    [
                        f"    if ({result_name}.error == 0) {{",
                        f"        {result_name}.value = "
                        f"{collection_name}.storage->values[{length} - 1];",
                        f"        --{length};",
                        f"        {result_name}.is_ok = 1;",
                        "    }",
                        "}",
                    ]
                )
            else:
                assert index_name is not None or name == "append"
                index_error = (
                    f"{bounds_test(index_name, allow_end=name == 'insert')}"
                    if index_name is not None
                    else "0"
                )
                prelude.extend(
                    [
                        f"if ({index_error}) {{",
                        f"    {result_name}.error = 1;",
                        "} else {",
                    ]
                )
                self._list_make_unique(
                    list_type, collection_name, prelude, "    ", result_name
                )
                capacity = f"{collection_name}.storage->capacity"
                needs_growth = (
                    f"{length} == {capacity}"
                    if name in {"append", "insert"}
                    else "0"
                )
                prelude.extend(
                    [
                        f"    if ({result_name}.error == 0 && {needs_growth}) {{",
                        f"        size_t aether_new_capacity_{self.temp_index} = "
                        f"{capacity} == 0 ? 4 : {capacity} * 2;",
                        f"        if (aether_new_capacity_{self.temp_index} < {capacity} || "
                        f"aether_new_capacity_{self.temp_index} > "
                        f"(SIZE_MAX - sizeof({storage})) / sizeof({element_c})) {{",
                        f"            {result_name}.error = 4;",
                        "        } else {",
                        f"            {storage} *aether_new_storage_{self.temp_index} = "
                        f"({storage} *)realloc({collection_name}.storage, "
                        f"sizeof({storage}) + aether_new_capacity_{self.temp_index} * "
                        f"sizeof({element_c}));",
                        f"            if (aether_new_storage_{self.temp_index} == NULL) {{",
                        f"                {result_name}.error = 3;",
                        "            } else {",
                        f"                {collection_name}.storage = "
                        f"aether_new_storage_{self.temp_index};",
                        f"                {collection_name}.storage->capacity = "
                        f"aether_new_capacity_{self.temp_index};",
                        "            }",
                        "        }",
                        "    }",
                        f"    if ({result_name}.error == 0) {{",
                    ]
                )
                if name in {"insert", "remove"}:
                    if name == "remove":
                        prelude.append(
                            f"        {result_name}.value = "
                            f"{collection_name}.storage->values[{index_name}];"
                        )
                    prelude.extend(
                        [
                            f"        for (size_t aether_move_i_{self.temp_index} = "
                            f"{length}; aether_move_i_{self.temp_index} > "
                            f"(size_t)({index_name}); --aether_move_i_{self.temp_index}) {{",
                            f"            {collection_name}.storage->values["
                            f"aether_move_i_{self.temp_index}] = "
                            f"{collection_name}.storage->values["
                            f"aether_move_i_{self.temp_index} - 1];",
                            "        }",
                        ]
                        if name == "insert"
                        else [
                            f"        for (size_t aether_move_i_{self.temp_index} = "
                            f"(size_t)({index_name}); aether_move_i_{self.temp_index} + 1 < "
                            f"{length}; ++aether_move_i_{self.temp_index}) {{",
                            f"            {collection_name}.storage->values["
                            f"aether_move_i_{self.temp_index}] = "
                            f"{collection_name}.storage->values["
                            f"aether_move_i_{self.temp_index} + 1];",
                            "        }",
                        ]
                    )
                if name in {"append", "insert"}:
                    destination = (
                        f"(size_t)({index_name})"
                        if name == "insert"
                        else length
                    )
                    prelude.append(
                        f"        {collection_name}.storage->values[{destination}] = "
                        f"{value_name};"
                    )
                if name in {"append", "insert"}:
                    prelude.append(f"        ++{length};")
                elif name == "remove":
                    prelude.append(f"        --{length};")
                prelude.extend(
                    [
                        f"        {result_name}.is_ok = 1;",
                        "    }",
                        "}",
                    ]
                )
                self.temp_index += 1
        if (
            name in {"set", "append", "pop", "insert", "remove"}
            and collection_type.name in {"List", "Slice"}
        ):
            slot = (
                self._lookup_collection_slot(arguments[0].value)
                if arguments[0].kind == "identifier"
                else None
            )
            if slot is not None:
                prelude.append(
                    f"{self.collection_slots_name}[{slot}].storage = "
                    f"{collection_name}.storage;"
                )
        return _CValue(
            tuple(prelude),
            result_name,
            result_type,
            "owned" if name == "slice" else "plain",
            tuple(argument_cleanups),
        )

    def _list_make_unique(
        self,
        type_: AetherType,
        name: str,
        lines: list[str] | None = None,
        indentation: str = "",
        error_result: str | None = None,
    ) -> list[str]:
        target = lines if lines is not None else []
        storage = self._collection_c_name(type_) + "_storage"
        element = self.c_type(type_.arguments[0])
        temp = self.temp_index
        self.temp_index += 1
        target.extend(
            [
                f"{indentation}if ({name}.storage != NULL && "
                f"{name}.storage->references > 1) {{",
                f"{indentation}    if ({name}.storage->capacity > "
                f"(SIZE_MAX - sizeof({storage})) / sizeof({element})) {{",
                (
                    f"{indentation}        {error_result}.error = 4;"
                    if error_result
                    else f"{indentation}        aether_trap();"
                ),
                f"{indentation}    }} else {{",
                f"{indentation}        {storage} *aether_unique_{temp} = "
                f"({storage} *)malloc(sizeof({storage}) + "
                f"{name}.storage->capacity * sizeof({element}));",
                f"{indentation}        if (aether_unique_{temp} == NULL) {{",
                (
                    f"{indentation}            {error_result}.error = 3;"
                    if error_result
                    else f"{indentation}            aether_trap();"
                ),
                f"{indentation}        }} else {{",
                f"{indentation}            aether_unique_{temp}->references = 1;",
                f"{indentation}            aether_unique_{temp}->capacity = "
                f"{name}.storage->capacity;",
                f"{indentation}            for (size_t aether_unique_i_{temp} = 0; "
                f"aether_unique_i_{temp} < {name}.length; "
                f"++aether_unique_i_{temp}) {{",
                f"{indentation}                aether_unique_{temp}->values[aether_unique_i_{temp}] = "
                f"{name}.storage->values[aether_unique_i_{temp}];",
                f"{indentation}            }}",
                f"{indentation}            --{name}.storage->references;",
                f"{indentation}            {name}.storage = aether_unique_{temp};",
                f"{indentation}        }}",
                f"{indentation}    }}",
                f"{indentation}}}",
            ]
        )
        return target

    def _call_expression(self, expression: IRExpression) -> _CValue:
        callee = expression.children[0]
        if callee.kind == "identifier" and callee.value in {"Ok", "Err"}:
            argument = self.expression(expression.children[1])
            if callee.value == "Ok":
                prelude = list(argument.prelude)
                if (
                    (
                        self._contains_frame_buffer(argument.type)
                        or self._contains_owned_collection(argument.type)
                    )
                    and argument.ownership == "borrowed"
                ):
                    retain = self._retain_code(argument.code, argument.type)
                    if retain:
                        prelude.append(retain)
                code = (
                    f"({self.c_type(expression.type)})"
                    f"{{1, {argument.code}, 0}}"
                )
            else:
                prelude = list(argument.prelude)
                value_type = expression.type.arguments[0]
                zero_value = (
                    "{0}"
                    if value_type.name in {"str", "FrameBuffer"}
                    else "0"
                )
                code = (
                    f"({self.c_type(expression.type)})"
                    f"{{0, {zero_value}, {argument.code}}}"
                )
            return _CValue(
                tuple(prelude),
                code,
                expression.type,
                (
                    "owned"
                    if self._contains_frame_buffer(expression.type)
                    or self._contains_owned_collection(expression.type)
                    else "plain"
                ),
                argument.cleanup,
                self._contains_frame_buffer(expression.type)
                or self._contains_owned_collection(expression.type),
            )
        if callee.kind == "identifier" and callee.value in {
            "frame_buffer",
            "frame_buffer_len",
        }:
            return self._builtin_expression(expression)
        if callee.kind == "identifier" and callee.value in {
            "List",
            "get",
            "set",
            "slice",
            "append",
            "pop",
            "insert",
            "remove",
        }:
            return self._collection_expression(expression)
        if callee.kind == "identifier" and callee.value in {
            "io_write",
            "io_read_line",
            "fs_read_text",
            "fs_write_text",
            "fs_exists",
            "args_len",
            "args_get",
        }:
            return self._runtime_expression(expression)
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
            for index, argument_expression in enumerate(expression.children[1:]):
                self.active_expression_cleanups = (
                    *outer_cleanups,
                    *cleanup,
                )
                argument = self.expression(argument_expression)
                parameter = target.parameters[index]
                if (
                    target.external_abi is not None
                    and parameter.type.name in {"Slice", "SliceMut"}
                ):
                    if (
                        parameter.type.name == "SliceMut"
                        and argument_expression.kind == "identifier"
                    ):
                        prelude.extend(argument.prelude)
                        name = argument.code
                        argument_cleanup = argument.cleanup
                    else:
                        argument_lines, name, argument_cleanup = self._materialize(argument)
                        prelude.extend(argument_lines)
                    names.extend(
                        (
                            f"{name}.data",
                            f"{argument_expression.type.dimensions[0]}u",
                        )
                    )
                else:
                    if (
                        self._contains_owned_collection(parameter.type)
                        and argument.ownership == "owned"
                    ):
                        declaration = self._new_temp(argument.type)
                        name = declaration.rsplit(" ", 1)[1]
                        argument_cleanup = list(argument.cleanup)
                        prelude.extend(
                            [
                                *argument.prelude,
                                f"{declaration} = {argument.code};",
                            ]
                        )
                    else:
                        argument_lines, name, argument_cleanup = self._materialize(argument)
                        prelude.extend(argument_lines)
                    names.append(name)
                    if (
                        self._contains_owned_collection(parameter.type)
                        and argument.ownership == "borrowed"
                    ):
                        retain = self._retain_code(name, parameter.type)
                        if retain:
                            prelude.append(retain)
                cleanup.extend(argument_cleanup)
        finally:
            self.active_expression_cleanups = outer_cleanups
        return _CValue(
            tuple(prelude),
            f"{self._call_target(callee.value)}({', '.join(names)})",
            target.return_type,
            (
                "owned"
                if self._contains_frame_buffer(target.return_type)
                or self._contains_owned_collection(target.return_type)
                else "plain"
            ),
            tuple(cleanup),
            self._contains_frame_buffer(target.return_type)
            or self._contains_owned_collection(target.return_type),
        )

    def _runtime_expression(self, expression: IRExpression) -> _CValue:
        callee = expression.children[0]
        arguments = expression.children[1:]
        prelude: list[str] = []
        names: list[str] = []
        cleanup: list[str] = []
        outer_cleanups = self.active_expression_cleanups
        try:
            for argument_expression in arguments:
                self.active_expression_cleanups = (
                    *outer_cleanups,
                    *cleanup,
                )
                argument = self.expression(argument_expression)
                lines, name, argument_cleanup = self._materialize(argument)
                prelude.extend(lines)
                names.append(name)
                cleanup.extend(argument_cleanup)
        finally:
            self.active_expression_cleanups = outer_cleanups
        target = {
            "io_write": "aether_io_write",
            "io_read_line": "aether_io_read_line",
            "fs_read_text": "aether_fs_read_text",
            "fs_write_text": "aether_fs_write_text",
            "fs_exists": "aether_fs_exists",
            "args_len": "aether_args_len",
            "args_get": "aether_args_get",
        }[callee.value]
        return _CValue(
            tuple(prelude),
            f"{target}({', '.join(names)})",
            expression.type,
            cleanup=tuple(cleanup),
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

    def _fused_element_code(self, expression: IRExpression, index: str) -> str:
        if expression.kind == "literal":
            return self._literal_code(expression)
        if expression.kind == "identifier":
            name = _encoded_identifier("aether_v_", expression.value)
            if expression.type.name == "Tensor":
                return f"{name}.data[{index}]"
            return name
        if expression.kind == "group":
            return f"({self._fused_element_code(expression.children[0], index)})"
        if expression.kind == "unary":
            operand = self._fused_element_code(expression.children[0], index)
            code = f"({expression.value}{operand})"
        elif expression.kind == "binary":
            left = self._fused_element_code(expression.children[0], index)
            right = self._fused_element_code(expression.children[1], index)
            code = f"({left} {expression.value} {right})"
        else:
            raise ValueError(
                f"Unsupported fused tensor expression {expression.kind!r}"
            )
        value_type = (
            expression.type.element_type
            if expression.type.name == "Tensor"
            else expression.type
        )
        if value_type is None or value_type.name not in {"f32", "f64"}:
            raise ValueError("Fused tensor expressions require f32 or f64 values")
        return f"(({_C_TYPES[value_type.name]}){code})"

    def _fused_tensor_expression(self, expression: IRExpression) -> _CValue:
        root = expression.children[0]
        result_declaration = self._new_temp(expression.type)
        result_name = result_declaration.rsplit(" ", 1)[1]
        length = self._tensor_size(expression.type)
        index_name = f"aether_i_{self.temp_index}"
        self.temp_index += 1
        value_code = self._fused_element_code(root, index_name)
        prelude = [
            f"{result_declaration} = {{0}};",
            f"for (size_t {index_name} = 0; {index_name} < {length}; "
            f"++{index_name}) {{",
            f"    {result_name}.data[{index_name}] = {value_code};",
            "}",
        ]
        return _CValue(tuple(prelude), result_name, expression.type)

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
                f"    {result_name}.data[{index_name}] = "
                + (
                    f"aether_{result_type.element_type.name}_div({left_code}, {right_code});"
                    if result_type.element_type is not None
                    and result_type.element_type.name in _INTEGER_TYPES
                    else f"{left_code} / {right_code};"
                )
            )
            prelude.append("}")
        else:
            prelude.append(f"{result_declaration} = {{0}};")
            prelude.append(
                f"for (size_t {index_name} = 0; {index_name} < {length}; "
                f"++{index_name}) {{"
            )
            operation_type = result_type.element_type
            if (
                operation_type is not None
                and operation_type.name in _INTEGER_TYPES
                and operator in {"+", "-", "*"}
            ):
                operation_name = self._operation_name(operator)
                item_expression = (
                    f"aether_{operation_type.name}_{operation_name}"
                    f"({left_code}, {right_code})"
                )
            else:
                item_expression = f"{left_code} {operator} {right_code}"
            prelude.append(
                f"    {result_name}.data[{index_name}] = {item_expression};"
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
        if expression.kind == "binary" and expression.value == "|>":
            left, right = expression.children
            if right.kind == "call" and right.children:
                callee = right.children[0]
                arguments = (left, *right.children[1:])
            else:
                callee = right
                arguments = (left,)
            call = IRExpression(
                "call",
                expression.span,
                expression.type,
                children=(callee, *arguments),
                conversion=right.conversion or expression.conversion,
            )
            return self._gradient_tape_expression(call, names, lines, next_node)
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
            if name in {"scale", "subtract"}:
                if len(expression.children) != 3:
                    raise ValueError(
                        f"Autodiff call {name!r} requires two arguments"
                    )
                binary = IRExpression(
                    "binary",
                    expression.span,
                    expression.type,
                    "*" if name == "scale" else "-",
                    children=expression.children[1:],
                )
                return self._gradient_tape_expression(
                    binary,
                    names,
                    lines,
                    next_node,
                )
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
            source_type = values[0].type
            length = (
                str(source_type.dimensions[0])
                if source_type.name in {"Tensor", "Array"}
                else f"{names[0]}.length"
            )
            prelude.append(f"(void){names[0]};")
            return _CValue(
                tuple(prelude),
                f"((int64_t)INT64_C({length}))"
                if source_type.name in {"Tensor", "Array"}
                else f"((int64_t){length})",
                expression.type,
                cleanup=tuple(cleanup),
            )

        if name in {"sum", "mean", "max_abs"}:
            source_type = values[0].type
            element_type = source_type.element_type
            assert element_type is not None
            count = self._tensor_size(source_type)
            scalar_type = self.c_type(expression.type)
            declaration = self._new_temp(expression.type)
            result = declaration.rsplit(" ", 1)[1]
            i = f"aether_i_{self.temp_index}"
            self.temp_index += 1
            item = f"{names[0]}.data[{i}]"
            integer_element = element_type.name in _INTEGER_TYPES
            if name == "mean" and integer_element:
                accumulator = f"aether_sum_{i}"
                prelude.append(f"double {accumulator} = 0.0;")
                prelude.append(
                    f"for (size_t {i} = 0; {i} < {count}; ++{i}) {{"
                )
                prelude.append(f"    {accumulator} += (double){item};")
                prelude.append("}")
                prelude.append(
                    f"{declaration} = ({scalar_type})"
                    f"({accumulator} / (double){count});"
                )
                return _CValue(
                    tuple(prelude),
                    result,
                    expression.type,
                    cleanup=tuple(cleanup),
                )
            initial = (
                self._absolute_code(
                    f"{names[0]}.data[0]",
                    element_type.name,
                )
                if name == "max_abs"
                else "0"
                if integer_element
                else "0.0"
            )
            prelude.append(f"{declaration} = ({scalar_type})({initial});")
            prelude.append(
                f"for (size_t {i} = 0; {i} < {count}; ++{i}) {{"
            )
            if name == "sum" and integer_element:
                prelude.append(
                    f"    {result} = aether_{element_type.name}_add({result}, {item});"
                )
            elif name == "sum":
                prelude.append(f"    {result} += {item};")
            elif name == "mean":
                prelude.append(f"    {result} += {item};")
            else:
                absolute = self._absolute_code(item, element_type.name)
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
                self._absolute_code(names[0], values[0].type.name),
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
                element_type = result_type.element_type
                assert element_type is not None
                value = self._absolute_code(item, element_type.name)
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
    def _absolute_code(value: str, type_name: str) -> str:
        if type_name in _SIGNED_TYPES:
            return f"aether_{type_name}_abs({value})"
        if type_name in _INTEGER_TYPES:
            return value
        zero = "0.0f" if type_name == "f32" else "0.0"
        return f"({value} == {zero} ? {zero} : ({value} < {zero} ? -{value} : {value}))"

    @staticmethod
    def _operation_name(operator: str) -> str:
        return {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
            "%": "rem",
            "&": "band",
            "|": "bor",
            "^": "bxor",
            "<<": "shl",
            ">>": "shr",
        }[operator]

    @staticmethod
    def _decode_string_literal(value: str) -> str:
        if value.startswith('"'):
            decoded = json.loads(value)
        elif value.startswith("'") and value.endswith("'"):
            content = value[1:-1]
            decoded = re.sub(
                r"\\(.)",
                lambda match: {
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                    "\\": "\\",
                    "'": "'",
                }.get(match.group(1), match.group(1)),
                content,
            )
        else:
            raise ValueError("Malformed string literal")
        if not isinstance(decoded, str):
            raise ValueError("String literal did not decode to text")
        return decoded

    def _literal_code(self, expression: IRExpression) -> str:
        value = expression.value
        type_name = expression.type.name
        if type_name == "str":
            decoded = self._decode_string_literal(value).encode("utf-8")
            data = "".join(f"\\{byte:03o}" for byte in decoded)
            return (
                "((aether_string){(const unsigned char *)\""
                + data
                + f"\", {len(decoded)}u}})"
            )
        if type_name == "bool":
            return "1" if value == "true" else "0"
        if type_name in _INTEGER_TYPES:
            width = type_name[1:]
            macro = ("UINT" if type_name.startswith("u") else "INT") + f"{width}_C"
            integer = int(
                value.replace("_", ""),
                0 if value.lower().startswith(("0x", "0b")) else 10,
            )
            return f"(({_C_TYPES[type_name]}){macro}({integer}))"
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
            "#if !defined(_WIN32)",
            "#define _POSIX_C_SOURCE 200809L",
            "#endif",
            "#include <stdint.h>",
            "#include <stdbool.h>",
            "#include <limits.h>",
            "#include <stddef.h>",
        ]
        if (
            self.needs_host_io
            or self.needs_host_fs
            or self.needs_process_args
            or any(
                type_.name in {"List", "Slice"}
                for type_ in self.collection_types.values()
            )
        ):
            output.extend(
                [
                    "#include <errno.h>",
                    "#include <stdio.h>",
                    "#include <stdlib.h>",
                    "#include <string.h>",
                    "#if defined(_WIN32)",
                    "#include <direct.h>",
                    "#include <io.h>",
                    "#include <sys/stat.h>",
                    "#include <windows.h>",
                    "#else",
                    "#include <sys/stat.h>",
                    "#include <unistd.h>",
                    "#endif",
                ]
            )
        if self.needs_host_fs or self.needs_process_args:
            output.append("#include <stdint.h>")
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
        for declaration in self._ordered_struct_declarations():
            output.extend(
                [
                    "typedef struct {",
                    *[
                        f"    {self.c_type(field_type)} "
                        f"{_encoded_identifier('aether_field_', field_name)};"
                        for field_name, field_type in declaration.fields
                    ],
                    f"}} {self.c_type(AetherType(declaration.name))};",
                    "",
                ]
            )
        for name, type_ in sorted(self.collection_types.items()):
            element = self.c_type(type_.arguments[0])
            if type_.name == "Array":
                output.extend(
                    [
                        f"typedef struct {{ {element} data[{type_.dimensions[0]}]; }} "
                        f"{name};",
                        "",
                    ]
                )
            elif type_.name == "List":
                storage_name = name + "_storage"
                output.extend(
                    [
                        f"typedef struct {{ size_t references; size_t capacity; "
                        f"{element} values[]; }} {storage_name};",
                        f"typedef struct {{ {storage_name} *storage; "
                        f"size_t length; }} {name};",
                        "",
                    ]
                )
            else:
                storage_name = name + "_storage"
                output.extend(
                    [
                        f"typedef struct {{ size_t references; size_t capacity; "
                        f"{element} values[]; }} {storage_name};",
                        f"typedef struct {{ {storage_name} *storage; "
                        f"size_t start; size_t length; }} {name};",
                        "",
                    ]
                )
        if any(
            type_.name in {"List", "Slice"}
            for type_ in self.collection_types.values()
        ):
            output.extend(
                [
                    "typedef struct {",
                    "    void *storage;",
                    "    void (*release)(void *);",
                    "} aether_owned_slot;",
                    "",
                ]
            )
        if self.needs_string:
                output.extend(
                    [
                        "typedef struct {",
                        "    const unsigned char *data;",
                        "    size_t length;",
                        "} aether_string;",
                        "",
                    ]
                )
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
                        f"    {self._result_value_c_type(value_type)} value;",
                        "    int error;",
                        f"}} {name};",
                        "",
                    ]
                )
        if self.needs_host_io or self.needs_host_fs or self.needs_process_args:
            output.extend(self._emit_runtime_helpers())
            output.append("")
        if self.needs_trap:
            if self.target_mcu:
                output.extend(
                    [
                        "#ifndef AETHER_TRAP",
                        "#define AETHER_TRAP() do { for (;;) { } } while (0)",
                        "#endif",
                        "",
                    ]
                )
            else:
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
        if (
            self.needs_trap
            or self.helpers
            or any(
                type_.name in {"List", "Slice"}
                for type_ in self.collection_types.values()
            )
        ):
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
            if function.c_exported:
                output.extend(self._emit_c_export(function))
                output.append("")
        if self.include_entry_point:
            output.extend(self._emit_entry_point())
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
        parameters: list[str] = []
        for parameter in function.parameters:
            name = _encoded_identifier("aether_v_", parameter.name)
            if parameter.type.name in {"Slice", "SliceMut"}:
                pointer_type = "const uint8_t *" if parameter.type.name == "Slice" else "uint8_t *"
                parameters.extend(
                    (f"{pointer_type}{name}", f"size_t {name}_len")
                )
            else:
                parameters.append(f"{self.c_type(parameter.type)} {name}")
        symbol = function.external_symbol or function.name
        return f"extern {return_type} {symbol}({', '.join(parameters) or 'void'});"

    def _c_export_signature(self, function: IRFunction) -> str:
        parameters = ", ".join(
            f"{self.c_type(parameter.type)} "
            f"{_encoded_identifier('aether_v_', parameter.name)}"
            for parameter in function.parameters
        ) or "void"
        return f"{self.c_type(function.return_type)} {function.name}({parameters})"

    def _emit_c_export(self, function: IRFunction) -> list[str]:
        signature = self._c_export_signature(function)
        arguments = ", ".join(
            _encoded_identifier("aether_v_", parameter.name)
            for parameter in function.parameters
        )
        call = (
            f"{_encoded_identifier('aether_fn_', function.name)}({arguments})"
        )
        if function.return_type.name == "void":
            return [signature + " {", f"    {call};", "}"]
        return [signature + " {", f"    return {call};", "}"]

    def _call_target(self, name: str) -> str:
        function = self.functions[name]
        if function.external_abi is not None:
            return function.external_symbol or function.name
        return _encoded_identifier("aether_fn_", name)

    @staticmethod
    def _control_condition_code(
        expression: IRExpression,
        value: _CValue,
    ) -> str:
        if (
            expression.kind in {"binary", "unary"}
            and value.code.startswith("(")
            and value.code.endswith(")")
        ):
            return value.code
        return f"({value.code})"

    def _emit_control_statements(
        self,
        statements: tuple[IRStatement, ...],
        indentation: str,
        has_cleanup: bool,
    ) -> list[str]:
        lines: list[str] = []
        for statement in statements:
            if statement.kind == "if":
                assert statement.expression is not None
                condition = self.expression(statement.expression)
                lines.extend(indentation + item for item in condition.prelude)
                condition_code = self._control_condition_code(
                    statement.expression,
                    condition,
                )
                lines.append(indentation + f"if {condition_code} {{")
                self._enter_collection_scope()
                lines.extend(
                    self._emit_control_statements(
                        statement.body,
                        indentation + "    ",
                        has_cleanup,
                    )
                )
                lines.extend(
                    self._leave_collection_scope(indentation + "    ")
                )
                if statement.else_body:
                    lines.append(indentation + "} else {")
                    self._enter_collection_scope()
                    lines.extend(
                        self._emit_control_statements(
                            statement.else_body,
                            indentation + "    ",
                            has_cleanup,
                        )
                    )
                    lines.extend(
                        self._leave_collection_scope(indentation + "    ")
                    )
                lines.append(indentation + "}")
                lines.extend(indentation + item for item in condition.cleanup)
                continue
            if statement.kind == "while":
                assert statement.expression is not None
                lines.append(indentation + "while (1) {")
                condition = self.expression(statement.expression)
                lines.extend(indentation + "    " + item for item in condition.prelude)
                condition_code = self._control_condition_code(
                    statement.expression,
                    condition,
                )
                lines.append(
                    indentation + f"    if (!{condition_code}) break;"
                )
                loop_scope_depth = len(self.collection_scope_stack)
                self.collection_loop_scope_depths.append(loop_scope_depth)
                self._enter_collection_scope()
                lines.extend(
                    self._emit_control_statements(
                        statement.body,
                        indentation + "    ",
                        has_cleanup,
                    )
                )
                lines.extend(
                    self._leave_collection_scope(indentation + "    ")
                )
                self.collection_loop_scope_depths.pop()
                lines.extend(indentation + "    " + item for item in condition.cleanup)
                lines.append(indentation + "}")
                continue
            if statement.kind == "for":
                if len(statement.expressions) != 2:
                    continue
                start = self.expression(statement.expressions[0])
                end = self.expression(statement.expressions[1])
                start_temp = self._new_temp(start.type)
                end_temp = self._new_temp(end.type)
                start_name = start_temp.rsplit(" ", 1)[1]
                end_name = end_temp.rsplit(" ", 1)[1]
                loop_name = _encoded_identifier("aether_v_", statement.name)
                lines.append(indentation + "{")
                lines.extend(indentation + "    " + item for item in start.prelude)
                lines.append(
                    indentation + f"    const {start_temp} = {start.code};"
                )
                lines.extend(indentation + "    " + item for item in start.cleanup)
                lines.extend(indentation + "    " + item for item in end.prelude)
                lines.append(
                    indentation + f"    const {end_temp} = {end.code};"
                )
                lines.extend(indentation + "    " + item for item in end.cleanup)
                lines.append(
                    indentation
                    + f"    for ({self.c_type(start.type)} {loop_name} = "
                    + f"{start_name}; {loop_name} < {end_name}; ++{loop_name}) {{"
                )
                loop_scope_depth = len(self.collection_scope_stack)
                self.collection_loop_scope_depths.append(loop_scope_depth)
                self._enter_collection_scope()
                lines.extend(
                    self._emit_control_statements(
                        statement.body,
                        indentation + "        ",
                        has_cleanup,
                    )
                )
                lines.extend(
                    self._leave_collection_scope(indentation + "        ")
                )
                self.collection_loop_scope_depths.pop()
                lines.extend([indentation + "    }", indentation + "}"])
                continue
            if statement.kind in {"break", "continue"}:
                if self.collection_loop_scope_depths:
                    lines.extend(
                        self._collection_scope_release_lines(
                            indentation,
                            self.collection_loop_scope_depths[-1],
                        )
                    )
                lines.append(indentation + statement.kind + ";")
                continue
            expression = statement.expression
            if expression is None:
                if statement.kind == "return":
                    lines.append(
                        indentation
                        + (
                            f"goto {self.frame_cleanup_label};"
                            if has_cleanup
                            else "return;"
                        )
                    )
                continue
            value = self.expression(expression)
            if statement.kind == "assignment" and statement.operator != "=":
                value = self.expression(
                    IRExpression(
                        "binary",
                        statement.span,
                        expression.type,
                        statement.operator[:-1],
                        (
                            IRExpression(
                                "identifier",
                                statement.span,
                                expression.type,
                                statement.name,
                            ),
                            expression,
                        ),
                    )
                )
            lines.extend(indentation + item for item in value.prelude)
            if statement.kind == "let":
                qualifier = (
                    "const "
                    if statement.binding_kind in {"let", "const"}
                    else ""
                )
                name = _encoded_identifier("aether_v_", statement.name)
                lines.append(
                    indentation
                    + f"{qualifier}{self.c_type(value.type)} {name} = {value.code};"
                )
                if self._contains_owned_collection(value.type):
                    if value.ownership == "borrowed":
                        retain = self._retain_code(name, value.type)
                        if retain:
                            lines.append(indentation + retain)
                    slot = self._bind_collection_slot(statement.name)
                    lines.extend(
                        self._collection_slot_set_lines(
                            slot,
                            name,
                            value.type,
                            indentation,
                        )
                    )
            elif statement.kind == "assignment":
                name = _encoded_identifier("aether_v_", statement.name)
                if statement.field_name is None:
                    if self._contains_owned_collection(value.type):
                        temporary = self._new_temp(value.type)
                        temporary_name = temporary.rsplit(" ", 1)[1]
                        lines.append(
                            indentation + f"{temporary} = {value.code};"
                        )
                        if value.ownership == "borrowed":
                            retain = self._retain_code(
                                temporary_name,
                                value.type,
                            )
                            if retain:
                                lines.append(indentation + retain)
                        slot = self._lookup_collection_slot(statement.name)
                        if slot is not None:
                            lines.append(
                                indentation
                                + f"aether_owned_slot_release("
                                + f"&{self.collection_slots_name}[{slot}]);"
                            )
                        lines.append(
                            indentation + f"{name} = {temporary_name};"
                        )
                        if slot is not None:
                            lines.extend(
                                self._collection_slot_set_lines(
                                    slot,
                                    name,
                                    value.type,
                                    indentation,
                                )
                            )
                        lines.extend(indentation + item for item in value.cleanup)
                    else:
                        lines.append(indentation + f"{name} = {value.code};")
                        lines.extend(indentation + item for item in value.cleanup)
                else:
                    field = _encoded_identifier(
                        "aether_field_",
                        statement.field_name,
                    )
                    lines.append(
                        indentation + f"{name}.{field} = {value.code};"
                    )
                    lines.extend(indentation + item for item in value.cleanup)
            elif statement.kind == "return":
                if self.current_function is not None and (
                    self.current_function.return_type.name == "void"
                ):
                    if expression is not None:
                        lines.append(indentation + f"(void)({value.code});")
                    lines.extend(indentation + item for item in value.cleanup)
                    lines.append(
                        indentation
                        + (
                            f"goto {self.frame_cleanup_label};"
                            if has_cleanup
                            else "return;"
                        )
                    )
                elif has_cleanup and self.current_function is not None:
                    return_type = self.current_function.return_type
                    if return_type.name == "Result" and value.type.name != "Result":
                        return_value = (
                            f"({self.c_type(return_type)})"
                            f"{{1, {value.code}, 0}}"
                        )
                    else:
                        return_value = value.code
                    lines.append(
                        indentation
                        + f"{self.frame_return_name} = {return_value};"
                    )
                    if (
                        self._contains_frame_buffer(value.type)
                        or self._contains_owned_collection(value.type)
                    ) and value.ownership == "borrowed":
                        retain = self._retain_code(
                            self.frame_return_name,
                            return_type,
                        )
                        if retain:
                            lines.append(indentation + retain)
                    lines.extend(indentation + item for item in value.cleanup)
                    lines.append(
                        indentation + f"goto {self.frame_cleanup_label};"
                    )
                elif value.cleanup:
                    result_temp = self._new_temp(value.type)
                    result_name = result_temp.rsplit(" ", 1)[1]
                    lines.append(
                        indentation + f"{result_temp} = {value.code};"
                    )
                    lines.extend(indentation + item for item in value.cleanup)
                    lines.append(indentation + f"return {result_name};")
                else:
                    lines.append(indentation + f"return {value.code};")
                continue
            else:
                lines.append(indentation + f"(void)({value.code});")
            lines.extend(indentation + item for item in value.cleanup)
        return lines

    def _emit_function(self, function: IRFunction) -> list[str]:
        self.temp_index = 0
        self.current_function = function
        self.collection_slots_name = "aether_owned_slots"
        self.collection_slot_next = 0
        self.collection_bindings = {}
        self.collection_scope_stack = []
        self.collection_scope_slots = []
        self.collection_loop_scope_depths = []
        owned_collection_parameters = [
            parameter
            for parameter in function.parameters
            if self._contains_owned_collection(parameter.type)
        ]
        owned_collection_locals = [
            statement
            for statement in _walk_ir_statements(function.body)
            if statement.kind == "let"
            and statement.expression is not None
            and self._contains_owned_collection(statement.expression.type)
        ]
        has_owned_collections = bool(
            owned_collection_parameters
            or owned_collection_locals
            or self._contains_owned_collection(function.return_type)
            or any(
                self._contains_owned_collection(expression.type)
                for statement in _walk_ir_statements(function.body)
                for expression in _statement_expressions(statement)
            )
        )
        self.collection_slot_count = (
            len(owned_collection_parameters) + len(owned_collection_locals)
        )
        has_try = any(
            self._expression_contains_try(expression)
            for statement in _walk_ir_statements(function.body)
            for expression in _statement_expressions(statement)
        )
        self.frame_return_name = "aether_frame_return"
        self.frame_cleanup_label = "aether_frame_cleanup"
        has_cleanup = (
            has_try
            or bool(function.frame_resources)
            or self._contains_frame_buffer(function.return_type)
            or has_owned_collections
        )
        lines = [self._function_signature(function) + " {"]
        if self.collection_slot_count:
            lines.append(
                f"    aether_owned_slot {self.collection_slots_name}["
                f"{max(self.collection_slot_count, 1)}] = {{0}};"
            )
        self._enter_collection_scope()
        for parameter in owned_collection_parameters:
            name = _encoded_identifier("aether_v_", parameter.name)
            slot = self._bind_collection_slot(parameter.name)
            lines.extend(
                self._collection_slot_set_lines(
                    slot,
                    name,
                    parameter.type,
                    "    ",
                )
            )
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
            if statement.kind in {"if", "while", "for"}:
                lines.extend(
                    self._emit_control_statements(
                        (statement,),
                        "    ",
                        has_cleanup,
                    )
                )
                continue
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
            if statement.kind == "assignment" and statement.operator != "=":
                target_type = expression.type
                compound_expression = IRExpression(
                    "binary",
                    statement.span,
                    target_type,
                    statement.operator[:-1],
                    (
                        IRExpression(
                            "identifier",
                            statement.span,
                            target_type,
                            statement.name,
                        ),
                        expression,
                    ),
                )
                value = self.expression(compound_expression)
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
                elif self._contains_owned_collection(expression.type):
                    qualifier = (
                        "const "
                        if statement.binding_kind in {"let", "const"}
                        else ""
                    )
                    lines.append(
                        f"    {qualifier}{self.c_type(expression.type)} "
                        f"{name} = {value.code};"
                    )
                    if value.ownership == "borrowed":
                        retain = self._retain_code(name, expression.type)
                        if retain:
                            lines.append(f"    {retain}")
                    slot = self._bind_collection_slot(statement.name)
                    lines.extend(
                        self._collection_slot_set_lines(
                            slot,
                            name,
                            expression.type,
                            "    ",
                        )
                    )
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
                        or self._contains_owned_collection(value.type)
                    ) and value.ownership == "borrowed":
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
                target = (
                    name
                    if statement.field_name is None
                    else f"{name}."
                    + _encoded_identifier(
                        "aether_field_",
                        statement.field_name,
                    )
                )
                if (
                    self._contains_frame_buffer(expression.type)
                    or self._contains_owned_collection(expression.type)
                ):
                    temporary = f"aether_owner_{self.temp_index}"
                    self.temp_index += 1
                    lines.append(
                        f"    {self.c_type(expression.type)} {temporary} = "
                        f"{value.code};"
                    )
                    if value.ownership == "borrowed":
                        retain = self._retain_code(temporary, expression.type)
                        if retain:
                            lines.append(f"    {retain}")
                    lines.extend("    " + item for item in value.cleanup)
                    if self._contains_frame_buffer(expression.type):
                        release = self._release_code(name, expression.type)
                        if release:
                            lines.append(f"    {release}")
                    elif self._contains_owned_collection(expression.type):
                        slot = self._lookup_collection_slot(statement.name)
                        if slot is not None:
                            lines.append(
                                f"    aether_owned_slot_release("
                                f"&{self.collection_slots_name}[{slot}]);"
                            )
                    lines.append(f"    {target} = {temporary};")
                    if self._contains_owned_collection(expression.type):
                        slot = self._lookup_collection_slot(statement.name)
                        if slot is not None:
                            lines.extend(
                                self._collection_slot_set_lines(
                                    slot,
                                    target,
                                    expression.type,
                                    "    ",
                                )
                            )
                else:
                    lines.append(f"    {target} = {value.code};")
                    lines.extend("    " + item for item in value.cleanup)
            elif statement.kind == "expression":
                if (
                    self._contains_frame_buffer(value.type)
                    or self._contains_owned_collection(value.type)
                ) and value.ownership == "owned":
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
        lines.extend(self._leave_collection_scope("    "))
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
            if has_owned_collections:
                for slot in reversed(range(self.collection_slot_count)):
                    lines.append(
                        f"    aether_owned_slot_release("
                        f"&{self.collection_slots_name}[{slot}]);"
                    )
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

    def _emit_runtime_helpers(self) -> list[str]:
        if self.shared_runtime:
            lines: list[str] = ["#define AETHER_PATH_MAX 4096"]
            storage = "extern " if not self.include_entry_point else ""
            initializer = "" if not self.include_entry_point else " = 0"
            argv_initializer = "" if not self.include_entry_point else " = NULL"
            root_initializer = "" if not self.include_entry_point else " = {0}"
            lines.extend(
                [
                    f"{storage}int aether_runtime_initialized{initializer};",
                    f"{storage}int aether_runtime_argc{initializer};",
                    f"{storage}char **aether_runtime_argv{argv_initializer};",
                    f"{storage}char aether_project_root[AETHER_PATH_MAX]{root_initializer};",
                ]
            )
        else:
            lines = [
                "#define AETHER_PATH_MAX 4096",
                "static int aether_runtime_initialized = 0;",
                "static int aether_runtime_argc = 0;",
                "static char **aether_runtime_argv = NULL;",
                "static char aether_project_root[AETHER_PATH_MAX];",
            ]
        needs_owned_string = bool(
            self.module_runtime_calls & {"io_read_line", "fs_read_text"}
        )
        if needs_owned_string:
            lines.extend(
                [
                    "typedef struct aether_string_owner {",
                    "    unsigned char *data;",
                    "    struct aether_string_owner *next;",
                    "} aether_string_owner;",
                    (
                        f"{'extern ' if self.shared_runtime and not self.include_entry_point else ''}"
                        f"{'static ' if not self.shared_runtime else ''}"
                        "aether_string_owner *aether_string_owners"
                        f"{' = NULL' if self.include_entry_point or not self.shared_runtime else ''};"
                    ),
                ]
            )
            if self.runtime_calls & {"io_read_line", "fs_read_text"}:
                lines.extend(
                    [
                        "",
                        "static int aether_string_copy(",
                        "    const unsigned char *data, size_t length, aether_string *out",
                        ") {",
                        "    unsigned char *copy = (unsigned char *)malloc(length + 1u);",
                        "    aether_string_owner *owner =",
                        "        (aether_string_owner *)malloc(sizeof(*owner));",
                        "    if (copy == NULL || owner == NULL) {",
                        "        free(copy);",
                        "        free(owner);",
                        "        return 0;",
                        "    }",
                        "    if (length != 0u) memcpy(copy, data, length);",
                        "    copy[length] = 0u;",
                        "    owner->data = copy;",
                        "    owner->next = aether_string_owners;",
                        "    aether_string_owners = owner;",
                        "    out->data = copy;",
                        "    out->length = length;",
                        "    return 1;",
                        "}",
                    ]
                )
            if self.include_entry_point:
                lines.extend(
                    [
                        "",
                        "static void aether_runtime_cleanup(void) {",
                        "    while (aether_string_owners != NULL) {",
                        "        aether_string_owner *owner = aether_string_owners;",
                        "        aether_string_owners = owner->next;",
                        "        free(owner->data);",
                        "        free(owner);",
                        "    }",
                        "#if defined(_WIN32)",
                        "    if (aether_runtime_argv != NULL) {",
                        "        int i;",
                        "        for (i = 0; i < aether_runtime_argc; ++i)",
                        "            free(aether_runtime_argv[i]);",
                        "        free(aether_runtime_argv);",
                        "        aether_runtime_argv = NULL;",
                        "    }",
                        "#endif",
                        "}",
                    ]
                )
        elif self.include_entry_point:
            lines.extend(
                [
                    "",
                    "static void aether_runtime_cleanup(void) {",
                    "#if defined(_WIN32)",
                    "    if (aether_runtime_argv != NULL) {",
                    "        int i;",
                    "        for (i = 0; i < aether_runtime_argc; ++i)",
                    "            free(aether_runtime_argv[i]);",
                    "        free(aether_runtime_argv);",
                    "        aether_runtime_argv = NULL;",
                    "    }",
                    "#endif",
                    "}",
                ]
            )
        needs_utf8 = bool(
            self.runtime_calls
            & {
                "io_write",
                "io_read_line",
                "fs_read_text",
                "fs_write_text",
                "fs_exists",
                "args_get",
            }
        )
        if needs_utf8:
            lines.extend(
                [
                    "",
                    "static int aether_utf8_valid(",
                    "    const unsigned char *data, size_t length",
                    ") {",
                    "    size_t i = 0u;",
                    "    while (i < length) {",
                    "        unsigned char c = data[i++];",
                    "        if (c <= 0x7fu) continue;",
                    "        if (c >= 0xc2u && c <= 0xdfu) {",
                    "            if (i >= length || (data[i++] & 0xc0u) != 0x80u) return 0;",
                    "        } else if (c >= 0xe0u && c <= 0xefu) {",
                    "            unsigned char b1, b2;",
                    "            if (length - i < 2u) return 0;",
                    "            b1 = data[i++]; b2 = data[i++];",
                    "            if ((b1 & 0xc0u) != 0x80u || (b2 & 0xc0u) != 0x80u)",
                    "                return 0;",
                    "            if ((c == 0xe0u && b1 < 0xa0u) ||",
                    "                (c == 0xedu && b1 >= 0xa0u)) return 0;",
                    "        } else if (c >= 0xf0u && c <= 0xf4u) {",
                    "            unsigned char b1, b2, b3;",
                    "            if (length - i < 3u) return 0;",
                    "            b1 = data[i++]; b2 = data[i++]; b3 = data[i++];",
                    "            if ((b1 & 0xc0u) != 0x80u ||",
                    "                (b2 & 0xc0u) != 0x80u ||",
                    "                (b3 & 0xc0u) != 0x80u) return 0;",
                    "            if ((c == 0xf0u && b1 < 0x90u) ||",
                    "                (c == 0xf4u && b1 >= 0x90u)) return 0;",
                    "        } else {",
                    "            return 0;",
                    "        }",
                    "    }",
                    "    return 1;",
                    "}",
                ]
            )

        if self.runtime_calls or (
            self.include_entry_point and self.module_runtime_calls
        ):
            if self.shared_runtime and not self.include_entry_point:
                lines.extend(
                    [
                        "",
                        "int aether_runtime_init(int argc, char **argv);",
                    ]
                )
            else:
                lines.extend(
                    [
                        "",
                        (
                            "int aether_runtime_init(int argc, char **argv) {"
                            if self.shared_runtime
                            else "static int aether_runtime_init(int argc, char **argv) {"
                        ),
                    "    aether_runtime_argc = argc > 0 ? argc - 1 : 0;",
                    "#if defined(_WIN32)",
                    "    if (aether_runtime_argc != 0) {",
                    "        int i;",
                    "        aether_runtime_argv = (char **)calloc(",
                    "            (size_t)aether_runtime_argc, sizeof(char *));",
                    "        if (aether_runtime_argv == NULL) return 0;",
                    "        for (i = 0; i < aether_runtime_argc; ++i) {",
                    "            int wide_length = MultiByteToWideChar(",
                    "                CP_ACP, MB_ERR_INVALID_CHARS, argv[i + 1], -1, NULL, 0);",
                    "            wchar_t *wide;",
                    "            int utf8_length;",
                    "            char *utf8;",
                    "            if (wide_length == 0) { aether_runtime_argv[i] = NULL; continue; }",
                    "            wide = (wchar_t *)malloc((size_t)wide_length * sizeof(wchar_t));",
                    "            if (wide == NULL) return 0;",
                    "            if (MultiByteToWideChar(CP_ACP, MB_ERR_INVALID_CHARS,",
                    "                argv[i + 1], -1, wide, wide_length) == 0) {",
                    "                free(wide); aether_runtime_argv[i] = NULL; continue;",
                    "            }",
                    "            utf8_length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,",
                    "                wide, -1, NULL, 0, NULL, NULL);",
                    "            if (utf8_length == 0) { free(wide); aether_runtime_argv[i] = NULL; continue; }",
                    "            utf8 = (char *)malloc((size_t)utf8_length);",
                    "            if (utf8 == NULL) { free(wide); return 0; }",
                    "            if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,",
                    "                wide, -1, utf8, utf8_length, NULL, NULL) == 0) {",
                    "                free(wide); free(utf8); aether_runtime_argv[i] = NULL; continue;",
                    "            }",
                    "            free(wide); aether_runtime_argv[i] = utf8;",
                    "        }",
                    "    }",
                    "#else",
                    "    aether_runtime_argv = argc > 1 ? argv + 1 : NULL;",
                    "#endif",
                    "#if defined(_WIN32)",
                    "    if (_getcwd(aether_project_root, AETHER_PATH_MAX) == NULL)",
                    "        return 0;",
                    "    if (_fullpath(aether_project_root, aether_project_root,",
                    "                  AETHER_PATH_MAX) == NULL) return 0;",
                    "#else",
                    "    if (getcwd(aether_project_root, AETHER_PATH_MAX) == NULL)",
                    "        return 0;",
                    "    {",
                    "        char canonical[AETHER_PATH_MAX];",
                    "        if (realpath(aether_project_root, canonical) == NULL) return 0;",
                    "        memcpy(aether_project_root, canonical, strlen(canonical) + 1u);",
                    "    }",
                    "#endif",
                    "    aether_runtime_initialized = 1;",
                    "    return 1;",
                    "}",
                    ]
                )
            if self.runtime_calls:
                lines.extend(
                    [
                        "",
                        "static int aether_runtime_ensure(void) {",
                        "    char *argv[1] = {NULL};",
                        "    return aether_runtime_initialized ||",
                        "        aether_runtime_init(0, argv);",
                        "}",
                    ]
                )

        if "args_len" in self.runtime_calls:
            lines.extend(
                [
                    "",
                    "static int64_t aether_args_len(void) {",
                    "    if (!aether_runtime_ensure()) return 0;",
                    "    return (int64_t)aether_runtime_argc;",
                    "}",
                ]
            )
        if "args_get" in self.runtime_calls:
            result = self._result_c_name(
                AetherType("Result", (AetherType("str"), AetherType("ArgsError")))
            )
            lines.extend(
                [
                    "",
                    f"static {result} aether_args_get(int64_t index) {{",
                    f"    {result} result = {{0}};",
                    "    const unsigned char *data;",
                    "    size_t length;",
                    "    if (!aether_runtime_ensure() || index < 0 ||",
                    "        index >= (int64_t)aether_runtime_argc) {",
                    "        result.error = 1;",
                    "        return result;",
                    "    }",
                    "    if (aether_runtime_argv[index] == NULL) {",
                    "        result.error = 2; return result;",
                    "    }",
                    "    data = (const unsigned char *)aether_runtime_argv[index];",
                    "    length = strlen((const char *)data);",
                    "    if (!aether_utf8_valid(data, length)) {",
                    "        result.error = 2;",
                    "        return result;",
                    "    }",
                    "    result.is_ok = 1;",
                    "    result.value.data = data;",
                    "    result.value.length = length;",
                    "    return result;",
                    "}",
                ]
            )

        if "io_write" in self.runtime_calls:
            result = self._result_c_name(
                AetherType("Result", (AetherType("void"), AetherType("IoError")))
            )
            lines.extend(
                [
                    "",
                    f"static {result} aether_io_write(aether_string text) {{",
                    f"    {result} result = {{0}};",
                    "    if (!aether_runtime_ensure()) { result.error = 3; return result; }",
                    "    if (!aether_utf8_valid(text.data, text.length)) {",
                    "        result.error = 2; return result;",
                    "    }",
                    "    if ((text.length != 0u &&",
                    "         fwrite(text.data, 1u, text.length, stdout) != text.length) ||",
                    "        fflush(stdout) != 0) {",
                    "        result.error = 3; return result;",
                    "    }",
                    "    result.is_ok = 1;",
                    "    return result;",
                    "}",
                ]
            )

        if "io_read_line" in self.runtime_calls:
            result = self._result_c_name(
                AetherType("Result", (AetherType("str"), AetherType("IoError")))
            )
            lines.extend(
                [
                    "",
                    f"static {result} aether_io_read_line(void) {{",
                    f"    {result} result = {{0}};",
                    "    unsigned char *buffer = NULL;",
                    "    size_t length = 0u;",
                    "    size_t capacity = 0u;",
                    "    int ch;",
                    "    if (!aether_runtime_ensure()) { result.error = 3; return result; }",
                    "    while ((ch = fgetc(stdin)) != EOF && ch != '\\n') {",
                    "        unsigned char *next;",
                    "        if (length == capacity) {",
                    "            size_t new_capacity = capacity == 0u ? 128u : capacity * 2u;",
                    "            if (new_capacity < capacity) { free(buffer); result.error = 3; return result; }",
                    "            next = (unsigned char *)realloc(buffer, new_capacity);",
                    "            if (next == NULL) { free(buffer); result.error = 3; return result; }",
                    "            buffer = next; capacity = new_capacity;",
                    "        }",
                    "        buffer[length++] = (unsigned char)ch;",
                    "    }",
                    "    if (ferror(stdin)) { free(buffer); result.error = 3; return result; }",
                    "    if (ch == EOF && length == 0u) {",
                    "        free(buffer); result.error = 1; return result;",
                    "    }",
                    "    if (length != 0u && buffer[length - 1u] == '\\r') --length;",
                    "    if (!aether_utf8_valid(buffer, length)) {",
                    "        free(buffer); result.error = 2; return result;",
                    "    }",
                    "    if (!aether_string_copy(buffer, length, &result.value)) {",
                    "        free(buffer); result.error = 3; return result;",
                    "    }",
                    "    free(buffer);",
                    "    result.is_ok = 1;",
                    "    return result;",
                    "}",
                ]
            )

        if self.runtime_calls & {"fs_read_text", "fs_write_text", "fs_exists"}:
            lines.extend(self._emit_filesystem_helpers())
        return lines

    def _emit_filesystem_helpers(self) -> list[str]:
        lines = [
            "",
            "static int aether_path_within_root(const char *path) {",
            "    size_t root_length = strlen(aether_project_root);",
            "#if defined(_WIN32)",
            "    if (_strnicmp(path, aether_project_root, root_length) != 0) return 0;",
            "#else",
            "    if (strncmp(path, aether_project_root, root_length) != 0) return 0;",
            "#endif",
            "    if (root_length > 1u &&",
            "        (aether_project_root[root_length - 1u] == '/' ||",
            "         aether_project_root[root_length - 1u] == '\\\\')) return 1;",
            "    return path[root_length] == '\\0' || path[root_length] == '/' ||",
            "        path[root_length] == '\\\\';",
            "}",
            "#if defined(_WIN32)",
            "",
            "static int aether_path_has_reparse_point(",
            "    const char *root, const char *relative",
            ") {",
            "    char current[AETHER_PATH_MAX];",
            "    size_t length = strlen(root), i = 0u;",
            "    if (length >= sizeof(current)) return -1;",
            "    memcpy(current, root, length + 1u);",
            "    while (relative[i] != '\\0') {",
            "        size_t start = i;",
            "        DWORD attributes;",
            "        while (relative[i] != '\\0' && relative[i] != '/') ++i;",
            "        if (length != 0u && current[length - 1u] != '/' &&",
            "            current[length - 1u] != '\\\\') {",
            "            if (length + 1u >= sizeof(current)) return -1;",
            "            current[length++] = '\\\\';",
            "        }",
            "        if (length + i - start >= sizeof(current)) return -1;",
            "        memcpy(current + length, relative + start, i - start);",
            "        length += i - start;",
            "        current[length] = '\\0';",
            "        attributes = GetFileAttributesA(current);",
            "        if (attributes == INVALID_FILE_ATTRIBUTES) {",
            "            DWORD error = GetLastError();",
            "            if (error == ERROR_FILE_NOT_FOUND ||",
            "                error == ERROR_PATH_NOT_FOUND) return 0;",
            "            return -1;",
            "        }",
            "        if ((attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0u) return 1;",
            "        if (relative[i] == '/') ++i;",
            "    }",
            "    return 0;",
            "}",
            "#endif",
            "",
            "static int aether_fs_resolve(",
            "    aether_string path, int writing, char output[AETHER_PATH_MAX]",
            ") {",
            "    char relative[AETHER_PATH_MAX];",
            "    char candidate[AETHER_PATH_MAX];",
            "    size_t i, root_length, path_length;",
            "    if (!aether_runtime_ensure()) return 4;",
            "    if (!aether_utf8_valid(path.data, path.length)) return 3;",
            "    if (path.length == 0u || path.length >= sizeof(relative)) return 2;",
            "    for (i = 0u; i < path.length; ++i) {",
            "        if (path.data[i] == 0u || path.data[i] == ':') return 2;",
            "        relative[i] = path.data[i] == '\\\\' ? '/' : (char)path.data[i];",
            "    }",
            "    relative[path.length] = '\\0';",
            "    if (relative[0] == '/' || relative[0] == '\\\\') return 2;",
            "    for (i = 0u; i < path.length; ++i) {",
            "        size_t start;",
            "        if (i != 0u && relative[i - 1u] != '/') continue;",
            "        start = i;",
            "        while (i < path.length && relative[i] != '/') ++i;",
            "        if (i - start == 2u && relative[start] == '.' &&",
            "            relative[start + 1u] == '.') return 2;",
            "    }",
            "    root_length = strlen(aether_project_root);",
            "    path_length = strlen(relative);",
            "    if (root_length + path_length + 2u >= sizeof(candidate)) return 2;",
            "    memcpy(candidate, aether_project_root, root_length);",
            "    candidate[root_length] = '/';",
            "    memcpy(candidate + root_length + 1u, relative, path_length + 1u);",
            "#if defined(_WIN32)",
            "    {",
            "        int reparse = aether_path_has_reparse_point(",
            "            aether_project_root, relative);",
            "        if (reparse > 0) return 2;",
            "        if (reparse < 0) return 4;",
            "    }",
            "#endif",
            "    if (writing) {",
            "        char *separator = strrchr(candidate, '/');",
            "        char parent[AETHER_PATH_MAX];",
            "        char canonical[AETHER_PATH_MAX];",
            "        if (separator == NULL) return 2;",
            "        *separator = '\\0';",
            "#if defined(_WIN32)",
            "        if (_fullpath(canonical, candidate, sizeof(canonical)) == NULL) return errno == ENOENT ? 1 : 4;",
            "#else",
            "        if (realpath(candidate, canonical) == NULL) return errno == ENOENT ? 1 : 4;",
            "#endif",
            "        if (!aether_path_within_root(canonical)) return 2;",
            "        memcpy(parent, canonical, strlen(canonical) + 1u);",
            "        if (strlen(parent) + strlen(separator + 1) + 2u >= AETHER_PATH_MAX) return 2;",
            "        strcpy(output, parent);",
            "        strcat(output, \"/\");",
            "        strcat(output, separator + 1);",
            "#if !defined(_WIN32)",
            "        {",
            "            struct stat info;",
            "            char existing[AETHER_PATH_MAX];",
            "            if (lstat(output, &info) == 0) {",
            "                if (realpath(output, existing) == NULL) return 2;",
            "                if (!aether_path_within_root(existing)) return 2;",
            "                strcpy(output, existing);",
            "            } else if (errno != ENOENT) return 4;",
            "        }",
            "#endif",
            "        return 0;",
            "    }",
            "#if defined(_WIN32)",
            "    if (_fullpath(output, candidate, AETHER_PATH_MAX) == NULL)",
            "        return errno == ENOENT ? 1 : 4;",
            "#else",
            "    if (realpath(candidate, output) == NULL)",
            "        return errno == ENOENT ? 1 : 4;",
            "#endif",
            "    return aether_path_within_root(output) ? 0 : 2;",
            "}",
        ]
        if "fs_read_text" in self.runtime_calls:
            result = self._result_c_name(
                AetherType("Result", (AetherType("str"), AetherType("FsError")))
            )
            lines.extend(
                [
                    "",
                    f"static {result} aether_fs_read_text(aether_string path) {{",
                    f"    {result} result = {{0}};",
                    "    char resolved[AETHER_PATH_MAX];",
                    "    unsigned char *buffer = NULL;",
                    "    size_t length = 0u, capacity = 0u, count;",
                    "    FILE *file;",
                    "    int path_error = aether_fs_resolve(path, 0, resolved);",
                    "    if (path_error != 0) { result.error = path_error; return result; }",
                    "    file = fopen(resolved, \"rb\");",
                    "    if (file == NULL) { result.error = errno == ENOENT ? 1 : 4; return result; }",
                    "    for (;;) {",
                    "        unsigned char *next;",
                    "        if (capacity - length < 4096u) {",
                    "            size_t new_capacity = capacity == 0u ? 4096u : capacity * 2u;",
                    "            if (new_capacity < capacity) { fclose(file); free(buffer); result.error = 4; return result; }",
                    "            next = (unsigned char *)realloc(buffer, new_capacity);",
                    "            if (next == NULL) { fclose(file); free(buffer); result.error = 4; return result; }",
                    "            buffer = next; capacity = new_capacity;",
                    "        }",
                    "        count = fread(buffer + length, 1u, capacity - length, file);",
                    "        length += count;",
                    "        if (count == 0u) break;",
                    "    }",
                    "    {",
                    "        int read_error = ferror(file);",
                    "        int close_error = fclose(file) != 0;",
                    "        if (read_error || close_error) {",
                    "            free(buffer); result.error = 4; return result;",
                    "        }",
                    "    }",
                    "    if (!aether_utf8_valid(buffer, length)) {",
                    "        free(buffer); result.error = 3; return result;",
                    "    }",
                    "    if (!aether_string_copy(buffer, length, &result.value)) {",
                    "        free(buffer); result.error = 4; return result;",
                    "    }",
                    "    free(buffer); result.is_ok = 1; return result;",
                    "}",
                ]
            )
        if "fs_write_text" in self.runtime_calls:
            result = self._result_c_name(
                AetherType("Result", (AetherType("void"), AetherType("FsError")))
            )
            lines.extend(
                [
                    "",
                    f"static {result} aether_fs_write_text(",
                    "    aether_string path, aether_string text",
                    ") {",
                    f"    {result} result = {{0}};",
                    "    char resolved[AETHER_PATH_MAX];",
                    "    FILE *file;",
                    "    int path_error = aether_fs_resolve(path, 1, resolved);",
                    "    if (path_error != 0) { result.error = path_error; return result; }",
                    "    if (!aether_utf8_valid(text.data, text.length)) { result.error = 3; return result; }",
                    "    file = fopen(resolved, \"wb\");",
                    "    if (file == NULL) { result.error = errno == ENOENT ? 1 : 4; return result; }",
                    "    {",
                    "        int write_error = text.length != 0u &&",
                    "            fwrite(text.data, 1u, text.length, file) != text.length;",
                    "        int close_error = fclose(file) != 0;",
                    "        if (write_error || close_error) { result.error = 4; return result; }",
                    "    }",
                    "    result.is_ok = 1; return result;",
                    "}",
                ]
            )
        if "fs_exists" in self.runtime_calls:
            result = self._result_c_name(
                AetherType("Result", (AetherType("bool"), AetherType("FsError")))
            )
            lines.extend(
                [
                    "",
                    f"static {result} aether_fs_exists(aether_string path) {{",
                    f"    {result} result = {{0}};",
                    "    char resolved[AETHER_PATH_MAX];",
                    "    struct stat info;",
                    "    int path_error = aether_fs_resolve(path, 0, resolved);",
                    "    if (path_error == 1) { result.is_ok = 1; result.value = 0; return result; }",
                    "    if (path_error != 0) { result.error = path_error; return result; }",
                    "    if (stat(resolved, &info) != 0) {",
                    "        if (errno == ENOENT) { result.is_ok = 1; result.value = 0; }",
                    "        else result.error = 4;",
                    "        return result;",
                    "    }",
                    "    result.is_ok = 1; result.value = 1; return result;",
                    "}",
                ]
            )
        return lines

    def _emit_entry_point(self) -> list[str]:
        if self.needs_host_io or self.needs_host_fs or self.needs_process_args:
            lines = [
                "int main(int argc, char **argv) {",
                "    int status;",
                "    if (!aether_runtime_init(argc, argv)) return 127;",
            ]
            if self.functions["main"].return_type.name == "i32":
                lines.append("    status = (int)aether_fn_6d61696e();")
            else:
                lines.extend(
                    [
                        "    aether_fn_6d61696e();",
                        "    status = 0;",
                    ]
                )
            if self.module_runtime_calls & {
                "io_read_line",
                "fs_read_text",
                "args_len",
                "args_get",
            }:
                lines.append("    aether_runtime_cleanup();")
            lines.extend(["    return status;", "}", ""])
            return lines
        return [
            "int main(void) {",
            "    " + (
                "return aether_fn_6d61696e();"
                if self.functions["main"].return_type.name == "i32"
                else "aether_fn_6d61696e();\n    return 0;"
            ),
            "}",
            "",
        ]

    def _emit_helpers(self) -> list[str]:
        lines: list[str] = []
        if self.needs_trap:
            lines.append(
                "static void aether_trap(void) { "
                + ("AETHER_TRAP();" if self.target_mcu else "abort();")
                + " }"
            )
        used_owned_collection_types: set[str] = set()

        def collect_owned_collection_type(type_: AetherType) -> None:
            if type_.name in {"List", "Slice"}:
                used_owned_collection_types.add(self._collection_c_name(type_))
            for argument in type_.arguments:
                collect_owned_collection_type(argument)

        def collect_expression_collection_types(expression: IRExpression) -> None:
            collect_owned_collection_type(expression.type)
            for child in expression.children:
                collect_expression_collection_types(child)
            for arm in expression.arms:
                collect_expression_collection_types(arm.value)

        for function in self.module.functions:
            if function.external_library is not None:
                continue
            collect_owned_collection_type(function.return_type)
            for parameter in function.parameters:
                collect_owned_collection_type(parameter.type)
            for statement in _walk_ir_statements(function.body):
                for expression in _statement_expressions(statement):
                    collect_expression_collection_types(expression)

        owned_collection_types = {
            self._collection_c_name(type_): type_
            for type_ in self.collection_types.values()
            if type_.name in {"List", "Slice"}
            and self._collection_c_name(type_) in used_owned_collection_types
        }
        if owned_collection_types:
            lines.extend(
                [
                    "static inline void aether_owned_slot_release(aether_owned_slot *slot) {",
                    "    if (slot->storage != NULL) {",
                    "        slot->release(slot->storage);",
                    "        slot->storage = NULL;",
                    "        slot->release = NULL;",
                    "    }",
                    "}",
                    "",
                ]
            )
            for name, type_ in sorted(owned_collection_types.items()):
                storage = name + "_storage"
                lines.extend(
                    [
                        f"static inline void aether_collection_release_{name}(void *raw) {{",
                        f"    {storage} *storage = ({storage} *)raw;",
                        "    if (storage == NULL || storage->references == 0) {",
                        "        aether_trap();",
                        "    }",
                        "    --storage->references;",
                        "    if (storage->references == 0) free(storage);",
                        "}",
                        "",
                    ]
                )
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
        bitwise_types = {
            type_name
            for type_name, operation in self.helpers
            if operation in {"band", "bor", "bxor", "bnot", "shl", "shr"}
        }
        for type_name in sorted(bitwise_types & _SIGNED_TYPES):
            width = type_name[1:]
            lines.extend(
                [
                    f"static { _C_TYPES[type_name] } aether_{type_name}_from_bits("
                    f"uint{width}_t bits) {{",
                    f"    if (bits <= (uint{width}_t)INT{width}_MAX)",
                    f"        return ({_C_TYPES[type_name]})bits;",
                    f"    return ({_C_TYPES[type_name]})("
                    f"-1 - ({_C_TYPES[type_name]})(UINT{width}_MAX - bits));",
                    "}",
                ]
            )
        for type_name, operation in sorted(self.helpers):
            ctype = _C_TYPES[type_name]
            fn = f"aether_{type_name}_{operation}"
            if operation in {"band", "bor", "bxor", "bnot", "shl", "shr"}:
                width = type_name[1:]
                utype = f"uint{width}_t"
                if operation == "bnot":
                    lines.extend(
                        [
                            f"static {ctype} {fn}({ctype} value) {{",
                            f"    {utype} bits = ({utype})~({utype})value;",
                            (
                                f"    return aether_{type_name}_from_bits(bits);"
                                if type_name in _SIGNED_TYPES
                                else f"    return ({ctype})bits;"
                            ),
                            "}",
                        ]
                    )
                    continue
                if operation in {"band", "bor", "bxor"}:
                    operator = {"band": "&", "bor": "|", "bxor": "^"}[operation]
                    lines.extend(
                        [
                            f"static {ctype} {fn}({ctype} left, {ctype} right) {{",
                            f"    {utype} bits = ({utype})left {operator} ({utype})right;",
                            (
                                f"    return aether_{type_name}_from_bits(bits);"
                                if type_name in _SIGNED_TYPES
                                else f"    return ({ctype})bits;"
                            ),
                            "}",
                        ]
                    )
                    continue
                signed_shift_guard = (
                    "    if (right < 0) aether_trap();"
                    if type_name in _SIGNED_TYPES
                    else ""
                )
                if operation == "shl":
                    lines.extend(
                        [
                            f"static {ctype} {fn}({ctype} left, {ctype} right) {{",
                            signed_shift_guard,
                            "    uint64_t count = (uint64_t)right;",
                            f"    {utype} bits;",
                            f"    if (count >= {width}u) aether_trap();",
                            f"    bits = ({utype})left << count;",
                            (
                                f"    return aether_{type_name}_from_bits(bits);"
                                if type_name in _SIGNED_TYPES
                                else f"    return ({ctype})bits;"
                            ),
                            "}",
                        ]
                    )
                    continue
                lines.extend(
                    [
                        f"static {ctype} {fn}({ctype} left, {ctype} right) {{",
                        signed_shift_guard,
                        "    uint64_t count = (uint64_t)right;",
                        f"    {utype} bits = ({utype})left;",
                        f"    if (count >= {width}u) aether_trap();",
                        "    bits >>= count;",
                    ]
                )
                if type_name in _SIGNED_TYPES:
                    lines.extend(
                        [
                            "    if (left < 0 && count != 0u)",
                            f"        bits |= ({utype})(UINT{width}_MAX << ({width}u - count));",
                            f"    return aether_{type_name}_from_bits(bits);",
                        ]
                    )
                else:
                    lines.append(f"    return ({ctype})bits;")
                lines.append("}")
                continue
            if operation == "abs" and type_name in _SIGNED_TYPES:
                width = type_name[1:]
                lines.extend(
                    [
                        f"static {ctype} {fn}({ctype} value) {{",
                        f"    if (value == INT{width}_MIN) aether_trap();",
                        f"    return value < 0 ? ({ctype})(-value) : value;",
                        "}",
                    ]
                )
                continue
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
    optimize: bool = True,
) -> str:
    """Emit deterministic C99 for the supported checked-scalar subset."""
    if optimize:
        module = fuse_elementwise_tensors(module)
    return _C99Emitter(
        module,
        include_entry_point=include_entry_point,
    ).emit()


def emit_c99_module_header(module: IRModule) -> str:
    """Emit a C header containing only functions marked with @C."""
    emitter = _C99Emitter(module)
    emitter.validate()
    if emitter.diagnostics:
        raise C99LoweringError(tuple(emitter.diagnostics))
    lines = [
        "#ifndef AETHER_GENERATED_H",
        "#define AETHER_GENERATED_H",
        "",
        "#include <stdbool.h>",
        "#include <stdint.h>",
        "",
    ]
    for function in module.functions:
        if function.c_exported:
            lines.append(emitter._c_export_signature(function) + ";")
    if any(function.c_exported for function in module.functions):
        lines.append("")
    lines.extend(("#endif", ""))
    return "\n".join(lines)


def emit_c99_module_units(
    module: IRModule,
    entry_source_path: str,
    *,
    optimize: bool = True,
) -> tuple[tuple[str, str], ...]:
    """Emit one C99 translation unit per source file in a project."""
    if optimize:
        module = fuse_elementwise_tensors(module)
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
