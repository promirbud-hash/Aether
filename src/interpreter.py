"""Reference interpreter for the typed Aether subset.

Effectful built-ins are inert unless callers explicitly provide handlers.
"""

from __future__ import annotations

import math
import re
import struct
import sys
from dataclasses import dataclass
from typing import Callable, Mapping

from src.frontend import Module
from src.ir import IRExpression, IRFunction, IRModule, lower_module
from src.semantic import AetherType, SemanticResult, UnitConversion


@dataclass(frozen=True)
class ResultValue:
    is_ok: bool
    value: object

    @classmethod
    def ok(cls, value: object) -> ResultValue:
        return cls(True, value)

    @classmethod
    def err(cls, error: object) -> ResultValue:
        return cls(False, error)


@dataclass(frozen=True)
class Quantity:
    value: object
    type: AetherType


@dataclass(frozen=True)
class _ADValue:
    values: tuple[float, ...]
    nodes: tuple[int, ...]
    is_tensor: bool


@dataclass
class _FrameStorage:
    data: bytearray
    refcount: int = 1


class _FrameBuffer:
    def __init__(
        self,
        data: bytearray,
        *,
        _storage: _FrameStorage | None = None,
    ) -> None:
        self._storage = _storage or _FrameStorage(data)
        self._released = False

    @property
    def data(self) -> bytearray:
        return self._storage.data

    @property
    def length(self) -> int:
        return len(self._storage.data)

    @property
    def refcount(self) -> int:
        return self._storage.refcount

    @property
    def released(self) -> bool:
        return self._released

    def retain(self) -> _FrameBuffer:
        if self._released or self._storage.refcount < 1:
            raise AetherRuntimeError("Cannot retain a released FrameBuffer")
        if self._storage.refcount >= sys.maxsize:
            raise AetherRuntimeError("FrameBuffer reference count overflow")
        self._storage.refcount += 1
        return _FrameBuffer(self.data, _storage=self._storage)

    def release(self) -> None:
        if self._released or self._storage.refcount < 1:
            raise AetherRuntimeError("FrameBuffer released more than once")
        self._released = True
        self._storage.refcount -= 1
        if self._storage.refcount == 0:
            self._storage.data.clear()


class AetherRuntimeError(RuntimeError):
    """A runtime failure with the source span of its Aether operation."""


class _PropagateResult(Exception):
    def __init__(self, result: ResultValue) -> None:
        self.result = result


EffectHandler = Callable[..., ResultValue]
ExternalFunctionHandler = Callable[..., object]


def _has_unit(type_: AetherType) -> bool:
    return bool(
        type_.unit
        or type_.unit_symbol
        or type_.unit_scale != 1
        or type_.unit_offset
        or type_.unit_pi_power
    )


def _shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    child_shapes = tuple(_shape(item) for item in value)
    if any(shape != child_shapes[0] for shape in child_shapes):
        raise ValueError("tensor data must be rectangular")
    return (len(value),) + child_shapes[0]


def _map_unary(value: object, operation: Callable[[object], object]) -> object:
    if isinstance(value, (list, tuple)):
        return [_map_unary(item, operation) for item in value]
    return operation(value)


def _map_binary(
    left: object,
    right: object,
    operation: Callable[[object, object], object],
) -> object:
    if isinstance(left, (list, tuple)):
        if isinstance(right, (list, tuple)):
            if len(left) != len(right):
                raise ValueError("tensor operands must have matching shapes")
            return [
                _map_binary(a, b, operation)
                for a, b in zip(left, right)
            ]
        return [_map_binary(item, right, operation) for item in left]
    if isinstance(right, (list, tuple)):
        return [_map_binary(left, item, operation) for item in right]
    return operation(left, right)


def _flatten(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _flatten(child)]
    return [value]


def _reshape_flat(values: list[float], shape: tuple[int, ...]) -> object:
    cursor = 0

    def build(axis: int) -> object:
        nonlocal cursor
        if axis == len(shape):
            value = values[cursor]
            cursor += 1
            return value
        return [build(axis + 1) for _ in range(shape[axis])]

    return build(0)


def _divide(left: object, right: object) -> object:
    if (
        isinstance(left, int)
        and not isinstance(left, bool)
        and isinstance(right, int)
        and not isinstance(right, bool)
    ):
        quotient = abs(left) // abs(right)
        return -quotient if (left < 0) != (right < 0) else quotient
    return left / right


def _remainder(left: object, right: object) -> object:
    if (
        isinstance(left, int)
        and not isinstance(left, bool)
        and isinstance(right, int)
        and not isinstance(right, bool)
    ):
        return left - _divide(left, right) * right
    return left % right


def _round_f32(value: object) -> float:
    try:
        numeric = float(value)
    except OverflowError:
        return -math.inf if value < 0 else math.inf
    try:
        return struct.unpack("=f", struct.pack("=f", numeric))[0]
    except OverflowError:
        return math.copysign(math.inf, numeric)


def _round_to_type(value: object, type_: AetherType | None) -> object:
    if type_ is None:
        return value
    if type_.name == "Tensor":
        element_type = type_.element_type
        if element_type is not None:
            return _map_unary(value, lambda item: _round_to_type(item, element_type))
    if type_.name == "f32":
        return _round_f32(value)
    return value


class Interpreter:
    """Execute a lowered module with explicit, injectable effect handlers."""

    _EFFECTFUL_BUILTINS = {
        "io_write", "io_read_line", "fs_read_text", "fs_write_text",
        "fs_exists", "net_get", "net_post",
    }

    def __init__(
        self,
        module: IRModule,
        *,
        effect_handlers: Mapping[str, EffectHandler] | None = None,
        external_handlers: Mapping[str, ExternalFunctionHandler] | None = None,
        max_call_depth: int = 128,
    ) -> None:
        if max_call_depth < 1:
            raise ValueError("max_call_depth must be positive")
        self.module = module
        self.effect_handlers = dict(effect_handlers or {})
        self.external_handlers = dict(external_handlers or {})
        self.max_call_depth = max_call_depth
        self._call_depth = 0
        self._frame_resources: list[list[object | None]] = []
        self._frame_resource_slots: list[dict[str, int]] = []
        self._temporary_resources: list[list[object]] = []
        self._functions = {function.name: function for function in module.functions}
        self._types = {type_.name: type_ for type_ in module.types}

    @classmethod
    def from_source(
        cls,
        source_module: Module,
        *,
        semantics: SemanticResult | None = None,
        effect_handlers: Mapping[str, EffectHandler] | None = None,
        external_handlers: Mapping[str, ExternalFunctionHandler] | None = None,
        max_call_depth: int = 128,
    ) -> Interpreter:
        return cls(
            lower_module(source_module, semantics),
            effect_handlers=effect_handlers,
            external_handlers=external_handlers,
            max_call_depth=max_call_depth,
        )

    def run(self, function_name: str, *arguments: object, **keywords: object) -> object:
        """Call an Aether function with Python values as its external inputs."""
        try:
            function = self._functions[function_name]
        except KeyError as error:
            raise KeyError(f"Unknown Aether function {function_name!r}") from error
        return self._call(function, arguments, keywords)

    def _call(
        self,
        function: IRFunction,
        arguments: tuple[object, ...],
        keywords: Mapping[str, object],
    ) -> object:
        if self._call_depth >= self.max_call_depth:
            raise AetherRuntimeError(
                f"Maximum call depth ({self.max_call_depth}) exceeded in "
                f"{function.name!r}"
            )
        if len(arguments) > len(function.parameters):
            raise TypeError(
                f"{function.name} expects {len(function.parameters)} arguments, "
                f"got {len(arguments)}"
            )
        parameters = {parameter.name: parameter for parameter in function.parameters}
        environment: dict[str, object] = {}
        for parameter, value in zip(function.parameters, arguments):
            environment[parameter.name] = self._prepare_input(
                value, parameter.type, function.name, parameter.name
            )
        for name, value in keywords.items():
            if name not in parameters:
                raise TypeError(f"{function.name} has no parameter {name!r}")
            if name in environment:
                raise TypeError(f"Multiple values for parameter {name!r}")
            environment[name] = self._prepare_input(
                value, parameters[name].type, function.name, name
            )
        missing = [parameter.name for parameter in function.parameters if parameter.name not in environment]
        if missing:
            raise TypeError(
                f"{function.name} is missing argument(s): {', '.join(missing)}"
            )
        if function.external_abi is not None:
            handler = self.external_handlers.get(function.name)
            if handler is None:
                raise AetherRuntimeError(
                    f"Extern function {function.name!r} from library "
                    f"{function.external_library!r} has no injected handler; "
                    "the reference interpreter does not load native libraries"
                )
            result = handler(
                *(environment[parameter.name] for parameter in function.parameters)
            )
            return self._prepare_output(
                result,
                function.return_type,
                function.name,
            )

        self._call_depth += 1
        frame_resources: list[object | None] = [
            None for _ in function.frame_resources
        ]
        self._frame_resources.append(frame_resources)
        self._frame_resource_slots.append(
            {
                name: index
                for index, name in enumerate(function.frame_resources)
            }
        )
        try:
            try:
                result, returned = self._execute_body(function, environment)
            except _PropagateResult as propagation:
                return propagation.result
            if returned:
                if function.return_type.name == "Unknown":
                    return result
                if function.return_type.name == "Result":
                    if not isinstance(result, ResultValue):
                        result = ResultValue.ok(result)
                    return self._prepare_output(
                        result,
                        function.return_type,
                        function.name,
                    )
                return self._prepare_output(result, function.return_type, function.name)
            if (
                function.has_return_annotation
                and function.return_type.name not in {"void", "Unknown"}
            ):
                raise AetherRuntimeError(
                    f"Function {function.name!r} completed without returning a value"
                )
            if function.return_type.name == "Result":
                return self._prepare_output(
                    ResultValue.ok(None),
                    function.return_type,
                    function.name,
                )
            return None
        finally:
            for resource in reversed(frame_resources):
                if resource is not None:
                    self._release_owned_value(resource)
            if self._frame_resources[-1] is not frame_resources:
                raise AetherRuntimeError("Frame resource stack is corrupted")
            self._frame_resources.pop()
            self._frame_resource_slots.pop()
            self._call_depth -= 1

    @staticmethod
    def _contains_frame_buffer(type_: AetherType) -> bool:
        return type_.name == "FrameBuffer" or any(
            Interpreter._contains_frame_buffer(argument)
            for argument in type_.arguments
        )

    @staticmethod
    def _clone_owned_value(value: object) -> object:
        if isinstance(value, _FrameBuffer):
            return value.retain()
        if isinstance(value, ResultValue) and value.is_ok:
            payload = Interpreter._clone_owned_value(value.value)
            if payload is not value.value:
                return ResultValue.ok(payload)
        return value

    @staticmethod
    def _release_owned_value(value: object) -> None:
        if isinstance(value, _FrameBuffer):
            value.release()
        elif isinstance(value, ResultValue) and value.is_ok:
            Interpreter._release_owned_value(value.value)

    def _track_temporary(self, value: object) -> object:
        if self._temporary_resources and self._owns_frame_buffer(value):
            self._temporary_resources[-1].append(value)
        return value

    @staticmethod
    def _owns_frame_buffer(value: object) -> bool:
        return isinstance(value, _FrameBuffer) or (
            isinstance(value, ResultValue)
            and value.is_ok
            and isinstance(value.value, _FrameBuffer)
        )

    def _evaluate_statement_expression(
        self,
        expression: IRExpression,
        environment: dict[str, object],
        *,
        transfer: bool = False,
    ) -> object:
        temporaries: list[object] = []
        self._temporary_resources.append(temporaries)
        try:
            value = self._evaluate(expression, environment)
            if transfer and self._contains_frame_buffer(expression.type):
                if expression.ownership == "borrowed":
                    value = self._clone_owned_value(value)
                else:
                    for index in range(len(temporaries) - 1, -1, -1):
                        if temporaries[index] is value:
                            temporaries.pop(index)
                            break
            return value
        finally:
            if self._temporary_resources[-1] is not temporaries:
                raise AetherRuntimeError("Temporary resource stack is corrupted")
            for temporary in reversed(temporaries):
                self._release_owned_value(temporary)
            self._temporary_resources.pop()

    def _prepare_output(
        self,
        value: object,
        type_: AetherType,
        function: str,
    ) -> object:
        try:
            return self._prepare_input(value, type_, function, "return value")
        except TypeError as error:
            raise AetherRuntimeError(
                f"Function {function!r} returned a value that violates "
                f"{type_.display()}: {error}"
            ) from error

    def _prepare_input(
        self,
        value: object,
        type_: AetherType,
        function: str,
        parameter: str,
    ) -> object:
        if type_.name == "Result":
            return self._prepare_result(value, type_, function, parameter)
        if type_.name == "FrameBuffer":
            if not isinstance(value, _FrameBuffer) or value.released:
                raise TypeError(f"{function}.{parameter} expects a live FrameBuffer")
            return value
        if type_.name == "void":
            if value is not None:
                raise TypeError(f"{function}.{parameter} expects no value")
            return None
        declaration = self._types.get(type_.name)
        if declaration is not None:
            if not isinstance(value, Mapping):
                raise TypeError(f"{function}.{parameter} expects {type_.name} data")
            field_types = dict(declaration.fields)
            if any(not isinstance(name, str) for name in value):
                raise TypeError(
                    f"{function}.{parameter} field names must be strings"
                )
            if set(value) != set(field_types):
                missing = sorted(set(field_types) - set(value))
                extra = sorted(set(value) - set(field_types))
                details = []
                if missing:
                    details.append("missing " + ", ".join(missing))
                if extra:
                    details.append("unexpected " + ", ".join(extra))
                raise TypeError(
                    f"{function}.{parameter} has invalid {type_.name} fields: "
                    + "; ".join(details)
                )
            return {
                name: self._prepare_input(
                    value[name],
                    field_type,
                    f"{function}.{parameter}",
                    name,
                )
                for name, field_type in field_types.items()
            }
        raw = value.value if isinstance(value, Quantity) else value
        if type_.name == "Tensor":
            if isinstance(value, Quantity) and not _has_unit(type_):
                raise TypeError(
                    f"{function}.{parameter} does not accept a unitful tensor"
                )
            try:
                actual_shape = _shape(raw)
            except ValueError as error:
                raise TypeError(f"{function}.{parameter}: {error}") from error
            if actual_shape != type_.dimensions:
                raise TypeError(
                    f"{function}.{parameter} expects tensor shape "
                    f"{type_.dimensions}, got {actual_shape}"
                )
            if type_.arguments:
                element_type = type_.arguments[0]
                for index, item in enumerate(_flatten(raw)):
                    self._validate_scalar(
                        item,
                        element_type,
                        f"{function}.{parameter}[{index}]",
                    )
            raw = _round_to_type(raw, type_)
            if _has_unit(type_):
                if isinstance(value, Quantity) and value.type != type_:
                    raise TypeError(
                        f"{function}.{parameter} expects unit {type_.unit_symbol or type_.unit}, "
                        f"got {value.type.unit_symbol or value.type.unit}"
                    )
                return value if isinstance(value, Quantity) else Quantity(raw, type_)
            return raw
        if _has_unit(type_):
            if isinstance(value, Quantity):
                if value.type != type_:
                    raise TypeError(
                        f"{function}.{parameter} expects unit {type_.unit_symbol or type_.unit}, "
                        f"got {value.type.unit_symbol or value.type.unit}"
                    )
                return Quantity(_round_to_type(raw, type_), type_)
            self._validate_scalar(raw, type_, f"{function}.{parameter}")
            return Quantity(_round_to_type(raw, type_), type_)
        if isinstance(value, Quantity):
            raise TypeError(f"{function}.{parameter} does not accept a unitful value")
        self._validate_scalar(raw, type_, f"{function}.{parameter}")
        return _round_to_type(raw, type_)

    def _prepare_result(
        self,
        value: object,
        type_: AetherType,
        function: str,
        parameter: str,
    ) -> ResultValue:
        if not isinstance(value, ResultValue):
            raise TypeError(f"{function}.{parameter} expects a ResultValue")
        if not isinstance(value.is_ok, bool):
            raise TypeError(f"{function}.{parameter} has an invalid Result variant")
        if len(type_.arguments) != 2:
            raise TypeError(f"{function}.{parameter} has an invalid Result type")
        if not value.is_ok:
            return value
        payload_type = type_.arguments[0]
        payload = self._prepare_input(
            value.value,
            payload_type,
            function,
            f"{parameter} Ok payload",
        )
        return ResultValue.ok(payload)

    @staticmethod
    def _validate_scalar(value: object, type_: AetherType, location: str) -> None:
        if type_.name == "bool" and not isinstance(value, bool):
            raise TypeError(f"{location} expects bool")
        if type_.name == "str" and not isinstance(value, str):
            raise TypeError(f"{location} expects str")
        if type_.is_numeric:
            if type_.name.startswith(("i", "u")) or type_.name == "usize":
                valid = isinstance(value, int) and not isinstance(value, bool)
                if valid and type_.name != "usize":
                    bits = int(type_.name[1:])
                    if type_.name.startswith("i"):
                        valid = -(2 ** (bits - 1)) <= value < 2 ** (bits - 1)
                    else:
                        valid = 0 <= value < 2**bits
            else:
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not valid:
                raise TypeError(f"{location} expects {type_.name}")

    def _execute_body(
        self,
        function: IRFunction,
        environment: dict[str, object],
    ) -> tuple[object, bool]:
        for statement in function.body:
            if statement.kind == "let":
                assert statement.expression is not None
                value = self._evaluate_statement_expression(
                    statement.expression,
                    environment,
                    transfer=True,
                )
                environment[statement.name] = value
                if self._contains_frame_buffer(statement.expression.type):
                    slot = self._frame_resource_slots[-1].get(statement.name)
                    if slot is None:
                        raise AetherRuntimeError(
                            f"Missing ownership slot for {statement.name!r}"
                        )
                    self._frame_resources[-1][slot] = value
            elif statement.kind == "assignment":
                assert statement.expression is not None
                value = self._evaluate_statement_expression(
                    statement.expression,
                    environment,
                    transfer=True,
                )
                slot = self._frame_resource_slots[-1].get(statement.name)
                if slot is not None:
                    old_value = self._frame_resources[-1][slot]
                    if old_value is not None:
                        self._release_owned_value(old_value)
                    self._frame_resources[-1][slot] = value
                environment[statement.name] = value
            elif statement.kind == "expression":
                assert statement.expression is not None
                value = self._evaluate_statement_expression(
                    statement.expression,
                    environment,
                )
                if isinstance(value, ResultValue):
                    raise AetherRuntimeError(
                        f"Discarded Result value at line {statement.span.line}"
                    )
            elif statement.kind == "return":
                value = (
                    self._evaluate_statement_expression(
                        statement.expression,
                        environment,
                        transfer=True,
                    )
                    if statement.expression is not None
                    else None
                )
                return value, True
        return None, False

    def _evaluate(
        self,
        expression: IRExpression,
        environment: dict[str, object],
    ) -> object:
        kind = expression.kind
        if kind == "literal":
            return _round_to_type(self._literal(expression.value), expression.type)
        if kind == "identifier":
            error_codes = {"IoError": 1, "FsError": 2, "NetError": 3}
            if expression.value in error_codes:
                return error_codes[expression.value]
            try:
                return environment[expression.value]
            except KeyError as error:
                raise AetherRuntimeError(
                    f"Unknown value {expression.value!r} at line {expression.span.line}"
                ) from error
        if kind == "group":
            return self._evaluate(expression.children[0], environment)
        if kind == "unary":
            value = self._evaluate(expression.children[0], environment)
            raw = value.value if isinstance(value, Quantity) else value
            if expression.value == "!":
                return not raw
            result = _map_unary(raw, (lambda item: +item) if expression.value == "+" else (lambda item: -item))
            return self._wrap(result, expression.type)
        if kind == "try":
            value = self._evaluate(expression.children[0], environment)
            if not isinstance(value, ResultValue):
                raise AetherRuntimeError(
                    f"'?' expected Result at line {expression.span.line}"
                )
            if not value.is_ok:
                raise _PropagateResult(self._clone_owned_value(value))
            return self._track_temporary(
                self._clone_owned_value(value.value)
            )
        if kind == "match":
            subject = self._evaluate(expression.children[0], environment)
            if not isinstance(subject, ResultValue):
                raise AetherRuntimeError(
                    f"Result match expected Result at line {expression.span.line}"
                )
            variant = "Ok" if subject.is_ok else "Err"
            arm = next((candidate for candidate in expression.arms if candidate.variant == variant), None)
            if arm is None:
                raise AetherRuntimeError(
                    f"Missing {variant} match arm at line {expression.span.line}"
                )
            if arm.binding is None:
                return self._evaluate(arm.value, environment)
            scoped = dict(environment)
            scoped[arm.binding] = subject.value
            return self._evaluate(arm.value, scoped)
        if kind == "call":
            return self._evaluate_call(expression, environment)
        if kind == "gradient":
            input_value = self._evaluate(expression.children[0], environment)
            return self._evaluate_gradient(expression, input_value)
        if kind == "binary":
            if expression.value == "|>":
                left = self._evaluate(expression.children[0], environment)
                return self._evaluate_pipeline_right(
                    expression.children[1], left, environment
                )
            if expression.value == "&&":
                left = self._evaluate(expression.children[0], environment)
                return bool(left) and bool(self._evaluate(expression.children[1], environment))
            if expression.value == "||":
                left = self._evaluate(expression.children[0], environment)
                return bool(left) or bool(self._evaluate(expression.children[1], environment))
            left = self._evaluate(expression.children[0], environment)
            right = self._evaluate(expression.children[1], environment)
            return self._binary(expression.value, left, right, expression)
        if kind == "member":
            base = self._evaluate(expression.children[0], environment)
            member = expression.children[1].value
            raw = base.value if isinstance(base, Quantity) else base
            if isinstance(raw, Mapping):
                try:
                    return raw[member]
                except KeyError as error:
                    raise AetherRuntimeError(
                        f"Unknown member {member!r} at line {expression.span.line}"
                    ) from error
            if hasattr(raw, member):
                return getattr(raw, member)
            raise AetherRuntimeError(
                f"Unknown member {member!r} at line {expression.span.line}"
            )
        if kind == "index":
            base = self._evaluate(expression.children[0], environment)
            index = self._evaluate(expression.children[1], environment)
            raw = base.value if isinstance(base, Quantity) else base
            if isinstance(index, int) and not isinstance(index, bool) and index < 0:
                raise AetherRuntimeError(
                    f"Negative index at line {expression.span.line}"
                )
            try:
                value = raw[index]
            except (IndexError, KeyError, TypeError) as error:
                raise AetherRuntimeError(
                    f"Invalid index at line {expression.span.line}: {error}"
                ) from error
            return self._wrap(value, expression.type)
        raise AetherRuntimeError(
            f"Unsupported IR expression {kind!r} at line {expression.span.line}"
        )

    def _evaluate_gradient(
        self,
        expression: IRExpression,
        input_value: object,
    ) -> object:
        target = self._functions.get(expression.value)
        if target is None or len(target.parameters) != 1:
            raise AetherRuntimeError(
                f"Invalid autodiff target {expression.value!r} at line "
                f"{expression.span.line}"
            )
        try:
            prepared = self._prepare_input(
                input_value,
                target.parameters[0].type,
                "gradient",
                target.parameters[0].name,
            )
        except TypeError as error:
            raise AetherRuntimeError(
                f"Invalid autodiff input at line {expression.span.line}: {error}"
            ) from error
        if target.parameters[0].type.name == "Tensor":
            return self._evaluate_tensor_gradient(expression, target, prepared)
        if not isinstance(prepared, (int, float)) or isinstance(prepared, bool):
            raise AetherRuntimeError(
                f"Autodiff v0 requires a scalar input at line {expression.span.line}"
            )

        tape: list[tuple[float, tuple[tuple[int, float], ...]]] = [
            (float(prepared), ())
        ]
        environment: dict[str, _ADValue] = {
            target.parameters[0].name: _ADValue((float(prepared),), (0,), False)
        }
        returned: _ADValue | None = None
        for statement in target.body:
            if statement.kind == "let" and statement.expression is not None:
                environment[statement.name] = self._trace_ad_expression(
                    statement.expression,
                    environment,
                    tape,
                )
            elif statement.kind == "return" and statement.expression is not None:
                returned = self._trace_ad_expression(
                    statement.expression,
                    environment,
                    tape,
                )
                break
            else:
                raise AetherRuntimeError(
                    f"Unsupported autodiff statement at line {statement.span.line}"
                )
        if returned is None:
            raise AetherRuntimeError(
                f"Autodiff target did not return at line {target.span.line}"
            )

        adjoints = [0.0] * len(tape)
        if returned.is_tensor or len(returned.nodes) != 1:
            raise AetherRuntimeError(
                f"Scalar autodiff target returned a tensor at line {target.span.line}"
            )
        adjoints[returned.nodes[0]] = 1.0
        for node_index in range(len(tape) - 1, -1, -1):
            for parent, partial in tape[node_index][1]:
                adjoints[parent] += adjoints[node_index] * partial
        return self._wrap(adjoints[0], expression.type)

    def _evaluate_tensor_gradient(
        self,
        expression: IRExpression,
        target: IRFunction,
        prepared: object,
    ) -> object:
        parameter_type = target.parameters[0].type
        raw_input = prepared.value if isinstance(prepared, Quantity) else prepared
        input_values = tuple(float(value) for value in _flatten(raw_input))
        element_type = parameter_type.element_type
        if element_type is not None:
            input_values = tuple(
                _round_to_type(value, element_type) for value in input_values
            )
        tape: list[tuple[float, tuple[tuple[int, float], ...]]] = [
            (value, ()) for value in input_values
        ]
        environment: dict[str, _ADValue] = {
            target.parameters[0].name: _ADValue(
                input_values,
                tuple(range(len(input_values))),
                True,
            )
        }
        returned: _ADValue | None = None
        for statement in target.body:
            if statement.kind == "let" and statement.expression is not None:
                environment[statement.name] = self._trace_ad_expression(
                    statement.expression,
                    environment,
                    tape,
                )
            elif statement.kind == "return" and statement.expression is not None:
                returned = self._trace_ad_expression(
                    statement.expression,
                    environment,
                    tape,
                )
                break
            else:
                raise AetherRuntimeError(
                    f"Unsupported autodiff statement at line {statement.span.line}"
                )
        if returned is None or returned.is_tensor:
            raise AetherRuntimeError(
                f"Autodiff target did not return a scalar at line {target.span.line}"
            )

        adjoints = [0.0] * len(tape)
        adjoints[returned.nodes[0]] = 1.0
        for node_index in range(len(tape) - 1, -1, -1):
            for parent, partial in tape[node_index][1]:
                adjoints[parent] += adjoints[node_index] * partial
        gradient = adjoints[:len(input_values)]
        result = _reshape_flat(gradient, parameter_type.dimensions)
        return self._wrap(result, expression.type)

    def _trace_ad_expression(
        self,
        expression: IRExpression,
        environment: dict[str, _ADValue],
        tape: list[tuple[float, tuple[tuple[int, float], ...]]],
    ) -> _ADValue:
        kind = expression.kind
        if kind == "literal":
            value = _round_to_type(
                float(self._literal(expression.value)),
                expression.type,
            )
            tape.append((value, ()))
            return _ADValue((value,), (len(tape) - 1,), False)
        if kind == "identifier":
            try:
                return environment[expression.value]
            except KeyError as error:
                raise AetherRuntimeError(
                    f"Unknown autodiff value {expression.value!r} "
                    f"at line {expression.span.line}"
                ) from error
        if kind == "group":
            return self._trace_ad_expression(
                expression.children[0], environment, tape
            )
        if kind == "unary":
            operand = self._trace_ad_expression(
                expression.children[0], environment, tape
            )
            if expression.value == "+":
                return operand
            if expression.value == "-":
                values: list[float] = []
                nodes: list[int] = []
                element_type = (
                    expression.type.element_type
                    if expression.type.name == "Tensor"
                    else expression.type
                )
                for value, parent in zip(operand.values, operand.nodes):
                    result = _round_to_type(-value, element_type)
                    partial = _round_to_type(-1.0, element_type)
                    tape.append((result, ((parent, partial),)))
                    values.append(result)
                    nodes.append(len(tape) - 1)
                return _ADValue(tuple(values), tuple(nodes), operand.is_tensor)
        if kind == "binary":
            left = self._trace_ad_expression(
                expression.children[0], environment, tape
            )
            right = self._trace_ad_expression(
                expression.children[1], environment, tape
            )
            is_tensor = expression.type.name == "Tensor"
            count = len(left.values) if left.is_tensor else len(right.values)
            if left.is_tensor and right.is_tensor and len(left.values) != len(right.values):
                raise AetherRuntimeError(
                    f"Autodiff tensor shape mismatch at line {expression.span.line}"
                )
            left_values = left.values if left.is_tensor else left.values * count
            right_values = right.values if right.is_tensor else right.values * count
            left_nodes = left.nodes if left.is_tensor else left.nodes * count
            right_nodes = right.nodes if right.is_tensor else right.nodes * count
            values = []
            nodes = []
            element_type = (
                expression.type.element_type
                if expression.type.name == "Tensor"
                else expression.type
            )
            for a, b, parent_a, parent_b in zip(
                left_values, right_values, left_nodes, right_nodes
            ):
                if expression.value == "+":
                    value, parents = a + b, ((parent_a, 1.0), (parent_b, 1.0))
                elif expression.value == "-":
                    value, parents = a - b, ((parent_a, 1.0), (parent_b, -1.0))
                elif expression.value == "*":
                    value, parents = a * b, ((parent_a, b), (parent_b, a))
                elif expression.value == "/":
                    if b == 0.0:
                        raise AetherRuntimeError(
                            f"Division by zero at line {expression.span.line}"
                        )
                    value = a / b
                    parents = (
                        (parent_a, 1.0 / b),
                        (parent_b, -a / (b * b)),
                    )
                else:
                    raise AetherRuntimeError(
                        f"Unsupported autodiff operator at line {expression.span.line}"
                    )
                value = _round_to_type(value, element_type)
                parents = tuple(
                    (parent, _round_to_type(partial, element_type))
                    for parent, partial in parents
                )
                tape.append((value, parents))
                values.append(value)
                nodes.append(len(tape) - 1)
            return _ADValue(tuple(values), tuple(nodes), is_tensor)
        if kind == "call":
            callee = expression.children[0]
            if callee.kind != "identifier":
                raise AetherRuntimeError(
                    f"Unsupported autodiff call at line {expression.span.line}"
                )
            if callee.value == "convert" and len(expression.children) == 3:
                operand = self._trace_ad_expression(
                    expression.children[1], environment, tape
                )
                conversion = expression.conversion
                if (
                    conversion is None
                    or conversion.addend
                    or conversion.pi_power
                ):
                    raise AetherRuntimeError(
                        f"Nonlinear unit conversion in autodiff at line "
                        f"{expression.span.line}"
                    )
                factor = float(conversion.multiplier)
                element_type = (
                    expression.type.element_type
                    if expression.type.name == "Tensor"
                    else expression.type
                )
                values = tuple(
                    _round_to_type(
                        value * factor + float(conversion.addend),
                        element_type,
                    )
                    for value in operand.values
                )
                factor = _round_to_type(factor, element_type)
                nodes = []
                for value, parent in zip(values, operand.nodes):
                    tape.append((value, ((parent, factor),)))
                    nodes.append(len(tape) - 1)
                return _ADValue(
                    values,
                    tuple(nodes),
                    expression.type.name == "Tensor",
                )
            if callee.value not in {"sum", "mean"} or len(expression.children) != 2:
                raise AetherRuntimeError(
                    f"Unsupported autodiff call at line {expression.span.line}"
                )
            operand = self._trace_ad_expression(
                expression.children[1], environment, tape
            )
            if not operand.is_tensor or not operand.values:
                raise AetherRuntimeError(
                    f"Autodiff reduction requires a non-empty tensor "
                    f"at line {expression.span.line}"
                )
            element_type = expression.type
            value = 0.0
            for item in operand.values:
                value = _round_to_type(value + item, element_type)
            factor = 1.0
            if callee.value == "mean":
                factor = 1.0 / len(operand.values)
                value = _round_to_type(value / len(operand.values), element_type)
                factor = _round_to_type(factor, element_type)
            parents = tuple((node, factor) for node in operand.nodes)
            tape.append((value, parents))
            return _ADValue((value,), (len(tape) - 1,), False)
        raise AetherRuntimeError(
            f"Unsupported autodiff expression {kind!r} at line {expression.span.line}"
        )

    def _evaluate_pipeline_right(
        self,
        right: IRExpression,
        left: object,
        environment: dict[str, object],
    ) -> object:
        if right.kind == "call":
            callee = right.children[0]
            if callee.kind != "identifier":
                raise AetherRuntimeError(
                    f"Unsupported pipeline callee at line {right.span.line}"
                )
            arguments = [left]
            arguments.extend(
                self._evaluate(argument, environment)
                for argument in right.children[1:]
            )
            return self._invoke(callee.value, arguments, right)
        if right.kind == "identifier":
            return self._invoke(right.value, [left], right)
        return self._evaluate(right, environment)

    def _evaluate_call(
        self,
        expression: IRExpression,
        environment: dict[str, object],
    ) -> object:
        callee = expression.children[0]
        if callee.kind != "identifier":
            raise AetherRuntimeError(
                f"Unsupported call target at line {expression.span.line}"
            )
        arguments = [
            self._evaluate(argument, environment)
            for argument in expression.children[1:]
        ]
        return self._invoke(callee.value, arguments, expression)

    def _invoke(
        self,
        name: str,
        arguments: list[object],
        expression: IRExpression,
    ) -> object:
        if name == "frame_buffer" and len(arguments) == 1:
            size = arguments[0]
            if not isinstance(size, int) or isinstance(size, bool):
                raise AetherRuntimeError("FrameBuffer size must be an integer")
            if size <= 0 or size > sys.maxsize:
                raise AetherRuntimeError(
                    "FrameBuffer size is zero or exceeds the runtime allocation limit"
                )
            try:
                resource = _FrameBuffer(bytearray(size))
            except (MemoryError, OverflowError) as error:
                raise AetherRuntimeError("FrameBuffer allocation failed") from error
            return self._track_temporary(resource)
        if name == "frame_buffer_len" and len(arguments) == 1:
            resource = arguments[0]
            if not isinstance(resource, _FrameBuffer) or resource.released:
                raise AetherRuntimeError("Invalid or released FrameBuffer")
            return resource.length
        if name == "Ok" and len(arguments) == 1:
            return self._track_temporary(
                ResultValue.ok(self._clone_owned_value(arguments[0]))
            )
        if name == "Err" and len(arguments) == 1:
            return ResultValue.err(arguments[0])
        if name in self._functions:
            result = self._call(self._functions[name], tuple(arguments), {})
            return (
                self._track_temporary(result)
                if self._contains_frame_buffer(expression.type)
                else result
            )
        if name in self._EFFECTFUL_BUILTINS:
            if expression.type.name != "Result" or len(expression.type.arguments) != 2:
                raise AetherRuntimeError(
                    f"Effect operation {name!r} has no typed Result IR"
                )
            handler = self.effect_handlers.get(name)
            if handler is None:
                raise AetherRuntimeError(
                    f"Effectful operation {name!r} has no injected handler "
                    f"(line {expression.span.line})"
                )
            result = handler(*arguments)
            if not isinstance(result, ResultValue):
                raise AetherRuntimeError(
                    f"Effect handler {name!r} must return ResultValue"
                )
            if not result.is_ok:
                return result
            success_type = expression.type.arguments[0]
            if success_type.name == "void":
                if result.value is not None:
                    raise AetherRuntimeError(
                        f"Effect handler {name!r} must return Ok(None)"
                    )
                return result
            try:
                value = self._prepare_input(
                    result.value,
                    success_type,
                    f"effect handler {name}",
                    "success value",
                )
            except TypeError as error:
                raise AetherRuntimeError(
                    f"Effect handler {name!r} returned an invalid success value: {error}"
                ) from error
            return ResultValue.ok(value)
        return self._pure_builtin(name, arguments, expression)

    def _pure_builtin(
        self,
        name: str,
        arguments: list[object],
        expression: IRExpression,
    ) -> object:
        raw = [value.value if isinstance(value, Quantity) else value for value in arguments]
        if name in {"mean", "sum", "max_abs"} and len(raw) == 1:
            items = _flatten(raw[0])
            if not items:
                raise AetherRuntimeError(
                    f"{name} cannot operate on an empty tensor at line {expression.span.line}"
                )
            if name == "sum":
                if expression.type.name == "f32":
                    result = 0.0
                    for item in items:
                        result = _round_f32(result + item)
                else:
                    result = sum(items)
            elif name == "mean":
                if expression.type.name == "f32":
                    result = 0.0
                    for item in items:
                        result = _round_f32(result + item)
                    result = _round_f32(result / len(items))
                else:
                    result = sum(items) / len(items)
            else:
                result = max(abs(item) for item in items)
            return self._wrap(result, expression.type)
        if name == "abs" and len(raw) == 1:
            result = _map_unary(raw[0], abs)
            return self._wrap(result, expression.type)
        if name == "len" and len(raw) == 1:
            try:
                return len(raw[0])
            except TypeError as error:
                raise AetherRuntimeError(
                    f"len expects tensor data at line {expression.span.line}"
                ) from error
        if name in {"scale", "subtract"} and len(arguments) == 2:
            return self._binary(
                "*" if name == "scale" else "-",
                arguments[0],
                arguments[1],
                expression,
            )
        if name in {"min", "max"} and len(arguments) == 2:
            left = arguments[0].value if isinstance(arguments[0], Quantity) else arguments[0]
            right = arguments[1].value if isinstance(arguments[1], Quantity) else arguments[1]
            operation = min if name == "min" else max
            try:
                result = _map_binary(left, right, operation)
            except (TypeError, ValueError) as error:
                raise AetherRuntimeError(
                    f"Invalid {name} operation at line {expression.span.line}: {error}"
                ) from error
            return self._wrap(result, expression.type)
        if name == "convert" and len(arguments) == 2:
            if expression.conversion is None:
                raise AetherRuntimeError(
                    f"Missing conversion metadata at line {expression.span.line}"
                )
            return self._convert(arguments[0], expression.conversion, expression.type)
        raise AetherRuntimeError(
            f"Unsupported operation {name!r} at line {expression.span.line}"
        )

    def _binary(
        self,
        operator: str,
        left: object,
        right: object,
        expression: IRExpression,
    ) -> object:
        left_raw = left.value if isinstance(left, Quantity) else left
        right_raw = right.value if isinstance(right, Quantity) else right
        operation: dict[str, Callable[[object, object], object]] = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": _divide,
            "%": _remainder,
            "**": lambda a, b: a**b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            "<": lambda a, b: a < b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            ">=": lambda a, b: a >= b,
        }
        try:
            raw_result = _map_binary(left_raw, right_raw, operation[operator])
        except ZeroDivisionError as error:
            raise AetherRuntimeError(
                f"Division by zero at line {expression.span.line}"
            ) from error
        except (TypeError, ValueError, OverflowError) as error:
            raise AetherRuntimeError(
                f"Invalid {operator} operation at line {expression.span.line}: {error}"
            ) from error
        if expression.type.name == "bool":
            return raw_result
        return self._wrap(raw_result, expression.type)

    def _convert(
        self,
        value: object,
        conversion: UnitConversion,
        target_type: AetherType,
    ) -> object:
        raw = value.value if isinstance(value, Quantity) else value
        multiplier = float(conversion.multiplier)
        multiplier *= math.pi ** conversion.pi_power
        addend = float(conversion.addend)
        converted = _map_unary(
            raw,
            lambda item: item * multiplier + addend,
        )
        return self._wrap(converted, target_type)

    @staticmethod
    def _wrap(value: object, type_: AetherType) -> object:
        value = _round_to_type(value, type_)
        return Quantity(value, type_) if _has_unit(type_) else value

    @staticmethod
    def _literal(value: str) -> object:
        if value == "true":
            return True
        if value == "false":
            return False
        if value == "none":
            return None
        if value.startswith(("'", '"')):
            quote = value[0]
            content = value[1:-1]
            if quote == '"':
                import json

                return json.loads(value)
            return re.sub(
                r"\\(.)",
                lambda match: {
                    "n": "\n", "r": "\r", "t": "\t", "\\": "\\", "'": "'",
                }.get(match.group(1), match.group(1)),
                content,
            )
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError as error:
                raise AetherRuntimeError(f"Invalid numeric literal {value!r}") from error


def run_module(
    source_module: Module,
    function_name: str,
    arguments: tuple[object, ...] = (),
    semantics: SemanticResult | None = None,
    effect_handlers: Mapping[str, EffectHandler] | None = None,
    named_arguments: Mapping[str, object] | None = None,
) -> object:
    """Lower and execute one function from a frontend Module."""
    interpreter = Interpreter.from_source(
        source_module,
        semantics=semantics,
        effect_handlers=effect_handlers,
    )
    return interpreter.run(
        function_name,
        *arguments,
        **dict(named_arguments or {}),
    )
