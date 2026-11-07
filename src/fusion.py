"""Conservative, deterministic tensor expression fusion for typed IR."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from src.ir import IRExpression, IRFunction, IRModule, IRStatement


def fuse_elementwise_tensors(module: IRModule) -> IRModule:
    """Fuse eligible single-use tensor temporaries into one elementwise node."""
    gradient_targets = {
        expression.value
        for function in module.functions
        for statement in function.body
        if statement.expression is not None
        for expression in _walk(statement.expression)
        if expression.kind == "gradient"
    }
    functions = tuple(
        _fuse_function(function)
        if function.name not in gradient_targets
        else function
        for function in module.functions
    )
    if functions == module.functions:
        return module
    return replace(module, functions=functions)


def _walk(expression: IRExpression):
    yield expression
    for child in expression.children:
        yield from _walk(child)
    for arm in expression.arms:
        yield from _walk(arm.value)


def _fuse_function(function: IRFunction) -> IRFunction:
    if (
        function.external_abi is not None
        or function.required_effects
        or not function.body
        or function.body[-1].kind != "return"
        or function.body[-1].expression is None
        or function.body[-1].expression.type.name != "Tensor"
        or any(statement.kind != "let" for statement in function.body[:-1])
    ):
        return function

    bindings: dict[str, IRExpression] = {}
    names: set[str] = set()
    for statement in function.body[:-1]:
        if statement.name in names:
            return function
        names.add(statement.name)
        expression = statement.expression
        if (
            expression is not None
            and expression.type.name == "Tensor"
            and _is_elementwise(expression, expression.type)
        ):
            bindings[statement.name] = expression

    counts = Counter(
        expression.value
        for statement in function.body
        if statement.expression is not None
        for expression in _walk(statement.expression)
        if expression.kind == "identifier"
    )
    consumed: set[str] = set()

    def expand(expression: IRExpression, active: frozenset[str] = frozenset()) -> IRExpression:
        if (
            expression.kind == "identifier"
            and expression.value in bindings
            and counts[expression.value] == 1
            and expression.value not in active
        ):
            expanded = expand(
                bindings[expression.value],
                active | {expression.value},
            )
            if _is_elementwise(expanded, function.body[-1].expression.type):
                consumed.add(expression.value)
                return expanded
            return expression
        if not expression.children:
            return expression
        children = tuple(expand(child, active) for child in expression.children)
        return replace(expression, children=children)

    returned = function.body[-1].expression
    if returned is None:
        return function
    expanded = expand(returned)
    if (
        not consumed
        or not _is_elementwise(expanded, expanded.type)
        or _tensor_operation_count(expanded) < 1
    ):
        return function

    fused = IRExpression(
        "fused_tensor",
        expanded.span,
        expanded.type,
        children=(expanded,),
    )
    body = tuple(
        statement
        for statement in function.body[:-1]
        if statement.name not in consumed
    ) + (replace(function.body[-1], expression=fused),)
    return replace(function, body=body)


def _is_elementwise(expression: IRExpression, result_type) -> bool:
    element_type = result_type.element_type
    if element_type is None or element_type.name not in {"f32", "f64"}:
        return False
    if expression.kind == "literal":
        return expression.type.name == element_type.name
    if expression.kind == "identifier":
        if expression.type.name == "Tensor":
            return (
                expression.type.element_type == element_type
                and expression.type.dimensions == result_type.dimensions
            )
        return expression.type.name == element_type.name
    if expression.kind == "group":
        return bool(expression.children) and _is_elementwise(
            expression.children[0], result_type
        )
    if expression.kind == "unary":
        return (
            expression.value in {"+", "-"}
            and bool(expression.children)
            and _is_elementwise(expression.children[0], result_type)
        )
    if expression.kind != "binary" or expression.value not in {"+", "-", "*"}:
        return False
    if len(expression.children) != 2:
        return False
    left, right = expression.children
    if not _is_elementwise(left, result_type) or not _is_elementwise(
        right, result_type
    ):
        return False
    left_tensor = left.type.name == "Tensor"
    right_tensor = right.type.name == "Tensor"
    if expression.type.name == "Tensor":
        return (
            (left_tensor or right_tensor)
            and expression.type.element_type == element_type
            and expression.type.dimensions == result_type.dimensions
        )
    return (
        not left_tensor
        and not right_tensor
        and expression.type.name == element_type.name
    )


def _tensor_operation_count(expression: IRExpression) -> int:
    current = int(
        expression.type.name == "Tensor"
        and expression.kind in {"binary", "unary"}
    )
    return current + sum(
        _tensor_operation_count(child) for child in expression.children
    )
