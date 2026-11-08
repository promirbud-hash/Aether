"""Typed intermediate representation for the supported Aether source subset."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

from src.frontend import (
    AssignmentStmt,
    Expr,
    ExprStmt,
    ForStmt,
    FunctionDecl,
    IfStmt,
    LetStmt,
    LoopControlStmt,
    Module,
    ReturnStmt,
    SourceSpan,
    Statement,
    TypeDecl,
    WhileStmt,
)
from src.semantic import (
    AetherType,
    Diagnostic,
    SemanticResult,
    UnitConversion,
    analyze,
    resolve_type,
)


@dataclass(frozen=True)
class IRMatchArm:
    variant: str
    binding: str | None
    value: IRExpression
    span: SourceSpan


@dataclass(frozen=True)
class IRExpression:
    kind: str
    span: SourceSpan
    type: AetherType
    value: str = ""
    children: tuple[IRExpression, ...] = ()
    arms: tuple[IRMatchArm, ...] = ()
    conversion: UnitConversion | None = None
    ownership: str = "plain"
    temporary_owner: bool = False
    field_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class IROwnershipAction:
    kind: str
    target: str = ""
    source: str = ""


@dataclass(frozen=True)
class IRStatement:
    kind: str
    span: SourceSpan
    name: str = ""
    binding_kind: str = ""
    expression: IRExpression | None = None
    ownership_actions: tuple[IROwnershipAction, ...] = ()
    operator: str = "="
    expressions: tuple[IRExpression, ...] = ()
    body: tuple[IRStatement, ...] = ()
    else_body: tuple[IRStatement, ...] = ()
    field_name: str | None = None


@dataclass(frozen=True)
class IRParameter:
    name: str
    type: AetherType


@dataclass(frozen=True)
class IRFunction:
    name: str
    parameters: tuple[IRParameter, ...]
    return_type: AetherType
    has_return_annotation: bool
    declared_effects: tuple[str, ...]
    required_effects: tuple[str, ...]
    body: tuple[IRStatement, ...]
    span: SourceSpan
    external_abi: str | None = None
    external_library: str | None = None
    frame_resources: tuple[str, ...] = ()
    frame_resource_types: tuple[tuple[str, AetherType], ...] = ()
    c_exported: bool = False
    external_symbol: str | None = None


def walk_statements(
    statements: tuple[IRStatement, ...],
) -> Iterator[IRStatement]:
    for statement in statements:
        yield statement
        yield from walk_statements(statement.body)
        yield from walk_statements(statement.else_body)


@dataclass(frozen=True)
class IRTypeDeclaration:
    name: str
    fields: tuple[tuple[str, AetherType], ...]
    span: SourceSpan


@dataclass(frozen=True)
class IRModule:
    functions: tuple[IRFunction, ...]
    types: tuple[IRTypeDeclaration, ...]
    source_paths: tuple[str, ...] = ()
    source_hashes: tuple[tuple[str, str], ...] = ()
    target: str | None = None

    def function(self, name: str) -> IRFunction:
        for function in self.functions:
            if function.name == name:
                return function
        raise KeyError(f"Unknown Aether function {name!r}")


class IRLoweringError(ValueError):
    def __init__(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        codes = ", ".join(
            getattr(diagnostic, "code", "unknown")
            for diagnostic in diagnostics
        )
        super().__init__(f"Cannot lower a module with semantic errors: {codes}")


def lower_module(module: Module, semantics: SemanticResult | None = None) -> IRModule:
    """Lower a semantically valid supported-subset module into typed IR."""
    semantics = semantics or analyze(module)
    if any(diagnostic.severity == "error" for diagnostic in semantics.diagnostics):
        raise IRLoweringError(semantics.diagnostics)

    expression_types = {span: type_ for span, type_ in semantics.expression_types}
    conversions = {conversion.span: conversion for conversion in semantics.conversions}
    declarations = {
        declaration.name: declaration
        for declaration in module.declarations
        if isinstance(declaration, FunctionDecl)
    }

    def contains_frame_buffer(type_: AetherType) -> bool:
        return type_.name == "FrameBuffer" or any(
            contains_frame_buffer(argument) for argument in type_.arguments
        )

    def lower_expression(expression: Expr) -> IRExpression:
        if (
            expression.kind == "call"
            and expression.children
            and expression.children[0].kind == "identifier"
            and expression.children[0].value == "gradient"
            and len(expression.children) == 3
            and expression.children[1].kind == "identifier"
        ):
            return IRExpression(
                "gradient",
                expression.span,
                expression_types.get(expression.span, AetherType("Unknown")),
                expression.children[1].value,
                (lower_expression(expression.children[2]),),
            )
        children = tuple(lower_expression(child) for child in expression.children)
        if (
            expression.kind == "binary"
            and expression.value == "|>"
            and len(children) == 2
            and children[1].kind in {"call", "identifier"}
            and children[1].type.name == "Unknown"
        ):
            children = (
                children[0],
                replace(
                    children[1],
                    type=expression_types.get(
                        expression.span,
                        AetherType("Unknown"),
                    ),
                ),
            )
        expression_type = expression_types.get(expression.span)
        if expression_type is None and expression.kind == "literal":
            if expression.value.startswith(("'", '"')):
                expression_type = AetherType("str")
            elif expression.value in {"true", "false"}:
                expression_type = AetherType("bool")
            elif expression.value == "none":
                expression_type = AetherType("None")
            else:
                expression_type = AetherType(
                    "f32"
                    if not expression.value.lower().startswith(("0x", "0b"))
                    and any(char in expression.value for char in ".eE")
                    else "i32"
                )
        arms = tuple(
            IRMatchArm(
                arm.variant,
                arm.binding,
                lower_expression(arm.value),
                arm.span,
            )
            for arm in expression.arms
        )
        return IRExpression(
            expression.kind,
            expression.span,
            expression_type or AetherType("Unknown"),
            expression.value,
            children,
            arms,
            conversions.get(expression.span),
            (
                children[0].ownership
                if expression.kind == "group"
                and children
                and contains_frame_buffer(expression_type)
                else "borrowed"
                if expression_type is not None
                and contains_frame_buffer(expression_type)
                and expression.kind == "identifier"
                else "owned"
                if expression_type is not None
                and contains_frame_buffer(expression_type)
                and expression.kind == "call"
                else "owned"
                if expression_type is not None
                and expression_type.name == "FrameBuffer"
                and expression.kind == "try"
                else "borrowed"
                if expression_type is not None
                and contains_frame_buffer(expression_type)
                else "plain"
            ),
            bool(
                expression_type is not None
                and contains_frame_buffer(expression_type)
                and (
                    expression.kind in {"call", "try"}
                    or (
                        expression.kind == "group"
                        and children
                        and children[0].temporary_owner
                    )
                )
            ),
            expression.field_names,
        )

    def untyped_expressions(
        expression: IRExpression,
        *,
        callee: bool = False,
    ) -> list[Diagnostic]:
        errors: list[Diagnostic] = []
        if not callee and expression.type.name == "Unknown":
            errors.append(
                Diagnostic(
                    "E_IR_UNTYPED",
                    f"Cannot lower untyped {expression.kind} expression",
                    expression.span,
                )
            )
        if expression.kind == "call":
            if expression.children:
                errors.extend(untyped_expressions(expression.children[0], callee=True))
                for child in expression.children[1:]:
                    errors.extend(untyped_expressions(child))
        elif expression.kind == "member":
            if expression.children:
                errors.extend(untyped_expressions(expression.children[0]))
        else:
            for child in expression.children:
                errors.extend(untyped_expressions(child))
        for arm in expression.arms:
            errors.extend(untyped_expressions(arm.value))
        return errors

    functions: list[IRFunction] = []
    types: list[IRTypeDeclaration] = []
    effect_summaries = {summary.function: summary for summary in semantics.effects}
    for declaration in module.declarations:
        if isinstance(declaration, TypeDecl):
            types.append(
                IRTypeDeclaration(
                    declaration.name,
                    tuple(
                        (field.name, resolve_type(field.type))
                        for field in declaration.fields
                    ),
                    declaration.span,
                )
            )
            continue
        if not isinstance(declaration, FunctionDecl):
            continue
        def lower_statements(
            statements: tuple[Statement, ...],
        ) -> tuple[IRStatement, ...]:
            body: list[IRStatement] = []
            for statement in statements:
                if isinstance(statement, IfStmt):
                    body.append(
                        IRStatement(
                            "if",
                            statement.span,
                            expression=lower_expression(statement.condition),
                            body=lower_statements(statement.body),
                            else_body=lower_statements(statement.else_body),
                        )
                    )
                elif isinstance(statement, WhileStmt):
                    body.append(
                        IRStatement(
                            "while",
                            statement.span,
                            expression=lower_expression(statement.condition),
                            body=lower_statements(statement.body),
                        )
                    )
                elif isinstance(statement, ForStmt):
                    body.append(
                        IRStatement(
                            "for",
                            statement.span,
                            name=statement.name,
                            expressions=(
                                lower_expression(statement.start),
                                lower_expression(statement.end),
                            ),
                            body=lower_statements(statement.body),
                        )
                    )
                elif isinstance(statement, LoopControlStmt):
                    body.append(IRStatement(statement.kind, statement.span))
                elif isinstance(statement, LetStmt):
                    expression = lower_expression(statement.value)
                    ownership_actions: tuple[IROwnershipAction, ...] = ()
                    if contains_frame_buffer(expression.type):
                        if expression.ownership == "borrowed":
                            ownership_actions = (
                                IROwnershipAction(
                                    "retain",
                                    statement.name,
                                    statement.value.value
                                    if statement.value.kind == "identifier"
                                    else "",
                                ),
                            )
                        else:
                            ownership_actions = (
                                IROwnershipAction("transfer", statement.name),
                            )
                    body.append(
                        IRStatement(
                            "let",
                            statement.span,
                            statement.name,
                            statement.binding_kind,
                            expression,
                            ownership_actions,
                        )
                    )
                elif isinstance(statement, AssignmentStmt):
                    expression = lower_expression(statement.value)
                    actions: list[IROwnershipAction] = []
                    if contains_frame_buffer(expression.type):
                        if expression.ownership == "borrowed":
                            actions.append(
                                IROwnershipAction(
                                    "retain",
                                    statement.name,
                                    statement.value.value
                                    if statement.value.kind == "identifier"
                                    else "",
                                )
                            )
                        actions.extend(
                            (
                                IROwnershipAction("release", statement.name),
                                IROwnershipAction("transfer", statement.name),
                            )
                        )
                    body.append(
                        IRStatement(
                            "assignment",
                            statement.span,
                            statement.name,
                            expression=expression,
                            ownership_actions=tuple(actions),
                            operator=statement.operator,
                            field_name=statement.field_name,
                        )
                    )
                elif isinstance(statement, ReturnStmt):
                    expression = (
                        lower_expression(statement.value)
                        if statement.value is not None
                        else None
                    )
                    actions: tuple[IROwnershipAction, ...] = ()
                    if expression is not None and contains_frame_buffer(expression.type):
                        if expression.ownership == "borrowed":
                            actions = (IROwnershipAction("retain_return"),)
                        else:
                            actions = (IROwnershipAction("transfer_return"),)
                    body.append(
                        IRStatement(
                            "return",
                            statement.span,
                            expression=expression,
                            ownership_actions=actions,
                        )
                    )
                elif isinstance(statement, ExprStmt):
                    body.append(
                        IRStatement(
                            "expression",
                            statement.span,
                            expression=lower_expression(statement.expression),
                        )
                    )
            return tuple(body)

        body = lower_statements(declaration.body)
        effect_summary = effect_summaries.get(declaration.name)
        functions.append(
            IRFunction(
                declaration.name,
                tuple(
                    IRParameter(
                        parameter.name,
                        resolve_type(parameter.type),
                    )
                    for parameter in declaration.parameters
                ),
                (
                    resolve_type(declaration.return_type)
                    if declaration.return_type is not None
                    else AetherType("Unknown")
                ),
                declaration.return_type is not None,
                effect_summary.declared if effect_summary else (),
                effect_summary.required if effect_summary else (),
                body,
                declaration.span,
                declaration.external_abi,
                declaration.external_library,
                tuple(
                    statement.name
                    for statement in walk_statements(body)
                    if statement.kind == "let"
                    and statement.expression is not None
                    and contains_frame_buffer(statement.expression.type)
                ),
                tuple(
                    (
                        statement.name,
                        statement.expression.type,
                    )
                    for statement in walk_statements(body)
                    if statement.kind == "let"
                    and statement.expression is not None
                    and contains_frame_buffer(statement.expression.type)
                ),
                any(attribute.name == "C" for attribute in declaration.attributes),
                declaration.external_symbol,
            )
        )
    target = next(
        (
            attribute.arguments[0]
            for declaration in module.declarations
            for attribute in declaration.attributes
            if attribute.name == "target" and attribute.arguments
        ),
        None,
    )
    ir_module = IRModule(tuple(functions), tuple(types), target=target)
    errors = tuple(
        diagnostic
        for function in ir_module.functions
        for statement in walk_statements(function.body)
        for expression in (
            *((statement.expression,) if statement.expression is not None else ()),
            *statement.expressions,
        )
        for diagnostic in untyped_expressions(expression)
    )
    if errors:
        raise IRLoweringError(errors)
    return ir_module
