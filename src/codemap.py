"""Extract a compact project topology from an Aether concrete syntax tree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.frontend import (
    AssignmentStmt,
    Expr,
    FunctionDecl,
    LetStmt,
    Module,
    ReturnStmt,
    TypeDecl,
    TypeExpr,
    UnitExpr,
    parse_source,
    parse_source_file,
)
from src.parser import CSTNode, ParseError


_TRIVIA = {"WHITESPACE", "NEWLINE", "COMMENT"}
_KEYWORDS = {
    "fn", "func", "type", "struct", "let", "var", "const", "return",
    "if", "else", "for", "while", "match", "true", "false", "none",
}
_UNITS = {
    "mm", "cm", "m", "km", "um", "nm", "V", "mV", "uV", "A", "mA",
    "s", "ms", "us", "ns", "Hz", "kHz", "N", "Pa", "kPa", "J", "W",
    "rad", "deg", "degC", "degF", "K", "kg", "g",
}


@dataclass(frozen=True)
class MapNode:
    id: str
    kind: str
    label: str
    signature: str = ""
    permissions: tuple[str, ...] = ()
    units: tuple[str, ...] = ()


@dataclass(frozen=True)
class MapEdge:
    source: str
    target: str
    kind: str


@dataclass(frozen=True)
class CodeMap:
    nodes: tuple[MapNode, ...]
    edges: tuple[MapEdge, ...]


def _significant(children: tuple[CSTNode, ...] | list[CSTNode]) -> list[CSTNode]:
    return [child for child in children if child.kind not in _TRIVIA]


def _tokens(node: CSTNode):
    if node.is_leaf:
        if node.kind not in _TRIVIA:
            yield node
        return
    for child in node.children:
        yield from _tokens(child)


def _compact(nodes) -> str:
    output = ""
    previous_kind = ""
    for token in (leaf for node in nodes for leaf in _tokens(node)):
        if (
            previous_kind in {"IDENTIFIER", "KEYWORD", "NUMBER"}
            and token.kind in {"IDENTIFIER", "KEYWORD", "NUMBER"}
        ):
            output += " "
        output += token.value
        previous_kind = token.kind
    return output


def _identifiers(node: CSTNode) -> list[str]:
    return [token.value for token in _tokens(node) if token.kind == "IDENTIFIER"]


def _units(node: CSTNode) -> tuple[str, ...]:
    return tuple(sorted(set(_identifiers(node)) & _UNITS))


def _build_codemap_from_cst(tree: CSTNode | Module) -> CodeMap:
    """Extract function/type declarations, permissions, units, and call edges."""
    if isinstance(tree, Module):
        return _build_ast_codemap(tree)
    items = _significant(tree.children)
    declarations: list[MapNode] = []
    function_bodies: dict[str, CSTNode] = {}
    type_names: set[str] = set()
    function_names: set[str] = set()
    external_function_names: set[str] = set()
    function_types: dict[str, set[str]] = {}
    external_calls: dict[str, set[str]] = {}
    pending_permissions: list[str] = []
    index = 0

    while index < len(items):
        item = items[index]
        if item.value == "@" and index + 1 < len(items):
            permission = items[index + 1].value
            if permission in {"io", "net", "fs", "ffi"}:
                pending_permissions.append(permission)
            index += 2
            continue

        if (
            item.value == "extern"
            and index + 3 < len(items)
            and items[index + 1].kind == "STRING"
            and items[index + 2].value == "fn"
        ):
            name_node = items[index + 3]
            if name_node.kind != "IDENTIFIER":
                index += 1
                continue
            params_index = next(
                (
                    i for i in range(index + 4, len(items))
                    if items[i].kind == "group_paren"
                ),
                None,
            )
            if params_index is None:
                index += 1
                continue
            semicolon_index = next(
                (
                    i for i in range(params_index + 1, len(items))
                    if items[i].value == ";"
                ),
                None,
            )
            if semicolon_index is None:
                index += 1
                continue
            name = name_node.value
            abi = items[index + 1].value.strip("\"'")
            signature = (
                f'extern"{abi}" '
                + _compact(items[index + 3 : semicolon_index])
            )
            function_names.add(name)
            external_function_names.add(name)
            declarations.append(
                MapNode(
                    id=f"x:{name}",
                    kind="external",
                    label=name,
                    signature=signature,
                    permissions=("ffi",),
                )
            )
            index = semicolon_index + 1
            pending_permissions.clear()
            continue

        if item.value in {"fn", "func"} and index + 1 < len(items):
            name_node = items[index + 1]
            if name_node.kind != "IDENTIFIER":
                index += 1
                continue
            name = name_node.value
            params_index = next(
                (i for i in range(index + 2, len(items)) if items[i].kind == "group_paren"),
                None,
            )
            if params_index is None:
                index += 2
                continue
            body_index = next(
                (
                    i
                    for i in range(params_index + 1, len(items))
                    if items[i].kind == "group_brace"
                ),
                None,
            )
            if body_index is None:
                index += 2
                continue

            body = items[body_index]
            signature = _compact(items[index + 2 : body_index])
            function_names.add(name)
            function_bodies[name] = body
            function_types[name] = set(_identifiers(items[params_index])) | set(
                _identifiers_from_nodes(items[params_index + 1 : body_index])
            )
            declarations.append(
                MapNode(
                    id=f"f:{name}",
                    kind="function",
                    label=name,
                    signature=signature,
                    permissions=tuple(dict.fromkeys(pending_permissions)),
                    units=tuple(sorted(set(_units(items[params_index]) + _units(body)))),
                )
            )
            pending_permissions.clear()
            index = body_index + 1
            continue

        if item.value in {"type", "struct"} and index + 1 < len(items):
            name_node = items[index + 1]
            if name_node.kind != "IDENTIFIER":
                index += 1
                continue
            body_index = next(
                (
                    i
                    for i in range(index + 2, len(items))
                    if items[i].kind == "group_brace"
                ),
                None,
            )
            if body_index is None:
                index += 2
                continue
            name = name_node.value
            body = items[body_index]
            type_names.add(name)
            declarations.append(
                MapNode(
                    id=f"t:{name}",
                    kind="type",
                    label=name,
                    signature=_compact([body]),
                    units=_units(body),
                )
            )
            pending_permissions.clear()
            index = body_index + 1
            continue

        pending_permissions.clear()
        index += 1

    edges: set[MapEdge] = set()
    for name, body in function_bodies.items():
        caller = f"f:{name}"
        for called in _calls(body):
            if called == name:
                continue
            if called in function_names:
                target = (
                    f"x:{called}"
                    if called in external_function_names
                    else f"f:{called}"
                )
            else:
                target = f"x:{called}"
                external_calls.setdefault(called, set()).add(target)
            edges.add(MapEdge(caller, target, "call"))
        for referenced_type in function_types.get(name, set()) & type_names:
            edges.add(MapEdge(caller, f"t:{referenced_type}", "type"))

    declarations.extend(
        MapNode(id=f"x:{name}", kind="external", label=name)
        for name in sorted(external_calls)
    )
    return CodeMap(
        nodes=tuple(sorted(declarations, key=lambda node: node.id)),
        edges=tuple(sorted(edges, key=lambda edge: (edge.source, edge.kind, edge.target))),
    )


def _type_text(type_expr: TypeExpr) -> str:
    if type_expr.name == "[]":
        return "[" + ",".join(type_expr.dimensions) + "]"
    if type_expr.name == "Tensor" and len(type_expr.arguments) == 2:
        unit = (
            "," + _unit_expression_text(type_expr.unit_expression)
            if type_expr.unit_expression is not None
            else ""
        )
        return (
            f"{type_expr.name}<"
            + ",".join(_type_text(argument) for argument in type_expr.arguments)
            + unit
            + ">"
        )
    if type_expr.unit_expression is not None:
        return (
            f"{type_expr.name}<"
            + _unit_expression_text(type_expr.unit_expression)
            + ">"
        )
    if type_expr.arguments:
        return f"{type_expr.name}<" + ",".join(
            _type_text(argument) for argument in type_expr.arguments
        ) + ">"
    return type_expr.name


def _unit_expression_text(expression: UnitExpr) -> str:
    if expression.kind == "symbol":
        return expression.value.replace("µ", "u")
    if expression.kind == "dimensionless":
        return "1"
    if expression.kind == "group":
        return "(" + _unit_expression_text(expression.children[0]) + ")"
    if expression.kind == "power":
        return (
            _unit_expression_text(expression.children[0])
            + "^"
            + str(expression.exponent)
        )
    operator = "*" if expression.kind == "multiply" else "/"
    return (
        _unit_expression_text(expression.children[0])
        + operator
        + _unit_expression_text(expression.children[1])
    )


def _type_names(type_expr: TypeExpr) -> set[str]:
    names = {type_expr.name} if type_expr.name not in {"[]"} else set()
    for argument in type_expr.arguments:
        names.update(_type_names(argument))
    return names


def _type_units(type_expr: TypeExpr) -> set[str]:
    units = _type_names(type_expr) & _UNITS
    if type_expr.unit_expression is not None:
        units.update(_unit_expression_names(type_expr.unit_expression))
    return units


def _unit_expression_names(expression: UnitExpr) -> set[str]:
    names = {expression.value.replace("µ", "u")} if expression.kind == "symbol" else set()
    for child in expression.children:
        names.update(_unit_expression_names(child))
    return names


def _expression_nodes(expression: Expr):
    yield expression
    for child in expression.children:
        yield from _expression_nodes(child)
    for arm in expression.arms:
        yield from _expression_nodes(arm.value)


def _statement_expression(statement):
    if isinstance(statement, (LetStmt, AssignmentStmt)):
        return statement.value
    if isinstance(statement, ReturnStmt):
        return statement.value
    return statement.expression


def _build_ast_codemap(module: Module) -> CodeMap:
    functions = {
        declaration.name: declaration
        for declaration in module.declarations
        if isinstance(declaration, FunctionDecl)
    }
    types = {
        declaration.name: declaration
        for declaration in module.declarations
        if isinstance(declaration, TypeDecl)
    }
    nodes: list[MapNode] = []
    edges: set[MapEdge] = set()
    external_calls: set[str] = set()

    for name, declaration in functions.items():
        params = ", ".join(
            f"{parameter.name}:{_type_text(parameter.type)}"
            for parameter in declaration.parameters
        )
        signature = f"({params})"
        if declaration.return_type is not None:
            signature += "->" + _type_text(declaration.return_type)
        if declaration.external_abi is not None:
            signature = (
                f'extern"{declaration.external_abi}" '
                f"@link({declaration.external_library}) {signature}"
            )
            nodes.append(
                MapNode(
                    f"x:{name}",
                    "external",
                    name,
                    signature,
                    ("ffi",),
                )
            )
            continue
        used_types = set()
        units: set[str] = set()
        for parameter in declaration.parameters:
            used_types.update(_type_names(parameter.type))
            units.update(_type_units(parameter.type))
        if declaration.return_type is not None:
            used_types.update(_type_names(declaration.return_type))
            units.update(_type_units(declaration.return_type))
        for statement in declaration.body:
            expression = _statement_expression(statement)
            if expression is None:
                continue
            for node in _expression_nodes(expression):
                if node.kind == "call" and node.children[0].kind == "identifier":
                    called = node.children[0].value
                    if called in functions:
                        target = (
                            f"x:{called}"
                            if functions[called].external_abi is not None
                            else f"f:{called}"
                        )
                        edges.add(MapEdge(f"f:{name}", target, "call"))
                    else:
                        external_calls.add(called)
                        edges.add(MapEdge(f"f:{name}", f"x:{called}", "call"))
        for referenced_type in used_types & types.keys():
            edges.add(MapEdge(f"f:{name}", f"t:{referenced_type}", "type"))
        permissions = tuple(
            attribute.name
            for attribute in declaration.attributes
            if attribute.name in {"io", "net", "fs", "ffi"}
        )
        if not permissions:
            permissions = ("pure",)
        nodes.append(
            MapNode(
                f"f:{name}",
                "function",
                name,
                signature,
                permissions,
                tuple(sorted(units)),
            )
        )

    for name, declaration in types.items():
        used_types: set[str] = set()
        units: set[str] = set()
        fields = []
        for field in declaration.fields:
            field_type = _type_text(field.type)
            fields.append(f"{field.name}:{field_type}")
            used_types.update(_type_names(field.type))
            units.update(_type_units(field.type))
        signature = "{" + ",".join(fields) + "}"
        nodes.append(
            MapNode(
                f"t:{name}",
                "type",
                name,
                signature,
                (),
                tuple(sorted(units)),
            )
        )
        for referenced_type in used_types & types.keys():
            if referenced_type != name:
                edges.add(MapEdge(f"t:{name}", f"t:{referenced_type}", "type"))

    existing_external_names = {
        node.label for node in nodes if node.kind == "external"
    }
    nodes.extend(
        MapNode(f"x:{name}", "external", name)
        for name in sorted(external_calls - existing_external_names)
    )
    return CodeMap(
        nodes=tuple(sorted(nodes, key=lambda node: node.id)),
        edges=tuple(sorted(edges, key=lambda edge: (edge.source, edge.kind, edge.target))),
    )


def _identifiers_from_nodes(nodes: list[CSTNode]) -> list[str]:
    return [token.value for node in nodes for token in _tokens(node) if token.kind == "IDENTIFIER"]


def _calls(node: CSTNode) -> set[str]:
    tokens = list(_tokens(node))
    calls: set[str] = set()
    for index, token in enumerate(tokens[:-1]):
        if (
            token.kind == "IDENTIFIER"
            and token.value not in _KEYWORDS
            and tokens[index + 1].value == "("
        ):
            calls.add(token.value)
    return calls


def _source_text(node: CSTNode) -> str:
    if node.is_leaf:
        return node.value
    return "".join(_source_text(child) for child in node.children)


def build_codemap(tree: CSTNode | Module) -> CodeMap:
    """Build the same codemap from concrete syntax and AST inputs."""
    if isinstance(tree, Module):
        return _build_ast_codemap(tree)
    try:
        module = parse_source(_source_text(tree), validate=False).ast
    except ParseError:
        return _build_codemap_from_cst(tree)
    return _build_ast_codemap(module)


def render_map(code_map: CodeMap) -> str:
    """Serialize a topology as short, line-oriented `.map` text."""
    lines = ["@map1"]
    for node in code_map.nodes:
        prefix = {"function": "F", "type": "T", "external": "X"}[node.kind]
        parts = [f"{prefix} {node.label}"]
        if node.signature:
            parts.append(node.signature)
        if node.permissions:
            parts.append("@" + ",".join(node.permissions))
        if node.units:
            parts.append("u:" + ",".join(node.units))
        lines.append(" ".join(parts))
    lines.extend(
        f"{'C' if edge.kind == 'call' else 'R'} "
        f"{edge.source.split(':', 1)[1]}>{edge.target.split(':', 1)[1]}"
        for edge in code_map.edges
    )
    return "\n".join(lines)


def render_active_slice(code_map: CodeMap, active_node: str) -> str:
    """Render the active node and its directly connected nodes and edges."""
    node_ids = {node.id for node in code_map.nodes}
    if active_node not in node_ids:
        raise ValueError(f"Unknown active node {active_node!r}")
    selected = {active_node}
    selected.update(
        edge.target if edge.source == active_node else edge.source
        for edge in code_map.edges
        if edge.source == active_node or edge.target == active_node
    )
    selected_nodes = [
        node for node in code_map.nodes if node.id in selected and node.kind != "external"
    ]
    lines = ["@map1"]
    for node in selected_nodes:
        prefix = "F" if node.kind == "function" else "T"
        line = f"{prefix} {node.label}{node.signature}"
        if node.permissions:
            line += " @" + ",".join(node.permissions)
        if node.units:
            line += " u:" + ",".join(node.units)
        lines.append(line)

    call_targets: dict[str, set[str]] = {}
    for edge in code_map.edges:
        if (
            edge.kind == "call"
            and edge.source in selected
            and edge.target in selected
        ):
            call_targets.setdefault(edge.source, set()).add(
                edge.target.split(":", 1)[1]
            )
        elif (
            edge.kind == "type"
            and edge.source in selected
            and edge.target in selected
        ):
            lines.append(
                f"R {edge.source.split(':', 1)[1]}>{edge.target.split(':', 1)[1]}"
            )
    for source, targets in sorted(call_targets.items()):
        lines.append(f"C {source.split(':', 1)[1]}>{','.join(sorted(targets))}")
    return "\n".join(lines)


def generate_codemap_from_file(path: str | Path) -> CodeMap:
    """Parse a source file and extract its topology."""
    result = parse_source_file(path)
    errors = [diagnostic for diagnostic in result.diagnostics if diagnostic.severity == "error"]
    if errors:
        joined = "; ".join(
            f"{item.code} line {item.span.line}: {item.message}" for item in errors
        )
        raise ValueError(f"Cannot generate codemap: {joined}")
    return build_codemap(result.ast)
