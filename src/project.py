"""Relative-import project loading and single-IR lowering."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path, PureWindowsPath

from src.frontend import (
    AssignmentStmt,
    Diagnostic,
    Expr,
    ExprStmt,
    FunctionDecl,
    ImportDecl,
    LetStmt,
    Module,
    ReturnStmt,
    SourceSpan,
    TypeDecl,
    TypeExpr,
    parse_source,
    validate_module,
)
from src.ir import IRLoweringError, IRModule, lower_module
from src.parser import ParseError
from src.semantic import analyze


class ProjectCompilationError(ValueError):
    """Raised when a project cannot be loaded or resolved."""

    def __init__(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "; ".join(f"{item.code}: {item.message}" for item in diagnostics)
            or "Aether project compilation failed"
        )


def _source_digest(source: str) -> str:
    canonical_source = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical_source.encode("utf-8")).hexdigest()


def compile_project_ir(entry_path: str | os.PathLike[str]) -> IRModule:
    """Load a relative-import graph, resolve symbols, and lower it to one IR."""
    root = Path(entry_path).expanduser().resolve()
    project_root = root.parent
    if root.suffix.lower() != ".hl":
        raise ProjectCompilationError(
            (
                Diagnostic(
                    "E_PROJECT_ENTRY",
                    "Project entry path must name a .hl file",
                    SourceSpan(0, 0, 1, 1, str(root)),
                ),
            )
        )
    diagnostics: list[Diagnostic] = []
    modules: dict[Path, Module] = {}
    source_hashes: dict[Path, str] = {}
    states: dict[Path, str] = {}
    stack: list[Path] = []

    def add_error(code: str, message: str, span: SourceSpan) -> None:
        diagnostics.append(Diagnostic(code, message, span))

    def visit(path: Path, import_span: SourceSpan | None = None) -> None:
        state = states.get(path)
        if state == "loaded":
            return
        if state == "loading":
            cycle_start = stack.index(path)
            cycle = (*stack[cycle_start:], path)
            location = import_span or SourceSpan(0, 0, 1, 1, str(path))
            add_error(
                "E_IMPORT_CYCLE",
                "Import cycle: " + " -> ".join(_display_path(item, root.parent) for item in cycle),
                location,
            )
            return
        states[path] = "loading"
        stack.append(path)
        try:
            with path.open("r", encoding="utf-8", newline="") as source_file:
                source = source_file.read()
        except (OSError, ValueError) as error:
            location = import_span or SourceSpan(0, 0, 1, 1, str(path))
            add_error("E_PROJECT_IO", f"Cannot read source file {path}: {error}", location)
            stack.pop()
            states[path] = "loaded"
            return
        source_hashes[path] = _source_digest(source)
        try:
            parsed = parse_source(source, source_path=str(path), validate=False)
        except ParseError as error:
            add_error(
                "E_LEXICAL",
                str(error),
                SourceSpan(error.offset, error.offset, error.line, error.column, str(path)),
            )
            stack.pop()
            states[path] = "loaded"
            return
        modules[path] = parsed.ast
        diagnostics.extend(parsed.diagnostics)
        aliases: dict[str, ImportDecl] = {}
        for import_decl in parsed.ast.imports:
            previous = aliases.get(import_decl.alias)
            if previous is not None:
                add_error(
                    "E_DUPLICATE_IMPORT_ALIAS",
                    f"Import alias {import_decl.alias!r} is already declared",
                    import_decl.span,
                )
                continue
            aliases[import_decl.alias] = import_decl
            import_path = Path(import_decl.path)
            windows_path = PureWindowsPath(import_decl.path)
            if (
                not import_decl.path
                or any(ord(char) < 32 for char in import_decl.path)
                or import_path.is_absolute()
                or windows_path.is_absolute()
                or bool(windows_path.drive)
                or import_path.suffix.lower() != ".hl"
            ):
                add_error(
                    "E_IMPORT_PATH",
                    "Import path must be a non-empty relative .hl path",
                    import_decl.span,
                )
                continue
            try:
                target = (path.parent / import_path).resolve()
            except (OSError, RuntimeError, ValueError) as error:
                add_error(
                    "E_IMPORT_PATH",
                    f"Cannot resolve import path {import_decl.path!r}: {error}",
                    import_decl.span,
                )
                continue
            if not target.is_relative_to(project_root):
                add_error(
                    "E_IMPORT_PATH",
                    "Import path must resolve within the project entry directory",
                    import_decl.span,
                )
                continue
            visit(target, import_decl.span)
        stack.pop()
        states[path] = "loaded"

    visit(root)
    if diagnostics:
        raise ProjectCompilationError(tuple(diagnostics))

    type_declarations = {
        path: {
            declaration.name: declaration
            for declaration in module.declarations
            if isinstance(declaration, TypeDecl)
        }
        for path, module in modules.items()
    }
    function_declarations = {
        path: {
            declaration.name: declaration
            for declaration in module.declarations
            if isinstance(declaration, FunctionDecl)
        }
        for path, module in modules.items()
    }
    aliases_by_path = {
        path: {
            import_decl.alias: (path.parent / import_decl.path).resolve()
            for import_decl in module.imports
        }
        for path, module in modules.items()
    }

    def internal_name(path: Path, symbol: str, *, is_type: bool = False) -> str:
        if path == root:
            return symbol
        kind = "type" if is_type else "function"
        digest = hashlib.sha256(_display_path(path, root.parent).encode("utf-8")).hexdigest()[:16]
        return f"$module:{digest}:{kind}::{symbol}"

    rewritten_declarations = []
    for path, module in modules.items():
        local_functions = {
            name: internal_name(path, name)
            for name in function_declarations[path]
        }
        local_types = {
            name: internal_name(path, name, is_type=True)
            for name in type_declarations[path]
        }
        imported_modules = aliases_by_path[path]

        def resolve_imported_type(type_expr: TypeExpr) -> TypeExpr:
            arguments = tuple(resolve_imported_type(arg) for arg in type_expr.arguments)
            if type_expr.qualifier is None:
                type_name = local_types.get(type_expr.name, type_expr.name)
                return replace(type_expr, name=type_name, arguments=arguments)
            target = imported_modules.get(type_expr.qualifier)
            if target is None:
                add_error(
                    "E_UNKNOWN_IMPORT_ALIAS",
                    f"Unknown import alias {type_expr.qualifier!r}",
                    type_expr.span,
                )
                return replace(type_expr, arguments=arguments)
            if type_expr.name not in type_declarations.get(target, {}):
                add_error(
                    "E_IMPORT_UNKNOWN_TYPE",
                    f"Module alias {type_expr.qualifier!r} has no type {type_expr.name!r}",
                    type_expr.span,
                )
                return replace(type_expr, arguments=arguments)
            return replace(
                type_expr,
                name=internal_name(target, type_expr.name, is_type=True),
                qualifier=None,
                arguments=arguments,
            )

        def resolve_callee(callee: Expr) -> Expr:
            if callee.kind == "qualified":
                alias, separator, symbol = callee.value.partition("::")
                target = imported_modules.get(alias) if separator else None
                if target is None:
                    add_error(
                        "E_UNKNOWN_IMPORT_ALIAS",
                        f"Unknown import alias {alias!r}",
                        callee.span,
                    )
                    return replace(callee, kind="identifier")
                target_function = function_declarations.get(target, {}).get(symbol)
                if target_function is None:
                    add_error(
                        "E_IMPORT_UNKNOWN_FUNCTION",
                        f"Module alias {alias!r} has no function {symbol!r}",
                        callee.span,
                    )
                    return replace(callee, kind="identifier")
                return replace(
                    callee,
                    kind="identifier",
                    value=internal_name(target, symbol),
                    children=(),
                )
            if callee.kind == "identifier":
                return replace(
                    callee,
                    value=local_functions.get(callee.value, callee.value),
                )
            add_error(
                "E_IMPORT_QUALIFIED_CALL",
                "Imported functions must be called with alias::function(...)",
                callee.span,
            )
            return callee

        def rewrite_expression(expression: Expr) -> Expr:
            if expression.kind == "call":
                callee = resolve_callee(expression.children[0])
                raw_arguments = expression.children[1:]
                if (
                    expression.children[0].kind == "identifier"
                    and expression.children[0].value == "gradient"
                    and raw_arguments
                    and raw_arguments[0].kind in {"identifier", "qualified"}
                ):
                    arguments = (
                        resolve_callee(raw_arguments[0]),
                        *(rewrite_expression(child) for child in raw_arguments[1:]),
                    )
                else:
                    arguments = tuple(
                        rewrite_expression(child) for child in raw_arguments
                    )
                return replace(expression, children=(callee, *arguments))
            if expression.kind == "binary" and expression.value == "|>":
                left, right = expression.children
                if right.kind == "call":
                    rewritten_right = rewrite_expression(right)
                elif right.kind in {"identifier", "qualified"}:
                    rewritten_right = resolve_callee(right)
                else:
                    rewritten_right = rewrite_expression(right)
                return replace(
                    expression,
                    children=(rewrite_expression(left), rewritten_right),
                )
            if expression.kind == "qualified":
                add_error(
                    "E_IMPORT_QUALIFIED_CALL",
                    "Qualified imports may only be used as function calls",
                    expression.span,
                )
                return expression
            return replace(
                expression,
                children=tuple(rewrite_expression(child) for child in expression.children),
                arms=tuple(
                    replace(arm, value=rewrite_expression(arm.value))
                    for arm in expression.arms
                ),
            )

        for declaration in module.declarations:
            if isinstance(declaration, TypeDecl):
                rewritten_declarations.append(
                    replace(
                        declaration,
                        name=local_types[declaration.name],
                        fields=tuple(
                            replace(field, type=resolve_imported_type(field.type))
                            for field in declaration.fields
                        ),
                    )
                )
                continue
            if not isinstance(declaration, FunctionDecl):
                continue
            if declaration.external_abi is not None and path != root:
                add_error(
                    "E_PROJECT_FFI_UNSUPPORTED",
                    "Imported extern declarations are not yet supported by project execution",
                    declaration.span,
                )
            rewritten_body = []
            for statement in declaration.body:
                if isinstance(statement, LetStmt):
                    rewritten_body.append(
                        replace(
                            statement,
                            annotation=(
                                resolve_imported_type(statement.annotation)
                                if statement.annotation is not None
                                else None
                            ),
                            value=rewrite_expression(statement.value),
                        )
                    )
                elif isinstance(statement, AssignmentStmt):
                    rewritten_body.append(
                        replace(
                            statement,
                            value=rewrite_expression(statement.value),
                        )
                    )
                elif isinstance(statement, ReturnStmt):
                    rewritten_body.append(
                        replace(
                            statement,
                            value=(
                                rewrite_expression(statement.value)
                                if statement.value is not None
                                else None
                            ),
                        )
                    )
                elif isinstance(statement, ExprStmt):
                    rewritten_body.append(
                        replace(
                            statement,
                            expression=rewrite_expression(statement.expression),
                        )
                    )
            rewritten_declarations.append(
                replace(
                    declaration,
                    name=local_functions[declaration.name],
                    parameters=tuple(
                        replace(
                            parameter,
                            type=resolve_imported_type(parameter.type),
                        )
                        for parameter in declaration.parameters
                    ),
                    return_type=(
                        resolve_imported_type(declaration.return_type)
                        if declaration.return_type is not None
                        else None
                    ),
                    body=tuple(rewritten_body),
                )
            )

    if diagnostics:
        raise ProjectCompilationError(tuple(diagnostics))

    combined = Module(tuple(rewritten_declarations))
    syntax_diagnostics = validate_module(combined)
    semantic_result = analyze(combined)
    errors = tuple(
        diagnostic
        for diagnostic in (*syntax_diagnostics, *semantic_result.diagnostics)
        if diagnostic.severity == "error"
    )
    if errors:
        raise ProjectCompilationError(errors)
    try:
        ir_module = lower_module(combined, semantic_result)
        return replace(
            ir_module,
            source_paths=tuple(str(path) for path in modules),
            source_hashes=tuple(
                (str(path), source_hashes[path])
                for path in modules
            ),
        )
    except IRLoweringError as error:
        raise ProjectCompilationError(error.diagnostics) from error


def _display_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return os.path.relpath(path, base).replace("\\", "/")
