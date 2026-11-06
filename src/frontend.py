"""Syntax parser and initial semantic analysis for the Aether v0 subset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.parser import (
    CSTNode,
    ParseError,
    Token,
    _OPEN_TO_CLOSE,
    parse_tokens,
    tokenize,
)


_TRIVIA = {"WHITESPACE", "NEWLINE", "COMMENT"}
_BINARY_PRECEDENCE = {
    "|>": 1,
    "??": 2,
    "||": 3,
    "&&": 4,
    "==": 5,
    "!=": 5,
    "<": 5,
    ">": 5,
    "<=": 5,
    ">=": 5,
    "+": 6,
    "-": 6,
    "*": 7,
    "/": 7,
    "%": 7,
    "**": 8,
}
_BUILTIN_CALLS = {
    "mean", "max_abs", "subtract", "scale", "abs", "min", "max", "sum", "len",
    "convert", "io_write", "io_read_line", "fs_read_text", "fs_write_text",
    "fs_exists", "net_get", "net_post",
}
_KNOWN_ATTRIBUTES = {
    "pure", "io", "net", "fs", "ffi", "link", "persist", "contract", "C", "target",
}
_BUILTIN_TYPES = {
    "bool", "str", "char", "void", "usize",
    "i8", "i16", "i32", "i64", "i128",
    "u8", "u16", "u32", "u64", "u128",
    "f16", "bf16", "f32", "f64", "fp8", "i4", "t2",
    "Tensor", "Slice", "Pool", "Id", "Result",
    "IoError", "FsError", "NetError",
}
_UNIT_NAMES = {
    "mm", "cm", "m", "km", "um", "nm", "V", "mV", "uV", "A", "mA",
    "s", "ms", "us", "ns", "Hz", "kHz", "N", "Pa", "kPa", "J", "W",
    "rad", "deg", "degC", "degF", "K", "kg", "g",
}
_CST_KINDS = {
    "FunctionDecl": "function_declaration",
    "TypeDecl": "type_declaration",
    "ImportDecl": "import_declaration",
}


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int
    line: int
    column: int
    source_path: str | None = None


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    span: SourceSpan
    severity: str = "error"


@dataclass(frozen=True)
class Attribute:
    name: str
    arguments: tuple[str, ...]
    span: SourceSpan


@dataclass(frozen=True)
class TypeExpr:
    name: str
    arguments: tuple[TypeExpr, ...]
    dimensions: tuple[str, ...]
    span: SourceSpan
    qualifier: str | None = None


@dataclass(frozen=True)
class Parameter:
    name: str
    type: TypeExpr
    span: SourceSpan


@dataclass(frozen=True)
class FieldDecl:
    name: str
    type: TypeExpr
    span: SourceSpan


@dataclass(frozen=True)
class MatchArm:
    variant: str
    binding: str | None
    value: Expr
    span: SourceSpan


@dataclass(frozen=True)
class Expr:
    kind: str
    span: SourceSpan
    value: str = ""
    children: tuple[Expr, ...] = ()
    arms: tuple[MatchArm, ...] = ()


@dataclass(frozen=True)
class LetStmt:
    name: str
    binding_kind: str
    annotation: TypeExpr | None
    value: Expr
    span: SourceSpan


@dataclass(frozen=True)
class ReturnStmt:
    value: Expr | None
    span: SourceSpan


@dataclass(frozen=True)
class ExprStmt:
    expression: Expr
    span: SourceSpan


Statement = LetStmt | ReturnStmt | ExprStmt


@dataclass(frozen=True)
class FunctionDecl:
    name: str
    parameters: tuple[Parameter, ...]
    return_type: TypeExpr | None
    body: tuple[Statement, ...]
    attributes: tuple[Attribute, ...]
    span: SourceSpan
    external_abi: str | None = None
    external_library: str | None = None


@dataclass(frozen=True)
class TypeDecl:
    name: str
    fields: tuple[FieldDecl, ...]
    attributes: tuple[Attribute, ...]
    span: SourceSpan


@dataclass(frozen=True)
class ImportDecl:
    path: str
    alias: str
    span: SourceSpan


Declaration = FunctionDecl | TypeDecl


@dataclass(frozen=True)
class Module:
    declarations: tuple[Declaration, ...]
    imports: tuple[ImportDecl, ...] = ()


@dataclass(frozen=True)
class FrontendResult:
    cst: CSTNode
    ast: Module
    diagnostics: tuple[Diagnostic, ...]


class _SyntaxError(Exception):
    def __init__(self, message: str, token: Token) -> None:
        self.message = message
        self.token = token
        super().__init__(message)


class _FrontendParser:
    def __init__(self, source: str, source_path: str | None = None) -> None:
        self.source = source
        self.source_path = source_path
        all_tokens = tokenize(source)
        self.tokens = [token for token in all_tokens if token.kind not in _TRIVIA]
        try:
            self.lexical_cst = parse_tokens(source, all_tokens)
        except ParseError:
            self.lexical_cst = CSTNode(
                "source_file",
                0,
                len(source),
                children=tuple(
                    CSTNode(token.kind, token.start, token.end, value=token.value)
                    for token in all_tokens
                ),
            )
        self.position = 0
        self.diagnostics: list[Diagnostic] = []

    def current(self) -> Token:
        if self.position >= len(self.tokens):
            return Token("EOF", "", len(self.source), len(self.source))
        return self.tokens[self.position]

    def advance(self) -> Token:
        token = self.current()
        if self.position < len(self.tokens):
            self.position += 1
        return token

    def match(self, *values: str) -> Token | None:
        if self.current().value in values:
            return self.advance()
        return None

    def expect(self, value: str, message: str | None = None) -> Token:
        token = self.match(value)
        if token is None:
            raise _SyntaxError(message or f"Expected {value!r}", self.current())
        return token

    def expect_identifier(self, message: str = "Expected an identifier") -> Token:
        token = self.current()
        if token.kind != "IDENTIFIER":
            raise _SyntaxError(message, token)
        return self.advance()

    def span(self, start: int, end: int) -> SourceSpan:
        line = self.source.count("\n", 0, start) + 1
        last_newline = self.source.rfind("\n", 0, start)
        return SourceSpan(
            start,
            end,
            line,
            start - last_newline,
            self.source_path,
        )

    def diagnose(self, code: str, message: str, token: Token) -> None:
        self.diagnostics.append(
            Diagnostic(code, message, self.span(token.start, token.end))
        )

    def parse(self) -> Module:
        declarations: list[Declaration] = []
        imports: list[ImportDecl] = []
        while self.current().kind != "EOF":
            start_position = self.position
            attributes: list[Attribute] = []
            try:
                if self.current().value == "import":
                    if attributes:
                        raise _SyntaxError(
                            "Imports cannot have declaration attributes",
                            self.current(),
                        )
                    imports.append(self.parse_import())
                    continue
                while self.current().value == "@":
                    attributes.append(self.parse_attribute())
                token = self.current()
                if token.value == "extern":
                    self.advance()
                    abi = self.current()
                    if abi.kind != "STRING":
                        raise _SyntaxError(
                            'Expected ABI string after "extern"',
                            abi,
                        )
                    self.advance()
                    if abi.value not in {'"C"', "'C'"}:
                        raise _SyntaxError(
                            'Only the extern "C" ABI is supported',
                            abi,
                        )
                    if self.current().value != "fn":
                        raise _SyntaxError(
                            'Expected "fn" after extern ABI',
                            self.current(),
                        )
                    declarations.append(
                        self.parse_function(tuple(attributes), external_abi="C")
                    )
                elif token.value in {"fn", "func"}:
                    declarations.append(self.parse_function(tuple(attributes)))
                elif token.value in {"type", "struct"}:
                    declarations.append(self.parse_type_decl(tuple(attributes)))
                else:
                    raise _SyntaxError(
                        "Expected an import, function, or type declaration", token
                    )
            except _SyntaxError as error:
                self.diagnose("E_SYNTAX", error.message, error.token)
                self.recover_declaration(start_position)
            if self.position == start_position:
                self.advance()
        return Module(tuple(declarations), tuple(imports))

    def parse_import(self) -> ImportDecl:
        start = self.expect("import").start
        path_token = self.current()
        if path_token.kind != "STRING" or not path_token.value.startswith('"'):
            raise _SyntaxError('Expected a quoted relative path after "import"', path_token)
        self.advance()
        try:
            import ast as python_ast

            path = python_ast.literal_eval(path_token.value)
        except (SyntaxError, ValueError) as error:
            raise _SyntaxError("Invalid import path string", path_token) from error
        if not isinstance(path, str):
            raise _SyntaxError("Import path must be a string", path_token)
        self.expect("as", 'Expected "as" after import path')
        alias = self.expect_identifier("Expected an import alias")
        end = self.expect(";", "Expected ';' after import declaration").end
        return ImportDecl(path, alias.value, self.span(start, end))

    def recover_declaration(self, start_position: int) -> None:
        if self.position == start_position and self.current().kind != "EOF":
            self.advance()
        depth = 0
        while self.current().kind != "EOF":
            token = self.current()
            if token.value in _OPEN_TO_CLOSE:
                depth += 1
            elif token.value in {")", "]", "}"}:
                depth = max(0, depth - 1)
                self.advance()
                if depth == 0:
                    return
                continue
            elif depth == 0 and token.value in {
                "import", "fn", "func", "type", "struct", "@"
            }:
                return
            self.advance()

    def parse_attribute(self) -> Attribute:
        start = self.expect("@").start
        name = self.expect_identifier("Expected an attribute name")
        arguments: list[str] = []
        end = name.end
        if self.match("("):
            while self.current().value != ")" and self.current().kind != "EOF":
                argument = self.advance()
                if argument.value != ",":
                    arguments.append(argument.value)
                end = argument.end
            end = self.expect(")", "Expected ')' after attribute arguments").end
        return Attribute(name.value, tuple(arguments), self.span(start, end))

    def parse_function(
        self,
        attributes: tuple[Attribute, ...],
        *,
        external_abi: str | None = None,
    ) -> FunctionDecl:
        start = self.advance().start
        name_token = self.expect_identifier("Expected a function name")
        self.expect("(")
        parameters: list[Parameter] = []
        while self.current().value not in {")", ""}:
            parameter_start = self.current().start
            name = self.expect_identifier("Expected a parameter name")
            self.expect(":", "Expected ':' after parameter name")
            type_expr = self.parse_type()
            parameters.append(
                Parameter(name.value, type_expr, self.span(parameter_start, type_expr.span.end))
            )
            if not self.match(","):
                break
        self.expect(")", "Expected ')' after function parameters")
        return_type = None
        if self.match("->"):
            return_type = self.parse_type()
        if external_abi is not None:
            end = self.expect(";", "Expected ';' after extern function declaration").end
            body: tuple[Statement, ...] = ()
        else:
            body, end = self.parse_block()
        link_attribute = next(
            (attribute for attribute in attributes if attribute.name == "link"),
            None,
        )
        external_library = None
        if link_attribute is not None and len(link_attribute.arguments) == 1:
            raw_library = link_attribute.arguments[0]
            if len(raw_library) >= 2 and raw_library[0] in {'"', "'"}:
                external_library = raw_library[1:-1]
        return FunctionDecl(
            name_token.value,
            tuple(parameters),
            return_type,
            body,
            attributes,
            self.span(attributes[0].span.start if attributes else start, end),
            external_abi,
            external_library,
        )

    def parse_type_decl(self, attributes: tuple[Attribute, ...]) -> TypeDecl:
        start = self.advance().start
        name = self.expect_identifier("Expected a type name")
        self.expect("{", "Expected '{' before type fields")
        fields: list[FieldDecl] = []
        while self.current().value not in {"}", ""}:
            field_start = self.current().start
            field_name = self.expect_identifier("Expected a field name")
            self.expect(":", "Expected ':' after field name")
            field_type = self.parse_type()
            fields.append(
                FieldDecl(
                    field_name.value,
                    field_type,
                    self.span(field_start, field_type.span.end),
                )
            )
            if not self.match(","):
                if self.current().value != "}":
                    continue
        end = self.expect("}", "Expected '}' after type fields").end
        return TypeDecl(
            name.value,
            tuple(fields),
            attributes,
            self.span(attributes[0].span.start if attributes else start, end),
        )

    def parse_type(self) -> TypeExpr:
        start_token = self.current()
        if self.match("["):
            dimensions: list[str] = []
            while self.current().value not in {"]", ""}:
                if self.current().value != ",":
                    dimensions.append(self.advance().value)
                else:
                    self.advance()
            end = self.expect("]", "Expected ']' after dimensions").end
            return TypeExpr("[]", (), tuple(dimensions), self.span(start_token.start, end))

        name = self.expect_identifier("Expected a type name")
        qualifier = None
        if self.match("::"):
            qualifier = name.value
            name = self.expect_identifier("Expected a type name after '::'")
        arguments: list[TypeExpr] = []
        end = name.end
        if self.match("<"):
            while self.current().value not in {">", ""}:
                arguments.append(self.parse_type())
                if not self.match(","):
                    break
            end = self.expect(">", "Expected '>' after type arguments").end
        return TypeExpr(
            name.value,
            tuple(arguments),
            (),
            self.span(start_token.start, end),
            qualifier,
        )

    def parse_block(self) -> tuple[tuple[Statement, ...], int]:
        self.expect("{", "Expected a function body")
        statements: list[Statement] = []
        while self.current().value not in {"}", ""}:
            start_position = self.position
            try:
                statements.append(self.parse_statement())
                self.match(";")
            except _SyntaxError as error:
                self.diagnose("E_SYNTAX", error.message, error.token)
                self.recover_statement(start_position)
            if self.position == start_position:
                self.advance()
        end = self.expect("}", "Expected '}' after function body").end
        return tuple(statements), end

    def recover_statement(self, start_position: int) -> None:
        if self.position == start_position and self.current().kind != "EOF":
            self.advance()
        while self.current().kind != "EOF" and self.current().value not in {";", "}"}:
            self.advance()
        self.match(";")

    def parse_statement(self) -> Statement:
        token = self.current()
        if token.value in {"let", "var", "const"}:
            start = self.advance()
            name = self.expect_identifier("Expected a binding name")
            annotation = None
            if self.match(":"):
                annotation = self.parse_type()
            self.expect("=", "Expected '=' in binding")
            value = self.parse_expression()
            return LetStmt(
                name.value,
                start.value,
                annotation,
                value,
                self.span(start.start, value.span.end),
            )
        if token.value == "return":
            start = self.advance()
            value = None
            if self.current().value not in {";", "}", ""}:
                value = self.parse_expression()
            return ReturnStmt(
                value,
                self.span(start.start, value.span.end if value else start.end),
            )
        expression = self.parse_expression()
        return ExprStmt(expression, expression.span)

    def parse_expression(self, minimum_precedence: int = 0) -> Expr:
        left = self.parse_prefix()
        while True:
            token = self.current()
            if token.value == "?":
                end = self.advance().end
                left = Expr("try", self.span(left.span.start, end), children=(left,))
                continue
            if token.value == "(":
                left = self.parse_call(left)
                continue
            if token.value == "::":
                operator = self.advance()
                member = self.expect_identifier("Expected a qualified symbol name")
                if left.kind != "identifier":
                    raise _SyntaxError(
                        "A qualified name must start with an import alias",
                        operator,
                    )
                left = Expr(
                    "qualified",
                    self.span(left.span.start, member.end),
                    f"{left.value}::{member.value}",
                )
                continue
            if token.value in {".", "?."}:
                operator = self.advance()
                member = self.expect_identifier("Expected a member name")
                left = Expr(
                    "member",
                    self.span(left.span.start, member.end),
                    operator.value,
                    (left, Expr("identifier", self.span(member.start, member.end), member.value)),
                )
                continue
            if token.value == "[":
                self.advance()
                index = self.parse_expression()
                end = self.expect("]", "Expected ']' after index").end
                left = Expr("index", self.span(left.span.start, end), children=(left, index))
                continue
            precedence = _BINARY_PRECEDENCE.get(token.value)
            if precedence is None or precedence < minimum_precedence:
                break
            operator = self.advance()
            next_precedence = precedence if operator.value == "**" else precedence + 1
            right = self.parse_expression(next_precedence)
            left = Expr(
                "binary",
                self.span(left.span.start, right.span.end),
                operator.value,
                (left, right),
            )
        return left

    def parse_prefix(self) -> Expr:
        token = self.current()
        if token.value == "match":
            return self.parse_match_expression()
        if token.value in {"!", "-", "+"}:
            operator = self.advance()
            value = self.parse_expression(9)
            return Expr("unary", self.span(operator.start, value.span.end), operator.value, (value,))
        if token.value == "(":
            start = self.advance()
            expression = self.parse_expression()
            end = self.expect(")", "Expected ')' after expression").end
            return Expr("group", self.span(start.start, end), children=(expression,))
        if token.kind in {"NUMBER", "STRING"} or token.value in {"true", "false", "none"}:
            self.advance()
            return Expr("literal", self.span(token.start, token.end), token.value)
        if token.kind == "IDENTIFIER":
            self.advance()
            return Expr("identifier", self.span(token.start, token.end), token.value)
        raise _SyntaxError("Expected an expression", token)

    def parse_match_expression(self) -> Expr:
        start = self.expect("match")
        subject = self.parse_expression()
        self.expect("{", "Expected '{' after match value")
        arms: list[MatchArm] = []
        while self.current().value not in {"}", ""}:
            arm_start = self.current()
            variant = self.expect_identifier("Expected Result variant Ok or Err")
            if variant.value not in {"Ok", "Err"}:
                raise _SyntaxError("Expected Result variant Ok or Err", variant)
            self.expect("(", "Expected '(' after Result variant")
            binding_token = self.expect_identifier(
                "Expected a payload binding or '_'"
            )
            binding = None if binding_token.value == "_" else binding_token.value
            self.expect(")", "Expected ')' after Result payload binding")
            self.expect("=>", "Expected '=>' after Result pattern")
            value = self.parse_expression()
            arms.append(
                MatchArm(
                    variant.value,
                    binding,
                    value,
                    self.span(arm_start.start, value.span.end),
                )
            )
            if not self.match(","):
                break
        end = self.expect("}", "Expected '}' after Result match arms")
        return Expr(
            "match",
            self.span(start.start, end.end),
            children=(subject,),
            arms=tuple(arms),
        )

    def parse_call(self, callee: Expr) -> Expr:
        self.expect("(")
        arguments: list[Expr] = []
        while self.current().value not in {")", ""}:
            arguments.append(self.parse_expression())
            if not self.match(","):
                break
        end = self.expect(")", "Expected ')' after call arguments").end
        return Expr(
            "call",
            self.span(callee.span.start, end),
            children=(callee, *arguments),
        )


def _build_syntax_cst(source: str, ast: Module, lexical_cst: CSTNode) -> CSTNode:
    nodes: list[CSTNode] = []
    children = lexical_cst.children
    cursor = 0
    top_level_nodes = sorted(
        (*ast.imports, *ast.declarations),
        key=lambda node: node.span.start,
    )
    for declaration in top_level_nodes:
        declaration_children = []
        while cursor < len(children) and children[cursor].end <= declaration.span.start:
            nodes.append(children[cursor])
            cursor += 1
        while (
            cursor < len(children)
            and children[cursor].start >= declaration.span.start
            and children[cursor].end <= declaration.span.end
        ):
            declaration_children.append(children[cursor])
            cursor += 1
        kind = _CST_KINDS[type(declaration).__name__]
        nodes.append(
            CSTNode(
                kind,
                declaration.span.start,
                declaration.span.end,
                children=tuple(declaration_children),
            )
        )
    nodes.extend(children[cursor:])
    return CSTNode("module", 0, len(source), children=tuple(nodes))


def _analyze(module: Module, parser: _FrontendParser) -> None:
    for import_decl in module.imports:
        parser.diagnostics.append(
            Diagnostic(
                "E_IMPORT_PROJECT_ONLY",
                "Imports require compile_project() and cannot be used with compile_source()",
                import_decl.span,
            )
        )
    declarations: dict[str, Declaration] = {}
    functions = {declaration.name for declaration in module.declarations if isinstance(declaration, FunctionDecl)}
    type_names = {declaration.name for declaration in module.declarations if isinstance(declaration, TypeDecl)}
    for declaration in module.declarations:
        if declaration.name in declarations:
            parser.diagnostics.append(
                Diagnostic("E_DUPLICATE_DECL", f"Duplicate declaration {declaration.name!r}", declaration.span)
            )
        else:
            declarations[declaration.name] = declaration
        seen_attributes: set[str] = set()
        for attribute in declaration.attributes:
            if attribute.name not in _KNOWN_ATTRIBUTES:
                parser.diagnostics.append(
                    Diagnostic(
                        "E_UNKNOWN_ATTRIBUTE",
                        f"Unknown attribute @{attribute.name}",
                        attribute.span,
                    )
                )
            if attribute.name in seen_attributes:
                parser.diagnostics.append(
                    Diagnostic(
                        "E_DUPLICATE_ATTRIBUTE",
                        f"Duplicate attribute @{attribute.name}",
                        attribute.span,
                    )
                )
            seen_attributes.add(attribute.name)
        if "pure" in seen_attributes and seen_attributes & {"io", "net", "fs", "ffi"}:
            parser.diagnostics.append(
                Diagnostic(
                    "E_CONFLICTING_EFFECTS",
                    "A declaration cannot combine @pure with an effect attribute",
                    declaration.span,
                )
            )
        link_attributes = [
            attribute for attribute in declaration.attributes
            if attribute.name == "link"
        ]
        for attribute in link_attributes:
            if (
                len(attribute.arguments) != 1
                or not (
                    len(attribute.arguments[0]) >= 2
                    and attribute.arguments[0][0] in {'"', "'"}
                    and attribute.arguments[0][-1] == attribute.arguments[0][0]
                )
            ):
                parser.diagnostics.append(
                    Diagnostic(
                        "E_LINK_ATTRIBUTE",
                        '@link requires exactly one quoted library name',
                        attribute.span,
                    )
                )
            elif attribute.arguments[0][1:-1] == "":
                parser.diagnostics.append(
                    Diagnostic(
                        "E_LINK_ATTRIBUTE",
                        "Library name in @link cannot be empty",
                        attribute.span,
                    )
                )

        if isinstance(declaration, FunctionDecl):
            has_ffi = "ffi" in seen_attributes
            if declaration.external_abi is not None and not has_ffi:
                parser.diagnostics.append(
                    Diagnostic(
                        "E_EXTERN_EFFECT",
                        'An extern "C" declaration must declare @ffi',
                        declaration.span,
                    )
                )
            if declaration.external_abi is None and link_attributes:
                parser.diagnostics.append(
                    Diagnostic(
                        "E_EXTERN_ATTRIBUTE",
                        "@link may only be used on extern declarations",
                        declaration.span,
                    )
                )
        elif link_attributes or "ffi" in seen_attributes:
            parser.diagnostics.append(
                Diagnostic(
                    "E_EXTERN_ATTRIBUTE",
                    "@ffi and @link may only be used on extern declarations",
                    declaration.span,
                )
            )

    for declaration in module.declarations:
        if isinstance(declaration, TypeDecl):
            field_names: set[str] = set()
            for field in declaration.fields:
                if field.name in field_names:
                    parser.diagnostics.append(
                        Diagnostic(
                            "E_DUPLICATE_FIELD",
                            f"Duplicate field {field.name!r}",
                            field.span,
                        )
                    )
                field_names.add(field.name)
                _analyze_type(field.type, type_names, parser.diagnostics)
            continue
        if not isinstance(declaration, FunctionDecl):
            continue
        for parameter in declaration.parameters:
            _analyze_type(parameter.type, type_names, parser.diagnostics)
        if declaration.return_type is not None:
            _analyze_type(declaration.return_type, type_names, parser.diagnostics)
        locals_: set[str] = set()
        for parameter in declaration.parameters:
            if parameter.name in locals_:
                parser.diagnostics.append(
                    Diagnostic("E_DUPLICATE_BINDING", f"Duplicate parameter {parameter.name!r}", parameter.span)
                )
            locals_.add(parameter.name)

        for statement in declaration.body:
            expression = (
                statement.value if isinstance(statement, (LetStmt, ReturnStmt))
                else statement.expression
            )
            if expression is not None:
                _analyze_expression(
                    expression,
                    locals_,
                    functions,
                    parser.diagnostics,
                )
            if isinstance(statement, LetStmt):
                if statement.annotation is not None:
                    _analyze_type(statement.annotation, type_names, parser.diagnostics)
                if statement.name in locals_:
                    parser.diagnostics.append(
                        Diagnostic(
                            "E_DUPLICATE_BINDING",
                            f"Duplicate local binding {statement.name!r}",
                            statement.span,
                        )
                    )
                locals_.add(statement.name)


def _analyze_type(
    type_expr: TypeExpr,
    declared_types: set[str],
    diagnostics: list[Diagnostic],
) -> None:
    if type_expr.qualifier is not None:
        diagnostics.append(
            Diagnostic(
                "E_IMPORT_PROJECT_ONLY",
                "Qualified types require compile_project()",
                type_expr.span,
            )
        )
    if (
        type_expr.name != "[]"
        and type_expr.qualifier is None
        and type_expr.name not in _BUILTIN_TYPES
        and type_expr.name not in declared_types
        and type_expr.name not in _UNIT_NAMES
    ):
        diagnostics.append(
            Diagnostic(
                "E_UNKNOWN_TYPE",
                f"Unknown type {type_expr.name!r}",
                type_expr.span,
            )
        )
    for argument in type_expr.arguments:
        _analyze_type(argument, declared_types, diagnostics)


def _analyze_expression(
    expression: Expr,
    locals_: set[str],
    functions: set[str],
    diagnostics: list[Diagnostic],
) -> None:
    if expression.kind == "identifier":
        if expression.value not in locals_ | functions | _BUILTIN_CALLS:
            diagnostics.append(
                Diagnostic(
                    "E_UNKNOWN_NAME",
                    f"Unknown name {expression.value!r}",
                    expression.span,
                )
            )
        return
    if expression.kind == "call":
        callee = expression.children[0]
        if (
            callee.kind == "identifier"
            and callee.value not in locals_ | functions | _BUILTIN_CALLS
        ):
            diagnostics.append(
                Diagnostic(
                    "E_UNKNOWN_CALL",
                    f"Unknown function {callee.value!r}",
                    callee.span,
                )
            )
        elif callee.kind != "identifier":
            _analyze_expression(callee, locals_, functions, diagnostics)
        for argument in expression.children[1:]:
            _analyze_expression(argument, locals_, functions, diagnostics)
        return
    if expression.kind == "member":
        _analyze_expression(expression.children[0], locals_, functions, diagnostics)
        return
    if expression.kind == "qualified":
        diagnostics.append(
            Diagnostic(
                "E_IMPORT_PROJECT_ONLY",
                "Qualified names require compile_project()",
                expression.span,
            )
        )
        return
    for child in expression.children:
        _analyze_expression(child, locals_, functions, diagnostics)
    for arm in expression.arms:
        arm_locals = locals_ | ({arm.binding} if arm.binding else set())
        _analyze_expression(arm.value, arm_locals, functions, diagnostics)


def _render_type(type_expr: TypeExpr) -> str:
    if type_expr.name == "[]":
        return "[" + ",".join(type_expr.dimensions) + "]"
    name = (
        f"{type_expr.qualifier}::{type_expr.name}"
        if type_expr.qualifier
        else type_expr.name
    )
    if type_expr.arguments:
        return f"{name}<" + ",".join(
            _render_type(argument) for argument in type_expr.arguments
        ) + ">"
    return name


def _render_expression(expression: Expr, parent_precedence: int = 0) -> str:
    if expression.kind in {"identifier", "literal"}:
        return expression.value
    if expression.kind == "qualified":
        return expression.value
    if expression.kind == "group":
        return "(" + _render_expression(expression.children[0]) + ")"
    if expression.kind == "unary":
        return expression.value + _render_expression(expression.children[0], 9)
    if expression.kind == "binary":
        precedence = _BINARY_PRECEDENCE[expression.value]
        right_precedence = precedence if expression.value == "**" else precedence + 1
        rendered = (
            _render_expression(expression.children[0], precedence)
            + f" {expression.value} "
            + _render_expression(expression.children[1], right_precedence)
        )
        return f"({rendered})" if precedence < parent_precedence else rendered
    if expression.kind == "call":
        return _render_expression(expression.children[0], 10) + "(" + ", ".join(
            _render_expression(argument) for argument in expression.children[1:]
        ) + ")"
    if expression.kind == "member":
        return (
            _render_expression(expression.children[0], 10)
            + expression.value
            + _render_expression(expression.children[1])
        )
    if expression.kind == "index":
        return (
            _render_expression(expression.children[0], 10)
            + "["
            + _render_expression(expression.children[1])
            + "]"
        )
    if expression.kind == "try":
        return _render_expression(expression.children[0], 10) + "?"
    if expression.kind == "match":
        rendered_arms = ", ".join(
            f"{arm.variant}({arm.binding or '_'}) => "
            f"{_render_expression(arm.value)}"
            for arm in expression.arms
        )
        return (
            "match "
            + _render_expression(expression.children[0])
            + " { "
            + rendered_arms
            + " }"
        )
    raise ValueError(f"Unsupported expression kind {expression.kind!r}")


def _escape_import_path(path: str) -> str:
    return path.replace("\\", "\\\\").replace('"', '\\"')


def render_module(module: Module) -> str:
    """Render a canonical source form from the parsed AST (comments are omitted)."""
    lines: list[str] = [
        f'import "{_escape_import_path(import_decl.path)}" as {import_decl.alias};'
        for import_decl in module.imports
    ]
    for declaration in module.declarations:
        lines.extend(
            f"@{attribute.name}"
            + (
                "(" + ",".join(attribute.arguments) + ")"
                if attribute.arguments
                else ""
            )
            for attribute in declaration.attributes
        )
        if isinstance(declaration, TypeDecl):
            lines.append(f"type {declaration.name} {{")
            lines.extend(
                f"    {field.name}: {_render_type(field.type)},"
                for field in declaration.fields
            )
            lines.append("}")
            continue
        parameters = ", ".join(
            f"{parameter.name}: {_render_type(parameter.type)}"
            for parameter in declaration.parameters
        )
        return_type = (
            f" -> {_render_type(declaration.return_type)}"
            if declaration.return_type
            else ""
        )
        if declaration.external_abi is not None:
            lines.append(
                f'extern "{declaration.external_abi}" fn '
                f"{declaration.name}({parameters}){return_type};"
            )
            continue
        lines.append(f"fn {declaration.name}({parameters}){return_type} {{")
        for statement in declaration.body:
            if isinstance(statement, LetStmt):
                annotation = (
                    f": {_render_type(statement.annotation)}"
                    if statement.annotation
                    else ""
                )
                rendered = (
                    f"{statement.binding_kind} {statement.name}{annotation} = "
                    f"{_render_expression(statement.value)};"
                )
            elif isinstance(statement, ReturnStmt):
                rendered = "return"
                if statement.value is not None:
                    rendered += " " + _render_expression(statement.value)
                rendered += ";"
            else:
                rendered = _render_expression(statement.expression) + ";"
            lines.append("    " + rendered)
        lines.append("}")
    return "\n".join(lines) + ("\n" if lines else "")


def validate_module(module: Module) -> tuple[Diagnostic, ...]:
    """Run declaration and local-name validation on an already parsed module."""
    parser = _FrontendParser("")
    _analyze(module, parser)
    return tuple(parser.diagnostics)


def parse_source(
    source: str,
    *,
    source_path: str | None = None,
    validate: bool = True,
) -> FrontendResult:
    """Parse source into syntax-aware CST and AST, returning recoverable diagnostics."""
    parser = _FrontendParser(source, source_path)
    ast = parser.parse()
    if validate:
        parser.diagnostics.extend(validate_module(ast))
        from src.semantic import analyze

        parser.diagnostics.extend(analyze(ast).diagnostics)
    cst = _build_syntax_cst(source, ast, parser.lexical_cst)
    return FrontendResult(cst, ast, tuple(parser.diagnostics))


def parse_source_file(path: str | Path, *, encoding: str = "utf-8") -> FrontendResult:
    """Read and parse a source file without newline normalization."""
    source_path = Path(path).resolve()
    with source_path.open("r", encoding=encoding, newline="") as source_file:
        return parse_source(source_file.read(), source_path=str(source_path))
