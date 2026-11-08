"""Lossless lexical CST parser for the first Aether prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


_KEYWORDS = {
    "fn",
    "func",
    "type",
    "struct",
    "let",
    "var",
    "const",
    "return",
    "import",
    "as",
    "if",
    "else",
    "for",
    "while",
    "match",
    "break",
    "continue",
    "true",
    "false",
    "none",
}
_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
_CLOSE_TO_OPEN = {closing: opening for opening, closing in _OPEN_TO_CLOSE.items()}
_GROUP_KINDS = {"(": "group_paren", "[": "group_bracket", "{": "group_brace"}
_OPERATORS = (
    "|>",
    "->",
    "=>",
    "::",
    "?.",
    "??",
    "==",
    "!=",
    "<=",
    ">=",
    "<<",
    ">>",
    "&&",
    "||",
    "+=",
    "-=",
    "*=",
    "/=",
    "**",
    "..",
    "∇",
)
_NUMBER_RE = re.compile(
    r"(?:"
    r"0[xX][0-9a-fA-F](?:_?[0-9a-fA-F])*"
    r"|0[bB][01](?:_?[01])*"
    r"|\d+\.(?!\.)\d*"
    r"|\d+\.(?!\.)"
    r"|\.\d+"
    r"|\d+"
    r")(?:[eE][+-]?\d+)?"
)


class ParseError(ValueError):
    """Raised when the source has an unterminated token or unbalanced delimiter."""

    def __init__(self, message: str, source: str, offset: int) -> None:
        self.offset = offset
        self.line = source.count("\n", 0, offset) + 1
        last_newline = source.rfind("\n", 0, offset)
        self.column = offset - last_newline
        super().__init__(f"{message} at line {self.line}, column {self.column}")


@dataclass(frozen=True)
class CSTNode:
    """A concrete syntax node; leaves retain exact source text and offsets."""

    kind: str
    start: int
    end: int
    value: str = ""
    children: tuple[CSTNode, ...] = field(default_factory=tuple)

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    start: int
    end: int


def _is_identifier_start(char: str) -> bool:
    return char == "_" or char.isalpha()


def _is_identifier_continue(char: str) -> bool:
    return char == "_" or char.isalnum()


def _scan_string(source: str, start: int) -> int:
    quote = source[start]
    triple = source.startswith(quote * 3, start)
    delimiter = quote * (3 if triple else 1)
    cursor = start + len(delimiter)

    while cursor < len(source):
        if source[cursor] == "\\":
            cursor += 2
            continue
        if source.startswith(delimiter, cursor):
            return cursor + len(delimiter)
        if not triple and source[cursor] in "\r\n":
            raise ParseError("Unterminated string literal", source, start)
        cursor += 1

    raise ParseError("Unterminated string literal", source, start)


def tokenize(source: str) -> list[Token]:
    """Tokenize source while retaining trivia and exact character offsets."""
    tokens: list[Token] = []
    cursor = 0

    while cursor < len(source):
        start = cursor
        char = source[cursor]

        if char in " \t\f\v":
            cursor += 1
            while cursor < len(source) and source[cursor] in " \t\f\v":
                cursor += 1
            kind = "WHITESPACE"
        elif char in "\r\n":
            if source.startswith("\r\n", cursor):
                cursor += 2
            else:
                cursor += 1
            kind = "NEWLINE"
        elif source.startswith("//", cursor) or char == "#":
            newline = source.find("\n", cursor)
            cursor = len(source) if newline == -1 else newline
            kind = "COMMENT"
        elif source.startswith("/*", cursor):
            end = source.find("*/", cursor + 2)
            if end == -1:
                raise ParseError("Unterminated block comment", source, start)
            cursor = end + 2
            kind = "COMMENT"
        elif char in "'\"":
            cursor = _scan_string(source, cursor)
            kind = "STRING"
        elif _is_identifier_start(char):
            cursor += 1
            while cursor < len(source) and _is_identifier_continue(source[cursor]):
                cursor += 1
            value = source[start:cursor]
            kind = "KEYWORD" if value in _KEYWORDS else "IDENTIFIER"
        elif char.isdigit() or (
            char == "." and cursor + 1 < len(source) and source[cursor + 1].isdigit()
        ):
            match = _NUMBER_RE.match(source, cursor)
            assert match is not None
            cursor = match.end()
            kind = "NUMBER"
        else:
            operator = next(
                (candidate for candidate in _OPERATORS if source.startswith(candidate, cursor)),
                None,
            )
            if operator is not None:
                cursor += len(operator)
                kind = "OPERATOR"
            elif char in "()[]{}":
                cursor += 1
                kind = "DELIMITER"
            else:
                cursor += 1
                kind = "OPERATOR" if char in "+-*/%=!<>:&|^~?@∇" else "PUNCTUATION"

        tokens.append(Token(kind, source[start:cursor], start, cursor))

    return tokens


_tokenize = tokenize


def parse(source: str) -> CSTNode:
    """Parse Aether source into a lossless, delimiter-grouped concrete syntax tree.

    This initial parser intentionally preserves unknown syntax as tokens rather than
    rejecting it; only malformed strings, comments, or delimiter nesting are errors.
    """
    return parse_tokens(source, tokenize(source))


def parse_tokens(source: str, tokens: list[Token]) -> CSTNode:
    """Build a lossless delimiter-grouped CST from a token stream for this source."""
    root: list[CSTNode] = []
    stack: list[tuple[str, int, list[CSTNode]]] = [("", 0, root)]

    for token in tokens:
        if token.value in _OPEN_TO_CLOSE:
            children = [
                CSTNode(token.kind, token.start, token.end, value=token.value)
            ]
            stack.append((token.value, token.start, children))
        elif token.value in _CLOSE_TO_OPEN:
            if len(stack) == 1 or stack[-1][0] != _CLOSE_TO_OPEN[token.value]:
                raise ParseError(f"Unexpected closing delimiter {token.value!r}", source, token.start)
            opening, start, children = stack.pop()
            children.append(CSTNode(token.kind, token.start, token.end, value=token.value))
            stack[-1][2].append(
                CSTNode(_GROUP_KINDS[opening], start, token.end, children=tuple(children))
            )
        else:
            stack[-1][2].append(
                CSTNode(token.kind, token.start, token.end, value=token.value)
            )

    if len(stack) > 1:
        opening, start, _ = stack[-1]
        raise ParseError(f"Unclosed delimiter {opening!r}", source, start)

    return CSTNode("source_file", 0, len(source), children=tuple(root))


def parse_file(path: str | Path, *, encoding: str = "utf-8") -> CSTNode:
    """Read and parse a UTF-8 Aether source file."""
    with Path(path).open("r", encoding=encoding, newline="") as source_file:
        return parse(source_file.read())


def render(node: CSTNode) -> str:
    """Reconstruct the exact source represented by a CST."""
    if node.is_leaf:
        return node.value
    return "".join(render(child) for child in node.children)
