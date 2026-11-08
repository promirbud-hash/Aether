"""Small deterministic LSP/JSON-RPC transceiver for Aether source documents."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import sys
from typing import BinaryIO

from src.ai_view import (
    AIView,
    ProfileError,
    default_profile,
    expand_ai_view,
    generate_ai_view_with_map,
    load_profiles,
)
from src.codemap import build_codemap, render_active_slice, render_map
from src.frontend import (
    Diagnostic,
    FrontendResult,
    SourceSpan,
    parse_source,
    render_module,
)
from src.parser import CSTNode, ParseError


_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_MAX_HEADER_BYTES = 8192


class LspProtocolError(ValueError):
    """Raised for malformed or oversized LSP frames."""


class InvalidPosition(ValueError):
    """Raised for a malformed or out-of-range UTF-16 LSP position."""


@dataclass(frozen=True)
class DocumentAnalysis:
    frontend: FrontendResult | None
    diagnostics: tuple[Diagnostic, ...]
    lexical_error: ParseError | None = None


@dataclass
class _Document:
    uri: str
    text: str
    version: int
    analysis: DocumentAnalysis | None = None
    cst: CSTNode | None = None
    ai_views: dict[str, AIView] = field(default_factory=dict)
    map_text: str | None = None

    def invalidate(self, text: str, version: int) -> None:
        self.text = text
        self.version = version
        self.analysis = None
        self.cst = None
        self.ai_views.clear()
        self.map_text = None


def _line_starts(text: str) -> list[int]:
    starts = [0]
    starts.extend(index + 1 for index, char in enumerate(text) if char == "\n")
    return starts


def _offset_at_position(text: str, position: dict) -> int:
    line = position.get("line")
    character = position.get("character")
    if (
        not isinstance(line, int)
        or isinstance(line, bool)
        or not isinstance(character, int)
        or isinstance(character, bool)
        or line < 0
        or character < 0
    ):
        raise InvalidPosition("LSP position requires non-negative integer line and character")
    starts = _line_starts(text)
    if line >= len(starts):
        raise InvalidPosition(f"Line {line} is outside the document")
    start = starts[line]
    end = text.find("\n", start)
    if end == -1:
        end = len(text)
    elif end > start and text[end - 1] == "\r":
        end -= 1
    units = 0
    for offset in range(start, end):
        if units == character:
            return offset
        char_units = len(text[offset].encode("utf-16-le")) // 2
        if units < character < units + char_units:
            raise InvalidPosition("Position splits a UTF-16 surrogate pair")
        units += char_units
    if units == character:
        return end
    raise InvalidPosition(
        f"Character {character} is outside line {line} (length {units} UTF-16 units)"
    )


def _position_at_offset(text: str, offset: int) -> dict[str, int]:
    if offset < 0 or offset > len(text):
        raise InvalidPosition(f"Offset {offset} is outside the document")
    line = text.count("\n", 0, offset)
    line_start = text.rfind("\n", 0, offset) + 1
    character = len(text[line_start:offset].encode("utf-16-le")) // 2
    return {"line": line, "character": character}


def _range_at_offsets(text: str, start: int, end: int) -> dict:
    return {
        "start": _position_at_offset(text, start),
        "end": _position_at_offset(text, end),
    }


def _diagnostic_to_lsp(text: str, diagnostic: Diagnostic) -> dict:
    return {
        "range": _range_at_offsets(text, diagnostic.span.start, diagnostic.span.end),
        "severity": 1 if diagnostic.severity == "error" else 2,
        "code": diagnostic.code,
        "source": "aether",
        "message": diagnostic.message,
    }


def _lexical_diagnostic(error: ParseError, source_length: int) -> Diagnostic:
    return Diagnostic(
        "E_LEX",
        str(error),
        span=SourceSpan(
            error.offset,
            min(error.offset + 1, source_length),
            error.line,
            error.column,
        ),
    )


def _position_dict(position) -> dict[str, int]:
    if not isinstance(position, dict):
        raise InvalidPosition("Expected a position object")
    return position


def _nearest_mapping(view: AIView, offset: int):
    for mapping in view.mappings:
        if mapping.view_start <= offset < mapping.view_end:
            return mapping
    if not view.mappings:
        return None
    return min(
        view.mappings,
        key=lambda mapping: min(
            abs(offset - mapping.view_start),
            abs(offset - mapping.view_end),
        ),
    )


def _apply_text_edits(text: str, edits: list) -> str:
    resolved: list[tuple[int, int, str]] = []
    for edit in edits:
        if not isinstance(edit, dict) or not isinstance(edit.get("newText"), str):
            raise ValueError("Each AI-view edit must contain newText")
        range_value = edit.get("range")
        if not isinstance(range_value, dict):
            raise InvalidPosition("Each AI-view edit requires a range")
        start = _offset_at_position(text, _position_dict(range_value.get("start")))
        end = _offset_at_position(text, _position_dict(range_value.get("end")))
        if end < start:
            raise InvalidPosition("AI-view edit range end precedes start")
        resolved.append((start, end, edit["newText"]))
    resolved.sort(key=lambda item: (item[0], item[1]))
    for previous, current in zip(resolved, resolved[1:]):
        if current[0] < previous[1] or current[0] == previous[0]:
            raise InvalidPosition("AI-view edit ranges overlap or are ambiguous")
    for start, end, replacement in reversed(resolved):
        text = text[:start] + replacement + text[end:]
    return text


class AetherLanguageServer:
    """Document-scoped analysis and Tri-View service; no external AI is loaded."""

    def __init__(self) -> None:
        self.documents: dict[str, _Document] = {}
        self.profiles = {profile.name: profile for profile in load_profiles()}
        default = default_profile()
        self.profiles.setdefault(default.name, default)
        self.encoders = {}
        self.shutdown_requested = False
        self.exit_requested = False

    def _document(self, uri: str) -> _Document:
        document = self.documents.get(uri)
        if document is None:
            raise KeyError(f"Document is not open: {uri}")
        return document

    def _analysis(self, document: _Document) -> DocumentAnalysis:
        if document.analysis is None:
            try:
                frontend = parse_source(document.text)
            except ParseError as error:
                document.analysis = DocumentAnalysis(
                    None,
                    (_lexical_diagnostic(error, len(document.text)),),
                    error,
                )
            else:
                document.analysis = DocumentAnalysis(
                    frontend,
                    frontend.diagnostics,
                )
        return document.analysis

    def _view(self, document: _Document, profile_name: str) -> AIView:
        cached = document.ai_views.get(profile_name)
        if cached is not None:
            return cached
        profile = self.profiles.get(profile_name)
        if profile is None:
            raise KeyError(f"Unknown tokenizer profile {profile_name!r}")
        analysis = self._analysis(document)
        if analysis.frontend is None:
            raise ValueError(f"Cannot generate views: {analysis.diagnostics[0].message}")
        errors = [item for item in analysis.diagnostics if item.severity == "error"]
        if errors:
            joined = "; ".join(
                f"{item.code} line {item.span.line}: {item.message}" for item in errors
            )
            raise ValueError(f"Cannot generate views: {joined}")
        document.cst = analysis.frontend.cst
        view = generate_ai_view_with_map(
            document.cst,
            profile,
            self.encoders.get(profile_name),
        )
        document.ai_views[profile_name] = view
        return view

    def _map_text(self, document: _Document) -> str:
        if document.map_text is not None:
            return document.map_text
        analysis = self._analysis(document)
        if analysis.frontend is None:
            raise ValueError(f"Cannot generate map: {analysis.diagnostics[0].message}")
        errors = [item for item in analysis.diagnostics if item.severity == "error"]
        if errors:
            joined = "; ".join(
                f"{item.code} line {item.span.line}: {item.message}" for item in errors
            )
            raise ValueError(f"Cannot generate map: {joined}")
        document.map_text = render_map(build_codemap(analysis.frontend.ast))
        return document.map_text

    def _publish_diagnostics(self, document: _Document) -> dict:
        diagnostics = self._analysis(document).diagnostics
        return {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": document.uri,
                "version": document.version,
                "diagnostics": [
                    _diagnostic_to_lsp(document.text, item) for item in diagnostics
                ],
            },
        }

    def _apply_changes(self, document: _Document, params: dict) -> dict | None:
        version = params.get("textDocument", {}).get("version")
        changes = params.get("contentChanges")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError("textDocument/didChange requires an integer document version")
        if version <= document.version:
            return None
        if not isinstance(changes, list) or not changes:
            raise ValueError("textDocument/didChange requires non-empty contentChanges")
        new_text = document.text
        for change in changes:
            if not isinstance(change, dict) or not isinstance(change.get("text"), str):
                raise ValueError("Each content change must contain text")
            text = change["text"]
            range_value = change.get("range")
            if range_value is None:
                new_text = text
                continue
            if not isinstance(range_value, dict):
                raise InvalidPosition("Change range must be a range object")
            start = _offset_at_position(new_text, _position_dict(range_value.get("start")))
            end = _offset_at_position(new_text, _position_dict(range_value.get("end")))
            if end < start:
                raise InvalidPosition("Change range end precedes start")
            range_length = change.get("rangeLength")
            if range_length is not None:
                actual_length = len(new_text[start:end].encode("utf-16-le")) // 2
                if range_length != actual_length:
                    raise InvalidPosition(
                        f"rangeLength {range_length} does not match actual {actual_length}"
                    )
            new_text = new_text[:start] + text + new_text[end:]
        document.invalidate(new_text, version)
        return self._publish_diagnostics(document)

    def _views_result(self, params: dict) -> dict:
        uri = params.get("textDocument", {}).get("uri")
        if not isinstance(uri, str):
            raise ValueError("aether/getViews requires textDocument.uri")
        document = self._document(uri)
        profile_name = params.get("profile", default_profile().name)
        if not isinstance(profile_name, str):
            raise ValueError("profile must be a string")
        ai_view = self._view(document, profile_name)
        code_map = self._map_text(document)
        active = params.get("activeNode")
        active_slice = (
            render_active_slice(
                build_codemap(self._analysis(document).frontend.ast),
                active,
            )
            if active is not None
            else None
        )
        return {
            "uri": uri,
            "version": document.version,
            "profile": profile_name,
            "ai": ai_view.text,
            "map": code_map,
            "active": active_slice,
            "mappings": [
                {
                    "sourceRange": _range_at_offsets(
                        document.text,
                        mapping.source_start,
                        mapping.source_end,
                    ),
                    "viewRange": _range_at_offsets(
                        ai_view.text,
                        mapping.view_start,
                        mapping.view_end,
                    ),
                    "kind": mapping.kind,
                }
                for mapping in ai_view.mappings
            ],
        }

    def _source_position(self, params: dict) -> dict:
        uri = params.get("textDocument", {}).get("uri")
        if not isinstance(uri, str):
            raise ValueError("aether/sourcePosition requires textDocument.uri")
        document = self._document(uri)
        profile_name = params.get("profile", default_profile().name)
        view = self._view(document, profile_name)
        offset = _offset_at_position(view.text, _position_dict(params.get("position")))
        mapping = _nearest_mapping(view, offset)
        if mapping is None:
            raise InvalidPosition("AI view contains no source-mappable tokens")
        return {
            "uri": uri,
            "version": document.version,
            "sourceRange": _range_at_offsets(
                document.text,
                mapping.source_start,
                mapping.source_end,
            ),
            "kind": mapping.kind,
        }

    def _view_position(self, params: dict) -> dict:
        uri = params.get("textDocument", {}).get("uri")
        if not isinstance(uri, str):
            raise ValueError("aether/viewPosition requires textDocument.uri")
        document = self._document(uri)
        profile_name = params.get("profile", default_profile().name)
        view = self._view(document, profile_name)
        source_offset = _offset_at_position(
            document.text,
            _position_dict(params.get("position")),
        )
        matches = [
            mapping
            for mapping in view.mappings
            if mapping.source_start <= source_offset < mapping.source_end
        ]
        if not matches:
            raise InvalidPosition("Source position does not correspond to an AI-view token")
        mapping = matches[0]
        return {
            "uri": uri,
            "version": document.version,
            "viewRange": _range_at_offsets(
                view.text,
                mapping.view_start,
                mapping.view_end,
            ),
            "kind": mapping.kind,
        }

    def _apply_ai_view_edits(self, params: dict) -> dict:
        item = params.get("textDocument")
        if not isinstance(item, dict) or not isinstance(item.get("uri"), str):
            raise ValueError("aether/applyAIViewEdits requires textDocument.uri")
        uri = item["uri"]
        document = self._document(uri)
        source_version = params.get("version")
        target_version = params.get("newVersion")
        if (
            not isinstance(source_version, int)
            or isinstance(source_version, bool)
            or not isinstance(target_version, int)
            or isinstance(target_version, bool)
        ):
            raise ValueError(
                "aether/applyAIViewEdits requires integer version and newVersion"
            )
        if source_version != document.version:
            raise ValueError(
                f"Stale AI-view edit: expected source version {document.version}, "
                f"received {source_version}"
            )
        if target_version <= source_version:
            raise ValueError("newVersion must be greater than the source version")
        profile_name = params.get("profile", default_profile().name)
        if not isinstance(profile_name, str):
            raise ValueError("profile must be a string")
        edits = params.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ValueError("aether/applyAIViewEdits requires non-empty edits")

        profile = self.profiles.get(profile_name)
        if profile is None:
            raise KeyError(f"Unknown tokenizer profile {profile_name!r}")
        current_view = self._view(document, profile_name)
        edited_view = _apply_text_edits(current_view.text, edits)
        expanded_source = expand_ai_view(edited_view, profile)
        analysis = parse_source(expanded_source)
        candidate = _Document(uri, expanded_source, target_version)
        diagnostics = [
            _diagnostic_to_lsp(expanded_source, diagnostic)
            for diagnostic in analysis.diagnostics
        ]
        if any(diagnostic.severity == "error" for diagnostic in analysis.diagnostics):
            return {
                "accepted": False,
                "uri": uri,
                "sourceVersion": source_version,
                "version": target_version,
                "diagnostics": diagnostics,
            }

        candidate_source = render_module(analysis.ast)
        candidate = _Document(uri, candidate_source, target_version)
        candidate_view = self._view(candidate, profile_name)
        candidate_map = self._map_text(candidate)
        mappings = [
            {
                "sourceRange": _range_at_offsets(
                    candidate_source,
                    mapping.source_start,
                    mapping.source_end,
                ),
                "viewRange": _range_at_offsets(
                    candidate_view.text,
                    mapping.view_start,
                    mapping.view_end,
                ),
                "kind": mapping.kind,
            }
            for mapping in candidate_view.mappings
        ]
        return {
            "accepted": True,
            "uri": uri,
            "sourceVersion": source_version,
            "version": target_version,
            "source": candidate_source,
            "ai": candidate_view.text,
            "map": candidate_map,
            "diagnostics": [
                _diagnostic_to_lsp(candidate_source, diagnostic)
                for diagnostic in self._analysis(candidate).diagnostics
            ],
            "mappings": mappings,
            "workspaceEdit": {
                "documentChanges": [
                    {
                        "textDocument": {
                            "uri": uri,
                            "version": source_version,
                        },
                        "edits": [
                            {
                                "range": _range_at_offsets(
                                    document.text,
                                    0,
                                    len(document.text),
                                ),
                                "newText": candidate_source,
                            }
                        ],
                    }
                ]
            },
        }

    @staticmethod
    def _error(request_id, code: int, message: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    def handle(self, message: dict) -> dict | None:
        """Handle one decoded JSON-RPC request/notification."""
        request_id = message.get("id")
        method = message.get("method")
        if message.get("jsonrpc") != "2.0" or not isinstance(method, str):
            if "id" in message:
                return self._error(request_id, -32600, "Invalid Request")
            return None
        params = message.get("params", {})
        if not isinstance(params, dict):
            if request_id is None:
                print("aether LSP notification error: params must be an object", file=sys.stderr)
                return None
            return self._error(request_id, -32602, "params must be an object")
        try:
            if method == "initialize":
                result = {
                    "capabilities": {
                        "textDocumentSync": {
                            "openClose": True,
                            "change": 2,
                        },
                        "diagnosticProvider": {
                            "interFileDependencies": False,
                            "workspaceDiagnostics": False,
                        },
                    },
                    "serverInfo": {"name": "aether-deterministic", "version": "0.1.0"},
                }
            elif method in {"initialized", "$/setTrace", "$/cancelRequest"}:
                return None
            elif method == "shutdown":
                self.shutdown_requested = True
                result = None
            elif method == "exit":
                self.exit_requested = True
                return None
            elif method == "textDocument/didOpen":
                item = params.get("textDocument", {})
                uri = item.get("uri")
                text = item.get("text")
                version = item.get("version")
                if (
                    not isinstance(uri, str)
                    or not isinstance(text, str)
                    or not isinstance(version, int)
                    or isinstance(version, bool)
                ):
                    raise ValueError("didOpen requires uri, text, and integer version")
                document = _Document(uri, text, version)
                self.documents[uri] = document
                return self._publish_diagnostics(document)
            elif method == "textDocument/didChange":
                uri = params.get("textDocument", {}).get("uri")
                if not isinstance(uri, str):
                    raise ValueError("didChange requires textDocument.uri")
                notification = self._apply_changes(self._document(uri), params)
                if notification is None:
                    return None
                return notification
            elif method == "textDocument/didClose":
                uri = params.get("textDocument", {}).get("uri")
                if not isinstance(uri, str):
                    raise ValueError("didClose requires textDocument.uri")
                self.documents.pop(uri, None)
                return {
                    "jsonrpc": "2.0",
                    "method": "textDocument/publishDiagnostics",
                    "params": {"uri": uri, "diagnostics": []},
                }
            elif method == "textDocument/diagnostic":
                uri = params.get("textDocument", {}).get("uri")
                if not isinstance(uri, str):
                    raise ValueError("diagnostic requires textDocument.uri")
                document = self._document(uri)
                diagnostics = self._analysis(document).diagnostics
                result = {
                    "kind": "full",
                    "items": [
                        _diagnostic_to_lsp(document.text, item) for item in diagnostics
                    ],
                }
            elif method == "aether/getViews":
                result = self._views_result(params)
            elif method == "aether/sourcePosition":
                result = self._source_position(params)
            elif method == "aether/viewPosition":
                result = self._view_position(params)
            elif method == "aether/applyAIViewEdits":
                result = self._apply_ai_view_edits(params)
            elif request_id is not None:
                return self._error(request_id, -32601, f"Method not found: {method}")
            else:
                return None
        except KeyError as error:
            return self._error(request_id, -32002, str(error)) if request_id is not None else None
        except (ValueError, ProfileError) as error:
            if request_id is None:
                print(f"aether LSP notification error: {error}", file=sys.stderr)
            return self._error(request_id, -32602, str(error)) if request_id is not None else None
        if request_id is None:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}


def read_message(stream: BinaryIO) -> dict | None:
    """Read one Content-Length framed JSON-RPC message."""
    header_bytes = 0
    headers: dict[str, str] = {}
    while True:
        line = stream.readline(_MAX_HEADER_BYTES + 1)
        if line == b"" and not headers:
            return None
        header_bytes += len(line)
        if header_bytes > _MAX_HEADER_BYTES:
            raise LspProtocolError("LSP header exceeds size limit")
        if not line:
            raise LspProtocolError("Unexpected EOF in LSP header")
        if line in {b"\r\n", b"\n"}:
            break
        try:
            name, value = line.decode("ascii").split(":", 1)
        except (UnicodeDecodeError, ValueError) as error:
            raise LspProtocolError("Malformed LSP header") from error
        normalized = name.strip().lower()
        if normalized in headers:
            raise LspProtocolError(f"Duplicate LSP header {name}")
        headers[normalized] = value.strip()
    length_value = headers.get("content-length")
    if length_value is None or not length_value.isdecimal():
        raise LspProtocolError("Missing or invalid Content-Length")
    length = int(length_value)
    if length > _MAX_MESSAGE_BYTES:
        raise LspProtocolError("LSP message exceeds size limit")
    chunks = []
    remaining = length
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise LspProtocolError("Unexpected EOF in LSP payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    try:
        message = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LspProtocolError(f"Invalid JSON-RPC payload: {error}") from error
    if not isinstance(message, dict):
        raise LspProtocolError("JSON-RPC payload must be an object")
    return message


def write_message(stream: BinaryIO, message: dict) -> None:
    """Write one UTF-8 Content-Length framed JSON-RPC message."""
    payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    frame = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii") + payload
    stream.write(frame)
    stream.flush()


def serve(input_stream: BinaryIO, output_stream: BinaryIO) -> None:
    """Serve LSP messages until EOF or an `exit` notification."""
    server = AetherLanguageServer()
    while not server.exit_requested:
        message = read_message(input_stream)
        if message is None:
            return
        response = server.handle(message)
        if response is not None:
            write_message(output_stream, response)
        if server.shutdown_requested and server.exit_requested:
            return


def main() -> None:
    try:
        serve(sys.stdin.buffer, sys.stdout.buffer)
    except LspProtocolError as error:
        print(f"aether LSP protocol error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
