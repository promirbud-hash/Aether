from io import BytesIO
import unittest

from src.lsp import (
    AetherLanguageServer,
    InvalidPosition,
    _offset_at_position,
    read_message,
    serve,
    write_message,
)


class LspTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = AetherLanguageServer()
        self.uri = "file:///workspace/sensor.hl"

    def open(self, text: str, version: int = 1) -> dict:
        return self.server.handle(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": self.uri,
                        "languageId": "aether",
                        "version": version,
                        "text": text,
                    }
                },
            }
        )

    def test_initialize_advertises_incremental_sync_and_pull_diagnostics(self) -> None:
        response = self.server.handle(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )

        self.assertEqual(response["result"]["capabilities"]["textDocumentSync"]["change"], 2)
        self.assertIn("diagnosticProvider", response["result"]["capabilities"])

    def test_open_reports_diagnostics_and_full_diagnostic_request(self) -> None:
        notification = self.open("fn broken() { let = 1; }\n")
        diagnostics = notification["params"]["diagnostics"]
        response = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "textDocument/diagnostic",
                "params": {"textDocument": {"uri": self.uri}},
            }
        )

        self.assertTrue(any(item["code"] == "E_SYNTAX" for item in diagnostics))
        self.assertEqual(response["result"]["kind"], "full")
        self.assertEqual(response["result"]["items"], diagnostics)

    def test_get_views_includes_versioned_source_mapping_and_map(self) -> None:
        source = "fn identity(value: i32) -> i32 { return value; }\n"
        self.open(source)
        response = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "aether/getViews",
                "params": {
                    "textDocument": {"uri": self.uri},
                    "activeNode": "f:identity",
                },
            }
        )
        result = response["result"]

        self.assertEqual(result["version"], 1)
        self.assertIn("@map1", result["map"])
        self.assertIn("F identity", result["active"])
        self.assertTrue(result["mappings"])
        self.assertEqual(self.server.encoders, {})
        token_mapping = next(item for item in result["mappings"] if item["kind"] == "token")
        self.assertIn("sourceRange", token_mapping)
        self.assertIn("viewRange", token_mapping)

    def test_view_and_source_positions_roundtrip_for_token(self) -> None:
        source = "fn identity(value: i32) -> i32 { return value; }\n"
        self.open(source)
        view_result = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "aether/getViews",
                "params": {"textDocument": {"uri": self.uri}},
            }
        )["result"]
        mapping = next(
            item
            for item in view_result["mappings"]
            if item["kind"] == "token"
            and item["sourceRange"]["start"]["line"] == 0
        )
        source_result = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "aether/sourcePosition",
                "params": {
                    "textDocument": {"uri": self.uri},
                    "position": mapping["viewRange"]["start"],
                },
            }
        )
        view_position = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "aether/viewPosition",
                "params": {
                    "textDocument": {"uri": self.uri},
                    "position": mapping["sourceRange"]["start"],
                },
            }
        )

        self.assertEqual(source_result["result"]["sourceRange"], mapping["sourceRange"])
        self.assertEqual(view_position["result"]["viewRange"], mapping["viewRange"])

    def test_apply_ai_view_edits_returns_versioned_source_edit_and_new_mappings(self) -> None:
        self.open("fn answer() -> i32 { return 1; }\n")
        view = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 20,
                "method": "aether/getViews",
                "params": {"textDocument": {"uri": self.uri}},
            }
        )["result"]["ai"]
        start = view.index("1")
        response = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 21,
                "method": "aether/applyAIViewEdits",
                "params": {
                    "textDocument": {"uri": self.uri},
                    "version": 1,
                    "newVersion": 2,
                    "edits": [
                        {
                            "range": {
                                "start": {"line": 0, "character": start},
                                "end": {"line": 0, "character": start + 1},
                            },
                            "newText": "2",
                        }
                    ],
                },
            }
        )["result"]

        self.assertTrue(response["accepted"])
        self.assertEqual(response["version"], 2)
        document_edit = response["workspaceEdit"]["documentChanges"][0]
        self.assertEqual(document_edit["textDocument"]["version"], 1)
        self.assertEqual(
            document_edit["edits"][0]["newText"],
            "fn answer() -> i32 {\n    return 2;\n}\n",
        )
        self.assertTrue(response["mappings"])
        self.assertEqual(self.server.documents[self.uri].version, 1)

        notification = self.server.handle(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": {"uri": self.uri, "version": 2},
                    "contentChanges": [{"text": document_edit["edits"][0]["newText"]}],
                },
            }
        )
        self.assertEqual(notification["params"]["version"], 2)

    def test_apply_ai_view_edits_rejects_stale_version_and_bad_syntax(self) -> None:
        self.open("fn answer() -> i32 { return 1; }\n")
        result = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 22,
                "method": "aether/applyAIViewEdits",
                "params": {
                    "textDocument": {"uri": self.uri},
                    "version": 0,
                    "newVersion": 2,
                    "edits": [],
                },
            }
        )
        self.assertIn("Stale AI-view edit", result["error"]["message"])

        view = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 23,
                "method": "aether/getViews",
                "params": {"textDocument": {"uri": self.uri}},
            }
        )["result"]["ai"]
        body_start = view.index(")") + 1
        start = view.index("ret", body_start)
        invalid = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 24,
                "method": "aether/applyAIViewEdits",
                "params": {
                    "textDocument": {"uri": self.uri},
                    "version": 1,
                    "newVersion": 2,
                    "edits": [
                        {
                            "range": {
                                "start": {"line": 0, "character": start},
                                "end": {"line": 0, "character": start + len("ret")},
                            },
                            "newText": "invalid",
                        }
                    ],
                },
            }
        )["result"]
        self.assertFalse(invalid["accepted"])
        self.assertTrue(invalid["diagnostics"])

    def test_apply_ai_view_edits_rejects_out_of_range_edits(self) -> None:
        self.open("fn answer() -> i32 { return 1; }\n")
        view = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 25,
                "method": "aether/getViews",
                "params": {"textDocument": {"uri": self.uri}},
            }
        )["result"]["ai"]
        response = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 26,
                "method": "aether/applyAIViewEdits",
                "params": {
                    "textDocument": {"uri": self.uri},
                    "version": 1,
                    "newVersion": 2,
                    "edits": [
                        {
                            "range": {
                                "start": {"line": 0, "character": len(view) + 1},
                                "end": {"line": 0, "character": len(view) + 2},
                            },
                            "newText": "x",
                        }
                    ],
                },
            }
        )
        self.assertIn("outside line", response["error"]["message"])

    def test_incremental_change_invalidates_only_modified_document_cache(self) -> None:
        other_uri = "file:///workspace/other.hl"
        self.open("fn first() {}\n")
        self.server.handle(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": other_uri,
                        "version": 1,
                        "text": "fn second() {}\n",
                    }
                },
            }
        )
        first = self.server.documents[self.uri]
        other = self.server.documents[other_uri]
        self.server._analysis(first)
        first_analysis = first.analysis
        other_analysis = self.server._analysis(other)

        notification = self.server.handle(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": {"uri": self.uri, "version": 2},
                    "contentChanges": [
                        {
                            "range": {
                                "start": {"line": 0, "character": 3},
                                "end": {"line": 0, "character": 8},
                            },
                            "rangeLength": 5,
                            "text": "third",
                        }
                    ],
                },
            }
        )

        self.assertEqual(first.text, "fn third() {}\n")
        self.assertIsNot(first.analysis, first_analysis)
        self.assertIs(other.analysis, other_analysis)
        self.assertEqual(notification["params"]["version"], 2)

    def test_stale_document_versions_are_ignored(self) -> None:
        self.open("fn original() {}\n", 4)
        result = self.server.handle(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": {"uri": self.uri, "version": 3},
                    "contentChanges": [{"text": "fn stale() {}\n"}],
                },
            }
        )

        self.assertIsNone(result)
        self.assertEqual(self.server.documents[self.uri].text, "fn original() {}\n")

    def test_utf16_positions_handle_astral_characters_and_crlf(self) -> None:
        source = "// 😀\r\nfn f() {}\r\n"

        self.assertEqual(_offset_at_position(source, {"line": 0, "character": 5}), 4)
        self.assertEqual(_offset_at_position(source, {"line": 1, "character": 3}), source.index("f()"))
        with self.assertRaisesRegex(InvalidPosition, "surrogate"):
            _offset_at_position(source, {"line": 0, "character": 4})

    def test_json_rpc_frames_roundtrip_utf8_and_content_length(self) -> None:
        output = BytesIO()
        message = {"jsonrpc": "2.0", "method": "test", "params": {"text": "ą😀"}}
        write_message(output, message)
        frame = output.getvalue()
        header, payload = frame.split(b"\r\n\r\n", 1)
        self.assertIn(f"Content-Length: {len(payload)}".encode("ascii"), header)
        self.assertEqual(read_message(BytesIO(frame)), message)
        self.assertIsNone(read_message(BytesIO(b"")))

    def test_stdio_server_serves_lifecycle_and_notification_frames(self) -> None:
        messages = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": self.uri,
                        "version": 1,
                        "text": "fn identity(value: i32) { return value; }\n",
                    }
                },
            },
            {"jsonrpc": "2.0", "id": 2, "method": "shutdown"},
            {"jsonrpc": "2.0", "method": "exit"},
        ]
        input_stream = BytesIO()
        for message in messages:
            write_message(input_stream, message)
        input_stream.seek(0)
        output_stream = BytesIO()

        serve(input_stream, output_stream)

        output_stream.seek(0)
        responses = []
        while (response := read_message(output_stream)) is not None:
            responses.append(response)
        self.assertEqual(responses[0]["id"], 1)
        self.assertEqual(responses[1]["method"], "textDocument/publishDiagnostics")
        self.assertEqual(responses[2], {"jsonrpc": "2.0", "id": 2, "result": None})

    def test_unknown_profile_is_returned_as_rpc_error(self) -> None:
        response = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "aether/getViews",
                "params": {"textDocument": {"uri": self.uri}, "profile": "missing"},
            }
        )

        self.assertEqual(response["error"]["code"], -32002)

    def test_profile_switch_changes_view_without_loading_tokenizer(self) -> None:
        self.open(
            "type SensorFrame { voltage: f32, }\n"
            "fn normalize_voltage(frame: SensorFrame) { return frame; }\n"
        )
        first = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "aether/getViews",
                "params": {
                    "textDocument": {"uri": self.uri},
                    "profile": "qwen2.5-coder",
                },
            }
        )["result"]["ai"]
        second = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "aether/getViews",
                "params": {
                    "textDocument": {"uri": self.uri},
                    "profile": "llama3.1",
                },
            }
        )["result"]["ai"]

        self.assertNotEqual(first, second)
        self.assertEqual(self.server.encoders, {})


if __name__ == "__main__":
    unittest.main()
