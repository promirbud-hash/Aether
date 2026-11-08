from pathlib import Path
import re
import tempfile
import unittest

from src.ai_view import (
    TokenizerProfile,
    ai_view_to_source,
    create_encoder,
    default_profile,
    expand_ai_view,
    generate_ai_view,
    load_profiles,
)
from src.benchmark import (
    benchmark,
    benchmark_profiles,
    count_tokens,
    format_profile_report,
    savings_percent,
)
from src.codemap import (
    build_codemap,
    render_active_slice,
    render_map,
)
from src.frontend import parse_source, render_module
from src.parser import parse


ROOT = Path(__file__).parents[1]


class ViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (ROOT / "examples" / "sensor.hl").read_text(encoding="utf-8")
        cls.tree = parse(cls.source)

    def test_ai_view_compacts_whitespace_and_uses_semantic_aliases(self) -> None:
        view = generate_ai_view(self.tree)

        self.assertIn("@a(", view)
        self.assertIn("≫=|>", view)
        self.assertNotIn("// Sensor data", view)
        self.assertNotIn("\n", view)

    def test_ai_view_aliases_expand_to_canonical_source(self) -> None:
        source = (
            "fn measure(value: i32) -> i32 {\n"
            "  let centered = value;\n"
            "  return centered + centered;\n"
            "}\n"
        )
        profile = default_profile()
        view = generate_ai_view(parse(source), profile)

        self.assertIn("centered", view.split(")", 1)[0])
        self.assertEqual(
            ai_view_to_source(view, profile),
            "fn measure(value: i32) -> i32 {\n"
            "    let centered = value;\n"
            "    return centered + centered;\n"
            "}\n",
        )

    def test_sample_ai_view_roundtrips_to_canonical_source(self) -> None:
        expected = render_module(parse_source(self.source).ast)
        view = generate_ai_view(self.tree)

        self.assertEqual(ai_view_to_source(view), expected)

    def test_ai_view_rejects_missing_stale_and_duplicate_alias_headers(self) -> None:
        profile = default_profile()
        with self.assertRaisesRegex(ValueError, "missing"):
            expand_ai_view("fn f() {}", profile)
        with self.assertRaisesRegex(ValueError, "does not match"):
            expand_ai_view("@a(other-profile;)fn f(){}", profile)
        with self.assertRaisesRegex(ValueError, "duplicate alias"):
            expand_ai_view(
                f"@a({profile.name};rtn=return,rtn=return)ret;",
                profile,
            )

    def test_codemap_contains_declarations_permissions_units_and_calls(self) -> None:
        code_map = build_codemap(self.tree)
        text = render_map(code_map)

        self.assertIn("F normalize_voltage", text)
        self.assertIn("@io", text)
        self.assertIn("T SensorFrame", text)
        self.assertIn("u:V", text)
        self.assertIn("C normalize_voltage>mean", text)
        self.assertIn("R normalize_voltage>SensorFrame", text)

    def test_active_slice_includes_node_and_adjacent_nodes(self) -> None:
        code_map = build_codemap(self.tree)
        active = render_active_slice(code_map, "f:normalize_voltage")

        self.assertIn("F normalize_voltage", active)
        self.assertIn("T SensorFrame", active)
        self.assertIn("C normalize_voltage>", active)
        self.assertNotIn("X mean", active)
        self.assertNotIn("Unknown active node", active)

    def test_unknown_active_node_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown active node"):
            render_active_slice(build_codemap(self.tree), "f:missing")

    def test_active_type_slice_includes_incoming_type_relationship(self) -> None:
        active = render_active_slice(
            build_codemap(self.tree),
            "t:SensorFrame",
        )

        self.assertIn("T SensorFrame", active)
        self.assertIn("F normalize_voltage", active)
        self.assertIn("R normalize_voltage>SensorFrame", active)

    def test_benchmark_uses_exact_encoder_counts(self) -> None:
        class FakeEncoder:
            def encode(self, text: str) -> list[int]:
                return list(text.encode("utf-8"))

        results = benchmark("python", "human", "compact", "slice", FakeEncoder())

        self.assertEqual([count for _, count, _ in results], [6, 5, 7, 5])
        self.assertEqual(savings_percent(10, 7), 30.0)
        self.assertEqual(count_tokens("abc", FakeEncoder()), 3)

    def test_profiles_load_and_use_distinct_operator_dictionaries(self) -> None:
        profiles = {profile.name: profile for profile in load_profiles()}
        qwen = generate_ai_view(self.tree, profiles["qwen2.5-coder"])
        llama = generate_ai_view(self.tree, profiles["llama3.1"])

        self.assertIn("⊳=|>", qwen)
        self.assertIn("⟫=|>", llama)
        self.assertNotEqual(qwen, llama)

    def test_encoder_optimizes_alias_selection_by_full_view_cost(self) -> None:
        class MergeAwareEncoder:
            def encode(self, text: str) -> list[int]:
                pieces = re.findall(
                    r"repeated_identifier|\$\d+|[A-Za-z_]+|\d+|[^\w\s]",
                    text,
                )
                return [
                    0
                    for piece in pieces
                    for _ in range(30 if piece == "repeated_identifier" else 1)
                ]

        profile = TokenizerProfile(
            name="test-profile",
            backend="tiktoken",
            tokenizer="unused",
            operator_aliases={},
            idiom_aliases={},
            semantic_aliases={"repeated_identifier": ("$1",)},
            alias_candidates=("q",),
        )
        tree = parse(
            "fn repeated_identifier(input: i32) { return repeated_identifier(input); }"
        )
        encoded_view = generate_ai_view(tree, profile, MergeAwareEncoder())
        fallback_view = generate_ai_view(tree, profile)

        self.assertIn("$1=repeated_identifier", encoded_view)
        self.assertNotEqual(encoded_view, fallback_view)
        self.assertLess(
            len(MergeAwareEncoder().encode(encoded_view)),
            len(MergeAwareEncoder().encode(fallback_view)),
        )

    def test_profile_optimizer_selects_best_operator_spelling(self) -> None:
        class OperatorEncoder:
            def encode(self, text: str) -> list[int]:
                pieces = re.findall(r"\|>|Z|Y|.", text)
                return [
                    0
                    for piece in pieces
                    for _ in range({"|>": 10, "Y": 3}.get(piece, 1))
                ]

        profile = TokenizerProfile(
            name="operator-test",
            backend="tiktoken",
            tokenizer="unused",
            operator_aliases={"|>": ("Y", "Z")},
            idiom_aliases={},
            semantic_aliases={},
            alias_candidates=("$1",),
        )
        tree = parse("a |> a |> a |> a |> a |> a")
        encoder = OperatorEncoder()
        view = generate_ai_view(tree, profile, encoder)

        self.assertIn("Z=|>", view)
        self.assertLess(len(encoder.encode(view)), len(encoder.encode("a |> a |> a |> a |> a |> a")))

    def test_ai_view_spacing_uses_aliased_token_classes(self) -> None:
        profile = TokenizerProfile(
            name="class-changing-alias",
            backend="tiktoken",
            tokenizer="unused",
            operator_aliases={"|>": ("pipeword",)},
            idiom_aliases={},
            semantic_aliases={},
            alias_candidates=("$1",),
        )
        tree = parse("left |> right")

        view = generate_ai_view(tree, profile)

        self.assertIn("left pipeword right", view)

    def test_benchmark_runs_independent_tokenizer_profiles(self) -> None:
        profiles = load_profiles()

        class ByteEncoder:
            def encode(self, text: str) -> list[int]:
                return list(text.encode("utf-8"))

        encoders = {profile.name: ByteEncoder() for profile in profiles}
        results = benchmark_profiles(
            profiles,
            "python",
            self.source,
            "map",
            self.tree,
            encoders,
        )

        self.assertEqual(len(results), len(profiles))
        self.assertTrue(all(result.counts for result in results))
        report = format_profile_report(results)
        self.assertIn("qwen2.5-coder", report)
        self.assertIn("llama3.1", report)
        self.assertIn("claude3-o200k-proxy", report)
        self.assertIn("not Claude's tokenizer", report)

    def test_profile_json_rejects_invalid_alias_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(
                '{"name":"bad","tokenizer":{"backend":"tiktoken","name":"x"},'
                '"operators":{},"idioms":{},"alias_candidates":[]}',
                encoding="utf-8",
            )
            from src.ai_view import ProfileError, load_profile

            with self.assertRaisesRegex(ProfileError, "alias_candidates"):
                load_profile(path)

    def test_real_bpe_encoder_is_available(self) -> None:
        self.assertGreater(count_tokens(self.source), 0)

    def test_ai_and_active_graph_views_reduce_sample_token_count(self) -> None:
        code_map = build_codemap(self.tree)
        human_tokens = count_tokens(self.source)
        profile = default_profile()
        encoder = create_encoder(profile)
        ai_tokens = count_tokens(generate_ai_view(self.tree, profile, encoder), encoder)
        graph_tokens = count_tokens(render_active_slice(code_map, "f:normalize_voltage"))

        self.assertLess(ai_tokens, human_tokens)
        self.assertLess(graph_tokens, human_tokens)


if __name__ == "__main__":
    unittest.main()
