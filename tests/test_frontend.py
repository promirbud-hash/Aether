from pathlib import Path
import random
import unittest

from src.ai_view import generate_ai_view
from src.codemap import build_codemap, render_map
from src.frontend import (
    AssignmentStmt,
    ForStmt,
    FunctionDecl,
    IfStmt,
    LetStmt,
    LoopControlStmt,
    ReturnStmt,
    TypeDecl,
    WhileStmt,
    parse_source,
    render_module,
)
from src.parser import parse_tokens, render, tokenize


ROOT = Path(__file__).parents[1]


class FrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "examples" / "sensor.hl").open(
            "r", encoding="utf-8", newline=""
        ) as source_file:
            cls.source = source_file.read()

    def test_sample_produces_clean_ast_and_syntax_cst(self) -> None:
        result = parse_source(self.source)

        self.assertEqual(result.diagnostics, ())
        self.assertIsInstance(result.ast.declarations[0], TypeDecl)
        function = result.ast.declarations[1]
        self.assertIsInstance(function, FunctionDecl)
        self.assertEqual(function.attributes[0].name, "io")
        self.assertIsInstance(function.body[0], LetStmt)
        self.assertIsInstance(function.body[1], ReturnStmt)
        self.assertEqual(render(result.cst), self.source)
        self.assertTrue(any(node.kind == "type_declaration" for node in result.cst.children))
        self.assertTrue(
            any(node.kind == "function_declaration" for node in result.cst.children)
        )

    def test_parser_recovers_and_keeps_following_declarations(self) -> None:
        source = "fn broken() { let = 1; }\nfn good() { return 1; }\n"
        result = parse_source(source)

        self.assertEqual(
            [declaration.name for declaration in result.ast.declarations],
            ["broken", "good"],
        )
        self.assertTrue(any(item.code == "E_SYNTAX" for item in result.diagnostics))
        self.assertEqual(render(result.cst), source)

    def test_semantic_diagnostics_report_duplicates_unknown_names_and_attributes(self) -> None:
        source = (
            "@mystery\n"
            "fn f(value: i32, value: i32) {\n"
            "  let result = missing(value, ghost);\n"
            "}\n"
            "fn f() {}\n"
        )
        result = parse_source(source)
        codes = [item.code for item in result.diagnostics]

        self.assertIn("E_UNKNOWN_ATTRIBUTE", codes)
        self.assertIn("E_DUPLICATE_DECL", codes)
        self.assertIn("E_DUPLICATE_BINDING", codes)
        self.assertIn("E_UNKNOWN_CALL", codes)
        self.assertIn("E_UNKNOWN_NAME", codes)
        self.assertEqual(len([item for item in result.diagnostics if item.code == "E_UNKNOWN_CALL"]), 1)

    def test_unknown_type_and_conflicting_effects_are_diagnosed(self) -> None:
        result = parse_source("@pure\n@io\nfn f(value: Mystery) {}\n")
        codes = [item.code for item in result.diagnostics]

        self.assertIn("E_UNKNOWN_TYPE", codes)
        self.assertIn("E_CONFLICTING_EFFECTS", codes)

    def test_target_attribute_accepts_one_mcu_target(self) -> None:
        self.assertEqual(
            parse_source("@target(mcu)\nfn main() -> i32 { return 0; }\n").diagnostics,
            (),
        )
        invalid = parse_source(
            "@target(host)\nfn main() -> i32 { return 0; }\n"
            "@target(mcu)\nfn helper() {}\n"
        )
        self.assertEqual(
            sum(item.code == "E_TARGET_ATTRIBUTE" for item in invalid.diagnostics),
            2,
        )

    def test_expression_precedence_and_pipeline_shape(self) -> None:
        result = parse_source(
            "fn f(a: i32, b: i32, c: i32) { return a |> scale(b * c + 1); }\n"
        )
        function = result.ast.declarations[0]
        expression = function.body[0].value

        self.assertEqual(expression.kind, "binary")
        self.assertEqual(expression.value, "|>")
        call = expression.children[1]
        self.assertEqual(call.kind, "call")
        addition = call.children[1]
        self.assertEqual(addition.kind, "binary")
        self.assertEqual(addition.value, "+")
        self.assertEqual(addition.children[0].value, "*")
        self.assertEqual(result.diagnostics, ())

    def test_bitwise_precedence_and_base_integer_literals(self) -> None:
        source = (
            "fn value(input: i32) -> i32 {\n"
            "  return input | 0xFF & 0b1010_0001 ^ 0x1 << 2;\n"
            "}\n"
        )
        result = parse_source(source)
        expression = result.ast.declarations[0].body[0].value

        self.assertEqual(result.diagnostics, ())
        self.assertEqual(expression.value, "|")
        self.assertEqual(expression.children[1].value, "^")
        self.assertEqual(expression.children[1].children[0].value, "&")
        self.assertEqual(expression.children[1].children[1].value, "<<")
        self.assertEqual(
            parse_source(render_module(result.ast)).diagnostics,
            (),
        )

    def test_control_flow_roundtrips_with_nested_blocks(self) -> None:
        source = (
            "fn count(limit: i32) -> i32 {\n"
            "  var total = 0;\n"
            "  for index in 0..limit {\n"
            "    if index == 2 { continue; }\n"
            "    while false { break; }\n"
            "    total += index;\n"
            "  }\n"
            "  return total;\n"
            "}\n"
        )
        parsed = parse_source(source)
        function = parsed.ast.declarations[0]

        self.assertEqual(parsed.diagnostics, ())
        self.assertIsInstance(function.body[1], ForStmt)
        self.assertIsInstance(function.body[1].body[0], IfStmt)
        self.assertIsInstance(function.body[1].body[0].body[0], LoopControlStmt)
        self.assertIsInstance(function.body[1].body[1], WhileStmt)
        self.assertEqual(parse_source(render_module(parsed.ast)).diagnostics, ())

    def test_control_flow_rejects_bad_conditions_ranges_and_loop_control(self) -> None:
        parsed = parse_source(
            "fn invalid() -> i32 {\n"
            "  break;\n"
            "  if 1 { return 1; }\n"
            "  for index in 0..1.5 { return index; }\n"
            "}\n"
            "fn escaped_local() -> i32 {\n"
            "  if true { let local = 1; }\n"
            "  return local;\n"
            "}\n"
        )
        codes = {diagnostic.code for diagnostic in parsed.diagnostics}

        self.assertIn("E_LOOP_CONTROL_CONTEXT", codes)
        self.assertIn("E_CONDITION_TYPE", codes)
        self.assertIn("E_RANGE_TYPE", codes)
        self.assertIn("E_RETURN_PATH", codes)
        self.assertIn("E_UNKNOWN_NAME", codes)
        self.assertEqual(
            sum(
                diagnostic.code == "E_LOOP_CONTROL_CONTEXT"
                for diagnostic in parsed.diagnostics
            ),
            1,
        )

    def test_var_rebinding_parses_and_renders_as_assignment(self) -> None:
        source = "fn update() -> i32 { var value = 1; value = value + 1; return value; }\n"
        result = parse_source(source)
        function = result.ast.declarations[0]

        self.assertEqual(result.diagnostics, ())
        self.assertIsInstance(function.body[1], AssignmentStmt)
        canonical = render_module(result.ast)
        self.assertIn("value = value + 1;", canonical)
        self.assertEqual(parse_source(canonical).diagnostics, ())

    def test_compound_assignment_parses_and_renders(self) -> None:
        source = (
            "fn update() -> i32 {\n"
            "  var value = 1;\n"
            "  value += 2;\n"
            "  value -= 1;\n"
            "  value *= 3;\n"
            "  value /= 2;\n"
            "  return value;\n"
            "}\n"
        )
        result = parse_source(source)
        function = result.ast.declarations[0]

        self.assertEqual(result.diagnostics, ())
        assignments = function.body[1:5]
        self.assertTrue(all(isinstance(item, AssignmentStmt) for item in assignments))
        self.assertEqual(
            [item.operator for item in assignments],
            ["+=", "-=", "*=", "/="],
        )
        self.assertIn("value += 2;", render_module(result.ast))

    def test_ast_rendering_is_parseable_and_view_generation_is_deterministic(self) -> None:
        first = parse_source(self.source)
        canonical = render_module(first.ast)
        reparsed = parse_source(canonical)

        self.assertEqual(reparsed.diagnostics, ())
        self.assertEqual(
            generate_ai_view(first.ast),
            generate_ai_view(reparsed.ast),
        )

    def test_unit_type_expressions_roundtrip_in_scalar_and_tensor_types(self) -> None:
        source = (
            "fn slope(value: Tensor<f32, [4], V/mm>) -> f32<V/mm> {\n"
            "  return sum(value);\n"
            "}\n"
            "fn wavelength(value: f32<µm>) -> f32<um> { return value; }\n"
            "fn inverse_time(value: f64<1/ms>) -> f64<kHz> { return value; }\n"
        )
        result = parse_source(source)
        canonical = render_module(result.ast)

        self.assertEqual(result.diagnostics, ())
        self.assertIn("Tensor<f32,[4],V/mm>", canonical)
        self.assertIn("-> f32<V/mm>", canonical)
        self.assertEqual(parse_source(canonical).diagnostics, ())
        codemap = render_map(build_codemap(result.ast))
        self.assertIn("value:f32<um>", codemap)

    def test_codemap_from_ast_uses_declarations_and_calls(self) -> None:
        result = parse_source(self.source)
        text = render_map(build_codemap(result.ast))

        self.assertTrue(text.startswith("@map1\n"))
        self.assertIn("F normalize_voltage (frame:SensorFrame)->Tensor<f32,[4,4]> @io", text)
        self.assertIn("T SensorFrame", text)
        self.assertIn("C normalize_voltage>max_abs", text)
        self.assertIn("C normalize_voltage>mean", text)
        self.assertIn("C normalize_voltage>scale", text)
        self.assertIn("C normalize_voltage>subtract", text)
        self.assertIn("R normalize_voltage>SensorFrame", text)

    def test_codemap_marks_unannotated_functions_as_pure(self) -> None:
        module = parse_source("fn identity(value: i32) { return value; }\n").ast
        text = render_map(build_codemap(module))

        self.assertIn("F identity (value:i32) @pure", text)

    def test_crlf_source_roundtrips_without_newline_normalization(self) -> None:
        source = "type T {\r\n    value: i32,\r\n}\r\n\r\nfn f() {}\r\n"
        result = parse_source(source)

        self.assertEqual(render(result.cst), source)
        self.assertEqual(result.diagnostics, ())

    def test_seeded_source_variants_roundtrip_losslessly(self) -> None:
        generator = random.Random(20261106)
        for index in range(40):
            newline = generator.choice(("\n", "\r\n"))
            name = f"sensor_{index}"
            factor = generator.choice(("2.0", "0.5", "1.25"))
            source = newline.join(
                (
                    f"// generated case {index}",
                    f"fn {name}(input: f32) -> f32 {{",
                    f"    let output = input |> scale({factor});",
                    "    return output;",
                    "}",
                    "",
                )
            )
            result = parse_source(source)

            self.assertEqual(render(result.cst), source)
            self.assertEqual(result.diagnostics, ())

    def test_malformed_delimiters_are_recoverable_frontend_diagnostics(self) -> None:
        result = parse_source("fn broken() { return (1; }\nfn good() {}\n")

        self.assertTrue(any(item.code == "E_SYNTAX" for item in result.diagnostics))
        self.assertIn("good", {item.name for item in result.ast.declarations})

    def test_extern_c_declaration_roundtrips_through_ast(self) -> None:
        source = (
            '@ffi\n'
            '@link("sensor")\n'
            'extern "C" fn read_channel(channel: u32) -> i32;\n'
            "@ffi\n"
            '@link("sensor")\n'
            'extern "C" fn reset() -> void;\n'
        )
        parsed = parse_source(source)

        self.assertEqual(parsed.diagnostics, ())
        first = parsed.ast.declarations[0]
        self.assertEqual(first.external_abi, "C")
        self.assertEqual(first.external_library, "sensor")
        self.assertEqual(first.body, ())
        canonical = render_module(parsed.ast)
        reparsed = parse_source(canonical)
        self.assertEqual(reparsed.diagnostics, ())
        self.assertEqual(render_module(reparsed.ast), canonical)

    def test_struct_initializer_and_field_assignment_roundtrip(self) -> None:
        source = (
            "type Point { x: i32, y: i32, }\n"
            "fn update(point: Point) -> Point {\n"
            "  var copy = point;\n"
            "  copy.x = 2;\n"
            "  return Point { y: copy.y, x: copy.x };\n"
            "}\n"
        )
        parsed = parse_source(source)
        canonical = render_module(parsed.ast)
        reparsed = parse_source(canonical)

        self.assertEqual(parsed.diagnostics, ())
        self.assertEqual(reparsed.diagnostics, ())
        self.assertEqual(render_module(reparsed.ast), canonical)

    def test_collection_literal_and_types_roundtrip(self) -> None:
        source = (
            "fn values() -> Array<i32, [2]> {\n"
            "  let result: Array<i32, [2]> = [4, 5];\n"
            "  return result;\n"
            "}\n"
        )
        parsed = parse_source(source)
        canonical = render_module(parsed.ast)
        reparsed = parse_source(canonical)

        self.assertEqual(parsed.diagnostics, ())
        self.assertEqual(reparsed.diagnostics, ())
        self.assertEqual(render_module(reparsed.ast), canonical)

    def test_invalid_extern_syntax_is_rejected(self) -> None:
        wrong_abi = parse_source(
            '@ffi\n@link("sensor")\nextern "python" fn read() -> i32;\n'
        )
        missing_semicolon = parse_source(
            '@ffi\n@link("sensor")\nextern "C" fn read() -> i32\n'
        )

        self.assertIn("E_SYNTAX", [item.code for item in wrong_abi.diagnostics])
        self.assertIn("E_SYNTAX", [item.code for item in missing_semicolon.diagnostics])

    def test_codemap_marks_extern_symbol_and_call_edge(self) -> None:
        source = (
            '@ffi\n@link("sensor")\nextern "C" fn read() -> i32;\n'
            "@ffi\n"
            "fn sample() -> i32 { return read(); }\n"
        )
        parsed = parse_source(source)
        ast_map = build_codemap(parsed.ast)
        cst_map = build_codemap(parse_tokens(source, tokenize(source)))
        text = render_map(ast_map)
        cst_text = render_map(cst_map)

        self.assertEqual(parsed.diagnostics, ())
        self.assertEqual(cst_map, ast_map)
        self.assertIn("X read", text)
        self.assertIn("C sample>read", text)
        self.assertIn("X read", cst_text)
        self.assertIn("C sample>read", cst_text)

    def test_codemap_ast_and_cst_inputs_match_for_nested_calls_and_purity(self) -> None:
        source = (
            "fn helper(value: i32) -> i32 { return value; }\n"
            "fn caller(value: i32) -> i32 {\n"
            "  let nested = helper(value + helper(1));\n"
            "  return helper(nested);\n"
            "}\n"
        )
        parsed = parse_source(source)
        self.assertEqual(parsed.diagnostics, ())

        ast_map = build_codemap(parsed.ast)
        cst_map = build_codemap(parse_tokens(source, tokenize(source)))
        syntax_cst_map = build_codemap(parsed.cst)

        self.assertEqual(cst_map, ast_map)
        self.assertEqual(syntax_cst_map, ast_map)
        self.assertEqual(render_map(cst_map), render_map(ast_map))


if __name__ == "__main__":
    unittest.main()
