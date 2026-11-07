import unittest
from fractions import Fraction

from src.api import CompilationError, compile_source
from src.frontend import _BUILTIN_CALLS, parse_source, render_module
from src.semantic import analyze


class SemanticTests(unittest.TestCase):
    def test_frame_buffer_containing_locals_cannot_be_const_or_annotated(self) -> None:
        result = parse_source(
            "fn direct_const(size: u64) {\n"
            "  const bytes = frame_buffer(size);\n"
            "}\n"
            "fn direct_annotation(size: u64) {\n"
            "  let bytes: FrameBuffer = frame_buffer(size);\n"
            "}\n"
            "fn nested_const(size: u64) -> Result<FrameBuffer, IoError> {\n"
            "  const result = Ok(frame_buffer(size));\n"
            "  return result;\n"
            "}\n"
            "fn nested_annotation(size: u64) -> Result<FrameBuffer, IoError> {\n"
            "  let result: Result<FrameBuffer, IoError> = Ok(frame_buffer(size));\n"
            "  return result;\n"
            "}\n"
        )

        diagnostics = analyze(result.ast).diagnostics
        ownership_errors = [
            item for item in diagnostics if item.code == "E_FRAME_BUFFER_OWNERSHIP"
        ]
        self.assertEqual(len(ownership_errors), 4)

    def test_all_builtin_calls_are_reserved_function_names(self) -> None:
        source = "".join(f"fn {name}() {{}}\n" for name in sorted(_BUILTIN_CALLS))
        parsed = parse_source(source)

        diagnostics = analyze(parsed.ast).diagnostics
        reserved_names = {
            item.message.split("'")[1]
            for item in diagnostics
            if item.code == "E_RESERVED_NAME"
        }
        self.assertEqual(reserved_names, _BUILTIN_CALLS)

    def test_frame_buffer_is_local_owned_and_cleaned(self) -> None:
        valid = parse_source(
            "fn inspect(size: u64) -> u64 {\n"
            "  let bytes = frame_buffer(size);\n"
            "  return frame_buffer_len(bytes);\n"
            "}\n"
        )
        self.assertEqual(valid.diagnostics, ())
        self.assertEqual(analyze(valid.ast).diagnostics, ())

        ownership = parse_source(
            "fn copy(size: u64) -> u64 {\n"
            "  let bytes = frame_buffer(size);\n"
            "  let duplicate = bytes;\n"
            "  return frame_buffer_len(duplicate);\n"
            "}\n"
            "fn escape(size: u64) -> FrameBuffer {\n"
            "  return frame_buffer(size);\n"
            "}\n"
            "fn mutable(size: u64) -> u64 {\n"
            "  var bytes = frame_buffer(size);\n"
            "  return frame_buffer_len(bytes);\n"
            "}\n"
        )
        self.assertEqual(ownership.diagnostics, ())
        self.assertEqual(analyze(ownership.ast).diagnostics, ())

    def test_assignment_requires_var_and_preserves_binding_type(self) -> None:
        result = parse_source(
            "fn valid() -> i32 { var value = 1; value = 2; return value; }\n"
            "fn immutable() -> i32 { let value = 1; value = 2; return value; }\n"
            "fn parameter(value: i32) -> i32 { value = 2; return value; }\n"
            "fn unknown() { missing = 2; }\n"
            "fn mismatch() { var value = 1; value = true; }\n"
        )

        self.assertIn("E_ASSIGN_IMMUTABLE", [item.code for item in result.diagnostics])
        self.assertIn("E_UNKNOWN_NAME", [item.code for item in result.diagnostics])
        self.assertIn("E_TYPE_MISMATCH", [item.code for item in result.diagnostics])
        self.assertEqual(
            sum(item.code == "E_ASSIGN_IMMUTABLE" for item in result.diagnostics),
            2,
        )

    def test_sensor_pipeline_infers_tensor_units_and_cancels_normalization_unit(self) -> None:
        source = (
            "type Frame { samples: Tensor<f32, [4, 4], V>, }\n"
            "fn normalize(frame: Frame) -> Tensor<f32, [4, 4]> {\n"
            "  let centered = frame.samples |> subtract(mean(frame.samples));\n"
            "  return centered |> scale(1.0 / max_abs(centered));\n"
            "}\n"
        )
        result = parse_source(source)
        self.assertEqual(result.diagnostics, ())

        semantic = analyze(result.ast)
        return_expr = result.ast.declarations[1].body[-1].value
        inferred = dict(semantic.expression_types)[return_expr.span]
        self.assertEqual(inferred.name, "Tensor")
        self.assertEqual(inferred.dimensions, (4, 4))
        self.assertEqual(inferred.unit, ())

    def test_unit_and_shape_mismatches_are_diagnosed(self) -> None:
        result = parse_source(
            "fn add(a: Tensor<f32, [2, 3], V>, "
            "b: Tensor<f32, [2, 3], mm>) {\n"
            "  let invalid = a + b;\n"
            "}\n"
        )
        codes = [item.code for item in result.diagnostics]

        self.assertIn("E_UNIT_MISMATCH", codes)

    def test_tensor_shape_mismatch_has_dedicated_diagnostic(self) -> None:
        result = parse_source(
            "fn add(a: Tensor<f32, [2, 3]>, b: Tensor<f32, [3, 2]>) {\n"
            "  let invalid = a + b;\n"
            "}\n"
        )

        self.assertIn("E_SHAPE_MISMATCH", [item.code for item in result.diagnostics])

    def test_tensor_comparisons_are_rejected_without_boolean_tensor_types(self) -> None:
        result = parse_source(
            "fn equal(a: Tensor<f32, [2]>, b: Tensor<f32, [2]>) -> bool {\n"
            "  return a == b;\n"
            "}\n"
            "fn less(a: Tensor<f32, [2]>, b: Tensor<f32, [2]>) -> bool {\n"
            "  return a < b;\n"
            "}\n"
        )

        tensor_comparison_diagnostics = [
            item
            for item in result.diagnostics
            if item.code == "E_TENSOR_COMPARISON"
        ]
        self.assertEqual(len(tensor_comparison_diagnostics), 2)
        with self.assertRaisesRegex(CompilationError, "E_TENSOR_COMPARISON"):
            compile_source(
                "fn equal(a: Tensor<f32, [2]>, b: Tensor<f32, [2]>) -> bool {\n"
                "  return a == b;\n"
                "}\n"
            )

    def test_tensor_contract_rejects_unsupported_broadcasting_and_operators(self) -> None:
        result = parse_source(
            "fn divide_scalar(value: f32, values: Tensor<f32, [2]>) {\n"
            "  let invalid = value / values;\n"
            "}\n"
            "fn multiply_tensors(left: Tensor<f32, [2]>, "
            "right: Tensor<f32, [2]>) {\n"
            "  let invalid = left * right;\n"
            "}\n"
            "fn divide_tensors(left: Tensor<f32, [2]>, "
            "right: Tensor<f32, [2]>) {\n"
            "  let invalid = left / right;\n"
            "}\n"
            "fn broadcast_rows(left: Tensor<f32, [2, 1]>, "
            "right: Tensor<f32, [1, 2]>) {\n"
            "  let invalid = left + right;\n"
            "}\n"
            "fn remainder_tensor(values: Tensor<i32, [2]>) {\n"
            "  let invalid = values % 2;\n"
            "}\n"
        )

        operator_diagnostics = [
            item
            for item in result.diagnostics
            if item.code == "E_TYPE_OPERATOR"
        ]
        shape_diagnostics = [
            item for item in result.diagnostics if item.code == "E_SHAPE_MISMATCH"
        ]
        self.assertEqual(len(operator_diagnostics), 4)
        self.assertEqual(len(shape_diagnostics), 1)

    def test_unit_scales_are_not_implicitly_mixed(self) -> None:
        result = parse_source(
            "fn add(a: Tensor<f32, [2], m>, b: Tensor<f32, [2], mm>) {\n"
            "  let invalid = a + b;\n"
            "}\n"
        )

        self.assertIn("E_UNIT_MISMATCH", [item.code for item in result.diagnostics])

    def test_explicit_conversion_changes_tensor_unit_and_scale(self) -> None:
        result = parse_source(
            "fn convert_length(value: Tensor<f32, [2], mm>) "
            "-> Tensor<f32, [2], m> {\n"
            "  return value |> convert(\"m\");\n"
            "}\n"
        )

        self.assertEqual(result.diagnostics, ())
        returned = result.ast.declarations[0].body[0].value
        inferred = dict(analyze(result.ast).expression_types)[returned.span]
        self.assertEqual(inferred.unit_symbol, "m")
        self.assertEqual(inferred.unit_scale.numerator, 1)
        self.assertEqual(inferred.unit_scale.denominator, 1)
        conversion = analyze(result.ast).conversions[0]
        self.assertEqual(conversion.multiplier.numerator, 1)
        self.assertEqual(conversion.multiplier.denominator, 1000)
        self.assertEqual(conversion.addend, 0)

    def test_si_prefixes_resolve_exactly_and_preserve_conversion_scales(self) -> None:
        result = parse_source(
            "fn to_kilohertz(value: Tensor<f64, [2], MHz>) "
            "-> Tensor<f64, [2], kHz> {\n"
            "  return convert(value, \"kHz\");\n"
            "}\n"
            "fn to_hertz(value: Tensor<f64, [2], MHz>) "
            "-> Tensor<f64, [2], Hz> { return convert(value, \"Hz\"); }\n"
            "fn to_micrometers(value: Tensor<f32, [2], µm>) "
            "-> Tensor<f32, [2], um> { return value; }\n"
            "fn prefixed_existing(value: f64<mkm>) -> f64<m> { return value; }\n"
            "fn decameter(value: f64<dam>) -> f64<dam> { return value; }\n"
        )

        self.assertEqual(result.diagnostics, ())
        semantic = analyze(result.ast)
        frequency_conversion = semantic.conversions[0]
        self.assertEqual(frequency_conversion.multiplier, Fraction(1000))
        self.assertEqual(frequency_conversion.addend, 0)
        self.assertEqual(semantic.conversions[1].multiplier, Fraction(1_000_000))
        self.assertEqual(
            dict(semantic.expression_types)[
                result.ast.declarations[2].body[0].value.span
            ].unit_scale,
            Fraction(1, 1_000_000),
        )
        self.assertEqual(
            dict(semantic.expression_types)[
                result.ast.declarations[2].body[0].value.span
            ].unit_symbol,
            "um",
        )
        self.assertEqual(
            dict(semantic.expression_types)[
                result.ast.declarations[4].body[0].value.span
            ].unit_scale,
            Fraction(10),
        )

    def test_compound_units_normalize_for_type_compatibility_and_conversion(self) -> None:
        result = parse_source(
            "fn slope(value: Tensor<f32, [2], V/mm>) -> f32<kV/m> {\n"
            "  return mean(value);\n"
            "}\n"
            "fn convert_slope(value: Tensor<f32, [2], V/mm>) "
            "-> Tensor<f32, [2], kV/m> { return convert(value, \"kV/m\"); }\n"
        )

        self.assertEqual(result.diagnostics, ())
        semantic = analyze(result.ast)
        returned = result.ast.declarations[0].body[0].value
        inferred = dict(semantic.expression_types)[returned.span]
        self.assertEqual(inferred.unit, (("length", -1), ("voltage", 1)))
        self.assertEqual(inferred.unit_scale, Fraction(1000))
        self.assertEqual(inferred.display(), "f32<V/mm>")
        conversion = semantic.conversions[0]
        self.assertEqual(conversion.multiplier, 1)

    def test_compound_units_support_grouping_powers_and_left_associative_division(self) -> None:
        result = parse_source(
            "fn grouped(value: f32<V/(mm*s)>) -> f32<V/mm/s> { return value; }\n"
            "fn squared(value: f32<(V/m)^2>) -> f32<V^2/m^2> { return value; }\n"
            "fn inverse_time(value: f64<1/ms>) -> f64<kHz> { return value; }\n"
        )

        self.assertEqual(result.diagnostics, ())
        semantic = analyze(result.ast)
        expression_types = dict(semantic.expression_types)
        grouped = expression_types[result.ast.declarations[0].body[0].value.span]
        squared = expression_types[result.ast.declarations[1].body[0].value.span]
        inverse_time = expression_types[result.ast.declarations[2].body[0].value.span]
        self.assertEqual(grouped.unit, (("length", -1), ("time", -1), ("voltage", 1)))
        self.assertEqual(grouped.unit_scale, Fraction(1000))
        self.assertEqual(squared.unit, (("length", -2), ("voltage", 2)))
        self.assertEqual(squared.unit_scale, Fraction(1))
        self.assertEqual(inverse_time.unit, (("time", -1),))
        self.assertEqual(inverse_time.unit_symbol, "1/ms")

    def test_unit_prefix_resolution_rejects_affine_and_unknown_combinations(self) -> None:
        result = parse_source(
            "fn invalid(value: Tensor<f32, [2], mdegC>) {}\n"
            "fn unknown(value: Tensor<f32, [2], furlongs>) {}\n"
            "fn affine(value: Tensor<f32, [2], degC/s>) {}\n"
        )

        codes = [item.code for item in result.diagnostics]
        self.assertIn("E_UNKNOWN_UNIT", codes)
        self.assertIn("E_UNIT_EXPRESSION", codes)

    def test_explicit_celsius_to_fahrenheit_conversion_is_affine(self) -> None:
        result = parse_source(
            "fn convert_temperature(value: Tensor<f32, [1], degC>) "
            "-> Tensor<f32, [1], degF> {\n"
            "  return convert(value, \"degF\");\n"
            "}\n"
        )

        self.assertEqual(result.diagnostics, ())
        conversion = analyze(result.ast).conversions[0]
        self.assertEqual(conversion.multiplier, Fraction(9, 5))
        self.assertEqual(conversion.addend, Fraction(32))

    def test_degree_to_radian_conversion_keeps_pi_symbolic(self) -> None:
        result = parse_source(
            "fn convert_angle(value: Tensor<f32, [1], deg>) "
            "-> Tensor<f32, [1], rad> {\n"
            "  return convert(value, \"rad\");\n"
            "}\n"
        )

        self.assertEqual(result.diagnostics, ())
        conversion = analyze(result.ast).conversions[0]
        self.assertEqual(conversion.multiplier, Fraction(1, 180))
        self.assertEqual(conversion.pi_power, 1)

    def test_temperature_arithmetic_requires_explicit_temperature_semantics(self) -> None:
        result = parse_source(
            "fn add(a: Tensor<f32, [2], degC>, "
            "b: Tensor<f32, [2], degC>) {\n"
            "  let invalid = a + b;\n"
            "}\n"
        )

        self.assertIn(
            "E_TEMPERATURE_ARITHMETIC",
            [item.code for item in result.diagnostics],
        )

    def test_unknown_units_and_invalid_builtin_calls_are_reported(self) -> None:
        result = parse_source(
            "fn invalid(value: Tensor<f32, [2], furlong>) {\n"
            "  let average = mean();\n"
            "}\n"
        )
        codes = [item.code for item in result.diagnostics]

        self.assertIn("E_UNKNOWN_UNIT", codes)
        self.assertIn("E_BUILTIN_ARITY", codes)

    def test_non_numeric_addition_is_rejected(self) -> None:
        result = parse_source("fn invalid(a: bool, b: bool) { let value = a + b; }\n")

        self.assertIn("E_TYPE_OPERATOR", [item.code for item in result.diagnostics])

    def test_tensor_dimensions_must_be_positive_static_integers(self) -> None:
        result = parse_source("fn invalid(value: Tensor<f32, [0, 3]>) {}\n")

        self.assertIn("E_TENSOR_DIMENSION", [item.code for item in result.diagnostics])

    def test_tensor_shape_must_have_at_least_one_dimension(self) -> None:
        result = parse_source("fn invalid(value: Tensor<f32, []>) {}\n")

        self.assertIn("E_TENSOR_SHAPE", [item.code for item in result.diagnostics])

    def test_effect_requirements_propagate_through_calls(self) -> None:
        result = parse_source(
            "@net\n"
            "fn request() -> Result<str, NetError> {\n"
            "  let response = net_get(\"https://example.test/data\"); return response;\n"
            "}\n"
            "fn caller() { return request(); }\n"
            "@net\n"
            "fn authorized() { return request(); }\n"
        )

        violations = [item for item in result.diagnostics if item.code == "E_EFFECT_VIOLATION"]
        self.assertEqual(len(violations), 1)
        self.assertIn("@net", violations[0].message)
        summaries = {item.function: item for item in analyze(result.ast).effects}
        self.assertEqual(summaries["request"].required, ("net",))
        self.assertEqual(summaries["caller"].required, ("net",))

    def test_effectful_builtins_require_declared_effects(self) -> None:
        result = parse_source(
            "fn invalid() { let content = fs_read_text(\"data.txt\"); return content; }\n"
            "@fs\n"
            "fn authorized() { let content = fs_read_text(\"data.txt\"); return content; }\n"
        )

        violations = [item for item in result.diagnostics if item.code == "E_EFFECT_VIOLATION"]
        self.assertEqual(len(violations), 1)
        self.assertIn("@fs", violations[0].message)

    def test_pure_annotation_rejects_direct_effectful_operations(self) -> None:
        result = parse_source(
            "@pure\n"
            "fn invalid() -> Result<str, NetError> {\n"
            "  let response = net_get(\"https://example.test/data\"); return response;\n"
            "}\n"
        )

        self.assertIn("E_EFFECT_VIOLATION", [item.code for item in result.diagnostics])

    def test_effects_propagate_across_recursive_call_graphs(self) -> None:
        result = parse_source(
            "@net\n"
            "fn fetch() -> Result<str, NetError> {\n"
            "  let response = net_get(\"https://example.test/data\"); caller(); return response;\n"
            "}\n"
            "fn caller() { return fetch(); }\n"
        )

        self.assertEqual(
            [item.code for item in result.diagnostics].count("E_EFFECT_VIOLATION"),
            1,
        )
        summaries = {item.function: item for item in analyze(result.ast).effects}
        self.assertEqual(summaries["caller"].required, ("net",))

    def test_ffi_effect_is_required_and_propagates_through_calls(self) -> None:
        result = parse_source(
            '@ffi\n@link("sensor")\n'
            'extern "C" fn read_channel(channel: u32) -> i32;\n'
            "fn unauthorized(channel: u32) -> i32 { return read_channel(channel); }\n"
            "@ffi\n"
            "fn authorized(channel: u32) -> i32 { return read_channel(channel); }\n"
        )

        violations = [
            item for item in result.diagnostics
            if item.code == "E_EFFECT_VIOLATION"
        ]
        self.assertEqual(len(violations), 1)
        self.assertIn("@ffi", violations[0].message)
        summaries = {item.function: item for item in analyze(result.ast).effects}
        self.assertEqual(summaries["read_channel"].required, ("ffi",))
        self.assertEqual(summaries["authorized"].required, ("ffi",))
        self.assertEqual(summaries["unauthorized"].required, ("ffi",))

    def test_ffi_rejects_missing_effect_link_and_non_scalar_types(self) -> None:
        missing_ffi = parse_source(
            '@link("sensor")\nextern "C" fn read() -> i32;\n'
        )
        missing_link = parse_source(
            '@ffi\nextern "C" fn read() -> i32;\n'
        )
        unsupported = parse_source(
            '@ffi\n@link("sensor")\n'
            'extern "C" fn read(value: Tensor<f32, [2]>) -> Result<i32, IoError>;\n'
        )

        self.assertIn(
            "E_EXTERN_EFFECT",
            [item.code for item in missing_ffi.diagnostics],
        )
        self.assertIn(
            "E_EXTERN_LINK",
            [item.code for item in missing_link.diagnostics],
        )
        self.assertGreaterEqual(
            [item.code for item in unsupported.diagnostics].count("E_FFI_TYPE"),
            2,
        )

    def test_ffi_rejects_pure_or_malformed_library_declarations(self) -> None:
        pure_extern = parse_source(
            '@pure\n@ffi\n@link("sensor")\n'
            'extern "C" fn read() -> i32;\n'
        )
        path_library = parse_source(
            '@ffi\n@link("../sensor")\n'
            'extern "C" fn read() -> i32;\n'
        )
        link_on_normal_function = parse_source(
            '@link("sensor")\nfn read() -> i32 { return 1; }\n'
        )

        self.assertIn(
            "E_CONFLICTING_EFFECTS",
            [item.code for item in pure_extern.diagnostics],
        )
        self.assertIn("E_EXTERN_LINK", [item.code for item in path_library.diagnostics])
        self.assertIn(
            "E_EXTERN_ATTRIBUTE",
            [item.code for item in link_on_normal_function.diagnostics],
        )

    def test_result_rejects_unsupported_error_types(self) -> None:
        result = parse_source("fn invalid() -> Result<str, CustomError> {}\n")

        self.assertIn(
            "E_RESULT_ERROR_TYPE",
            [item.code for item in result.diagnostics],
        )

    def test_match_result_is_exhaustive_and_types_payloads(self) -> None:
        result = parse_source(
            "@io\n"
            "fn read_or_default() -> str {\n"
            "  let result = io_read_line();\n"
            "  return match result {\n"
            "    Ok(value) => value,\n"
            "    Err(error) => \"fallback\",\n"
            "  };\n"
            "}\n"
        )

        self.assertEqual(result.diagnostics, ())
        returned = result.ast.declarations[0].body[-1].value
        self.assertEqual(returned.kind, "match")
        self.assertEqual(dict(analyze(result.ast).expression_types)[returned.span].name, "str")

    def test_match_result_requires_both_variants_and_compatible_arms(self) -> None:
        result = parse_source(
            "@io\n"
            "fn invalid() -> Result<str, IoError> {\n"
            "  let value = io_read_line();\n"
            "  return match value { Ok(text) => text };\n"
            "}\n"
        )

        self.assertIn(
            "E_MATCH_NON_EXHAUSTIVE",
            [item.code for item in result.diagnostics],
        )

    def test_match_result_rejects_duplicate_variants_and_mismatched_arm_types(self) -> None:
        result = parse_source(
            "@io\n"
            "fn invalid() -> str {\n"
            "  let value = io_read_line();\n"
            "  return match value {\n"
            "    Ok(text) => text,\n"
            "    Ok(other) => other,\n"
            "    Err(error) => false,\n"
            "  };\n"
            "}\n"
        )
        codes = [item.code for item in result.diagnostics]

        self.assertIn("E_MATCH_DUPLICATE_ARM", codes)
        self.assertIn("E_TYPE_MISMATCH", codes)

    def test_match_and_try_survive_ast_canonical_roundtrip(self) -> None:
        source = (
            "@io\n"
            "fn read() -> Result<str, IoError> { return io_read_line(); }\n"
            "@io\n"
            "fn wrapper() -> str {\n"
            "  let result = read();\n"
            "  return match result { Ok(value) => value, Err(error) => \"fallback\" };\n"
            "}\n"
        )
        parsed = parse_source(source)
        canonical = render_module(parsed.ast)
        reparsed = parse_source(canonical)

        self.assertEqual(parsed.diagnostics, ())
        self.assertEqual(reparsed.diagnostics, ())
        self.assertIn("match result", canonical)

    def test_question_operator_propagates_matching_error_type(self) -> None:
        result = parse_source(
            "@io\n"
            "fn read_value() -> Result<str, IoError> { return io_read_line(); }\n"
            "@io\n"
            "fn wrapper() -> Result<str, IoError> { return read_value()?; }\n"
        )

        self.assertEqual(result.diagnostics, ())
        propagated = result.ast.declarations[1].body[0].value
        self.assertEqual(dict(analyze(result.ast).expression_types)[propagated.span].name, "str")

    def test_result_constructors_and_try_require_matching_result_context(self) -> None:
        valid = parse_source(
            "fn source() -> Result<i32, IoError> { return Ok(4); }\n"
            "fn forward() -> Result<i32, IoError> {\n"
            "  let value = source()?;\n"
            "  return Ok(value + 1);\n"
            "}\n"
            "fn fail() -> Result<i32, IoError> { return Err(IoError); }\n"
        )
        self.assertEqual(valid.diagnostics, ())

        invalid = parse_source(
            "fn bad_error() -> Result<i32, IoError> { return Err(FsError); }\n"
            "fn no_context() -> i32 { return Ok(1); }\n"
        )
        self.assertIn(
            "E_RESULT_CONSTRUCTOR_TYPE",
            [item.code for item in invalid.diagnostics],
        )
        self.assertIn(
            "E_RESULT_CONSTRUCTOR_CONTEXT",
            [item.code for item in invalid.diagnostics],
        )

    def test_question_operator_rejects_wrong_context_and_error_type(self) -> None:
        result = parse_source(
            "@io\n"
            "fn read_value() -> Result<str, IoError> { return io_read_line(); }\n"
            "fn invalid_context() -> str { return read_value()?; }\n"
            "@io\n"
            "fn invalid_error() -> Result<str, FsError> { return read_value()?; }\n"
        )
        codes = [item.code for item in result.diagnostics]

        self.assertIn("E_TRY_CONTEXT", codes)
        self.assertIn("E_TRY_ERROR_TYPE", codes)

    def test_result_values_cannot_be_silently_discarded(self) -> None:
        result = parse_source(
            "@io\n"
            "fn invalid() { io_read_line(); }\n"
            "@io\n"
            "fn also_invalid() { let ignored = io_read_line(); }\n"
        )

        self.assertEqual(
            [item.code for item in result.diagnostics].count("E_UNHANDLED_RESULT"),
            2,
        )

    def test_result_binding_requires_a_handling_use_not_any_reference(self) -> None:
        result = parse_source(
            "@io\n"
            "fn invalid() {\n"
            "  let value = io_read_line();\n"
            "  let unrelated = value == value;\n"
            "}\n"
            "@io\n"
            "fn handled() -> str {\n"
            "  let value = io_read_line();\n"
            "  let alias = value;\n"
            "  return match alias { Ok(text) => text, Err(error) => \"fallback\" };\n"
            "}\n"
        )

        unhandled = [
            item for item in result.diagnostics
            if item.code == "E_UNHANDLED_RESULT"
        ]
        self.assertEqual(len(unhandled), 1)
        self.assertIn("value", unhandled[0].message)

    def test_result_in_short_circuited_match_is_not_definitely_handled(self) -> None:
        result = parse_source(
            "@io\n"
            "fn maybe() -> bool {\n"
            "  let value = io_read_line();\n"
            "  return true || match value {\n"
            "    Ok(_) => false,\n"
            "    Err(_) => false,\n"
            "  };\n"
            "}\n"
        )

        self.assertIn("E_UNHANDLED_RESULT", [item.code for item in result.diagnostics])

    def test_result_use_after_return_does_not_count_as_handling(self) -> None:
        result = parse_source(
            "@io\n"
            "fn maybe() -> str {\n"
            "  let value = io_read_line();\n"
            "  return \"done\";\n"
            "  return match value { Ok(text) => text, Err(_) => \"fallback\" };\n"
            "}\n"
        )

        self.assertIn("E_UNHANDLED_RESULT", [item.code for item in result.diagnostics])

    def test_result_binding_cannot_be_consumed_more_than_once(self) -> None:
        result = parse_source(
            "@io\n"
            "fn invalid() -> str {\n"
            "  let value = io_read_line();\n"
            "  let alias = value;\n"
            "  let second_alias = alias;\n"
            "  let first = match second_alias {\n"
            "    Ok(text) => text,\n"
            "    Err(_) => \"first fallback\",\n"
            "  };\n"
            "  return match value {\n"
            "    Ok(text) => text,\n"
            "    Err(_) => \"second fallback\",\n"
            "  };\n"
            "}\n"
        )

        self.assertIn("E_RESULT_REUSE", [item.code for item in result.diagnostics])

    def test_optional_member_access_is_rejected_without_an_option_type(self) -> None:
        result = parse_source(
            "type Point { x: f32, }\n"
            "fn read(point: Point) -> f32 { return point?.x; }\n"
        )

        self.assertIn(
            "E_OPTIONAL_MEMBER_UNSUPPORTED",
            [item.code for item in result.diagnostics],
        )

    def test_coalescing_operator_is_rejected_until_option_semantics_exist(self) -> None:
        result = parse_source(
            "fn fallback(value: bool) -> bool { return value ?? false; }\n"
        )

        self.assertIn("E_TYPE_OPERATOR", [item.code for item in result.diagnostics])

    def test_effectful_builtin_signatures_return_typed_results(self) -> None:
        result = parse_source(
            "@io\n"
            "fn read_input() -> Result<str, IoError> { return io_read_line(); }\n"
            "@io\n"
            "fn write_output(text: str) -> Result<void, IoError> {\n"
            "  return io_write(text);\n"
            "}\n"
            "@fs\n"
            "fn read_file(path: str) -> Result<str, FsError> {\n"
            "  return fs_read_text(path);\n"
            "}\n"
            "@fs\n"
            "fn write_file(path: str, text: str) -> Result<void, FsError> {\n"
            "  return fs_write_text(path, text);\n"
            "}\n"
            "@fs\n"
            "fn file_exists(path: str) -> Result<bool, FsError> {\n"
            "  return fs_exists(path);\n"
            "}\n"
            "@net\n"
            "fn get_data(url: str) -> Result<str, NetError> { return net_get(url); }\n"
            "@net\n"
            "fn post_data(url: str, body: str) -> Result<str, NetError> {\n"
            "  return net_post(url, body);\n"
            "}\n"
        )

        self.assertEqual(result.diagnostics, ())

    def test_effectful_builtin_argument_and_result_types_are_checked(self) -> None:
        result = parse_source(
            "@fs\n"
            "fn invalid() -> Result<bool, FsError> {\n"
            "  return fs_read_text(false);\n"
            "}\n"
        )
        codes = [item.code for item in result.diagnostics]

        self.assertIn("E_ARGUMENT_TYPE", codes)
        self.assertIn("E_RETURN_TYPE", codes)

    def test_return_and_argument_types_are_checked(self) -> None:
        result = parse_source(
            "fn takes(value: i32) -> i32 { return value; }\n"
            "fn caller() -> bool { return takes(true); }\n"
        )
        codes = [item.code for item in result.diagnostics]

        self.assertIn("E_ARGUMENT_TYPE", codes)
        self.assertIn("E_RETURN_TYPE", codes)

    def test_numeric_argument_types_are_not_silently_coerced(self) -> None:
        result = parse_source(
            "fn takes(value: i32) { return value; }\n"
            "fn caller() { takes(1.0); }\n"
        )

        self.assertIn("E_ARGUMENT_TYPE", [item.code for item in result.diagnostics])

    def test_autodiff_v0_accepts_pure_unary_float_functions(self) -> None:
        result = parse_source(
            "fn objective(x: f64) -> f64 {\n"
            "  let square = x * x;\n"
            "  return square * square + x;\n"
            "}\n"
            "fn derivative(x: f64) -> f64 { return gradient(objective, x); }\n"
        )

        self.assertEqual(result.diagnostics, ())
        gradient_call = result.ast.declarations[1].body[0].value
        self.assertEqual(
            dict(analyze(result.ast).expression_types)[gradient_call.span].name,
            "f64",
        )
        lowered = compile_source(
            "fn objective(x: f64) -> f64 { return x * x; }\n"
            "fn derivative(x: f64) -> f64 { return gradient(objective, x); }\n"
        )
        expression = lowered.function("derivative").body[0].expression
        self.assertEqual(expression.kind, "gradient")
        self.assertEqual(expression.value, "objective")

    def test_autodiff_v0_rejects_unsupported_targets_and_inputs(self) -> None:
        cases = (
            (
                "fn gradient_target(x: f32) -> f32 { return x; }\n"
                "fn caller(x: f32) -> f32 { return gradient(missing, x); }\n",
                "E_GRADIENT_TARGET",
            ),
            (
                "fn objective(x: i32) -> i32 { return x * x; }\n"
                "fn caller(x: i32) -> i32 { return gradient(objective, x); }\n",
                "E_GRADIENT_TYPE",
            ),
            (
                "fn objective(x: f32) -> f32 { return abs(x); }\n"
                "fn caller(x: f32) -> f32 { return gradient(objective, x); }\n",
                "E_GRADIENT_OPERATION",
            ),
            (
                "@io\nfn objective(x: f32) -> f32 { return x; }\n"
                "fn caller(x: f32) -> f32 { return gradient(objective, x); }\n",
                "E_GRADIENT_EFFECT",
            ),
            (
                "fn objective(x: Tensor<f32, [2]>) -> f32 { return sum(x); }\n"
                "fn caller(x: Tensor<f32, [3]>) -> Tensor<f32, [3]> "
                "{ return gradient(objective, x); }\n",
                "E_GRADIENT_TYPE",
            ),
        )
        for source, expected in cases:
            with self.subTest(expected=expected):
                result = parse_source(source)
                self.assertIn(expected, [item.code for item in result.diagnostics])

    def test_tensor_autodiff_v0_infers_shape_and_quotient_unit(self) -> None:
        source = (
            "fn objective(values: Tensor<f64, [2], mm>) -> f64<V> {\n"
            "  return sum(abs(values));\n"
            "}\n"
            "fn derivative(values: Tensor<f64, [2], mm>) "
            "-> Tensor<f64, [2], V/mm> {\n"
            "  return gradient(objective, values);\n"
            "}\n"
        )
        result = parse_source(source)
        self.assertIn("E_GRADIENT_OPERATION", [item.code for item in result.diagnostics])

        supported = parse_source(
            "fn objective(values: Tensor<f32, [2], cm>) -> f32<m> {\n"
            "  return convert(sum(values), \"m\");\n"
            "}\n"
            "fn derivative(values: Tensor<f32, [2], cm>) "
            "-> Tensor<f32, [2], m/cm> {\n"
            "  return gradient(objective, values);\n"
            "}\n"
        )
        self.assertEqual(supported.diagnostics, ())
        gradient_call = supported.ast.declarations[1].body[0].value
        inferred = dict(analyze(supported.ast).expression_types)[gradient_call.span]
        self.assertEqual(inferred.name, "Tensor")
        self.assertEqual(inferred.dimensions, (2,))
        self.assertEqual(inferred.element_type.name, "f32")
        self.assertEqual(inferred.unit, ())
        self.assertEqual(inferred.unit_scale, Fraction(100))
        self.assertEqual(inferred.unit_symbol, "m/cm")

    def test_tensor_autodiff_v0_rejects_nonlinear_units_and_shape_mismatches(self) -> None:
        cases = (
            (
                "fn objective(x: Tensor<f32, [2], degC>) -> f32<V> "
                "{ return sum(x); }\n"
                "fn caller(x: Tensor<f32, [2], degC>) "
                "-> Tensor<f32, [2], V/degC> { return gradient(objective, x); }\n",
                "E_GRADIENT_UNIT",
            ),
            (
                "fn objective(x: Tensor<f32, [2], rad>) -> f32<rad> "
                "{ return sum(x); }\n"
                "fn caller(x: Tensor<f32, [2], rad>) "
                "-> Tensor<f32, [2]> { return gradient(objective, x); }\n",
                "E_GRADIENT_UNIT",
            ),
            (
                "fn objective(x: Tensor<f32, [2]>) -> f32 { return sum(x); }\n"
                "fn caller(x: Tensor<f32, [3]>) "
                "-> Tensor<f32, [3]> { return gradient(objective, x); }\n",
                "E_GRADIENT_TYPE",
            ),
        )
        for source, expected in cases:
            with self.subTest(expected=expected):
                result = parse_source(source)
                self.assertIn(expected, [item.code for item in result.diagnostics])


if __name__ == "__main__":
    unittest.main()
