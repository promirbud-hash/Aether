import math
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

from src.frontend import parse_source
from src.api import CompilationError, compile_source, run_source
from src.interpreter import (
    AetherRuntimeError,
    Interpreter,
    Quantity,
    ResultValue,
    _FrameBuffer,
)
from src.ir import IRLoweringError, lower_module


class TypedIRTests(unittest.TestCase):
    def test_ir_records_frame_resource_construction_order(self) -> None:
        parsed = parse_source(
            "fn inspect(size: u64, other: u64) -> u64 {\n"
            "  let first = frame_buffer(size);\n"
            "  let second = frame_buffer(other);\n"
            "  return frame_buffer_len(first) + frame_buffer_len(second);\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        function = lower_module(parsed.ast).function("inspect")
        self.assertEqual(function.frame_resources, ("first", "second"))

    def test_ir_records_frame_buffer_retain_and_return_ownership(self) -> None:
        parsed = parse_source(
            "fn identity(bytes: FrameBuffer) -> FrameBuffer { return bytes; }\n"
            "fn copy(size: u64) -> u64 {\n"
            "  var bytes = frame_buffer(size);\n"
            "  let duplicate = bytes;\n"
            "  bytes = bytes;\n"
            "  return frame_buffer_len(duplicate);\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        module = lower_module(parsed.ast)
        identity = module.function("identity")
        copy = module.function("copy")

        self.assertEqual(identity.body[0].ownership_actions[0].kind, "retain_return")
        self.assertEqual(copy.body[1].ownership_actions[0].kind, "retain")
        self.assertEqual(
            [action.kind for action in copy.body[2].ownership_actions],
            ["retain", "release", "transfer"],
        )
        self.assertTrue(copy.body[0].expression.temporary_owner)

    def test_lowering_preserves_expression_types_and_spans(self) -> None:
        parsed = parse_source(
            "fn add(left: i32, right: i32) -> i32 { return left + right; }\n"
        )
        lowered = lower_module(parsed.ast)
        function = lowered.function("add")
        expression = function.body[0].expression

        self.assertEqual(function.return_type.name, "i32")
        self.assertEqual(expression.type.name, "i32")
        self.assertEqual(expression.span.line, 1)

    def test_ir_retains_declared_and_transitive_effects(self) -> None:
        parsed = parse_source(
            "@io\n"
            "fn read() -> Result<str, IoError> { return io_read_line(); }\n"
            "@io\n"
            "fn wrapper() -> Result<str, IoError> { return read(); }\n"
        )
        wrapper = lower_module(parsed.ast).function("wrapper")

        self.assertEqual(wrapper.declared_effects, ("io",))
        self.assertEqual(wrapper.required_effects, ("io",))

    def test_ir_preserves_extern_abi_library_and_signature(self) -> None:
        parsed = parse_source(
            '@ffi\n@link("sensor")\n'
            'extern "C" fn read_channel(channel: u32) -> i32;\n'
        )
        self.assertEqual(parsed.diagnostics, ())
        external = lower_module(parsed.ast).function("read_channel")

        self.assertEqual(external.external_abi, "C")
        self.assertEqual(external.external_library, "sensor")
        self.assertEqual(external.required_effects, ("ffi",))
        self.assertEqual(external.parameters[0].type.name, "u32")
        self.assertEqual(external.return_type.name, "i32")
        self.assertEqual(external.body, ())

    def test_ir_preserves_unit_conversion_metadata_and_match_payload_types(self) -> None:
        parsed = parse_source(
            "fn to_fahrenheit(value: Tensor<f32, [1], degC>) "
            "-> Tensor<f32, [1], degF> { return convert(value, \"degF\"); }\n"
            "fn inspect(value: Result<str, IoError>) -> str {\n"
            "  return match value { Ok(text) => text, Err(_) => \"failed\" };\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        module = lower_module(parsed.ast)
        conversion = module.function("to_fahrenheit").body[0].expression
        match = module.function("inspect").body[0].expression

        self.assertEqual(conversion.type.unit_symbol, "degF")
        self.assertIsNotNone(conversion.conversion)
        self.assertEqual(match.type.name, "str")
        self.assertEqual(match.children[0].type.name, "Result")
        self.assertEqual(match.arms[0].value.type.name, "str")

    def test_lowering_refuses_semantic_errors(self) -> None:
        parsed = parse_source(
            "fn invalid(left: bool, right: bool) { let value = left + right; }\n"
        )

        with self.assertRaises(IRLoweringError):
            lower_module(parsed.ast)

    def test_lowering_refuses_an_expression_without_an_inferred_type(self) -> None:
        parsed = parse_source(
            "fn caller() { return callee(); }\n"
            "fn callee() { return 1; }\n"
        )

        with self.assertRaises(IRLoweringError) as raised:
            lower_module(parsed.ast)
        self.assertIn("E_IR_UNTYPED", {item.code for item in raised.exception.diagnostics})


class InterpreterTests(unittest.TestCase):
    def test_frame_resources_release_lifo_on_nested_calls_and_question_error(self) -> None:
        source = (
            "fn inner(size: u64) -> u64 {\n"
            "  let temporary = frame_buffer(size);\n"
            "  return frame_buffer_len(temporary);\n"
            "}\n"
            "fn fail() -> Result<u64, IoError> { return Err(IoError); }\n"
            "fn outer(size: u64, next: u64, nested: u64) -> Result<u64, IoError> {\n"
            "  let first = frame_buffer(size);\n"
            "  let second = frame_buffer(next);\n"
            "  let nested_size = inner(nested);\n"
            "  let failure = fail();\n"
            "  let value = failure?;\n"
            "  return Ok(frame_buffer_len(first) + frame_buffer_len(second) + nested_size + value);\n"
            "}\n"
        )
        released_sizes: list[int] = []
        original_release = _FrameBuffer.release

        def record_release(resource: _FrameBuffer) -> None:
            released_sizes.append(resource.length)
            original_release(resource)

        with patch.object(_FrameBuffer, "release", record_release):
            result = run_source(source, "outer", 5, 6, 7)

        self.assertEqual(result, ResultValue.err(1))
        self.assertEqual(released_sizes, [7, 6, 5])

    def test_frame_resources_release_on_success_and_reject_invalid_size(self) -> None:
        source = (
            "fn measure(size: u64) -> u64 {\n"
            "  let bytes = frame_buffer(size);\n"
            "  return frame_buffer_len(bytes);\n"
            "}\n"
            "fn ok(value: u64) -> Result<u64, IoError> { return Ok(value); }\n"
            "fn propagate(size: u64) -> Result<u64, IoError> {\n"
            "  let bytes = frame_buffer(size);\n"
            "  let result = ok(size);\n"
            "  let value = result?;\n"
            "  return Ok(frame_buffer_len(bytes) + value);\n"
            "}\n"
        )
        released_sizes: list[int] = []
        original_release = _FrameBuffer.release

        def record_release(resource: _FrameBuffer) -> None:
            released_sizes.append(resource.length)
            original_release(resource)

        with patch.object(_FrameBuffer, "release", record_release):
            self.assertEqual(run_source(source, "measure", 4), 4)
            self.assertEqual(
                run_source(source, "propagate", 4),
                ResultValue.ok(8),
            )
        self.assertEqual(released_sizes, [4, 4])

        with self.assertRaisesRegex(AetherRuntimeError, "size is zero"):
            run_source(source, "measure", 0)
        with self.assertRaisesRegex(AetherRuntimeError, "allocation limit"):
            run_source(source, "measure", (1 << 64) - 1)

    def test_frame_buffer_alias_return_result_and_temporary_ownership(self) -> None:
        source = (
            "fn identity(bytes: FrameBuffer) -> FrameBuffer { return bytes; }\n"
            "fn alias(size: u64) -> u64 {\n"
            "  let bytes = frame_buffer(size);\n"
            "  let duplicate = bytes;\n"
            "  return frame_buffer_len(bytes) + frame_buffer_len(duplicate);\n"
            "}\n"
            "fn rebind(size: u64) -> u64 {\n"
            "  var bytes = frame_buffer(size);\n"
            "  bytes = bytes;\n"
            "  return frame_buffer_len(bytes);\n"
            "}\n"
            "fn return_ok(size: u64) -> Result<FrameBuffer, IoError> {\n"
            "  let bytes = frame_buffer(size);\n"
            "  return Ok(bytes);\n"
            "}\n"
            "fn inline_ok(size: u64) -> Result<FrameBuffer, IoError> {\n"
            "  let bytes = Ok(frame_buffer(size))?;\n"
            "  return Ok(bytes);\n"
            "}\n"
            "fn unwrap(input: Result<FrameBuffer, IoError>) "
            "-> Result<FrameBuffer, IoError> {\n"
            "  let bytes = input?;\n"
            "  return Ok(bytes);\n"
            "}\n"
            "fn discard(size: u64) -> u64 {\n"
            "  identity(frame_buffer(size));\n"
            "  return size - size;\n"
            "}\n"
        )
        original = _FrameBuffer(bytearray(4))
        returned = run_source(source, "identity", original)
        self.assertIsNot(returned, original)
        self.assertIs(returned.data, original.data)
        self.assertEqual(original.refcount, 2)
        returned.release()
        self.assertEqual(original.refcount, 1)
        original.release()
        self.assertEqual(original.refcount, 0)
        self.assertEqual(run_source(source, "alias", 5), 10)
        self.assertEqual(run_source(source, "rebind", 6), 6)

        result = run_source(source, "return_ok", 3)
        self.assertIsInstance(result, ResultValue)
        self.assertEqual(result.value.refcount, 1)
        result.value.release()

        inline = run_source(source, "inline_ok", 4)
        self.assertIsInstance(inline, ResultValue)
        self.assertEqual(inline.value.length, 4)
        self.assertEqual(inline.value.refcount, 1)
        inline.value.release()

        borrowed = _FrameBuffer(bytearray(2))
        unwrapped = run_source(
            source,
            "unwrap",
            ResultValue.ok(borrowed),
        )
        self.assertEqual(unwrapped.value.refcount, 2)
        unwrapped.value.release()
        borrowed.release()

        released: list[tuple[int, int]] = []
        original_release = _FrameBuffer.release

        def record_release(resource: _FrameBuffer) -> None:
            released.append((resource.length, resource.refcount))
            original_release(resource)

        with patch.object(_FrameBuffer, "release", record_release):
            self.assertEqual(run_source(source, "discard", 8), 0)
        self.assertEqual(released, [(8, 2), (8, 1)])

        overflow = _FrameBuffer(bytearray(1))
        overflow._storage.refcount = sys.maxsize
        with self.assertRaisesRegex(AetherRuntimeError, "reference count overflow"):
            overflow.retain()

    def test_var_rebinding_preserves_value_semantics_for_tensors(self) -> None:
        source = (
            "fn rebind(values: Tensor<f32, [2]>) -> f32 {\n"
            "  var current = values;\n"
            "  let snapshot = current;\n"
            "  current = current * 2.0;\n"
            "  return sum(snapshot) + sum(current);\n"
            "}\n"
        )

        self.assertEqual(run_source(source, "rebind", [1.0, 3.0]), 12.0)

    def test_executes_numeric_functions_and_user_calls(self) -> None:
        parsed = parse_source(
            "fn add(left: i32, right: i32) -> i32 { return left + right; }\n"
            "fn twice(value: i32) -> i32 { return add(value, value); }\n"
            "fn divide(left: i32, right: i32) -> i32 { return left / right; }\n"
            "fn operations(left: i32, right: i32) -> i32 {\n"
            "  let remainder = left % right;\n"
            "  return -(remainder + left ** 2);\n"
            "}\n"
            "fn logic(left: bool, right: bool) -> bool {\n"
            "  return left && right || !left;\n"
            "}\n"
            "fn fail() -> bool { return 1 / 0 == 0; }\n"
            "fn short_circuit() -> bool { return false && fail(); }\n"
            "fn stats(values: Tensor<f32, [2]>) -> f32 { return sum(values); }\n"
            "fn average(values: Tensor<i32, [2]>) -> f32 { return mean(values); }\n"
            "fn size(values: Tensor<f32, [2]>) -> i64 { return len(values); }\n"
            "fn extremes(values: Tensor<f32, [2]>) -> Tensor<f32, [2]> {\n"
            "  return max(abs(values), values);\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        self.assertEqual(interpreter.run("twice", 21), 42)
        self.assertEqual(interpreter.run("divide", -3, 2), -1)
        self.assertEqual(interpreter.run("operations", -5, 2), -24)
        self.assertTrue(interpreter.run("logic", True, True))
        self.assertFalse(interpreter.run("logic", True, False))
        self.assertTrue(interpreter.run("logic", False, False))
        self.assertFalse(interpreter.run("short_circuit"))
        self.assertEqual(interpreter.run("stats", [2.0, 4.0]), 6.0)
        self.assertEqual(interpreter.run("average", [1, 2]), 1.5)
        self.assertEqual(interpreter.run("size", [2.0, 4.0]), 2)
        self.assertEqual(interpreter.run("extremes", [-2.0, 1.0]), [2.0, 1.0])

    def test_f32_rounds_inputs_intermediate_results_reductions_and_gradients(self) -> None:
        source = (
            "fn identity(value: f32) -> f32 { return value; }\n"
            "fn add_one(value: f32) -> f32 { return value + 1.0; }\n"
            "fn multiply(left: f32, right: f32) -> f32 { return left * right; }\n"
            "fn total(values: Tensor<f32, [4]>) -> f32 { return sum(values); }\n"
            "fn average(values: Tensor<f32, [4]>) -> f32 { return mean(values); }\n"
            "fn objective(value: f32) -> f32 {\n"
            "  let square = value * value;\n"
            "  return square + value;\n"
            "}\n"
            "fn derivative(value: f32) -> f32 {\n"
            "  return gradient(objective, value);\n"
            "}\n"
            "fn tensor_objective(values: Tensor<f32, [4]>) -> f32 {\n"
            "  return sum(values * 2.0);\n"
            "}\n"
            "fn tensor_derivative(values: Tensor<f32, [4]>) "
            "-> Tensor<f32, [4]> { return gradient(tensor_objective, values); }\n"
        )
        values = [16777216.0, 1.0, -16777216.0, 1.0]
        interpreter = Interpreter.from_source(parse_source(source).ast)

        self.assertEqual(interpreter.run("identity", 16777217), 16777216.0)
        self.assertEqual(interpreter.run("add_one", 16777216.0), 16777216.0)
        self.assertTrue(math.isinf(interpreter.run("multiply", 3.4e38, 2.0)))
        subnormal = interpreter.run("multiply", 1.0e-30, 1.0e-10)
        self.assertGreater(subnormal, 0.0)
        self.assertLess(subnormal, 1.1754943508222875e-38)
        self.assertEqual(interpreter.run("total", values), 1.0)
        self.assertEqual(interpreter.run("average", values), 0.25)
        self.assertEqual(interpreter.run("derivative", 4097.25), 8195.5)
        self.assertEqual(
            interpreter.run("tensor_derivative", values),
            [2.0, 2.0, 2.0, 2.0],
        )

    def test_f64_reductions_accumulate_sequentially_in_row_major_order(self) -> None:
        parsed = parse_source(
            "fn total(values: Tensor<f64, [2, 2]>) -> f64 {\n"
            "  return sum(values);\n"
            "}\n"
            "fn average(values: Tensor<f64, [2, 2]>) -> f64 {\n"
            "  return mean(values);\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        first_order = [[1e16, 1.0], [-1e16, 1.0]]
        second_order = [[1e16, -1e16], [1.0, 1.0]]
        self.assertEqual(interpreter.run("total", first_order), 1.0)
        self.assertEqual(interpreter.run("average", first_order), 0.25)
        self.assertEqual(interpreter.run("total", second_order), 2.0)
        self.assertEqual(interpreter.run("average", second_order), 0.5)

    def test_reverse_mode_gradient_matches_analytic_scalar_derivative(self) -> None:
        source = (
            "fn objective(x: f64) -> f64 {\n"
            "  let square = x * x;\n"
            "  return square * square + x;\n"
            "}\n"
            "fn derivative(x: f64) -> f64 {\n"
            "  return gradient(objective, x);\n"
            "}\n"
        )

        self.assertAlmostEqual(run_source(source, "derivative", 2.0), 33.0)
        self.assertAlmostEqual(run_source(source, "derivative", -1.0), -3.0)

    def test_scalar_autodiff_supports_linear_unit_conversion(self) -> None:
        source = (
            "fn objective(value: f64) -> f64 {\n"
            "  return convert(value, \"1\");\n"
            "}\n"
            "fn derivative(value: f64) -> f64 {\n"
            "  return gradient(objective, value);\n"
            "}\n"
        )
        result = Interpreter.from_source(parse_source(source).ast).run(
            "derivative", 1.0
        )
        self.assertEqual(result, 1.0)

    def test_tensor_reverse_mode_gradient_handles_reductions_and_shared_values(self) -> None:
        parsed = parse_source(
            "fn objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  let doubled = values * 2.0;\n"
            "  return sum(doubled + values);\n"
            "}\n"
            "fn derivative(values: Tensor<f32, [2]>) -> Tensor<f32, [2]> {\n"
            "  return gradient(objective, values);\n"
            "}\n"
            "fn average(values: Tensor<f32, [2]>) -> f32 {\n"
            "  return mean(values * 2.0);\n"
            "}\n"
            "fn average_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> { return gradient(average, values); }\n"
            "fn negative(values: Tensor<f32, [2]>) -> f32 {\n"
            "  return sum(-values + 2.0);\n"
            "}\n"
            "fn negative_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> { return gradient(negative, values); }\n"
            "fn scalar_product(values: Tensor<f32, [2]>) -> f32 {\n"
            "  return mean(values) * sum(values);\n"
            "}\n"
            "fn scalar_product_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> { return gradient(scalar_product, values); }\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        self.assertEqual(interpreter.run("derivative", [1.0, -3.0]), [3.0, 3.0])
        self.assertEqual(
            interpreter.run("average_gradient", [1.0, -3.0]),
            [1.0, 1.0],
        )
        self.assertEqual(interpreter.run("negative_gradient", [1.0, -3.0]), [-1.0, -1.0])
        self.assertEqual(
            interpreter.run("scalar_product_gradient", [1.0, -3.0]),
            [-2.0, -2.0],
        )

    def test_f32_tensor_gradient_chains_shared_reductions_and_unit_conversion(self) -> None:
        source = (
            "fn objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  let scaled = values * 0.1;\n"
            "  let average = mean(scaled);\n"
            "  let total = sum(scaled);\n"
            "  return total + average * 3.0;\n"
            "}\n"
            "fn derivative(values: Tensor<f32, [2]>) -> Tensor<f32, [2]> {\n"
            "  return gradient(objective, values);\n"
            "}\n"
            "fn objective_cm(values: Tensor<f32, [2], cm>) -> f32<m> {\n"
            "  let meters = convert(values, \"m\");\n"
            "  let average = mean(meters);\n"
            "  let total = sum(meters);\n"
            "  return average + total * 0.25;\n"
            "}\n"
            "fn derivative_cm(values: Tensor<f32, [2], cm>) "
            "-> Tensor<f32, [2], m/cm> {\n"
            "  return gradient(objective_cm, values);\n"
            "}\n"
        )
        parsed = parse_source(source)
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)
        values = [1.1, -3.2]

        gradient = interpreter.run("derivative", values)
        self.assertEqual(gradient, [0.25, 0.25])

        unit_gradient = interpreter.run("derivative_cm", values)
        self.assertIsInstance(unit_gradient, Quantity)
        self.assertEqual(unit_gradient.type.unit_symbol, "m/cm")
        for component in unit_gradient.value:
            self.assertAlmostEqual(component, 0.0075, delta=1e-9)

    def test_f32_autodiff_division_gradients(self) -> None:
        source = (
            "fn tensor_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  let divisor = mean(values) + 2.0;\n"
            "  return sum(values / divisor);\n"
            "}\n"
            "fn tensor_gradient(values: Tensor<f32, [2]>) -> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_objective, values);\n"
            "}\n"
            "fn tensor_small_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  let divisor = mean(values) + 1e-10;\n"
            "  return sum(values / divisor);\n"
            "}\n"
            "fn tensor_small_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_small_objective, values);\n"
            "}\n"
            "fn scalar_objective(divisor: f32) -> f32 {\n"
            "  return 3.0 / divisor;\n"
            "}\n"
            "fn scalar_gradient(divisor: f32) -> f32 {\n"
            "  return gradient(scalar_objective, divisor);\n"
            "}\n"
        )
        parsed = parse_source(source)
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        for values, expected in (
            ([1.0, 3.0], [0.125, 0.125]),
            ([-5.0, -3.0], [0.5, 0.5]),
        ):
            with self.subTest(values=values):
                actual = interpreter.run("tensor_gradient", values)
                for component, reference in zip(actual, expected):
                    self.assertAlmostEqual(component, reference, delta=max(1e-6, abs(reference) * 1e-6))

        small = interpreter.run("tensor_small_gradient", [0.0, 0.0])
        for component in small:
            self.assertAlmostEqual(component, 1.0e10, delta=1.0e4)

        for divisor in (2.0, -2.0, 1.0e-10):
            with self.subTest(divisor=divisor):
                actual = interpreter.run("scalar_gradient", divisor)
                expected = -3.0 / (divisor * divisor)
                self.assertAlmostEqual(actual, expected, delta=max(1e-6, abs(expected) * 1e-6))

    def test_f32_autodiff_adjoint_accumulation_with_cancellation(self) -> None:
        source = (
            "fn cancellation_objective(value: f32) -> f32 {\n"
            "  let large = value * 16777216.0;\n"
            "  let small = value;\n"
            "  let negative = value * -16777216.0;\n"
            "  return large + small + negative;\n"
            "}\n"
            "fn cancellation_gradient(value: f32) -> f32 {\n"
            "  return gradient(cancellation_objective, value);\n"
            "}\n"
            "fn full_cancellation_objective(value: f32) -> f32 {\n"
            "  let large = value * 16777216.0;\n"
            "  let negative = value * -16777216.0;\n"
            "  return large + negative;\n"
            "}\n"
            "fn full_cancellation_gradient(value: f32) -> f32 {\n"
            "  return gradient(full_cancellation_objective, value);\n"
            "}\n"
            "fn tensor_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  let large = values * 16777216.0;\n"
            "  let small = values;\n"
            "  let negative = values * -16777216.0;\n"
            "  return sum(large + small + negative);\n"
            "}\n"
            "fn tensor_gradient(values: Tensor<f32, [2]>) -> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_objective, values);\n"
            "}\n"
        )
        parsed = parse_source(source)
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        self.assertEqual(interpreter.run("cancellation_gradient", 1.0), 1.0)
        self.assertEqual(interpreter.run("full_cancellation_gradient", 1.0), 0.0)
        self.assertEqual(interpreter.run("tensor_gradient", [1.0, -2.0]), [1.0, 1.0])

    def test_f32_autodiff_accumulates_many_cancelling_paths(self) -> None:
        source = (
            "fn many_paths_objective(value: f32) -> f32 {\n"
            "  let p1 = value * 16777216.0;\n"
            "  let p2 = value * 16777216.0;\n"
            "  let p3 = value * 16777216.0;\n"
            "  let p4 = value * 16777216.0;\n"
            "  let residual = value;\n"
            "  let n1 = value * -16777216.0;\n"
            "  let n2 = value * -16777216.0;\n"
            "  let n3 = value * -16777216.0;\n"
            "  let n4 = value * -16777216.0;\n"
            "  return p1 + p2 + p3 + p4 + residual + n1 + n2 + n3 + n4;\n"
            "}\n"
            "fn many_paths_gradient(value: f32) -> f32 {\n"
            "  return gradient(many_paths_objective, value);\n"
            "}\n"
            "fn residual_first_objective(value: f32) -> f32 {\n"
            "  let residual = value;\n"
            "  let p1 = value * 16777216.0;\n"
            "  let n1 = value * -16777216.0;\n"
            "  let p2 = value * 16777216.0;\n"
            "  let n2 = value * -16777216.0;\n"
            "  let p3 = value * 16777216.0;\n"
            "  let n3 = value * -16777216.0;\n"
            "  let p4 = value * 16777216.0;\n"
            "  let n4 = value * -16777216.0;\n"
            "  return residual + p1 + n1 + p2 + n2 + p3 + n3 + p4 + n4;\n"
            "}\n"
            "fn residual_first_gradient(value: f32) -> f32 {\n"
            "  return gradient(residual_first_objective, value);\n"
            "}\n"
            "fn many_tensor_paths_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  let p1 = values * 16777216.0;\n"
            "  let p2 = values * 16777216.0;\n"
            "  let p3 = values * 16777216.0;\n"
            "  let residual = values;\n"
            "  let n1 = values * -16777216.0;\n"
            "  let n2 = values * -16777216.0;\n"
            "  let n3 = values * -16777216.0;\n"
            "  return sum(p1 + p2 + p3 + residual + n1 + n2 + n3);\n"
            "}\n"
            "fn many_tensor_paths_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> {\n"
            "  return gradient(many_tensor_paths_objective, values);\n"
            "}\n"
        )
        parsed = parse_source(source)
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        self.assertEqual(interpreter.run("many_paths_gradient", 1.0), 1.0)
        self.assertEqual(interpreter.run("residual_first_gradient", 1.0), 1.0)
        self.assertEqual(
            interpreter.run("many_tensor_paths_gradient", [1.0, -2.0]),
            [1.0, 1.0],
        )

    def test_f32_autodiff_gradients_at_subnormal_and_overflow_boundaries(self) -> None:
        source = (
            "fn precision_objective(value: f32) -> f32 {\n"
            "  let branch = value * value;\n"
            "  return branch + branch;\n"
            "}\n"
            "fn precision_gradient(value: f32) -> f32 {\n"
            "  return gradient(precision_objective, value);\n"
            "}\n"
            "fn shared_objective(value: f32) -> f32 {\n"
            "  let branch = value * 0.1;\n"
            "  return branch + branch;\n"
            "}\n"
            "fn shared_gradient(value: f32) -> f32 {\n"
            "  return gradient(shared_objective, value);\n"
            "}\n"
            "fn subnormal_objective(value: f32) -> f32 {\n"
            "  let branch = value * 1e-40;\n"
            "  return branch + branch;\n"
            "}\n"
            "fn subnormal_gradient(value: f32) -> f32 {\n"
            "  return gradient(subnormal_objective, value);\n"
            "}\n"
            "fn overflow_objective(value: f32) -> f32 {\n"
            "  let branch = value * 3e38;\n"
            "  return branch + branch;\n"
            "}\n"
            "fn overflow_gradient(value: f32) -> f32 {\n"
            "  return gradient(overflow_objective, value);\n"
            "}\n"
            "fn tensor_subnormal_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  let branch = values * 1e-40;\n"
            "  return sum(branch + branch);\n"
            "}\n"
            "fn tensor_subnormal_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_subnormal_objective, values);\n"
            "}\n"
            "fn tensor_overflow_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  let branch = values * 3e38;\n"
            "  return sum(branch + branch);\n"
            "}\n"
            "fn tensor_overflow_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_overflow_objective, values);\n"
            "}\n"
        )
        parsed = parse_source(source)
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)
        f32 = lambda value: struct.unpack("=f", struct.pack("=f", value))[0]

        precision = interpreter.run("precision_gradient", 16777216.0)
        self.assertEqual(precision, 67108864.0)

        shared = interpreter.run("shared_gradient", 16777216.0)
        self.assertEqual(shared, f32(f32(0.1) * 2.0))

        subnormal = interpreter.run("subnormal_gradient", 1.0)
        self.assertEqual(subnormal, f32(f32(1e-40) * 2.0))
        self.assertGreater(subnormal, 0.0)
        self.assertLess(subnormal, 1.1754943508222875e-38)

        overflow = interpreter.run("overflow_gradient", 0.0)
        self.assertTrue(math.isinf(overflow))
        self.assertGreater(overflow, 0.0)

        tensor_subnormal = interpreter.run("tensor_subnormal_gradient", [1.0, -1.0])
        self.assertEqual(
            tensor_subnormal,
            [f32(f32(1e-40) * 2.0), f32(f32(1e-40) * 2.0)],
        )
        self.assertTrue(
            all(0.0 < value < 1.1754943508222875e-38 for value in tensor_subnormal)
        )

        tensor_overflow = interpreter.run("tensor_overflow_gradient", [0.0, 0.0])
        self.assertTrue(all(math.isinf(value) and value > 0.0 for value in tensor_overflow))

    def test_f32_autodiff_rounds_accumulated_gradients_ties_to_even(self) -> None:
        source = (
            "fn tie_objective(value: f32) -> f32 {\n"
            "  let scaled = value * 5.9604644775390625e-8;\n"
            "  return value + scaled;\n"
            "}\n"
            "fn tie_gradient(value: f32) -> f32 {\n"
            "  return gradient(tie_objective, value);\n"
            "}\n"
            "fn above_objective(value: f32) -> f32 {\n"
            "  let scaled = value * 5.9604651880817983e-8;\n"
            "  return value + scaled;\n"
            "}\n"
            "fn above_gradient(value: f32) -> f32 {\n"
            "  return gradient(above_objective, value);\n"
            "}\n"
            "fn tie_odd_objective(value: f32) -> f32 {\n"
            "  let scaled = value * 1.78813934326171875e-7;\n"
            "  return value + scaled;\n"
            "}\n"
            "fn tie_odd_gradient(value: f32) -> f32 {\n"
            "  return gradient(tie_odd_objective, value);\n"
            "}\n"
            "fn tie_negative_objective(value: f32) -> f32 {\n"
            "  let scaled = value * 1.78813934326171875e-7;\n"
            "  return -(value + scaled);\n"
            "}\n"
            "fn tie_negative_gradient(value: f32) -> f32 {\n"
            "  return gradient(tie_negative_objective, value);\n"
            "}\n"
            "fn tensor_tie_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  let scaled = values * 5.9604644775390625e-8;\n"
            "  return sum(values + scaled);\n"
            "}\n"
            "fn tensor_tie_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_tie_objective, values);\n"
            "}\n"
            "fn tensor_above_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  let scaled = values * 5.9604651880817983e-8;\n"
            "  return sum(values + scaled);\n"
            "}\n"
            "fn tensor_above_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_above_objective, values);\n"
            "}\n"
            "fn tensor_tie_odd_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  let scaled = values * 1.78813934326171875e-7;\n"
            "  return sum(values + scaled);\n"
            "}\n"
            "fn tensor_tie_odd_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_tie_odd_objective, values);\n"
            "}\n"
            "fn tensor_tie_negative_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  let scaled = values * 1.78813934326171875e-7;\n"
            "  return -sum(values + scaled);\n"
            "}\n"
            "fn tensor_tie_negative_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_tie_negative_objective, values);\n"
            "}\n"
        )
        parsed = parse_source(source)
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        tie = interpreter.run("tie_gradient", 1.0)
        above = interpreter.run("above_gradient", 1.0)
        tie_odd = interpreter.run("tie_odd_gradient", 1.0)
        tie_negative = interpreter.run("tie_negative_gradient", 1.0)
        tensor_tie = interpreter.run("tensor_tie_gradient", [1.0, -2.0])
        tensor_above = interpreter.run("tensor_above_gradient", [1.0, -2.0])
        tensor_tie_odd = interpreter.run("tensor_tie_odd_gradient", [1.0, -2.0])
        tensor_tie_negative = interpreter.run(
            "tensor_tie_negative_gradient", [1.0, -2.0]
        )

        next_f32 = struct.unpack("=f", struct.pack("=f", 1.0 + 2.0**-23))[0]
        upper_even_f32 = struct.unpack("=f", struct.pack("=f", 1.0 + 2.0**-22))[0]
        self.assertEqual(tie, 1.0)
        self.assertEqual(above, next_f32)
        self.assertEqual(tie_odd, upper_even_f32)
        self.assertEqual(tie_negative, -upper_even_f32)
        self.assertEqual(tensor_tie, [1.0, 1.0])
        self.assertEqual(tensor_above, [next_f32, next_f32])
        self.assertEqual(tensor_tie_odd, [upper_even_f32, upper_even_f32])
        self.assertEqual(
            tensor_tie_negative, [-upper_even_f32, -upper_even_f32]
        )

    def test_f32_autodiff_rounds_gradients_at_subnormal_ties_to_even(self) -> None:
        source = (
            "fn half_subnormal_objective(value: f32) -> f32 {\n"
            "  let scaled = value * 1.401298464324817e-45;\n"
            "  return scaled * 0.5;\n"
            "}\n"
            "fn half_subnormal_gradient(value: f32) -> f32 {\n"
            "  return gradient(half_subnormal_objective, value);\n"
            "}\n"
            "fn negative_half_subnormal_objective(value: f32) -> f32 {\n"
            "  let scaled = value * 1.401298464324817e-45;\n"
            "  return -(scaled * 0.5);\n"
            "}\n"
            "fn negative_half_subnormal_gradient(value: f32) -> f32 {\n"
            "  return gradient(negative_half_subnormal_objective, value);\n"
            "}\n"
            "fn odd_subnormal_tie_objective(value: f32) -> f32 {\n"
            "  let scaled = value * 1.401298464324817e-45;\n"
            "  return scaled * 1.5;\n"
            "}\n"
            "fn odd_subnormal_tie_gradient(value: f32) -> f32 {\n"
            "  return gradient(odd_subnormal_tie_objective, value);\n"
            "}\n"
            "fn negative_odd_subnormal_tie_objective(value: f32) -> f32 {\n"
            "  let scaled = value * 1.401298464324817e-45;\n"
            "  return -(scaled * 1.5);\n"
            "}\n"
            "fn negative_odd_subnormal_tie_gradient(value: f32) -> f32 {\n"
            "  return gradient(negative_odd_subnormal_tie_objective, value);\n"
            "}\n"
            "fn tensor_half_subnormal_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  return mean(values * 1.401298464324817e-45);\n"
            "}\n"
            "fn tensor_half_subnormal_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_half_subnormal_objective, values);\n"
            "}\n"
            "fn tensor_negative_half_subnormal_objective(values: Tensor<f32, [2]>) "
            "-> f32 {\n"
            "  return -mean(values * 1.401298464324817e-45);\n"
            "}\n"
            "fn tensor_negative_half_subnormal_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_negative_half_subnormal_objective, values);\n"
            "}\n"
            "fn tensor_odd_subnormal_tie_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  return mean(values * 1.401298464324817e-45) * 3.0;\n"
            "}\n"
            "fn tensor_odd_subnormal_tie_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_odd_subnormal_tie_objective, values);\n"
            "}\n"
            "fn tensor_negative_odd_subnormal_tie_objective("
            "values: Tensor<f32, [2]>) -> f32 {\n"
            "  return -(mean(values * 1.401298464324817e-45) * 3.0);\n"
            "}\n"
            "fn tensor_negative_odd_subnormal_tie_gradient("
            "values: Tensor<f32, [2]>) -> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_negative_odd_subnormal_tie_objective, values);\n"
            "}\n"
        )
        parsed = parse_source(source)
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        half = interpreter.run("half_subnormal_gradient", 1.0)
        negative_half = interpreter.run("negative_half_subnormal_gradient", 1.0)
        odd_tie = interpreter.run("odd_subnormal_tie_gradient", 1.0)
        negative_odd_tie = interpreter.run(
            "negative_odd_subnormal_tie_gradient", 1.0
        )
        tensor_half = interpreter.run(
            "tensor_half_subnormal_gradient", [1.0, -2.0]
        )
        tensor_negative_half = interpreter.run(
            "tensor_negative_half_subnormal_gradient", [1.0, -2.0]
        )
        tensor_odd_tie = interpreter.run(
            "tensor_odd_subnormal_tie_gradient", [1.0, -2.0]
        )
        tensor_negative_odd_tie = interpreter.run(
            "tensor_negative_odd_subnormal_tie_gradient", [1.0, -2.0]
        )

        min_subnormal = struct.unpack("=f", struct.pack("=I", 1))[0]
        second_subnormal = struct.unpack("=f", struct.pack("=I", 2))[0]
        self.assertEqual(half, 0.0)
        self.assertEqual(math.copysign(1.0, half), 1.0)
        self.assertEqual(negative_half, 0.0)
        self.assertEqual(math.copysign(1.0, negative_half), -1.0)
        self.assertEqual(odd_tie, second_subnormal)
        self.assertEqual(negative_odd_tie, -second_subnormal)
        self.assertEqual(tensor_half, [0.0, 0.0])
        self.assertEqual(tensor_negative_half, [-0.0, -0.0])
        self.assertEqual(tensor_odd_tie, [second_subnormal, second_subnormal])
        self.assertEqual(
            tensor_negative_odd_tie,
            [-second_subnormal, -second_subnormal],
        )
        self.assertGreater(min_subnormal, 0.0)

    def test_f32_autodiff_rounds_gradients_around_two_and_half_subnormals(self) -> None:
        source = (
            "fn below_objective(value: f32) -> f32 {\n"
            "  let scaled = value * 1.401298464324817e-45;\n"
            "  return scaled * 2.4999997615814209;\n"
            "}\n"
            "fn below_gradient(value: f32) -> f32 {\n"
            "  return gradient(below_objective, value);\n"
            "}\n"
            "fn tie_objective(value: f32) -> f32 {\n"
            "  let scaled = value * 1.401298464324817e-45;\n"
            "  return scaled * 2.5;\n"
            "}\n"
            "fn tie_gradient(value: f32) -> f32 {\n"
            "  return gradient(tie_objective, value);\n"
            "}\n"
            "fn above_objective(value: f32) -> f32 {\n"
            "  let scaled = value * 1.401298464324817e-45;\n"
            "  return scaled * 2.5000002384185791;\n"
            "}\n"
            "fn above_gradient(value: f32) -> f32 {\n"
            "  return gradient(above_objective, value);\n"
            "}\n"
            "fn negative_below_objective(value: f32) -> f32 {\n"
            "  let scaled = value * 1.401298464324817e-45;\n"
            "  return -(scaled * 2.4999997615814209);\n"
            "}\n"
            "fn negative_below_gradient(value: f32) -> f32 {\n"
            "  return gradient(negative_below_objective, value);\n"
            "}\n"
            "fn negative_tie_objective(value: f32) -> f32 {\n"
            "  let scaled = value * 1.401298464324817e-45;\n"
            "  return -(scaled * 2.5);\n"
            "}\n"
            "fn negative_tie_gradient(value: f32) -> f32 {\n"
            "  return gradient(negative_tie_objective, value);\n"
            "}\n"
            "fn negative_above_objective(value: f32) -> f32 {\n"
            "  let scaled = value * 1.401298464324817e-45;\n"
            "  return -(scaled * 2.5000002384185791);\n"
            "}\n"
            "fn negative_above_gradient(value: f32) -> f32 {\n"
            "  return gradient(negative_above_objective, value);\n"
            "}\n"
            "fn tensor_below_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  return mean(values * 1.401298464324817e-45) * 4.999999523162842;\n"
            "}\n"
            "fn tensor_below_gradient(values: Tensor<f32, [2]>) -> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_below_objective, values);\n"
            "}\n"
            "fn tensor_tie_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  return mean(values * 1.401298464324817e-45) * 5.0;\n"
            "}\n"
            "fn tensor_tie_gradient(values: Tensor<f32, [2]>) -> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_tie_objective, values);\n"
            "}\n"
            "fn tensor_above_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  return mean(values * 1.401298464324817e-45) * 5.000000476837158;\n"
            "}\n"
            "fn tensor_above_gradient(values: Tensor<f32, [2]>) -> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_above_objective, values);\n"
            "}\n"
            "fn tensor_negative_below_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  return -(mean(values * 1.401298464324817e-45) * 4.999999523162842);\n"
            "}\n"
            "fn tensor_negative_below_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_negative_below_objective, values);\n"
            "}\n"
            "fn tensor_negative_tie_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  return -(mean(values * 1.401298464324817e-45) * 5.0);\n"
            "}\n"
            "fn tensor_negative_tie_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_negative_tie_objective, values);\n"
            "}\n"
            "fn tensor_negative_above_objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  return -(mean(values * 1.401298464324817e-45) * 5.000000476837158);\n"
            "}\n"
            "fn tensor_negative_above_gradient(values: Tensor<f32, [2]>) "
            "-> Tensor<f32, [2]> {\n"
            "  return gradient(tensor_negative_above_objective, values);\n"
            "}\n"
        )
        parsed = parse_source(source)
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        min_subnormal = struct.unpack("=f", struct.pack("=I", 1))[0]
        even_result = struct.unpack("=f", struct.pack("=I", 2))[0]
        odd_result = struct.unpack("=f", struct.pack("=I", 3))[0]
        cases = (
            ("below_gradient", even_result),
            ("tie_gradient", even_result),
            ("above_gradient", odd_result),
            ("negative_below_gradient", -even_result),
            ("negative_tie_gradient", -even_result),
            ("negative_above_gradient", -odd_result),
        )
        for name, expected in cases:
            with self.subTest(function=name):
                self.assertEqual(interpreter.run(name, 1.0), expected)

        tensor_cases = (
            ("tensor_below_gradient", [even_result, even_result]),
            ("tensor_tie_gradient", [even_result, even_result]),
            ("tensor_above_gradient", [odd_result, odd_result]),
            ("tensor_negative_below_gradient", [-even_result, -even_result]),
            ("tensor_negative_tie_gradient", [-even_result, -even_result]),
            ("tensor_negative_above_gradient", [-odd_result, -odd_result]),
        )
        for name, expected in tensor_cases:
            with self.subTest(function=name):
                self.assertEqual(interpreter.run(name, [1.0, -2.0]), expected)
        self.assertEqual(even_result, 2.0 * min_subnormal)
        self.assertEqual(odd_result, 3.0 * min_subnormal)

    def test_f32_autodiff_rounds_gradients_around_half_subnormal(self) -> None:
        scalar_coefficients = {
            "below": "0.4999999701976776",
            "tie": "0.5",
            "above": "0.5000000596046448",
        }
        tensor_coefficients = {
            "below": "0.9999999403953552",
            "tie": "1.0",
            "above": "1.0000001192092896",
        }
        declarations: list[str] = []
        for prefix, coefficient in scalar_coefficients.items():
            for sign, unary in (("", ""), ("negative_", "-")):
                objective = f"{sign}{prefix}_objective"
                gradient = f"{sign}{prefix}_gradient"
                declarations.append(
                    f"fn {objective}(value: f32) -> f32 {{ "
                    f"let scaled = value * 1.401298464324817e-45; "
                    f"return {unary}(scaled * {coefficient}); }}"
                )
                declarations.append(
                    f"fn {gradient}(value: f32) -> f32 {{ "
                    f"return gradient({objective}, value); }}"
                )
        for prefix, coefficient in tensor_coefficients.items():
            for sign, unary in (("", ""), ("negative_", "-")):
                objective = f"tensor_{sign}{prefix}_objective"
                gradient = f"tensor_{sign}{prefix}_gradient"
                declarations.append(
                    f"fn {objective}(values: Tensor<f32, [2]>) -> f32 {{ "
                    f"return {unary}(mean(values * 1.401298464324817e-45) "
                    f"* {coefficient}); }}"
                )
                declarations.append(
                    f"fn {gradient}(values: Tensor<f32, [2]>) "
                    f"-> Tensor<f32, [2]> {{ "
                    f"return gradient({objective}, values); }}"
                )
        parsed = parse_source("\n".join(declarations))
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        min_subnormal = struct.unpack("=f", struct.pack("=I", 1))[0]
        scalar_cases = (
            ("below_gradient", 0.0),
            ("negative_below_gradient", -0.0),
            ("tie_gradient", 0.0),
            ("negative_tie_gradient", -0.0),
            ("above_gradient", min_subnormal),
            ("negative_above_gradient", -min_subnormal),
        )
        for name, expected in scalar_cases:
            with self.subTest(function=name):
                actual = interpreter.run(name, 1.0)
                self.assertEqual(actual, expected)
                if expected == 0.0:
                    self.assertEqual(
                        math.copysign(1.0, actual),
                        math.copysign(1.0, expected),
                    )

        tensor_cases = (
            ("tensor_below_gradient", [0.0, 0.0]),
            ("tensor_negative_below_gradient", [-0.0, -0.0]),
            ("tensor_tie_gradient", [0.0, 0.0]),
            ("tensor_negative_tie_gradient", [-0.0, -0.0]),
            ("tensor_above_gradient", [min_subnormal, min_subnormal]),
            (
                "tensor_negative_above_gradient",
                [-min_subnormal, -min_subnormal],
            ),
        )
        for name, expected in tensor_cases:
            with self.subTest(function=name):
                actual = interpreter.run(name, [1.0, -2.0])
                self.assertEqual(actual, expected)
                for actual_value, expected_value in zip(actual, expected):
                    if expected_value == 0.0:
                        self.assertEqual(
                            math.copysign(1.0, actual_value),
                            math.copysign(1.0, expected_value),
                        )

    def test_tensor_reverse_mode_gradient_supports_f64_multidimensional_tensors(self) -> None:
        parsed = parse_source(
            "fn objective(values: Tensor<f64, [2, 2]>) -> f64 {\n"
            "  let average = mean(values);\n"
            "  return sum(average + values);\n"
            "}\n"
            "fn derivative(values: Tensor<f64, [2, 2]>) "
            "-> Tensor<f64, [2, 2]> { return gradient(objective, values); }\n"
            "fn ratio(values: Tensor<f64, [2, 2]>) -> f64 {\n"
            "  return sum(values / mean(values));\n"
            "}\n"
            "fn ratio_gradient(values: Tensor<f64, [2, 2]>) "
            "-> Tensor<f64, [2, 2]> { return gradient(ratio, values); }\n"
            "fn single(values: Tensor<f64, [1]>) -> f64 { return mean(values); }\n"
            "fn single_gradient(values: Tensor<f64, [1]>) "
            "-> Tensor<f64, [1]> { return gradient(single, values); }\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        self.assertEqual(
            interpreter.run("derivative", [[1.0, 2.0], [3.0, 4.0]]),
            [[2.0, 2.0], [2.0, 2.0]],
        )
        ratio_gradient = interpreter.run(
            "ratio_gradient",
            [[1.0, 2.0], [3.0, 4.0]],
        )
        for row in ratio_gradient:
            for value in row:
                self.assertAlmostEqual(value, 0.0, delta=1e-12)
        self.assertEqual(interpreter.run("single_gradient", [42.0]), [1.0])

    def test_f32_mean_gradient_rounds_local_derivative_factor(self) -> None:
        parsed = parse_source(
            "fn objective(values: Tensor<f32, [3]>) -> f32 {\n"
            "  return mean(values) * -6185394176.0;\n"
            "}\n"
            "fn derivative(values: Tensor<f32, [3]>) -> Tensor<f32, [3]> {\n"
            "  return gradient(objective, values);\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        self.assertEqual(
            interpreter.run("derivative", [0.0, 0.0, 0.0]),
            [-2061798144.0, -2061798144.0, -2061798144.0],
        )

    def test_tensor_reverse_mode_gradient_preserves_unit_scale(self) -> None:
        parsed = parse_source(
            "fn objective(values: Tensor<f32, [2], cm>) -> f32<m> {\n"
            "  return convert(sum(values), \"m\");\n"
            "}\n"
            "fn derivative(values: Tensor<f32, [2], cm>) "
            "-> Tensor<f32, [2], m/cm> { return gradient(objective, values); }\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        result = Interpreter.from_source(parsed.ast).run("derivative", [2.0, 5.0])

        self.assertIsInstance(result, Quantity)
        self.assertEqual(result.type.unit_symbol, "m/cm")
        for value in result.value:
            self.assertAlmostEqual(value, 0.01, delta=1e-8)

    def test_tensor_reverse_mode_gradient_rejects_division_by_zero(self) -> None:
        source = (
            "fn objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  return sum(values / 0.0);\n"
            "}\n"
            "fn derivative(values: Tensor<f32, [2]>) -> Tensor<f32, [2]> {\n"
            "  return gradient(objective, values);\n"
            "}\n"
        )
        with self.assertRaisesRegex(AetherRuntimeError, "Division by zero"):
            run_source(source, "derivative", [1.0, 2.0])

    def test_reverse_mode_gradient_propagates_division_by_zero(self) -> None:
        source = (
            "fn objective(x: f32) -> f32 { return x / x; }\n"
            "fn derivative(x: f32) -> f32 { return gradient(objective, x); }\n"
        )

        with self.assertRaisesRegex(AetherRuntimeError, "Division by zero"):
            run_source(source, "derivative", 0.0)

    def test_executes_tensor_pipeline(self) -> None:
        parsed = parse_source(
            "fn center(values: Tensor<f32, [2]>) -> Tensor<f32, [2]> {\n"
            "  let average = mean(values);\n"
            "  return values |> subtract(average);\n"
            "}\n"
        )

        self.assertEqual(parsed.diagnostics, ())
        self.assertEqual(Interpreter.from_source(parsed.ast).run("center", [1.0, 3.0]), [-1.0, 1.0])

    def test_static_tensor_operations_match_inferred_shapes(self) -> None:
        parsed = parse_source(
            "fn add(values: Tensor<f32, [2, 2]>, offset: f32) "
            "-> Tensor<f32, [2, 2]> { return values + offset; }\n"
            "fn reverse_subtract(offset: f32, values: Tensor<f32, [2, 2]>) "
            "-> Tensor<f32, [2, 2]> { return offset - values; }\n"
            "fn multiply(values: Tensor<f32, [2, 2]>, factor: f32) "
            "-> Tensor<f32, [2, 2]> { return factor * values; }\n"
            "fn divide(values: Tensor<f32, [2, 2]>, divisor: f32) "
            "-> Tensor<f32, [2, 2]> { return values / divisor; }\n"
            "fn first_row(values: Tensor<f32, [2, 2]>) "
            "-> Tensor<f32, [2]> { return values[0]; }\n"
            "fn row_count(values: Tensor<f32, [2, 2]>) -> i64 { return len(values); }\n"
            "fn total(values: Tensor<f32, [2, 2]>) -> f32 { return sum(values); }\n"
            "fn average(values: Tensor<f32, [2, 2]>) -> f32 { return mean(values); }\n"
            "fn last_row(values: Tensor<f32, [2, 2]>) "
            "-> Tensor<f32, [2]> { return values[1]; }\n"
            "fn matrix_item(values: Tensor<f32, [2, 2]>) -> f32 "
            "{ return values[1][0]; }\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        matrix = [[1.0, 2.0], [3.0, 4.0]]
        self.assertEqual(
            interpreter.run("add", matrix, 2.0),
            [[3.0, 4.0], [5.0, 6.0]],
        )
        self.assertEqual(
            interpreter.run("reverse_subtract", 5.0, matrix),
            [[4.0, 3.0], [2.0, 1.0]],
        )
        self.assertEqual(
            interpreter.run("multiply", matrix, 2.0),
            [[2.0, 4.0], [6.0, 8.0]],
        )
        self.assertEqual(
            interpreter.run("divide", matrix, 2.0),
            [[0.5, 1.0], [1.5, 2.0]],
        )
        self.assertEqual(interpreter.run("first_row", matrix), [1.0, 2.0])
        self.assertEqual(interpreter.run("last_row", matrix), [3.0, 4.0])
        self.assertEqual(interpreter.run("matrix_item", matrix), 3.0)
        self.assertEqual(interpreter.run("row_count", matrix), 2)
        self.assertEqual(interpreter.run("total", matrix), 10.0)
        self.assertEqual(interpreter.run("average", matrix), 2.5)

    def test_executes_sensor_example_end_to_end(self) -> None:
        source_path = Path(__file__).parents[1] / "examples" / "sensor.hl"
        with source_path.open("r", encoding="utf-8", newline="") as source_file:
            parsed = parse_source(source_file.read())
        self.assertEqual(parsed.diagnostics, ())

        samples = [[float(row * 4 + column) for column in range(4)] for row in range(4)]
        result = Interpreter.from_source(parsed.ast).run(
            "normalize_voltage",
            {"samples": samples, "timestamp": 1},
        )

        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0][0], -1.0)
        self.assertAlmostEqual(result[3][3], 1.0)

    def test_match_handles_injected_result_and_effects_are_opt_in(self) -> None:
        parsed = parse_source(
            "@io\n"
            "fn read_or_default() -> str {\n"
            "  let result = io_read_line();\n"
            "  return match result { Ok(value) => value, Err(error) => \"fallback\" };\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())

        with self.assertRaisesRegex(AetherRuntimeError, "no injected handler"):
            Interpreter.from_source(parsed.ast).run("read_or_default")
        with self.assertRaisesRegex(AetherRuntimeError, "invalid success value"):
            Interpreter.from_source(
                parsed.ast,
                effect_handlers={"io_read_line": lambda: ResultValue.ok(False)},
            ).run("read_or_default")
        interpreter = Interpreter.from_source(
            parsed.ast,
            effect_handlers={
                "io_read_line": lambda: ResultValue.err("offline")
            },
        )
        self.assertEqual(interpreter.run("read_or_default"), "fallback")

    def test_result_parameters_validate_success_payloads(self) -> None:
        parsed = parse_source(
            "fn read(result: Result<str, IoError>) -> str {\n"
            "  return match result { Ok(value) => value, Err(_) => \"fallback\" };\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        self.assertEqual(interpreter.run("read", ResultValue.ok("ok")), "ok")
        with self.assertRaisesRegex(TypeError, "Ok payload expects str"):
            interpreter.run("read", ResultValue.ok(12))

    def test_question_operator_propagates_error_and_wraps_success(self) -> None:
        parsed = parse_source(
            "@io\n"
            "fn read() -> Result<str, IoError> { return io_read_line(); }\n"
            "@io\n"
            "fn wrapper() -> Result<str, IoError> {\n"
            "  let value = read();\n"
            "  return value?;\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        failing = Interpreter.from_source(
            parsed.ast,
            effect_handlers={"io_read_line": lambda: ResultValue.err("EOF")},
        ).run("wrapper")
        succeeding = Interpreter.from_source(
            parsed.ast,
            effect_handlers={"io_read_line": lambda: ResultValue.ok("hello")},
        ).run("wrapper")

        self.assertEqual(failing, ResultValue.err("EOF"))
        self.assertEqual(succeeding, ResultValue.ok("hello"))

    def test_question_operator_runs_inline_result_constructor_paths(self) -> None:
        source = (
            "fn source() -> Result<i32, IoError> { return Ok(40); }\n"
            "fn fail() -> Result<i32, IoError> { return Err(IoError); }\n"
            "fn forward() -> Result<i32, IoError> {\n"
            "  let value = source()?;\n"
            "  return Ok(value + 1);\n"
            "}\n"
            "fn propagate() -> Result<i32, IoError> {\n"
            "  let value = fail()?;\n"
            "  return Ok(value + 1);\n"
            "}\n"
        )
        self.assertEqual(run_source(source, "forward"), ResultValue.ok(41))
        self.assertEqual(run_source(source, "propagate"), ResultValue.err(1))

    def test_executes_explicit_unit_conversion(self) -> None:
        parsed = parse_source(
            "fn to_meters(value: Tensor<f32, [2], mm>) "
            "-> Tensor<f32, [2], m> { return convert(value, \"m\"); }\n"
        )
        result = Interpreter.from_source(parsed.ast).run("to_meters", [1000.0, 2500.0])

        self.assertIsInstance(result, Quantity)
        self.assertEqual(result.type.unit_symbol, "m")
        self.assertEqual(result.value, [1.0, 2.5])

    def test_executes_si_prefix_and_compound_unit_conversions(self) -> None:
        parsed = parse_source(
            "fn to_kilohertz(value: Tensor<f64, [2], MHz>) "
            "-> Tensor<f64, [2], kHz> { return convert(value, \"kHz\"); }\n"
            "fn to_slope(value: Tensor<f64, [2], V/mm>) "
            "-> Tensor<f64, [2], kV/m> { return convert(value, \"kV/m\"); }\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        frequencies = interpreter.run("to_kilohertz", [1.0, 2.5])
        slopes = interpreter.run("to_slope", [3.0, -4.0])

        self.assertIsInstance(frequencies, Quantity)
        self.assertEqual(frequencies.value, [1000.0, 2500.0])
        self.assertIsInstance(slopes, Quantity)
        self.assertEqual(slopes.type.unit_symbol, "kV/m")
        self.assertEqual(slopes.value, [3.0, -4.0])

    def test_pipeline_conversion_and_tensor_unit_cancellation(self) -> None:
        parsed = parse_source(
            "fn to_meters(value: Tensor<f32, [2], mm>) "
            "-> Tensor<f32, [2], m> { return value |> convert(\"m\"); }\n"
            "fn normalize(value: Tensor<f32, [2], V>) "
            "-> Tensor<f32, [2]> {\n"
            "  return value |> scale(1.0 / max_abs(value));\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        converted = interpreter.run("to_meters", [1000.0, 2500.0])
        normalized = interpreter.run("normalize", [1.0, 3.0])
        self.assertIsInstance(converted, Quantity)
        self.assertEqual(converted.type.unit_symbol, "m")
        self.assertEqual(converted.value, [1.0, 2.5])
        self.assertAlmostEqual(normalized[0], 1.0 / 3.0, delta=1e-7)
        self.assertEqual(normalized[1], 1.0)

    def test_executes_affine_and_symbolic_pi_unit_conversions(self) -> None:
        parsed = parse_source(
            "fn fahrenheit(value: Tensor<f32, [1], degC>) "
            "-> Tensor<f32, [1], degF> { return convert(value, \"degF\"); }\n"
            "fn radians(value: Tensor<f32, [1], deg>) "
            "-> Tensor<f32, [1], rad> { return convert(value, \"rad\"); }\n"
        )
        interpreter = Interpreter.from_source(parsed.ast)

        fahrenheit = interpreter.run("fahrenheit", [0.0])
        radians = interpreter.run("radians", [180.0])
        self.assertAlmostEqual(fahrenheit.value[0], 32.0)
        self.assertAlmostEqual(radians.value[0], 3.141592653589793, delta=1e-6)

    def test_checks_runtime_tensor_shapes_and_division_by_zero(self) -> None:
        parsed = parse_source(
            "fn first(values: Tensor<f32, [2]>) -> f32 { return values[0]; }\n"
            "fn negative(values: Tensor<f32, [2]>) -> f32 { return values[-1]; }\n"
            "fn out_of_bounds(values: Tensor<f32, [2]>) -> f32 { return values[2]; }\n"
            "fn tensor_divide(values: Tensor<f32, [2]>, divisor: f32) "
            "-> Tensor<f32, [2]> { return values / divisor; }\n"
            "fn divide(left: f32, right: f32) -> f32 { return left / right; }\n"
            "fn missing() -> f32 {}\n"
            "fn recurse() -> i32 { return recurse(); }\n"
        )
        interpreter = Interpreter.from_source(parsed.ast)

        with self.assertRaisesRegex(TypeError, "expects tensor shape"):
            interpreter.run("first", [1.0])
        with self.assertRaisesRegex(AetherRuntimeError, "Division by zero"):
            interpreter.run("divide", 1.0, 0.0)
        with self.assertRaisesRegex(AetherRuntimeError, "Negative index"):
            interpreter.run("negative", [1.0, 2.0])
        with self.assertRaisesRegex(AetherRuntimeError, "Invalid index"):
            interpreter.run("out_of_bounds", [1.0, 2.0])
        with self.assertRaisesRegex(AetherRuntimeError, "Division by zero"):
            interpreter.run("tensor_divide", [1.0, 2.0], 0.0)
        with self.assertRaisesRegex(AetherRuntimeError, "completed without returning"):
            interpreter.run("missing")
        with self.assertRaisesRegex(AetherRuntimeError, "Maximum call depth"):
            Interpreter.from_source(parsed.ast, max_call_depth=4).run("recurse")

    def test_tensor_inputs_validate_rectangular_shape_and_element_types(self) -> None:
        parsed = parse_source(
            "fn accept(values: Tensor<f32, [2, 2]>) -> f32 { return mean(values); }\n"
        )
        interpreter = Interpreter.from_source(parsed.ast)

        with self.assertRaisesRegex(TypeError, "rectangular"):
            interpreter.run("accept", [[1.0], [2.0, 3.0]])
        with self.assertRaisesRegex(TypeError, r"values\[3\] expects f32"):
            interpreter.run("accept", [[1.0, 2.0], [3.0, "bad"]])

    def test_struct_arguments_validate_fields_and_field_types(self) -> None:
        parsed = parse_source(
            "type Point { x: f32, y: f32, }\n"
            "fn x(point: Point) -> f32 { return point.x; }\n"
        )
        interpreter = Interpreter.from_source(parsed.ast)

        self.assertEqual(interpreter.run("x", {"x": 2.0, "y": 3.0}), 2.0)
        with self.assertRaisesRegex(TypeError, "invalid Point fields"):
            interpreter.run("x", {"x": 2.0})
        with self.assertRaisesRegex(TypeError, "point.x expects f32"):
            interpreter.run("x", {"x": "bad", "y": 3.0})

    def test_public_api_compiles_and_runs_clean_source_with_injected_effects(self) -> None:
        pure_source = "fn double(value: i32) -> i32 { return value * 2; }\n"
        self.assertEqual(run_source(pure_source, "double", 21), 42)

        effect_source = (
            "@io\n"
            "fn read() -> Result<str, IoError> { return io_read_line(); }\n"
            "@io\n"
            "fn wrapper() -> Result<str, IoError> {\n"
            "  let value = read();\n"
            "  return value?;\n"
            "}\n"
        )
        self.assertEqual(
            run_source(
                effect_source,
                "wrapper",
                effect_handlers={
                    "io_read_line": lambda: ResultValue.ok("ready")
                },
            ),
            ResultValue.ok("ready"),
        )
        self.assertEqual(compile_source(pure_source).function("double").name, "double")

    def test_extern_calls_only_use_explicit_handlers(self) -> None:
        source = (
            '@ffi\n@link("sensor")\n'
            'extern "C" fn read_channel(channel: u32) -> i32;\n'
            "@ffi\n"
            "fn sample(channel: u32) -> i32 { return read_channel(channel); }\n"
            '@ffi\n@link("sensor")\n'
            'extern "C" fn reset() -> void;\n'
        )
        module = compile_source(source)
        interpreter = Interpreter(module)

        with self.assertRaisesRegex(AetherRuntimeError, "does not load native libraries"):
            interpreter.run("sample", 4)
        with self.assertRaisesRegex(TypeError, "expects u32"):
            Interpreter(
                module,
                external_handlers={"read_channel": lambda channel: channel},
            ).run("sample", -1)
        with self.assertRaisesRegex(TypeError, "expects u32"):
            Interpreter(
                module,
                external_handlers={"read_channel": lambda channel: channel},
            ).run("sample", 2**32)
        with self.assertRaisesRegex(AetherRuntimeError, "violates i32"):
            Interpreter(
                module,
                external_handlers={"read_channel": lambda channel: "not an integer"},
            ).run("sample", 4)
        with self.assertRaisesRegex(AetherRuntimeError, "violates void"):
            Interpreter(
                module,
                external_handlers={
                    "read_channel": lambda channel: channel,
                    "reset": lambda: "unexpected",
                },
            ).run("reset")

        calls = []
        result = run_source(
            source,
            "sample",
            7,
            external_handlers={
                "read_channel": lambda channel: channel * 3,
                "reset": lambda: calls.append("reset"),
            },
        )
        self.assertEqual(result, 21)
        self.assertIsNone(
            run_source(
                source,
                "reset",
                external_handlers={
                    "read_channel": lambda channel: channel,
                    "reset": lambda: calls.append("reset"),
                },
            )
        )
        self.assertEqual(calls, ["reset"])

    def test_public_api_rejects_parse_and_semantic_errors(self) -> None:
        with self.assertRaises(CompilationError) as syntax_error:
            compile_source('fn broken() { return "unterminated; }\n')
        self.assertEqual(syntax_error.exception.diagnostics[0].code, "E_LEXICAL")

        with self.assertRaises(CompilationError) as semantic_error:
            compile_source("fn broken() { let x = true + false; }\n")
        self.assertIn(
            "E_TYPE_OPERATOR",
            {item.code for item in semantic_error.exception.diagnostics},
        )

        with self.assertRaises(CompilationError) as lowering_error:
            compile_source(
                "fn caller() { return callee(); }\n"
                "fn callee() { return 1; }\n"
            )
        self.assertIn(
            "E_IR_UNTYPED",
            {item.code for item in lowering_error.exception.diagnostics},
        )


if __name__ == "__main__":
    unittest.main()
