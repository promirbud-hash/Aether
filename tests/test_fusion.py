import unittest

from src.api import compile_source
from src.fusion import fuse_elementwise_tensors


class TensorFusionTests(unittest.TestCase):
    def test_fuses_single_elementwise_temporary_for_cuda_subset(self) -> None:
        module = compile_source(
            "fn single(values: Tensor<f32, [4]>, scale: f32) "
            "-> Tensor<f32, [4]> {\n"
            "  let scaled = values * scale;\n"
            "  return scaled;\n"
            "}\n"
        )

        optimized = fuse_elementwise_tensors(module)
        function = optimized.function("single")

        self.assertEqual(len(function.body), 1)
        self.assertEqual(function.body[0].expression.kind, "fused_tensor")

    def test_fuses_single_use_elementwise_temporary_chain_deterministically(self) -> None:
        module = compile_source(
            "fn chain(values: Tensor<f32, [4]>) -> Tensor<f32, [4]> {\n"
            "  let doubled = values * 2.0;\n"
            "  let shifted = doubled + 1.0;\n"
            "  return shifted - values;\n"
            "}\n"
        )

        optimized = fuse_elementwise_tensors(module)
        function = optimized.function("chain")

        self.assertEqual(len(function.body), 1)
        self.assertEqual(function.body[0].kind, "return")
        self.assertEqual(function.body[0].expression.kind, "fused_tensor")
        self.assertEqual(fuse_elementwise_tensors(optimized), optimized)

    def test_does_not_fuse_multi_use_or_gradient_target_functions(self) -> None:
        module = compile_source(
            "fn shared(values: Tensor<f64, [2]>) -> Tensor<f64, [2]> {\n"
            "  let scaled = values + mean(values);\n"
            "  return scaled + scaled;\n"
            "}\n"
            "fn objective(values: Tensor<f64, [2]>) -> f64 {\n"
            "  let shifted = values + mean(values);\n"
            "  return sum(shifted);\n"
            "}\n"
            "fn derivative(values: Tensor<f64, [2]>) -> Tensor<f64, [2]> {\n"
            "  return gradient(objective, values);\n"
            "}\n"
        )

        optimized = fuse_elementwise_tensors(module)

        self.assertEqual(
            optimized.function("shared"),
            module.function("shared"),
        )
        self.assertEqual(
            optimized.function("objective"),
            module.function("objective"),
        )

    def test_does_not_fuse_reduction_expressions(self) -> None:
        module = compile_source(
            "fn reduced(values: Tensor<f32, [2]>) -> f32 {\n"
            "  let shifted = values + 1.0;\n"
            "  return sum(shifted);\n"
            "}\n"
        )

        optimized = fuse_elementwise_tensors(module)

        self.assertEqual(optimized, module)


if __name__ == "__main__":
    unittest.main()
