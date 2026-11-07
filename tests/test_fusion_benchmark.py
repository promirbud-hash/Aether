import os
from pathlib import Path
import shutil
import unittest

from src.fusion_benchmark import measure_fusion, measure_fusion_suite


class FusionBenchmarkTests(unittest.TestCase):
    def compiler(self) -> str | None:
        configured = os.environ.get("AETHER_C99_COMPILER")
        if configured:
            path = Path(configured)
            return str(path.resolve()) if path.is_file() else shutil.which(configured)
        return shutil.which("gcc") or shutil.which("clang")

    def test_measure_compares_identical_native_results(self) -> None:
        compiler = self.compiler()
        if compiler is None:
            self.skipTest("No C99 compiler is available")
        result = measure_fusion(
            compiler=compiler,
            elements=32,
            loops_per_sample=5,
            samples=2,
        )
        self.assertEqual(result.elements, 32)
        self.assertEqual(result.samples, 2)
        self.assertGreaterEqual(result.baseline_median_ms, 0.0)
        self.assertGreaterEqual(result.fused_median_ms, 0.0)
        self.assertGreater(result.speedup, 0.0)

    def test_measure_suite_compares_representative_chains(self) -> None:
        compiler = self.compiler()
        if compiler is None:
            self.skipTest("No C99 compiler is available")
        results = measure_fusion_suite(
            compiler=compiler,
            elements=32,
            loops_per_sample=5,
            samples=2,
        )
        self.assertEqual(
            [result.workload for result in results],
            ["scale_shift_restore", "offset_cancel", "negated_scale"],
        )
        for result in results:
            with self.subTest(workload=result.workload):
                self.assertGreaterEqual(result.baseline_median_ms, 0.0)
                self.assertGreaterEqual(result.fused_median_ms, 0.0)
                self.assertGreater(result.speedup, 0.0)

    def test_measure_rejects_non_positive_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "elements"):
            measure_fusion(elements=0)
        with self.assertRaisesRegex(ValueError, "workload"):
            measure_fusion(workload="unknown")


if __name__ == "__main__":
    unittest.main()
