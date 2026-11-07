import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from src.api import emit_c99, run_source
from src.cuda import CudaLoweringError, emit_cuda


_SOURCE = (
    "fn chain(values: Tensor<f32, [4]>, scale: f32, offset: f32) "
    "-> Tensor<f32, [4]> {\n"
    "  let scaled = values * scale;\n"
    "  let shifted = scaled + offset;\n"
    "  return shifted - values;\n"
    "}\n"
)

def _boundary_source(dimensions: tuple[int, ...]) -> str:
    shape = ", ".join(str(dimension) for dimension in dimensions)
    return (
        f"fn boundary(left: Tensor<f32, [{shape}]>, "
        f"right: Tensor<f32, [{shape}]>, scale: f32, offset: f32) "
        f"-> Tensor<f32, [{shape}]> {{\n"
        "  let added = left + right;\n"
        "  let delta = right - left;\n"
        "  let scaled = scale * delta;\n"
        "  let shifted = offset + scaled;\n"
        "  let negated = -shifted;\n"
        "  return added + negated;\n"
        "}\n"
    )


def _f32_ulp_distance(left: float, right: float) -> int:
    def ordered(value: float) -> int:
        bits = struct.unpack("=I", struct.pack("=f", value))[0]
        magnitude = bits & 0x7FFFFFFF
        return 0x80000000 - magnitude if bits & 0x80000000 else 0x80000000 + bits

    return abs(ordered(left) - ordered(right))


def _nest(values: list[float], dimensions: tuple[int, ...]) -> list:
    if len(dimensions) == 1:
        return values
    stride = 1
    for dimension in dimensions[1:]:
        stride *= dimension
    return [
        _nest(values[index * stride:(index + 1) * stride], dimensions[1:])
        for index in range(dimensions[0])
    ]


def _flatten(values: list) -> list[float]:
    if not values or not isinstance(values[0], list):
        return values
    return [item for child in values for item in _flatten(child)]


def _compile_cuda(nvcc: str, source: str, root: Path, name: str) -> Path:
    if os.name == "nt" and shutil.which("cl.exe") is None:
        raise unittest.SkipTest("Initialize the MSVC x64 compiler environment first")
    source_path = root / f"{name}.cu"
    executable = root / (f"{name}.exe" if os.name == "nt" else name)
    source_path.write_text(source, encoding="utf-8", newline="\n")
    result = subprocess.run(
        [
            nvcc,
            "--std=c++14",
            "--fmad=false",
            "--ftz=false",
            str(source_path),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode:
        raise AssertionError(
            f"CUDA compilation failed: {result.stderr or result.stdout}"
        )
    return executable


def _run_cuda(executable: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _compile_c99(compiler: str, source: str, root: Path, name: str) -> Path:
    source_path = root / f"{name}.c"
    executable = root / (f"{name}.exe" if os.name == "nt" else name)
    source_path.write_text(source, encoding="utf-8", newline="\n")
    result = subprocess.run(
        [
            compiler,
            "-std=c99",
            "-pedantic",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffp-contract=off",
            str(source_path),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode:
        raise AssertionError(
            f"C99 compilation failed: {result.stderr or result.stdout}"
        )
    return executable


class CudaBackendTests(unittest.TestCase):
    def test_emission_is_deterministic_and_has_explicit_host_statuses(self) -> None:
        generated = emit_cuda(_SOURCE, "chain")

        self.assertEqual(generated, emit_cuda(_SOURCE, "chain"))
        self.assertIn("__global__ void aether_kernel_636861696e", generated)
        self.assertIn('extern "C" int aether_cuda_636861696e', generated)
        self.assertIn("AETHER_CUDA_NO_DEVICE 1", generated)
        self.assertIn("AETHER_CUDA_INVALID_ARGUMENT 2", generated)
        self.assertIn("AETHER_CUDA_RUNTIME_ERROR 3", generated)
        self.assertIn("cudaGetDeviceCount", generated)
        self.assertIn("cudaMemcpyHostToDevice", generated)
        self.assertIn("cudaMemcpyDeviceToHost", generated)
        self.assertIn("index += (size_t)blockDim.x * gridDim.x", generated)
        self.assertEqual(generated.count("__global__ void"), 1)
        self.assertLess(
            generated.index("aether_v_76616c756573 == NULL"),
            generated.index("cudaGetDeviceCount"),
        )

    def test_emits_single_operation_without_a_temporary_chain(self) -> None:
        generated = emit_cuda(
            "fn negate(values: Tensor<f32, [2]>) -> Tensor<f32, [2]> "
            "{ return -values; }\n",
            "negate",
        )

        self.assertIn(
            "aether_output[index] = "
            "static_cast<float>((-aether_v_76616c756573[index]));",
            generated,
        )

    def test_accepts_single_let_tensor_operation(self) -> None:
        generated = emit_cuda(
            "fn scaled(values: Tensor<f32, [2]>, factor: f32) "
            "-> Tensor<f32, [2]> {\n"
            "  let result = values * factor;\n"
            "  return result;\n"
            "}\n",
            "scaled",
        )

        self.assertIn("__global__ void aether_kernel_7363616c6564", generated)

    def test_rejects_features_outside_cuda_v0(self) -> None:
        for source, function in (
            (
                "fn unsupported(values: Tensor<f64, [2]>) "
                "-> Tensor<f64, [2]> { return values + values; }\n",
                "unsupported",
            ),
            (
                "fn unsupported(values: Tensor<f32, [2]>) -> f32 "
                "{ return sum(values); }\n",
                "unsupported",
            ),
        ):
            with self.subTest(function=function, source=source):
                with self.assertRaises(CudaLoweringError):
                    emit_cuda(source, function)

    def test_cuda_kernel_matches_interpreter_within_one_f32_ulp(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("CUDA compiler is not available")
        if os.name == "nt" and shutil.which("cl.exe") is None:
            self.skipTest("Initialize the MSVC x64 compiler environment first")

        with tempfile.TemporaryDirectory(prefix="aether-cuda-test-") as directory:
            root = Path(directory)
            executable = _compile_cuda(
                nvcc,
                emit_cuda(_SOURCE, "chain") + r"""
#include <stdio.h>
int main(void) {
    aether_tensor_f32_4 input = {{16777216.0f, -2.0f, 3.5f, 0.125f}};
    aether_tensor_f32_4 output = {{0}};
    if (aether_cuda_636861696e(NULL, 2.0f, 1.0f, &output)
        != AETHER_CUDA_INVALID_ARGUMENT) return 4;
    if (aether_cuda_636861696e(&input, 2.0f, 1.0f, NULL)
        != AETHER_CUDA_INVALID_ARGUMENT) return 5;
    int status = aether_cuda_636861696e(&input, 2.0f, 1.0f, &output);
    if (status != AETHER_CUDA_SUCCESS) return status;
    for (size_t index = 0; index < 4; ++index)
        printf("%.9g ", (double)output.data[index]);
    return 0;
}
""",
                root,
                "single",
            )
            result = _run_cuda(executable)
            if result.returncode == 1:
                self.skipTest("No CUDA device is available to run the kernel")
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = [float(value) for value in result.stdout.split()]
            expected = run_source(
                _SOURCE,
                "chain",
                [16777216.0, -2.0, 3.5, 0.125],
                2.0,
                1.0,
            )
            self.assertEqual(len(actual), len(expected))
            for native, reference in zip(actual, expected):
                self.assertLessEqual(_f32_ulp_distance(native, reference), 1)

    def test_cuda_elementwise_matrix_runs_across_grid_stride_boundary(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("CUDA compiler is not available")
        if os.name == "nt" and shutil.which("cl.exe") is None:
            self.skipTest("Initialize the MSVC x64 compiler environment first")
        cpu_compiler = (
            shutil.which("gcc")
            or shutil.which("clang")
            or shutil.which("cc")
        )
        if cpu_compiler is None:
            self.skipTest("A host C99 compiler is required for CPU/GPU comparison")

        for dimensions in ((1,), (255,), (256,), (257,), (513,), (16, 16)):
            count = 1
            for dimension in dimensions:
                count *= dimension
            shape = "x".join(str(dimension) for dimension in dimensions)
            source = _boundary_source(dimensions)
            harness = fr"""
#include <stdio.h>
int main(void) {{
    aether_tensor_f32_{shape} left = {{0}};
    aether_tensor_f32_{shape} right = {{0}};
    aether_tensor_f32_{shape} output = {{0}};
    for (size_t index = 0; index < {count}; ++index) {{
        left.data[index] = (float)((int)(index % 17) - 8) / 8.0f;
        right.data[index] = (float)((int)(index % 11) - 5) / 16.0f;
    }}
    left.data[0] = 16777216.0f;
    right.data[0] = 1.0f;
    if ({count} > 1) {{
        left.data[1] = 1.40129846e-45f;
        right.data[1] = 0.0f;
    }}
    int status = aether_cuda_626f756e64617279(&left, &right, 1.0f, 0.125f, &output);
    if (status != AETHER_CUDA_SUCCESS) return status;
    for (size_t index = 0; index < {count}; ++index)
        printf("%.9g\n", (double)output.data[index]);
    return 0;
}}
"""
            cpu_harness = harness.replace(
                "int status = aether_cuda_626f756e64617279("
                "&left, &right, 1.0f, 0.125f, &output);",
                "output = aether_fn_626f756e64617279("
                "left, right, 1.0f, 0.125f);",
            ).replace("    if (status != AETHER_CUDA_SUCCESS) return status;\n", "")
            with self.subTest(dimensions=dimensions), tempfile.TemporaryDirectory(
                prefix="aether-cuda-grid-test-"
            ) as directory:
                root = Path(directory)
                executable = _compile_cuda(
                    nvcc,
                    emit_cuda(source, "boundary") + harness,
                    root,
                    "boundary",
                )
                result = _run_cuda(executable)
                if result.returncode == 1:
                    self.skipTest("No CUDA device is available to run the kernel")
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = [float(value) for value in result.stdout.splitlines()]
                left = [(index % 17 - 8) / 8.0 for index in range(count)]
                right = [(index % 11 - 5) / 16.0 for index in range(count)]
                left[0], right[0] = 16777216.0, 1.0
                if count > 1:
                    left[1], right[1] = 1.40129846e-45, 0.0
                expected = run_source(
                    source,
                    "boundary",
                    _nest(left, dimensions),
                    _nest(right, dimensions),
                    1.0,
                    0.125,
                )
                expected_flat = _flatten(expected)
                self.assertEqual(len(actual), count)
                self.assertEqual(len(expected_flat), count)
                for index, (native, reference) in enumerate(zip(actual, expected_flat)):
                    with self.subTest(index=index, dimensions=dimensions):
                        self.assertLessEqual(
                            _f32_ulp_distance(native, reference),
                            1,
                        )

                cpu_executable = _compile_c99(
                    cpu_compiler,
                    emit_c99(source) + cpu_harness,
                    root,
                    "cpu",
                )
                cpu_result = subprocess.run(
                    [str(cpu_executable)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                self.assertEqual(cpu_result.returncode, 0, cpu_result.stderr)
                cpu_values = [
                    float(value) for value in cpu_result.stdout.splitlines()
                ]
                self.assertEqual(len(cpu_values), count)
                for index, (native, cpu) in enumerate(zip(actual, cpu_values)):
                    with self.subTest(index=index, dimensions=dimensions, target="cpu"):
                        self.assertLessEqual(_f32_ulp_distance(native, cpu), 1)

    def test_cuda_host_reports_device_and_runtime_failures(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("CUDA compiler is not available")
        if os.name == "nt" and shutil.which("cl.exe") is None:
            self.skipTest("Initialize the MSVC x64 compiler environment first")

        cases = (
            (
                "no_device",
                r"""
#include <stdlib.h>
static cudaError_t test_device_count(int *count) {
    *count = 0;
    return cudaErrorNoDevice;
}
#define cudaGetDeviceCount test_device_count
int main(void) {
    aether_tensor_f32_4 input = {{0}};
    aether_tensor_f32_4 output = {{0}};
    if (aether_cuda_636861696e(NULL, 1.0f, 0.0f, &output)
        != AETHER_CUDA_INVALID_ARGUMENT) return 1;
    if (aether_cuda_636861696e(&input, 1.0f, 0.0f, &output)
        != AETHER_CUDA_NO_DEVICE) return 2;
    return 0;
}
""",
            ),
            (
                "device_query_failure",
                r"""
static cudaError_t test_device_count(int *count) {
    *count = 0;
    return cudaErrorInsufficientDriver;
}
#define cudaGetDeviceCount test_device_count
int main(void) {
    aether_tensor_f32_4 input = {{0}};
    aether_tensor_f32_4 output = {{0}};
    return aether_cuda_636861696e(&input, 1.0f, 0.0f, &output)
        == AETHER_CUDA_RUNTIME_ERROR ? 0 : 1;
}
""",
            ),
            (
                "allocation_failure",
                r"""
#include <stdlib.h>
static cudaError_t test_device_count(int *count) {
    *count = 1;
    return cudaSuccess;
}
static cudaError_t test_malloc(void **pointer, size_t size) {
    (void)size;
    *pointer = NULL;
    return cudaErrorMemoryAllocation;
}
#define cudaGetDeviceCount test_device_count
#define cudaMalloc test_malloc
int main(void) {
    aether_tensor_f32_4 input = {{0}};
    aether_tensor_f32_4 output = {{0}};
    return aether_cuda_636861696e(&input, 1.0f, 0.0f, &output)
        == AETHER_CUDA_RUNTIME_ERROR ? 0 : 1;
}
""",
            ),
            (
                "copy_failure",
                r"""
#include <stdlib.h>
static int test_free_count;
static cudaError_t test_device_count(int *count) {
    *count = 1;
    return cudaSuccess;
}
static cudaError_t test_malloc(void **pointer, size_t size) {
    *pointer = malloc(size);
    return *pointer == NULL ? cudaErrorMemoryAllocation : cudaSuccess;
}
static cudaError_t test_memcpy(void *destination, const void *source,
                               size_t size, enum cudaMemcpyKind kind) {
    (void)destination; (void)source; (void)size; (void)kind;
    return cudaErrorInvalidValue;
}
static cudaError_t test_free(void *pointer) {
    ++test_free_count;
    free(pointer);
    return cudaSuccess;
}
#define cudaGetDeviceCount test_device_count
#define cudaMalloc test_malloc
#define cudaMemcpy test_memcpy
#define cudaFree test_free
int main(void) {
    aether_tensor_f32_4 input = {{0}};
    aether_tensor_f32_4 output = {{0}};
    int status = aether_cuda_636861696e(&input, 1.0f, 0.0f, &output);
    return status == AETHER_CUDA_RUNTIME_ERROR && test_free_count == 1 ? 0 : 1;
}
""",
            ),
        )
        for name, harness in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                prefix=f"aether-cuda-{name}-"
            ) as directory:
                generated = emit_cuda(_SOURCE, "chain")
                shim, separator, main_body = harness.partition("int main(void) {")
                self.assertTrue(separator)
                wrapper_declaration = 'extern "C" int aether_cuda_636861696e'
                injected = generated.replace(
                    wrapper_declaration,
                    shim + wrapper_declaration,
                    1,
                ) + "int main(void) {" + main_body
                executable = _compile_cuda(
                    nvcc,
                    injected,
                    Path(directory),
                    name,
                )
                result = _run_cuda(executable)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
