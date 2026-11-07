import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
import uuid

from src.api import (
    CompilationError,
    build_executable,
    build_object,
    emit_c99,
    run_source,
)
from src.build import (
    C99BuildError,
    C99LinkConfigurationError,
    C99ToolchainError,
    compile_c99_object,
)
from src.interpreter import Quantity


_STRICT_C99_FLAGS = ("-std=c99", "-pedantic", "-Wall", "-Wextra", "-Werror")


class C99BackendTests(unittest.TestCase):
    def test_frame_resources_cleanup_lifo_on_return_and_question_error(self) -> None:
        source = (
            "fn inner(size: u64) -> u64 {\n"
            "  let temporary = frame_buffer(size);\n"
            "  return frame_buffer_len(temporary);\n"
            "}\n"
            "fn fail() -> Result<u64, IoError> { return Err(IoError); }\n"
            "fn early(size: u64) -> Result<u64, IoError> {\n"
            "  let failure = fail();\n"
            "  let value = failure?;\n"
            "  let later = frame_buffer(size);\n"
            "  return Ok(frame_buffer_len(later) + value);\n"
            "}\n"
            "fn outer(size: u64, next: u64, nested: u64) -> Result<u64, IoError> {\n"
            "  let first = frame_buffer(size);\n"
            "  let second = frame_buffer(next);\n"
            "  let nested_size = inner(nested);\n"
            "  let failure = fail();\n"
            "  let value = failure?;\n"
            "  return Ok(frame_buffer_len(first) + frame_buffer_len(second) + nested_size + value);\n"
            "}\n"
            "fn success(size: u64) -> u64 {\n"
            "  let bytes = frame_buffer(size);\n"
            "  return frame_buffer_len(bytes);\n"
            "}\n"
            "fn ok(value: u64) -> Result<u64, IoError> { return Ok(value); }\n"
            "fn success_question(size: u64) -> Result<u64, IoError> {\n"
            "  let bytes = frame_buffer(size);\n"
            "  let result = ok(size);\n"
            "  let value = result?;\n"
            "  return Ok(frame_buffer_len(bytes) + value);\n"
            "}\n"
        )
        generated = emit_c99(source)
        harness = r"""
static void *allocated[8];
static void *released[8];
static size_t allocation_count;
static size_t release_count;

int main(void) {
    aether_result_u64_IoError early =
        aether_fn_6561726c79(UINT64_MAX);
    if (early.is_ok || early.error != 1) return 10;
    if (allocation_count != 0 || release_count != 0) return 11;
    aether_result_u64_IoError result =
        aether_fn_6f75746572(UINT64_C(5), UINT64_C(6), UINT64_C(7));
    if (result.is_ok || result.error != 1) return 1;
    if (allocation_count != 3 || release_count != 3) return 2;
    if (released[0] != allocated[2]
        || released[1] != allocated[1]
        || released[2] != allocated[0]) return 3;
    if (aether_fn_73756363657373(UINT64_C(4)) != UINT64_C(4)) return 4;
    if (allocation_count != 4 || release_count != 4) return 5;
    if (released[3] != allocated[3]) return 6;
    aether_result_u64_IoError success =
        aether_fn_737563636573735f7175657374696f6e(UINT64_C(8));
    if (!success.is_ok || success.value != UINT64_C(16)) return 7;
    if (allocation_count != 5 || release_count != 5) return 8;
    if (released[4] != allocated[4]) return 9;
    return 0;
}
"""
        prefix = r"""
#include <stddef.h>
#include <stdlib.h>
static void *aether_test_alloc(size_t size);
static void aether_test_free(void *pointer);
#define AETHER_FRAME_ALLOC aether_test_alloc
#define AETHER_FRAME_FREE aether_test_free
"""
        harness += r"""
static void *aether_test_alloc(size_t size) {
    void *pointer = malloc(size);
    if (pointer != NULL) allocated[allocation_count++] = pointer;
    return pointer;
}

static void aether_test_free(void *pointer) {
    released[release_count++] = pointer;
    free(pointer);
}
"""
        executable = self.compile_program(generated, harness, prefix=prefix)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_frame_buffer_retain_release_and_owned_return_paths(self) -> None:
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
        generated = emit_c99(source)
        harness = r"""
static void *allocated[8];
static void *released[8];
static size_t allocation_count;
static size_t release_count;

int main(void) {
    aether_frame_buffer input = aether_frame_buffer_create(UINT64_C(4));
    aether_frame_buffer output = aether_fn_6964656e74697479(input);
    if (input.storage != output.storage || input.storage->refcount != 2) return 1;
    aether_frame_buffer_release(&output);
    aether_frame_buffer_release(&input);
    if (release_count != 1) return 2;

    aether_result_FrameBuffer_IoError result =
        aether_fn_72657475726e5f6f6b(UINT64_C(3));
    if (!result.is_ok || result.value.storage->refcount != 1) return 3;
    aether_frame_buffer_release(&result.value);
    if (release_count != 2) return 4;

    aether_result_FrameBuffer_IoError inline_result =
        aether_fn_696e6c696e655f6f6b(UINT64_C(4));
    if (!inline_result.is_ok || inline_result.value.storage->refcount != 1) return 5;
    aether_frame_buffer_release(&inline_result.value);
    if (release_count != 3) return 6;

    aether_frame_buffer borrowed = aether_frame_buffer_create(UINT64_C(2));
    aether_result_FrameBuffer_IoError input_result = {1, borrowed, 0};
    aether_result_FrameBuffer_IoError unwrapped =
        aether_fn_756e77726170(input_result);
    if (!unwrapped.is_ok || unwrapped.value.storage != borrowed.storage) return 7;
    if (borrowed.storage->refcount != 2) return 8;
    aether_frame_buffer_release(&unwrapped.value);
    aether_frame_buffer_release(&borrowed);
    if (release_count != 4) return 9;

    if (aether_fn_64697363617264(UINT64_C(8)) != 0) return 10;
    if (release_count != 5 || released[4] != allocated[4]) return 11;
    if (aether_fn_616c696173(UINT64_C(5)) != UINT64_C(10)) return 12;
    if (release_count != 6 || released[5] != allocated[5]) return 13;
    if (aether_fn_726562696e64(UINT64_C(6)) != UINT64_C(6)) return 14;
    if (release_count != 7 || released[6] != allocated[6]) return 15;
    return 0;
}
"""
        prefix = r"""
#include <stddef.h>
#include <stdlib.h>
static void *aether_test_alloc(size_t size);
static void aether_test_free(void *pointer);
#define AETHER_FRAME_ALLOC aether_test_alloc
#define AETHER_FRAME_FREE aether_test_free
"""
        harness += r"""
static void *aether_test_alloc(size_t size) {
    void *pointer = malloc(size);
    if (pointer != NULL) allocated[allocation_count++] = pointer;
    return pointer;
}

static void aether_test_free(void *pointer) {
    released[release_count++] = pointer;
    free(pointer);
}
"""
        executable = self.compile_program(generated, harness, prefix=prefix)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_try_failure_releases_previously_evaluated_expression_temporaries(self) -> None:
        source = (
            "fn fail() -> Result<u64, IoError> { return Err(IoError); }\n"
            "fn succeed(value: u64) -> Result<u64, IoError> { return Ok(value); }\n"
            "fn failing_expression(size: u64) -> Result<u64, IoError> {\n"
            "  return Ok(frame_buffer_len(frame_buffer(size)) + fail()?);\n"
            "}\n"
            "fn successful_expression(size: u64) -> Result<u64, IoError> {\n"
            "  return Ok(frame_buffer_len(frame_buffer(size)) + succeed(size)?);\n"
            "}\n"
        )
        generated = emit_c99(source)
        harness = r"""
static size_t allocation_count;
static size_t release_count;

int main(void) {
    aether_result_u64_IoError failed =
        aether_fn_6661696c696e675f65787072657373696f6e(UINT64_C(8));
    if (failed.is_ok || allocation_count != 1 || release_count != 1) return 1;
    aether_result_u64_IoError succeeded =
        aether_fn_7375636365737366756c5f65787072657373696f6e(UINT64_C(8));
    if (!succeeded.is_ok || succeeded.value != UINT64_C(16)) return 2;
    if (allocation_count != 2 || release_count != 2) return 3;
    return 0;
}
"""
        prefix = r"""
#include <stddef.h>
#include <stdlib.h>
static void *aether_test_alloc(size_t size);
static void aether_test_free(void *pointer);
#define AETHER_FRAME_ALLOC aether_test_alloc
#define AETHER_FRAME_FREE aether_test_free
"""
        harness += r"""
static void *aether_test_alloc(size_t size) {
    void *pointer = malloc(size);
    if (pointer != NULL) ++allocation_count;
    return pointer;
}

static void aether_test_free(void *pointer) {
    ++release_count;
    free(pointer);
}
"""
        executable = self.compile_program(generated, harness, prefix=prefix)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_frame_buffer_allocation_failure_traps(self) -> None:
        source = (
            "fn allocate(size: u64) -> u64 {\n"
            "  let bytes = frame_buffer(size);\n"
            "  return frame_buffer_len(bytes);\n"
            "}\n"
        )
        prefix = r"""
#include <stddef.h>
static void *aether_test_alloc(size_t size);
#define AETHER_FRAME_ALLOC aether_test_alloc
static void *aether_test_alloc(size_t size) {
    (void)size;
    return NULL;
}
"""
        harness = r"""
int main(void) {
    return (int)aether_fn_616c6c6f63617465(UINT64_C(1));
}
"""
        executable = self.compile_program(emit_c99(source), harness, prefix=prefix)
        result = self.run_program(executable)
        self.assertNotEqual(result.returncode, 0)

    def test_frame_buffer_size_overflow_traps_before_allocation(self) -> None:
        source = (
            "fn allocate(size: u64) -> u64 {\n"
            "  let bytes = frame_buffer(size);\n"
            "  return frame_buffer_len(bytes);\n"
            "}\n"
        )
        prefix = r"""
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
static void *aether_test_alloc(size_t size);
#define AETHER_FRAME_ALLOC aether_test_alloc
#define AETHER_FRAME_SIZE_MAX UINT32_MAX
"""
        harness = r"""
static void *aether_test_alloc(size_t size) {
    (void)size;
    return malloc(1);
}

int main(void) {
    uint64_t too_large = UINT64_C(4294967296);
    return aether_fn_616c6c6f63617465(too_large) == too_large ? 0 : 1;
}
"""
        executable = self.compile_program(
            emit_c99(source),
            harness,
            prefix=prefix,
        )
        result = self.run_program(executable)
        self.assertNotEqual(result.returncode, 0)

    def test_frame_buffer_reference_count_overflow_traps(self) -> None:
        source = (
            "fn identity(bytes: FrameBuffer) -> FrameBuffer { return bytes; }\n"
        )
        harness = r"""
int main(void) {
    aether_frame_buffer bytes = aether_frame_buffer_create(UINT64_C(1));
    bytes.storage->refcount = SIZE_MAX;
    (void)aether_fn_6964656e74697479(bytes);
    return 0;
}
"""
        executable = self.compile_program(emit_c99(source), harness)
        result = self.run_program(executable)
        self.assertNotEqual(result.returncode, 0)

    def test_result_question_propagation_uses_frame_cleanup_epilogue(self) -> None:
        source = (
            "fn success() -> Result<i32, IoError> { return Ok(40); }\n"
            "fn failure() -> Result<i32, IoError> { return Err(IoError); }\n"
            "fn forward() -> Result<i32, IoError> {\n"
            "  let result = success();\n"
            "  let value = result?;\n"
            "  return Ok(value + 1);\n"
            "}\n"
            "fn propagate() -> Result<i32, IoError> {\n"
            "  let result = failure();\n"
            "  let value = result?;\n"
            "  return Ok(value + 1);\n"
            "}\n"
        )
        generated = emit_c99(source)
        self.assertIn("goto aether_frame_cleanup;", generated)
        self.assertIn("aether_frame_cleanup:", generated)
        harness = r"""
int main(void) {
    aether_result_i32_IoError success = aether_fn_666f7277617264();
    aether_result_i32_IoError failure = aether_fn_70726f706167617465();
    if (!success.is_ok || success.value != 41) return 1;
    if (failure.is_ok || failure.error != 1) return 2;
    return 0;
}
"""
        executable = self.compile_program(generated, harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_var_rebinding_copies_tensor_values_in_c99(self) -> None:
        source = (
            "fn rebind(values: Tensor<f32, [2]>) -> f32 {\n"
            "  var current = values;\n"
            "  let snapshot = current;\n"
            "  current = current * 2.0;\n"
            "  return sum(snapshot) + sum(current);\n"
            "}\n"
        )
        generated = emit_c99(source)
        harness = r"""
int main(void) {
    aether_tensor_f32_2 values = {{1.0f, 3.0f}};
    return aether_fn_726562696e64(values) == 12.0f ? 0 : 1;
}
"""
        executable = self.compile_program(generated, harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def compiler(self) -> tuple[str, ...]:
        configured_compiler = os.environ.get("AETHER_C99_COMPILER")
        if configured_compiler:
            path = shutil.which(configured_compiler)
            if path is not None:
                if Path(path).stem.lower() == "zig":
                    return (path, "cc")
                return (path,)
            message = (
                f"Configured C99 compiler {configured_compiler!r} is not installed"
            )
            if os.environ.get("AETHER_REQUIRE_C99_COMPILER") == "1":
                self.fail(message)
            self.skipTest(message)

        compiler = next(
            (
                (path,)
                for name in ("cc", "gcc", "clang")
                if (path := shutil.which(name)) is not None
            ),
            None,
        )
        if compiler is None:
            zig = shutil.which("zig")
            if zig is not None:
                compiler = (zig, "cc")
        if compiler is None:
            message = "No C99 compiler is installed"
            if os.environ.get("AETHER_REQUIRE_C99_COMPILER") == "1":
                self.fail(f"{message}; this test environment requires one")
            self.skipTest(message)
        return compiler

    def compile_program(
        self,
        generated: str,
        harness: str,
        *,
        prefix: str = "",
    ) -> Path:
        directory = Path(__file__).resolve().parent / f".c99-build-{uuid.uuid4().hex}"
        directory.mkdir()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        source_path = directory / "generated.c"
        executable_path = directory / "generated"
        source_path.write_text(prefix + generated + "\n" + harness, encoding="utf-8")
        result = subprocess.run(
            [
                *self.compiler(),
                *_STRICT_C99_FLAGS,
                str(source_path),
                "-o",
                str(executable_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"C99 compilation failed:\n{result.stdout}\n{result.stderr}",
        )
        return executable_path

    def run_program(
        self,
        executable_path: Path,
        *arguments: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(executable_path), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=executable_path.parent,
        )

    def compile_and_run(
        self,
        generated: str,
        harness: str,
        *arguments: str,
    ) -> subprocess.CompletedProcess[str]:
        executable = self.compile_program(generated, harness)
        return self.run_program(executable, *arguments)

    def test_generated_c99_compiles_when_a_c_compiler_is_available(self) -> None:
        compiler = self.compiler()
        generated = emit_c99(
            "fn add(left: i32, right: i32) -> i32 { return left + right; }\n"
            "fn divide(left: i32, right: i32) -> i32 { return left / right; }\n"
        )
        directory = Path(__file__).resolve().parent / f".c99-build-{uuid.uuid4().hex}"
        directory.mkdir()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        source_path = directory / "generated.c"
        object_path = directory / "generated.o"
        source_path.write_text(generated, encoding="utf-8")
        subprocess.run(
            [
                *compiler,
                *_STRICT_C99_FLAGS,
                "-c",
                str(source_path),
                "-o",
                str(object_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_build_object_writes_a_compiled_object_without_linking(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = (
                Path(temporary_directory)
                / "artifacts with spaces"
                / "module.o"
            )
            actual_path = build_object(
                "fn add(left: i32, right: i32) -> i32 { return left + right; }\n",
                output_path,
                compiler=compiler,
            )

            self.assertEqual(actual_path, output_path.resolve())
            self.assertTrue(actual_path.is_file())
            self.assertGreater(actual_path.stat().st_size, 0)

    def test_build_object_compiles_supported_c_abi_scalar_prototypes(self) -> None:
        compiler = self.compiler()
        source = "\n".join(
            f'@ffi @link("native") extern "C" fn ffi_{type_name}(value: {type_name}) -> {result};'
            for type_name, result in (
                ("i8", "i8"),
                ("i16", "i16"),
                ("i32", "i32"),
                ("i64", "i64"),
                ("u8", "u8"),
                ("u16", "u16"),
                ("u32", "u32"),
                ("u64", "u64"),
                ("f32", "f32"),
                ("f64", "f64"),
            )
        ) + '\n@ffi @link("native") extern "C" fn ffi_void(value: i32) -> void;\n'
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = build_object(
                source,
                Path(temporary_directory) / "ffi-prototypes.o",
                compiler=compiler,
            )
            self.assertTrue(output.is_file())

    def test_build_object_preserves_artifact_on_compiler_failure(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "existing.o"
            output_path.write_bytes(b"previous object")
            with self.assertRaises(C99BuildError):
                compile_c99_object(
                    "this is not valid C;\n",
                    output_path,
                    compiler=compiler,
                )
            self.assertEqual(output_path.read_bytes(), b"previous object")

    def test_build_object_reports_missing_compiler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(C99ToolchainError):
                compile_c99_object(
                    "int value(void) { return 1; }\n",
                    Path(temporary_directory) / "module.o",
                    compiler=("aether-missing-c-compiler",),
                )

    def test_build_executable_runs_i32_entry_point(self) -> None:
        compiler = self.compiler()
        source = "fn main() -> i32 { return 23; }\n"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "aether-program.exe"
            executable = build_executable(
                source,
                output_path,
                compiler=compiler,
            )
            result = subprocess.run(
                [str(executable)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

        self.assertEqual(result.returncode, run_source(source, "main"))

    def test_build_executable_runs_void_entry_point_as_success(self) -> None:
        compiler = self.compiler()
        source = "fn main() -> void { return; }\n"
        with tempfile.TemporaryDirectory() as temporary_directory:
            executable = build_executable(
                source,
                Path(temporary_directory) / "aether-program.exe",
                compiler=compiler,
            )
            result = subprocess.run(
                [str(executable)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

        self.assertEqual(result.returncode, 0)

    def test_build_executable_links_real_c_abi_library(self) -> None:
        compiler = self.compiler()
        source = (
            '@ffi\n@link("sensor")\n'
            'extern "C" fn read_channel(channel: i32) -> i32;\n'
            "@ffi\n"
            "fn main() -> i32 { return read_channel(19); }\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            library_object = Path(temporary_directory) / "sensor.o"
            compile_c99_object(
                "#include <stdint.h>\n"
                "int32_t read_channel(int32_t channel) {\n"
                "    return channel + 7;\n"
                "}\n",
                library_object,
                compiler=compiler,
            )
            executable = build_executable(
                source,
                Path(temporary_directory) / "aether-program.exe",
                compiler=compiler,
                libraries={"sensor": library_object},
            )
            result = subprocess.run(
                [str(executable)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

        self.assertEqual(result.returncode, 26, result.stderr)

    def test_build_executable_roundtrips_all_supported_scalar_ffi_types(self) -> None:
        compiler = self.compiler()
        signatures = (
            ("i8", "i8"),
            ("i16", "i16"),
            ("i32", "i32"),
            ("i64", "i64"),
            ("u8", "u8"),
            ("u16", "u16"),
            ("u32", "u32"),
            ("u64", "u64"),
            ("f32", "f32"),
            ("f64", "f64"),
        )
        declarations = "\n".join(
            declaration
            for name, type_name in signatures
            for declaration in (
                f'@ffi @link("native") extern "C" fn get_{name}() -> {type_name};',
                f'@ffi @link("native") extern "C" fn check_{name}(value: {type_name}) -> i32;',
            )
        )
        source = (
            declarations
            + '\n@ffi @link("native") extern "C" fn mark_void() -> void;\n'
            + '@ffi @link("native") extern "C" fn check_void() -> i32;\n'
            + "@ffi fn main() -> i32 {\n"
            + "    mark_void();\n"
            + "    return "
            + " + ".join(
                f"check_{name}(get_{name}())" for name, _ in signatures
            )
            + " + check_void();\n}\n"
        )
        c_functions = (
            "#include <stdint.h>\n"
            "#include <stdbool.h>\n"
            "int8_t get_i8(void) { return INT8_C(-101); }\n"
            "int32_t check_i8(int8_t value) { return value == INT8_C(-101) ? 0 : 1; }\n"
            "int16_t get_i16(void) { return INT16_C(-30001); }\n"
            "int32_t check_i16(int16_t value) { return value == INT16_C(-30001) ? 0 : 1; }\n"
            "int32_t get_i32(void) { return INT32_C(-2000000001); }\n"
            "int32_t check_i32(int32_t value) { return value == INT32_C(-2000000001) ? 0 : 1; }\n"
            "int64_t get_i64(void) { return INT64_C(-9000000000000000001); }\n"
            "int32_t check_i64(int64_t value) { return value == INT64_C(-9000000000000000001) ? 0 : 1; }\n"
            "uint8_t get_u8(void) { return UINT8_C(250); }\n"
            "int32_t check_u8(uint8_t value) { return value == UINT8_C(250) ? 0 : 1; }\n"
            "uint16_t get_u16(void) { return UINT16_C(65000); }\n"
            "int32_t check_u16(uint16_t value) { return value == UINT16_C(65000) ? 0 : 1; }\n"
            "uint32_t get_u32(void) { return UINT32_C(4000000000); }\n"
            "int32_t check_u32(uint32_t value) { return value == UINT32_C(4000000000) ? 0 : 1; }\n"
            "uint64_t get_u64(void) { return UINT64_C(18000000000000000000); }\n"
            "int32_t check_u64(uint64_t value) { return value == UINT64_C(18000000000000000000) ? 0 : 1; }\n"
            "float get_f32(void) { return 1.25f; }\n"
            "int32_t check_f32(float value) { return value == 1.25f ? 0 : 1; }\n"
            "double get_f64(void) { return 2.5; }\n"
            "int32_t check_f64(double value) { return value == 2.5 ? 0 : 1; }\n"
            "static int void_called;\n"
            "void mark_void(void) { void_called = 1; }\n"
            "int32_t check_void(void) { return void_called ? 0 : 1; }\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            library_object = Path(temporary_directory) / "native.o"
            compile_c99_object(
                c_functions,
                library_object,
                compiler=compiler,
            )
            executable = build_executable(
                source,
                Path(temporary_directory) / "aether-program.exe",
                compiler=compiler,
                libraries={"native": library_object},
            )
            result = subprocess.run(
                [str(executable)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_build_executable_requires_exact_explicit_library_mapping(self) -> None:
        source = (
            '@ffi\n@link("sensor")\n'
            'extern "C" fn read_channel(channel: i32) -> i32;\n'
            "@ffi\n"
            "fn main() -> i32 { return read_channel(1); }\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(C99LinkConfigurationError):
                build_executable(
                    source,
                    Path(temporary_directory) / "program",
                    compiler=("aether-missing-c-compiler",),
                )
            with self.assertRaises(C99LinkConfigurationError):
                build_executable(
                    "fn main() -> i32 { return 0; }\n",
                    Path(temporary_directory) / "program",
                    compiler=("aether-missing-c-compiler",),
                    libraries={"unused": "unused.o"},
                )

    def test_build_executable_rejects_output_that_would_overwrite_library(self) -> None:
        compiler = self.compiler()
        source = (
            '@ffi @link("sensor") extern "C" fn read_value() -> i32;\n'
            "@ffi fn main() -> i32 { return read_value(); }\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            library_object = Path(temporary_directory) / "sensor.o"
            compile_c99_object(
                "int read_value(void) { return 1; }\n",
                library_object,
                compiler=compiler,
            )
            with self.assertRaises(C99LinkConfigurationError):
                build_executable(
                    source,
                    library_object,
                    compiler=compiler,
                    libraries={"sensor": library_object},
                )

    def test_build_executable_preserves_previous_artifact_on_link_error(self) -> None:
        compiler = self.compiler()
        source = (
            '@ffi\n@link("sensor")\n'
            'extern "C" fn missing_symbol(channel: i32) -> i32;\n'
            "@ffi\n"
            "fn main() -> i32 { return missing_symbol(1); }\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            library_object = Path(temporary_directory) / "sensor.o"
            compile_c99_object(
                "#include <stdint.h>\n"
                "int32_t other_symbol(uint32_t channel) { return (int32_t)channel; }\n",
                library_object,
                compiler=compiler,
            )
            executable_path = Path(temporary_directory) / "existing.exe"
            executable_path.write_bytes(b"previous executable")
            with self.assertRaises(C99BuildError):
                build_executable(
                    source,
                    executable_path,
                    compiler=compiler,
                    libraries={"sensor": library_object},
                )
            self.assertEqual(
                executable_path.read_bytes(),
                b"previous executable",
            )

    def test_build_executable_rejects_missing_or_invalid_entry_point(self) -> None:
        cases = (
            ("fn helper() -> i32 { return 0; }\n", "E_C99_ENTRY_POINT"),
            (
                "fn main(value: i32) -> i32 { return value; }\n",
                "E_C99_ENTRY_POINT",
            ),
            ("fn main() -> f32 { return 0.0; }\n", "E_C99_ENTRY_POINT"),
        )
        for source, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(CompilationError) as raised:
                    _ = build_executable(
                        source,
                        Path("unused.exe"),
                        compiler=("aether-missing-c-compiler",),
                    )
                self.assertIn(
                    expected_code,
                    {item.code for item in raised.exception.diagnostics},
                )

    def test_emits_deterministic_c99_with_checked_integer_arithmetic(self) -> None:
        source = (
            "fn add(left: i32, right: i32) -> i32 { return left + right; }\n"
            "fn twice(value: i32) -> i32 { return add(value, value); }\n"
        )

        generated = emit_c99(source)

        self.assertEqual(generated, emit_c99(source))
        self.assertIn(
            "#if defined(__clang__)\n#pragma STDC FP_CONTRACT OFF\n#endif",
            generated,
        )
        self.assertIn("#include <stdint.h>", generated)
        self.assertIn("INT32_MAX - right", generated)
        self.assertIn("INT32_MIN - right", generated)
        self.assertIn("aether_fn_616464", generated)
        self.assertIn("aether_fn_7477696365", generated)
        self.assertIn("aether_i32_add", generated)

    def test_checks_division_remainder_and_unsigned_overflow(self) -> None:
        generated = emit_c99(
            "fn divide(left: i64, right: i64) -> i64 { return left / right; }\n"
            "fn remainder(left: i32, right: i32) -> i32 { return left % right; }\n"
            "fn add(left: u8, right: u8) -> u8 { return left + right; }\n"
        )

        self.assertIn("left == INT64_MIN && right == -1", generated)
        self.assertIn("left == INT32_MIN && right == -1", generated)
        self.assertIn("UINT8_MAX - right", generated)

    def test_preserves_short_circuit_and_pipeline_call_order(self) -> None:
        generated = emit_c99(
            "fn accept(value: i32, offset: i32) -> i32 { return value + offset; }\n"
            "fn positive(value: i32) -> bool { return value > 0; }\n"
            "fn both(left: bool, value: i32) -> bool {\n"
            "  return left && positive(value);\n"
            "}\n"
            "fn pipeline(value: i32, offset: i32) -> i32 {\n"
            "  return value |> accept(offset);\n"
            "}\n"
        )

        self.assertIn("if (aether_tmp_0)", generated)
        self.assertIn("aether_fn_616363657074", generated)
        self.assertIn("aether_fn_706f736974697665", generated)
        self.assertIn("aether_fn_616363657074(aether_tmp_0, aether_tmp_1)", generated)

    def test_handles_signed_minimum_literal_without_overflow(self) -> None:
        generated = emit_c99(
            "fn min_value() -> i32 { return -2147483648; }\n"
        )

        self.assertIn("return INT32_MIN;", generated)

    def test_rejects_literals_outside_c99_type_ranges(self) -> None:
        for source in (
            "fn too_large() -> i32 { return 2147483648; }\n",
            "fn too_small() -> i32 { return -2147483649; }\n",
            "fn float_overflow() -> f32 { return 1e39; }\n",
        ):
            with self.subTest(source=source):
                with self.assertRaises(CompilationError) as raised:
                    emit_c99(source)
                self.assertIn(
                    "E_C99_LITERAL_RANGE",
                    {diagnostic.code for diagnostic in raised.exception.diagnostics},
                )

    def test_emits_void_functions_and_float_division_guard(self) -> None:
        generated = emit_c99(
            "fn noop() -> void { return; }\n"
            "fn ratio(left: f32, right: f32) -> f32 { return left / right; }\n"
        )

        self.assertIn("void aether_fn_6e6f6f70(void)", generated)
        self.assertIn("if (right == 0.0) aether_trap();", generated)
        self.assertIn("float aether_f32_div", generated)

    def test_compiles_static_float_tensor_operations(self) -> None:
        source = (
            "fn add_scalar(value: Tensor<f32, [2, 2]>, scalar: f32) "
            "-> Tensor<f32, [2, 2]> { return value + scalar; }\n"
            "fn add_tensors(left: Tensor<f32, [2, 2]>, right: Tensor<f32, [2, 2]>) "
            "-> Tensor<f32, [2, 2]> { return left + right; }\n"
            "fn add_scalar_left(scalar: f32, value: Tensor<f32, [2, 2]>) "
            "-> Tensor<f32, [2, 2]> { return scalar + value; }\n"
            "fn scale_value(value: Tensor<f32, [2, 2]>, scalar: f32) "
            "-> Tensor<f32, [2, 2]> { return scale(value, scalar); }\n"
            "fn multiply_scalar_left(scalar: f32, value: Tensor<f32, [2, 2]>) "
            "-> Tensor<f32, [2, 2]> { return scalar * value; }\n"
            "fn divide_value(value: Tensor<f32, [2, 2]>, scalar: f32) "
            "-> Tensor<f32, [2, 2]> { return value / scalar; }\n"
            "fn absolute(value: Tensor<f32, [2, 2]>) -> Tensor<f32, [2, 2]> "
            "{ return abs(value); }\n"
            "fn minimum(left: Tensor<f32, [2, 2]>, right: Tensor<f32, [2, 2]>) "
            "-> Tensor<f32, [2, 2]> { return min(left, right); }\n"
            "fn maximum(left: Tensor<f32, [2, 2]>, right: Tensor<f32, [2, 2]>) "
            "-> Tensor<f32, [2, 2]> { return max(left, right); }\n"
            "fn row(value: Tensor<f32, [2, 2]>, index: i32) "
            "-> Tensor<f32, [2]> { return value[index]; }\n"
            "fn matrix_item(value: Tensor<f32, [2, 2]>, row: i32, column: i32) "
            "-> f32 { return value[row][column]; }\n"
            "fn item(value: Tensor<f32, [2]>, index: i32) -> f32 "
            "{ return value[index]; }\n"
            "fn total(value: Tensor<f32, [2, 2]>) -> f32 { return sum(value); }\n"
            "fn average(value: Tensor<f32, [2, 2]>) -> f32 { return mean(value); }\n"
            "fn pipeline_average(value: Tensor<f32, [2, 2]>) -> f32 "
            "{ return value |> mean; }\n"
            "fn largest_magnitude(value: Tensor<f32, [2, 2]>) -> f32 "
            "{ return max_abs(value); }\n"
            "fn leading_length(value: Tensor<f32, [2, 2]>) -> i64 "
            "{ return len(value); }\n"
            "fn to_meters(value: Tensor<f32, [2], mm>) "
            "-> Tensor<f32, [2], m> { return convert(value, \"m\"); }\n"
            "fn add64(value: Tensor<f64, [2]>, scalar: f64) "
            "-> Tensor<f64, [2]> { return value + scalar; }\n"
        )
        generated = emit_c99(source)
        input_value = [[1.0, -2.0], [3.0, 4.0]]
        self.assertEqual(
            run_source(source, "add_scalar", input_value, 2.0),
            [[3.0, 0.0], [5.0, 6.0]],
        )
        self.assertEqual(
            run_source(source, "scale_value", input_value, 2.0),
            [[2.0, -4.0], [6.0, 8.0]],
        )
        converted_reference = run_source(source, "to_meters", [1000.0, 2500.0])
        self.assertEqual(converted_reference.value, [1.0, 2.5])
        harness = r"""
#include <math.h>
int main(void) {
    aether_tensor_f32_2x2 value = {{1.0f, -2.0f, 3.0f, 4.0f}};
    aether_tensor_f32_2x2 added = aether_fn_6164645f7363616c6172(value, 2.0f);
    aether_tensor_f32_2x2 pair = {{2.0f, -1.0f, 4.0f, 3.0f}};
    aether_tensor_f32_2x2 pair_sum = aether_fn_6164645f74656e736f7273(value, pair);
    aether_tensor_f32_2x2 left_added = aether_fn_6164645f7363616c61725f6c656674(2.0f, value);
    aether_tensor_f32_2x2 scaled = aether_fn_7363616c655f76616c7565(value, 2.0f);
    aether_tensor_f32_2x2 left_scaled = aether_fn_6d756c7469706c795f7363616c61725f6c656674(2.0f, value);
    aether_tensor_f32_2x2 divided = aether_fn_6469766964655f76616c7565(value, 2.0f);
    aether_tensor_f32_2x2 absolute = aether_fn_6162736f6c757465(value);
    aether_tensor_f32_2x2 minimum = aether_fn_6d696e696d756d(value, pair);
    aether_tensor_f32_2x2 maximum = aether_fn_6d6178696d756d(value, pair);
    aether_tensor_f32_2x2 left_nan = {{NAN, 1.0f, NAN, 4.0f}};
    aether_tensor_f32_2x2 right_nan = {{2.0f, NAN, 3.0f, NAN}};
    aether_tensor_f32_2x2 nan_minimum = aether_fn_6d696e696d756d(left_nan, right_nan);
    aether_tensor_f32_2x2 nan_maximum = aether_fn_6d6178696d756d(left_nan, right_nan);
    aether_tensor_f32_2 row = aether_fn_726f77(value, 1);
    aether_tensor_f32_2 meters = aether_fn_746f5f6d6574657273(
        (aether_tensor_f32_2){{1000.0f, 2500.0f}});
    aether_tensor_f64_2 wide = aether_fn_6164643634(
        (aether_tensor_f64_2){{1.0, 3.0}}, 0.5);
    if (added.data[2] != 5.0f || pair_sum.data[2] != 7.0f) return 1;
    if (left_added.data[1] != 0.0f || scaled.data[1] != -4.0f) return 2;
    if (left_scaled.data[0] != 2.0f || absolute.data[1] != 2.0f) return 3;
    if (minimum.data[1] != -2.0f || maximum.data[3] != 4.0f) return 4;
    if (!isnan(nan_minimum.data[0]) || nan_minimum.data[1] != 1.0f) return 14;
    if (!isnan(nan_maximum.data[2]) || nan_maximum.data[3] != 4.0f) return 15;
    if (divided.data[3] != 2.0f || row.data[0] != 3.0f) return 5;
    if (aether_fn_6d61747269785f6974656d(value, 1, 0) != 3.0f) return 16;
    if (aether_fn_6974656d(row, 1) != 4.0f) return 6;
    if (aether_fn_746f74616c(value) != 6.0f) return 7;
    if (aether_fn_61766572616765(value) != 1.5f) return 8;
    if (aether_fn_706970656c696e655f61766572616765(value) != 1.5f) return 9;
    if (aether_fn_6c6172676573745f6d61676e6974756465(value) != 4.0f) return 10;
    if (aether_fn_6c656164696e675f6c656e677468(value) != 2) return 11;
    if (meters.data[0] != 1.0f || meters.data[1] != 2.5f) return 12;
    if (wide.data[0] != 1.5 || wide.data[1] != 3.5) return 13;
    return 0;
}
"""
        executable = self.compile_program(generated, harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_multidimensional_f64_tensor_results_match_interpreter(self) -> None:
        source = (
            "fn add_scalar(value: Tensor<f64, [2, 2, 2]>, scalar: f64) "
            "-> Tensor<f64, [2, 2, 2]> { return value + scalar; }\n"
            "fn reverse_subtract(scalar: f64, value: Tensor<f64, [2, 2, 2]>) "
            "-> Tensor<f64, [2, 2, 2]> { return scalar - value; }\n"
            "fn multiply_left(scalar: f64, value: Tensor<f64, [2, 2, 2]>) "
            "-> Tensor<f64, [2, 2, 2]> { return scalar * value; }\n"
            "fn divide(value: Tensor<f64, [2, 2, 2]>, scalar: f64) "
            "-> Tensor<f64, [2, 2, 2]> { return value / scalar; }\n"
            "fn minimum(left: Tensor<f64, [2, 2, 2]>, "
            "right: Tensor<f64, [2, 2, 2]>) -> Tensor<f64, [2, 2, 2]> "
            "{ return min(left, right); }\n"
            "fn maximum(left: Tensor<f64, [2, 2, 2]>, "
            "right: Tensor<f64, [2, 2, 2]>) -> Tensor<f64, [2, 2, 2]> "
            "{ return max(left, right); }\n"
            "fn plane(value: Tensor<f64, [2, 2, 2]>) "
            "-> Tensor<f64, [2, 2]> { return value[1]; }\n"
            "fn element(value: Tensor<f64, [2, 2, 2]>) -> f64 "
            "{ return value[1][0][1]; }\n"
            "fn total(value: Tensor<f64, [2, 2, 2]>) -> f64 "
            "{ return sum(value); }\n"
            "fn average(value: Tensor<f64, [2, 2, 2]>) -> f64 "
            "{ return mean(value); }\n"
            "fn largest_magnitude(value: Tensor<f64, [2, 2, 2]>) -> f64 "
            "{ return max_abs(value); }\n"
        )
        values = [
            [[-1.25, 2.5], [3.75, -4.5]],
            [[5.25, -6.5], [7.75, 8.5]],
        ]
        other = [
            [[0.5, 3.0], [4.25, -2.0]],
            [[6.0, -7.25], [2.5, 9.0]],
        ]
        scalar = 1.5
        function_calls = (
            ("add_scalar", "value, scalar", (values, scalar)),
            ("reverse_subtract", "scalar, value", (scalar, values)),
            ("multiply_left", "scalar, value", (scalar, values)),
            ("divide", "value, scalar", (values, scalar)),
            ("minimum", "value, other", (values, other)),
            ("maximum", "value, other", (values, other)),
            ("plane", "value", (values,)),
            ("element", "value", (values,)),
            ("total", "value", (values,)),
            ("average", "value", (values,)),
            ("largest_magnitude", "value", (values,)),
        )
        generated = emit_c99(source)
        harness_lines = [
            "#include <stdio.h>",
            "int main(void) {",
            "    aether_tensor_f64_2x2x2 value = "
            "{{-1.25, 2.5, 3.75, -4.5, 5.25, -6.5, 7.75, 8.5}};",
            "    aether_tensor_f64_2x2x2 other = "
            "{{0.5, 3.0, 4.25, -2.0, 6.0, -7.25, 2.5, 9.0}};",
        ]
        for name, arguments, _ in function_calls:
            if name in {"add_scalar", "divide"}:
                call_args = "value, 1.5"
            elif name in {"reverse_subtract", "multiply_left"}:
                call_args = "1.5, value"
            elif name in {"minimum", "maximum"}:
                call_args = "value, other"
            else:
                call_args = arguments
            result_type = (
                "aether_tensor_f64_2x2"
                if name == "plane"
                else "aether_tensor_f64_2x2x2"
                if name in {"add_scalar", "reverse_subtract", "multiply_left",
                            "divide", "minimum", "maximum"}
                else "double"
            )
            result_name = f"result_{name}"
            harness_lines.append(
                f"    {result_type} {result_name} = "
                f"aether_fn_{name.encode('utf-8').hex()}({call_args});"
            )
            if name in {"add_scalar", "reverse_subtract", "multiply_left",
                        "divide", "minimum", "maximum", "plane"}:
                element_count = 4 if name == "plane" else 8
                harness_lines.extend(
                    f'    printf("%.17g\\n", {result_name}.data[{index}]);'
                    for index in range(element_count)
                )
            else:
                harness_lines.append(
                    f'    printf("%.17g\\n", {result_name});'
                )
        harness_lines.extend(("    return 0;", "}"))
        executable = self.compile_program(generated, "\n".join(harness_lines))
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = [float(item) for item in result.stdout.splitlines()]

        expected: list[float] = []
        for name, _, arguments in function_calls:
            reference = run_source(source, name, *arguments)
            raw = reference.value if hasattr(reference, "value") else reference
            if isinstance(raw, (list, tuple)):
                def flatten(items: object) -> list[float]:
                    if isinstance(items, (list, tuple)):
                        return [number for item in items for number in flatten(item)]
                    return [float(items)]

                expected.extend(flatten(raw))
            else:
                expected.append(float(raw))

        self.assertEqual(len(actual), len(expected))
        for index, (native, reference) in enumerate(zip(actual, expected)):
            with self.subTest(result=index):
                self.assertTrue(
                    math.isclose(native, reference, rel_tol=1e-14, abs_tol=1e-14),
                    f"C99={native}, interpreter={reference}",
                )

    def test_tensor_scalar_broadcasting_matches_interpreter_across_ranks(self) -> None:
        source = (
            "fn rank1_add(value: Tensor<f32, [3]>, scalar: f32) "
            "-> Tensor<f32, [3]> { return value + scalar; }\n"
            "fn rank1_reverse_subtract(scalar: f32, value: Tensor<f32, [3]>) "
            "-> Tensor<f32, [3]> { return scalar - value; }\n"
            "fn rank2_subtract(left: Tensor<f64, [2, 2]>, "
            "right: Tensor<f64, [2, 2]>) -> Tensor<f64, [2, 2]> "
            "{ return left - right; }\n"
            "fn rank2_multiply_left(scalar: f64, value: Tensor<f64, [2, 2]>) "
            "-> Tensor<f64, [2, 2]> { return scalar * value; }\n"
            "fn rank3_add(value: Tensor<f32, [2, 1, 2]>, scalar: f32) "
            "-> Tensor<f32, [2, 1, 2]> { return value + scalar; }\n"
            "fn rank3_divide(value: Tensor<f32, [2, 1, 2]>, scalar: f32) "
            "-> Tensor<f32, [2, 1, 2]> { return value / scalar; }\n"
        )
        cases = (
            (
                "rank1_add",
                "aether_tensor_f32_3",
                "value, 0.5f",
                ([1.0, -2.0, 3.5], 0.5),
                3,
            ),
            (
                "rank1_reverse_subtract",
                "aether_tensor_f32_3",
                "2.0f, value",
                (2.0, [1.0, -2.0, 3.5]),
                3,
            ),
            (
                "rank2_subtract",
                "aether_tensor_f64_2x2",
                "left, right",
                ([[1.0, 2.5], [-3.0, 4.0]], [[0.5, 1.5], [-1.0, 2.0]]),
                4,
            ),
            (
                "rank2_multiply_left",
                "aether_tensor_f64_2x2",
                "1.5, value2",
                (1.5, [[1.0, 2.5], [-3.0, 4.0]]),
                4,
            ),
            (
                "rank3_add",
                "aether_tensor_f32_2x1x2",
                "value3, 0.25f",
                ([[[1.0, 2.0]], [[-3.0, 4.0]]], 0.25),
                4,
            ),
            (
                "rank3_divide",
                "aether_tensor_f32_2x1x2",
                "value3, 2.0f",
                ([[[1.0, 2.0]], [[-3.0, 4.0]]], 2.0),
                4,
            ),
        )
        harness_lines = [
            "#include <stdio.h>",
            "int main(void) {",
            "    aether_tensor_f32_3 value = {{1.0f, -2.0f, 3.5f}};",
            "    aether_tensor_f64_2x2 left = {{1.0, 2.5, -3.0, 4.0}};",
            "    aether_tensor_f64_2x2 right = {{0.5, 1.5, -1.0, 2.0}};",
            "    aether_tensor_f64_2x2 value2 = {{1.0, 2.5, -3.0, 4.0}};",
            "    aether_tensor_f32_2x1x2 value3 = {{1.0f, 2.0f, -3.0f, 4.0f}};",
        ]
        for name, result_type, c_arguments, _, count in cases:
            result_name = f"result_{name}"
            harness_lines.append(
                f"    {result_type} {result_name} = "
                f"aether_fn_{name.encode('utf-8').hex()}({c_arguments});"
            )
            harness_lines.extend(
                f'    printf("%.17g\\n", (double){result_name}.data[{index}]);'
                for index in range(count)
            )
        harness_lines.extend(("    return 0;", "}"))

        executable = self.compile_program(emit_c99(source), "\n".join(harness_lines))
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = [float(item) for item in result.stdout.splitlines()]

        def flatten(items: object) -> list[float]:
            if isinstance(items, (list, tuple)):
                return [number for item in items for number in flatten(item)]
            return [float(items)]

        expected = [
            number
            for name, _, _, arguments, _ in cases
            for number in flatten(run_source(source, name, *arguments))
        ]
        self.assertEqual(actual, expected)

    def test_f32_tensor_operations_match_interpreter_at_f32_precision(self) -> None:
        source = (
            "fn add_scalar(value: Tensor<f32, [4]>, scalar: f32) "
            "-> Tensor<f32, [4]> { return value + scalar; }\n"
            "fn subtract_scalar(value: Tensor<f32, [4]>, scalar: f32) "
            "-> Tensor<f32, [4]> { return value - scalar; }\n"
            "fn reverse_subtract(scalar: f32, value: Tensor<f32, [4]>) "
            "-> Tensor<f32, [4]> { return scalar - value; }\n"
            "fn multiply_scalar(scalar: f32, value: Tensor<f32, [4]>) "
            "-> Tensor<f32, [4]> { return scalar * value; }\n"
            "fn divide_scalar(value: Tensor<f32, [4]>, scalar: f32) "
            "-> Tensor<f32, [4]> { return value / scalar; }\n"
            "fn total(value: Tensor<f32, [4]>) -> f32 { return sum(value); }\n"
            "fn average(value: Tensor<f32, [4]>) -> f32 { return mean(value); }\n"
        )

        def as_f32(value: float) -> float:
            return struct.unpack("f", struct.pack("f", value))[0]

        values = [
            as_f32(value)
            for value in (12345.678, 0.0001234, 0.3333333, 9876.543)
        ]
        scalar = as_f32(0.1234567)
        functions = (
            ("add_scalar", "value, scalar", (values, scalar), True),
            ("subtract_scalar", "value, scalar", (values, scalar), True),
            ("reverse_subtract", "scalar, value", (scalar, values), True),
            ("multiply_scalar", "scalar, value", (scalar, values), True),
            ("divide_scalar", "value, scalar", (values, scalar), True),
            ("total", "value", (values,), False),
            ("average", "value", (values,), False),
        )
        lines = [
            "#include <stdio.h>",
            "int main(void) {",
            "    aether_tensor_f32_4 value = "
            f"{{{{{', '.join(f'{item:.9g}f' for item in values)}}}}};",
        ]
        for name, c_arguments, _, returns_tensor in functions:
            call_arguments = c_arguments.replace("scalar", f"{scalar:.9g}f")
            result_name = f"result_{name}"
            if returns_tensor:
                lines.append(
                    f"    aether_tensor_f32_4 {result_name} = "
                    f"aether_fn_{name.encode('utf-8').hex()}({call_arguments});"
                )
                lines.extend(
                    f'    printf("%.17g\\n", (double){result_name}.data[{index}]);'
                    for index in range(len(values))
                )
            else:
                lines.append(
                    f'    printf("%.17g\\n", (double)aether_fn_'
                    f'{name.encode("utf-8").hex()}({call_arguments}));'
                )
        lines.extend(("    return 0;", "}"))

        executable = self.compile_program(emit_c99(source), "\n".join(lines))
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = [float(value) for value in result.stdout.splitlines()]

        expected: list[float] = []
        for name, _, arguments, _ in functions:
            reference = run_source(source, name, *arguments)
            raw = reference.value if hasattr(reference, "value") else reference
            if isinstance(raw, (list, tuple)):
                expected.extend(float(item) for item in raw)
            else:
                expected.append(float(raw))

        self.assertEqual(len(actual), len(expected))
        for index, (native, reference) in enumerate(zip(actual, expected)):
            with self.subTest(result=index):
                self.assertEqual(native, reference)

    def test_f32_rounding_matches_interpreter_for_reductions_and_gradients(self) -> None:
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
        generated = emit_c99(source)
        harness = r"""
#include <stdio.h>
int main(void) {
    aether_tensor_f32_4 values = {{16777216.0f, 1.0f, -16777216.0f, 1.0f}};
    aether_tensor_f32_4 gradient =
        aether_fn_74656e736f725f64657269766174697665(values);
    printf("%.17g\n", (double)aether_fn_6964656e74697479(16777217.0f));
    printf("%.17g\n", (double)aether_fn_6164645f6f6e65(16777216.0f));
    printf("%.17g\n", (double)aether_fn_6d756c7469706c79(3.4e38f, 2.0f));
    printf("%.17g\n", (double)aether_fn_6d756c7469706c79(1.0e-30f, 1.0e-10f));
    printf("%.17g\n", (double)aether_fn_746f74616c(values));
    printf("%.17g\n", (double)aether_fn_61766572616765(values));
    printf("%.17g\n", (double)aether_fn_64657269766174697665(4097.25f));
    for (size_t index = 0; index < 4; ++index)
        printf("%.17g\n", (double)gradient.data[index]);
    return 0;
}
"""
        executable = self.compile_program(generated, harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = [float(value) for value in result.stdout.splitlines()]

        expected = [
            run_source(source, "identity", 16777217),
            run_source(source, "add_one", 16777216.0),
            run_source(source, "multiply", 3.4e38, 2.0),
            run_source(source, "multiply", 1.0e-30, 1.0e-10),
            run_source(source, "total", values),
            run_source(source, "average", values),
            run_source(source, "derivative", 4097.25),
            *run_source(source, "tensor_derivative", values),
        ]
        self.assertEqual(actual, expected)

    def test_tensor_indexing_traps_negative_and_out_of_bounds_indexes(self) -> None:
        source = (
            "fn item(value: Tensor<f32, [2]>, index: i32) -> f32 "
            "{ return value[index]; }\n"
        )
        harness = r"""
int main(int argc, char **argv) {
    aether_tensor_f32_2 value = {{1.0f, 2.0f}};
    if (argc != 2) return 3;
    if (argv[1][0] == 'n') (void)aether_fn_6974656d(value, -1);
    else (void)aether_fn_6974656d(value, 2);
    return 0;
}
"""
        executable = self.compile_program(emit_c99(source), harness)
        for index in ("negative", "out-of-bounds"):
            result = self.run_program(executable, index)
            self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_tensor_reductions_and_unit_conversions_match_interpreter(self) -> None:
        source = (
            "fn total(value: Tensor<f32, [3]>) -> f32 { return sum(value); }\n"
            "fn average(value: Tensor<f32, [3]>) -> f32 { return mean(value); }\n"
            "fn largest_magnitude(value: Tensor<f32, [3]>) -> f32 "
            "{ return max_abs(value); }\n"
            "fn to_meters(value: Tensor<f32, [2], mm>) "
            "-> Tensor<f32, [2], m> { return convert(value, \"m\"); }\n"
            "fn fahrenheit(value: Tensor<f32, [2], degC>) "
            "-> Tensor<f32, [2], degF> { return convert(value, \"degF\"); }\n"
            "fn radians(value: Tensor<f32, [2], deg>) "
            "-> Tensor<f32, [2], rad> { return convert(value, \"rad\"); }\n"
        )
        generated = emit_c99(source)
        harness = r"""
#include <math.h>
#include <stdio.h>
int main(void) {
    aether_tensor_f32_3 normal = {{1.0f, -2.0f, 3.0f}};
    aether_tensor_f32_3 infinities = {{INFINITY, -INFINITY, 1.0f}};
    aether_tensor_f32_3 nan_first = {{NAN, 2.0f, 3.0f}};
    aether_tensor_f32_3 nan_middle = {{1.0f, NAN, 3.0f}};
    aether_tensor_f32_3 negative_zeros = {{-0.0f, -0.0f, -0.0f}};
    aether_tensor_f32_2 meters = aether_fn_746f5f6d6574657273(
        (aether_tensor_f32_2){{1000.0f, 2500.0f}});
    aether_tensor_f32_2 fahrenheit = aether_fn_66616872656e68656974(
        (aether_tensor_f32_2){{0.0f, 100.0f}});
    aether_tensor_f32_2 radians = aether_fn_72616469616e73(
        (aether_tensor_f32_2){{180.0f, -90.0f}});
    printf("%.17g\n", (double)aether_fn_746f74616c(normal));
    printf("%.17g\n", (double)aether_fn_61766572616765(normal));
    printf("%.17g\n", (double)aether_fn_6c6172676573745f6d61676e6974756465(normal));
    printf("%.17g\n", (double)aether_fn_746f74616c(infinities));
    printf("%.17g\n", (double)aether_fn_61766572616765(infinities));
    printf("%.17g\n", (double)aether_fn_6c6172676573745f6d61676e6974756465(nan_first));
    printf("%.17g\n", (double)aether_fn_6c6172676573745f6d61676e6974756465(nan_middle));
    printf("%.17g\n", (double)aether_fn_6c6172676573745f6d61676e6974756465(negative_zeros));
    printf("%.17g\n", (double)meters.data[0]);
    printf("%.17g\n", (double)meters.data[1]);
    printf("%.17g\n", (double)fahrenheit.data[0]);
    printf("%.17g\n", (double)fahrenheit.data[1]);
    printf("%.17g\n", (double)radians.data[0]);
    printf("%.17g\n", (double)radians.data[1]);
    return 0;
}
"""
        executable = self.compile_program(generated, harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = [float(value) for value in result.stdout.splitlines()]

        normal = [1.0, -2.0, 3.0]
        infinities = [math.inf, -math.inf, 1.0]
        nan_first = [math.nan, 2.0, 3.0]
        nan_middle = [1.0, math.nan, 3.0]
        negative_zeros = [-0.0, -0.0, -0.0]
        expected = [
            run_source(source, "total", normal),
            run_source(source, "average", normal),
            run_source(source, "largest_magnitude", normal),
            run_source(source, "total", infinities),
            run_source(source, "average", infinities),
            run_source(source, "largest_magnitude", nan_first),
            run_source(source, "largest_magnitude", nan_middle),
            run_source(source, "largest_magnitude", negative_zeros),
        ]
        for function, values in (
            ("to_meters", [1000.0, 2500.0]),
            ("fahrenheit", [0.0, 100.0]),
            ("radians", [180.0, -90.0]),
        ):
            converted = run_source(source, function, values)
            expected.extend(converted.value)

        self.assertEqual(len(actual), len(expected))
        for index, (native, reference) in enumerate(zip(actual, expected)):
            with self.subTest(result=index):
                if math.isnan(reference):
                    self.assertTrue(math.isnan(native))
                elif math.isinf(reference):
                    self.assertEqual(native, reference)
                else:
                    self.assertAlmostEqual(native, reference, delta=1e-6)
                    if reference == 0.0:
                        self.assertEqual(
                            math.copysign(1.0, native),
                            math.copysign(1.0, reference),
                        )

    def test_f64_reductions_match_sequential_interpreter_order(self) -> None:
        source = (
            "fn total(values: Tensor<f64, [2, 2]>) -> f64 {\n"
            "  return sum(values);\n"
            "}\n"
            "fn average(values: Tensor<f64, [2, 2]>) -> f64 {\n"
            "  return mean(values);\n"
            "}\n"
        )
        harness = r"""
#include <stdio.h>
int main(void) {
    aether_tensor_f64_2x2 first = {{1e16, 1.0, -1e16, 1.0}};
    aether_tensor_f64_2x2 second = {{1e16, -1e16, 1.0, 1.0}};
    printf("%.17g %.17g %.17g %.17g\n",
        aether_fn_746f74616c(first),
        aether_fn_61766572616765(first),
        aether_fn_746f74616c(second),
        aether_fn_61766572616765(second));
    return 0;
}
"""
        executable = self.compile_program(emit_c99(source), harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = [float(value) for value in result.stdout.split()]
        first_order = [[1e16, 1.0], [-1e16, 1.0]]
        second_order = [[1e16, -1e16], [1.0, 1.0]]
        expected = [
            run_source(source, "total", first_order),
            run_source(source, "average", first_order),
            run_source(source, "total", second_order),
            run_source(source, "average", second_order),
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(actual, [1.0, 0.25, 2.0, 0.5])

    def test_tensor_division_by_zero_traps(self) -> None:
        source = (
            "fn divide(value: Tensor<f32, [2]>, scalar: f32) "
            "-> Tensor<f32, [2]> { return value / scalar; }\n"
        )
        harness = r"""
int main(void) {
    aether_tensor_f32_2 value = {{1.0f, 2.0f}};
    (void)aether_fn_646976696465(value, 0.0f);
    return 0;
}
"""
        executable = self.compile_program(emit_c99(source), harness)
        result = self.run_program(executable)
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_compiles_reverse_mode_scalar_gradient(self) -> None:
        source = (
            "fn objective(x: f64) -> f64 {\n"
            "  let square = x * x;\n"
            "  return square * square + x;\n"
            "}\n"
            "fn derivative(x: f64) -> f64 {\n"
            "  return gradient(objective, x);\n"
            "}\n"
            "fn ratio(x: f32) -> f32 { return x / x; }\n"
            "fn ratio_derivative(x: f32) -> f32 "
            "{ return gradient(ratio, x); }\n"
        )
        generated = emit_c99(source)
        harness = r"""
#include <stdio.h>
int main(void) {
    printf("%.12g\n", aether_fn_64657269766174697665(2.0));
    printf("%.12g\n", aether_fn_64657269766174697665(-1.0));
    printf("%.9g\n", aether_fn_726174696f5f64657269766174697665(2.0f));
    return 0;
}
"""
        executable = self.compile_program(generated, harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                format(run_source(source, "derivative", 2.0), ".12g"),
                format(run_source(source, "derivative", -1.0), ".12g"),
                format(run_source(source, "ratio_derivative", 2.0), ".9g"),
            ],
        )

    def test_compiled_scalar_autodiff_supports_linear_unit_conversion(self) -> None:
        source = (
            "fn objective(value: f64) -> f64 {\n"
            "  return convert(value, \"1\");\n"
            "}\n"
            "fn derivative(value: f64) -> f64 {\n"
            "  return gradient(objective, value);\n"
            "}\n"
        )
        generated = emit_c99(source)
        harness = r"""
#include <stdio.h>
int main(void) {
    printf("%.17g\n", aether_fn_64657269766174697665(1.0));
    return 0;
}
"""
        executable = self.compile_program(generated, harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        reference = run_source(source, "derivative", 1.0)
        self.assertEqual(float(result.stdout.strip()), reference)

    def test_pi_power_unit_conversion_matches_interpreter_exactly(self) -> None:
        source = (
            "fn radian_area(value: f64<deg^2>) -> f64<rad^2> {\n"
            "  return convert(value, \"rad^2\");\n"
            "}\n"
        )
        generated = emit_c99(source)
        self.assertNotIn("3.14159265358979323846264338327950288", generated)
        harness = r"""
#include <stdio.h>
int main(void) {
    printf("%.17g\n", aether_fn_72616469616e5f61726561(1.0));
    return 0;
}
"""
        executable = self.compile_program(generated, harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        reference = run_source(source, "radian_area", 1.0)
        self.assertEqual(float(result.stdout.strip()), reference.value)

    def test_compiles_tensor_reverse_mode_gradient(self) -> None:
        source = (
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
            "fn objective_cm(values: Tensor<f32, [2], cm>) -> f32<m> {\n"
            "  return convert(sum(values), \"m\");\n"
            "}\n"
            "fn derivative_cm(values: Tensor<f32, [2], cm>) "
            "-> Tensor<f32, [2], m/cm> {\n"
            "  return gradient(objective_cm, values);\n"
            "}\n"
            "fn objective_f64(values: Tensor<f64, [2, 2]>) -> f64 {\n"
            "  let average = mean(values);\n"
            "  return sum(average + values);\n"
            "}\n"
            "fn derivative_f64(values: Tensor<f64, [2, 2]>) "
            "-> Tensor<f64, [2, 2]> {\n"
            "  return gradient(objective_f64, values);\n"
            "}\n"
            "fn single(values: Tensor<f64, [1]>) -> f64 { return mean(values); }\n"
            "fn single_gradient(values: Tensor<f64, [1]>) "
            "-> Tensor<f64, [1]> { return gradient(single, values); }\n"
        )
        generated = emit_c99(source)
        self.assertIn("SIZE_MAX / sizeof(*tape)", generated)
        self.assertIn("malloc((size_t)", generated)
        self.assertIn("free(tape);", generated)
        harness = r"""
#include <stdio.h>
int main(void) {
    aether_tensor_f32_2 values = {{1.0f, -3.0f}};
    aether_tensor_f32_2 gradient = aether_fn_64657269766174697665(values);
    aether_tensor_f32_2 average_gradient =
        aether_fn_617665726167655f6772616469656e74(values);
    aether_tensor_f32_2 negative_gradient =
        aether_fn_6e656761746976655f6772616469656e74(values);
    aether_tensor_f32_2 scalar_product_gradient =
        aether_fn_7363616c61725f70726f647563745f6772616469656e74(values);
    aether_tensor_f32_2 centimeters = {{2.0f, 5.0f}};
    aether_tensor_f32_2 unit_gradient =
        aether_fn_646572697661746976655f636d(centimeters);
    aether_tensor_f64_2x2 matrix = {{1.0, 2.0, 3.0, 4.0}};
    aether_tensor_f64_2x2 matrix_gradient =
        aether_fn_646572697661746976655f663634(matrix);
    aether_tensor_f64_1 singleton = {{42.0}};
    aether_tensor_f64_1 singleton_gradient =
        aether_fn_73696e676c655f6772616469656e74(singleton);
    printf("%.7g %.7g %.7g %.7g %.7g %.7g "
           "%.7g %.7g %.7g %.7g %.7g %.7g %.7g %.7g %.7g\n",
        gradient.data[0], gradient.data[1],
        average_gradient.data[0], average_gradient.data[1],
        negative_gradient.data[0], negative_gradient.data[1],
        scalar_product_gradient.data[0], scalar_product_gradient.data[1],
        unit_gradient.data[0], unit_gradient.data[1],
        matrix_gradient.data[0], matrix_gradient.data[1],
        matrix_gradient.data[2], matrix_gradient.data[3],
        singleton_gradient.data[0]);
    return 0;
}
"""
        executable = self.compile_program(generated, harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "3 3 1 1 -1 -1 -2 -2 0.01 0.01 2 2 2 2 1",
        )

    def test_compiled_f32_tensor_gradient_chains_shared_reductions_and_units(self) -> None:
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
        values = [1.1, -3.2]
        generated = emit_c99(source)
        harness = r"""
#include <stdio.h>
int main(void) {
    aether_tensor_f32_2 values = {{1.1f, -3.2f}};
    aether_tensor_f32_2 gradient =
        aether_fn_64657269766174697665(values);
    aether_tensor_f32_2 unit_gradient =
        aether_fn_646572697661746976655f636d(values);
    printf("%.9g %.9g %.9g %.9g\n",
        gradient.data[0], gradient.data[1],
        unit_gradient.data[0], unit_gradient.data[1]);
    return 0;
}
"""
        executable = self.compile_program(generated, harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = [float(item) for item in result.stdout.split()]

        reference = run_source(source, "derivative", values)
        unit_reference = run_source(source, "derivative_cm", values)
        self.assertIsInstance(unit_reference, Quantity)
        self.assertEqual(reference, [0.25, 0.25])
        self.assertEqual(len(actual), 4)
        for index in range(2):
            with self.subTest(component=index):
                self.assertAlmostEqual(actual[index], reference[index], delta=1e-7)
                self.assertAlmostEqual(actual[index], 0.25, delta=1e-7)
                self.assertAlmostEqual(
                    actual[index + 2],
                    unit_reference.value[index],
                    delta=1e-9,
                )
                self.assertAlmostEqual(actual[index + 2], 0.0075, delta=1e-9)

    def test_compiled_f32_mean_gradient_rounds_local_derivative_factor(self) -> None:
        source = (
            "fn objective(values: Tensor<f32, [3]>) -> f32 {\n"
            "  return mean(values) * -6185394176.0;\n"
            "}\n"
            "fn derivative(values: Tensor<f32, [3]>) -> Tensor<f32, [3]> {\n"
            "  return gradient(objective, values);\n"
            "}\n"
        )
        harness = r"""
int main(void) {
    aether_tensor_f32_3 values = {{0.0f, 0.0f, 0.0f}};
    aether_tensor_f32_3 gradient =
        aether_fn_64657269766174697665(values);
    if (gradient.data[0] != -2061798144.0f ||
        gradient.data[1] != -2061798144.0f ||
        gradient.data[2] != -2061798144.0f) return 1;
    return 0;
}
"""
        executable = self.compile_program(emit_c99(source), harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_compiled_f32_autodiff_division_gradients(self) -> None:
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
        generated = emit_c99(source)
        harness = r"""
#include <stdio.h>
int main(void) {
    aether_tensor_f32_2 positive = {{1.0f, 3.0f}};
    aether_tensor_f32_2 negative = {{-5.0f, -3.0f}};
    aether_tensor_f32_2 small = {{0.0f, 0.0f}};
    aether_tensor_f32_2 positive_gradient =
        aether_fn_74656e736f725f6772616469656e74(positive);
    aether_tensor_f32_2 negative_gradient =
        aether_fn_74656e736f725f6772616469656e74(negative);
    aether_tensor_f32_2 small_gradient =
        aether_fn_74656e736f725f736d616c6c5f6772616469656e74(small);
    printf("%.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
        positive_gradient.data[0], positive_gradient.data[1],
        negative_gradient.data[0], negative_gradient.data[1],
        small_gradient.data[0], small_gradient.data[1],
        aether_fn_7363616c61725f6772616469656e74(2.0f),
        aether_fn_7363616c61725f6772616469656e74(-2.0f),
        aether_fn_7363616c61725f6772616469656e74(1.0e-10f));
    return 0;
}
"""
        executable = self.compile_program(generated, harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = [float(value) for value in result.stdout.split()]

        cases = (
            ("tensor_gradient", ([1.0, 3.0],), [0.125, 0.125]),
            ("tensor_gradient", ([-5.0, -3.0],), [0.5, 0.5]),
            ("tensor_small_gradient", ([0.0, 0.0],), [1.0e10, 1.0e10]),
            ("scalar_gradient", (2.0,), [-0.75]),
            ("scalar_gradient", (-2.0,), [-0.75]),
            ("scalar_gradient", (1.0e-10,), [-3.0e20]),
        )
        expected: list[float] = []
        for function, arguments, analytic in cases:
            reference = run_source(source, function, *arguments)
            raw = reference.value if isinstance(reference, Quantity) else reference
            values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
            self.assertEqual(len(values), len(analytic))
            for value, expected_value in zip(values, analytic):
                self.assertAlmostEqual(
                    float(value),
                    expected_value,
                    delta=max(1e-6, abs(expected_value) * 1e-6),
                )
            expected.extend(float(value) for value in values)

        self.assertEqual(len(actual), len(expected))
        for index, (native, reference) in enumerate(zip(actual, expected)):
            with self.subTest(result=index):
                self.assertAlmostEqual(
                    native,
                    reference,
                    delta=max(1e-6, abs(reference) * 1e-6),
                )

    def test_compiled_f32_autodiff_adjoint_accumulation_with_cancellation(self) -> None:
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
        generated = emit_c99(source)
        harness = r"""
#include <stdio.h>
int main(void) {
    aether_tensor_f32_2 values = {{1.0f, -2.0f}};
    aether_tensor_f32_2 gradient =
        aether_fn_74656e736f725f6772616469656e74(values);
    printf("%.9g %.9g %.9g\n",
        aether_fn_63616e63656c6c6174696f6e5f6772616469656e74(1.0f),
        aether_fn_66756c6c5f63616e63656c6c6174696f6e5f6772616469656e74(1.0f),
        gradient.data[0]);
    printf("%.9g\n", gradient.data[1]);
    return 0;
}
"""
        executable = self.compile_program(generated, harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = [float(value) for value in result.stdout.split()]

        cases = (
            ("cancellation_gradient", (1.0,), 1.0),
            ("full_cancellation_gradient", (1.0,), 0.0),
            ("tensor_gradient", ([1.0, -2.0],), [1.0, 1.0]),
        )
        expected: list[float] = []
        for function, arguments, analytic in cases:
            reference = run_source(source, function, *arguments)
            raw = reference.value if isinstance(reference, Quantity) else reference
            values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
            analytic_values = list(analytic) if isinstance(analytic, (list, tuple)) else [analytic]
            self.assertEqual(values, analytic_values)
            expected.extend(float(value) for value in values)

        self.assertEqual(actual, expected)

    def test_compiled_f32_autodiff_accumulates_many_cancelling_paths(self) -> None:
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
        generated = emit_c99(source)
        harness = r"""
#include <stdio.h>
int main(void) {
    aether_tensor_f32_2 values = {{1.0f, -2.0f}};
    aether_tensor_f32_2 gradient =
        aether_fn_6d616e795f74656e736f725f70617468735f6772616469656e74(values);
    printf("%.9g %.9g %.9g %.9g\n",
        aether_fn_6d616e795f70617468735f6772616469656e74(1.0f),
        aether_fn_726573696475616c5f66697273745f6772616469656e74(1.0f),
        gradient.data[0], gradient.data[1]);
    return 0;
}
"""
        executable = self.compile_program(generated, harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            [float(value) for value in result.stdout.split()],
            [1.0, 1.0, 1.0, 1.0],
        )
        for function, arguments in (
            ("many_paths_gradient", (1.0,)),
            ("residual_first_gradient", (1.0,)),
            ("many_tensor_paths_gradient", ([1.0, -2.0],)),
        ):
            reference = run_source(source, function, *arguments)
            raw = reference.value if isinstance(reference, Quantity) else reference
            self.assertEqual(raw, 1.0 if isinstance(raw, float) else [1.0, 1.0])

    def test_compiled_f32_autodiff_gradients_at_numeric_boundaries(self) -> None:
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
        generated = emit_c99(source)
        harness = r"""
#include <stdio.h>
int main(void) {
    aether_tensor_f32_2 subnormal_input = {{1.0f, -1.0f}};
    aether_tensor_f32_2 overflow_input = {{0.0f, 0.0f}};
    aether_tensor_f32_2 subnormal =
        aether_fn_74656e736f725f7375626e6f726d616c5f6772616469656e74(
            subnormal_input);
    aether_tensor_f32_2 overflow =
        aether_fn_74656e736f725f6f766572666c6f775f6772616469656e74(
            overflow_input);
    printf("%.17g\n", (double)aether_fn_707265636973696f6e5f6772616469656e74(16777216.0f));
    printf("%.17g\n", (double)aether_fn_7368617265645f6772616469656e74(16777216.0f));
    printf("%.17g\n", (double)aether_fn_7375626e6f726d616c5f6772616469656e74(1.0f));
    printf("%.17g\n", (double)aether_fn_6f766572666c6f775f6772616469656e74(0.0f));
    for (size_t index = 0; index < 2; ++index)
        printf("%.17g\n", (double)subnormal.data[index]);
    for (size_t index = 0; index < 2; ++index)
        printf("%.17g\n", (double)overflow.data[index]);
    return 0;
}
"""
        executable = self.compile_program(generated, harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = [float(value) for value in result.stdout.splitlines()]

        inputs = {
            "precision_gradient": (16777216.0,),
            "shared_gradient": (16777216.0,),
            "subnormal_gradient": (1.0,),
            "overflow_gradient": (0.0,),
            "tensor_subnormal_gradient": ([1.0, -1.0],),
            "tensor_overflow_gradient": ([0.0, 0.0],),
        }
        expected: list[float] = []
        for name, arguments in inputs.items():
            reference = run_source(source, name, *arguments)
            raw = reference.value if isinstance(reference, Quantity) else reference
            if isinstance(raw, (list, tuple)):
                expected.extend(float(item) for item in raw)
            else:
                expected.append(float(raw))

        self.assertEqual(len(actual), len(expected))
        for index, (native, reference) in enumerate(zip(actual, expected)):
            with self.subTest(result=index):
                self.assertEqual(native, reference)
        self.assertEqual(expected[0], 67108864.0)
        self.assertEqual(expected[1], struct.unpack("=f", struct.pack("=f", 0.1 * 2))[0])
        self.assertGreater(expected[2], 0.0)
        self.assertLess(expected[2], 1.1754943508222875e-38)
        self.assertTrue(math.isinf(expected[3]) and expected[3] > 0.0)
        self.assertTrue(all(0.0 < value < 1.1754943508222875e-38 for value in expected[4:6]))
        self.assertTrue(all(math.isinf(value) and value > 0.0 for value in expected[6:]))

    def test_compiled_f32_autodiff_rounds_accumulated_gradients_ties_to_even(self) -> None:
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
        generated = emit_c99(source)
        harness = r"""
#include <stdio.h>
int main(void) {
    aether_tensor_f32_2 input = {{1.0f, -2.0f}};
    aether_tensor_f32_2 tensor_tie =
        aether_fn_74656e736f725f7469655f6772616469656e74(input);
    aether_tensor_f32_2 tensor_above =
        aether_fn_74656e736f725f61626f76655f6772616469656e74(input);
    printf("%.17g\n", (double)aether_fn_7469655f6772616469656e74(1.0f));
    printf("%.17g\n", (double)aether_fn_61626f76655f6772616469656e74(1.0f));
    printf("%.17g\n", (double)aether_fn_7469655f6f64645f6772616469656e74(1.0f));
    printf("%.17g\n", (double)aether_fn_7469655f6e656761746976655f6772616469656e74(1.0f));
    for (size_t index = 0; index < 2; ++index)
        printf("%.17g\n", (double)tensor_tie.data[index]);
    for (size_t index = 0; index < 2; ++index)
        printf("%.17g\n", (double)tensor_above.data[index]);
    aether_tensor_f32_2 tensor_tie_odd =
        aether_fn_74656e736f725f7469655f6f64645f6772616469656e74(input);
    aether_tensor_f32_2 tensor_tie_negative =
        aether_fn_74656e736f725f7469655f6e656761746976655f6772616469656e74(input);
    for (size_t index = 0; index < 2; ++index)
        printf("%.17g\n", (double)tensor_tie_odd.data[index]);
    for (size_t index = 0; index < 2; ++index)
        printf("%.17g\n", (double)tensor_tie_negative.data[index]);
    return 0;
}
"""
        executable = self.compile_program(generated, harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = [float(value) for value in result.stdout.splitlines()]

        next_f32 = struct.unpack("=f", struct.pack("=f", 1.0 + 2.0**-23))[0]
        upper_even_f32 = struct.unpack("=f", struct.pack("=f", 1.0 + 2.0**-22))[0]
        expected = [
            1.0,
            next_f32,
            upper_even_f32,
            -upper_even_f32,
            1.0,
            1.0,
            next_f32,
            next_f32,
            upper_even_f32,
            upper_even_f32,
            -upper_even_f32,
            -upper_even_f32,
        ]
        self.assertEqual(actual, expected)
        for name, arguments, value in (
            ("tie_gradient", (1.0,), 1.0),
            ("above_gradient", (1.0,), next_f32),
            ("tie_odd_gradient", (1.0,), upper_even_f32),
            ("tie_negative_gradient", (1.0,), -upper_even_f32),
            ("tensor_tie_gradient", ([1.0, -2.0],), [1.0, 1.0]),
            (
                "tensor_above_gradient",
                ([1.0, -2.0],),
                [next_f32, next_f32],
            ),
            (
                "tensor_tie_odd_gradient",
                ([1.0, -2.0],),
                [upper_even_f32, upper_even_f32],
            ),
            (
                "tensor_tie_negative_gradient",
                ([1.0, -2.0],),
                [-upper_even_f32, -upper_even_f32],
            ),
        ):
            with self.subTest(function=name):
                reference = run_source(source, name, *arguments)
                raw = reference.value if isinstance(reference, Quantity) else reference
                self.assertEqual(raw, value)

    def test_compiled_f32_autodiff_rounds_gradients_at_subnormal_ties_to_even(self) -> None:
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
        harness = r"""
#include <stdio.h>
int main(void) {
    aether_tensor_f32_2 values = {{1.0f, -2.0f}};
    aether_tensor_f32_2 tensor_half =
        aether_fn_74656e736f725f68616c665f7375626e6f726d616c5f6772616469656e74(values);
    aether_tensor_f32_2 tensor_negative_half =
        aether_fn_74656e736f725f6e656761746976655f68616c665f7375626e6f726d616c5f6772616469656e74(values);
    aether_tensor_f32_2 tensor_odd_tie =
        aether_fn_74656e736f725f6f64645f7375626e6f726d616c5f7469655f6772616469656e74(values);
    aether_tensor_f32_2 tensor_negative_odd_tie =
        aether_fn_74656e736f725f6e656761746976655f6f64645f7375626e6f726d616c5f7469655f6772616469656e74(values);
    printf("%.17g\n", (double)aether_fn_68616c665f7375626e6f726d616c5f6772616469656e74(1.0f));
    printf("%.17g\n", (double)aether_fn_6e656761746976655f68616c665f7375626e6f726d616c5f6772616469656e74(1.0f));
    printf("%.17g\n", (double)aether_fn_6f64645f7375626e6f726d616c5f7469655f6772616469656e74(1.0f));
    printf("%.17g\n", (double)aether_fn_6e656761746976655f6f64645f7375626e6f726d616c5f7469655f6772616469656e74(1.0f));
    for (size_t index = 0; index < 2; ++index)
        printf("%.17g\n", (double)tensor_half.data[index]);
    for (size_t index = 0; index < 2; ++index)
        printf("%.17g\n", (double)tensor_negative_half.data[index]);
    for (size_t index = 0; index < 2; ++index)
        printf("%.17g\n", (double)tensor_odd_tie.data[index]);
    for (size_t index = 0; index < 2; ++index)
        printf("%.17g\n", (double)tensor_negative_odd_tie.data[index]);
    return 0;
}
"""
        executable = self.compile_program(emit_c99(source), harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = [float(value) for value in result.stdout.splitlines()]

        min_subnormal = struct.unpack("=f", struct.pack("=I", 1))[0]
        second_subnormal = struct.unpack("=f", struct.pack("=I", 2))[0]
        expected = [
            0.0,
            -0.0,
            second_subnormal,
            -second_subnormal,
            0.0,
            0.0,
            -0.0,
            -0.0,
            second_subnormal,
            second_subnormal,
            -second_subnormal,
            -second_subnormal,
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(math.copysign(1.0, actual[1]), -1.0)
        self.assertTrue(all(math.copysign(1.0, value) < 0.0 for value in actual[6:8]))
        for name, arguments, value in (
            ("half_subnormal_gradient", (1.0,), 0.0),
            ("negative_half_subnormal_gradient", (1.0,), -0.0),
            ("odd_subnormal_tie_gradient", (1.0,), second_subnormal),
            (
                "negative_odd_subnormal_tie_gradient",
                (1.0,),
                -second_subnormal,
            ),
            ("tensor_half_subnormal_gradient", ([1.0, -2.0],), [0.0, 0.0]),
            (
                "tensor_negative_half_subnormal_gradient",
                ([1.0, -2.0],),
                [-0.0, -0.0],
            ),
            (
                "tensor_odd_subnormal_tie_gradient",
                ([1.0, -2.0],),
                [second_subnormal, second_subnormal],
            ),
            (
                "tensor_negative_odd_subnormal_tie_gradient",
                ([1.0, -2.0],),
                [-second_subnormal, -second_subnormal],
            ),
        ):
            with self.subTest(function=name):
                reference = run_source(source, name, *arguments)
                raw = reference.value if isinstance(reference, Quantity) else reference
                self.assertEqual(raw, value)
        self.assertGreater(min_subnormal, 0.0)

    def test_compiled_f32_autodiff_rounds_gradients_around_two_and_half_subnormals(self) -> None:
        scalar_coefficients = {
            "below": "2.4999997615814209",
            "tie": "2.5",
            "above": "2.5000002384185791",
        }
        tensor_coefficients = {
            "below": "4.999999523162842",
            "tie": "5.0",
            "above": "5.000000476837158",
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
        source = "\n".join(declarations)

        scalar_names = [
            f"{sign}{prefix}_gradient"
            for prefix in scalar_coefficients
            for sign in ("", "negative_")
        ]
        tensor_names = [
            f"tensor_{sign}{prefix}_gradient"
            for prefix in tensor_coefficients
            for sign in ("", "negative_")
        ]
        scalar_output = "\n".join(
            f'    printf("%.17g\\n", (double)aether_fn_{name.encode("utf-8").hex()}(1.0f));'
            for name in scalar_names
        )
        tensor_initializers = ",\n        ".join(
            f"aether_fn_{name.encode('utf-8').hex()}(values)"
            for name in tensor_names
        )
        harness = f"""
#include <stdio.h>
int main(void) {{
    aether_tensor_f32_2 values = {{{{1.0f, -2.0f}}}};
{scalar_output}
    aether_tensor_f32_2 gradients[] = {{
        {tensor_initializers}
    }};
    for (size_t gradient = 0; gradient < 6; ++gradient)
        for (size_t index = 0; index < 2; ++index)
            printf("%.17g\\n", (double)gradients[gradient].data[index]);
    return 0;
}}
"""
        executable = self.compile_program(emit_c99(source), harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = [float(value) for value in result.stdout.splitlines()]

        even_result = struct.unpack("=f", struct.pack("=I", 2))[0]
        odd_result = struct.unpack("=f", struct.pack("=I", 3))[0]
        scalar_expected = [
            even_result,
            -even_result,
            even_result,
            -even_result,
            odd_result,
            -odd_result,
        ]
        tensor_expected = [
            value
            for gradient in (
                [even_result, even_result],
                [-even_result, -even_result],
                [even_result, even_result],
                [-even_result, -even_result],
                [odd_result, odd_result],
                [-odd_result, -odd_result],
            )
            for value in gradient
        ]
        expected = scalar_expected + tensor_expected
        self.assertEqual(actual, expected)
        self.assertTrue(all(value < 0.0 for value in actual[1:2] + actual[3:4] + actual[5:6]))
        for name in scalar_names:
            with self.subTest(function=name):
                reference = run_source(source, name, 1.0)
                raw = reference.value if isinstance(reference, Quantity) else reference
                self.assertEqual(raw, scalar_expected[scalar_names.index(name)])
        for index, name in enumerate(tensor_names):
            with self.subTest(function=name):
                reference = run_source(source, name, [1.0, -2.0])
                raw = reference.value if isinstance(reference, Quantity) else reference
                self.assertEqual(raw, tensor_expected[index * 2 : index * 2 + 2])

    def test_compiled_f32_autodiff_rounds_gradients_around_half_subnormal(self) -> None:
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
        source = "\n".join(declarations)

        scalar_names = [
            f"{sign}{prefix}_gradient"
            for prefix in scalar_coefficients
            for sign in ("", "negative_")
        ]
        tensor_names = [
            f"tensor_{sign}{prefix}_gradient"
            for prefix in tensor_coefficients
            for sign in ("", "negative_")
        ]
        scalar_output = "\n".join(
            f'    printf("%.17g\\n", (double)aether_fn_{name.encode("utf-8").hex()}(1.0f));'
            for name in scalar_names
        )
        tensor_initializers = ",\n        ".join(
            f"aether_fn_{name.encode('utf-8').hex()}(values)"
            for name in tensor_names
        )
        harness = f"""
#include <stdio.h>
int main(void) {{
    aether_tensor_f32_2 values = {{{{1.0f, -2.0f}}}};
{scalar_output}
    aether_tensor_f32_2 gradients[] = {{
        {tensor_initializers}
    }};
    for (size_t gradient = 0; gradient < 6; ++gradient)
        for (size_t index = 0; index < 2; ++index)
            printf("%.17g\\n", (double)gradients[gradient].data[index]);
    return 0;
}}
"""
        executable = self.compile_program(emit_c99(source), harness)
        result = self.run_program(executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = [float(value) for value in result.stdout.splitlines()]

        min_subnormal = struct.unpack("=f", struct.pack("=I", 1))[0]
        scalar_expected = [
            0.0,
            -0.0,
            0.0,
            -0.0,
            min_subnormal,
            -min_subnormal,
        ]
        tensor_expected = [
            value
            for gradient in (
                [0.0, 0.0],
                [-0.0, -0.0],
                [0.0, 0.0],
                [-0.0, -0.0],
                [min_subnormal, min_subnormal],
                [-min_subnormal, -min_subnormal],
            )
            for value in gradient
        ]
        self.assertEqual(actual, scalar_expected + tensor_expected)
        for index in (1, 3, 5, 8, 9, 12, 13, 16, 17):
            self.assertEqual(math.copysign(1.0, actual[index]), -1.0)
        for name, expected in zip(scalar_names, scalar_expected):
            with self.subTest(function=name):
                reference = run_source(source, name, 1.0)
                raw = reference.value if isinstance(reference, Quantity) else reference
                self.assertEqual(raw, expected)
        for index, name in enumerate(tensor_names):
            expected = tensor_expected[index * 2 : index * 2 + 2]
            with self.subTest(function=name):
                reference = run_source(source, name, [1.0, -2.0])
                raw = reference.value if isinstance(reference, Quantity) else reference
                self.assertEqual(raw, expected)

    def test_compiled_tensor_gradient_traps_division_by_zero(self) -> None:
        source = (
            "fn objective(values: Tensor<f32, [2]>) -> f32 {\n"
            "  return sum(values / mean(values));\n"
            "}\n"
            "fn derivative(values: Tensor<f32, [2]>) -> Tensor<f32, [2]> {\n"
            "  return gradient(objective, values);\n"
            "}\n"
        )
        harness = r"""
int main(void) {
    aether_tensor_f32_2 values = {{0.0f, 0.0f}};
    (void)aether_fn_64657269766174697665(values);
    return 0;
}
"""
        executable = self.compile_program(emit_c99(source), harness)
        result = self.run_program(executable)
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_compiled_reverse_mode_gradient_traps_division_by_zero(self) -> None:
        source = (
            "fn ratio(x: f32) -> f32 { return x / x; }\n"
            "fn ratio_derivative(x: f32) -> f32 "
            "{ return gradient(ratio, x); }\n"
        )
        harness = r"""
int main(void) {
    (void)aether_fn_726174696f5f64657269766174697665(0.0f);
    return 0;
}
"""
        executable = self.compile_program(emit_c99(source), harness)
        result = self.run_program(executable)
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_compiled_program_matches_interpreter_for_scalar_features(self) -> None:
        source = (
            "fn add(left: i32, right: i32) -> i32 { return left + right; }\n"
            "fn accept(value: i32, offset: i32) -> i32 { return value + offset; }\n"
            "fn pipeline(value: i32, offset: i32) -> i32 { "
            "return value |> accept(offset); }\n"
            "fn danger(value: i32) -> bool { return value / 0 > 0; }\n"
            "fn short_and() -> bool { return false && danger(1); }\n"
            "fn short_or() -> bool { return true || danger(1); }\n"
            "fn half() -> f32 { return 0.5; }\n"
            "fn min_value() -> i32 { return -2147483648; }\n"
            "fn unused_parameter(value: i32) -> i32 { return 7; }\n"
            "fn noop() -> void { return; }\n"
            "fn call_noop() -> void { noop(); return; }\n"
        )
        generated = emit_c99(source)
        harness = r"""
#include <stdio.h>
int main(void) {
    printf("%d\n", (int)aether_fn_616464(9, -4));
    printf("%d\n", (int)aether_fn_706970656c696e65(8, 6));
    printf("%d\n", (int)aether_fn_73686f72745f616e64());
    printf("%d\n", (int)aether_fn_73686f72745f6f72());
    printf("%.9g\n", (double)aether_fn_68616c66());
    printf("%d\n", (int)aether_fn_6d696e5f76616c7565());
    printf("%d\n", (int)aether_fn_756e757365645f706172616d65746572(3));
    aether_fn_63616c6c5f6e6f6f70();
    puts("void");
    return 0;
}
"""
        result = self.compile_and_run(generated, harness)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = result.stdout.splitlines()
        expected = [
            str(run_source(source, "add", 9, -4)),
            str(run_source(source, "pipeline", 8, 6)),
            str(int(run_source(source, "short_and"))),
            str(int(run_source(source, "short_or"))),
            format(run_source(source, "half"), ".9g"),
            str(run_source(source, "min_value")),
            str(run_source(source, "unused_parameter", 3)),
            "void",
        ]
        self.assertEqual(actual, expected)

    def test_compiled_integer_boundaries_match_interpreter(self) -> None:
        source = (
            "fn add_i64(left: i64, right: i64) -> i64 { return left + right; }\n"
            "fn sub_i64(left: i64, right: i64) -> i64 { return left - right; }\n"
            "fn mul_i64(left: i64, right: i64) -> i64 { return left * right; }\n"
            "fn add_u64(left: u64, right: u64) -> u64 { return left + right; }\n"
            "fn mul_u64(left: u64, right: u64) -> u64 { return left * right; }\n"
            "fn div_i32(left: i32, right: i32) -> i32 { return left / right; }\n"
            "fn rem_i32(left: i32, right: i32) -> i32 { return left % right; }\n"
        )
        harness = r"""
#include <stdio.h>
int main(void) {
    printf("%lld\n", (long long)aether_fn_6164645f693634(INT64_MIN, 0));
    printf("%lld\n", (long long)aether_fn_6164645f693634(INT64_MAX, 0));
    printf("%lld\n", (long long)aether_fn_7375625f693634(INT64_MIN, 0));
    printf("%lld\n", (long long)aether_fn_6d756c5f693634(INT64_MIN, 1));
    printf("%lld\n", (long long)aether_fn_6d756c5f693634(0, INT64_MIN));
    printf("%llu\n", (unsigned long long)aether_fn_6164645f753634(UINT64_MAX, 0));
    printf("%llu\n", (unsigned long long)aether_fn_6d756c5f753634(UINT64_MAX, 1));
    printf("%d\n", (int)aether_fn_6469765f693332(-7, 3));
    printf("%d\n", (int)aether_fn_72656d5f693332(-7, 3));
    return 0;
}
"""
        result = self.compile_and_run(emit_c99(source), harness)
        self.assertEqual(result.returncode, 0, result.stderr)
        test_cases = (
            ("add_i64", (-(1 << 63), 0)),
            ("add_i64", ((1 << 63) - 1, 0)),
            ("sub_i64", (-(1 << 63), 0)),
            ("mul_i64", (-(1 << 63), 1)),
            ("mul_i64", (0, -(1 << 63))),
            ("add_u64", ((1 << 64) - 1, 0)),
            ("mul_u64", ((1 << 64) - 1, 1)),
            ("div_i32", (-7, 3)),
            ("rem_i32", (-7, 3)),
        )
        expected = [
            str(run_source(source, name, *arguments))
            for name, arguments in test_cases
        ]
        self.assertEqual(result.stdout.splitlines(), expected)

    def test_compiled_arithmetic_traps_invalid_operations(self) -> None:
        source = (
            "fn add_i32(left: i32, right: i32) -> i32 { return left + right; }\n"
            "fn add_u8(left: u8, right: u8) -> u8 { return left + right; }\n"
            "fn mul_i64(left: i64, right: i64) -> i64 { return left * right; }\n"
            "fn div_i32(left: i32, right: i32) -> i32 { return left / right; }\n"
            "fn rem_i32(left: i32, right: i32) -> i32 { return left % right; }\n"
            "fn neg_i64(value: i64) -> i64 { return -value; }\n"
            "fn div_f32(left: f32, right: f32) -> f32 { return left / right; }\n"
        )
        harness = r"""
#include <string.h>
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    if (strcmp(argv[1], "signed_add") == 0)
        (void)aether_fn_6164645f693332(INT32_MAX, 1);
    else if (strcmp(argv[1], "unsigned_add") == 0)
        (void)aether_fn_6164645f7538((uint8_t)255, (uint8_t)1);
    else if (strcmp(argv[1], "signed_multiply") == 0)
        (void)aether_fn_6d756c5f693634(INT64_MAX, 2);
    else if (strcmp(argv[1], "signed_divide_zero") == 0)
        (void)aether_fn_6469765f693332(1, 0);
    else if (strcmp(argv[1], "signed_divide_min") == 0)
        (void)aether_fn_6469765f693332(INT32_MIN, -1);
    else if (strcmp(argv[1], "signed_remainder_zero") == 0)
        (void)aether_fn_72656d5f693332(1, 0);
    else if (strcmp(argv[1], "signed_remainder_min") == 0)
        (void)aether_fn_72656d5f693332(INT32_MIN, -1);
    else if (strcmp(argv[1], "signed_negate_min") == 0)
        (void)aether_fn_6e65675f693634(INT64_MIN);
    else if (strcmp(argv[1], "float_divide_zero") == 0)
        (void)aether_fn_6469765f663332(1.0f, 0.0f);
    else return 3;
    return 0;
}
"""
        generated = emit_c99(source)
        executable = self.compile_program(generated, harness)
        for case in (
            "signed_add",
            "unsigned_add",
            "signed_multiply",
            "signed_divide_zero",
            "signed_divide_min",
            "signed_remainder_zero",
            "signed_remainder_min",
            "signed_negate_min",
            "float_divide_zero",
        ):
            with self.subTest(operation=case):
                result = self.run_program(executable, case)
                self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_rejects_features_outside_backend_subset_with_diagnostics(self) -> None:
        cases = (
            (
                "fn power(value: i32, exponent: i32) -> i32 "
                "{ return value ** exponent; }\n",
                "E_C99_OPERATOR",
            ),
            (
                "fn text() -> str { return \"hello\"; }\n",
                "E_C99_TYPE",
            ),
            (
                "fn tensor(value: Tensor<i32, [2]>) -> Tensor<i32, [2]> "
                "{ return value; }\n",
                "E_C99_TYPE",
            ),
            (
                '@ffi\n@link("sensor")\nextern "C" fn read_tensor('
                "value: Tensor<f32, [2]>) -> void;\n",
                "E_FFI_TYPE",
            ),
            (
                "fn result(value: Result<str, IoError>) -> i32 { return 1; }\n",
                "E_C99_TYPE",
            ),
            (
                "@io fn read() -> Result<str, IoError> { return io_read_line(); }\n",
                "E_C99_EFFECT",
            ),
            (
                '@ffi\n@link("sensor")\nextern "C" fn café() -> i32;\n',
                "E_C99_FFI_SYMBOL",
            ),
            (
                "fn missing_return(value: i32) -> i32 { let copy = value; }\n",
                "E_C99_RETURN",
            ),
            (
                "type Pair { left: i32, right: i32 }\n"
                "fn value() -> i32 { return 1; }\n",
                "E_C99_TYPE_DECL",
            ),
        )
        for source, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(CompilationError) as raised:
                    emit_c99(source)
                self.assertIn(
                    expected_code,
                    {diagnostic.code for diagnostic in raised.exception.diagnostics},
                )


if __name__ == "__main__":
    unittest.main()
