from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

from src.api import (
    CompilationError,
    build_project_executable,
    build_project_object,
    compile_project,
    compile_source,
    emit_c99_project,
    run_project,
)
from src.build import C99LinkConfigurationError, compile_c99_object
from src.frontend import parse_source, render_module
from src.interpreter import ResultValue


class ProjectTests(unittest.TestCase):
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

        for name in ("cc", "gcc", "clang"):
            path = shutil.which(name)
            if path is not None:
                return (path,)
        zig = shutil.which("zig")
        if zig is not None:
            return (zig, "cc")
        message = "No C99 compiler is installed"
        if os.environ.get("AETHER_REQUIRE_C99_COMPILER") == "1":
            self.fail(f"{message}; this test environment requires one")
        self.skipTest(message)

    def test_import_syntax_roundtrips_and_requires_project_api(self) -> None:
        source = (
            'import "lib\\\\sensor.hl" as sensor;\n'
            "fn read(point: sensor::Point) -> i32 {\n"
            "    return sensor::read(point);\n"
            "}\n"
        )
        parsed = parse_source(source, validate=False)

        self.assertEqual(parsed.ast.imports[0].path, "lib\\sensor.hl")
        self.assertEqual(parsed.ast.imports[0].alias, "sensor")
        self.assertIn("import_declaration", {node.kind for node in parsed.cst.children})
        self.assertEqual(render_module(parsed.ast), source)
        with self.assertRaises(CompilationError) as raised:
            compile_source(source)
        self.assertIn(
            "E_IMPORT_PROJECT_ONLY",
            {diagnostic.code for diagnostic in raised.exception.diagnostics},
        )

    def test_project_calls_imported_functions_with_qualified_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "math.hl").write_text(
                "fn double(value: i32) -> i32 { return value * 2; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "math.hl" as math;\n'
                "fn main(value: i32) -> i32 {\n"
                "    return (value |> math::double) + math::double(1);\n"
                "}\n",
                encoding="utf-8",
            )

            result = run_project(entry, "main", 20)
            module = compile_project(entry)

        self.assertEqual(result, 42)
        self.assertEqual(len(module.functions), 2)

    def test_project_emits_and_builds_c99_across_modules(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "left.hl").write_text(
                "fn double(value: i32) -> i32 { return value * 2; }\n",
                encoding="utf-8",
            )
            (project / "right.hl").write_text(
                "fn double(value: i32) -> i32 { return value * 2; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "left.hl" as left;\n'
                'import "right.hl" as right;\n'
                "fn main() -> i32 {\n"
                "    return (20 |> left::double) + right::double(1);\n"
                "}\n",
                encoding="utf-8",
            )

            reference_result = run_project(entry, "main")
            generated = emit_c99_project(entry)
            self.assertEqual(generated, emit_c99_project(entry))
            object_file = build_project_object(
                entry,
                project / "program.o",
                compiler=compiler,
            )
            self.assertTrue(object_file.is_file())
            executable = build_project_executable(
                entry,
                project / "program.exe",
                compiler=compiler,
            )
            result = subprocess.run(
                [str(executable)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        self.assertIn("aether_fn_", generated)
        self.assertEqual(result.returncode, reference_result, result.stderr)

    def test_project_c99_rejects_features_outside_backend_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "types.hl").write_text(
                "type Point { x: i32, }\n"
                "fn identity(point: Point) -> Point { return point; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "types.hl" as types;\n'
                "fn main() -> i32 { return 0; }\n",
                encoding="utf-8",
            )

            with self.assertRaises(CompilationError) as raised:
                emit_c99_project(entry)

        diagnostic = next(
            item
            for item in raised.exception.diagnostics
            if item.code == "E_C99_TYPE_DECL"
        )
        self.assertEqual(
            diagnostic.span.source_path,
            str((project / "types.hl").resolve()),
        )

    def test_project_executable_links_entry_module_c_abi_library(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "math.hl").write_text(
                "fn double(value: i32) -> i32 { return value * 2; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                '@ffi @link("native") extern "C" fn native_value() -> i32;\n'
                'import "math.hl" as math;\n'
                "@ffi fn main() -> i32 {\n"
                "    return math::double(native_value());\n"
                "}\n",
                encoding="utf-8",
            )
            library_object = project / "native.o"
            compile_c99_object(
                "int native_value(void) { return 21; }\n",
                library_object,
                compiler=compiler,
            )
            with self.assertRaises(C99LinkConfigurationError):
                build_project_executable(
                    entry,
                    project / "missing-library.exe",
                    compiler=compiler,
                )
            executable = build_project_executable(
                entry,
                project / "program.exe",
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
            reference_result = run_project(
                entry,
                "main",
                external_handlers={"native_value": lambda: 21},
            )

        self.assertEqual(result.returncode, reference_result, result.stderr)

    def test_project_resolves_types_and_calls_across_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "geometry.hl").write_text(
                "type Point { x: i32, }\n"
                "fn get_x(point: Point) -> i32 { return point.x; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "geometry.hl" as geometry;\n'
                "fn main(point: geometry::Point) -> i32 {\n"
                "    return geometry::get_x(point);\n"
                "}\n",
                encoding="utf-8",
            )

            result = run_project(entry, "main", {"x": 17})

        self.assertEqual(result, 17)

    def test_project_propagates_effects_across_imported_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "input.hl").write_text(
                "@io fn read() -> Result<str, IoError> {\n"
                "    return io_read_line();\n"
                "}\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "input.hl" as input;\n'
                "@io fn main() -> Result<str, IoError> {\n"
                "    return input::read();\n"
                "}\n",
                encoding="utf-8",
            )

            result = run_project(
                entry,
                "main",
                effect_handlers={
                    "io_read_line": lambda: ResultValue.ok("ready"),
                },
            )

        self.assertEqual(result, ResultValue.ok("ready"))

    def test_project_keeps_same_named_functions_in_distinct_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "left.hl").write_text(
                "fn value() -> i32 { return 12; }\n",
                encoding="utf-8",
            )
            (project / "right.hl").write_text(
                "fn value() -> i32 { return 30; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "left.hl" as left;\n'
                'import "right.hl" as right;\n'
                "fn main() -> i32 { return left::value() + right::value(); }\n",
                encoding="utf-8",
            )

            result = run_project(entry, "main")

        self.assertEqual(result, 42)

    def test_project_resolves_relative_imports_from_importing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "nested").mkdir()
            (project / "leaf.hl").write_text(
                "fn value() -> i32 { return 42; }\n",
                encoding="utf-8",
            )
            (project / "nested" / "middle.hl").write_text(
                'import "../leaf.hl" as leaf;\n'
                "fn value() -> i32 { return leaf::value(); }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "nested/middle.hl" as middle;\n'
                "fn main() -> i32 { return middle::value(); }\n",
                encoding="utf-8",
            )

            result = run_project(entry, "main")

        self.assertEqual(result, 42)

    def test_project_allows_multiple_aliases_for_the_same_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "lib.hl").write_text(
                "fn value() -> i32 { return 20; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "lib.hl" as first;\n'
                'import "./lib.hl" as second;\n'
                "fn main() -> i32 {\n"
                "    return first::value() + second::value();\n"
                "}\n",
                encoding="utf-8",
            )

            result = run_project(entry, "main")
            module = compile_project(entry)

        self.assertEqual(result, 40)
        self.assertEqual(
            len([function for function in module.functions if function.name.endswith("value")]),
            1,
        )

    def test_project_allows_alias_to_match_a_local_function_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "lib.hl").write_text(
                "fn value() -> i32 { return 20; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "lib.hl" as helper;\n'
                "fn helper() -> i32 { return 2; }\n"
                "fn main() -> i32 { return helper() + helper::value(); }\n",
                encoding="utf-8",
            )

            result = run_project(entry, "main")

        self.assertEqual(result, 22)

    def test_project_rejects_missing_import_and_reports_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            entry = Path(temporary_directory) / "main.hl"
            entry.write_text(
                'import "missing.hl" as missing;\n'
                "fn main() -> i32 { return 0; }\n",
                encoding="utf-8",
            )
            with self.assertRaises(CompilationError) as raised:
                compile_project(entry)

        diagnostic = raised.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, "E_PROJECT_IO")
        self.assertEqual(diagnostic.span.source_path, str(entry.resolve()))
        self.assertIn("main.hl:", str(raised.exception))

    def test_project_rejects_import_paths_outside_entry_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            project = project_root / "project"
            project.mkdir()
            (project_root / "outside.hl").write_text(
                "fn value() -> i32 { return 42; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "../outside.hl" as outside;\n'
                "fn main() -> i32 { return outside::value(); }\n",
                encoding="utf-8",
            )

            with self.assertRaises(CompilationError) as raised:
                compile_project(entry)

        self.assertIn(
            "E_IMPORT_PATH",
            {diagnostic.code for diagnostic in raised.exception.diagnostics},
        )

    def test_project_rejects_absolute_non_hl_and_control_character_paths(self) -> None:
        imports = (
            'import "C:\\\\outside.hl" as absolute;\n',
            'import "not-source.txt" as wrong_extension;\n',
            'import "bad\\nname.hl" as control;\n',
        )
        for import_declaration in imports:
            with self.subTest(import_declaration=import_declaration):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    entry = Path(temporary_directory) / "main.hl"
                    entry.write_text(
                        import_declaration + "fn main() -> i32 { return 0; }\n",
                        encoding="utf-8",
                    )

                    with self.assertRaises(CompilationError) as raised:
                        compile_project(entry)

                self.assertIn(
                    "E_IMPORT_PATH",
                    {
                        diagnostic.code
                        for diagnostic in raised.exception.diagnostics
                    },
                )

    def test_project_semantic_diagnostics_keep_originating_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            library = project / "lib.hl"
            library.write_text(
                "fn value() -> i32 { return true + 1; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "lib.hl" as lib;\n'
                "fn main() -> i32 { return lib::value(); }\n",
                encoding="utf-8",
            )

            with self.assertRaises(CompilationError) as raised:
                compile_project(entry)

        diagnostic = next(
            item
            for item in raised.exception.diagnostics
            if item.code == "E_TYPE_OPERATOR"
        )
        self.assertEqual(diagnostic.span.source_path, str(library.resolve()))

    def test_project_rejects_cycles_with_cycle_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            first = project / "first.hl"
            second = project / "second.hl"
            first.write_text(
                'import "second.hl" as second;\n'
                "fn first() -> i32 { return second::second(); }\n",
                encoding="utf-8",
            )
            second.write_text(
                'import "first.hl" as first;\n'
                "fn second() -> i32 { return first::first(); }\n",
                encoding="utf-8",
            )

            with self.assertRaises(CompilationError) as raised:
                compile_project(first)

        diagnostic = next(
            item
            for item in raised.exception.diagnostics
            if item.code == "E_IMPORT_CYCLE"
        )
        self.assertIn("first.hl -> second.hl -> first.hl", diagnostic.message)

    def test_project_rejects_unqualified_imported_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "lib.hl").write_text(
                "fn value() -> i32 { return 42; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "lib.hl" as lib;\n'
                "fn main() -> i32 { return value(); }\n",
                encoding="utf-8",
            )

            with self.assertRaises(CompilationError) as raised:
                compile_project(entry)

        self.assertIn(
            "E_UNKNOWN_CALL",
            {diagnostic.code for diagnostic in raised.exception.diagnostics},
        )

    def test_project_rejects_unknown_alias_and_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "lib.hl").write_text(
                "fn value() -> i32 { return 42; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "lib.hl" as lib;\n'
                "fn main() -> i32 { return other::value() + lib::missing(); }\n",
                encoding="utf-8",
            )

            with self.assertRaises(CompilationError) as raised:
                compile_project(entry)

        codes = {diagnostic.code for diagnostic in raised.exception.diagnostics}
        self.assertIn("E_UNKNOWN_IMPORT_ALIAS", codes)
        self.assertIn("E_IMPORT_UNKNOWN_FUNCTION", codes)

    def test_project_rejects_duplicate_import_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "a.hl").write_text(
                "fn a() -> i32 { return 1; }\n",
                encoding="utf-8",
            )
            (project / "b.hl").write_text(
                "fn b() -> i32 { return 2; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "a.hl" as shared;\n'
                'import "b.hl" as shared;\n'
                "fn main() -> i32 { return 0; }\n",
                encoding="utf-8",
            )

            with self.assertRaises(CompilationError) as raised:
                compile_project(entry)

        self.assertIn(
            "E_DUPLICATE_IMPORT_ALIAS",
            {diagnostic.code for diagnostic in raised.exception.diagnostics},
        )

    def test_project_rejects_imported_extern_declarations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "native.hl").write_text(
                '@ffi @link("native") extern "C" fn read() -> i32;\n',
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "native.hl" as native;\n'
                "fn main() -> i32 { return native::read(); }\n",
                encoding="utf-8",
            )

            with self.assertRaises(CompilationError) as raised:
                compile_project(entry)

        self.assertIn(
            "E_PROJECT_FFI_UNSUPPORTED",
            {diagnostic.code for diagnostic in raised.exception.diagnostics},
        )


if __name__ == "__main__":
    unittest.main()
