import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.api import (
    CompilationError,
    build_project_executable,
    build_project_object,
    build_project_objects,
    compile_project,
    compile_source,
    emit_c99_project,
    link_project_objects,
    run_project,
)
from src.build import (
    C99BuildError,
    C99LinkConfigurationError,
    C99ToolchainError,
    compile_c99_object,
)
from src.c99 import emit_c99_module_units
from src.frontend import parse_source, render_module
from src.interpreter import ResultValue


class ProjectTests(unittest.TestCase):
    def test_native_imported_module_uses_entry_runtime_arguments(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            entry = root / "main.hl"
            entry.write_text(
                'import "arguments.hl" as arguments;\n'
                "fn main() -> i32 { return arguments::has_argument(); }\n",
                encoding="utf-8",
            )
            (root / "arguments.hl").write_text(
                "fn has_argument() -> i32 {\n"
                "  return match args_get(args_len() - args_len()) "
                "{ Ok(_) => 0, Err(_) => 1 };\n"
                "}\n",
                encoding="utf-8",
            )
            executable = build_project_executable(
                entry,
                root / "program",
                compiler=compiler,
            )
            with_argument = subprocess.run(
                [str(executable), "héllo"],
                check=False,
                capture_output=True,
                timeout=10,
                cwd=root,
            )
            without_argument = subprocess.run(
                [str(executable)],
                check=False,
                capture_output=True,
                timeout=10,
                cwd=root,
            )
            self.assertEqual(with_argument.returncode, 0, with_argument.stderr)
            self.assertEqual(without_argument.returncode, 1, without_argument.stderr)

    def test_imported_calls_in_control_flow_are_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "values.hl").write_text(
                "fn positive(value: i32) -> bool { return value > 0; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "values.hl" as values;\n'
                "fn main(value: i32) -> i32 {\n"
                "  if values::positive(value) { return 1; } else { return 0; }\n"
                "}\n",
                encoding="utf-8",
            )

            result = run_project(entry, "main", 4)

        self.assertEqual(result, 1)

    def test_imported_var_rebinding_is_preserved_in_project_ir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "values.hl").write_text(
                "fn update() -> i32 {\n"
                "    var value = 1;\n"
                "    value = 42;\n"
                "    return value;\n"
                "}\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "values.hl" as values;\n'
                "fn main() -> i32 { return values::update(); }\n",
                encoding="utf-8",
            )

            result = run_project(entry, "main")
            module = compile_project(entry)

        self.assertEqual(result, 42)
        imported_update = next(
            function for function in module.functions if function.name.endswith("update")
        )
        self.assertTrue(any(statement.kind == "assignment" for statement in imported_update.body))

    def test_project_unit_emission_applies_tensor_fusion(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "helper.hl").write_text(
                "fn chain(values: Tensor<f32, [2]>, offset: f32) "
                "-> Tensor<f32, [2]> {\n"
                "    let scaled = values * 2.0;\n"
                "    let shifted = scaled + offset;\n"
                "    return shifted - values;\n"
                "}\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "helper.hl" as helper;\n'
                "fn main() -> i32 { return 0; }\n",
                encoding="utf-8",
            )

            module = compile_project(entry)
            optimized_units = emit_c99_module_units(module, str(entry.resolve()))
            baseline_units = emit_c99_module_units(
                module,
                str(entry.resolve()),
                optimize=False,
            )
            optimized_helper = next(
                unit for path, unit in optimized_units
                if path == "helper.hl"
            )
            baseline_helper = next(
                unit for path, unit in baseline_units
                if path == "helper.hl"
            )
            self.assertEqual(optimized_helper.count("for (size_t aether_i_"), 1)
            self.assertEqual(baseline_helper.count("for (size_t aether_i_"), 3)
            self.assertEqual(len(optimized_units), len(baseline_units))
            object_file = build_project_object(
                entry,
                project / "fusion.o",
                compiler=compiler,
            )
            self.assertTrue(object_file.is_file())

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

    def controllable_compiler(
        self,
        project: Path,
        delegate: tuple[str, ...],
    ) -> tuple[tuple[str, ...], Path]:
        configuration = project / "compiler.json"
        script = project / "compiler_wrapper.py"
        script.write_text(
            "import json, pathlib, subprocess, sys\n"
            f"delegate = {list(delegate)!r}\n"
            f"configuration = pathlib.Path({str(configuration)!r})\n"
            "settings = json.loads(configuration.read_text(encoding='utf-8'))\n"
            "arguments = sys.argv[1:]\n"
            "if arguments[-1:] == ['--version']:\n"
            "    print(settings['version'])\n"
            "    raise SystemExit(0)\n"
            "if arguments[-1:] == ['-dumpmachine']:\n"
            "    print(settings['target'])\n"
            "    raise SystemExit(0)\n"
            "result = subprocess.run([*delegate, *arguments], check=False)\n"
            "raise SystemExit(result.returncode)\n",
            encoding="utf-8",
        )
        return (sys.executable, str(script)), configuration

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

    def test_project_gradient_resolves_imported_target(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "math.hl").write_text(
                "fn square(value: f32) -> f32 { return value * value; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "math.hl" as math;\n'
                "fn derivative(value: f32) -> f32 {\n"
                "    return gradient(math::square, value);\n"
                "}\n"
                "fn main() -> i32 { return 0; }\n",
                encoding="utf-8",
            )

            interpreted = run_project(entry, "derivative", 3.0)
            executable = build_project_executable(
                entry,
                project / "program.exe",
                compiler=compiler,
            )
            native = subprocess.run(
                [str(executable)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

        self.assertAlmostEqual(interpreted, 6.0)
        self.assertEqual(native.returncode, 0, native.stderr)

    def test_project_emits_and_builds_c99_across_modules(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory) / "project with spaces"
            project.mkdir()
            (project / "helper.hl").write_text(
                "fn scale(value: i32, factor: i32) -> i32 {\n"
                "    return value * factor;\n"
                "}\n",
                encoding="utf-8",
            )
            (project / "left.hl").write_text(
                'import "helper.hl" as helper;\n'
                "fn double(value: i32) -> i32 {\n"
                "    return helper::scale(value, 2);\n"
                "}\n",
                encoding="utf-8",
            )
            (project / "right.hl").write_text(
                "fn double(value: i32) -> i32 { return value * 2; }\n",
                encoding="utf-8",
            )
            (project / "empty.hl").write_text("", encoding="utf-8")
            entry = project / "main.hl"
            entry.write_text(
                'import "left.hl" as left;\n'
                'import "right.hl" as right;\n'
                'import "empty.hl" as empty;\n'
                "fn main() -> i32 {\n"
                "    return (20 |> left::double) + right::double(1);\n"
                "}\n",
                encoding="utf-8",
            )

            reference_result = run_project(entry, "main")
            generated = emit_c99_project(entry)
            self.assertEqual(generated, emit_c99_project(entry))
            project_module = compile_project(entry)
            generated_units = emit_c99_module_units(
                project_module,
                str(entry.resolve()),
            )
            self.assertEqual(len(generated_units), 5)
            self.assertEqual(
                sum("int main(void)" in unit for _, unit in generated_units),
                1,
            )
            for function in project_module.functions:
                signature = (
                    "int32_t aether_fn_"
                    + function.name.encode("utf-8").hex()
                    + "("
                )
                self.assertEqual(
                    sum(
                        unit.count(signature) == 2
                        for _, unit in generated_units
                    ),
                    1,
                )
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

    def test_build_project_objects_returns_separately_linkable_module_objects(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory) / "source tree with spaces"
            project.mkdir()
            (project / "left").mkdir()
            (project / "right").mkdir()
            (project / "left" / "worker.hl").write_text(
                "fn value() -> i32 { return 12; }\n",
                encoding="utf-8",
            )
            (project / "right" / "worker.hl").write_text(
                "fn value() -> i32 { return 30; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "left/worker.hl" as left;\n'
                'import "right/worker.hl" as right;\n'
                "fn main() -> i32 { return left::value() + right::value(); }\n",
                encoding="utf-8",
            )
            reference_result = run_project(entry, "main")
            output_directory = Path(temporary_directory) / "objects with spaces"

            object_paths = build_project_objects(
                entry,
                output_directory,
                compiler=compiler,
            )
            manifest = json.loads(
                (output_directory / "aether-objects.json").read_text(
                    encoding="utf-8"
                )
            )
            repeated_object_paths = build_project_objects(
                entry,
                Path(temporary_directory) / "repeat objects",
                compiler=compiler,
            )
            self.assertEqual(
                set(object_paths),
                {Path("main.hl"), Path("left/worker.hl"), Path("right/worker.hl")},
            )
            self.assertEqual(len({path.name for path in object_paths.values()}), 3)
            self.assertEqual(
                {source: path.name for source, path in object_paths.items()},
                {
                    source: path.name
                    for source, path in repeated_object_paths.items()
                },
            )
            self.assertEqual(
                {
                    source: path.read_bytes()
                    for source, path in object_paths.items()
                },
                {
                    source: path.read_bytes()
                    for source, path in repeated_object_paths.items()
                },
            )
            self.assertTrue(all(path.is_file() for path in object_paths.values()))
            self.assertEqual(manifest["format"], "aether-object-set")
            self.assertEqual(manifest["version"], 2)
            self.assertEqual(
                set(manifest["toolchain"]),
                {
                    "executable",
                    "command_sha256",
                    "version",
                    "target",
                    "compile_flags",
                    "link_flags",
                },
            )
            self.assertEqual(
                manifest["toolchain"]["compile_flags"],
                [
                    "-std=c99",
                    "-pedantic",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-ffp-contract=off",
                    "-frandom-seed=aether",
                    "-c",
                ],
            )
            self.assertEqual(
                {record["source"] for record in manifest["modules"]},
                {"main.hl", "left/worker.hl", "right/worker.hl"},
            )
            self.assertTrue(
                all(len(record["object_sha256"]) == 64 for record in manifest["modules"])
            )

            executable = Path(temporary_directory) / "linked.exe"
            linked = link_project_objects(
                entry,
                object_paths,
                executable,
                compiler=compiler,
            )
            self.assertEqual(linked, executable.resolve())
            execution = subprocess.run(
                [str(executable)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

        self.assertEqual(execution.returncode, reference_result)

    def test_project_source_hash_and_objects_ignore_line_ending_style(self) -> None:
        compiler = self.compiler()
        source = (
            "fn helper(value: i32) -> i32 {\n"
            "    return value + 1;\n"
            "}\n"
            "fn main() -> i32 {\n"
            "    return helper(41);\n"
            "}\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            outputs: list[tuple[dict[Path, Path], dict[str, str]]] = []
            for name, line_ending in (("lf", "\n"), ("crlf", "\r\n")):
                project = root / name
                project.mkdir()
                entry = project / "main.hl"
                entry.write_bytes(source.replace("\n", line_ending).encode("utf-8"))
                output_directory = project / "objects"
                objects = build_project_objects(
                    entry,
                    output_directory,
                    compiler=compiler,
                )
                manifest = json.loads(
                    (output_directory / "aether-objects.json").read_text(
                        encoding="utf-8"
                    )
                )
                hashes = {
                    record["source"]: record["source_sha256"]
                    for record in manifest["modules"]
                }
                outputs.append((objects, hashes))

            lf_objects, lf_hashes = outputs[0]
            crlf_objects, crlf_hashes = outputs[1]
            self.assertEqual(lf_hashes, crlf_hashes)
            self.assertEqual(
                {
                    source: object_path.read_bytes()
                    for source, object_path in lf_objects.items()
                },
                {
                    source: object_path.read_bytes()
                    for source, object_path in crlf_objects.items()
                },
            )

    def test_link_project_objects_rejects_incomplete_or_foreign_object_maps(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "lib.hl").write_text(
                "fn value() -> i32 { return 12; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "lib.hl" as lib;\n'
                "fn main() -> i32 { return lib::value(); }\n",
                encoding="utf-8",
            )
            objects = build_project_objects(
                entry,
                project / "objects",
                compiler=compiler,
            )
            output = project / "invalid.exe"

            with self.assertRaises(C99LinkConfigurationError):
                link_project_objects(
                    entry,
                    {Path("main.hl"): objects[Path("main.hl")]},
                    output,
                    compiler=compiler,
                )
            with self.assertRaises(C99LinkConfigurationError):
                link_project_objects(
                    entry,
                    {
                        **objects,
                        Path("other.hl"): objects[Path("lib.hl")],
                    },
                    output,
                    compiler=compiler,
                )

            self.assertFalse(output.exists())

    def test_link_project_objects_uses_explicit_ffi_libraries(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            entry = project / "main.hl"
            entry.write_text(
                '@ffi @link("native") extern "C" fn native_value() -> i32;\n'
                "@ffi fn main() -> i32 { return native_value(); }\n",
                encoding="utf-8",
            )
            objects = build_project_objects(
                entry,
                project / "objects",
                compiler=compiler,
            )
            library = project / "native.o"
            compile_c99_object(
                "int native_value(void) { return 37; }\n",
                library,
                compiler=compiler,
            )
            output = project / "linked.exe"
            with self.assertRaises(C99LinkConfigurationError):
                link_project_objects(
                    entry,
                    objects,
                    output,
                    compiler=compiler,
                )
            linked = link_project_objects(
                entry,
                objects,
                output,
                compiler=compiler,
                libraries={"native": library},
            )
            execution = subprocess.run(
                [str(linked)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            reference = run_project(
                entry,
                "main",
                external_handlers={"native_value": lambda: 37},
            )

        self.assertEqual(execution.returncode, reference, execution.stderr)

    def test_link_project_objects_preserves_output_on_link_error(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            entry = project / "main.hl"
            entry.write_text(
                '@ffi @link("native") extern "C" fn missing_symbol() -> i32;\n'
                "@ffi fn main() -> i32 { return missing_symbol(); }\n",
                encoding="utf-8",
            )
            objects = build_project_objects(
                entry,
                project / "objects",
                compiler=compiler,
            )
            library = project / "native.o"
            compile_c99_object(
                "int unrelated_symbol(void) { return 7; }\n",
                library,
                compiler=compiler,
            )
            output = project / "existing.exe"
            output.write_bytes(b"previous executable")

            with self.assertRaises(C99BuildError):
                link_project_objects(
                    entry,
                    objects,
                    output,
                    compiler=compiler,
                    libraries={"native": library},
                )

            self.assertEqual(output.read_bytes(), b"previous executable")

    def test_link_project_objects_rejects_changed_source(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "lib.hl").write_text(
                "fn value() -> i32 { return 12; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "lib.hl" as lib;\n'
                "fn main() -> i32 { return lib::value(); }\n",
                encoding="utf-8",
            )
            objects = build_project_objects(
                entry,
                project / "objects",
                compiler=compiler,
            )
            (project / "lib.hl").write_text(
                "fn value() -> i32 { return 13; }\n",
                encoding="utf-8",
            )
            output = project / "stale.exe"

            with self.assertRaisesRegex(
                C99LinkConfigurationError,
                "has changed since its object was built",
            ):
                link_project_objects(
                    entry,
                    objects,
                    output,
                    compiler=compiler,
                )

            self.assertFalse(output.exists())

    def test_link_project_objects_rejects_modified_object(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            entry = project / "main.hl"
            entry.write_text(
                "fn main() -> i32 { return 12; }\n",
                encoding="utf-8",
            )
            objects = build_project_objects(
                entry,
                project / "objects",
                compiler=compiler,
            )
            objects[Path("main.hl")].write_bytes(
                objects[Path("main.hl")].read_bytes() + b"tampered"
            )
            output = project / "corrupt.exe"

            with self.assertRaisesRegex(
                C99LinkConfigurationError,
                "has changed since its object was built",
            ):
                link_project_objects(
                    entry,
                    objects,
                    output,
                    compiler=compiler,
                )

            self.assertFalse(output.exists())

    def test_link_project_objects_rejects_malformed_manifests(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "lib.hl").write_text(
                "fn value() -> i32 { return 12; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "lib.hl" as lib;\n'
                "fn main() -> i32 { return lib::value(); }\n",
                encoding="utf-8",
            )
            objects = build_project_objects(
                entry,
                project / "objects",
                compiler=compiler,
            )
            manifest_path = project / "objects" / "aether-objects.json"
            valid_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            output = project / "preserved.exe"
            preserved_output = b"previous executable"

            malformed_manifests = [
                ("unsupported format", lambda value: value.update(format="other")),
                ("legacy manifest version", lambda value: value.update(version=1)),
                ("boolean version", lambda value: value.update(version=True)),
                ("missing toolchain", lambda value: value.pop("toolchain")),
                ("invalid toolchain", lambda value: value.update(toolchain=[])),
                ("invalid entry path", lambda value: value.update(entry_source="./main.hl")),
                ("invalid module list", lambda value: value.update(modules={})),
                (
                    "missing module",
                    lambda value: value.update(modules=value["modules"][:1]),
                ),
                (
                    "duplicate source",
                    lambda value: value["modules"].append(
                        dict(value["modules"][0])
                    ),
                ),
                (
                    "duplicate object",
                    lambda value: value["modules"][1].update(
                        object=value["modules"][0]["object"]
                    ),
                ),
                (
                    "traversal source",
                    lambda value: value["modules"][0].update(source="../main.hl"),
                ),
                (
                    "noncanonical source",
                    lambda value: value["modules"][0].update(source="./main.hl"),
                ),
                (
                    "traversal object",
                    lambda value: value["modules"][0].update(object="../escape.o"),
                ),
                (
                    "invalid digest",
                    lambda value: value["modules"][0].update(source_sha256="g" * 64),
                ),
                (
                    "missing digest",
                    lambda value: value["modules"][0].pop("object_sha256"),
                ),
                ("non-object record", lambda value: value["modules"].__setitem__(0, None)),
            ]
            for label, mutate in malformed_manifests:
                with self.subTest(manifest=label):
                    mutated = json.loads(json.dumps(valid_manifest))
                    mutate(mutated)
                    manifest_path.write_text(
                        json.dumps(mutated),
                        encoding="utf-8",
                    )
                    output.write_bytes(preserved_output)

                    with self.assertRaises(C99LinkConfigurationError):
                        link_project_objects(
                            entry,
                            objects,
                            output,
                            compiler=compiler,
                        )

                    self.assertEqual(output.read_bytes(), preserved_output)

            invalid_encodings = [
                ("invalid json", b"{"),
                ("invalid utf-8", b"\xff"),
                (
                    "duplicate json key",
                    b'{"format":"aether-object-set","format":"aether-object-set"}',
                ),
            ]
            for label, contents in invalid_encodings:
                with self.subTest(manifest=label):
                    manifest_path.write_bytes(contents)
                    output.write_bytes(preserved_output)

                    with self.assertRaises(C99LinkConfigurationError):
                        link_project_objects(
                            entry,
                            objects,
                            output,
                            compiler=compiler,
                        )

                    self.assertEqual(output.read_bytes(), preserved_output)

            manifest_path.unlink()
            output.write_bytes(preserved_output)
            with self.assertRaisesRegex(
                C99LinkConfigurationError,
                "Cannot read object manifest",
            ):
                link_project_objects(
                    entry,
                    objects,
                    output,
                    compiler=compiler,
                )
            self.assertEqual(output.read_bytes(), preserved_output)

    def test_link_project_objects_normalizes_mapping_keys_and_rejects_collisions(
        self,
    ) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "lib.hl").write_text(
                "fn value() -> i32 { return 12; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "lib.hl" as lib;\n'
                "fn main() -> i32 { return lib::value(); }\n",
                encoding="utf-8",
            )
            objects = build_project_objects(
                entry,
                project / "objects",
                compiler=compiler,
            )
            equivalent_mapping = {
                ("./" + source.as_posix()): object_path
                for source, object_path in objects.items()
            }
            output = project / "linked.exe"
            link_project_objects(
                entry,
                equivalent_mapping,
                output,
                compiler=compiler,
            )
            self.assertTrue(output.is_file())

            collision_mapping = dict(equivalent_mapping)
            collision_mapping["main.hl"] = objects[Path("main.hl")]
            with self.assertRaisesRegex(
                C99LinkConfigurationError,
                "Duplicate object mapping",
            ):
                link_project_objects(
                    entry,
                    collision_mapping,
                    project / "collision.exe",
                    compiler=compiler,
                )

            traversal_mapping = dict(equivalent_mapping)
            traversal_mapping["..\\outside.hl"] = objects[Path("main.hl")]
            with self.assertRaisesRegex(
                C99LinkConfigurationError,
                "project-relative source path",
            ):
                link_project_objects(
                    entry,
                    traversal_mapping,
                    project / "traversal.exe",
                    compiler=compiler,
                )

    def test_link_project_objects_wraps_missing_object_error(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            entry = project / "main.hl"
            entry.write_text(
                "fn main() -> i32 { return 12; }\n",
                encoding="utf-8",
            )
            objects = build_project_objects(
                entry,
                project / "objects",
                compiler=compiler,
            )
            objects[Path("main.hl")].unlink()
            output = project / "preserved.exe"
            output.write_bytes(b"previous executable")

            with self.assertRaisesRegex(
                C99LinkConfigurationError,
                "Cannot read object file",
            ):
                link_project_objects(
                    entry,
                    objects,
                    output,
                    compiler=compiler,
                )

            self.assertEqual(output.read_bytes(), b"previous executable")

    def test_project_object_link_rejects_toolchain_mismatches(self) -> None:
        delegate = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            entry = project / "main.hl"
            entry.write_text(
                "fn main() -> i32 { return 12; }\n",
                encoding="utf-8",
            )
            wrapper, configuration_path = self.controllable_compiler(
                project,
                delegate,
            )
            baseline = {"version": "test cc 1.0", "target": "test-target-1"}

            for label, change, expected_field in (
                ("version", {"version": "test cc 2.0"}, "version"),
                ("target", {"target": "test-target-2"}, "target"),
                (
                    "compiler command",
                    {"command_argument": "alternate"},
                    "command_sha256",
                ),
            ):
                with self.subTest(mismatch=label):
                    configuration_path.write_text(
                        json.dumps(baseline),
                        encoding="utf-8",
                    )
                    objects = build_project_objects(
                        entry,
                        project / f"objects-{label.replace(' ', '-')}",
                        compiler=wrapper,
                    )
                    configuration_path.write_text(
                        json.dumps(
                            {
                                **baseline,
                                **{
                                    key: value
                                    for key, value in change.items()
                                    if key != "command_argument"
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    output = project / f"{label.replace(' ', '-')}.exe"
                    output.write_bytes(b"previous executable")
                    link_compiler = wrapper
                    if "command_argument" in change:
                        link_compiler = (*wrapper, change["command_argument"])

                    with self.assertRaisesRegex(
                        C99LinkConfigurationError,
                        f"mismatched fields: {expected_field}",
                    ):
                        link_project_objects(
                            entry,
                            objects,
                            output,
                            compiler=link_compiler,
                        )

                    self.assertEqual(output.read_bytes(), b"previous executable")

    def test_project_object_link_rejects_toolchain_flag_mismatch(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            entry = project / "main.hl"
            entry.write_text(
                "fn main() -> i32 { return 12; }\n",
                encoding="utf-8",
            )
            objects = build_project_objects(
                entry,
                project / "objects",
                compiler=compiler,
            )
            output = project / "flags.exe"
            output.write_bytes(b"previous executable")

            with patch(
                "src.build._C99_LINK_FLAGS",
                ("-std=c99", "-pedantic", "-Werror", "-fno-common"),
            ):
                with self.assertRaisesRegex(
                    C99LinkConfigurationError,
                    "mismatched fields: link_flags",
                ):
                    link_project_objects(
                        entry,
                        objects,
                        output,
                        compiler=compiler,
                    )

            self.assertEqual(output.read_bytes(), b"previous executable")

    def test_project_object_build_rejects_failed_toolchain_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            entry = project / "main.hl"
            entry.write_text(
                "fn main() -> i32 { return 12; }\n",
                encoding="utf-8",
            )
            wrapper, configuration_path = self.controllable_compiler(
                project,
                self.compiler(),
            )
            for label, identity, error_message in (
                (
                    "empty version",
                    {"version": "", "target": "test-target"},
                    "empty or oversized version",
                ),
                (
                    "empty target",
                    {"version": "test cc 1.0", "target": ""},
                    "invalid target identity",
                ),
            ):
                with self.subTest(probe=label):
                    configuration_path.write_text(
                        json.dumps(identity),
                        encoding="utf-8",
                    )
                    output_directory = project / f"objects-{label.replace(' ', '-')}"

                    with self.assertRaisesRegex(
                        C99ToolchainError,
                        error_message,
                    ):
                        build_project_objects(
                            entry,
                            output_directory,
                            compiler=wrapper,
                        )

                    self.assertFalse(output_directory.exists())

    def test_project_object_link_wraps_failed_toolchain_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            entry = project / "main.hl"
            entry.write_text(
                "fn main() -> i32 { return 12; }\n",
                encoding="utf-8",
            )
            wrapper, configuration_path = self.controllable_compiler(
                project,
                self.compiler(),
            )
            configuration_path.write_text(
                json.dumps({"version": "test cc 1.0", "target": "test-target"}),
                encoding="utf-8",
            )
            objects = build_project_objects(
                entry,
                project / "objects",
                compiler=wrapper,
            )
            configuration_path.write_text(
                json.dumps({"version": "", "target": "test-target"}),
                encoding="utf-8",
            )
            output = project / "preserved.exe"
            output.write_bytes(b"previous executable")

            with self.assertRaisesRegex(
                C99LinkConfigurationError,
                "Cannot identify C99 toolchain",
            ):
                link_project_objects(
                    entry,
                    objects,
                    output,
                    compiler=wrapper,
                )

            self.assertEqual(output.read_bytes(), b"previous executable")

    def test_build_project_objects_publishes_nothing_on_compiler_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "lib.hl").write_text(
                "fn value() -> i32 { return 12; }\n",
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "lib.hl" as lib;\n'
                "fn main() -> i32 { return lib::value(); }\n",
                encoding="utf-8",
            )
            call_count = project / "calls.txt"
            fake_compiler = project / "fake_compiler.py"
            fake_compiler.write_text(
                "import pathlib, sys\n"
                "if sys.argv[-1:] == ['--version']:\n"
                "    print('fake compiler 1.0')\n"
                "    raise SystemExit(0)\n"
                "if sys.argv[-1:] == ['-dumpmachine']:\n"
                "    print('fake-target')\n"
                "    raise SystemExit(0)\n"
                f"count_path = pathlib.Path({str(call_count)!r})\n"
                "count = int(count_path.read_text() if count_path.exists() else '0')\n"
                "count_path.write_text(str(count + 1))\n"
                "if count == 0:\n"
                "    output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
                "    output.write_bytes(b'fake object')\n"
                "    raise SystemExit(0)\n"
                "print('intentional compiler failure', file=sys.stderr)\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            output_directory = project / "objects"

            with self.assertRaises(C99BuildError):
                build_project_objects(
                    entry,
                    output_directory,
                    compiler=(sys.executable, str(fake_compiler)),
                )

            self.assertFalse(output_directory.exists())
            self.assertEqual(call_count.read_text(encoding="utf-8"), "2")

    def test_build_project_objects_rejects_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            entry = project / "main.hl"
            entry.write_text(
                "fn main() -> i32 { return 0; }\n",
                encoding="utf-8",
            )
            output_directory = project / "objects"
            output_directory.mkdir()
            marker = output_directory / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                build_project_objects(
                    entry,
                    output_directory,
                    compiler=("missing-c-compiler",),
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_project_executable_preserves_artifact_when_module_link_fails(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            entry = project / "main.hl"
            entry.write_text(
                '@ffi @link("native") extern "C" fn missing_symbol() -> i32;\n'
                "@ffi fn main() -> i32 { return missing_symbol(); }\n",
                encoding="utf-8",
            )
            library_object = project / "native.o"
            compile_c99_object(
                "int unrelated_symbol(void) { return 7; }\n",
                library_object,
                compiler=compiler,
            )
            output = project / "existing.exe"
            output.write_bytes(b"previous executable")

            with self.assertRaises(C99BuildError):
                build_project_executable(
                    entry,
                    output,
                    compiler=compiler,
                    libraries={"native": library_object},
                )
            self.assertEqual(output.read_bytes(), b"previous executable")

    def test_project_c99_passes_imported_struct_values(self) -> None:
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
                "fn main() -> i32 {\n"
                "  let point = types::Point { x: 7 };\n"
                "  return types::identity(point).x;\n"
                "}\n",
                encoding="utf-8",
            )

            executable = project / "program.exe"
            build_project_executable(
                entry,
                executable,
                compiler=self.compiler(),
            )
            result = subprocess.run(
                [str(executable)],
                capture_output=True,
                check=False,
                timeout=10,
                cwd=project,
            )
            self.assertEqual(result.returncode, 7, result.stderr)

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

    def test_project_rejects_reserved_and_unsupported_types_in_imported_modules(self) -> None:
        cases = (
            (
                "type i32 { value: i32, }\n"
                "fn echo(value: i32) -> i32 { return value; }\n",
                "E_RESERVED_TYPE",
            ),
            (
                "type Nested { value: Result<Id, IoError>, }\n"
                "fn accept(value: char) {}\n",
                "E_UNSUPPORTED_TYPE",
            ),
        )
        for library_source, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    project = Path(temporary_directory)
                    library = project / "types.hl"
                    library.write_text(library_source, encoding="utf-8")
                    entry = project / "main.hl"
                    entry.write_text(
                        'import "types.hl" as types;\n'
                        "fn main() -> i32 { return 0; }\n",
                        encoding="utf-8",
                    )

                    with self.assertRaises(CompilationError) as raised:
                        compile_project(entry)

                matching = [
                    diagnostic for diagnostic in raised.exception.diagnostics
                    if diagnostic.code == expected_code
                ]
                self.assertTrue(matching)
                self.assertTrue(
                    all(item.span.source_path == str(library.resolve()) for item in matching)
                )

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

    def test_project_links_imported_extern_and_injects_interpreter_handler(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            library_source = project / "native.hl"
            library_source.write_text(
                '@ffi @link("sensor") extern "C" fn read() -> i32;\n',
                encoding="utf-8",
            )
            entry = project / "main.hl"
            entry.write_text(
                'import "native.hl" as native;\n'
                "@ffi fn main() -> i32 { return native::read(); }\n",
                encoding="utf-8",
            )
            library_object = project / "sensor.o"
            compile_c99_object(
                "#include <stdint.h>\nint32_t read(void) { return 73; }\n",
                library_object,
                compiler=compiler,
            )

            module = compile_project(entry)
            imported_extern = next(
                function
                for function in module.functions
                if function.external_abi == "C"
            )
            self.assertTrue(imported_extern.name.startswith("$module:"))
            self.assertEqual(imported_extern.external_symbol, "read")
            self.assertEqual(imported_extern.external_library, "sensor")

            with self.assertRaises(C99LinkConfigurationError):
                build_project_executable(
                    entry,
                    project / "missing.exe",
                    compiler=compiler,
                )
            executable = build_project_executable(
                entry,
                project / "program.exe",
                compiler=compiler,
                libraries={"sensor": library_object},
            )
            result = subprocess.run(
                [str(executable)],
                capture_output=True,
                check=False,
                timeout=10,
            )
            interpreted = run_project(
                entry,
                "main",
                external_handlers={"sensor:read": lambda: 73},
            )

        self.assertEqual(result.returncode, interpreted, result.stderr)
        self.assertEqual(interpreted, 73)

    def test_imported_extern_effect_propagates_to_project_callers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "native.hl").write_text(
                '@ffi @link("sensor") extern "C" fn read() -> i32;\n',
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
            "E_EFFECT_VIOLATION",
            {diagnostic.code for diagnostic in raised.exception.diagnostics},
        )


if __name__ == "__main__":
    unittest.main()
