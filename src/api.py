"""Stable parse, compile, and execution entry points for the Aether prototype."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import tempfile
from typing import Mapping, Sequence

from src.build import (
    C99LinkConfigurationError,
    C99ToolchainError,
    c99_toolchain_fingerprint,
    compile_c99_object,
    compile_c99_sources_to_objects,
    link_c99_executable,
    link_c99_objects,
    link_c99_executable_sources,
)
from src.c99 import C99LoweringError, emit_c99_module, emit_c99_module_units
from src.frontend import Diagnostic, SourceSpan, parse_source
from src.interpreter import (
    EffectHandler,
    ExternalFunctionHandler,
    Interpreter,
)
from src.ir import IRLoweringError, IRModule, lower_module
from src.parser import ParseError
from src.project import ProjectCompilationError, compile_project_ir


class CompilationError(ValueError):
    """Compilation stopped because parsing, analysis, or lowering failed."""

    def __init__(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        summary = "; ".join(
            (
                f"{diagnostic.span.source_path}:"
                f"{diagnostic.span.line}:{diagnostic.span.column}: "
                if diagnostic.span.source_path
                else f"line {diagnostic.span.line}, column {diagnostic.span.column}: "
            )
            + f"{diagnostic.code}: {diagnostic.message}"
            for diagnostic in diagnostics
        )
        super().__init__(summary or "Aether compilation failed")


_OBJECT_MANIFEST_NAME = "aether-objects.json"
_OBJECT_MANIFEST_FORMAT = "aether-object-set"
_OBJECT_MANIFEST_VERSION = 2


class _DuplicateManifestKeyError(ValueError):
    """Raised when an object manifest contains an ambiguous JSON object."""


def _unique_manifest_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateManifestKeyError(
                f"Duplicate object manifest key: {key!r}"
            )
        result[key] = value
    return result


def _is_normalized_manifest_source(source: str) -> bool:
    if not source or "\\" in source or "\0" in source or ":" in source:
        return False
    parts = source.split("/")
    return all(part not in ("", ".", "..") for part in parts)


def _is_manifest_object_name(object_name: str) -> bool:
    return (
        bool(object_name)
        and object_name not in (".", "..")
        and "/" not in object_name
        and "\\" not in object_name
        and "\0" not in object_name
        and ":" not in object_name
        and not PureWindowsPath(object_name).drive
    )


def _is_sha256_digest(value: str) -> bool:
    return len(value) == 64 and all(
        "0" <= character <= "9" or "a" <= character <= "f"
        for character in value
    )


def compile_source(source: str) -> IRModule:
    """Parse, analyze, and lower source, refusing all reported errors."""
    try:
        parsed = parse_source(source)
    except ParseError as error:
        diagnostic = Diagnostic(
            "E_LEXICAL",
            str(error),
            SourceSpan(error.offset, error.offset, error.line, error.column),
        )
        raise CompilationError((diagnostic,)) from error

    errors = tuple(
        diagnostic
        for diagnostic in parsed.diagnostics
        if diagnostic.severity == "error"
    )
    if errors:
        raise CompilationError(errors)
    try:
        return lower_module(parsed.ast)
    except IRLoweringError as error:
        raise CompilationError(error.diagnostics) from error


def compile_project(entry_path: str | os.PathLike[str]) -> IRModule:
    """Compile a .hl project entry file and its relative imports to one IR module."""
    try:
        return compile_project_ir(entry_path)
    except ProjectCompilationError as error:
        raise CompilationError(error.diagnostics) from error


def emit_c99(source: str) -> str:
    """Compile source and emit C99 for the checked scalar backend subset."""
    module = compile_source(source)
    try:
        return emit_c99_module(module)
    except C99LoweringError as error:
        raise CompilationError(error.diagnostics) from error


def emit_c99_project(entry_path: str | os.PathLike[str]) -> str:
    """Compile a multi-file project and emit its supported C99 subset."""
    module = compile_project(entry_path)
    try:
        return emit_c99_module(module)
    except C99LoweringError as error:
        raise CompilationError(error.diagnostics) from error


def build_object(
    source: str,
    output_path: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | Sequence[str] | None = None,
) -> Path:
    """Compile supported Aether source to a host C99 object file.

    This invokes a C compiler with strict C99 flags but does not link an
    executable or load any native libraries.
    """
    generated = emit_c99(source)
    return compile_c99_object(
        generated,
        output_path,
        compiler=compiler,
    )


def build_project_object(
    entry_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | Sequence[str] | None = None,
) -> Path:
    """Compile a supported multi-file project to a host C99 object file."""
    generated = emit_c99_project(entry_path)
    return compile_c99_object(
        generated,
        output_path,
        compiler=compiler,
    )


def build_project_objects(
    entry_path: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | Sequence[str] | None = None,
) -> dict[Path, Path]:
    """Compile each project module to an object in a new output directory.

    The returned mapping uses paths relative to the project entry file as keys.
    The output directory must not already exist; objects are published there
    together only after every module compiles successfully.
    """
    root_path = Path(entry_path).expanduser().resolve()
    module = compile_project(root_path)
    try:
        generated_units = emit_c99_module_units(module, os.fspath(root_path))
    except C99LoweringError as error:
        raise CompilationError(error.diagnostics) from error

    sources_by_name: dict[str, str] = {}
    source_paths_by_name: dict[str, Path] = {}
    source_hashes = {
        Path(source_path).expanduser().resolve(): digest
        for source_path, digest in module.source_hashes
    }
    manifest_modules = []
    for relative_source, c_source in generated_units:
        digest = hashlib.sha256(relative_source.encode("utf-8")).hexdigest()
        object_name = f"aether_{digest}.o"
        sources_by_name[object_name] = c_source
        source_path = root_path.parent / Path(relative_source)
        source_paths_by_name[object_name] = Path(relative_source)
        manifest_modules.append(
            {
                "source": relative_source,
                "source_sha256": source_hashes[source_path],
                "generated_c_sha256": hashlib.sha256(
                    c_source.encode("utf-8")
                ).hexdigest(),
                "object": object_name,
            }
        )

    requested_output = Path(output_directory).expanduser()
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(
            f"Object output directory already exists: {requested_output}"
        )
    toolchain = c99_toolchain_fingerprint(compiler)
    output = requested_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".aether-project-objects-",
        dir=output.parent,
    ) as temporary_directory:
        staging = Path(temporary_directory) / "objects"
        staged_objects = compile_c99_sources_to_objects(
            sources_by_name,
            staging,
            compiler=compiler,
        )
        for record in manifest_modules:
            object_path = staging / record["object"]
            record["object_sha256"] = hashlib.sha256(
                object_path.read_bytes()
            ).hexdigest()
        manifest = {
            "format": _OBJECT_MANIFEST_FORMAT,
            "version": _OBJECT_MANIFEST_VERSION,
            "entry_source": root_path.relative_to(root_path.parent).as_posix(),
            "toolchain": toolchain,
            "modules": manifest_modules,
        }
        (staging / _OBJECT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.rename(staging, output)

    return {
        source_paths_by_name[object_name]: output / object_name
        for object_name in staged_objects
    }


def _validate_library_mappings(
    module: IRModule,
    libraries: Mapping[str, str | os.PathLike[str]] | None,
) -> None:
    declared_libraries = {
        function.external_library
        for function in module.functions
        if function.external_abi is not None
        and function.external_library is not None
    }
    supplied_libraries = set(libraries or {})
    missing = declared_libraries - supplied_libraries
    unexpected = supplied_libraries - declared_libraries
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing @link file(s) for " + ", ".join(sorted(missing)))
        if unexpected:
            details.append("unused library mapping(s) " + ", ".join(sorted(unexpected)))
        raise C99LinkConfigurationError("; ".join(details))


def _build_executable_from_ir(
    module: IRModule,
    output_path: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | Sequence[str] | None,
    libraries: Mapping[str, str | os.PathLike[str]] | None,
) -> Path:
    try:
        generated = emit_c99_module(module, include_entry_point=True)
    except C99LoweringError as error:
        raise CompilationError(error.diagnostics) from error
    _validate_library_mappings(module, libraries)
    return link_c99_executable(
        generated,
        output_path,
        compiler=compiler,
        libraries=libraries,
    )


def build_executable(
    source: str,
    output_path: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | Sequence[str] | None = None,
    libraries: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Path:
    """Build a host executable, resolving each @link name to a library file."""
    module = compile_source(source)
    return _build_executable_from_ir(
        module,
        output_path,
        compiler=compiler,
        libraries=libraries,
    )


def build_project_executable(
    entry_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | Sequence[str] | None = None,
    libraries: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Path:
    """Build a host executable by compiling each project module separately."""
    root_path = Path(entry_path).expanduser().resolve()
    module = compile_project(root_path)
    try:
        generated_units = emit_c99_module_units(module, os.fspath(root_path))
    except C99LoweringError as error:
        raise CompilationError(error.diagnostics) from error
    _validate_library_mappings(module, libraries)
    return link_c99_executable_sources(
        tuple(source for _, source in generated_units),
        output_path,
        compiler=compiler,
        libraries=libraries,
    )


def link_project_objects(
    entry_path: str | os.PathLike[str],
    object_paths: Mapping[
        str | os.PathLike[str],
        str | os.PathLike[str],
    ],
    output_path: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | Sequence[str] | None = None,
    libraries: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Path:
    """Link one previously built object per module of the current project."""
    root_path = Path(entry_path).expanduser().resolve()
    module = compile_project(root_path)
    try:
        generated_units = emit_c99_module_units(module, os.fspath(root_path))
    except C99LoweringError as error:
        raise CompilationError(error.diagnostics) from error
    _validate_library_mappings(module, libraries)

    expected_sources = {
        Path(source_path).expanduser().resolve().relative_to(root_path.parent).as_posix()
        for source_path in module.source_paths
    }
    supplied_objects: dict[str, Path] = {}
    for source_path, object_path in object_paths.items():
        relative = Path(source_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise C99LinkConfigurationError(
                f"Object mapping key must be a project-relative source path: {source_path!r}"
            )
        normalized = relative.as_posix()
        if normalized in supplied_objects:
            raise C99LinkConfigurationError(
                f"Duplicate object mapping for project source {normalized!r}"
            )
        supplied_objects[normalized] = Path(object_path).expanduser().resolve()

    missing = expected_sources - supplied_objects.keys()
    unexpected = supplied_objects.keys() - expected_sources
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing object(s) for " + ", ".join(sorted(missing)))
        if unexpected:
            details.append("unexpected project source(s) " + ", ".join(sorted(unexpected)))
        raise C99LinkConfigurationError("; ".join(details))

    object_directories = {path.parent for path in supplied_objects.values()}
    if len(object_directories) != 1:
        raise C99LinkConfigurationError(
            "All project object files must be in the same manifest directory"
        )
    object_directory = next(iter(object_directories))
    manifest_path = object_directory / _OBJECT_MANIFEST_NAME
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_manifest_object,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateManifestKeyError,
    ) as error:
        raise C99LinkConfigurationError(
            f"Cannot read object manifest {manifest_path}: {error}"
        ) from error

    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != _OBJECT_MANIFEST_FORMAT
        or type(manifest.get("version")) is not int
        or manifest.get("version") != _OBJECT_MANIFEST_VERSION
    ):
        raise C99LinkConfigurationError(
            f"Unsupported or invalid object manifest: {manifest_path}"
        )
    try:
        current_toolchain = c99_toolchain_fingerprint(compiler)
    except C99ToolchainError as error:
        raise C99LinkConfigurationError(
            f"Cannot identify C99 toolchain for object linking: {error}"
        ) from error
    recorded_toolchain = manifest.get("toolchain")
    if not isinstance(recorded_toolchain, dict):
        raise C99LinkConfigurationError(
            f"Invalid toolchain metadata in object manifest: {manifest_path}"
        )
    mismatched_toolchain_fields = sorted(
        key
        for key in set(current_toolchain) | set(recorded_toolchain)
        if recorded_toolchain.get(key) != current_toolchain.get(key)
    )
    if mismatched_toolchain_fields:
        raise C99LinkConfigurationError(
            "C99 toolchain does not match the object manifest "
            f"(mismatched fields: {', '.join(mismatched_toolchain_fields)})"
        )
    entry_source = root_path.relative_to(root_path.parent).as_posix()
    recorded_entry_source = manifest.get("entry_source")
    if (
        not isinstance(recorded_entry_source, str)
        or not _is_normalized_manifest_source(recorded_entry_source)
        or recorded_entry_source != entry_source
    ):
        raise C99LinkConfigurationError(
            "Object manifest belongs to a different project entry file"
        )
    manifest_records = manifest.get("modules")
    if not isinstance(manifest_records, list):
        raise C99LinkConfigurationError(
            f"Invalid module list in object manifest: {manifest_path}"
        )

    current_source_hashes = {
        Path(source_path).expanduser().resolve().relative_to(root_path.parent).as_posix():
        digest
        for source_path, digest in module.source_hashes
    }
    recorded_modules: dict[str, dict[str, str]] = {}
    recorded_object_names: set[str] = set()
    for record in manifest_records:
        if not isinstance(record, dict):
            raise C99LinkConfigurationError(
                f"Invalid module record in object manifest: {manifest_path}"
            )
        source = record.get("source")
        object_name = record.get("object")
        source_hash = record.get("source_sha256")
        generated_hash = record.get("generated_c_sha256")
        object_hash = record.get("object_sha256")
        if not all(
            isinstance(value, str)
            for value in (
                source,
                object_name,
                source_hash,
                generated_hash,
                object_hash,
            )
        ):
            raise C99LinkConfigurationError(
                f"Invalid hashes or paths in object manifest: {manifest_path}"
            )
        if (
            not _is_normalized_manifest_source(source)
            or not _is_manifest_object_name(object_name)
            or not all(
                _is_sha256_digest(digest)
                for digest in (source_hash, generated_hash, object_hash)
            )
        ):
            raise C99LinkConfigurationError(
                f"Invalid hashes or paths in object manifest: {manifest_path}"
            )
        if source in recorded_modules:
            raise C99LinkConfigurationError(
                f"Duplicate source entry in object manifest: {source!r}"
            )
        if object_name in recorded_object_names:
            raise C99LinkConfigurationError(
                f"Duplicate object entry in object manifest: {object_name!r}"
            )
        recorded_object_names.add(object_name)
        recorded_modules[source] = {
            "object": object_name,
            "source_sha256": source_hash,
            "generated_c_sha256": generated_hash,
            "object_sha256": object_hash,
        }

    if recorded_modules.keys() != expected_sources:
        raise C99LinkConfigurationError(
            "Object manifest module list does not match the current project"
        )
    generated_hashes = {
        source: hashlib.sha256(c_source.encode("utf-8")).hexdigest()
        for source, c_source in generated_units
    }
    for source, record in recorded_modules.items():
        object_path = supplied_objects[source]
        if object_path.parent != object_directory:
            raise C99LinkConfigurationError(
                "Project object file does not reside beside its manifest"
            )
        if record["object"] != object_path.name:
            raise C99LinkConfigurationError(
                f"Object manifest names a different file for source {source!r}"
            )
        if record["source_sha256"] != current_source_hashes[source]:
            raise C99LinkConfigurationError(
                f"Source {source!r} has changed since its object was built"
            )
        if record["generated_c_sha256"] != generated_hashes[source]:
            raise C99LinkConfigurationError(
                f"Generated C for source {source!r} does not match the manifest"
            )
        try:
            actual_object_hash = hashlib.sha256(object_path.read_bytes()).hexdigest()
        except OSError as error:
            raise C99LinkConfigurationError(
                f"Cannot read object file for source {source!r}: {error}"
            ) from error
        if record["object_sha256"] != actual_object_hash:
            raise C99LinkConfigurationError(
                f"Object file for source {source!r} has changed since its object was built"
            )

    return link_c99_objects(
        tuple(supplied_objects[source] for source in sorted(expected_sources)),
        output_path,
        compiler=compiler,
        libraries=libraries,
    )


def run_source(
    source: str,
    function_name: str,
    *arguments: object,
    effect_handlers: Mapping[str, EffectHandler] | None = None,
    external_handlers: Mapping[str, ExternalFunctionHandler] | None = None,
    max_call_depth: int = 128,
    **keywords: object,
) -> object:
    """Compile source and invoke one function with explicit effect handlers."""
    module = compile_source(source)
    interpreter = Interpreter(
        module,
        effect_handlers=effect_handlers,
        external_handlers=external_handlers,
        max_call_depth=max_call_depth,
    )
    return interpreter.run(function_name, *arguments, **keywords)


def run_project(
    entry_path: str | os.PathLike[str],
    function_name: str,
    *arguments: object,
    effect_handlers: Mapping[str, EffectHandler] | None = None,
    external_handlers: Mapping[str, ExternalFunctionHandler] | None = None,
    max_call_depth: int = 128,
    **keywords: object,
) -> object:
    """Compile a project and invoke a function from its entry module."""
    module = compile_project(entry_path)
    interpreter = Interpreter(
        module,
        effect_handlers=effect_handlers,
        external_handlers=external_handlers,
        max_call_depth=max_call_depth,
    )
    return interpreter.run(function_name, *arguments, **keywords)
