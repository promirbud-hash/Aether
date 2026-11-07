"""Host C99 toolchain driver for compiling Aether-generated C to an object."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Mapping, Sequence


_C99_FLAGS = (
    "-std=c99",
    "-pedantic",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-ffp-contract=off",
    "-frandom-seed=aether",
    "-c",
)
_C99_LINK_FLAGS = (
    "-std=c99",
    "-pedantic",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-ffp-contract=off",
    "-frandom-seed=aether",
)
_COMPILER_TIMEOUT_SECONDS = 120
_COMPILER_PROBE_TIMEOUT_SECONDS = 10
_MAX_COMPILER_VERSION_LENGTH = 16_384


class C99ToolchainError(RuntimeError):
    """Raised when no usable C99 compiler command can be found or started."""


class C99LinkConfigurationError(ValueError):
    """Raised when explicit native-library inputs do not match FFI declarations."""


@dataclass(frozen=True)
class C99BuildError(RuntimeError):
    """Compiler failure with the full command and captured diagnostics."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def __str__(self) -> str:
        details = self.stderr.strip() or self.stdout.strip() or "(no compiler output)"
        return (
            f"C99 compiler exited with status {self.returncode}: "
            f"{subprocess.list2cmdline(self.command)}\n{details}"
        )


def _resolve_compiler(
    compiler: str | os.PathLike[str] | Sequence[str] | None,
) -> tuple[str, ...]:
    if compiler is None:
        for name in ("cc", "gcc", "clang"):
            path = shutil.which(name)
            if path is not None:
                return (path,)
        zig = shutil.which("zig")
        if zig is not None:
            return (zig, "cc")
        raise C99ToolchainError(
            "No supported C compiler found (looked for cc, gcc, clang, or zig cc)"
        )

    if isinstance(compiler, (str, os.PathLike)):
        command = (os.fspath(compiler),)
    else:
        command = tuple(os.fspath(part) for part in compiler)
    if not command or any(not part for part in command):
        raise ValueError("compiler must be a non-empty command")

    executable = command[0]
    resolved_executable = (
        Path(executable).expanduser().resolve()
        if Path(executable).expanduser().is_file()
        else shutil.which(executable)
    )
    if resolved_executable is None:
        raise C99ToolchainError(
            f"C99 compiler executable {executable!r} was not found"
        )
    return (str(Path(resolved_executable).resolve()), *command[1:])


def c99_toolchain_fingerprint(
    compiler: str | os.PathLike[str] | Sequence[str] | None = None,
) -> dict[str, object]:
    """Return stable compiler, target, and flag metadata for object manifests."""
    command_prefix = _resolve_compiler(compiler)

    def probe(argument: str, *, require_single_line: bool = False) -> str:
        command = (*command_prefix, argument)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
                timeout=_COMPILER_PROBE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise C99ToolchainError(
                f"Unable to identify C99 compiler using "
                f"{subprocess.list2cmdline(command)}: {error}"
            ) from error
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip()
            raise C99ToolchainError(
                f"C99 compiler identity probe failed for "
                f"{subprocess.list2cmdline(command)}"
                + (f": {details}" if details else "")
            )
        output = result.stdout.strip()
        if argument == "--version":
            output = "\n".join(
                part.strip()
                for part in (result.stdout, result.stderr)
                if part.strip()
            )
            if not output or len(output) > _MAX_COMPILER_VERSION_LENGTH:
                raise C99ToolchainError(
                    "C99 compiler returned an empty or oversized version string"
                )
        elif require_single_line:
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if len(lines) != 1:
                raise C99ToolchainError(
                    "C99 compiler returned an invalid target identity"
                )
            output = lines[0]
        if not output:
            raise C99ToolchainError(
                f"C99 compiler returned no output for {argument!r}"
            )
        return output

    command_digest = hashlib.sha256(
        json.dumps(command_prefix, ensure_ascii=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "executable": command_prefix[0],
        "command_sha256": command_digest,
        "version": probe("--version"),
        "target": probe("-dumpmachine", require_single_line=True),
        "compile_flags": list(_C99_FLAGS),
        "link_flags": list(_C99_LINK_FLAGS),
    }


def compile_c99_object(
    c_source: str,
    output_path: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | Sequence[str] | None = None,
) -> Path:
    """Compile a C99 translation unit to an object without invoking a linker.

    The object is written atomically after a successful compiler run. Compiler
    arguments are passed directly to the process without a shell.
    """
    command_prefix = _resolve_compiler(compiler)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".aether-c99-",
        dir=output.parent,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        source_file = temporary / "module.c"
        temporary_object = temporary / "module.o"
        source_file.write_text(c_source, encoding="utf-8", newline="\n")
        command = (
            *command_prefix,
            *_C99_FLAGS,
            source_file.name,
            "-o",
            str(temporary_object),
        )
        try:
            result = subprocess.run(
                command,
                cwd=temporary,
                capture_output=True,
                text=True,
                check=False,
                timeout=_COMPILER_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as error:
            raise C99ToolchainError(
                f"Unable to start C99 compiler {command_prefix[0]!r}: {error}"
            ) from error
        except subprocess.TimeoutExpired as error:
            stdout = (
                error.stdout.decode(errors="replace")
                if isinstance(error.stdout, bytes)
                else error.stdout or ""
            )
            stderr = (
                error.stderr.decode(errors="replace")
                if isinstance(error.stderr, bytes)
                else error.stderr or ""
            )
            raise C99BuildError(
                command,
                -1,
                stdout,
                stderr or "Compiler timed out",
            ) from error

        if result.returncode != 0:
            raise C99BuildError(
                command,
                result.returncode,
                result.stdout,
                result.stderr,
            )
        if not temporary_object.is_file():
            raise C99ToolchainError(
                "C99 compiler reported success but did not produce an object file"
            )
        os.replace(temporary_object, output)
    return output


def compile_c99_sources_to_objects(
    c_sources: Mapping[str, str],
    output_directory: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | Sequence[str] | None = None,
) -> dict[str, Path]:
    """Compile named C99 sources and atomically publish their object directory."""
    if not c_sources:
        raise ValueError("at least one C source file is required")
    output = Path(output_directory).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Object output directory already exists: {output}")

    source_names = tuple(sorted(c_sources))
    for name in source_names:
        relative = Path(name)
        if (
            not name
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.name in {"", ".", ".."}
        ):
            raise ValueError(f"Object file name must be a safe relative path: {name!r}")

    command_prefix = _resolve_compiler(compiler)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".aether-objects-",
        dir=output.parent,
    ) as temporary_directory:
        staging = Path(temporary_directory) / "objects"
        staging.mkdir()
        for name in source_names:
            compile_c99_object(
                c_sources[name],
                staging / name,
                compiler=command_prefix,
            )
        os.rename(staging, output)

    return {name: output / name for name in source_names}


def link_c99_executable(
    c_source: str,
    output_path: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | Sequence[str] | None = None,
    libraries: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Path:
    """Compile and link one C99 translation unit with explicit libraries."""
    return link_c99_executable_sources(
        (c_source,),
        output_path,
        compiler=compiler,
        libraries=libraries,
    )


def link_c99_executable_sources(
    c_sources: Sequence[str],
    output_path: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | Sequence[str] | None = None,
    libraries: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Path:
    """Compile separate C99 translation units and link them atomically."""
    if not c_sources:
        raise ValueError("at least one C source file is required")
    command_prefix = _resolve_compiler(compiler)
    output = Path(output_path).expanduser().resolve()
    library_paths = {
        Path(path).expanduser().resolve()
        for path in (libraries or {}).values()
    }
    if output in library_paths:
        raise C99LinkConfigurationError(
            "Executable output path cannot overwrite an input library"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".aether-c99-",
        dir=output.parent,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        temporary_executable = temporary / (
            "module.exe" if os.name == "nt" else "module"
        )
        object_files = []
        for index, c_source in enumerate(c_sources):
            object_path = temporary / f"module_{index}.o"
            compile_c99_object(
                c_source,
                object_path,
                compiler=command_prefix,
            )
            object_files.append(object_path)
        link_c99_objects(
            object_files,
            temporary_executable,
            compiler=command_prefix,
            libraries=libraries,
        )
        if not temporary_executable.is_file():
            raise C99ToolchainError(
                "C compiler reported success but did not produce an executable"
            )
        os.replace(temporary_executable, output)
    return output


def link_c99_objects(
    object_files: Sequence[str | os.PathLike[str]],
    output_path: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | Sequence[str] | None = None,
    libraries: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Path:
    """Link existing object files and explicitly named native libraries."""
    if not object_files:
        raise ValueError("at least one object file is required")
    command_prefix = _resolve_compiler(compiler)
    objects = tuple(Path(path).expanduser().resolve() for path in object_files)
    if len(set(objects)) != len(objects):
        raise C99LinkConfigurationError("Object file inputs must be unique")
    for path in objects:
        if not path.is_file():
            raise C99LinkConfigurationError(
                f"Object file does not exist: {path}"
            )

    library_paths: list[Path] = []
    for name, library_path in sorted((libraries or {}).items()):
        if not name or Path(name).name != name:
            raise C99LinkConfigurationError(
                f"Invalid @link library name {name!r}"
            )
        path = Path(library_path).expanduser().resolve()
        if not path.is_file():
            raise C99LinkConfigurationError(
                f"Library {name!r} does not resolve to a file: {path}"
            )
        library_paths.append(path)

    output = Path(output_path).expanduser().resolve()
    if output in (*objects, *library_paths):
        raise C99LinkConfigurationError(
            "Executable output path cannot overwrite an input object or library"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".aether-link-",
        dir=output.parent,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        temporary_executable = temporary / (
            "module.exe" if os.name == "nt" else "module"
        )
        command = (
            *command_prefix,
            *_C99_LINK_FLAGS,
            *(str(path) for path in objects),
            *(str(path) for path in library_paths),
            "-o",
            str(temporary_executable),
        )
        try:
            result = subprocess.run(
                command,
                cwd=temporary,
                capture_output=True,
                text=True,
                check=False,
                timeout=_COMPILER_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as error:
            raise C99ToolchainError(
                f"Unable to start C99 compiler {command_prefix[0]!r}: {error}"
            ) from error
        except subprocess.TimeoutExpired as error:
            stdout = (
                error.stdout.decode(errors="replace")
                if isinstance(error.stdout, bytes)
                else error.stdout or ""
            )
            stderr = (
                error.stderr.decode(errors="replace")
                if isinstance(error.stderr, bytes)
                else error.stderr or ""
            )
            raise C99BuildError(
                command,
                -1,
                stdout,
                stderr or "Compiler/linker timed out",
            ) from error

        if result.returncode != 0:
            raise C99BuildError(
                command,
                result.returncode,
                result.stdout,
                result.stderr,
            )
        if not temporary_executable.is_file():
            raise C99ToolchainError(
                "C compiler reported success but did not produce an executable"
            )
        os.replace(temporary_executable, output)
    return output
