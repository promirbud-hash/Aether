"""Shared runtime contracts for host I/O, files, and process arguments."""

from __future__ import annotations

from enum import IntEnum
import os
from pathlib import Path
from typing import TypeAlias


class IoErrorCode(IntEnum):
    END_OF_INPUT = 1
    INVALID_UTF8 = 2
    IO_FAILURE = 3


class FsErrorCode(IntEnum):
    NOT_FOUND = 1
    INVALID_PATH = 2
    INVALID_UTF8 = 3
    IO_FAILURE = 4


class ArgsErrorCode(IntEnum):
    OUT_OF_RANGE = 1
    INVALID_UTF8 = 2


class CollectionErrorCode(IntEnum):
    OUT_OF_RANGE = 1
    EMPTY = 2
    ALLOCATION_FAILURE = 3
    CAPACITY_OVERFLOW = 4


class ProjectPathError(ValueError):
    """A path is not a valid project-relative path."""


class ProjectPathNotFound(FileNotFoundError):
    """A project-relative path does not exist."""


PathLike: TypeAlias = str | os.PathLike[str]


def resolve_project_path(
    project_root: PathLike,
    path: str,
    *,
    for_write: bool = False,
) -> Path:
    """Resolve a relative path while keeping symlink targets inside the root."""
    root = Path(project_root).expanduser().resolve(strict=True)
    if not path or "\0" in path:
        raise ProjectPathError("path is empty or contains NUL")
    candidate = Path(path)
    if candidate.is_absolute() or candidate.drive:
        raise ProjectPathError("path must be relative to the project root")
    if any(part == ".." for part in path.replace("\\", "/").split("/")):
        raise ProjectPathError("parent traversal is not allowed")

    unresolved = root / candidate
    try:
        if for_write and unresolved.is_symlink():
            resolved = unresolved.resolve(strict=False)
        elif for_write and not unresolved.exists():
            parent = unresolved.parent.resolve(strict=True)
            resolved = parent / unresolved.name
        else:
            resolved = unresolved.resolve(strict=True)
    except FileNotFoundError as error:
        raise ProjectPathNotFound(path) from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ProjectPathError("path resolves outside the project root") from error
    return resolved
