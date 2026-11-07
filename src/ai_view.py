"""Deterministic, tokenizer-profiled AI view generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

from src.frontend import Module, parse_source_file, render_module
from src.parser import CSTNode, parse


_WORD_KINDS = {"IDENTIFIER", "KEYWORD", "NUMBER", "STRING"}
_MERGING_OPERATORS = {
    "?.", "|>", "++", "--", "**", "//", "/*", "<<", ">>", "&&", "||",
    "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=", "::",
}
_RESERVED = {
    "fn", "func", "type", "struct", "let", "var", "const", "return",
    "if", "else", "for", "while", "match", "true", "false", "none",
}
_PROFILE_DIR = Path(__file__).resolve().parents[1] / "profiles"


class Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


class ProfileError(ValueError):
    """Raised when a tokenizer profile is invalid or cannot be loaded."""


class ProfileUnavailable(ProfileError):
    """Raised when a profile's tokenizer is not installed or cached locally."""


@dataclass(frozen=True)
class TokenizerProfile:
    name: str
    backend: str
    tokenizer: str
    operator_aliases: dict[str, tuple[str, ...]]
    idiom_aliases: dict[str, tuple[str, ...]]
    semantic_aliases: dict[str, tuple[str, ...]]
    alias_candidates: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class ViewMapping:
    source_start: int
    source_end: int
    view_start: int
    view_end: int
    kind: str


@dataclass(frozen=True)
class AIView:
    text: str
    mappings: tuple[ViewMapping, ...]
    profile: str


def load_profile(path: str | Path) -> TokenizerProfile:
    """Load and validate one JSON tokenizer profile."""
    profile_path = Path(path)
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"Cannot load tokenizer profile {profile_path}: {error}") from error
    required = {"name", "tokenizer", "operators", "idioms", "alias_candidates"}
    missing = required - data.keys()
    if missing:
        raise ProfileError(
            f"Tokenizer profile {profile_path} is missing: {', '.join(sorted(missing))}"
        )
    tokenizer = data["tokenizer"]
    if not isinstance(tokenizer, dict) or tokenizer.get("backend") not in {
        "tiktoken",
        "huggingface",
    }:
        raise ProfileError(f"Unsupported tokenizer backend in {profile_path}")
    semantic = data.get("semantic_aliases", {})
    if not isinstance(semantic, dict) or any(
        not isinstance(value, list) or not all(isinstance(item, str) for item in value)
        for value in semantic.values()
    ):
        raise ProfileError(f"semantic_aliases must map names to arrays in {profile_path}")
    aliases = tuple(data["alias_candidates"])
    if not aliases or len(set(aliases)) != len(aliases):
        raise ProfileError(f"alias_candidates must be a non-empty unique list in {profile_path}")
    alias_tables = {}
    for key in ("operators", "idioms"):
        table = data[key]
        if not isinstance(table, dict):
            raise ProfileError(f"{key} must be an alias map in {profile_path}")
        normalized: dict[str, tuple[str, ...]] = {}
        for source, value in table.items():
            choices = (value,) if isinstance(value, str) else tuple(value) if isinstance(value, list) else ()
            if (
                not isinstance(source, str)
                or not choices
                or not all(isinstance(alias, str) and alias for alias in choices)
            ):
                raise ProfileError(
                    f"{key} entries must map strings to aliases or alias arrays in {profile_path}"
                )
            normalized[source] = choices
        alias_tables[key] = normalized
    return TokenizerProfile(
        name=data["name"],
        backend=tokenizer["backend"],
        tokenizer=tokenizer["name"],
        operator_aliases=alias_tables["operators"],
        idiom_aliases=alias_tables["idioms"],
        semantic_aliases={
            name: tuple(candidates) for name, candidates in semantic.items()
        },
        alias_candidates=aliases,
        description=data.get("description", ""),
    )


def load_profiles(directory: str | Path = _PROFILE_DIR) -> list[TokenizerProfile]:
    """Load all JSON profiles from a directory in stable filename order."""
    paths = sorted(Path(directory).glob("*.json"))
    if not paths:
        raise ProfileError(f"No tokenizer profiles found in {directory}")
    profiles = [load_profile(path) for path in paths]
    names = [profile.name for profile in profiles]
    if len(set(names)) != len(names):
        raise ProfileError(f"Duplicate tokenizer profile names in {directory}")
    return profiles


def default_profile() -> TokenizerProfile:
    """Return the backward-compatible GPT-4o/o200k profile."""
    return load_profile(_PROFILE_DIR / "gpt-4o-o200k.json")


def create_encoder(profile: TokenizerProfile) -> Encoder:
    """Create the profile tokenizer without network access or model downloads."""
    if profile.backend == "tiktoken":
        try:
            import tiktoken
        except ImportError as error:
            raise ProfileUnavailable(
                f"Profile {profile.name!r} requires tiktoken; install requirements.txt"
            ) from error
        try:
            return tiktoken.get_encoding(profile.tokenizer)
        except ValueError as error:
            raise ProfileError(
                f"Unknown tiktoken encoding {profile.tokenizer!r} for {profile.name!r}"
            ) from error

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ProfileUnavailable(
            f"Profile {profile.name!r} requires optional package transformers"
        ) from error
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            profile.tokenizer,
            local_files_only=True,
            trust_remote_code=False,
        )
    except OSError as error:
        raise ProfileUnavailable(
            f"Tokenizer {profile.tokenizer!r} for {profile.name!r} is not cached locally"
        ) from error

    class HuggingFaceEncoder:
        def encode(self, text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False)

    return HuggingFaceEncoder()


def _leaves(node: CSTNode):
    if node.is_leaf:
        yield node
        return
    for child in node.children:
        yield from _leaves(child)


def _view(
    leaves: list[tuple[str, str, int, int]],
    profile: TokenizerProfile,
    aliases: dict[str, str],
) -> AIView:
    output: list[str] = []
    mappings: list[ViewMapping] = []
    view_offset = 0

    def append(text: str) -> tuple[int, int]:
        nonlocal view_offset
        start = view_offset
        output.append(text)
        view_offset += len(text)
        return start, view_offset

    declarations = sorted(aliases.items(), key=lambda item: item[1])
    if declarations:
        header = f"@a({profile.name};"
        append(header)
        for index, (source_value, alias) in enumerate(declarations):
            if index:
                append(",")
            start, end = append(f"{alias}={source_value}")
            origin = next(
                ((source_start, source_end) for _, value, source_start, source_end in leaves if value == source_value),
                None,
            )
            if origin is not None:
                mappings.append(
                    ViewMapping(origin[0], origin[1], start, end, "alias_declaration")
                )
        append(")")

    previous_rendered = ""
    previous_is_word = False
    for kind, value, source_start, source_end in leaves:
        rendered = aliases.get(value, value)
        current_is_word = (
            rendered[:1].isalnum()
            or rendered[:1] == "_"
            or (value not in aliases and kind in _WORD_KINDS)
        )
        if output:
            if (
                previous_is_word
                and current_is_word
            ) or previous_rendered + rendered in _MERGING_OPERATORS:
                append(" ")
        start, end = append(rendered)
        mappings.append(
            ViewMapping(source_start, source_end, start, end, "token")
        )
        previous_rendered = rendered
        previous_is_word = (
            rendered[-1:].isalnum()
            or rendered[-1:] == "_"
            or (value not in aliases and kind in _WORD_KINDS)
        )
    return AIView("".join(output), tuple(mappings), profile.name)


def _collect_leaves(tree: CSTNode) -> list[tuple[str, str, int, int]]:
    return [
        (leaf.kind, leaf.value, leaf.start, leaf.end)
        for leaf in _leaves(tree)
        if leaf.kind not in {"WHITESPACE", "NEWLINE", "COMMENT"}
    ]


def _select_aliases(
    leaves: list[tuple[str, str, int, int]],
    profile: TokenizerProfile,
    encoder: Encoder | None,
) -> dict[str, str]:
    counts = Counter(value for kind, value, _, _ in leaves if kind == "IDENTIFIER")
    reserved = set(counts) | _RESERVED
    candidates_by_name = {
        name: profile.semantic_aliases.get(name, profile.alias_candidates)
        for name, count in counts.items()
        if count >= 2 and len(name) >= 5
    }
    alias_options = {**profile.operator_aliases, **profile.idiom_aliases}
    alias_options.update(candidates_by_name)
    present = {value for _, value, _, _ in leaves}
    alias_options = {
        source: choices
        for source, choices in alias_options.items()
        if source in present
    }
    aliases: dict[str, str] = {}

    if encoder is None:
        used_aliases: set[str] = set()
        for source in sorted(
            alias_options,
            key=lambda name: (-counts.get(name, 1) * len(name), name),
        ):
            alias = next(
                (
                    candidate
                    for candidate in alias_options[source]
                    if candidate not in used_aliases
                    and candidate not in reserved
                ),
                None,
            )
            if alias is not None:
                aliases[source] = alias
                used_aliases.add(alias)
        return aliases

    while True:
        current_cost = len(encoder.encode(_view(leaves, profile, aliases).text))
        best: tuple[int, str, str] | None = None
        used_aliases = set(aliases.values())
        for name in sorted(alias_options):
            current_alias = aliases.get(name)
            if current_alias is not None:
                trial = dict(aliases)
                del trial[name]
                trial_cost = len(encoder.encode(_view(leaves, profile, trial).text))
                choice = (trial_cost, name, "")
                if trial_cost < current_cost and (best is None or choice < best):
                    best = choice
            for alias in alias_options[name]:
                if (
                    alias == current_alias
                    or alias in used_aliases - {current_alias}
                    or (alias in reserved and alias != name)
                ):
                    continue
                trial = {**aliases, name: alias}
                trial_cost = len(encoder.encode(_view(leaves, profile, trial).text))
                choice = (trial_cost, name, alias)
                if trial_cost < current_cost and (best is None or choice < best):
                    best = choice
        if best is None:
            return aliases
        _, name, alias = best
        if alias:
            aliases[name] = alias
        else:
            del aliases[name]


def generate_ai_view_with_map(
    tree: CSTNode | Module,
    profile: TokenizerProfile | None = None,
    encoder: Encoder | None = None,
) -> AIView:
    """Generate a compact view and precise source/view character-range mappings.

    When an encoder is provided, a deterministic coordinate-descent pass measures
    the complete output, including its alias-expansion header, and keeps only changes
    that reduce token count. Without one, the profile's ordered alias candidates and
    stable frequency ordering provide a deterministic fallback, not an optimality
    guarantee.
    """
    profile = profile or default_profile()
    if isinstance(tree, Module):
        tree = parse(render_module(tree))
    leaves = _collect_leaves(tree)
    aliases = _select_aliases(leaves, profile, encoder)
    return _view(leaves, profile, aliases)


def generate_ai_view(
    tree: CSTNode | Module,
    profile: TokenizerProfile | None = None,
    encoder: Encoder | None = None,
) -> str:
    """Generate a deterministic compact view optimized for the given profile."""
    return generate_ai_view_with_map(tree, profile, encoder).text


def generate_ai_view_from_file(
    path: str | Path,
    profile: TokenizerProfile | None = None,
    encoder: Encoder | None = None,
) -> str:
    """Parse a source file and generate its profile-specific AI view."""
    result = parse_source_file(path)
    errors = [diagnostic for diagnostic in result.diagnostics if diagnostic.severity == "error"]
    if errors:
        joined = "; ".join(
            f"{item.code} line {item.span.line}: {item.message}" for item in errors
        )
        raise ValueError(f"Cannot generate AI view: {joined}")
    return generate_ai_view(result.ast, profile, encoder)
