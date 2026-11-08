"""Deterministic, tokenizer-profiled AI view generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

from src.frontend import Module, parse_source_file, render_module
from src.parser import CSTNode, parse, tokenize


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

    def lexical_tokens(text: str) -> tuple[tuple[str, str], ...]:
        return tuple(
            (token.kind, token.value)
            for token in tokenize(text)
            if token.kind not in {"WHITESPACE", "NEWLINE", "COMMENT"}
        )

    declarations = sorted(aliases.items(), key=lambda item: item[1])
    append(f"@a({profile.name};")
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
            joined_tokens = lexical_tokens(previous_rendered + rendered)
            separate_tokens = lexical_tokens(previous_rendered) + lexical_tokens(rendered)
            if (
                previous_is_word
                and current_is_word
            ) or previous_rendered + rendered in _MERGING_OPERATORS or (
                joined_tokens != separate_tokens
            ):
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


def expand_ai_view(
    text: str,
    profile: TokenizerProfile | None = None,
) -> str:
    """Expand a profile-tagged `.ai` view back to compact Aether source text."""
    profile = profile or default_profile()
    if not text.startswith("@a("):
        raise ValueError("AI view is missing its @a expansion header")
    header_end = text.find(")", 3)
    if header_end < 0:
        raise ValueError("AI view has an unterminated @a expansion header")
    header = text[3:header_end]
    profile_name, separator, declarations = header.partition(";")
    if not separator or profile_name != profile.name:
        raise ValueError(
            f"AI view profile header does not match {profile.name!r}"
        )
    body = text[header_end + 1 :]
    if "@a(" in body:
        raise ValueError("AI view contains an unexpected second @a header")

    alias_to_source: dict[str, str] = {}
    source_to_alias: dict[str, str] = {}
    known_aliases = {
        **profile.operator_aliases,
        **profile.idiom_aliases,
        **profile.semantic_aliases,
    }
    for declaration in declarations.split(",") if declarations else ():
        alias, equals, source = declaration.partition("=")
        if not equals or not alias or not source:
            raise ValueError("AI view contains a malformed alias declaration")
        if alias in alias_to_source:
            raise ValueError(f"AI view contains duplicate alias {alias!r}")
        if source in source_to_alias:
            raise ValueError(f"AI view contains duplicate source alias {source!r}")
        alias_tokens = tuple(
            token.value
            for token in tokenize(alias)
            if token.kind not in {"WHITESPACE", "NEWLINE", "COMMENT"}
        )
        source_token_list = tuple(
            token
            for token in tokenize(source)
            if token.kind not in {"WHITESPACE", "NEWLINE", "COMMENT"}
        )
        if (
            not alias_tokens
            or not source_token_list
            or "".join(alias_tokens) != alias
        ):
            raise ValueError(f"AI view contains an invalid alias {alias!r}")
        if (
            len(source_token_list) != 1
            or source_token_list[0].value != source
        ):
            raise ValueError(
                f"AI view source alias {source!r} must be a single Aether token"
            )
        choices = known_aliases.get(source)
        if choices is not None:
            if alias not in choices:
                raise ValueError(
                    f"AI view alias {alias!r} is not valid for {source!r}"
                )
        elif (
            source_token_list[0].kind != "IDENTIFIER"
            or alias not in profile.alias_candidates
        ):
            raise ValueError(
                f"AI view alias {alias!r} is not valid for source identifier {source!r}"
            )
        alias_to_source[alias] = source
        source_to_alias[source] = alias

    tokenized_aliases = sorted(
        (
            (
                tuple(
                    token.value
                    for token in tokenize(alias)
                    if token.kind not in {"WHITESPACE", "NEWLINE", "COMMENT"}
                ),
                source,
            )
            for alias, source in alias_to_source.items()
        ),
        key=lambda item: (-len(item[0]), item[0]),
    )
    tokens = tokenize(body)
    expanded_tokens: list[tuple[str, str]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.kind in {"WHITESPACE", "NEWLINE", "COMMENT"}:
            index += 1
            continue
        replacement = None
        replacement_end = index
        for alias_tokens, source in tokenized_aliases:
            matched_indexes: list[int] = []
            candidate_index = index
            for alias_token in alias_tokens:
                if (
                    candidate_index >= len(tokens)
                    or tokens[candidate_index].value != alias_token
                ):
                    break
                matched_indexes.append(candidate_index)
                candidate_index += 1
            if len(matched_indexes) != len(alias_tokens):
                continue
            if any(
                tokens[left].end != tokens[right].start
                for left, right in zip(matched_indexes, matched_indexes[1:])
            ):
                continue
            replacement = source
            replacement_end = matched_indexes[-1]
            break
        if replacement is None:
            expanded_tokens.append((token.value, token.kind))
            index += 1
            continue
        expanded_tokens.extend(
            (source_token.value, source_token.kind)
            for source_token in tokenize(replacement)
            if source_token.kind not in {"WHITESPACE", "NEWLINE", "COMMENT"}
        )
        index = replacement_end + 1
    compact: list[str] = []
    previous_value = ""
    previous_is_word = False
    for value, kind in expanded_tokens:
        current_is_word = kind in _WORD_KINDS
        if compact:
            merged = [
                token
                for token in tokenize(previous_value + value)
                if token.kind not in {"WHITESPACE", "NEWLINE", "COMMENT"}
            ]
            if (
                previous_is_word and current_is_word
                or len(merged) != 2
                or merged[0].value != previous_value
                or merged[1].value != value
            ):
                compact.append(" ")
        compact.append(value)
        previous_value = value
        previous_is_word = current_is_word
    return "".join(compact)


def ai_view_to_source(
    text: str,
    profile: TokenizerProfile | None = None,
) -> str:
    """Expand an edited `.ai` view and render it as canonical `.hl` source."""
    from src.frontend import parse_source, render_module

    expanded = expand_ai_view(text, profile)
    result = parse_source(expanded)
    errors = [item for item in result.diagnostics if item.severity == "error"]
    if errors:
        joined = "; ".join(
            f"{item.code} line {item.span.line}: {item.message}" for item in errors
        )
        raise ValueError(f"Edited AI view is not valid Aether source: {joined}")
    return render_module(result.ast)
