from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_vocab(vocab_path: str | Path) -> list[str]:
    p = Path(vocab_path)
    with p.open("r", encoding="utf-8") as f:
        tokens = json.load(f)
    if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) for t in tokens):
        raise ValueError("Invalid vocab JSON")
    return tokens


def load_aliases(path: Optional[str | Path]) -> dict[str, list[str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Alias file must be a JSON object")
    out: dict[str, list[str]] = {}
    for canonical, variants in raw.items():
        if not isinstance(canonical, str) or not isinstance(variants, list):
            continue
        out[canonical] = [str(v) for v in variants]
    return out


_WS_RE = re.compile(r"\s+")


def normalize_gloss(gloss: str) -> str:
    """Normalize dataset gloss into a comparable token.

    We keep it conservative:
    - trim
    - collapse whitespace
    - replace separators with underscore
    - uppercase
    """
    g = gloss.strip()
    g = _WS_RE.sub(" ", g)
    g = g.replace("/", " ").replace("+", " ").replace("-", " ")
    g = _WS_RE.sub("_", g)
    return g.upper()


def build_alias_lookup(aliases: dict[str, list[str]]) -> dict[str, str]:
    """Return mapping from normalized variant -> CANONICAL_TOKEN."""
    lookup: dict[str, str] = {}
    for canonical, variants in aliases.items():
        canon_norm = normalize_gloss(canonical)
        lookup[canon_norm] = canon_norm
        for v in variants:
            lookup[normalize_gloss(v)] = canon_norm
    return lookup


@dataclass(frozen=True)
class GlossMatch:
    token: str
    match_type: str  # 'exact' | 'alias'


def resolve_to_vocab_token(gloss: str, *, vocab_set: set[str], alias_lookup: dict[str, str]) -> Optional[str]:
    match = resolve_to_vocab_match(gloss, vocab_set=vocab_set, alias_lookup=alias_lookup)
    return match.token if match else None


def resolve_to_vocab_match(gloss: str, *, vocab_set: set[str], alias_lookup: dict[str, str]) -> Optional[GlossMatch]:
    norm = normalize_gloss(gloss)
    if norm in vocab_set:
        return GlossMatch(token=norm, match_type="exact")
    alias = alias_lookup.get(norm)
    if alias and alias in vocab_set:
        return GlossMatch(token=alias, match_type="alias")
    return None


def get_hf_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    # Common env vars
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if env:
        return env
    # huggingface-cli login stores token locally; detect without printing it.
    try:
        from huggingface_hub import HfFolder

        t = HfFolder.get_token()
        if t:
            return t
    except Exception:
        return None
    return None


def require_hf_token(cli_token: Optional[str], *, dataset_id: Optional[str] = None) -> str:
    token = get_hf_token(cli_token)
    if token:
        return token
    ds_msg = f" for dataset '{dataset_id}'" if dataset_id else ""
    raise SystemExit(
        "Missing HuggingFace token" + ds_msg + ".\n"
        "Provide via one of:\n"
        "- env: set HF_TOKEN\n"
        "- CLI: --hf-token <token>\n"
        "- login: run 'huggingface-cli login'\n"
        "\nNote: token will never be printed by this tool."
    )


def validate_hf_token(token: str) -> None:
    """Validate token without logging it."""
    try:
        from huggingface_hub import HfApi

        _ = HfApi().whoami(token=token)
    except Exception:
        raise SystemExit(
            "HuggingFace token validation failed.\n"
            "- Check the token is correct\n"
            "- Ensure it has access to the gated dataset\n"
            "- Or run 'huggingface-cli login' and retry"
        )


@dataclass(frozen=True)
class VideoRecord:
    video_path: Path
    gloss: str
    token: str
    split: str
    signer_id: Optional[str] = None
