#!/usr/bin/env python3
"""Shared fail-closed cohort policy for Biohub ILP screens."""

from __future__ import annotations

from pathlib import Path


DEFAULT_REP20 = (
    Path(__file__).resolve().parents[1]
    / "gpu"
    / "manifests"
    / "representative20_fold0_stems_v001.txt"
)


def add_scope_arguments(parser) -> None:
    parser.add_argument(
        "--evaluation-scope",
        choices=("screen20", "full40", "custom"),
        default="screen20",
        help=(
            "screen20 is the project default and uses the fixed Fold0 manifest; "
            "full40 must be requested explicitly for production/submission confirmation."
        ),
    )
    parser.add_argument(
        "--stem-file",
        type=Path,
        help="Explicit newline-delimited stem manifest (required for custom scope).",
    )


def read_stem_file(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"stem manifest missing: {path}")
    stems = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not stems or len(stems) != len(set(stems)):
        raise ValueError(f"stem manifest must be non-empty and unique: {path}")
    return stems


def resolve_scope_stems(
    available_stems: list[str] | set[str],
    *,
    evaluation_scope: str,
    fold: int,
    case_partition: str,
    stem_file: Path | None = None,
) -> list[str]:
    available = set(available_stems)
    if evaluation_scope == "full40":
        if stem_file is not None:
            raise ValueError("full40 cannot be combined with --stem-file")
        return sorted(available)

    if evaluation_scope == "screen20":
        if fold != 0 or case_partition != "val":
            raise ValueError(
                "screen20 is defined only for Fold0 val; use custom or explicit full40"
            )
        manifest = stem_file or DEFAULT_REP20
        selected = read_stem_file(manifest)
        if len(selected) != 20:
            raise ValueError(f"screen20 manifest must contain exactly 20 stems: {manifest}")
    elif evaluation_scope == "custom":
        selected = read_stem_file(stem_file) if stem_file is not None else sorted(available)
    else:
        raise ValueError(f"unknown evaluation scope: {evaluation_scope}")

    unknown = set(selected) - available
    if unknown:
        raise ValueError(f"stem manifest contains non-partition stems: {sorted(unknown)}")
    return selected
