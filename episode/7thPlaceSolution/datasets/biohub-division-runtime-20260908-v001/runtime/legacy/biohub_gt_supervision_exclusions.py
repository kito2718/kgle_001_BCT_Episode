#!/usr/bin/env python3
"""Shared, edge-specific masks for audited Biohub GT supervision.

The registry intentionally distinguishes a questionable lineage edge from its
endpoint nodes.  A suspect identity link invalidates motion/association/path
supervision that traverses the link, but does not by itself prove that either
cell centre is an invalid detection target.
"""

from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class SuspectTransition:
    stem: str
    source_t: int
    target_t: int
    source_id: int
    target_id: int
    source_zyx: tuple[int, int, int]
    target_zyx: tuple[int, int, int]


class GTSupervisionExclusions:
    def __init__(self, transitions: Iterable[SuspectTransition]):
        self.transitions = tuple(transitions)
        self._edge_ids = {
            (row.stem, row.source_id, row.target_id) for row in self.transitions
        }

    @classmethod
    def from_path(cls, path: Path | str) -> "GTSupervisionExclusions":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("version", -1)) != 1:
            raise ValueError(f"unsupported GT exclusion manifest version: {payload.get('version')}")
        rows = []
        for raw in payload.get("transitions", []):
            rows.append(
                SuspectTransition(
                    stem=str(raw["stem"]),
                    source_t=int(raw["source_t"]),
                    target_t=int(raw["target_t"]),
                    source_id=int(raw["source_id"]),
                    target_id=int(raw["target_id"]),
                    source_zyx=tuple(int(value) for value in raw["source_zyx"]),
                    target_zyx=tuple(int(value) for value in raw["target_zyx"]),
                )
            )
        if not rows:
            raise ValueError(f"GT exclusion manifest contains no transitions: {path}")
        return cls(rows)

    def excludes_edge_ids(self, stem: str, source_id: int, target_id: int) -> bool:
        return (str(stem), int(source_id), int(target_id)) in self._edge_ids

    def filtered_predecessors(
        self,
        stem: str,
        target_id: int,
        predecessors: Iterable[int],
    ) -> tuple[int, ...]:
        return tuple(
            int(source_id)
            for source_id in predecessors
            if not self.excludes_edge_ids(stem, int(source_id), int(target_id))
        )

    def has_excluded_outgoing(
        self,
        stem: str,
        source_id: int,
        children: Mapping[int, Iterable[int]],
    ) -> bool:
        return any(
            self.excludes_edge_ids(stem, int(source_id), int(target_id))
            for target_id in children.get(int(source_id), ())
        )

    def path_traverses_excluded_edge(
        self,
        stem: str,
        children: Mapping[int, Iterable[int]],
        source_id: int,
        target_id: int,
        max_steps: int,
    ) -> bool:
        """Return true when a source-to-target path uses a suspect edge."""

        frontier = [(int(source_id), False, 0)]
        seen: set[tuple[int, bool, int]] = set()
        while frontier:
            node, used_exclusion, depth = frontier.pop()
            state = (node, used_exclusion, depth)
            if state in seen:
                continue
            seen.add(state)
            if node == int(target_id):
                return used_exclusion
            if depth >= int(max_steps):
                continue
            for child in children.get(node, ()):
                child = int(child)
                frontier.append(
                    (
                        child,
                        used_exclusion or self.excludes_edge_ids(stem, node, child),
                        depth + 1,
                    )
                )
        return False


def load_optional(path: Path | str | None) -> GTSupervisionExclusions | None:
    return None if path is None or not str(path).strip() else GTSupervisionExclusions.from_path(path)


def resolve_configured_manifest(path: Path | str | None = None) -> Path | None:
    """Resolve explicit, environment, then repository-default manifest."""

    if path is not None and str(path).strip():
        return Path(path)
    configured = os.environ.get("BIOHUB_GT_EXCLUSION_MANIFEST", "").strip()
    if configured:
        return Path(configured)
    repository_default = (
        Path(__file__).resolve().parents[1]
        / "gpu/manifests/biohub_suspect_gt_supervision_v001.json"
    )
    return repository_default if repository_default.is_file() else None


def manifest_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract_metadata(path: Path | str) -> dict[str, object]:
    resolved = Path(path).resolve()
    exclusions = GTSupervisionExclusions.from_path(resolved)
    return {
        "version": 1,
        "manifest": str(resolved),
        "manifest_sha256": manifest_sha256(resolved),
        "transition_count": len(exclusions.transitions),
        "node_detection_loss_policy": "keep",
        "transition_dependent_loss_policy": "mask",
    }
