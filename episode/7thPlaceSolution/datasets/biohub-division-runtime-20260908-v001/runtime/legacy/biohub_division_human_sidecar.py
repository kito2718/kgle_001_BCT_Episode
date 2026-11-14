#!/usr/bin/env python3
"""Dependency-light validation for Division human-event sidecars."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def read_human_event_rows(
    path: Path,
    evaluation_fold: int,
    *,
    require_both_classes: bool = True,
) -> list[dict[str, str]]:
    """Read and revalidate event-only human labels before training uses them."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seen: set[str] = set()
    targets: Counter[int] = Counter()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValueError(f"duplicate human sample_id: {sample_id}")
        seen.add(sample_id)
        if row.get("split") != "human_train" or row.get("event_mask") != "1":
            raise ValueError(f"{sample_id}: event-only sidecar requires split=human_train and event_mask=1")
        if row.get("training_use_source") != "explicit_adjudication":
            raise ValueError(f"{sample_id}: non-explicit training use is forbidden")
        source_fold = int(row["source_fold"])
        if source_fold == evaluation_fold:
            raise ValueError(f"{sample_id}: source_fold leaks into evaluation fold {evaluation_fold}")
        target = int(row["target"])
        if target not in {0, 1} or int(row["label"]) != target:
            raise ValueError(f"{sample_id}: invalid or inconsistent event target")
        frame = int(row["frame"])
        expected_case = f"{row['stem']}_t{frame:03d}"
        if row["case_id"] != expected_case:
            raise ValueError(f"{sample_id}: case_id mismatch {row['case_id']!r} != {expected_case!r}")
        targets[target] += 1
    if not rows or (require_both_classes and (targets[0] == 0 or targets[1] == 0)):
        raise ValueError(f"human sidecar must contain both classes, got {dict(targets)}")
    return rows
