"""Safe selective extraction helpers for large external-data archives."""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path


def extract_matching_zip(
    archive_path: str | Path,
    output_root: str | Path,
    predicate: Callable[[str], bool],
) -> list[Path]:
    """Extract matching regular files while rejecting path traversal."""
    archive_path = Path(archive_path)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            item for item in archive.infolist()
            if not item.is_dir() and predicate(item.filename)]
        for item in members:
            target = (output_root / item.filename).resolve()
            if output_root not in target.parents:
                raise ValueError(f"unsafe archive member {item.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            extracted.append(target)
    return extracted
