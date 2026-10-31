"""Deterministic hashes for files and directory-shaped prediction artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of the exact bytes at *path*."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def hash_artifact(path: str | Path) -> dict[str, Any]:
    """Hash one file or a directory tree without depending on timestamps.

    Directory hashes cover each relative POSIX path, byte length, and file
    digest in sorted order. This works for Zarr/GEFF directories as well as a
    directory of per-sample parquet predictions.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return {
            "path": str(path),
            "kind": "file",
            "files": 1,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    files = sorted(p for p in path.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    entries = []
    for item in files:
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        file_hash = sha256_file(item)
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        entries.append({"path": relative, "bytes": size, "sha256": file_hash})
    return {
        "path": str(path),
        "kind": "directory",
        "files": len(files),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "entries": entries,
    }


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Write canonical, stable UTF-8 JSON atomically."""
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    atomic_write_bytes(path, content)
