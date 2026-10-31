"""Pinned organizer-scorer provenance and runtime integrity verification."""

from __future__ import annotations

import inspect
import hashlib
from pathlib import Path

OFFICIAL_SCORER_REPOSITORY = (
    "https://github.com/royerlab/kaggle-cell-tracking-competition"
)
OFFICIAL_SCORER_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
OFFICIAL_SCORER_COMMIT_DATE = "2026-07-17T17:03:46-07:00"
# SHA-256 after CRLF/CR -> LF normalization. Git/checkouts may materialize the
# organizer's text with different line endings; normalized bytes must still be
# identical. The upstream Git blob IDs below retain a second provenance anchor.
OFFICIAL_SCORER_FILES = {
    "__init__.py": "f9df2098dd85934d0b0d48d477420b1f9ed20a9a55ca6294a1874b964e9166e6",
    "metrics.py": "cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444",
    "division_metrics.py": "0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9",
    "LICENSE": "3910a8b578783928cbbf981c1b204591ab3368e70017dc695d6536f71d272796",
    "METRIC_SPEC.md": "d8a2d3ff21b507242dd23e9b7c687d725261d7b4c79cff086cccb4093141d696",
}
OFFICIAL_SCORER_GIT_BLOBS = {
    "__init__.py": "943b14d7fbd618490528ce4c112dcc32aca22769",
    "metrics.py": "e536cdc9f0877542ab227ec701ef0fdbb667189a",
    "division_metrics.py": "9afa8630f3d7a294f9a25fca81cce1e0c7c7aeca",
    "LICENSE": "901f81c0982414b9459397210c17048f3ea70ddf",
    "METRIC_SPEC.md": "df84639194776282af9b788828ad6a9b870e19bb",
}


def _canonical_text_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def verify_official_scorer() -> dict:
    """Fail if imports resolve outside the vendored, byte-pinned scorer."""
    import tracking_cellmot

    package_dir = Path(inspect.getfile(tracking_cellmot)).resolve().parent
    actual = {}
    mismatches = {}
    for relative, expected in OFFICIAL_SCORER_FILES.items():
        path = package_dir / relative
        if not path.is_file():
            mismatches[relative] = {"expected": expected, "actual": None}
            continue
        value = _canonical_text_sha256(path)
        actual[relative] = value
        if value != expected:
            mismatches[relative] = {"expected": expected, "actual": value}
    if mismatches:
        raise RuntimeError(f"official scorer integrity check failed: {mismatches}")
    return {
        "repository": OFFICIAL_SCORER_REPOSITORY,
        "commit": OFFICIAL_SCORER_COMMIT,
        "commit_date": OFFICIAL_SCORER_COMMIT_DATE,
        "package_dir": str(package_dir),
        "canonical_text_sha256": actual,
        "upstream_git_blobs": OFFICIAL_SCORER_GIT_BLOBS,
    }
