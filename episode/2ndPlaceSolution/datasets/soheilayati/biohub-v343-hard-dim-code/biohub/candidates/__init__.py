"""Where the candidate pool lives.

Stage 1 built a measuring rig — the oracle ladder, the error taxonomy, the
count predictor, the deployability check — and every part of it reads a
directory of per-dataset candidate tables. Stage 2's entire job is to produce a
*better* such directory, so the rig has to point at either one without editing
eight scripts.

Resolution order, so a worker process spawned by `multiprocessing` sees the same
pool as its parent (on Windows the child re-imports this module rather than
inheriting globals, so an environment variable is the only thing that survives):

    explicit argument  ->  $BIOHUB_CANDIDATES  ->  candidates_v1

A bare name resolves under `artifacts/`; a path is used as given.

    python scripts/oracle_candidates.py --candidates candidates_v2
    BIOHUB_CANDIDATES=candidates_v2 python scripts/error_taxonomy.py
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO / "artifacts"
DEFAULT_POOL = "candidates_v1"
ENV_VAR = "BIOHUB_CANDIDATES"

# Columns every pool must provide. A Stage 2 detector writing this schema works
# with the whole rig unchanged; `score` is whatever that detector ranks by.
REQUIRED_COLUMNS = ("t", "z", "y", "x", "score")


def pool_dir(name: str | Path | None = None) -> Path:
    """Resolve the candidate pool directory."""
    name = name or os.environ.get(ENV_VAR) or DEFAULT_POOL
    p = Path(name)
    if p.is_absolute() or len(p.parts) > 1:
        return p
    return ARTIFACTS / p


def use_pool(name: str | Path | None) -> Path:
    """Select the pool for this process *and its children*. Returns the path."""
    if name:
        os.environ[ENV_VAR] = str(name)
    return pool_dir()


def add_pool_argument(parser) -> None:
    """Standard `--candidates` flag. Call `use_pool(args.candidates)` in main()."""
    parser.add_argument(
        "--candidates",
        default=None,
        metavar="NAME_OR_PATH",
        help=f"candidate pool to read (default: {DEFAULT_POOL}). A bare name "
             f"resolves under artifacts/. Also settable via ${ENV_VAR}.",
    )


def check_pool(names: list[str], directory: Path | None = None) -> Path:
    """Fail loudly and early if the pool is missing samples."""
    directory = directory or pool_dir()
    if not directory.exists():
        raise SystemExit(
            f"candidate pool not found: {directory}\n"
            f"run: python scripts/cache_candidates.py --out {directory}"
        )
    missing = [n for n in names if not (directory / f"{n}.parquet").exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(names)} samples missing from {directory} "
            f"(e.g. {missing[0]}). Run scripts/cache_candidates.py first."
        )
    return directory
