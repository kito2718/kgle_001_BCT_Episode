"""Phase 0: reproducible, leakage-audited official evaluation."""

from biohub.phase0.clusters import build_acquisition_clusters
from biohub.phase0.graph_control import build_backward_flow_graph
from biohub.phase0.evaluation import (
    audit_oof_manifest,
    enrich_summary,
    graphs_from_submission,
    load_fold_map,
    score_submission,
    summarise_groups,
)
from biohub.phase0.hashing import hash_artifact, sha256_file
from biohub.phase0.provenance import OFFICIAL_SCORER_COMMIT, verify_official_scorer

__all__ = [
    "OFFICIAL_SCORER_COMMIT",
    "audit_oof_manifest",
    "build_acquisition_clusters",
    "build_backward_flow_graph",
    "enrich_summary",
    "graphs_from_submission",
    "hash_artifact",
    "load_fold_map",
    "score_submission",
    "sha256_file",
    "summarise_groups",
    "verify_official_scorer",
]
