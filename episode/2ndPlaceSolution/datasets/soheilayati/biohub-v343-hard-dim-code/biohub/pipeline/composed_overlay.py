"""The V35 composed additive overlay, shared by the OOF rig and the kernel.

Composition order is frozen (Progress/v35-probe-program.md work item 2):

1. ``phase19 = seg(v21, v24)`` — the deployed Phase-19 base;
2. ``c1 = seg(phase19, stage6)`` — the independent strong-detector witness;
3. ``witness = seg(seg(v25, v21), v24)`` — the recovered-V25 witness;
4. ``final = seg(c1, witness)``.

Every pass is ``add_full_length_segments`` with its frozen constants; action
selection always runs on the raw graphs. When ``finals`` supplies
coordinate-blended variants of the four bank graphs, each pass is repeated
with ``output_base``/``output_witness`` so the returned graph carries blended
coordinates while the selected actions stay byte-identical to the raw path.
"""

from __future__ import annotations

from typing import Any

from biohub.io.submission import DatasetGraph
from biohub.pipeline.additive_segments import add_full_length_segments


def _dual(
    base_raw: DatasetGraph,
    witness_raw: DatasetGraph,
    base_out: DatasetGraph | None,
    witness_out: DatasetGraph | None,
) -> tuple[DatasetGraph, DatasetGraph, dict[str, Any]]:
    """One segment pass on the raw surface, optionally with blended output."""
    raw, audit = add_full_length_segments(base_raw, witness_raw)
    if base_out is None:
        return raw, raw, audit
    out, audit_out = add_full_length_segments(
        base_raw, witness_raw, output_base=base_out, output_witness=witness_out,
    )
    for key in ("segments", "added_nodes", "added_edges"):
        if audit_out[key] != audit[key]:
            raise RuntimeError(f"raw/output action divergence on {key}")
    if set(out.nodes) != set(raw.nodes) or len(out.edges) != len(raw.edges):
        raise RuntimeError("raw/output graph structure divergence")
    return raw, out, audit


def compose_overlay(
    v21: DatasetGraph,
    v24: DatasetGraph,
    v25: DatasetGraph,
    stage6: DatasetGraph,
    finals: dict[str, DatasetGraph] | None = None,
) -> tuple[DatasetGraph, DatasetGraph, dict[str, Any]]:
    """Return ``(final_raw, final_output, audits)`` for one movie.

    ``finals``, when given, maps ``{"v21", "v24", "v25", "stage6"}`` to the
    coordinate-blended variants of the same graphs (same node ids and edges).
    """
    f = finals or {}
    p19_raw, p19_out, audit_p19 = _dual(
        v21, v24, f.get("v21"), f.get("v24"))
    c1_raw, c1_out, audit_s6 = _dual(
        p19_raw, stage6, p19_out if f else None, f.get("stage6"))
    m_raw, m_out, audit_w21 = _dual(
        v25, v21, f.get("v25"), f.get("v21"))
    wit_raw, wit_out, audit_w24 = _dual(
        m_raw, v24, m_out if f else None, f.get("v24"))
    final_raw, final_out, audit_final = _dual(
        c1_raw, wit_raw, c1_out if f else None, wit_out if f else None)

    for name, audit in (
        ("phase19", audit_p19), ("stage6", audit_s6), ("final", audit_final),
    ):
        if not (
            audit["original_nodes_preserved"]
            and audit["original_edges_preserved"]
            and audit["division_forks_preserved"]
        ):
            raise RuntimeError(f"{name} pass violated incumbent preservation")

    audits = {
        "phase19": audit_p19,
        "stage6_pass": audit_s6,
        "witness_v21_pass": audit_w21,
        "witness_v24_pass": audit_w24,
        "final_pass": audit_final,
    }
    return final_raw, final_out, audits
