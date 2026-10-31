"""Apply the frozen V233 policy with one CV/deployment coordinate contract."""
import hashlib
import numpy as np
from biohub.pipeline.coordinate_contract import finalize_cv_graph
from biohub.pipeline.motion_cell_substitution import propose_substitutions, materialize_substitutions


def node_digest(graph):
    rows = np.asarray([(int(n), *point) for n, point in sorted(graph.nodes.items())], dtype='<i8')
    return hashlib.sha256(rows.tobytes()).hexdigest()


def edge_digest(graph):
    rows = np.asarray([(p, c) for p, c, _ in sorted(graph.edges, key=lambda e: -e[2])], dtype='<i8').reshape(-1, 2)
    return hashlib.sha256(rows.tobytes()).hexdigest()


def apply_motion_substitution(raw, output, ids, nodes, backward, forward, candidates):
    finalize_cv_graph(raw, output)
    before_nodes, before_edges = node_digest(raw), edge_digest(raw)
    actions, proposals = propose_substitutions(raw, ids, nodes, backward, forward, candidates)
    new_raw, controls = materialize_substitutions(raw, actions)
    new_output, output_controls = materialize_substitutions(output, actions)
    finalize_cv_graph(new_raw, new_output)
    if edge_digest(new_raw) != before_edges or controls != output_controls:
        raise RuntimeError('V233 raw/output topology or controls differ')
    return new_raw, new_output, dict(
        policy='frozen_v233_two_anchor_raw_substitution', coordinate_policy='v250_exact_cv_coordinates',
        input_node_sha256=before_nodes, output_node_sha256=node_digest(new_raw),
        ordered_edge_sha256=before_edges, actions=actions, proposals=proposals, controls=controls)
