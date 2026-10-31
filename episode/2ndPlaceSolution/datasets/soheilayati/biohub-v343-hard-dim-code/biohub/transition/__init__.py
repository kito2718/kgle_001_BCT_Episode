"""Joint cell-state and temporal-transition inference for the V34 redesign."""

from biohub.transition.decoder import (
    DecodeConfig,
    DecodeResult,
    decode_graph,
    decode_ordinary_graph,
)
from biohub.transition.distillation import (
    DistilledTrajectoryEvents,
    distill_ordinary_trajectory_events,
    edge_membership,
)
from biohub.transition.features import (
    FEATURE_COLUMNS,
    CandidateContext,
    build_candidate_context,
    build_transition_features,
)
from biohub.transition.graph import (
    TransitionGraph,
    build_duplicate_pairs,
    build_local_transition_edges,
)
from biohub.transition.model import TransitionEnergyMLP, listwise_null_loss
from biohub.transition.supervision import CandidateSupervision, label_candidate_graph
from biohub.transition.windowed import (
    WindowedDecodeConfig,
    decode_windowed,
    division_component_nodes,
    owner_windows,
    window_starts,
)
from biohub.transition.augmentation import apply_anchored_actions
from biohub.transition.state_supervision import (
    STATE_FEATURE_COLUMNS,
    STATE_OFFSETS,
    StateRows,
    best_edge_by_group,
    build_state_feature_matrix,
    choose_dark_background,
    choose_local_alternatives,
    trace_candidate_paths,
)
from biohub.transition.state_model import STATE_ARCHITECTURE, make_candidate_state_model
from biohub.transition.factorization import (
    ObjectiveAlignedFactors,
    ExistencePreservingFactors,
    calibrate_log_odds,
    conditional_child_log_probability,
    conditional_parent_log_probability,
    factor_objective_aligned_scores,
    grouped_log_probability,
    grouped_logsumexp,
    log_sigmoid,
    reduced_edge_utility,
    symmetric_transition_log_compatibility,
    existence_preserving_edge_utility,
)
from biohub.transition.event_model import CorrectedEventMLP, EVENT_FEATURE_COLUMNS
from biohub.transition.successor_supervision import (
    SuccessorRows,
    label_ordinary_successor_lists,
)
from biohub.transition.successor_model import (
    SuccessorEnergyMLP,
    grouped_softmax_loss,
)

__all__ = [
    "CandidateSupervision",
    "CandidateContext",
    "DecodeConfig",
    "DecodeResult",
    "TransitionGraph",
    "TransitionEnergyMLP",
    "WindowedDecodeConfig",
    "FEATURE_COLUMNS",
    "build_candidate_context",
    "build_duplicate_pairs",
    "build_local_transition_edges",
    "build_transition_features",
    "decode_graph",
    "decode_ordinary_graph",
    "decode_windowed",
    "division_component_nodes",
    "label_candidate_graph",
    "listwise_null_loss",
    "owner_windows",
    "window_starts",
    "apply_anchored_actions",
    "STATE_FEATURE_COLUMNS",
    "STATE_OFFSETS",
    "StateRows",
    "best_edge_by_group",
    "build_state_feature_matrix",
    "choose_dark_background",
    "choose_local_alternatives",
    "trace_candidate_paths",
    "STATE_ARCHITECTURE",
    "make_candidate_state_model",
    "ObjectiveAlignedFactors",
    "ExistencePreservingFactors",
    "calibrate_log_odds",
    "conditional_child_log_probability",
    "conditional_parent_log_probability",
    "factor_objective_aligned_scores",
    "grouped_log_probability",
    "grouped_logsumexp",
    "log_sigmoid",
    "reduced_edge_utility",
    "symmetric_transition_log_compatibility",
    "existence_preserving_edge_utility",
    "CorrectedEventMLP",
    "EVENT_FEATURE_COLUMNS",
    "SuccessorRows",
    "label_ordinary_successor_lists",
    "SuccessorEnergyMLP",
    "grouped_softmax_loss",
]
