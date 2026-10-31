"""Objective-aligned probabilistic factors for joint ordinary-lineage decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObjectiveAlignedFactors:
    """Calibrated node/event potentials in the decoder's additive energy system."""

    state_log_odds: np.ndarray
    incoming_log_odds: np.ndarray
    outgoing_log_odds: np.ndarray
    birth_score: np.ndarray
    death_score: np.ndarray
    conditional_parent_log_probability: np.ndarray
    edge_score: np.ndarray


@dataclass(frozen=True)
class ExistencePreservingFactors:
    """B7 link-existence log odds and conditional-identity utilities."""

    link_log_odds: np.ndarray
    conditional_log_probability: np.ndarray
    reduced_edge_utility: np.ndarray


def calibrate_log_odds(
    raw_logit: np.ndarray,
    *,
    temperature: float,
    offset: float,
) -> np.ndarray:
    """Apply the frozen additive-offset/temperature calibration contract."""

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")
    values = (np.asarray(raw_logit, dtype=np.float64) + float(offset)) / float(
        temperature
    )
    if not np.isfinite(values).all():
        raise ValueError("calibrated log odds are non-finite")
    return values


def log_sigmoid(logit: np.ndarray) -> np.ndarray:
    """Numerically stable ``log(sigmoid(logit))`` in float64."""

    values = -np.logaddexp(0.0, -np.asarray(logit, dtype=np.float64))
    if not np.isfinite(values).all():
        raise ValueError("log-sigmoid produced non-finite values")
    return values


def conditional_parent_log_probability(
    edges: np.ndarray,
    transition_energy: np.ndarray,
    *,
    n_nodes: int,
) -> np.ndarray:
    """Normalize real-parent energies independently for every child candidate.

    The listwise model's learned null state is intentionally excluded. Incoming
    existence is represented by a separately calibrated event factor.
    """

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    energy = np.asarray(transition_energy, dtype=np.float64).reshape(-1)
    if len(edges) != len(energy):
        raise ValueError("edge and transition-energy lengths differ")
    if n_nodes < 0:
        raise ValueError("n_nodes must be nonnegative")
    if len(edges) == 0:
        return np.zeros(0, dtype=np.float64)
    if not np.isfinite(energy).all():
        raise ValueError("transition energy is non-finite")
    if ((edges < 0) | (edges >= int(n_nodes))).any():
        raise ValueError("edge endpoint lies outside the node universe")

    return grouped_log_probability(edges[:, 1], energy, n_groups=n_nodes)


def grouped_log_probability(
    group: np.ndarray,
    energy: np.ndarray,
    *,
    n_groups: int,
) -> np.ndarray:
    """Return a numerically stable log-softmax within each integer group."""

    group = np.asarray(group, dtype=np.int64).reshape(-1)
    energy = np.asarray(energy, dtype=np.float64).reshape(-1)
    if len(group) != len(energy):
        raise ValueError("group and energy lengths differ")
    if n_groups < 0:
        raise ValueError("n_groups must be nonnegative")
    if len(group) == 0:
        return np.zeros(0, dtype=np.float64)
    if not np.isfinite(energy).all():
        raise ValueError("grouped energy is non-finite")
    if ((group < 0) | (group >= int(n_groups))).any():
        raise ValueError("group index lies outside the group universe")

    maximum = np.full(int(n_groups), -np.inf, dtype=np.float64)
    np.maximum.at(maximum, group, energy)
    shifted = np.exp(energy - maximum[group])
    denominator = np.zeros(int(n_groups), dtype=np.float64)
    np.add.at(denominator, group, shifted)
    if (denominator[group] <= 0).any() or not np.isfinite(denominator[group]).all():
        raise ValueError("invalid grouped log-softmax normalizer")
    output = energy - maximum[group] - np.log(denominator[group])
    if not np.isfinite(output).all():
        raise ValueError("grouped log probability is non-finite")
    return output


def grouped_logsumexp(
    group: np.ndarray,
    energy: np.ndarray,
    *,
    n_groups: int,
) -> np.ndarray:
    """Return one stable log-sum-exp value for each group in the universe."""

    group = np.asarray(group, dtype=np.int64).reshape(-1)
    energy = np.asarray(energy, dtype=np.float64).reshape(-1)
    if len(group) != len(energy):
        raise ValueError("group and energy lengths differ")
    if n_groups < 0:
        raise ValueError("n_groups must be nonnegative")
    output = np.full(int(n_groups), -np.inf, dtype=np.float64)
    if len(group) == 0:
        return output
    if not np.isfinite(energy).all():
        raise ValueError("grouped energy is non-finite")
    if ((group < 0) | (group >= int(n_groups))).any():
        raise ValueError("group index lies outside the group universe")
    maximum = np.full(int(n_groups), -np.inf, dtype=np.float64)
    np.maximum.at(maximum, group, energy)
    denominator = np.zeros(int(n_groups), dtype=np.float64)
    np.add.at(denominator, group, np.exp(energy - maximum[group]))
    present = denominator > 0
    output[present] = maximum[present] + np.log(denominator[present])
    if not np.isfinite(output[present]).all():
        raise ValueError("grouped log-sum-exp is non-finite for a present group")
    return output


def existence_preserving_edge_utility(
    *,
    edges: np.ndarray,
    b0_edge_energy: np.ndarray,
    b0_null_energy: float,
    identity_log_score: np.ndarray,
    n_nodes: int,
) -> ExistencePreservingFactors:
    """Construct B7 utilities preserving B0 real-link mass per child.

    ``identity_log_score`` may use any frozen pair evidence. It is normalized
    per child, so the resulting edge-utility log-sum-exp is exactly the B0
    real-link-versus-null log odds for every child with parent options.
    """

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    b0_energy = np.asarray(b0_edge_energy, dtype=np.float64).reshape(-1)
    identity = np.asarray(identity_log_score, dtype=np.float64).reshape(-1)
    if not (len(edges) == len(b0_energy) == len(identity)):
        raise ValueError("B7 edge arrays have different lengths")
    if not np.isfinite(b0_null_energy):
        raise ValueError("B0 null energy must be finite")
    if n_nodes < 0:
        raise ValueError("n_nodes must be nonnegative")
    if len(edges) and ((edges < 0) | (edges >= int(n_nodes))).any():
        raise ValueError("edge endpoint lies outside the node universe")
    child = edges[:, 1]
    b0_real_log_mass = grouped_logsumexp(child, b0_energy, n_groups=n_nodes)
    link_log_odds = b0_real_log_mass - float(b0_null_energy)
    conditional = grouped_log_probability(child, identity, n_groups=n_nodes)
    utility = link_log_odds[child] + conditional
    if not np.isfinite(utility).all():
        raise ValueError("existence-preserving edge utility is non-finite")
    return ExistencePreservingFactors(
        link_log_odds=link_log_odds,
        conditional_log_probability=conditional,
        reduced_edge_utility=utility,
    )


def conditional_child_log_probability(
    edges: np.ndarray,
    transition_energy: np.ndarray,
    *,
    n_nodes: int,
) -> np.ndarray:
    """Normalize real-child energies independently for every parent candidate."""

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    energy = np.asarray(transition_energy, dtype=np.float64).reshape(-1)
    if len(edges) != len(energy):
        raise ValueError("edge and transition-energy lengths differ")
    if n_nodes < 0:
        raise ValueError("n_nodes must be nonnegative")
    if len(edges) == 0:
        return np.zeros(0, dtype=np.float64)
    if ((edges < 0) | (edges >= int(n_nodes))).any():
        raise ValueError("edge endpoint lies outside the node universe")
    return grouped_log_probability(edges[:, 0], energy, n_groups=n_nodes)


def symmetric_transition_log_compatibility(
    edges: np.ndarray,
    transition_energy: np.ndarray,
    *,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the frozen B4 geometric-mean compatibility and both conditionals.

    The first output is ``0.5 * (log q(parent|child) +
    log q(child|parent))``. The identical pair energy is used in both choice
    sets, so this operation introduces no fitted scale or threshold.
    """

    incoming = conditional_parent_log_probability(
        edges, transition_energy, n_nodes=n_nodes
    )
    outgoing = conditional_child_log_probability(
        edges, transition_energy, n_nodes=n_nodes
    )
    compatibility = 0.5 * (incoming + outgoing)
    if not np.isfinite(compatibility).all():
        raise ValueError("symmetric transition compatibility is non-finite")
    return compatibility, incoming, outgoing


def factor_objective_aligned_scores(
    *,
    edges: np.ndarray,
    transition_energy: np.ndarray,
    state_logit: np.ndarray,
    incoming_logit: np.ndarray,
    outgoing_logit: np.ndarray,
    state_temperature: float,
    state_offset: float,
    incoming_temperature: float,
    incoming_offset: float,
    outgoing_temperature: float,
    outgoing_offset: float,
) -> ObjectiveAlignedFactors:
    """Construct the frozen B2 state, birth, death, and edge potentials."""

    state_raw = np.asarray(state_logit, dtype=np.float64).reshape(-1)
    incoming_raw = np.asarray(incoming_logit, dtype=np.float64).reshape(-1)
    outgoing_raw = np.asarray(outgoing_logit, dtype=np.float64).reshape(-1)
    if not (len(state_raw) == len(incoming_raw) == len(outgoing_raw)):
        raise ValueError("endpoint-logit lengths differ")
    n_nodes = len(state_raw)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)

    state = calibrate_log_odds(
        state_raw, temperature=state_temperature, offset=state_offset
    )
    incoming = calibrate_log_odds(
        incoming_raw, temperature=incoming_temperature, offset=incoming_offset
    )
    outgoing = calibrate_log_odds(
        outgoing_raw, temperature=outgoing_temperature, offset=outgoing_offset
    )
    conditional = conditional_parent_log_probability(
        edges, transition_energy, n_nodes=n_nodes
    )
    birth = log_sigmoid(-incoming)
    death = log_sigmoid(-outgoing)
    edge_score = (
        log_sigmoid(incoming[edges[:, 1]])
        + conditional
        + log_sigmoid(outgoing[edges[:, 0]])
    )
    if not np.isfinite(edge_score).all():
        raise ValueError("objective-aligned edge score is non-finite")
    return ObjectiveAlignedFactors(
        state_log_odds=state,
        incoming_log_odds=incoming,
        outgoing_log_odds=outgoing,
        birth_score=birth,
        death_score=death,
        conditional_parent_log_probability=conditional,
        edge_score=edge_score,
    )


def reduced_edge_utility(
    edges: np.ndarray,
    edge_score: np.ndarray,
    birth_score: np.ndarray,
    death_score: np.ndarray,
) -> np.ndarray:
    """Return the ordinary reduced-decoder utility for each transition."""

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    score = np.asarray(edge_score, dtype=np.float64).reshape(-1)
    birth = np.asarray(birth_score, dtype=np.float64).reshape(-1)
    death = np.asarray(death_score, dtype=np.float64).reshape(-1)
    if len(edges) != len(score):
        raise ValueError("edge and score lengths differ")
    if len(edges) and ((edges < 0) | (edges >= min(len(birth), len(death)))).any():
        raise ValueError("edge endpoint lies outside event-score arrays")
    output = score - birth[edges[:, 1]] - death[edges[:, 0]]
    if not np.isfinite(output).all():
        raise ValueError("reduced edge utility is non-finite")
    return output
