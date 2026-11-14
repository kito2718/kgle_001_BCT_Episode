#!/usr/bin/env python3
"""Build sparse motion graphs from nnU-Net peaks and solve them with tracksdata ILP."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td
from ilpy import Variable
from scipy.spatial import cKDTree
from tracksdata.constants import DEFAULT_ATTR_KEYS

from biohub_evaluation_scope import DEFAULT_REP20, add_scope_arguments, resolve_scope_stems

SPACING_ZYX_UM = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float64)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-final", type=Path, required=True)
    parser.add_argument("--predicted-nodes-dir", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--case-partition", choices=("train", "val", "all"), default="val"
    )
    parser.add_argument("--det-threshold", type=float, default=0.3)
    parser.add_argument("--motion-mode", choices=("learned", "zero"), default="learned")
    parser.add_argument("--dilations", default="1")
    parser.add_argument("--gate-um", type=float, default=7.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--distance-sigma-um", type=float, default=3.0)
    parser.add_argument("--confidence-power", type=float, default=0.5)
    parser.add_argument("--edge-weight", type=float, default=-1.0)
    parser.add_argument("--appearance-weight", type=float, default=0.1)
    parser.add_argument("--disappearance-weight", type=float, default=0.1)
    parser.add_argument("--division-weight", type=float, default=1.0)
    parser.add_argument("--division-prior-csv", type=Path)
    parser.add_argument(
        "--division-prior-score-field",
        default="fusion_history_image_w0p75",
    )
    parser.add_argument("--division-prior-threshold", type=float, default=0.9956686467138831)
    parser.add_argument("--division-prior-positive-weight", type=float, default=0.0)
    parser.add_argument("--division-prior-match-um", type=float, default=0.25)
    parser.add_argument("--division-pair-csv", type=Path)
    parser.add_argument("--division-pair-score-field", default="pair_rank_norm")
    parser.add_argument("--division-pair-bonus", type=float, default=0.0)
    parser.add_argument("--division-pair-match-um", type=float, default=0.25)
    parser.add_argument(
        "--secondary-edge-dir",
        type=Path,
        help="Per-stem Transformer scored-edge CSVs used only for a gated bonus.",
    )
    parser.add_argument("--secondary-score-field", default="transformer_edge_logit")
    parser.add_argument("--secondary-max-weight", type=float, default=0.15)
    parser.add_argument("--secondary-low-margin-max", type=float, default=0.35)
    parser.add_argument(
        "--minimum-secondary-target-coverage", type=float, default=0.0
    )
    parser.add_argument("--skip-ilp", action="store_true")
    add_scope_arguments(parser)
    parser.add_argument("--stem-limit", type=int)
    parser.add_argument(
        "--stem-list",
        default="",
        help="Optional comma-separated video stems; useful for prior-affected re-solves.",
    )
    parser.add_argument(
        "--max-frame",
        type=int,
        help="Smoke-test only: ignore validation cases after this frame.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def case_stem_frame(case_id: str) -> tuple[str, int]:
    stem, frame_text = case_id.rsplit("_t", 1)
    return stem, int(frame_text)


def load_nodes(path: Path, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.empty((0, 3)), np.empty((0,))
    coords: list[list[float]] = []
    scores: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            score = float(row["score"])
            if score < threshold:
                continue
            coords.append([float(row["z"]), float(row["y"]), float(row["x"])])
            scores.append(score)
    return (
        np.asarray(coords, dtype=np.float64).reshape(-1, 3),
        np.asarray(scores, dtype=np.float64),
    )


def load_motion(
    path: Path,
    coords: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"motion export missing: {path}")
    with np.load(path) as payload:
        exported_coords = np.asarray(payload["zyx"], dtype=np.float64)
        exported_scores = np.asarray(payload["score"], dtype=np.float64)
        vectors = np.asarray(
            payload["motion_zyx_um_per_frame"], dtype=np.float64
        )
    keep = exported_scores >= threshold
    exported_coords = exported_coords[keep]
    exported_scores = exported_scores[keep]
    vectors = vectors[keep]
    if (
        exported_coords.shape != coords.shape
        or not np.allclose(exported_coords, coords, atol=1e-5)
        or not np.allclose(exported_scores, scores, atol=1e-5)
    ):
        raise RuntimeError(
            f"node/motion order mismatch for {path}: "
            f"{exported_coords.shape} versus {coords.shape}"
        )
    return vectors


SAMPLE_COORD_PATTERN = re.compile(
    r"__z(?P<z>-?[0-9.]+)_y(?P<y>-?[0-9.]+)_x(?P<x>-?[0-9.]+)_r"
)


def load_division_priors(
    path: Path | None,
    score_field: str,
    threshold: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load high-confidence OOF Parent priors grouped by case."""
    if path is None:
        return {}
    grouped: dict[str, list[tuple[list[float], float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "sample_id", score_field}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"division prior missing columns: {sorted(missing)}")
        for row in reader:
            score = float(row[score_field])
            if not np.isfinite(score) or score < threshold:
                continue
            match = SAMPLE_COORD_PATTERN.search(row["sample_id"])
            if match is None:
                raise ValueError(f"cannot parse division prior coordinate: {row['sample_id']}")
            grouped[row["case_id"]].append(
                ([float(match["z"]), float(match["y"]), float(match["x"])], score)
            )
    return {
        case_id: (
            np.asarray([item[0] for item in items], dtype=np.float64),
            np.asarray([item[1] for item in items], dtype=np.float64),
        )
        for case_id, items in grouped.items()
    }


def match_division_prior_scores(
    coords_zyx: np.ndarray,
    case_priors: tuple[np.ndarray, np.ndarray] | None,
    match_um: float,
) -> np.ndarray:
    """Match prior rows back to the identical detector nodes in physical space."""
    scores = np.zeros(len(coords_zyx), dtype=np.float64)
    if case_priors is None or not len(coords_zyx):
        return scores
    prior_coords, prior_scores = case_priors
    if not len(prior_coords):
        return scores
    tree = cKDTree(prior_coords * SPACING_ZYX_UM)
    distances, indices = tree.query(coords_zyx * SPACING_ZYX_UM, k=1)
    keep = np.isfinite(distances) & (distances <= match_um)
    scores[keep] = prior_scores[indices[keep]]
    return scores


def load_division_pair_rows(
    path: Path | None,
    score_field: str,
    parent_threshold: float,
    fold: int,
) -> dict[str, list[dict[str, float | str]]]:
    """Load inference-only daughter-pair rows without consulting GT labels."""
    if path is None:
        return {}
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    coordinate_fields = (
        "parent_z", "parent_y", "parent_x",
        "daughter1_z", "daughter1_y", "daughter1_x",
        "daughter2_z", "daughter2_y", "daughter2_x",
    )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "fold", "case_id", "parent_prior", score_field, *coordinate_fields,
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"division pair file missing columns: {sorted(missing)}")
        for row in reader:
            if int(row["fold"]) != fold:
                continue
            parent_prior = float(row["parent_prior"])
            score = float(row[score_field])
            if (
                not np.isfinite(parent_prior)
                or not np.isfinite(score)
                or parent_prior < parent_threshold
            ):
                continue
            grouped[row["case_id"]].append({
                "case_id": row["case_id"],
                "parent_prior": parent_prior,
                "score": score,
                **{name: float(row[name]) for name in coordinate_fields},
            })
    return grouped


def _nearest_coordinate_index(
    coordinates: np.ndarray,
    target: np.ndarray,
    match_um: float,
) -> int | None:
    if not len(coordinates):
        return None
    distances = np.linalg.norm(
        (coordinates - target[None, :]) * SPACING_ZYX_UM[None, :], axis=1
    )
    index = int(np.argmin(distances))
    return index if float(distances[index]) <= match_um else None


def match_division_pair_terms(
    frame_payload: dict[int, dict[str, np.ndarray]],
    case_rows: dict[str, list[dict[str, float | str]]],
    edges: dict[tuple[int, int], tuple[float, float, int]],
    match_um: float,
) -> tuple[list[tuple[tuple[int, int], tuple[int, int], float]], dict[str, int]]:
    """Map parent/daughter coordinates to two existing candidate edges."""
    terms: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    audit = {"rows": 0, "coordinate_matched": 0, "edge_pair_matched": 0}
    for case_id, rows in case_rows.items():
        _, frame = case_stem_frame(case_id)
        source = frame_payload.get(frame)
        target = frame_payload.get(frame + 1)
        if source is None or target is None:
            continue
        for row in rows:
            audit["rows"] += 1
            source_local = _nearest_coordinate_index(
                source["coords"],
                np.asarray([row["parent_z"], row["parent_y"], row["parent_x"]]),
                match_um,
            )
            first_local = _nearest_coordinate_index(
                target["coords"],
                np.asarray([
                    row["daughter1_z"], row["daughter1_y"], row["daughter1_x"],
                ]),
                match_um,
            )
            second_local = _nearest_coordinate_index(
                target["coords"],
                np.asarray([
                    row["daughter2_z"], row["daughter2_y"], row["daughter2_x"],
                ]),
                match_um,
            )
            if None in (source_local, first_local, second_local):
                continue
            if first_local == second_local:
                continue
            audit["coordinate_matched"] += 1
            source_index = int(source["indices"][source_local])
            first_index = int(target["indices"][first_local])
            second_index = int(target["indices"][second_local])
            first_edge = (source_index, first_index)
            second_edge = (source_index, second_index)
            if first_edge not in edges or second_edge not in edges:
                continue
            audit["edge_pair_matched"] += 1
            key = tuple(sorted((first_edge, second_edge)))
            terms[key] = max(terms.get(key, -np.inf), float(row["score"]))
    return [(*key, score) for key, score in terms.items()], audit


class DivisionPairILPSolver(td.solvers.ILPSolver):
    """tracksdata ILP with a true binary term for a selected daughter-edge pair."""

    def __init__(
        self,
        *,
        pair_terms: list[tuple[int, int, float]],
        **kwargs,
    ) -> None:
        self.pair_terms = pair_terms
        super().__init__(**kwargs)

    def reset_model(self) -> None:
        super().reset_model()
        self._pair_vars: list[Variable] = []

    def _add_objective_and_variables(
        self,
        nodes_df: pl.DataFrame,
        edges_df: pl.DataFrame,
    ) -> None:
        super()._add_objective_and_variables(nodes_df, edges_df)
        usable = [
            term for term in self.pair_terms
            if term[0] in self._edge_vars and term[1] in self._edge_vars
        ]
        self._objective.resize(self._count + len(usable))
        for pair_index, (first_edge_id, second_edge_id, coefficient) in enumerate(usable):
            pair_var = Variable(f"division_pair_{pair_index}", index=self._count)
            self._objective.set_coefficient(self._count, coefficient)
            self._count += 1
            first = self._edge_vars[first_edge_id]
            second = self._edge_vars[second_edge_id]
            self._constraints.add(pair_var <= first)
            self._constraints.add(pair_var <= second)
            self._constraints.add(pair_var >= first + second - 1)
            self._pair_vars.append(pair_var)


def add_candidate_edges(
    edges: dict[tuple[int, int], tuple[float, float, int]],
    source_indices: np.ndarray,
    source_zyx: np.ndarray,
    source_scores: np.ndarray,
    target_indices: np.ndarray,
    target_zyx: np.ndarray,
    target_scores: np.ndarray,
    target_motion: np.ndarray,
    dilation: int,
    mode: str,
    gate_um: float,
    top_k: int,
    sigma_um: float,
    confidence_power: float,
) -> None:
    if not len(source_indices) or not len(target_indices):
        return
    source_um = source_zyx * SPACING_ZYX_UM
    target_um = target_zyx * SPACING_ZYX_UM
    projected_um = target_um.copy()
    if mode == "learned":
        projected_um += target_motion * float(dilation)

    tree = cKDTree(source_um)
    k = min(top_k, len(source_indices))
    distances, neighbors = tree.query(
        projected_um, k=k, distance_upper_bound=gate_um
    )
    if k == 1:
        distances = distances[:, None]
        neighbors = neighbors[:, None]
    for target_local in range(len(target_indices)):
        for distance, source_local in zip(
            distances[target_local], neighbors[target_local]
        ):
            if not np.isfinite(distance) or source_local >= len(source_indices):
                continue
            confidence = (
                source_scores[source_local] * target_scores[target_local]
            ) ** confidence_power
            probability = float(
                np.exp(-0.5 * (float(distance) / sigma_um) ** 2)
                * confidence
            )
            key = (
                int(source_indices[source_local]),
                int(target_indices[target_local]),
            )
            previous = edges.get(key)
            value = (probability, float(distance), dilation)
            if previous is None or probability > previous[0]:
                edges[key] = value


def load_secondary_edge_scores(
    path: Path,
    stem: str,
    score_field: str,
) -> dict[tuple[int, int], float]:
    """Load one secondary score per exact detector-node edge."""
    if not path.is_file():
        raise FileNotFoundError(f"secondary edge export missing: {path}")
    scores: dict[tuple[int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"stem", "source_uid", "target_uid", "dt", score_field}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"secondary edge CSV missing columns: {sorted(missing)}")
        for row in reader:
            if row["stem"] != stem:
                raise ValueError(f"secondary edge stem mismatch: {row['stem']} != {stem}")
            if int(row["dt"]) != 1:
                continue
            key = (int(row["source_uid"]), int(row["target_uid"]))
            score = float(row[score_field])
            if not np.isfinite(score):
                raise ValueError(f"non-finite secondary edge score for {stem} {key}")
            if key in scores and not np.isclose(scores[key], score, atol=1e-8):
                raise ValueError(f"conflicting duplicate secondary edge score: {stem} {key}")
            scores[key] = score
    return scores


def _stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exponent = np.exp(shifted)
    return exponent / float(exponent.sum())


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    output[~positive] = negative_exp / (1.0 + negative_exp)
    return output


def apply_low_margin_consensus_bonus(
    edges: dict[tuple[int, int], tuple[float, float, int]],
    secondary_scores: dict[tuple[int, int], float],
    *,
    max_weight: float,
    low_margin_max: float,
) -> list[dict[str, float | int | bool]]:
    """Blend aligned secondary logits only for low-margin same-top targets.

    The primary candidate set is immutable. Targets with incomplete secondary
    coverage, fewer than two primary candidates, high primary margin, or top-1
    disagreement receive exactly zero secondary weight.
    """
    if not 0.0 < max_weight < 1.0:
        raise ValueError("secondary max weight must be inside (0,1)")
    if not 0.0 < low_margin_max <= 1.0:
        raise ValueError("secondary low-margin max must be inside (0,1]")
    grouped: dict[int, list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    for key in edges:
        grouped[key[1]].append((key[0], key))

    details: list[dict[str, float | int | bool]] = []
    for target_uid, candidates in sorted(grouped.items()):
        candidates.sort()
        keys = [key for _, key in candidates]
        covered = [key in secondary_scores for key in keys]
        row: dict[str, float | int | bool] = {
            "target_uid": target_uid,
            "candidate_count": len(keys),
            "secondary_covered_count": int(sum(covered)),
            "full_secondary_coverage": bool(all(covered)),
            "primary_margin": float("nan"),
            "primary_top_source_uid": -1,
            "secondary_top_source_uid": -1,
            "same_top": False,
            "low_margin": False,
            "local_weight": 0.0,
        }
        if len(keys) < 2 or not all(covered):
            details.append(row)
            continue

        primary_probability = np.asarray(
            [edges[key][0] for key in keys], dtype=np.float64
        )
        primary_probability = np.clip(primary_probability, 1e-6, 1.0 - 1e-6)
        primary_logits = np.log(primary_probability) - np.log1p(-primary_probability)
        secondary_logits = np.asarray(
            [secondary_scores[key] for key in keys], dtype=np.float64
        )
        primary_center = float(primary_logits.mean())
        secondary_center = float(secondary_logits.mean())
        primary_scale = max(float(primary_logits.std()), 1e-4)
        secondary_scale = max(float(secondary_logits.std()), 1e-4)
        scale_ratio = float(np.clip(primary_scale / secondary_scale, 0.5, 2.0))
        secondary_aligned = (
            (secondary_logits - secondary_center) * scale_ratio + primary_center
        )

        primary_distribution = _stable_softmax(primary_logits)
        primary_order = np.argsort(-primary_distribution)
        primary_top = int(primary_order[0])
        primary_margin = float(
            primary_distribution[primary_order[0]]
            - primary_distribution[primary_order[1]]
        )
        secondary_top = int(np.argmax(secondary_aligned))
        same_top = primary_top == secondary_top
        low_margin = primary_margin < low_margin_max
        local_weight = (
            max_weight
            * float(np.clip((low_margin_max - primary_margin) / low_margin_max, 0.0, 1.0))
            if same_top and low_margin
            else 0.0
        )
        row.update({
            "primary_margin": primary_margin,
            "primary_top_source_uid": keys[primary_top][0],
            "secondary_top_source_uid": keys[secondary_top][0],
            "same_top": same_top,
            "low_margin": low_margin,
            "local_weight": local_weight,
        })
        if local_weight > 0.0:
            blended_logits = (
                (1.0 - local_weight) * primary_logits
                + local_weight * secondary_aligned
            )
            blended_probability = _sigmoid(blended_logits)
            for key, probability in zip(keys, blended_probability):
                _, distance, dilation = edges[key]
                edges[key] = (float(probability), distance, dilation)
        details.append(row)
    return details


def build_graph(
    node_rows: list[dict[str, float | int]],
    edges: dict[tuple[int, int], tuple[float, float, int]],
    *,
    return_mappings: bool = False,
) -> (
    td.graph.InMemoryGraph
    | tuple[
        td.graph.InMemoryGraph,
        list[int],
        dict[tuple[int, int], int],
    ]
):
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x", "node_score", "division_prior", "division_cost"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    node_ids = graph.bulk_add_nodes(node_rows)
    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.add_edge_attr_key("dilation", pl.Int64, 1)
        edge_items = list(edges.items())
        graph.bulk_add_edges([
            {
                "source_id": node_ids[source],
                "target_id": node_ids[target],
                "edge_prob": probability,
                "edge_dist": distance,
                "dilation": dilation,
            }
            for (source, target), (probability, distance, dilation) in edge_items
        ])
        local_index_by_node_id = {
            int(node_id): local_index
            for local_index, node_id in enumerate(node_ids)
        }
        edge_id_by_local_pair = {
            (
                local_index_by_node_id[int(row[DEFAULT_ATTR_KEYS.EDGE_SOURCE])],
                local_index_by_node_id[int(row[DEFAULT_ATTR_KEYS.EDGE_TARGET])],
            ): int(row[DEFAULT_ATTR_KEYS.EDGE_ID])
            for row in graph.edge_attrs().iter_rows(named=True)
        }
    else:
        edge_id_by_local_pair = {}
    if return_mappings:
        return graph, node_ids, edge_id_by_local_pair
    return graph


def save_graph(graph: td.graph.BaseGraph, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    graph.to_geff(output_path)


@contextlib.contextmanager
def suppress_solver_output():
    with open(os.devnull, "w") as null:
        with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
            yield


def main() -> None:
    args = arguments()
    if args.secondary_edge_dir is not None:
        if not 0.0 < args.secondary_max_weight < 1.0:
            raise ValueError("secondary max weight must be inside (0,1)")
        if not 0.0 < args.secondary_low_margin_max <= 1.0:
            raise ValueError("secondary low-margin max must be inside (0,1]")
        if not 0.0 <= args.minimum_secondary_target_coverage <= 1.0:
            raise ValueError("minimum secondary target coverage must be inside [0,1]")
    dilations = tuple(
        int(value) for value in args.dilations.replace(",", " ").split()
    )
    if not dilations or any(value not in (1, 2) for value in dilations):
        raise ValueError("dilations must be a non-empty subset of 1,2")

    splits = json.loads(args.splits_final.read_text())
    division_priors = load_division_priors(
        args.division_prior_csv,
        args.division_prior_score_field,
        args.division_prior_threshold,
    )
    division_pair_rows = load_division_pair_rows(
        args.division_pair_csv,
        args.division_pair_score_field,
        args.division_prior_threshold,
        args.fold,
    )
    if args.case_partition == "all":
        cases = list(
            dict.fromkeys(
                [*splits[args.fold]["train"], *splits[args.fold]["val"]]
            )
        )
    else:
        cases = list(splits[args.fold][args.case_partition])
    cases_by_stem: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for case_id in cases:
        stem, frame = case_stem_frame(case_id)
        if args.max_frame is not None and frame > args.max_frame:
            continue
        cases_by_stem[stem].append((frame, case_id))
    stems = resolve_scope_stems(
        sorted(cases_by_stem),
        evaluation_scope=args.evaluation_scope,
        fold=args.fold,
        case_partition=args.case_partition,
        stem_file=args.stem_file,
    )
    requested_stems = {
        value.strip() for value in args.stem_list.split(",") if value.strip()
    }
    if (
        args.evaluation_scope == "custom"
        and args.stem_file is None
        and not requested_stems
        and args.stem_limit is None
    ):
        raise ValueError(
            "custom evaluation scope requires --stem-file, --stem-list, or --stem-limit"
        )
    if requested_stems:
        if args.evaluation_scope != "custom":
            raise ValueError("--stem-list requires --evaluation-scope custom")
        unknown = requested_stems - set(stems)
        if unknown:
            raise ValueError(f"stem-list contains non-fold stems: {sorted(unknown)}")
        stems = [stem for stem in stems if stem in requested_stems]
    if args.stem_limit is not None:
        if args.evaluation_scope != "custom":
            raise ValueError("--stem-limit requires --evaluation-scope custom")
        stems = stems[: args.stem_limit]

    if args.output_dir.exists() and not args.resume:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_video: list[dict[str, float | int | str]] = []
    consensus_gate_rows: list[dict[str, float | int | bool | str]] = []
    matched_prior_nodes = 0
    pair_audit_total = {"rows": 0, "coordinate_matched": 0, "edge_pair_matched": 0}
    started = time.perf_counter()
    for stem_index, stem in enumerate(stems, start=1):
        output_path = args.output_dir / f"{stem}.geff"
        if args.resume and output_path.exists():
            continue
        frame_payload: dict[int, dict[str, np.ndarray]] = {}
        node_rows: list[dict[str, float | int]] = []
        for frame, case_id in sorted(cases_by_stem[stem]):
            coords, scores = load_nodes(
                args.predicted_nodes_dir / f"{case_id}.csv",
                args.det_threshold,
            )
            indices = np.arange(
                len(node_rows), len(node_rows) + len(coords), dtype=np.int64
            )
            prior_scores = match_division_prior_scores(
                coords,
                division_priors.get(case_id),
                args.division_prior_match_um,
            )
            matched_prior_nodes += int(
                np.sum(prior_scores >= args.division_prior_threshold)
            )
            node_rows.extend(
                {
                    "t": int(frame),
                    "z": float(coord[0]),
                    "y": float(coord[1]),
                    "x": float(coord[2]),
                    "node_score": float(score),
                    "division_prior": float(prior_score),
                    "division_cost": float(
                        args.division_prior_positive_weight
                        if prior_score >= args.division_prior_threshold
                        else args.division_weight
                    ),
                }
                for coord, score, prior_score in zip(coords, scores, prior_scores)
            )
            frame_payload[frame] = {
                "indices": indices,
                "coords": coords,
                "scores": scores,
            }

        edges: dict[tuple[int, int], tuple[float, float, int]] = {}
        for frame, case_id in sorted(cases_by_stem[stem]):
            target = frame_payload[frame]
            for dilation in dilations:
                if frame - dilation not in frame_payload:
                    continue
                vectors = load_motion(
                    args.motion_dir
                    / "cases"
                    / f"d{dilation}_{case_id}.npz",
                    target["coords"],
                    target["scores"],
                    args.det_threshold,
                )
                source = frame_payload[frame - dilation]
                add_candidate_edges(
                    edges,
                    source["indices"],
                    source["coords"],
                    source["scores"],
                    target["indices"],
                    target["coords"],
                    target["scores"],
                    vectors,
                    dilation,
                    args.motion_mode,
                    args.gate_um,
                    args.top_k,
                    args.distance_sigma_um,
                    args.confidence_power,
                )

        stem_gate_rows: list[dict[str, float | int | bool]] = []
        if args.secondary_edge_dir is not None:
            secondary_scores = load_secondary_edge_scores(
                args.secondary_edge_dir / f"{stem}.csv",
                stem,
                args.secondary_score_field,
            )
            stem_gate_rows = apply_low_margin_consensus_bonus(
                edges,
                secondary_scores,
                max_weight=args.secondary_max_weight,
                low_margin_max=args.secondary_low_margin_max,
            )
            consensus_gate_rows.extend(
                {"stem": stem, **row} for row in stem_gate_rows
            )
            stem_coverage = (
                sum(bool(row["full_secondary_coverage"]) for row in stem_gate_rows)
                / max(len(stem_gate_rows), 1)
            )
            if stem_coverage < args.minimum_secondary_target_coverage:
                raise RuntimeError(
                    f"secondary target coverage {stem_coverage:.6f} below "
                    f"minimum {args.minimum_secondary_target_coverage:.6f} for {stem}"
                )

        stem_pair_rows = {
            case_id: rows
            for case_id, rows in division_pair_rows.items()
            if case_id.startswith(f"{stem}_t")
        }
        local_pair_terms, pair_audit = match_division_pair_terms(
            frame_payload,
            stem_pair_rows,
            edges,
            args.division_pair_match_um,
        )
        for key in pair_audit_total:
            pair_audit_total[key] += pair_audit[key]
        graph, _, edge_id_by_local_pair = build_graph(
            node_rows, edges, return_mappings=True
        )
        solver_pair_terms = [
            (
                edge_id_by_local_pair[first_edge],
                edge_id_by_local_pair[second_edge],
                -args.division_pair_bonus * pair_score,
            )
            for first_edge, second_edge, pair_score in local_pair_terms
        ]
        raw_edges = int(graph.num_edges())
        solve_seconds = 0.0
        if not args.skip_ilp and raw_edges:
            solver_class = (
                DivisionPairILPSolver
                if args.division_pair_csv is not None and solver_pair_terms
                else td.solvers.ILPSolver
            )
            solver_kwargs = dict(
                edge_weight=args.edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=args.appearance_weight,
                disappearance_weight=args.disappearance_weight,
                division_weight=(
                    "division_cost"
                    if args.division_prior_csv is not None
                    else args.division_weight
                ),
            )
            if solver_class is DivisionPairILPSolver:
                solver_kwargs["pair_terms"] = solver_pair_terms
            solver = solver_class(**solver_kwargs)
            solve_started = time.perf_counter()
            with suppress_solver_output():
                graph = solver.solve(graph)
            solve_seconds = time.perf_counter() - solve_started
        save_graph(graph, output_path)
        row = {
            "stem": stem,
            "nodes": int(graph.num_nodes()),
            "candidate_edges": raw_edges,
            "selected_edges": int(graph.num_edges()),
            "solve_seconds": solve_seconds,
            "division_pair_terms": len(solver_pair_terms),
            "consensus_gate_targets": sum(
                float(item["local_weight"]) > 0.0 for item in stem_gate_rows
            ),
            "secondary_target_coverage": (
                sum(
                    bool(item["full_secondary_coverage"])
                    for item in stem_gate_rows
                ) / max(len(stem_gate_rows), 1)
            ),
        }
        per_video.append(row)
        print(
            f"[{stem_index}/{len(stems)}] {stem} "
            f"nodes={row['nodes']} edges={raw_edges}->{row['selected_edges']} "
            f"solve={solve_seconds:.2f}s",
            flush=True,
        )

    if per_video:
        with (args.output_dir / "graph_stats.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_video[0]))
            writer.writeheader()
            writer.writerows(per_video)
    if args.secondary_edge_dir is not None:
        audit_path = args.output_dir / "consensus_gate_audit.csv"
        if not consensus_gate_rows:
            raise RuntimeError("secondary consensus gate produced no target audit rows")
        with audit_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(consensus_gate_rows[0])
            )
            writer.writeheader()
            writer.writerows(consensus_gate_rows)
        full_coverage = sum(
            bool(row["full_secondary_coverage"]) for row in consensus_gate_rows
        )
        if full_coverage == 0:
            raise RuntimeError("secondary consensus gate has zero fully covered targets")
    summary = {
        "schema_version": 1,
        "fold": args.fold,
        "case_partition": args.case_partition,
        "evaluation_scope": args.evaluation_scope,
        "stem_file": str(args.stem_file or DEFAULT_REP20)
        if args.evaluation_scope == "screen20"
        else (str(args.stem_file) if args.stem_file else None),
        "stem_names": stems,
        "stems": len(stems),
        "max_frame": args.max_frame,
        "det_threshold": args.det_threshold,
        "motion_mode": args.motion_mode,
        "dilations": list(dilations),
        "gate_um": args.gate_um,
        "top_k": args.top_k,
        "distance_sigma_um": args.distance_sigma_um,
        "confidence_power": args.confidence_power,
        "skip_ilp": args.skip_ilp,
        "ilp": {
            "edge_weight": args.edge_weight,
            "appearance_weight": args.appearance_weight,
            "disappearance_weight": args.disappearance_weight,
            "division_weight": args.division_weight,
        },
        "division_prior": {
            "csv": str(args.division_prior_csv) if args.division_prior_csv else None,
            "score_field": args.division_prior_score_field,
            "threshold": args.division_prior_threshold,
            "positive_weight": args.division_prior_positive_weight,
            "match_um": args.division_prior_match_um,
            "matched_nodes": matched_prior_nodes,
        },
        "division_pair_prior": {
            "csv": str(args.division_pair_csv) if args.division_pair_csv else None,
            "score_field": args.division_pair_score_field,
            "bonus": args.division_pair_bonus,
            "match_um": args.division_pair_match_um,
            **pair_audit_total,
        },
        "secondary_consensus": {
            "edge_dir": str(args.secondary_edge_dir)
            if args.secondary_edge_dir is not None
            else None,
            "score_field": args.secondary_score_field,
            "max_weight": args.secondary_max_weight,
            "low_margin_max": args.secondary_low_margin_max,
            "minimum_target_coverage": args.minimum_secondary_target_coverage,
            "targets": len(consensus_gate_rows),
            "full_coverage_targets": sum(
                bool(row["full_secondary_coverage"])
                for row in consensus_gate_rows
            ),
            "low_margin_targets": sum(
                bool(row["low_margin"]) for row in consensus_gate_rows
            ),
            "same_top_targets": sum(
                bool(row["same_top"]) for row in consensus_gate_rows
            ),
            "gated_targets": sum(
                float(row["local_weight"]) > 0.0
                for row in consensus_gate_rows
            ),
            "mean_positive_weight": (
                float(np.mean([
                    float(row["local_weight"])
                    for row in consensus_gate_rows
                    if float(row["local_weight"]) > 0.0
                ]))
                if any(
                    float(row["local_weight"]) > 0.0
                    for row in consensus_gate_rows
                )
                else 0.0
            ),
        },
        "videos_completed_now": len(per_video),
        "nodes": int(sum(int(row["nodes"]) for row in per_video)),
        "candidate_edges": int(
            sum(int(row["candidate_edges"]) for row in per_video)
        ),
        "selected_edges": int(
            sum(int(row["selected_edges"]) for row in per_video)
        ),
        "solve_seconds": float(
            sum(float(row["solve_seconds"]) for row in per_video)
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
