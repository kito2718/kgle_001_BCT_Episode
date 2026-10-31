"""Event generators — births and faint cells, in track space first.

Every generated event is a LATENT (positions over frames + lineage) rendered
by transplanting real donor patches (compose.py). Constraints enforced per
frame, all measured constants:

  NMS_UM = 5.0        deployed selection NMS (`learned.py:44`): a synthetic
                      GT node closer than this to an existing node could
                      never be seated as a separate detection — the label
                      would be unlearnable. Enforced against host GT AND
                      previously accepted synthetic nodes, every frame.
  geometry            B2 samples (vec1, vec2) pairs from the empirical GT
                      event bank in host-med-scaled space, conditioned on the
                      host's real kept-daughter displacement, rotated about z
                      so the bank's vec1 aligns with the host's vec1 — the
                      joint (distance, opening-angle, dz) structure of real
                      divisions is preserved exactly. Per the twin-gate KILL,
                      d2's appearance donor is a generic local cell, never a
                      copy of c1.
  faint profile       target prominence drawn from the measured killed-cell
                      distribution (resolution_proof, prominence_44), placed
                      at jittered REAL killed sites (the deserted band:
                      nearest selected node q10-q90 = 7.3-14.8 um).
  motion              host-local bulk flow (GT edges within 20 um, control
                      edges as fallback) + the donor's own residual
                      micro-motion under the same dihedral op as its
                      appearance. No z flips anywhere near appearance.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from biohub.metrics.gt import SCALE_ARR, load_gt
from biohub.synthesis.compose import (
    PATCH_X, PATCH_Y, PATCH_Z,
    add_patches, blur_signal, donor_signal, hwhm_um_grid,
    measure_prominence_grid, noise_compensation, pooled_patch,
    residual_bank_from_frame,
)
from biohub.synthesis.donors import DonorBank

NMS_UM = 5.0
BULK_R_UM = 20.0
AMP_CLAMP = (0.6, 1.4)
FAINT_PROM_CLIP = (2.5, 5.5)
GEN_VERSION = "synth_v1"

# The 8 dihedral ops on (y, x): (transpose, flip_y, flip_x)
DIHEDRAL = [(tr, fy, fx) for tr in (0, 1) for fy in (0, 1) for fx in (0, 1)]

LINK_CONFLICT_FAMILIES = (
    "parent_steal",
    "child_steal",
    "double_conflict",
    "parallel_4_to_7_um",
    "crossing_or_overtake",
    "temporary_dropout",
)


def apply_dihedral_patch(p: np.ndarray, op: tuple[int, int, int]) -> np.ndarray:
    tr, fy, fx = op
    if tr:
        p = p.swapaxes(1, 2)
    if fy:
        p = p[:, ::-1, :]
    if fx:
        p = p[:, :, ::-1]
    return np.ascontiguousarray(p)


def apply_dihedral_vec(v_zyx: np.ndarray, op: tuple[int, int, int]) -> np.ndarray:
    z, y, x = float(v_zyx[0]), float(v_zyx[1]), float(v_zyx[2])
    tr, fy, fx = op
    if tr:
        y, x = x, y
    if fy:
        y = -y
    if fx:
        x = -x
    return np.array([z, y, x])


def rot_z(v_zyx: np.ndarray, theta: float) -> np.ndarray:
    """Rotate about the z axis (in the y-x plane) by theta radians."""
    c, s = np.cos(theta), np.sin(theta)
    z, y, x = v_zyx
    return np.array([z, c * y - s * x, s * y + c * x])


def link_conflict_offsets(
    family: str,
    span: int,
    separation_um: float,
    challenge: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonical two-identity paths and appearance profiles in physical units.

    The paths are later rotated about z and translated through a real host.
    Every frame keeps the two cells at least ``separation_um`` apart.  Only a
    small challenge window is ambiguous; the remaining frames provide the
    identity history/future needed by a contextual linker.
    """
    if family not in LINK_CONFLICT_FAMILIES:
        raise ValueError(f"unknown link-conflict family: {family}")
    if span < 8 or not 2 <= challenge <= span - 3:
        raise ValueError("link conflict requires span >= 8 and an interior challenge")
    if separation_um < NMS_UM:
        raise ValueError("link-conflict separation must respect deployed NMS")

    d = float(separation_um)
    a = np.zeros((span, 3), dtype=np.float64)
    b = np.zeros((span, 3), dtype=np.float64)
    a[:, 1] = -0.5 * d
    b[:, 1] = 0.5 * d
    amp_a = np.ones(span, dtype=np.float64)
    amp_b = np.ones(span, dtype=np.float64)

    if family == "parent_steal":
        # The whole pair shifts by one separation at the challenge.  A's true
        # parent is d away while B's parent sits at A's new position.
        a[challenge:, 1] += d
        b[challenge:, 1] += d
        amp_a[challenge] = 0.55
        amp_b[challenge - 1] = 1.20
    elif family == "child_steal":
        a[challenge:, 1] -= d
        b[challenge:, 1] -= d
        amp_b[challenge] = 0.55
        amp_a[challenge - 1] = 1.20
    elif family == "double_conflict":
        # Both identities exchange sites in one transition: both cross-links
        # are geometrically easier than the two true links.
        a[challenge:, 1] = 0.5 * d
        b[challenge:, 1] = -0.5 * d
        amp_a[challenge - 1:challenge + 1] = (1.15, 0.75)
        amp_b[challenge - 1:challenge + 1] = (0.75, 1.15)
    elif family == "parallel_4_to_7_um":
        phase = np.linspace(0.0, 2.0 * np.pi, span)
        amp_a = 0.85 + 0.25 * np.sin(phase)
        amp_b = 0.85 - 0.25 * np.sin(phase)
    elif family == "crossing_or_overtake":
        # Rotate the separation vector through pi over five frames.  The two
        # cells exchange sides without ever violating the 5 um NMS radius.
        lo = max(0, challenge - 2)
        hi = min(span - 1, challenge + 2)
        theta = np.zeros(span, dtype=np.float64)
        theta[hi + 1:] = np.pi
        theta[lo:hi + 1] = np.linspace(0.0, np.pi, hi - lo + 1)
        a[:, 1] = -0.5 * d * np.cos(theta)
        a[:, 2] = -0.5 * d * np.sin(theta)
        b[:, 1:] = -a[:, 1:]
    elif family == "temporary_dropout":
        amp_a[challenge] = 0.10
        if challenge + 1 < span:
            amp_a[challenge + 1] = 0.28
        amp_b[challenge - 1:challenge + 2] = 1.10

    return a, b, amp_a, amp_b


@dataclass
class Constants:
    root: Path

    def __post_init__(self):
        c = self.root / "experiments/synthesis_engine/constants"
        self.bank = pd.read_parquet(c / "division_bank.parquet")
        self.stats = json.loads((c / "sample_stats.json").read_text())
        self.killed = pd.read_parquet(c / "killed_covariates.parquet")
        self.curves = json.loads((c / "noise_curves.json").read_text())
        rp = pd.read_parquet(
            self.root / "experiments/stage6_autopsy/resolution_proof.parquet")
        self.killed_prom = rp[rp.kind == "killed"]["prominence_44"].dropna()
        self.killed_prom = self.killed_prom.clip(*FAINT_PROM_CLIP).to_numpy()


class Host:
    """Everything the generators need about one host sample."""

    def __init__(self, name: str, repo: Path, consts: Constants):
        self.name = name
        self.repo = repo
        self.consts = consts
        self.gt = load_gt(repo / "data/train" / f"{name}.geff")
        self.pos_um = self.gt.zyx * SCALE_ARR
        self.med = consts.stats[name]["med_um"]
        self.cache = np.load(repo / "artifacts/volumes_v1" / f"{name}.npy",
                             mmap_mode="r")
        self.shape_raw = tuple(int(s * d) for s, d in
                               zip(self.cache.shape[1:], (1, 4, 4)))
        self.n_frames = int(self.cache.shape[0])
        self.childs: dict[int, list[int]] = defaultdict(list)
        self.parent: dict[int, int] = {}
        for s, d in self.gt.edges:
            self.childs[int(s)].append(int(d))
            self.parent[int(d)] = int(s)
        self.bank = DonorBank(name, repo / "experiments/synthesis_engine/donor_bank")
        self.curve = consts.curves[name]
        k = consts.killed
        self.killed_rows = k[(k.dataset == name) & (~k.matched)]
        # control graph for bulk-velocity fallback
        gdir = repo / "experiments/stage6_strong_oof/graphs"
        cn = pd.read_parquet(gdir / f"{name}.nodes.parquet")
        ce = pd.read_parquet(gdir / f"{name}.edges.parquet")
        row = dict(zip(cn["node_id"].to_numpy(), range(len(cn))))
        cpos = cn[["z", "y", "x"]].to_numpy(np.float64) * SCALE_ARR
        ct = cn["t"].to_numpy(np.int64)
        self.ctrl_edges: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        src = np.array([row[s] for s in ce["source"]], dtype=np.int64)
        dst = np.array([row[d] for d in ce["target"]], dtype=np.int64)
        for t in np.unique(ct[src]):
            m = ct[src] == t
            self.ctrl_edges[int(t)] = (cpos[src[m]], cpos[dst[m]] - cpos[src[m]])
        self._gt_frame: dict[int, np.ndarray] = {}
        self._resid: dict[int, np.ndarray] = {}

    def gt_at(self, t: int) -> np.ndarray:
        if t not in self._gt_frame:
            self._gt_frame[t] = np.flatnonzero(self.gt.t == t)
        return self._gt_frame[t]

    def bulk_velocity(self, p_um: np.ndarray, t: int) -> np.ndarray:
        """Mean local motion t -> t+1 near p_um (GT first, control fallback)."""
        rows = self.gt_at(t)
        vecs = []
        for r in rows:
            kids = self.childs.get(int(r), [])
            if len(kids) == 1 and np.linalg.norm(
                    self.pos_um[r] - p_um) <= BULK_R_UM:
                vecs.append(self.pos_um[kids[0]] - self.pos_um[r])
        if vecs:
            return np.mean(vecs, axis=0)
        if t in self.ctrl_edges:
            cp, cv = self.ctrl_edges[t]
            d = np.linalg.norm(cp - p_um, axis=1)
            m = d <= BULK_R_UM
            if m.any():
                return cv[m].mean(axis=0)
        return np.zeros(3)

    def local_amp(self, p_um: np.ndarray, t: int, r_um: float = 30.0) -> float:
        """Median 3^3-max grid intensity of GT cells near a site."""
        rows = self.gt_at(t)
        if len(rows) == 0:
            return np.nan
        d = np.linalg.norm(self.pos_um[rows] - p_um, axis=1)
        rows = rows[d <= r_um]
        if len(rows) == 0:
            return np.nan
        frame = self.cache[t]
        vals = []
        for r in rows:
            gz, gy, gx = (int(round(v / dd)) for v, dd in
                          zip(self.gt.zyx[r], (1, 4, 4)))
            gz = min(max(gz, 1), frame.shape[0] - 2)
            gy = min(max(gy, 1), frame.shape[1] - 2)
            gx = min(max(gx, 1), frame.shape[2] - 2)
            vals.append(float(np.asarray(
                frame[gz - 1:gz + 2, gy - 1:gy + 2, gx - 1:gx + 2]).max()))
        return float(np.median(vals))

    def killed_profile(self, cap: int = 60) -> dict:
        """Measured prominence + HWHM of THIS sample's real killed cells —
        the faint arm's target distributions (iteration-1 amendment: global
        artifact targets measured 17% too prominent / 20% too narrow against
        the pilots' own killed populations)."""
        if not hasattr(self, "_killed_profile"):
            proms, hwhms = [], []
            rows = self.killed_rows.head(cap)
            for r in rows.itertuples():
                t = int(r.t)
                fr = np.asarray(self.cache[t], dtype=np.float32)
                q = measure_prominence_grid(fr, self.gt.zyx[int(r.row)])
                if q and np.isfinite(q["prominence"]):
                    proms.append(float(q["prominence"]))
                gc = [int(round(v / d)) for v, d in
                      zip(self.gt.zyx[int(r.row)], (1, 4, 4))]
                if all(8 <= c <= s - 8 for c, s in zip(gc, fr.shape)):
                    h = hwhm_um_grid(fr[gc[0] - 8:gc[0] + 8,
                                        gc[1] - 8:gc[1] + 8,
                                        gc[2] - 8:gc[2] + 8])
                    if np.isfinite(h):
                        hwhms.append(h)
            self._killed_profile = {
                "prom": np.array(proms) if len(proms) >= 8
                else self.consts.killed_prom,
                "hwhm": np.array(hwhms) if len(hwhms) >= 8 else np.array([6.0]),
            }
        return self._killed_profile

    def residual_bank(self, t: int, rng: np.random.Generator) -> np.ndarray:
        if t not in self._resid:
            self._resid[t] = residual_bank_from_frame(
                np.asarray(self.cache[t], dtype=np.float32), 20000, rng)
        return self._resid[t]

    def in_bounds_vox(self, zyx_vox: np.ndarray) -> bool:
        z, y, x = zyx_vox
        return (PATCH_Z <= z < self.shape_raw[0] - PATCH_Z
                and PATCH_Y <= y < self.shape_raw[1] - PATCH_Y
                and PATCH_X <= x < self.shape_raw[2] - PATCH_X)

    def boundary_um(self, zyx_vox: np.ndarray) -> float:
        """Physical distance to the closest source-volume boundary."""
        position = np.asarray(zyx_vox, dtype=np.float64) * SCALE_ARR
        extent = (np.asarray(self.shape_raw, dtype=np.float64) - 1.0) * SCALE_ARR
        return float(np.min(np.concatenate([position, extent - position])))


@dataclass
class Variant:
    host: Host
    seed: int
    rng: np.random.Generator = field(init=False)
    patches: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    syn_t: list[int] = field(default_factory=list)
    syn_zyx: list[np.ndarray] = field(default_factory=list)
    syn_parent: list[int] = field(default_factory=list)  # merged-index, -1 none
    events: list[dict] = field(default_factory=list)
    _syn_by_t: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    _donor_uses: dict = field(default_factory=dict)
    used_mothers: set = field(default_factory=set)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    # ---- constraints ----
    def nms_ok(self, p_um: np.ndarray, t: int,
               ignore_um: np.ndarray | None = None) -> bool:
        rows = self.host.gt_at(t)
        if len(rows):
            d = np.linalg.norm(self.host.pos_um[rows] - p_um, axis=1)
            if ignore_um is not None:
                d = d[np.linalg.norm(
                    self.host.pos_um[rows] - ignore_um, axis=1) > 1e-6]
            if len(d) and d.min() < NMS_UM:
                return False
        for j in self._syn_by_t.get(t, []):
            if np.linalg.norm(self.syn_zyx[j] * SCALE_ARR - p_um) < NMS_UM:
                return False
        return True

    def add_node(self, t: int, zyx_vox: np.ndarray, parent_merged: int) -> int:
        j = len(self.syn_t)
        self.syn_t.append(int(t))
        self.syn_zyx.append(np.asarray(zyx_vox, dtype=np.float64))
        self.syn_parent.append(int(parent_merged))
        self._syn_by_t[int(t)].append(j)
        return self.host.gt.n_nodes + j

    def local_grid(self, t: int, center_vox: np.ndarray, half: int = 8
                   ) -> tuple[np.ndarray, np.ndarray]:
        """Composited grid region around a raw-voxel center: cache + pending
        patches. Returns (region_float32, grid_origin)."""
        gz = int(round(center_vox[0]))
        gy = int(round(center_vox[1] / 4))
        gx = int(round(center_vox[2] / 4))
        f = self.host.cache[t]
        z0, y0, x0 = max(gz - half, 0), max(gy - half, 0), max(gx - half, 0)
        z1 = min(gz + half + 1, f.shape[0])
        y1 = min(gy + half + 1, f.shape[1])
        x1 = min(gx + half + 1, f.shape[2])
        reg = np.asarray(f[z0:z1, y0:y1, x0:x1], dtype=np.float32).copy()
        for (pz, py, px), p in self.patches.get(t, []):
            iz0, iy0, ix0 = max(pz, z0), max(py, y0), max(px, x0)
            iz1 = min(pz + p.shape[0], z1)
            iy1 = min(py + p.shape[1], y1)
            ix1 = min(px + p.shape[2], x1)
            if iz0 < iz1 and iy0 < iy1 and ix0 < ix1:
                reg[iz0 - z0:iz1 - z0, iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] += \
                    p[iz0 - pz:iz1 - pz, iy0 - py:iy1 - py, ix0 - px:ix1 - px]
        return reg, np.array([z0, y0, x0])

    def measure_prom_at(self, t: int, center_vox: np.ndarray,
                        extra: tuple | None = None) -> float | None:
        reg, org = self.local_grid(t, center_vox, half=8)
        if extra is not None:
            (pz, py, px), p = extra
            iz0 = max(pz, org[0]); iy0 = max(py, org[1]); ix0 = max(px, org[2])
            iz1 = min(pz + p.shape[0], org[0] + reg.shape[0])
            iy1 = min(py + p.shape[1], org[1] + reg.shape[1])
            ix1 = min(px + p.shape[2], org[2] + reg.shape[2])
            if iz0 < iz1 and iy0 < iy1 and ix0 < ix1:
                reg = reg.copy()
                reg[iz0 - org[0]:iz1 - org[0], iy0 - org[1]:iy1 - org[1],
                    ix0 - org[2]:ix1 - org[2]] += \
                    p[iz0 - pz:iz1 - pz, iy0 - py:iy1 - py, ix0 - px:ix1 - px]
        # measure() expects the full grid frame; fake it with offset coords
        center_local = np.asarray(center_vox, np.float64).copy()
        center_local[0] -= org[0] * 1
        center_local[1] -= org[1] * 4
        center_local[2] -= org[2] * 4
        q = measure_prominence_grid(reg, center_local)
        return None if q is None else q["prominence"]

    # ---- donor helpers ----
    # Per-donor use budget, split by arm: 2 birth-uses + 2 faint-uses. The
    # split keeps one arm from draining a thin bank before the other runs,
    # while total per-donor exposure stays at 4 instances per variant.
    MAX_USES_PER_ARM = 2
    MAX_EVENT_USES = 2

    def max_donor_span(self, group: str) -> int:
        bank = self.host.bank
        lens = [len(bank.tracklets[g]) for g in bank.clean_ids
                if self._donor_uses.get((group, g), 0) < self.MAX_USES_PER_ARM]
        return max(lens) if lens else 0

    def pick_tracklet(self, span: int, z_um: float, group: str,
                      exclude: set[int] | None = None) -> int | None:
        """Depth-weighted donor draw; unused donors strongly preferred —
        small banks (annotation-sparse samples) would otherwise clone one
        cell across the movie, which both the eye and a discriminator catch."""
        bank = self.host.bank
        exclude = exclude or set()
        ids = [g for g in bank.clean_ids
               if len(bank.tracklets[g]) >= span
               and g not in exclude
               and self._donor_uses.get((group, g), 0) < self.MAX_USES_PER_ARM]
        if not ids:
            return None
        zs = np.array([bank.tracklets[g].z_um.iloc[0] for g in ids])
        uses = np.array([self._donor_uses.get((group, g), 0) for g in ids])
        w = (np.exp(-np.abs(zs - z_um) / 10.0) + 1e-6) / (1.0 + 3.0 * uses)
        return int(self.rng.choice(ids, p=w / w.sum()))

    def render_tracklet(self, donor_id: int, positions: list[np.ndarray],
                        ts: list[int], alpha: float, amp: float,
                        op: tuple, compensate: bool,
                        start: int = 0, blur_um: float = 0.0,
                        feather_px: float = 3.0) -> list[dict] | None:
        """Paste donor patches (dihedral op, scaled) at integer positions.

        `start` picks the donor sub-segment: long donors serve many distinct
        appearance sequences, which is what makes capped reuse non-clone-y.
        `blur_um` applies depth-scatter blur (faint arm) before scaling.
        """
        bank = self.host.bank
        d = bank.tracklets[donor_id]
        recs = []
        for k, (pos_vox, t) in enumerate(zip(positions, ts)):
            row = d.iloc[start + k]
            patch = bank.patch(row.idx)
            sig, bg_med, bg_mad = donor_signal(patch, feather_px=feather_px)
            sig = blur_signal(sig, blur_um)
            sig = apply_dihedral_patch(sig, op) * (alpha * amp)
            out = pooled_patch(sig, pos_vox, self.host.shape_raw)
            if out is None:
                return None
            origin, pooled = out
            if compensate:
                f = self.host.cache[t]
                reg = np.asarray(
                    f[origin[0]:origin[0] + pooled.shape[0],
                      origin[1]:origin[1] + pooled.shape[1],
                      origin[2]:origin[2] + pooled.shape[2]], dtype=np.float32)
                pooled = noise_compensation(
                    pooled, reg, alpha, self.host.curve, self.rng,
                    self.host.residual_bank(t, self.rng))
            recs.append({"t": t, "origin": origin, "pooled": pooled,
                         "donor_idx": int(row.idx)})
        return recs

    def motion_path(self, start_um: np.ndarray, t0: int, span: int,
                    donor_steps: np.ndarray, op: tuple) -> list[np.ndarray]:
        """Latent positions (um): host bulk + donor residual micro-motion."""
        steps = np.array([apply_dihedral_vec(s, op) for s in donor_steps])
        resid = steps - steps.mean(axis=0) if len(steps) else np.zeros((0, 3))
        pos = [start_um.copy()]
        for k in range(span - 1):
            bulk = self.host.bulk_velocity(pos[-1], t0 + k)
            r = resid[k % max(len(resid), 1)] if len(resid) else np.zeros(3)
            pos.append(pos[-1] + bulk + r)
        return pos

    # ---- generators ----
    # Iteration-1 faint levers (pre-declared + amended before the rerun):
    # host-matched targets, depth-scatter blur to the host killed HWHM,
    # noise compensation OFF (measured overshoot 1.16), feather 3 -> 5.
    FAINT_FEATHER_PX = 5.0

    def try_faint(self, span_range=(5, 15), compensate=False) -> bool:
        host, rng = self.host, self.rng
        if len(host.killed_rows) == 0:
            return False
        max_span = self.max_donor_span("faint")
        if max_span < span_range[0]:
            return False  # bank exhausted for this variant
        seed_row = host.killed_rows.iloc[int(rng.integers(len(host.killed_rows)))]
        span = int(rng.integers(span_range[0],
                                min(span_range[1], max_span) + 1))
        t0 = int(seed_row.t)
        if t0 + span >= host.n_frames:
            t0 = max(1, host.n_frames - span - 1)
        # jitter the killed site by 3-8 um (z-compressed: anisotropy)
        base = host.gt.zyx[int(seed_row.row)].astype(np.float64)
        for _ in range(24):
            r = rng.uniform(3.0, 8.0)
            u = rng.normal(size=3) * np.array([0.35, 1.0, 1.0])
            u /= np.linalg.norm(u) + 1e-9
            p_um = base * SCALE_ARR + r * u
            p_vox = np.round(p_um / SCALE_ARR)
            if not host.in_bounds_vox(p_vox):
                continue
            if not self.nms_ok(p_vox * SCALE_ARR, t0):
                continue
            donor = self.pick_tracklet(span, float(p_um[0]), "faint")
            if donor is None:
                continue
            op = DIHEDRAL[int(rng.integers(8))]
            d = host.bank.tracklets[donor]
            start = int(rng.integers(0, len(d) - span + 1))
            seg = d.iloc[start:start + span]
            dsteps = np.diff(
                seg[["cz", "cy", "cx"]].to_numpy(np.float64) * SCALE_ARR,
                axis=0)
            path = self.motion_path(p_vox * SCALE_ARR, t0, span, dsteps, op)
            path_vox = [np.round(p / SCALE_ARR) for p in path]
            ts = list(range(t0, t0 + span))
            keep = 0
            for pv, t in zip(path_vox, ts):
                if not host.in_bounds_vox(pv) or not self.nms_ok(pv * SCALE_ARR, t):
                    break
                keep += 1
            if keep < span_range[0]:
                continue
            path_vox, ts = path_vox[:keep], ts[:keep]
            # host-matched targets: this sample's own killed cells set both
            # the prominence and the width the transplant must land on
            prof = host.killed_profile()
            target = float(rng.choice(prof["prom"]))
            target_hwhm = float(rng.choice(prof["hwhm"]))
            row0 = seg.iloc[0]
            sig, _, _ = donor_signal(host.bank.patch(row0.idx),
                                     feather_px=self.FAINT_FEATHER_PX)
            sig0 = apply_dihedral_patch(sig, op)
            pooled0 = pooled_patch(sig0, path_vox[0], host.shape_raw)
            if pooled0 is None:
                continue
            donor_hwhm = hwhm_um_grid(pooled0[1])
            blur_um = 0.0
            if np.isfinite(donor_hwhm) and target_hwhm > donor_hwhm:
                blur_um = float(np.sqrt(target_hwhm**2 - donor_hwhm**2)
                                / 1.1774)
            sig_b = apply_dihedral_patch(blur_signal(sig, blur_um), op)
            out = pooled_patch(sig_b, path_vox[0], host.shape_raw)
            if out is None:
                continue
            p1 = self.measure_prom_at(ts[0], path_vox[0], extra=out)
            if p1 is None or p1 <= 0:
                continue
            alpha = float(np.clip(target / p1, 0.05, 1.0))
            out_a = pooled_patch(sig_b * alpha, path_vox[0], host.shape_raw)
            p2 = self.measure_prom_at(ts[0], path_vox[0], extra=out_a)
            if p2 and p2 > 0:
                alpha = float(np.clip(alpha * target / p2, 0.05, 1.0))
            recs = self.render_tracklet(donor, path_vox, ts, alpha, 1.0, op,
                                        compensate, start=start,
                                        blur_um=blur_um,
                                        feather_px=self.FAINT_FEATHER_PX)
            if recs is None:
                continue
            self._donor_uses[("faint", donor)] =                 self._donor_uses.get(("faint", donor), 0) + 1
            prev = -1
            for rec, pv, t in zip(recs, path_vox, ts):
                self.patches[t].append((rec["origin"], rec["pooled"]))
                merged = self.add_node(t, pv, prev)
                prev = merged
            self.events.append({
                "type": "faint", "t0": int(ts[0]), "span": len(ts),
                "donor": int(donor), "start": start, "alpha": round(alpha, 4),
                "target_prom": round(target, 3),
                "target_hwhm": round(target_hwhm, 3),
                "blur_um": round(blur_um, 3),
                "site_vox": [float(v) for v in path_vox[0]], "op": list(op),
            })
            return True
        return False

    def _contested(self, d2_um: np.ndarray, t: int) -> tuple[bool, float]:
        """Is there a pixel-thief at the mother frame near the daughter site?

        Real lost events are lost BECAUSE a track passes through the
        daughter's position (thief resid ~1 um). Dose-1 synthetic births
        were all uncontested by the NMS placement rule — training never
        posed the real discrimination. A site counts as contested when a
        host GT node sits within 3.5 um at frame t, or the pixels there
        already hold a blob >= 0.6x the local cell amplitude.
        """
        host = self.host
        rows = host.gt_at(t)
        if len(rows):
            d = np.linalg.norm(host.pos_um[rows] - d2_um, axis=1)
            m = d.min() if len(d) else np.inf
            if m <= 3.5:
                return True, float(m)
        frame = host.cache[t]
        gz, gy, gx = (int(round(v / dd)) for v, dd in
                      zip(d2_um / SCALE_ARR, (1, 4, 4)))
        gz = min(max(gz, 1), frame.shape[0] - 2)
        gy = min(max(gy, 1), frame.shape[1] - 2)
        gx = min(max(gx, 1), frame.shape[2] - 2)
        peak = float(np.asarray(frame[gz - 1:gz + 2, gy - 1:gy + 2,
                                      gx - 1:gx + 2]).max())
        amp = host.local_amp(d2_um, t)
        if np.isfinite(amp) and amp > 1e-3 and peak >= 0.6 * amp:
            return True, np.nan
        return False, np.inf

    n_b2_contested: int = 0
    n_b2_free: int = 0

    def try_birth_b2(self, far_bias: float = 0.5, span_range=(5, 10),
                     contested_quota: float = 0.0) -> bool:
        host, rng, consts = self.host, self.rng, self.host.consts
        # host segment: out-degree-1 edge whose child continues >= 2 more
        cand = [(int(s), kids[0]) for s, kids in
                [(s, host.childs[s]) for s in host.childs]
                if len(kids) == 1]
        cand = [(m, c1) for m, c1 in cand
                if len(host.childs.get(c1, [])) == 1
                and int(host.gt.t[m]) >= 1
                and int(host.gt.t[m]) + 6 < host.n_frames
                and m not in self.used_mothers]
        if not cand:
            return False
        bank = consts.bank
        pool = bank[bank.ratio2 >= 4.0] if rng.random() < far_bias else bank
        if len(pool) == 0:
            pool = bank
        for _ in range(24):
            m, c1 = cand[int(rng.integers(len(cand)))]
            t = int(host.gt.t[m])
            vec1 = host.pos_um[c1] - host.pos_um[m]
            s = host.med / pool.med_um.to_numpy()
            v1b = pool[["v1z_um", "v1y_um", "v1x_um"]].to_numpy() * s[:, None]
            # condition on the host's real kept-daughter displacement
            dmag = np.abs(np.linalg.norm(v1b, axis=1) - np.linalg.norm(vec1))
            ddz = np.abs(np.abs(v1b[:, 0]) - np.abs(vec1[0]))
            score = dmag + 0.5 * ddz
            k = min(15, len(pool))
            pick = int(rng.choice(np.argsort(score)[:k]))
            ev = pool.iloc[pick]
            sc = host.med / ev.med_um
            v1e = np.array([ev.v1z_um, ev.v1y_um, ev.v1x_um]) * sc
            v2e = np.array([ev.v2z_um, ev.v2y_um, ev.v2x_um]) * sc
            if np.sign(v1e[0]) != np.sign(vec1[0]) and abs(vec1[0]) > 0.1:
                v1e[0], v2e[0] = -v1e[0], -v2e[0]  # geometry-only z mirror
            th_host = np.arctan2(vec1[2], vec1[1])
            th_bank = np.arctan2(v1e[2], v1e[1])
            v2 = rot_z(v2e, th_host - th_bank)
            v2 = rot_z(v2, rng.uniform(-0.17, 0.17)) * rng.uniform(0.9, 1.1)
            d2_um = host.pos_um[m] + v2
            d2_vox = np.round(d2_um / SCALE_ARR)
            if not host.in_bounds_vox(d2_vox):
                continue
            if not self.nms_ok(d2_vox * SCALE_ARR, t + 1):
                continue
            if np.linalg.norm(d2_vox * SCALE_ARR - host.pos_um[c1]) < NMS_UM:
                continue
            contested, thief_dist = self._contested(d2_vox * SCALE_ARR, t)
            if contested_quota > 0.0 and not contested:
                total = self.n_b2_contested + self.n_b2_free + 1
                if self.n_b2_free + 1 > total * (1.0 - contested_quota):
                    continue  # hold this slot for a contested site
            max_span = self.max_donor_span("birth")
            if max_span < span_range[0]:
                return False  # bank exhausted for this variant
            span = int(rng.integers(span_range[0],
                                    min(span_range[1], max_span) + 1))
            span = min(span, host.n_frames - (t + 1))
            donor = self.pick_tracklet(span, float(d2_um[0]), "birth")
            if donor is None:
                continue
            op = DIHEDRAL[int(rng.integers(8))]
            d = host.bank.tracklets[donor]
            start = int(rng.integers(0, len(d) - span + 1))
            seg = d.iloc[start:start + span]
            dsteps = np.diff(
                seg[["cz", "cy", "cx"]].to_numpy(np.float64) * SCALE_ARR,
                axis=0)
            path = self.motion_path(d2_vox * SCALE_ARR, t + 1, span, dsteps, op)
            path_vox = [np.round(p / SCALE_ARR) for p in path]
            ts = list(range(t + 1, t + 1 + span))
            keep = 0
            for pv, tt in zip(path_vox, ts):
                if not host.in_bounds_vox(pv) or not self.nms_ok(pv * SCALE_ARR, tt):
                    break
                keep += 1
            if keep < span_range[0]:
                continue
            path_vox, ts = path_vox[:keep], ts[:keep]
            amp_local = host.local_amp(d2_um, min(t + 1, host.n_frames - 1))
            donor_amp = float(seg.iloc[0].peak_grid)
            amp = 1.0
            if np.isfinite(amp_local) and donor_amp > 1e-3:
                amp = float(np.clip(amp_local / donor_amp, *AMP_CLAMP))
            recs = self.render_tracklet(donor, path_vox, ts, 1.0, amp, op,
                                        compensate=False, start=start)
            if recs is None:
                continue
            self._donor_uses[("birth", donor)] = \
                self._donor_uses.get(("birth", donor), 0) + 1
            self.used_mothers.add(m)
            if contested:
                self.n_b2_contested += 1
            else:
                self.n_b2_free += 1
            prev = int(m)
            for rec, pv, tt in zip(recs, path_vox, ts):
                self.patches[tt].append((rec["origin"], rec["pooled"]))
                merged = self.add_node(tt, pv, prev)
                prev = merged
            self.events.append({
                "type": "birth_b2", "t_mother": t, "mother_row": int(m),
                "c1_row": int(c1), "span": len(ts),
                "contested": bool(contested),
                "thief_dist_um": (round(float(thief_dist), 2)
                                  if np.isfinite(thief_dist) else None),
                "bank_event": int(ev.name), "donor": int(donor), "start": start,
                "amp": round(amp, 3), "dist2_um": round(float(np.linalg.norm(v2)), 2),
                "ratio2_host": round(float(np.linalg.norm(v2) / host.med), 2),
                "site_vox": [float(v) for v in path_vox[0]], "op": list(op),
            })
            return True
        return False

    def try_birth_b1(self, dt_range=(-8, 8)) -> bool:
        host, rng = self.host, self.rng
        bank = host.bank
        if not bank.events:
            return False
        eids = [e for e in sorted(bank.events)
                if self._donor_uses.get(("ev", e), 0) < self.MAX_EVENT_USES]
        if not eids:
            return False
        crowd = np.array([bank.events[e].n_peaks44.mean() for e in eids])
        w = np.exp(-np.clip(crowd - 1.0, 0, None)) + 1e-6
        for _ in range(24):
            ei = int(rng.choice(eids, p=w / w.sum()))
            ev = bank.events[ei]
            mother = ev[ev.role == "mother"].sort_values("seq")
            d1 = ev[ev.role == "d1"].sort_values("seq")
            d2 = ev[ev.role == "d2"].sort_values("seq")
            if len(mother) == 0 or len(d1) == 0 or len(d2) == 0:
                continue
            t_div = int(mother.t.iloc[-1])
            dt = int(rng.integers(dt_range[0], dt_range[1] + 1))
            all_rows = pd.concat([mother, d1, d2])
            ts_new = all_rows.t.to_numpy() + dt
            if ts_new.min() < 0 or ts_new.max() >= host.n_frames:
                continue
            op = DIHEDRAL[int(rng.integers(8))]
            p_mother = mother[["cz", "cy", "cx"]].to_numpy(np.float64)[-1]
            # translation: land the mother near tissue, NMS-clear for ALL nodes
            rows_t = host.gt_at(t_div + dt)
            if len(rows_t) == 0:
                continue
            anchor = host.gt.zyx[int(rng.choice(rows_t))].astype(np.float64)
            delta_um = (anchor * SCALE_ARR - p_mother * SCALE_ARR
                        + rng.uniform(8.0, 20.0) * self._rand_dir())
            ok = True
            placed = []
            for _, row in all_rows.iterrows():
                p0 = np.array([row.cz, row.cy, row.cx], dtype=np.float64)
                rel = apply_dihedral_vec(
                    (p0 - p_mother) * SCALE_ARR, op)
                p_um = p_mother * SCALE_ARR + rel + delta_um
                pv = np.round(p_um / SCALE_ARR)
                tt = int(row.t) + dt
                if not host.in_bounds_vox(pv) or not self.nms_ok(pv * SCALE_ARR, tt):
                    ok = False
                    break
                placed.append((row, pv, tt))
            if not ok:
                continue
            amp_local = host.local_amp(anchor * SCALE_ARR,
                                       min(t_div + dt, host.n_frames - 1))
            donor_amp = float(mother.peak_grid.iloc[-1])
            amp = 1.0
            if np.isfinite(amp_local) and donor_amp > 1e-3:
                amp = float(np.clip(amp_local / donor_amp, *AMP_CLAMP))
            merged_of: dict[int, int] = {}
            role_prev: dict[str, int] = {}
            for row, pv, tt in placed:
                patch = bank.patch(row.idx)
                sig, _, _ = donor_signal(patch)
                sig = apply_dihedral_patch(sig, op) * amp
                out = pooled_patch(sig, pv, host.shape_raw)
                if out is None:
                    ok = False
                    break
                self.patches[tt].append(out)
                if row.role == "mother":
                    prev = role_prev.get("mother", -1)
                else:
                    prev = (role_prev.get(row.role)
                            if row.role in role_prev
                            else merged_of.get("mother_last", -1))
                merged = self.add_node(tt, pv, prev if prev is not None else -1)
                role_prev[row.role] = merged
                if row.role == "mother":
                    merged_of["mother_last"] = merged
            if not ok:
                continue
            self._donor_uses[("ev", ei)] = \
                self._donor_uses.get(("ev", ei), 0) + 1
            self.events.append({
                "type": "birth_b1", "event_donor": int(ei), "dt": dt,
                "t_div_new": t_div + dt, "amp": round(amp, 3), "op": list(op),
                "n_nodes": len(placed),
            })
            return True
        return False

    def try_link_conflict(
        self,
        family: str,
        span_range: tuple[int, int] = (12, 18),
        separation_range_um: tuple[float, float] = (5.2, 6.8),
        boundary_range_um: tuple[float, float] = (11.0, 17.0),
    ) -> bool:
        """Render two continuation identities through one contested event.

        Only one short window is intentionally ambiguous.  Long flanks keep
        the raw hard-event rate near the real 3.6% scale instead of repeating
        V27's 22.8% synthetic mixture.  The two donors retain their own image
        sequences, so the correct association is identity continuity rather
        than geometric convenience.
        """
        if family not in LINK_CONFLICT_FAMILIES:
            raise ValueError(f"unknown link-conflict family: {family}")
        host, rng = self.host, self.rng
        max_span = min(
            self.max_donor_span("link_a"),
            self.max_donor_span("link_b"),
        )
        if max_span < span_range[0]:
            return False

        for _ in range(64):
            span = int(rng.integers(
                span_range[0], min(span_range[1], max_span) + 1
            ))
            challenge = int(rng.integers(3, span - 3))
            possible = np.flatnonzero(
                (host.gt.t >= 1) & (host.gt.t + span < host.n_frames)
            )
            if not len(possible):
                return False
            boundary = np.asarray([
                host.boundary_um(host.gt.zyx[row]) for row in possible
            ])
            preferred = possible[
                (boundary >= boundary_range_um[0])
                & (boundary <= boundary_range_um[1])
            ]
            anchor_row = int(rng.choice(preferred if len(preferred) else possible))
            t0 = int(host.gt.t[anchor_row])
            anchor_um = host.pos_um[anchor_row]

            # Offset tangentially to the nearest boundary so the generated
            # endpoint context stays in the real hard-case boundary band.
            extent = (np.asarray(host.shape_raw) - 1.0) * SCALE_ARR
            boundary_values = np.concatenate([anchor_um, extent - anchor_um])
            boundary_axis = int(np.argmin(boundary_values) % 3)
            tangent = self._rand_dir()
            tangent[boundary_axis] = 0.0
            tangent /= np.linalg.norm(tangent) + 1e-9
            center0 = anchor_um + rng.uniform(10.0, 18.0) * tangent

            centers = [center0]
            for k in range(span - 1):
                centers.append(
                    centers[-1] + host.bulk_velocity(centers[-1], t0 + k)
                )
            centers = np.asarray(centers)
            separation = float(rng.uniform(*separation_range_um))
            off_a, off_b, amp_a, amp_b = link_conflict_offsets(
                family, span, separation, challenge
            )
            angle = float(rng.uniform(-np.pi, np.pi))
            off_a = np.asarray([rot_z(value, angle) for value in off_a])
            off_b = np.asarray([rot_z(value, angle) for value in off_b])
            path_a = np.rint((centers + off_a) / SCALE_ARR)
            path_b = np.rint((centers + off_b) / SCALE_ARR)
            ts = list(range(t0, t0 + span))

            valid = True
            for pa, pb, t in zip(path_a, path_b, ts):
                pair_distance = np.linalg.norm((pa - pb) * SCALE_ARR)
                if pair_distance < NMS_UM:
                    valid = False
                    break
                if not host.in_bounds_vox(pa) or not host.in_bounds_vox(pb):
                    valid = False
                    break
                if not self.nms_ok(pa * SCALE_ARR, t):
                    valid = False
                    break
                if not self.nms_ok(pb * SCALE_ARR, t):
                    valid = False
                    break
            if not valid:
                continue
            start_boundary = min(
                host.boundary_um(path_a[0]), host.boundary_um(path_b[0])
            )
            if not (boundary_range_um[0] - 1.5 <= start_boundary
                    <= boundary_range_um[1] + 3.0):
                continue

            donor_a = self.pick_tracklet(
                span, float((path_a[0] * SCALE_ARR)[0]), "link_a"
            )
            donor_b = self.pick_tracklet(
                span, float((path_b[0] * SCALE_ARR)[0]), "link_b",
                exclude={donor_a} if donor_a is not None else None,
            )
            if donor_a is None or donor_b is None:
                continue
            track_a = host.bank.tracklets[donor_a]
            track_b = host.bank.tracklets[donor_b]
            start_a = int(rng.integers(0, len(track_a) - span + 1))
            start_b = int(rng.integers(0, len(track_b) - span + 1))
            op_a = DIHEDRAL[int(rng.integers(len(DIHEDRAL)))]
            op_b = DIHEDRAL[int(rng.integers(len(DIHEDRAL)))]

            local_a = host.local_amp(path_a[0] * SCALE_ARR, t0)
            local_b = host.local_amp(path_b[0] * SCALE_ARR, t0)
            donor_amp_a = float(track_a.peak_grid.iloc[start_a])
            donor_amp_b = float(track_b.peak_grid.iloc[start_b])
            base_a = (float(np.clip(local_a / donor_amp_a, *AMP_CLAMP))
                      if np.isfinite(local_a) and donor_amp_a > 1e-3 else 1.0)
            base_b = (float(np.clip(local_b / donor_amp_b, *AMP_CLAMP))
                      if np.isfinite(local_b) and donor_amp_b > 1e-3 else 1.0)
            rec_a = self.render_tracklet(
                donor_a, list(path_a), ts, 1.0, base_a, op_a,
                compensate=False, start=start_a,
            )
            rec_b = self.render_tracklet(
                donor_b, list(path_b), ts, 1.0, base_b, op_b,
                compensate=False, start=start_b,
            )
            if rec_a is None or rec_b is None:
                continue

            self._donor_uses[("link_a", donor_a)] = \
                self._donor_uses.get(("link_a", donor_a), 0) + 1
            self._donor_uses[("link_b", donor_b)] = \
                self._donor_uses.get(("link_b", donor_b), 0) + 1
            prev_a = -1
            for rec, position, t, scale in zip(rec_a, path_a, ts, amp_a):
                self.patches[t].append((rec["origin"], rec["pooled"] * scale))
                prev_a = self.add_node(t, position, prev_a)
            prev_b = -1
            for rec, position, t, scale in zip(rec_b, path_b, ts, amp_b):
                self.patches[t].append((rec["origin"], rec["pooled"] * scale))
                prev_b = self.add_node(t, position, prev_b)
            self.events.append({
                "type": "link_conflict",
                "family": family,
                "t0": t0,
                "span": span,
                "n_nodes": 2 * span,
                "challenge_offset": challenge,
                "separation_um": round(separation, 3),
                "start_boundary_um": round(start_boundary, 3),
                "donor_a": int(donor_a),
                "donor_b": int(donor_b),
                "start_a": start_a,
                "start_b": start_b,
                "base_amp_a": round(base_a, 3),
                "base_amp_b": round(base_b, 3),
                "amp_profile_a": np.round(amp_a, 3).tolist(),
                "amp_profile_b": np.round(amp_b, 3).tolist(),
                "site_vox": np.round(centers[0] / SCALE_ARR, 3).tolist(),
                "op_a": list(op_a),
                "op_b": list(op_b),
            })
            return True
        return False

    def _rand_dir(self) -> np.ndarray:
        u = self.rng.normal(size=3) * np.array([0.35, 1.0, 1.0])
        return u / (np.linalg.norm(u) + 1e-9)

    # ---- serialization ----
    def save(self, out_dir: Path, tag: str) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        origins, shapes, chunks = [], [], []
        for t in sorted(self.patches):
            for (gz, gy, gx), p in self.patches[t]:
                origins.append([t, gz, gy, gx])
                shapes.append(p.shape)
                chunks.append(p.astype(np.float16).ravel())
        data = (np.concatenate(chunks) if chunks
                else np.zeros(0, dtype=np.float16))
        offs = np.zeros(len(chunks) + 1, dtype=np.int64)
        if chunks:
            offs[1:] = np.cumsum([c.size for c in chunks])
        npz = out_dir / f"{tag}.npz"
        np.savez_compressed(
            npz,
            origins=np.asarray(origins, dtype=np.int32).reshape(-1, 4),
            shapes=np.asarray(shapes, dtype=np.int16).reshape(-1, 3),
            offsets=offs, data=data,
            syn_t=np.asarray(self.syn_t, dtype=np.int64),
            syn_zyx=np.asarray(self.syn_zyx, dtype=np.float64).reshape(-1, 3),
            syn_parent=np.asarray(self.syn_parent, dtype=np.int64),
        )
        manifest = {
            "generator": GEN_VERSION, "host": self.host.name,
            "seed": self.seed, "n_events": len(self.events),
            "n_syn_nodes": len(self.syn_t), "events": self.events,
        }
        (out_dir / f"{tag}.json").write_text(json.dumps(manifest, indent=1))
        return npz


def node_arms(manifest: dict) -> list[str]:
    """Per-synthetic-node arm label, reconstructed from the manifest.

    Nodes are appended in event-acceptance order; each event consumed a known
    count (faint/birth_b2: `span`, birth_b1: `n_nodes`). Pinned by test.
    """
    arms: list[str] = []
    for ev in manifest["events"]:
        n = ev["span"] if ev["type"] in ("faint", "birth_b2") else ev["n_nodes"]
        if ev["type"] == "faint":
            arm = "faint"
        elif ev["type"] == "link_conflict":
            arm = f"link:{ev['family']}"
        else:
            arm = "birth"
        arms.extend([arm] * n)
    return arms


def load_variant(npz_path: Path):
    """Returns (per-frame patch lists, syn arrays) from a saved variant."""
    z = np.load(npz_path)
    per_t: dict[int, list] = defaultdict(list)
    for k in range(len(z["origins"])):
        t, gz, gy, gx = (int(v) for v in z["origins"][k])
        sh = tuple(int(v) for v in z["shapes"][k])
        a, b = int(z["offsets"][k]), int(z["offsets"][k + 1])
        per_t[t].append(((gz, gy, gx),
                         z["data"][a:b].astype(np.float32).reshape(sh)))
    return per_t, {"t": z["syn_t"], "zyx": z["syn_zyx"],
                   "parent": z["syn_parent"]}


def compose_frame(cache_frame: np.ndarray, patch_list: list) -> np.ndarray:
    out = np.asarray(cache_frame, dtype=np.float32).copy()
    add_patches(out, patch_list)
    return out
