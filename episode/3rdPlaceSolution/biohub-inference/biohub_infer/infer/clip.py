"""Upstream: per test movie, stage ① detection -> ② matcher / flow / candidate graph -> ④ division scores.

Per movie it writes, under the upstream directory,

    graphs/<sid>.npz           nodes, LP wiring edges, radius-12 top-5 candidates, second-round top-5
    confidence/<sid>.npz       blurred detector heatmap value at every node
    div_scores/<sid>.npz       score (parent), pre, post, native_parent
    division_graphs/<sid>.npz  the radius-16 extra-candidate graph

Every model is the five-fold ensemble: detector members average their fold engines' heatmaps, the matcher
averages the fold logits (EnsembleMatcher), the div3f roles average the fold sigmoids.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from biohub_infer import config as C

SUBDIRS = ('graphs', 'confidence', 'div_scores', 'division_graphs')


# ---- ② matcher trunk: one decoded batch per frame, every fold's embeddings concatenated ----------------------
def detect_sample_embed(models, store, sid, cfg, device, det_peaks):
    """Per frame: (peaks, concat of each model's sampled embeddings along axis 1)."""
    import torch
    from biohub_infer.core.dataset import get_downsample, to_ds
    from biohub_infer.core.embedding import sample_embeddings
    ds = get_downsample(cfg)
    ds_on = bool((ds != 1).any())
    T = store.shape[sid][0]
    bs = int(cfg.test.frames_per_batch)
    amp = bool(cfg.test.get('amp', False))
    dev = device.type if hasattr(device, 'type') else str(device).split(':')[0]
    out = []
    with torch.no_grad():
        for t0 in range(0, T, bs):
            frames = [store.frame(sid, t) for t in range(t0, min(T, t0 + bs))]
            xt = torch.from_numpy(np.stack(frames))[:, None].to(device)
            embs = []
            for model in models:
                with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                    _, emb = model(xt)
                embs.append(emb.float())
            for b in range(xt.shape[0]):
                pk_np = det_peaks[t0 + b].astype(np.int32)
                pos_np = to_ds(pk_np, ds) if ds_on else pk_np
                pos = torch.from_numpy(pos_np.astype(np.float32)).to(device)
                ev = [sample_embeddings(emb[b], pos, tuple(xt.shape[2:])).cpu().numpy().astype(np.float32)
                      for emb in embs]
                out.append((pk_np, np.concatenate(ev, axis=1)))
    return out


class EnsembleMatcher:
    """Fold matchers on their slices of the concatenated features; the three outputs averaged over folds."""

    def __init__(self, matchers):
        self.matchers = list(matchers)

    def __call__(self, fa, ga, fb, gb, radius, candidate_offset_a=None):
        import torch
        n = len(self.matchers)
        outs = [m(a, ga, b, gb, radius, candidate_offset_a=candidate_offset_a)
                for m, a, b in zip(self.matchers, torch.chunk(fa, n, dim=1), torch.chunk(fb, n, dim=1))]
        return tuple(torch.stack([o[i] for o in outs]).mean(0) for i in range(3))


class EnsembleEmb:
    def __init__(self, models):
        self.models = list(models)
        self.matcher = EnsembleMatcher(m.matcher for m in self.models)


# ---- ④ div3f parent / pre / post with rot4 TTA and encoder reuse across time windows ---------------------
def div3f_scores_roles_reuse(roles, store, sid, nodes, device, batch=8):
    import torch
    from biohub_infer.core.dataset import to_ds
    from biohub_infer.div3f.data import apply_d4_points
    from biohub_infer.div3f.model import sample_logits
    from biohub_infer.infer.common import TTA_VARIANTS
    variants = TTA_VARIANTS[C.DIV_TTA]
    by_t = {}
    for i, node in enumerate(nodes):
        by_t.setdefault(int(node[0]), []).append(i)
    times = sorted(by_t)
    T = store.shape[sid][0]
    frames = torch.stack([torch.from_numpy(store.frame(sid, t)) for t in range(T)]).to(device)
    shape = tuple(frames.shape[1:])                     # (Z, Y, X) on the div3f grid; Y == X
    windows = {t: (max(t - 1, 0), t, min(t + 1, T - 1)) for t in times}
    pts = {t: to_ds(nodes[by_t[t], 1:].astype(np.float64), store.ds) for t in times}
    scores = {role: np.zeros(len(nodes), np.float32) for role in roles}
    dev = str(device).split(':')[0]
    denominators = {role: len(models) * len(variants) for role, models in roles.items()}
    with torch.no_grad():
        for k, flip in variants:
            view = torch.rot90(frames, k, dims=(-2, -1))
            if flip:
                view = torch.flip(view, dims=(-1,))
            moved = {t: torch.from_numpy(apply_d4_points(p, k, flip, shape).astype(np.float32)).to(device)
                     for t, p in pts.items()}
            for role, models in roles.items():
                for model in models:
                    cache = {}                          # frame index -> per-stage (1, C, ...) features
                    for start in range(0, len(times), batch):
                        chunk = times[start:start + batch]
                        new = sorted({u for t in chunk for u in windows[t]} - cache.keys())
                        with torch.autocast(dev, dtype=torch.float16, enabled=dev.startswith('cuda')):
                            if new:
                                feats = model.encode(view[new].unsqueeze(1))
                                for i, u in enumerate(new):
                                    cache[u] = [f[i:i + 1] for f in feats]
                            stacked = [torch.cat([torch.cat([cache[u][s] for u in windows[t]], 1) for t in chunk])
                                       for s in range(len(cache[chunk[0]]))]
                            logits = model.decode(stacked, shape)
                        for b, t in enumerate(chunk):
                            scores[role][by_t[t]] += (torch.sigmoid(sample_logits(logits[b:b + 1].float(), moved[t]))
                                                      .cpu().numpy() / denominators[role])
                        for u in [u for u in cache if u < chunk[-1]]:   # later windows start at >= chunk[-1]
                            del cache[u]
    return scores


def aligned_confidence(nodes, peaks, confidences):
    """Detector confidence per LP node; fails if the wiring reordered the detector nodes."""
    if len(peaks) != len(confidences):
        raise ValueError('confidence frame count differs from detections')
    expected, values = [], []
    for t, (p, c) in enumerate(zip(peaks, confidences)):
        p, c = np.asarray(p).reshape(-1, 3), np.asarray(c)
        if c.shape != (len(p),) or not np.isfinite(c).all():
            raise ValueError('invalid detector confidence')
        expected.append(np.column_stack([np.full(len(p), t), p]))
        values.append(c.astype(np.float16).astype(np.float32))   # the precision the calibrators were fit on
    expected = np.concatenate(expected) if expected else np.empty((0, 4))
    if not np.array_equal(nodes, expected):
        raise ValueError('matcher nodes do not align with detector confidence')
    return np.concatenate(values) if values else np.empty(0, np.float32)


class CachedZarrArray:
    """`array[t]` from the CellStore raw cache (filled by the detector pass), zarr on a miss."""

    def __init__(self, array, raw_cache, sid):
        self.array, self.raw, self.sid = array, raw_cache, sid
        self.shape = array.shape

    def __getitem__(self, t):
        key = (self.sid, int(t))
        raw = None if self.raw is None else self.raw.get(key)
        if raw is None:
            raw = self.array[t]
            if self.raw is not None:
                self.raw[key] = raw
        return raw


def resolve_weights(root):
    """{file name: path} for every file the run needs, each found exactly once under root."""
    paths, problems = {}, []
    for name in C.required_files():
        matches = sorted(Path(root).rglob(name))
        if len(matches) != 1:
            problems.append(f'{name}: {len(matches)} matches')
        else:
            paths[name] = matches[0]
    if problems:
        raise FileNotFoundError(f'weights under {root}: ' + '; '.join(problems))
    return paths


def load_models(paths, device):
    from biohub_infer.infer.common import load_matcher
    from biohub_infer.ssl_flow.inference import load_model as load_flow_model
    from biohub_infer.trt import TrtModel, load_div3f_roles
    det = [(TrtModel(paths[f'{name}_fold{f}_last.engine'], device), weight / C.FOLDS)
           for name, weight in C.DETECTORS for f in range(C.FOLDS)]
    cfgs, matchers = zip(*(load_matcher(paths[name], device) for name in C.MATCHER))
    if len({int(m.matcher.proj.in_features) for m in matchers}) != 1:
        raise ValueError('matcher folds differ in feature width')
    flow, cfg_flow, _ = load_flow_model(str(paths[C.FLOW]), device)
    roles = load_div3f_roles(paths, C.DIV_ROLES, paths[C.DIV_MANIFEST], device)
    for role, models in roles.items():
        if any(m.n_frames != C.DIV_FRAMES[role] for m in models):
            raise ValueError(f'{role}: engines must take {C.DIV_FRAMES[role]} frames')
    return det, (cfgs[0], EnsembleEmb(matchers)), (cfg_flow, flow), roles


def infer_clip(sid, models, stores, test_dir, device, out, raw_cache):
    import torch
    import zarr
    from omegaconf import OmegaConf
    from biohub_infer.infer import linking
    from biohub_infer.infer import common
    from biohub_infer.infer.div797_parent import flow_with_dense
    from biohub_infer.infer.div797_parent_reuse import parent_scores_reuse
    from biohub_infer.infer.second_round import second_round
    from biohub_infer.infer.lp_wiring import lp_wiring, top_candidates
    from biohub_infer.lineage.div797_features import radius16_graph
    det, (cfg_emb, emb), (cfg_flow, flow), roles = models
    store_det, store_emb, store_div = stores
    cfg_det = OmegaConf.create({'model': {'det_threshold': C.DET_THRESHOLD, 'nms_kernel': C.NMS_KERNEL},
                                'test': {'frames_per_batch': 1}})
    times = {}
    tic = time.monotonic()
    # ① detection
    peaks, conf = common.detect_tta(det, store_det, sid, device, cfg_det)
    torch.cuda.synchronize()
    times['det'] = time.monotonic() - tic
    tic = time.monotonic()
    # ② matcher embeddings, flow, scores, production cell, second round, LP wiring
    per_t = detect_sample_embed(emb.models, store_emb, sid, cfg_emb, device, peaks)
    z = zarr.open(str(Path(test_dir) / f'{sid}.zarr'), mode='r')
    array = CachedZarrArray(z['0'], raw_cache, sid)
    offsets, dense = flow_with_dense(array, dict(z.attrs)['image_statistics']['quantiles'], peaks, flow,
                                     cfg_flow.data.downsample, device)
    scores = linking.build_matcher_scores(per_t, emb, cfg_emb, device, offsets)
    sims, revs, syms = (f(scores) for f in [linking.matcher_bidir_probs, linking.matcher_rev_probs, linking.matcher_sym_probs])
    _, initial = linking.link_embed(per_t, common.MAX_LINK_UM, common.LAMBDA, common.DRIFT_ITERS, sims,
                                ext_um=common.GATE_EXT_UM, p_th=common.GATE_PTH, flow=offsets)
    r2i, r2p = second_round(per_t, initial, emb, cfg_emb, device, offsets, k=common.R2_TOPK)
    if r2i.shape != r2p.shape or r2i.ndim != 2 or r2i.shape[1] != common.R2_TOPK:
        raise ValueError(f'second round requires width {common.R2_TOPK}')
    nodes, edges = lp_wiring(per_t, sims, revs, syms, offsets, r2i, r2p)
    confidence = aligned_confidence(nodes, peaks, conf)
    ci, cp, cr, cs, cd = top_candidates(per_t, sims, revs, syms, offsets)
    torch.cuda.synchronize()
    times['matcher_flow_lp'] = time.monotonic() - tic
    tic = time.monotonic()
    # ④ division scores
    phase = div3f_scores_roles_reuse({k: roles[k] for k in ('parent', 'pre', 'post')}, store_div, sid, nodes, device)
    phase = dict(score=phase['parent'], pre=phase['pre'], post=phase['post'])
    phase['native_parent'] = parent_scores_reuse(roles['native_parent'], store_div, sid, nodes, device, dense)
    del dense
    for v in phase.values():
        if v.shape != (len(nodes),) or not np.isfinite(v).all() or np.any((v < 0) | (v > 1)):
            raise ValueError(f'{sid}: invalid phase scores')
    torch.cuda.synchronize()
    times['div3f'] = time.monotonic() - tic
    tic = time.monotonic()
    g = dict(nodes=nodes, edges=np.asarray(edges, np.int64).reshape(-1, 2), cand_idx=ci, cand_prob=cp, cand_rev=cr,
             cand_sym=cs, cand_dist=cd, r2_idx=r2i, r2_prob=r2p)
    wide = radius16_graph(g, phase, per_t, offsets, emb.matcher, device)
    times['radius16'] = time.monotonic() - tic
    out = Path(out)
    for name in SUBDIRS:
        (out / name).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'graphs' / f'{sid}.npz', **g)
    np.savez_compressed(out / 'confidence' / f'{sid}.npz', nodes=nodes, confidence=confidence)
    np.savez_compressed(out / 'div_scores' / f'{sid}.npz', nodes=nodes, **phase)
    np.savez_compressed(out / 'division_graphs' / f'{sid}.npz', **wide)
    return len(nodes), times


def run(ids, weights, test_dir, out, tag='[infer]'):
    """Infer every movie of ids on the visible GPU (one process, models loaded once)."""
    import torch
    from biohub_infer.core.dataset import CellStore, get_downsample
    from biohub_infer.infer.common import check_device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    check_device(device)
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = load_models(resolve_weights(weights), device)
    raw_cache = {}
    stores = [CellStore(test_dir, ids, with_gt=False, downsample=ds, raw_cache=raw_cache)
              for ds in [np.ones(3, int), get_downsample(models[1][0]), np.array(C.DIV_GRID)]]
    started = time.monotonic()
    for k, sid in enumerate(ids, 1):
        count, times = infer_clip(sid, models, stores, test_dir, device, out, raw_cache)
        raw_cache.clear()
        print(f'{tag} {k}/{len(ids)} {sid}: {count} nodes, ' + ', '.join(f'{n} {v:.1f}s' for n, v in times.items())
              + f', elapsed {time.monotonic() - started:.0f}s', flush=True)
