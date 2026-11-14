"""Frozen Current-TH inference, with a historical input and probability replay gate."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import zarr
from scipy.ndimage import shift

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B / 'legacy'))
from biohub_division_crop_classifier import SPACING_ZYX, crop_lazy
from biohub_division_detected_geometry_classifier import Peak, load_peaks
from biohub_division_flow_aligned_diff_classifier import SparseLocalFlow, read_samples
from biohub_division_bigru_shift9_ensemble import make_image
from biohub_division_sliding_t3_mil import SlidingT3Residual
from prepare_biohub_division_peak_heatmap_cache import rasterize

ROOT = Path(os.environ.get('BIOHUB_ROOT', '/biohub-gpu'))
SOURCE = ROOT / 'parent_models'
SEEDS = (20260817, 20260818)
SHAPE = (12, 48, 48)

def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2)); temp.replace(path)

class ParentScorer:
    def __init__(self, device=None):
        self.device = torch.device(device or os.environ.get('BIOHUB_PARENT_DEVICE','cuda')); self.models = []; self.provenance = []
        torch.set_num_threads(2)
        for seed in SEEDS:
            directory = SOURCE / f'seed{seed}_hard10_heatmap/arms/sliding_t3_bigru'
            path = directory / 'checkpoint_best.pt'
            summary = json.loads((directory / 'summary.json').read_text())
            mean = torch.tensor(summary['normalization']['mean'], dtype=torch.float32)
            std = torch.tensor(summary['normalization']['std'], dtype=torch.float32)
            if self.models:
                assert torch.equal(mean, self.mean) and torch.equal(std, self.std)
            self.mean, self.std = mean, std
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            model = SlidingT3Residual('bigru', 14)
            state = dict(checkpoint['model'])
            missing = set(model.state_dict()) - set(state)
            if missing:
                assert missing == {'role_residual.weight', 'role_residual.bias'}, missing
                for key in missing:
                    assert torch.count_nonzero(model.state_dict()[key]) == 0
                    state[key] = model.state_dict()[key]
            model.load_state_dict(state, strict=True)
            self.models.append(model.to(self.device).eval())
            self.provenance.append(dict(seed=seed, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def predict(self, raw, heatmap):
        images = torch.stack([make_image(r, h, np.zeros(3), np.zeros(3), self.mean, self.std) for r,h in zip(raw,heatmap)]).to(self.device)
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=os.environ.get('BIOHUB_PARENT_AMP','bf16')=='bf16'):
            values = [torch.sigmoid(model(images)['binary'].float()).cpu().numpy().reshape(-1) for model in self.models]
        result = np.stack(values, axis=1)
        if not np.isfinite(result).all(): raise RuntimeError('Nonfinite parent inference')
        return result

class VideoCrops:
    """Raw-zarr version of the training cache contract; no annotations are read."""
    def __init__(self, path, peaks):
        self.stem = Path(path).stem
        self.array = zarr.open_group(str(path), mode='r')['0']
        self.peaks = peaks
        self.flow = SparseLocalFlow(peaks, SimpleNamespace(residual_gate_um=7., exclude_radius_um=3., local_radius_um=50., max_local_support=64, min_local_support=3, sigma_um=20., alignment_method='gaussian_translation'))

    @lru_cache(maxsize=7)
    def frame(self, t):
        # Neighbour input is clipped to the movie boundary, as in nnU-Net.
        t = min(max(t, 0), self.array.shape[0]-1)
        a = np.asarray(self.array[t], dtype=np.float32)
        return (a-a.mean()) / max(float(a.std()), 1e-8)

    def inputs(self, t, center, width=3):
        current = f'{self.stem}_t{t:0{width}d}'
        center = np.asarray(center, dtype=np.float64)
        raw = np.zeros((9, *SHAPE), dtype=np.float16)
        heatmap = np.zeros((3, *SHAPE), dtype=np.float16)
        for offset in range(-2,3):
            frame = t+offset
            neighbor = f'{self.stem}_t{frame:0{width}d}' if frame >= 0 else None
            vector = np.zeros(3)
            if offset < 0:
                vector = self.flow.estimate(neighbor, current, center*SPACING_ZYX, 'target')[1] / SPACING_ZYX
            elif offset > 0:
                vector = -self.flow.estimate(current, neighbor, center*SPACING_ZYX, 'source')[1] / SPACING_ZYX
            crop = crop_lazy(self.frame(frame)[None], center.astype(int), SHAPE)[0].astype(np.float16).astype(np.float32)
            raw[offset+4] = shift(crop, vector, order=1, mode='constant', cval=0, prefilter=False) if offset else crop
            if abs(offset) <= 1:
                h = rasterize(self.peaks.get(neighbor, []), center, SHAPE, 1.5)
                heatmap[offset+1] = shift(h, vector, order=1, mode='constant', cval=0, prefilter=False) if offset else h
        return raw, heatmap

def peaks_from_rows(stem, rows):
    peaks = defaultdict(list)
    for idx,row in enumerate(rows):
        case = f'{stem}_t{int(row["t"]):03d}'
        peaks[case].append(Peak(case, row['z'], row['y'], row['x'], row['node_score'], 'unknown', 'submitted_detector', idx))
    return dict(peaks)

def score_video(scorer, stem, rows, data_dir, output):
    # Score all detector nodes: no GT gating, old-score substitution, or implicit negatives.
    crops = VideoCrops(Path(data_dir)/f'{stem}.zarr', peaks_from_rows(stem,rows))
    scored = []
    for start in range(0,len(rows),32):
        batch = rows[start:start+32]
        pairs = [crops.inputs(int(r['t']), [r[k] for k in ('z','y','x')]) for r in batch]
        probs = scorer.predict([v[0] for v in pairs],[v[1] for v in pairs])
        for i,(r,p) in enumerate(zip(batch,probs),start):
            scored.append(dict(node_index=i, **r, parent_prior=float(p.mean()), seed20260817=float(p[0]), seed20260818=float(p[1])))
        if start % 512 == 0: print('parent',stem,start,len(rows),flush=True)
    save(output,dict(stem=stem,nodes=scored,models=scorer.provenance,coverage_all_nodes=True))


