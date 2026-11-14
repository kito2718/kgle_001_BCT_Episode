"""Prediction-only daughter-pair appearance reranker for the one-stage worker."""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import zarr

SEEDS = (20260925, 20261934, 20262943, 20263952, 20264961)


def patch_features(frame, coordinate):
    z, y, x = (int(round(v)) for v in coordinate)
    assert 0 <= z < frame.shape[0] and 0 <= y < frame.shape[1] and 0 <= x < frame.shape[2]
    def cube(rz, ryx):
        return frame[max(0, z-rz):min(frame.shape[0], z+rz+1),
                     max(0, y-ryx):min(frame.shape[1], y+ryx+1),
                     max(0, x-ryx):min(frame.shape[2], x+ryx+1)]
    core, outer = cube(1, 2), cube(3, 5)
    return dict(centre=float(frame[z, y, x]), core_mean=float(core.mean()),
                core_max=float(core.max()), core_std=float(core.std()),
                outer_mean=float(outer.mean()),
                contrast=float(core.mean() - outer.mean()))


def rescore(out, stem, data_dir, model_dir, fold):
    out, data_dir, model_dir = map(Path, (out, data_dir, model_dir))
    candidate = pickle.loads((out / 'candidates' / f'{stem}.pkl').read_bytes())
    rows = candidate['node_rows']
    score_path = out / 'pair_scores_min01' / f'{stem}.json'
    scores = json.loads(score_path.read_text())
    pairs = scores['rows']
    names = json.loads((model_dir / 'feature_names.json').read_text())
    geom_names, image_names = names['geom_traj'], names['image']
    assert len(geom_names) == 73 and len(image_names) == 18
    if not pairs:
        scores['model'] = f'daughter_image_lgb5_heldout_fold{fold}'
        score_path.write_text(json.dumps(scores) + '\n')
        return dict(stem=stem, pairs=0, candidate_contract=candidate['contract'])
    indices = defaultdict(list)
    for i, row in enumerate(rows):
        indices[f'{stem}_t{int(row["t"]):03d}'].append(i)
    daughters = [(indices[row['target_case_id']][int(row['first'])],
                  indices[row['target_case_id']][int(row['second'])]) for row in pairs]
    needed = {node for pair in daughters for node in pair}
    by_frame = defaultdict(list)
    for node in needed:
        by_frame[int(rows[node]['t'])].append(node)
    zarr_array = zarr.open_group(str(data_dir / f'{stem}.zarr'), mode='r')['0']
    features = {}
    for frame_index, node_ids in sorted(by_frame.items()):
        frame = np.asarray(zarr_array[frame_index], np.float32)
        frame = (frame - frame.mean()) / max(float(frame.std()), 1e-8)
        for node in node_ids:
            coordinate = [rows[node][key] for key in 'zyx']
            features[node] = patch_features(frame, coordinate)
    matrix = []
    for row, (a, b) in zip(pairs, daughters):
        pair_features = {}
        for name in features[a]:
            left, right = features[a][name], features[b][name]
            pair_features[f'image_{name}_min'] = min(left, right)
            pair_features[f'image_{name}_max'] = max(left, right)
            pair_features[f'image_{name}_absdiff'] = abs(left - right)
        matrix.append([float(row[key]) for key in geom_names] +
                      [pair_features[key] for key in image_names])
    matrix = np.asarray(matrix, np.float32)
    assert matrix.shape == (len(pairs), 91) and np.isfinite(matrix).all()
    models = [lgb.Booster(model_file=str(model_dir / f'image_pair_fold{fold}_seed{seed}.txt'))
              for seed in SEEDS]
    prediction = np.mean([model.predict(matrix, num_threads=1) for model in models], axis=0)
    groups = defaultdict(list)
    for i, (row, value) in enumerate(zip(pairs, prediction)):
        row['pair_score'] = float(value)
        groups[int(row['node_index'])].append(i)
    for group in groups.values():
        for rank, i in enumerate(sorted(group, key=lambda i: (-prediction[i], i)), 1):
            pairs[i]['pair_rank'] = rank
            pairs[i]['pair_rank_norm'] = 1. - (rank-1) / max(1, len(group)-1)
    scores['model'] = f'daughter_image_lgb5_heldout_fold{fold}'
    scores['labels_read_at_inference'] = False
    score_path.write_text(json.dumps(scores) + '\n')
    return dict(stem=stem, pairs=len(pairs), candidate_contract=candidate['contract'],
                model=scores['model'], labels_read=False)


if __name__ == '__main__':
    import os
    result = rescore(os.environ['BIOHUB_CHOICE_OUT'],
                     (Path(os.environ['BIOHUB_CHOICE_OUT']) / 'config/one.txt').read_text().strip(),
                     os.environ['BIOHUB_DATA_DIR'], os.environ['BIOHUB_IMAGE_LGB5_MODELS'],
                     int(os.environ['BIOHUB_IMAGE_LGB5_FOLD']))
    print(json.dumps(result), flush=True)
