"""Fixed constants of the inference stack (submission 060) and the weight-file layout.

The weights directory is the weights dataset, flat: every file is looked up by name.
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent

FOLDS = 5
SEED = 42

# ---- stage ① detection: three members, each the mean of its five fold engines (TensorRT fp16) --------------
DETECTORS = [('dl_284', 0.3), ('dl_275', 0.3), ('dl_492', 0.4)]
DET_THRESHOLD = 0.2
NMS_KERNEL = [3, 7, 7]
DET_BLUR_SIGMA = 0.7
DET_TTA = 'none'

# ---- stage ② matcher (five folds, logit mean) and dense flow ----------------------------------------------
MATCHER = [f'dl_401_fold{f}_last.ckpt' for f in range(FOLDS)]
FLOW = 'ssl_flow_flow_164_ep48.pt'

# ---- stage ④ division models (div3f encoder / decoder engines, five folds per role) -----------------------
DIV_TTA = 'rot4'
DIV_GRID = (1, 2, 2)
DIV_ROLES = {
    'parent': [f'div_415_f{f}_calpick_ep{ep:03d}' for f, ep in enumerate((18, 18, 12, 12, 12))],
    'pre': [f'div_478_pre_fold{f}_calpick' for f in range(FOLDS)],
    'post': [f'div_478_post_fold{f}_calpick' for f in range(FOLDS)],
    'native_parent': [f'div_762_f{f}_calpick' for f in range(FOLDS)],
}
DIV_FRAMES = {'parent': 3, 'pre': 3, 'post': 3, 'native_parent': 5}
DIV_MANIFEST = 'div3f_engines.json'

# ---- calibration (fit-full on all train movies) and lineage optimization ----------------------------------
CALIBRATOR = 'model_fold-1.json'          # with div797.json, div846.json, native/wide .txt/.ubj beside it
LP_ARGS = ['--include-divisions', '--node-adjustment', 'clamped', '--division-postprocess', 'legacy',
           '--gap-lmax', '3', '--rounds', '3', '--time-limit', '120']
POOL_SPLITS = (106,)   # sorted positions that start a new LP pool: the hidden test's private [0,106) / public [106,D)


def engine_names():
    """Every TensorRT engine the run loads."""
    names = [f'{name}_fold{f}_last.engine' for name, _ in DETECTORS for f in range(FOLDS)]
    names += [f'{stem}_{kind}.engine' for stems in DIV_ROLES.values() for stem in stems for kind in ('enc', 'dec')]
    return names


MD5SUMS = 'md5sums.txt'   # `md5sum` output over every file of the weights dataset


def required_files():
    return [*engine_names(), DIV_MANIFEST, MD5SUMS, *MATCHER, FLOW, CALIBRATOR,
            'div797.json', 'div846.json', 'native.txt', 'wide.txt', 'native.ubj', 'wide.ubj']


def thread_env(n=1):
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
        os.environ.setdefault(key, str(n))
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
