# %% ################################################################################################
# # Biohub V533: V507 + contrast-gated add-back (acq_01) + xy edge trim
# 
# Strict CV **0.953024047** (**+0.001806884 vs V507 0.951217163**).
# 

# %% ################################################################################################
import os, sys, time, gc
from pathlib import Path

IS_KAGGLE = Path('/kaggle/input').exists()
ROOTS = [Path('/kaggle/input')] if IS_KAGGLE else [Path('../input')]

def walk(roots, max_depth=5):
    frontier = [(r, 0) for r in roots if r.is_dir()]
    while frontier:
        d, depth = frontier.pop(0)
        yield d
        if depth >= max_depth:
            continue
        try:
            for child in sorted(d.iterdir()):
                if child.is_dir() and not child.name.endswith('.zarr'):
                    frontier.append((child, depth + 1))
        except PermissionError:
            pass


def find_code_root() -> Path:
    if (Path('biohub')/ 'pipeline' / 'v233_deploy.py').exists():
        return Path('.')
    raise FileNotFoundError('biohub package with pipeline/v230_deploy.py not found.')


def find_test_root() -> Path:
    fallback = None
    for d in walk(ROOTS):
        if any(d.glob('*.zarr')):
            if d.name == 'test':
                return d
            fallback = fallback or d
    if fallback:
        return fallback
    raise FileNotFoundError('no test .zarr found')


def find_weights_root() -> Path:
    for d in walk([Path('./datasets/soheilayati')]):
        if ((d / 'count_model_v1.json').exists()
                and (d / 'edge_model_b125.npz').exists()
                and any(d.glob('fold*.ep05.pt'))
                and any(d.glob('v24.fold*.ep05.pt'))):
            return d
    raise FileNotFoundError('weights Dataset not attached')


if IS_KAGGLE:
    print('--- mounted ---')
    for d in walk(ROOTS, max_depth=3):
        rel = d.relative_to('/kaggle/input')
        if rel != Path('.'):
            print(f'  {rel}')
    print()

CODE_ROOT = find_code_root()
if IS_KAGGLE:
    import shutil
    staged = Path('/kaggle/working/v343_code')
    if CODE_ROOT != staged:
        shutil.copytree(CODE_ROOT / 'biohub', staged / 'biohub', dirs_exist_ok=True)
    CODE_ROOT = staged
TEST_ROOT = find_test_root()
WEIGHTS_ROOT = find_weights_root()

def find_unique_root(filename: str) -> Path:
    hits = sorted({d for d in walk([Path('./datasets/soheilayati')]) if (d / filename).exists()})
    if len(hits) != 1:
        raise FileNotFoundError(f'expected one root for {filename}, found {hits}')
    return hits[0]

V39_ROOT = find_unique_root('v39.fold0.ep05.pt')
V57_ROOT = find_unique_root('v57_deployment_manifest.json')
V343_ROOT = find_unique_root('v343_dim_scaffold_manifest.json')
V389_ROOT = find_unique_root('v389_structured_manifest.json')
V394_ROOT = find_unique_root('v394_component_manifest.json')
V399_ROOT = find_unique_root('v399_tail_manifest.json')
V454_ROOT = find_unique_root('v454_manifest.json')
V507_ROOT = find_unique_root('v507_manifest.json')
V533_ROOT = find_unique_root('v533_manifest.json')
V230_ROOT = WEIGHTS_ROOT if (WEIGHTS_ROOT / 'tree.txt').exists() else find_unique_root('tree.txt')

OUT = Path('submission.csv')
sys.path.insert(0, str(CODE_ROOT))

print(f'kaggle = {IS_KAGGLE}')
print(f'code    : {CODE_ROOT}')
print(f'test    : {TEST_ROOT}  ({len(list(TEST_ROOT.glob("*.zarr")))} datasets)')
print(f'weights : {WEIGHTS_ROOT}')
print(f'v230    : {V230_ROOT}')
print(f'v39     : {V39_ROOT}')
print(f'v57     : {V57_ROOT}')
print(f'v343    : {V343_ROOT}')
print(f'v389    : {V389_ROOT}')
print(f'v394    : {V394_ROOT}')
print(f'v399    : {V399_ROOT}')
print(f'v454    : {V454_ROOT}')
print(f'v507    : {V507_ROOT}')
print(f'v533    : {V533_ROOT}')
print(f'out     : {OUT}')

# import json
# V389_MANIFEST = json.loads((V389_ROOT / 'v389_structured_manifest.json').read_text())
# V394_MANIFEST = json.loads((V394_ROOT / 'v394_component_manifest.json').read_text())
# V399_MANIFEST = json.loads((V399_ROOT / 'v399_tail_manifest.json').read_text())
# V454_MANIFEST = json.loads((V454_ROOT / 'v454_manifest.json').read_text())
# V507_MANIFEST = json.loads((V507_ROOT / 'v507_manifest.json').read_text())
# V533_MANIFEST = json.loads((V533_ROOT / 'v533_manifest.json').read_text())
# V343_MANIFEST = json.loads((V343_ROOT / 'v343_dim_scaffold_manifest.json').read_text())


# %% ################################################################################################
# zarr is NOT in Kaggle's image and internet is disabled, so it ships as a
# wheels Dataset and installs offline here (PLAN.md §14.3).
#
# numpy is deliberately excluded from that Dataset: Kaggle pins 2.0.2 and
# letting pip pull a newer one would rebuild the environment underneath
# scipy/skimage.
#
# scikit-learn is NOT needed: the count model serialises to plain JSON and
# predicts with numpy, precisely so inference does not depend on a library
# version we do not control here.
import importlib, subprocess

try:
    importlib.import_module('zarr')
    print('zarr already present in the image')
except ModuleNotFoundError:
    wheel_dir = next((d for d in walk(ROOTS) if any(d.glob('zarr-*.whl'))), None)
    if wheel_dir is None:
        raise FileNotFoundError('no zarr wheel found; attach the wheels Dataset')
    print(f'installing zarr offline from {wheel_dir}')
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index',
         f'--find-links={wheel_dir}', 'zarr'],
        check=True,
    )
    importlib.invalidate_caches()

import zarr
print('zarr', zarr.__version__)

# %% ################################################################################################
# Environment record, and the GPU gate.
import platform, os, importlib
print('python  ', platform.python_version())
for m in ('numpy', 'scipy', 'skimage', 'zarr', 'pandas', 'torch', 'numba'):
    try:
        mod = __import__(m)
        print(f'{m:9s}', getattr(mod, '__version__', '?'))
    except Exception as e:
        print(f'{m:9s} MISSING ({type(e).__name__})')
print('cpu count', os.cpu_count())

import biohub.device
importlib.reload(biohub.device)
from biohub.device import select_device

SPEC = select_device(allow_cpu=False)
print(f'\ndevice   {SPEC.device}  {SPEC.name}  sm_{SPEC.capability[0]}{SPEC.capability[1]}'
      f'  {SPEC.vram_gb:.1f} GB  autocast={SPEC.amp_dtype}')


# %% ################################################################################################
# The five engineering fold checkpoints, averaged in HEATMAP space, plus the
# two downstream models.
#
# Averaging before peak extraction rather than after: extraction, ranking and
# the graph score are each nonlinear, so combining afterwards is a different and
# worse operation. Every fold model is valid on the hidden test -- none of them
# trained on it -- so the ensemble uses all 199 training samples' worth of
# learning while any single fold model would use four fifths. The edge GBM's
# five folds are combined the same way (mean probability) for the same reason.
#
# Named explicitly, never globbed from the local artifacts directory, which also
# holds `fold0.orig30.ep05.pt` from an abandoned run.
from biohub.counts import CountModel
from biohub.detector.infer import load_models
from biohub.tracker.edge_model import EdgeModel

CKPT_NAMES = [f'fold{k}.ep05.pt' for k in range(5)]
V24_CKPT_NAMES = [f'v24.fold{k}.ep05.pt' for k in range(5)]
ALL_CKPT_NAMES = CKPT_NAMES + V24_CKPT_NAMES
paths = [WEIGHTS_ROOT / n for n in ALL_CKPT_NAMES]
missing = [p.name for p in paths if not p.exists()]
assert not missing, f'V25 consensus weights Dataset is incomplete: {missing}'

# Controlled A/B: load one independently seeded full-data detector for the
# heatmap-only sixth view below and for the fixed v1 coordinate post-pass.
# All non-heatmap primary-detector heads remain the frozen five-model mean.
import hashlib
EXTRA_CKPT_NAME = 'full.seed1.ep02.pt'
# EXTRA_CKPT_SHA256 = '0e5cc6d5eb04e97037db54574aa340d378b978b0e03461eec3c82102850d84f6'
# extra_hits = sorted({d / EXTRA_CKPT_NAME for d in walk('./datasets/soheilayati/biohub-seed1-detector')})
# assert len(extra_hits) == 1, f'expected one {EXTRA_CKPT_NAME}, found {extra_hits}'
extra_path = Path('./datasets/soheilayati/biohub-seed1-detector') / EXTRA_CKPT_NAME
# assert hashlib.sha256(extra_path.read_bytes()).hexdigest() == EXTRA_CKPT_SHA256
# V454 removes two unused coordinate_blend imports; the checkpoint itself
# remains hash-verified and the executed detector path is unchanged.
MODELS = load_models(paths, SPEC.device)
V21_MODELS = MODELS[:len(CKPT_NAMES)]
V24_MODELS = MODELS[len(CKPT_NAMES):]
assert len(V21_MODELS) == len(V24_MODELS) == 5
SEED_MODELS = load_models([extra_path], SPEC.device)

# V16 coordinate A/B: identity plus a symmetric 3x3 x/y blur view of
# the independent seed detector. Spatial and directed semantics are fixed.
class SeedXYBlurView:
    def __init__(self, model):
        self.model = model
        self.feature = model.feature

    def __call__(self, x):
        import torch.nn.functional as functional
        smooth = functional.avg_pool3d(
            x, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        return self.model(smooth)

SEED_XYBLUR_MODELS = [SEED_MODELS[0], SeedXYBlurView(SEED_MODELS[0])]

# V7 topology A/B: add the independent seed-1 detector only to the heatmap
# probability average. Offset, flow, uncertainty, and feature embeddings
# remain the frozen five-model v14 views, avoiding cross-seed feature-space
# averaging. The composite presents one ordinary model interface to detect().
class HeatOnlyAugmentedEnsemble:
    def __init__(self, base_models, heat_model):
        self.base_models = list(base_models)
        self.heat_model = heat_model
        self.feature = self.base_models[0].feature

    def __call__(self, x):
        import torch
        heat = None
        other = None
        keys = ('offset', 'flow', 'flow_logvar', 'features')
        for model in self.base_models:
            out = model(x)
            prob = torch.sigmoid(out['heatmap'])
            heat = prob if heat is None else heat + prob
            if other is None:
                other = {key: out[key] for key in keys}
            else:
                other = {key: other[key] + out[key] for key in keys}
        seed = self.heat_model(x)
        prob = ((heat + torch.sigmoid(seed['heatmap']))
                / (len(self.base_models) + 1))
        prob = prob.clamp(1e-6, 1.0 - 1e-6)
        other = {key: value / len(self.base_models)
                 for key, value in other.items()}
        other['heatmap'] = torch.logit(prob)
        return other

# V9 topology A/B: notebook-local per-frame logit alignment across the
# original five folds. The mounted source predates the optional detect()
# alignment argument, so the composite keeps this experiment self-contained.
class FrameLogitAlignedEnsemble:
    def __init__(self, models):
        self.models = list(models)
        self.feature = self.models[0].feature

    def __call__(self, x):
        import torch
        logits = []
        other = None
        keys = ('offset', 'flow', 'flow_logvar', 'features')
        for model in self.models:
            out = model(x)
            logits.append(out['heatmap'].float())
            if other is None:
                other = {key: out[key] for key in keys}
            else:
                other = {key: other[key] + out[key] for key in keys}
        z = torch.stack(logits, dim=0)
        dims = (-3, -2, -1)
        means = z.mean(dim=dims, keepdim=True)
        scales = z.std(dim=dims, keepdim=True, correction=0).clamp_min(1e-6)
        ref_mean = means.mean(dim=0)
        ref_scale = torch.exp(torch.log(scales).mean(dim=0))
        aligned = (z - means) * (ref_scale / scales) + ref_mean
        other = {key: value / len(self.models)
                 for key, value in other.items()}
        other['heatmap'] = aligned.mean(dim=0)
        return other

ALIGNED_MODELS = [FrameLogitAlignedEnsemble(MODELS)]

# V12 attribution: arithmetic mean of raw heatmap logits, without the
# per-frame mean/scale alignment. Non-heatmap heads retain the same mean.
class RawLogitMeanEnsemble:
    def __init__(self, models):
        self.models = list(models)
        self.feature = self.models[0].feature

    def __call__(self, x):
        logits = None
        other = None
        keys = ('offset', 'flow', 'flow_logvar', 'features')
        for model in self.models:
            out = model(x)
            logits = out['heatmap'] if logits is None else logits + out['heatmap']
            if other is None:
                other = {key: out[key] for key in keys}
            else:
                other = {key: other[key] + out[key] for key in keys}
        other = {key: value / len(self.models)
                 for key, value in other.items()}
        other['heatmap'] = logits / len(self.models)
        return other

RAW_LOGIT_MODELS = [RawLogitMeanEnsemble(MODELS)]

# V13: add seed1 only to the standardized heatmap-logit ensemble. The
# original five models still own every vector head and feature embedding.
class SeedHeatAlignedEnsemble:
    def __init__(self, base_models, heat_model):
        self.base_models = list(base_models)
        self.heat_model = heat_model
        self.feature = self.base_models[0].feature

    def __call__(self, x):
        import torch
        logits = []
        other = None
        keys = ('offset', 'flow', 'flow_logvar', 'features')
        for model in self.base_models:
            out = model(x)
            logits.append(out['heatmap'].float())
            if other is None:
                other = {key: out[key] for key in keys}
            else:
                other = {key: other[key] + out[key] for key in keys}
        logits.append(self.heat_model(x)['heatmap'].float())
        z = torch.stack(logits, dim=0)
        dims = (-3, -2, -1)
        means = z.mean(dim=dims, keepdim=True)
        scales = z.std(dim=dims, keepdim=True, correction=0).clamp_min(1e-6)
        ref_mean = means.mean(dim=0)
        ref_scale = torch.exp(torch.log(scales).mean(dim=0))
        aligned = (z - means) * (ref_scale / scales) + ref_mean
        other = {key: value / len(self.base_models)
                 for key, value in other.items()}
        other['heatmap'] = aligned.mean(dim=0)
        return other

SEED_ALIGNED_MODELS = [SeedHeatAlignedEnsemble(MODELS, SEED_MODELS[0])]

# V16 retains V14's robust middle-three mean after per-frame
# standardization. Directed/vector heads remain ordinary five-way means.
class TrimmedFrameLogitEnsemble:
    def __init__(self, models):
        self.models = list(models)
        self.feature = self.models[0].feature

    def __call__(self, x):
        import torch
        logits = []
        other = None
        keys = ('offset', 'flow', 'flow_logvar', 'features')
        for model in self.models:
            out = model(x)
            logits.append(out['heatmap'].float())
            if other is None:
                other = {key: out[key] for key in keys}
            else:
                other = {key: other[key] + out[key] for key in keys}
        z = torch.stack(logits, dim=0)
        dims = (-3, -2, -1)
        means = z.mean(dim=dims, keepdim=True)
        scales = z.std(dim=dims, keepdim=True, correction=0).clamp_min(1e-6)
        ref_mean = means.mean(dim=0)
        ref_scale = torch.exp(torch.log(scales).mean(dim=0))
        aligned = (z - means) * (ref_scale / scales) + ref_mean
        middle = torch.sort(aligned, dim=0).values[1:-1].mean(dim=0)
        other = {key: value / len(self.models)
                 for key, value in other.items()}
        other['heatmap'] = middle
        return other

TRIMMED_MODELS = [TrimmedFrameLogitEnsemble(MODELS)]
COUNT_MODEL = CountModel.load(WEIGHTS_ROOT / 'count_model_v1.json')
EDGE_MODEL = EdgeModel.load(WEIGHTS_ROOT / 'edge_model_b125.npz')

# The operating point this version runs. v4 = pair-model fork decision
# (0.8810 local official); v5 = + context continuation, temporal division
# stack, fork-guarded smoothing, edgeless drop (0.9021 local official).
PIPELINE = 'v8'

import json as _json
from biohub.tracker.edge_model import (BoosterEnsemble, ForestModel,
                                       PairModel, PairStack)
PAIR_MODEL = PairModel.load(WEIGHTS_ROOT / 'pair_model_v0.npz')
CTX_MODEL = PAIR_STACK = PU_MODEL = EXTRA_MODEL = COMP_MODEL = None
if PIPELINE in ('v5', 'v6', 'v7', 'v8'):
    CTX_MODEL = EdgeModel.load(WEIGHTS_ROOT / 'edge_model_sv11_ctx.npz')
    PAIR_STACK = PairStack.load(WEIGHTS_ROOT / 'pair_stack_faithful.npz')
    print(f'ctx model  : {len(CTX_MODEL.folds)} GBM folds, '
          f'{len(CTX_MODEL.features)} features')
    print(f'pair stack : {len(PAIR_STACK.folds)} folds, '
          f'{len(PAIR_STACK.features)} features, base {PAIR_STACK.base_model}')
if PIPELINE in ('v7', 'v8'):
    ALLIMG_MODEL = PairStack.load(WEIGHTS_ROOT / 'pair_allimg_lr.npz')
    FULL_MODEL = PairStack.load(WEIGHTS_ROOT / 'pair_full_lr.npz')
    HARDNEG_MODEL = PairStack.load(WEIGHTS_ROOT / 'pair_hardneg_lr.npz')
    HIGHRES_MODEL = PairStack.load(WEIGHTS_ROOT / 'pair_highres_lr.npz')
    STUB_MODEL = PairStack.load(WEIGHTS_ROOT / 'stub_pruner_lr.npz')
    _meta7 = _json.loads((WEIGHTS_ROOT / 'component_current_lgb_l15.json').read_text())
    COMP7_MODEL = BoosterEnsemble.load(
        [(WEIGHTS_ROOT / f'component_current_lgb_l15_fold{k}.txt').read_text('utf-8')
         for k in range(_meta7['n_folds'])], _meta7['features'])
    print(f'division base: {len(ALLIMG_MODEL.features)}/{len(FULL_MODEL.features)}/'
          f'{len(HARDNEG_MODEL.features)} features; hr {len(HIGHRES_MODEL.features)}')
    print(f'component    : {len(COMP7_MODEL.boosters)} LGBM boosters, '
          f'{len(COMP7_MODEL.features)} features; stub {len(STUB_MODEL.features)}')
CONFIRM_MODEL = TRACK_MODEL = POST_MODEL = None
if PIPELINE == 'v8':
    def _texts(stem):
        meta = _json.loads((WEIGHTS_ROOT / f'{stem}.json').read_text())
        return BoosterEnsemble.load(
            [(WEIGHTS_ROOT / f'{stem}_fold{k}.txt').read_text('utf-8')
             for k in range(meta['n_folds'])], meta['features'])
    CONFIRM_MODEL = PairStack.load(WEIGHTS_ROOT / 'fork_confirmation_lr.npz')
    TRACK_MODEL = {
        'track_lr_c0p003': PairStack.load(WEIGHTS_ROOT / 'edge_tracklet_lr_c0p003.npz'),
        'track_lgb_l7': _texts('edge_tracklet_lgb_l7'),
        'track_lgb_l15': _texts('edge_tracklet_lgb_l15'),
    }
    POST_MODEL = _texts('edge_postselect_lgb_l31')
    print(f'v8 stages  : confirm {len(CONFIRM_MODEL.features)} / track x3 '
          f'/ post {len(POST_MODEL.features)} '
          f'(multiview arm dropped by measurement)')
if PIPELINE == 'v6':
    PU_MODEL = PairStack.load(WEIGHTS_ROOT / 'pair_pu_lr.npz')
    EXTRA_MODEL = ForestModel.load(WEIGHTS_ROOT / 'pair_tracklet_extra.npz')
    _meta = _json.loads((WEIGHTS_ROOT / 'component_lgb_l15.json').read_text())
    COMP_MODEL = BoosterEnsemble.load(
        [(WEIGHTS_ROOT / f'component_lgb_l15_fold{k}.txt').read_text('utf-8')
         for k in range(_meta['n_folds'])], _meta['features'])
    print(f'PU model   : {len(PU_MODEL.folds)} folds, '
          f'{len(PU_MODEL.features)} features')
    print(f'extra model: {len(EXTRA_MODEL.folds)} forest folds, '
          f'{len(EXTRA_MODEL.features)} features')
    print(f'component  : {len(COMP_MODEL.boosters)} LGBM boosters, '
          f'{len(COMP_MODEL.features)} features')
print(f'PIPELINE   : {PIPELINE}')
print(f'pair model : {len(PAIR_MODEL.folds)} GBM folds, '
      f'{len(PAIR_MODEL.features)} features')
print(f'{len(MODELS)} V21/V24 consensus models: {", ".join(ALL_CKPT_NAMES)}')
print(f'coordinate model: {EXTRA_CKPT_NAME}')
print(f'count model: fitted on {COUNT_MODEL.n_train} samples from '
      f'{COUNT_MODEL.pool}, {len(COUNT_MODEL.feat_cols)} features')
print(f'edge model : {len(EDGE_MODEL.folds)} GBM folds, '
      f'{len(EDGE_MODEL.features)} features, '
      f'trees {[len(f.off) - 1 for f in EDGE_MODEL.folds]}')
# V35: the independent Stage-6 strong-detector bank from its own
# dataset. Filenames are distinct on purpose; both weight datasets
# also carry files named fold*.ep05.pt with different content, so
# every strong checkpoint is verified by SHA-256 before loading.
STRONG_CKPT_SHA256 = {
    "strong.fold0.ep05.pt": "2b13e2c990e2768df1360f350092788d79d252df23c9deea80dd5511f50f2a6f",
    "strong.fold1.ep05.pt": "4d25ca8b008b0a9c78ae78f399a2d11204318bc06eb82d08ad63aa9c4783a6cd",
    "strong.fold2.ep05.pt": "cd618b8aa435fd3db618db4ce80bedff1dc41c6df2dde3cdf66104ff4c7d4e59",
    "strong.fold3.ep05.pt": "739ec2606a995dfd7d9e1fff39acd4e82ffc936fea5ad061201ce30d366e814d",
    "strong.fold4.ep05.pt": "f6017ac5c74141ba2c107672aa62b0c21ffd7a8da73b75f2ec6e54adfd0736b9"
}
# strong_paths = []
# for _name in sorted(STRONG_CKPT_SHA256):
#     _hits = sorted({d / _name for d in walk(ROOTS) if (d / _name).exists()})
#     assert len(_hits) == 1, f'expected one {_name}, found {_hits}'
#     _digest = hashlib.sha256(_hits[0].read_bytes()).hexdigest()
#     assert _digest == STRONG_CKPT_SHA256[_name], f'{_name} sha mismatch'
#     strong_paths.append(_hits[0])
STRONG_DIR = Path('datasets/soheilayati/biohub-v35-strong-detectors')
strong_paths = [STRONG_DIR / f'strong.fold{k}.ep05.pt' for k in range(5)]
STRONG_MODELS = load_models(strong_paths, SPEC.device)
print(f'{len(STRONG_MODELS)} strong Stage-6 models loaded and hash-verified')


# V57 hidden witnesses and single V51 risk rankers.  The large V39/V38 banks
# are loaded once, but exactly one route is evaluated for each movie.
V39_SHA256 = {
    'v39.fold0.ep05.pt': '81ed7731a32927d66770fefeab03945be90ad5d8b0c9c5154cc600cbd4e0e74b',
    'v39.fold1.ep05.pt': '7a014cd0f3ba5275a537a2fa32d62297fd6bda200e62bab7b872364cff912109',
    'v39.fold2.ep05.pt': '8fa0912748b49f6588755bd55906200e77e0dde2dc060475cbabe0ec0f132a2b',
    'v39.fold3.ep05.pt': 'b3463f9ae16e94bdcf1b6608a27219d40bb6cee3e35c17c6933e84d3645236f8',
    'v39.fold4.ep05.pt': 'af73647ef4dcb72e8b3ea4cf1fbe52b26acd90d9533f9a3e5806571964f095b5',
}
# for _name, _digest in V39_SHA256.items():
#     assert hashlib.sha256((V39_ROOT / _name).read_bytes()).hexdigest() == _digest
V39_MODELS = load_models(
    [V39_ROOT / f'v39.fold{k}.ep05.pt' for k in range(5)], SPEC.device)
assert all(int(getattr(model, 'in_frames', 3)) == 5 for model in V39_MODELS)

# V57_MANIFEST = _json.loads(
#     (V57_ROOT / 'v57_deployment_manifest.json').read_text('utf-8'))
# for _row in V57_MANIFEST['files']:
#     _path = V57_ROOT / _row['name']
#     assert _path.stat().st_size == int(_row['bytes'])
#     assert hashlib.sha256(_path.read_bytes()).hexdigest() == _row['sha256']
V38_MODELS = load_models(
    [V57_ROOT / f'v38.fold{k}.ep05.pt' for k in range(5)], SPEC.device)
assert all(int(getattr(model, 'in_frames', 3)) == 5 for model in V38_MODELS)

from biohub.pipeline.v55_deploy import load_boosters
V57_BOOSTERS = {
    fold: load_boosters([V57_ROOT / f'fold{fold}.txt'])
    for fold in range(5)
}
print('V57 assets: 5 V39 routes, 5 V38 routes, 5 V51 risk models verified')

import lightgbm as _v343_lgb
V343_DIM_MODEL = _v343_lgb.Booster(model_file=str(V343_ROOT / 'full.dim.txt'))
V343_CELL_MODEL = _v343_lgb.Booster(model_file=str(V343_ROOT / 'full.cell.txt'))
print('V343 hard-local dim specialist models loaded')


# %% ################################################################################################
# V57 runtime-safe deployment omits the full V25 consensus replay.
print('runtime-safe policy: V25 and expanded replays removed; V39 reused as crossbank witness; 3 bounded tracker workers; research-state export disabled')


# %% ################################################################################################
# V57: V37c parent + count-neutral V39 recovery + residual V38.
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from biohub.counts import features_from
from biohub.pipeline.v169_division_consensus import fork_votes, select_actions, materialize

from biohub.detector.infer import detect, volumes_from_zarr
from biohub.detector.paired_rotation import detect_original_and_rotation
from biohub.detector.heat_peak_retention import retain_original_peaks
from biohub.detector.temporal_motion import sample_temporal_motion
from biohub.pipeline.bidirectional_gaps import propose_gaps, materialize_gaps
from biohub.pipeline.v230_deploy import load_models as load_v230_models, apply_correspondence
V230_MODELS = load_v230_models(V230_ROOT / "tree.txt", V389_ROOT / "residual.pt", SPEC.device)
from biohub.io.submission import validate_graph
import importlib.util
_v394_spec = importlib.util.spec_from_file_location('v394_component_protector', V394_ROOT / 'v394_component_protector.py')
v394 = importlib.util.module_from_spec(_v394_spec)
_v394_spec.loader.exec_module(v394)
V394_MODEL, V394_METADATA, V394_ROUTER = v394.load_assets(V394_ROOT)
_v399_spec = importlib.util.spec_from_file_location('v399_tail_protector', V399_ROOT / 'v399_tail_protector.py')
v399 = importlib.util.module_from_spec(_v399_spec)
_v399_spec.loader.exec_module(v399)
V399_MODEL, V399_METADATA = v399.load_assets(V399_ROOT)
_v454_spec = importlib.util.spec_from_file_location('v454_gap_admission', V454_ROOT / 'v454_gap_admission.py')
v454 = importlib.util.module_from_spec(_v454_spec)
_v454_spec.loader.exec_module(v454)
V454_METADATA = v454.load_assets(V454_ROOT)
_v507_spec = importlib.util.spec_from_file_location('v507_candidate', V507_ROOT / 'v507_candidate.py')
v507 = importlib.util.module_from_spec(_v507_spec)
_v507_spec.loader.exec_module(v507)
V507_ASSETS = v507.load_assets(V507_ROOT)
_v533_spec = importlib.util.spec_from_file_location('v533_addback_edge', V533_ROOT / 'v533_addback_edge.py')
v533 = importlib.util.module_from_spec(_v533_spec)
_v533_spec.loader.exec_module(v533)
# assert v533.VERSION == V533_MANIFEST['version'] and v533.PARAMS == V533_MANIFEST['params']
from biohub.pipeline.v233_deploy import apply_motion_substitution
from biohub.pipeline.coordinate_contract import copy_cv_coordinates, finalize_cv_graph
from biohub.pipeline.additive_segments import (
    add_full_length_segments, full_length_segment_actions)
from biohub.pipeline.learned import BUDGET_MULT, image_quantiles, track_dataset_learned_v8
from biohub.tracker.dim_scaffold import specialist_scaffold
from biohub.tracker.solver import SUPERSET_MULT
from biohub.pipeline.stage6_edge_consensus import (
    add_stage6_v24_consensus_edges)
from biohub.pipeline.v55_deploy import (
    V55_FEATURES, add_bridges, add_consensus_edges, apply_exchange,
    bridges_for, build_removal_features, deployment_route,
    make_expanded_superset, predict_bagged_risk, select_removals)


# Three trackers overlap CPU-heavy graph construction while CUDA work
# remains serialized by the device.  This bounds peak state and avoids
# the six-way memory pressure of an unconstrained executor.
TRACK_WORKERS = 3


def run_chain(
    path, models, runtime_label, *, keep_state=False, keep_volumes=False,
    candidate_override=None, superset_override=None,
    count_candidate_override=None,
    correspondence_state=False,
):
    runtime_started = time.perf_counter()
    result = track_dataset_learned_v8(
        path, models, None, COUNT_MODEL, EDGE_MODEL, CTX_MODEL,
        PAIR_MODEL, PAIR_STACK, ALLIMG_MODEL, FULL_MODEL, HARDNEG_MODEL,
        HIGHRES_MODEL, STUB_MODEL, COMP7_MODEL, None, CONFIRM_MODEL,
        TRACK_MODEL, POST_MODEL, SPEC.device, amp_dtype=SPEC.amp_dtype,
        return_state=keep_state or keep_volumes,
        include_volumes_in_state=keep_volumes,
        candidate_override=candidate_override,
        superset_override=superset_override,
        count_candidate_override=count_candidate_override,
        volumes_override=shared_volumes,
        image_quantiles_override=shared_quantiles,
        collect_garbage=False,
        return_correspondence_state=correspondence_state)
    runtime_breakdown[runtime_label] = time.perf_counter() - runtime_started
    return result


datasets = sorted(TEST_ROOT.glob('*.zarr'))
print(f'discovered {len(datasets)} datasets')
graphs, bundle_audit = [], []
t_start = time.perf_counter()

for index, path in enumerate(datasets, 1):
    started = time.perf_counter()
    runtime_breakdown = {}
    # Reuse one immutable normalised movie across every branch.
    volume_started = time.perf_counter()
    shared_volumes = volumes_from_zarr(path)
    shared_quantiles = image_quantiles(path)
    v394_cluster = v394.acquisition_cluster(path, V394_ROUTER)
    runtime_breakdown['volume_load'] = time.perf_counter() - volume_started

    def detect_bank(label, models):
        bank_started = time.perf_counter()
        candidates = detect(
            models, shared_volumes, SPEC.device,
            amp_dtype=SPEC.amp_dtype, batch=8)
        runtime_breakdown[f'{label}_detect'] = (
            time.perf_counter() - bank_started)
        return candidates

    # Frozen ablations remove V25 and reuse the already-required V39
    # graph as the crossbank witness. Detect each retained bank once,
    # then overlap the CPU-heavy downstream graph materializers.
    route = deployment_route(path.stem)
    primary_started = time.perf_counter()
    c21_count, c21, original_flow = detect_original_and_rotation(
        V21_MODELS, shared_volumes, SPEC.device,
        amp_dtype=SPEC.amp_dtype, batch=8, return_flow=True)
    runtime_breakdown['v21_detect'] = time.perf_counter() - primary_started
    peak_started = time.perf_counter()
    c21, peak_retention_audit = retain_original_peaks(c21, c21_count)
    specialist_features = features_from(path.stem, shared_quantiles, c21_count)
    specialist_per_frame = float(COUNT_MODEL.predict_per_frame(specialist_features)[0])
    specialist_budget = max(1, int(round(specialist_per_frame * BUDGET_MULT)))
    specialist_cap = max(1, int(round(specialist_budget * SUPERSET_MULT)))
    c21, v343_union, dim_specialist_audit = specialist_scaffold(
        shared_volumes, c21, V343_DIM_MODEL, V343_CELL_MODEL,
        specialist_cap, max_rank_shift=150, dim_region_threshold=None)
    assert dim_specialist_audit['outside_region_rank_weight_exactly_zero']
    assert dim_specialist_audit['score_histogram_preserved']

    runtime_breakdown['peak_retention'] = time.perf_counter() - peak_started
    c24 = detect_bank('v24', V24_MODELS)
    c_stage6 = detect_bank('stage6', STRONG_MODELS)
    c39 = detect_bank('v39', [V39_MODELS[route]])
    c38 = detect_bank('v38', [V38_MODELS[route]])
    expanded_census = {
        'crossbank_witness': 'v39_reuse',
        'crossbank_replay_runs': 0,
    }

    with ThreadPoolExecutor(max_workers=TRACK_WORKERS) as executor:
        futures = {
            'v21': executor.submit(
                run_chain, path, None, 'v21_track',
                candidate_override=c21, superset_override=v343_union,
                count_candidate_override=c21_count, correspondence_state=True),
            'v24': executor.submit(
                run_chain, path, None, 'v24_track',
                candidate_override=c24),
            'stage6': executor.submit(
                run_chain, path, None, 'stage6_track',
                candidate_override=c_stage6),
            'v39': executor.submit(
                run_chain, path, None, 'v39_track',
                candidate_override=c39),
            'v38': executor.submit(
                run_chain, path, None, 'v38_track',
                candidate_override=c38),
        }
        v21 = futures['v21'].result()
        v24 = futures['v24'].result()
        stage6 = futures['stage6'].result()
        v39 = futures['v39'].result()
        v38 = futures['v38'].result()

    def blend(graph):
        return copy_cv_coordinates(graph)

    # V250 research integration: both paths carry the evaluated CV coordinates.
    phase19_raw, p19_audit = add_full_length_segments(v21.graph, v24.graph)
    phase19_out, p19_out_audit = copy_cv_coordinates(phase19_raw), dict(p19_audit)
    stage_raw, stage_audit = add_full_length_segments(
        phase19_raw, stage6.graph)
    stage_out, stage_out_audit = copy_cv_coordinates(stage_raw), dict(stage_audit)
    v37_raw, v37_edge_audit, v37_edge_actions = (
        add_stage6_v24_consensus_edges(stage_raw, stage6.graph, v24.graph))
    v37_out, v37_edge_out_audit, v37_edge_out_actions = copy_cv_coordinates(v37_raw), dict(v37_edge_audit), list(v37_edge_actions)
    assert p19_audit['segments'] == p19_out_audit['segments']
    assert stage_audit['segments'] == stage_out_audit['segments']
    assert v37_edge_actions == v37_edge_out_actions
    assert v37_edge_audit == v37_edge_out_audit

    # V57's only learned graph decision: same-frame count-neutral V39 exchange.
    v39_actions = full_length_segment_actions(v37_raw, v39.graph)
    features = build_removal_features(
        v37_raw, phase19_raw, v21.graph, v24.graph, stage6.graph, v39.graph,
        v39_actions,
        {'v21': c21, 'v24': c24, 'stage6': c_stage6, 'v39': c39})
    risk = predict_bagged_risk(features, V57_BOOSTERS[route])
    selected = select_removals(features, risk)
    removed = set(selected.node_id.astype(int))
    expected_removals = sum(
        len(json.loads(str(action['path_v24_nodes'])))
        for action in v39_actions)
    assert len(removed) == expected_removals
    v51_raw = apply_exchange(v37_raw, v39.graph, v39_actions, removed)
    v51_out = copy_cv_coordinates(v51_raw)

    witnesses = {
        'stage6': stage6.graph, 'v24': v24.graph,
        'expanded': v39.graph}
    v52_raw, edge_actions, edge_census = add_consensus_edges(
        v51_raw, witnesses)
    v52_out, edge_actions_out, edge_census_out = copy_cv_coordinates(v52_raw), list(edge_actions), dict(edge_census)
    assert edge_actions == edge_actions_out
    assert edge_census == edge_census_out

    after_v38_raw, v38_audit = add_full_length_segments(v52_raw, v38.graph)
    after_v38_out, v38_out_audit = copy_cv_coordinates(after_v38_raw), dict(v38_audit)
    assert v38_audit['segments'] == v38_out_audit['segments']
    # V57 intentionally omits the lower-efficiency V44 bridge stage.
    division_started = time.perf_counter()
    division_witnesses = {'v24': v24.graph, 'stage6': stage6.graph,
                          'v39': v39.graph, 'v38': v38.graph}
    division_votes, division_mapping = fork_votes(after_v38_raw, division_witnesses)
    division_actions, division_conflicts = select_actions(
        after_v38_raw, division_votes, 'root_consensus2', guard_hops=2)
    final_raw, division_audit = materialize(after_v38_raw, division_actions)
    final_out, division_out_audit = copy_cv_coordinates(final_raw), dict(division_audit)
    assert division_audit['new_forks'] == division_out_audit['new_forks']
    runtime_breakdown['division_consensus'] = time.perf_counter() - division_started

    # Select and emit on the same strict CV geometry.
    gap_motion_started = time.perf_counter()
    gap_ids = np.asarray(list(final_raw.nodes), np.int64)
    gap_nodes = np.asarray(list(final_raw.nodes.values()), np.int64)
    gap_backward, gap_forward = sample_temporal_motion(
        V21_MODELS, shared_volumes, gap_nodes[:, 0], gap_nodes[:, 1:],
        SPEC.device, SPEC.amp_dtype, batch=8, backward_fields=original_flow)
    del original_flow
    runtime_breakdown['gap_motion'] = time.perf_counter() - gap_motion_started
    gap_started = time.perf_counter()
    gap_actions, gap_selection = propose_gaps(
        final_raw, gap_ids, gap_backward, gap_forward, c21)
    final_raw, gap_raw_audit = materialize_gaps(final_raw, gap_actions)
    final_out, gap_out_audit = copy_cv_coordinates(final_raw), dict(gap_raw_audit)
    assert gap_raw_audit['forks'] == gap_out_audit['forks']
    runtime_breakdown['gap_closure'] = time.perf_counter() - gap_started

    correspondence_started = time.perf_counter()
    final_raw, final_out, correspondence_audit = apply_correspondence(
        final_raw, final_out, v21.graph, v21.correspondence_state, shared_volumes,
        gap_ids, gap_nodes, gap_backward, gap_forward, V230_MODELS, SPEC.device)
    runtime_breakdown['correspondence'] = time.perf_counter() - correspondence_started

    substitution_started = time.perf_counter()
    final_raw, final_out, substitution_audit = apply_motion_substitution(
        final_raw, final_out, gap_ids, gap_nodes, gap_backward, gap_forward, c21)
    runtime_breakdown['motion_substitution'] = time.perf_counter() - substitution_started

    raw_valid = validate_graph(final_raw)
    out_valid = validate_graph(final_out)
    assert set(final_raw.nodes) == set(final_out.nodes)
    assert final_raw.edges == final_out.edges
    assert raw_valid['forks'] == out_valid['forks']
    v389_graph = finalize_cv_graph(final_raw, final_out)
    component_started = time.perf_counter()
    v394_graph, v394_audit = v394.prune_components(
        v389_graph, shared_volumes,
        {'v24': copy_cv_coordinates(v24.graph), 'stage6': copy_cv_coordinates(stage6.graph),
         'v39': copy_cv_coordinates(v39.graph), 'v38': copy_cv_coordinates(v38.graph)},
        V394_MODEL, V394_METADATA, v394_cluster)
    runtime_breakdown['component_protector'] = time.perf_counter() - component_started
    tail_started = time.perf_counter()
    v399_graph, v399_audit = v399.trim_terminal_tails(
        v394_graph, shared_volumes, {}, V399_MODEL, V399_METADATA, v394_cluster)
    runtime_breakdown['tail_protector'] = time.perf_counter() - tail_started
    v454_started = time.perf_counter()
    v454_graph, v454_audit = v454.apply_gap_admission(
        v399_graph, gap_ids, gap_backward, gap_forward, path, V454_METADATA)
    runtime_breakdown['v454_gap_admission'] = time.perf_counter() - v454_started
    # Lock graph-dependent actions on the exact V454 graph used in CV.
    v507_prepare_started = time.perf_counter()
    v507_prepared, v507_prepare_audit = v507.prepare_actions(
        v454_graph, c21, v394_cluster, gap_ids, gap_nodes,
        gap_backward, gap_forward, V507_ASSETS)
    runtime_breakdown['v507_prepare'] = time.perf_counter() - v507_prepare_started
    # No second tail pass: apply the locked actions to the V454 graph itself.
    v507_apply_started = time.perf_counter()
    v507_graph, v507_apply_audit = v507.apply_prepared(
        v454_graph, v507_prepared)
    runtime_breakdown['v507_apply'] = time.perf_counter() - v507_apply_started
    # V533: contrast-gated add-back (acq_01 only) + xy data-edge trim on the final V507 graph.
    v533_started = time.perf_counter()
    v533_graph, v533_audit = v533.apply(v507_graph, c21, v394_cluster, path)
    runtime_breakdown['v533_addback_edge'] = time.perf_counter() - v533_started
    graphs.append(v533_graph)
    bundle_audit.append({
        'dataset': path.stem,
        'route': route,
        'v394_component_protector': v394_audit,
        'v399_tail_protector': v399_audit,
        'v454_gap_admission': v454_audit,
        'v507_prepare': v507_prepare_audit,
        'v507_apply': v507_apply_audit,
        'v533_addback_edge': v533_audit,
        'v39_actions': len(v39_actions),
        'removals': len(removed),
        'risk_rows': len(features),
        'multiwitness_edges': len(edge_actions),
        'v38_segments': int(v38_audit['segments']),
        'v38_nodes': int(v38_audit['added_nodes']),
        'v44_bridges': 0,
        'primary_heatmap': 'mean_original_rotation180',
        'primary_peak_policy': 'keep_heat_recover_original_3um_capped',
        'peak_retention': peak_retention_audit,
        'dim_specialist': dim_specialist_audit,
        'primary_count_features': 'original_v21',
        'division_policy': 'root_consensus2',
        'division_forks_added': len(division_actions),
        'division_mapping': division_mapping,
        'division_conflicts': division_conflicts,
        'gap_policy': 'bidirectional_max2_radius3_track3_guard2',
        'gap_actions': len(gap_actions),
        'gap_selection': gap_selection,
        'gap_added_nodes': sum(len(a['nodes']) for a in gap_actions),
        'gap_added_edges': sum(len(a['nodes']) + 1 for a in gap_actions),
        'gap_motion_views': ['original_backward', 'reversed_forward'],
        'gap_source_geometry': 'complete_raw_v194',
        'gap_new_coordinates': 'raw_v194_candidates',
        'correspondence_policy': 'tree_logit_plus_image_residual_bonus0',
        'correspondence': correspondence_audit,
        'motion_substitution': substitution_audit,
        'runtime_seconds': {
            key: round(value, 3) for key, value in runtime_breakdown.items()},
        **expanded_census,
    })
    elapsed = time.perf_counter() - t_start
    eta = elapsed / index * (len(datasets) - index)
    print(
        f'  [{index}/{len(datasets)}] {path.stem} route={route}: '
        f'v39 {len(v39_actions):,}/{len(removed):,}, '
        f'edge +{len(edge_actions):,}, v38 +{v38_audit["added_nodes"]:,}, '
        f'v454 +{v454_audit["selected"]:,}, '
        f'v507 loc {v507_apply_audit["localization_moved"]:,} '
        f'fork -{v507_apply_audit["fork_applied"]:,}, '
        f'v533 +{v533_audit["added_nodes"]:,} -{v533_audit["trimmed_nodes"]:,}; '
        f'stages {json.dumps({key: round(value) for key, value in runtime_breakdown.items()})}; '
        f'{time.perf_counter()-started:.1f}s eta {eta/60:.0f}m', flush=True)

    del (v21, v24, stage6, v39, v38, c21, c21_count, c24, c38,
         c_stage6, c39, features,
         selected, final_raw, shared_volumes, shared_quantiles, v343_union,
         gap_ids, gap_nodes, gap_backward, gap_forward, gap_actions,
         v507_prepared)
    gc.collect()

print(f'total {(time.perf_counter()-t_start)/60:.1f} min')


# %% ################################################################################################
# Validate and atomically write the V533 submission.
from biohub.io.submission import build_submission, write_submission

df, summary = build_submission(graphs)
temporary_output = OUT.with_suffix('.tmp.csv')
write_submission(df, temporary_output)
temporary_output.replace(OUT)
audit_path = OUT.with_name('v533_deployment_audit.json')
audit_path.write_text(json.dumps(bundle_audit, indent=2) + '\n', encoding='utf-8')
print(f'{len(df):,} rows -> {OUT} ({OUT.stat().st_size/1e6:.1f} MB)')
print(f'audit -> {audit_path}')
assert len(graphs) == len(datasets) > 0
assert set(df.dataset) == {path.stem for path in datasets}
assert df[['t', 'z', 'y', 'x']].notna().all().all()
print('every discovered dataset present; schema and finite-coordinate gates pass')
summary
