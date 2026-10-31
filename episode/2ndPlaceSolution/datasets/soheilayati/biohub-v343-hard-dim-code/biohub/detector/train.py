"""Detector training loop.

Deliberately small: the interesting decisions live in `targets.py` (what is
supervised) and `dataset.py` (what the model sees). This just runs them.

**Progress is read off `monitor`, never `total`.** The uncertainty term is a
Gaussian NLL and unbounded below, so `total` goes negative once the flow head is
accurate and reads as spectacular progress regardless of what the other three
terms did — measured 0.98 -> -0.09 on the overfit test.

**Validation reports recall at a fixed peak budget, not just a loss.** A
detector can improve its heatmap loss while moving neither budgeted recall nor
D2 (Stage 2 progress §2), and the loss is computed on annotated voxels only —
the failure being fixed is about what happens to the *other* peaks. The
in-training numbers are a screen; the promotion decision goes through the Stage 1
rig on out-of-fold candidates.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from biohub.detector.dataset import DatasetConfig, DetectorDataset, collate
from biohub.detector.losses import LossWeights, detector_loss
from biohub.detector.model import TemporalUNet3d, count_parameters
from biohub.candidates.nms import select_frame
from biohub.detector.targets import VOXEL_UM, grid_to_src
from biohub.device import select_device

# Peak budgets for the in-training recall screen. True density spans 38-786
# cells/frame with a median near 237, so these bracket the real operating point
# without reading any ground-truth count.
VAL_BUDGETS = (100, 250, 500)
MATCH_UM = 7.0
# Suppression radius for the screen. 5.0 um is the Stage 1 operating point;
# the NMS/score curve is re-swept on the real pool, not assumed here.
VAL_NMS_UM = 5.0


@dataclass
class TrainConfig:
    fold: int | None = 0
    epochs: int = 30
    # Epochs the cosine schedule is drawn over, if different from the number
    # actually run. Stopping a 30-epoch cosine at 12 is NOT the same
    # experiment as a 12-epoch cosine: to reconstruct checkpoints faithfully,
    # epoch k must see the learning rate it originally saw.
    schedule_epochs: int | None = None
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-2
    workers: int = 6
    warmup_frac: float = 0.03
    grad_clip: float = 1.0
    val_every: int = 3
    val_batches: int = 60
    # Steps between progress lines. An epoch is ~1,900 steps and ~8 minutes, so
    # logging only at epoch boundaries leaves a run with no signal for minutes at
    # a time -- long enough that a stalled job and a healthy one look identical.
    log_every: int = 200
    # Training samples re-evaluated WITHOUT augmentation, to make the
    # generalization gap readable. A slice, not all of them: the point is a
    # comparable number, not a precise one.
    clean_train_samples: int = 16
    # Early stopping, PRE-COMMITTED before the fold runs rather than chosen after
    # seeing its curve. Selecting a different epoch per fold once its validation
    # score is visible makes the out-of-fold estimate optimistic — the fold's own
    # held-out data would be choosing the model that is then measured on it.
    # `patience` counts consecutive validations that fail to improve
    # `val_monitor`. 0 disables.
    patience: int = 0
    seed: int = 0
    data: DatasetConfig = field(default_factory=DatasetConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    widths: tuple[int, ...] = (24, 48, 96, 192)
    feature_ch: int = 48
    # Exponential moving average of the weights. 0 disables. When enabled the
    # EMA model is evaluated at every validation and saved beside the raw
    # checkpoint as `<tag>.ep<k>.ema.pt` — selection between raw and EMA is
    # pre-committed by the experiment registration, never chosen per fold.
    ema_decay: float = 0.0
    # Draw training examples so the four embryo x acquisition strata
    # contribute equally per epoch (with replacement). The natural mix is
    # ~60/40 acq_00/acq_01 by annotated frames while acq_01 carries most of
    # the residual loss; balancing is a training-time choice only and leaves
    # inference untouched.
    balanced_strata: bool = False
    # Optional model-only initialization. Optimizer and scheduler deliberately
    # start fresh, which makes a short real-only calibration tail distinct from
    # resuming the synthetic training trajectory.
    init_checkpoint: Path | None = None
    out_dir: Path = Path("artifacts/detector")


def fold_split(folds_path: Path, fold: int) -> tuple[list[str], list[str]]:
    spec = json.loads(Path(folds_path).read_text())
    val = list(spec["folds"][str(fold)])
    train = [n for k, v in spec["folds"].items() if int(k) != fold for n in v]
    return sorted(train), sorted(val)


def loeo_split(splits_path: Path, train_embryo: str) -> tuple[list[str], list[str]]:
    """`train_embryo` names the embryo to TRAIN on; the other is validation.

    Matched on the `train_<embryo>__` prefix, not by substring: the keys are
    `train_6bba__val_44b6` and `train_44b6__val_6bba`, so `"44b6" in key` is
    true for both and would silently select the wrong direction.
    """
    spec = json.loads(Path(splits_path).read_text())["splits"]
    key = f"train_{train_embryo}__"
    matches = [k for k in spec if k.startswith(key)]
    if len(matches) != 1:
        raise KeyError(f"expected one split starting {key!r}, got {matches}")
    return sorted(spec[matches[0]]["train"]), sorted(spec[matches[0]]["val"])


def _loader(names: list[str], cfg: TrainConfig, train: bool, span: int | None = None):
    """`span` caps an unshuffled loader to that many examples, spread EVENLY
    across the whole index rather than taken from its front.

    This is not a detail. `DetectorDataset.index` is built in sorted dataset
    order, so every `44b6_*` frame precedes every `6bba_*` one. An unshuffled
    validation loader truncated to 60 batches therefore saw 480 consecutive
    frames from the front -- all 44b6 -- and reported them as the held-out
    result. 6bba carries 84.6 % of the score weight and never appeared.
    Measured: `nodes_44b6` equalled the entire validation node count.
    """
    from torch.utils.data import DataLoader, Subset

    dcfg = DatasetConfig(
        data_root=cfg.data.data_root, target=cfg.data.target,
        augment=cfg.data.augment, train=train,
        in_frames=cfg.data.in_frames,
        norm_lo_q=cfg.data.norm_lo_q, norm_hi_q=cfg.data.norm_hi_q,
        volume_cache=cfg.data.volume_cache, legacy_seed=cfg.data.legacy_seed,
        # `DatasetConfig` gates this on `train` itself, so validation keeps its
        # complete annotation even though the same spec is passed to both loaders.
        label_subset=cfg.data.label_subset,
    )
    if any("::" in name for name in names):
        # Synthetic variants deliberately live outside the detector package so
        # ordinary training and inference never acquire a synthesis dependency.
        # Select the compositing dataset only when an explicit variant key is
        # present; all-real runs retain the historical DetectorDataset path.
        from biohub.synthesis.dataset import SynthDetectorDataset

        ds: object = SynthDetectorDataset(names, dcfg)
    else:
        ds = DetectorDataset(names, dcfg)
    if span is not None and span < len(ds):
        # Deterministic even stride: reproducible across epochs and runs, and
        # proportional in both embryos by construction.
        idx = np.linspace(0, len(ds) - 1, span).astype(int)
        ds = Subset(ds, sorted(set(idx.tolist())))
    sampler = None
    if train and cfg.balanced_strata:
        from torch.utils.data import WeightedRandomSampler

        clusters = _acquisition_clusters()
        strata = [
            f"{name.split('_', 1)[0]}/{clusters[name]}" for name, _ in ds.index
        ]
        counts: dict[str, int] = {}
        for key in strata:
            counts[key] = counts.get(key, 0) + 1
        weights = torch.as_tensor(
            [1.0 / counts[key] for key in strata], dtype=torch.double
        )
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        print(f"balanced strata sampler over {counts}", flush=True)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=train and sampler is None,
        sampler=sampler, num_workers=cfg.workers,
        collate_fn=collate, drop_last=train, persistent_workers=cfg.workers > 0,
        prefetch_factor=2 if cfg.workers > 0 else None,
    )


def _acquisition_clusters() -> dict[str, str]:
    import csv

    path = Path("data/acquisition_clusters_v1.csv")
    with path.open(newline="", encoding="utf-8") as f:
        return {row["dataset"]: row["acquisition_cluster"] for row in csv.DictReader(f)}


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, weights, max_batches: int) -> dict:
    """Held-out loss plus annotated-node recall at fixed peak budgets."""
    from skimage.feature import peak_local_max

    model.eval()
    sums: dict[str, float] = {}
    n = 0
    hits = {b: 0 for b in VAL_BUDGETS}
    total_nodes = 0
    flow_err: list[float] = []
    # Per embryo as well as aggregate. 6bba carries 84.6 % of the score weight
    # and the two embryos fail differently, so an aggregate that improves can be
    # 44b6 alone -- and 44b6 is 15.4 % of the weight (Stage 2 progress §5).
    by_emb: dict[str, dict] = {}

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        tgt = {k: v.to(device, non_blocking=True) for k, v in batch.items()
               if isinstance(v, torch.Tensor)}
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(x)
        out = {k: v.float() for k, v in out.items() if k != "features"}
        _, parts = detector_loss(out, tgt, weights)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1

        heat = torch.sigmoid(out["heatmap"][:, 0]).cpu().numpy()
        flow = out["flow"].cpu().numpy()
        pos = batch["positive"].numpy() > 0
        tflow = batch["flow"].numpy()
        fmask = batch["flow_mask"].numpy() > 0

        for b in range(len(heat)):
            # Annotated centres are recoverable from the positive zone's own
            # local maxima in the target, which avoids threading coordinates
            # through the collate function.
            centres = peak_local_max(batch["heatmap"][b].numpy(), min_distance=2,
                                     threshold_abs=0.5, exclude_border=False)
            if not len(centres):
                continue
            total_nodes += len(centres)
            emb = batch["dataset"][b].split("_")[0]
            slot = by_emb.setdefault(emb, {"nodes": 0, **{f"h{k}": 0 for k in VAL_BUDGETS}})
            slot["nodes"] += len(centres)
            peaks = peak_local_max(heat[b], min_distance=1, exclude_border=False)
            if not len(peaks):
                continue
            # Physical suppression before the budget, exactly as the real
            # selector does. Without it the top-N is a cluster of adjacent
            # voxels around the strongest cell and recall is understated --
            # which is the whole reason `select_frame` orders it this way.
            scores = heat[b][tuple(peaks.T)]
            kept = select_frame(grid_to_src(peaks), scores, VAL_NMS_UM)
            peaks = peaks[kept]
            for budget in VAL_BUDGETS:
                take = peaks[:budget] * VOXEL_UM
                d = np.linalg.norm(
                    centres[:, None, :] * VOXEL_UM - take[None, :, :], axis=2
                )
                got = int((d.min(axis=1) <= MATCH_UM).sum())
                hits[budget] += got
                slot[f"h{budget}"] += got

            if fmask[b].any():
                m = fmask[b]
                flow_err.append(float(
                    np.linalg.norm((flow[b] - tflow[b])[:, m], axis=0).mean()
                ))

    model.train()
    res = {f"val_{k}": v / max(n, 1) for k, v in sums.items()}
    for budget in VAL_BUDGETS:
        res[f"recall@{budget}"] = hits[budget] / max(total_nodes, 1)
    res["flow_mae_um"] = float(np.mean(flow_err)) if flow_err else float("nan")
    res["val_nodes"] = total_nodes
    for emb, slot in sorted(by_emb.items()):
        for budget in VAL_BUDGETS:
            res[f"recall@{budget}_{emb}"] = slot[f"h{budget}"] / max(slot["nodes"], 1)
        res[f"nodes_{emb}"] = slot["nodes"]
    return res


def train_detector(cfg: TrainConfig, train_names: list[str], val_names: list[str]) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    spec = select_device(allow_cpu=True)
    device = spec.device
    print(f"device: {spec.describe()}", flush=True)

    model = TemporalUNet3d(cfg.data.in_frames, cfg.widths, cfg.feature_ch).to(device)
    if cfg.init_checkpoint is not None:
        checkpoint = torch.load(
            cfg.init_checkpoint, map_location="cpu", weights_only=False
        )
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state, strict=True)
        print(f"initialized model from {cfg.init_checkpoint}", flush=True)
    print(f"model: {count_parameters(model):,} parameters", flush=True)

    ema_model = None
    if cfg.ema_decay > 0.0:
        import copy

        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        print(f"weight EMA enabled, decay {cfg.ema_decay}", flush=True)

    train_loader = _loader(train_names, cfg, train=True)
    val_loader = _loader(val_names, cfg, train=False,
                         span=cfg.val_batches * cfg.batch_size)
    # A CLEAN pass over a slice of the training samples. Without it, comparing
    # `train_monitor` (augmented, and therefore harder) against `val_monitor`
    # (clean) is not a generalization gap at all -- and it is biased in the
    # flattering direction, because a clean-train figure would be lower still.
    # `clean_monitor` vs `val_monitor` is the honest comparison.
    clean_loader = _loader(train_names[:cfg.clean_train_samples], cfg, train=False,
                           span=max(8, cfg.val_batches // 4) * cfg.batch_size)
    steps = max(1, len(train_loader)) * (cfg.schedule_epochs or cfg.epochs)
    print(f"lr schedule over {cfg.schedule_epochs or cfg.epochs} epochs, "
          f"running {cfg.epochs}", flush=True)
    print(f"train {len(train_names)} samples / {len(train_loader.dataset):,} frames; "
          f"val {len(val_names)} / {len(val_loader.dataset):,}; {steps:,} steps",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.warmup_frac * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=spec.amp_dtype is torch.float16)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"fold{cfg.fold}" if cfg.fold is not None else "full"
    log_path = cfg.out_dir / f"{tag}.log.jsonl"
    ckpt_path = cfg.out_dir / f"{tag}.pt"
    best = float("inf")
    stale = 0
    t0 = time.perf_counter()
    step = 0

    for epoch in range(cfg.epochs):
        run: dict[str, float] = {}
        seen = 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            tgt = {k: v.to(device, non_blocking=True) for k, v in batch.items()
                   if isinstance(v, torch.Tensor)}

            with torch.autocast("cuda", dtype=spec.amp_dtype,
                                enabled=spec.amp_dtype is not None):
                out = model(x)
            out = {k: v.float() for k, v in out.items() if k != "features"}
            loss, parts = detector_loss(out, tgt, cfg.loss)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema_model is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.lerp_(p, 1.0 - cfg.ema_decay)

            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v
            seen += 1
            step += 1

            if cfg.log_every and step % cfg.log_every == 0:
                # Running mean over this epoch so far, not the single step --
                # a per-step figure on a batch of 8 is too noisy to read a trend
                # from, which is the only reason to print it.
                print(f"    step {step:>6}  epoch {epoch}  "
                      f"monitor {run['monitor']/seen:.4f}  "
                      f"heatmap {run['heatmap']/seen:.4f}  "
                      f"{step * cfg.batch_size / (time.perf_counter() - t0):.1f} ex/s",
                      flush=True)

        row = {"epoch": epoch, "step": step, "lr": sched.get_last_lr()[0],
               "secs": round(time.perf_counter() - t0, 1),
               **{f"train_{k}": round(v / max(seen, 1), 5) for k, v in run.items()}}

        if (epoch + 1) % cfg.val_every == 0 or epoch == cfg.epochs - 1:
            row.update({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in evaluate(model, val_loader, device,
                                             spec.amp_dtype, cfg.loss,
                                             len(val_loader)).items()})
            clean = evaluate(model, clean_loader, device, spec.amp_dtype,
                             cfg.loss, len(clean_loader))
            row["clean_monitor"] = round(clean["val_monitor"], 5)
            row["clean_recall@250"] = round(clean["recall@250"], 5)
            # The gap that actually means generalization: both sides clean.
            row["gen_gap"] = round(row["val_monitor"] - row["clean_monitor"], 5)
            state = {"model": model.state_dict(), "cfg": tag, "epoch": epoch,
                     "val_monitor": row["val_monitor"], "widths": cfg.widths,
                     "feature_ch": cfg.feature_ch,
                     "in_frames": cfg.data.in_frames,
                     "sigma_um": cfg.data.target.sigma_um,
                     "init_checkpoint": (str(cfg.init_checkpoint)
                                         if cfg.init_checkpoint is not None
                                         else None)}
            # EVERY validated epoch is kept, not only the best-so-far. The
            # selection criterion here is `val_monitor`, which is last in the
            # decision hierarchy -- the exact deployable score comes first, and
            # it cannot be computed until candidates are extracted and scored.
            # Keeping only the val_monitor winner therefore discards checkpoints
            # that may score better, which is exactly what happened on fold 0:
            # epoch 2 led at budgets 250 and 500 and was overwritten by epoch 5.
            # 14 MB each, ~10 per fold. Storage is not the constraint; a lost
            # checkpoint is unrecoverable without a full retrain.
            torch.save(state, cfg.out_dir / f"{tag}.ep{epoch:02d}.pt")
            row["ckpt"] = f"{tag}.ep{epoch:02d}.pt"
            if ema_model is not None:
                ema_metrics = evaluate(ema_model, val_loader, device,
                                       spec.amp_dtype, cfg.loss, len(val_loader))
                row["val_monitor_ema"] = round(ema_metrics["val_monitor"], 5)
                for budget in VAL_BUDGETS:
                    row[f"recall@{budget}_ema"] = round(
                        ema_metrics[f"recall@{budget}"], 5)
                row["flow_mae_um_ema"] = round(ema_metrics["flow_mae_um"], 5)
                ema_state = dict(state, model=ema_model.state_dict(),
                                 val_monitor=ema_metrics["val_monitor"],
                                 ema_decay=cfg.ema_decay)
                torch.save(ema_state, cfg.out_dir / f"{tag}.ep{epoch:02d}.ema.pt")
            if row["val_monitor"] < best:
                best = row["val_monitor"]
                stale = 0
                torch.save(state, ckpt_path)
                row["saved"] = True
            else:
                stale += 1
                row["stale"] = stale

        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

        if cfg.patience and stale >= cfg.patience:
            print(f"\nstopping: {stale} consecutive validations without "
                  f"improving val_monitor (patience {cfg.patience})", flush=True)
            break

    print(f"\nbest val_monitor {best:.5f} -> {ckpt_path}", flush=True)
    print("every validated epoch is also kept as "
          f"{tag}.ep??.pt -- select on the deployable score, not on this.",
          flush=True)
    return ckpt_path
