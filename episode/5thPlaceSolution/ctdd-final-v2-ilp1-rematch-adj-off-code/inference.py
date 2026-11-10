# Inference writes graph predictions and a submission CSV.
#
# Per clip: detect cell centres frame by frame, score every (t, t+1) pair, then hand the
# whole video's candidate edges to the ILP solver in one go.
# Run: python inference.py

import json
import os
import sys
from functools import lru_cache

import numpy as np
import torch

from config import *
from dataset import *
from models import *
from utils import *
from inference_fast import InferenceUNetRunner, fast_model_context, model_execution_order

sys.path.append(InferConfig.vendor_src)

from tracking_cellmot.io import *  # noqa: F403,E402


@lru_cache(maxsize=8)
def _fast_unet_runner(batch_size, oom_fallback):
    # The runner holds weak model references, retaining OOM batch decisions across
    # clips without retaining checkpoints after a worker finishes.
    return InferenceUNetRunner(batch_size=batch_size, oom_fallback=oom_fallback)


def predict_clip(cfg, models, volume, meta, collect_prethreshold=False,
                 point_thresholds=None):
    """Share UNet features across thresholds; score each full candidate set separately.

    Multi-threshold calls return {threshold: (prediction, cache)}. The cache always
    includes shared peak IDs; dense scores/features are collected only on request.
    """
    thresholds = tuple(point_thresholds) if point_thresholds is not None else (cfg.point_threshold,)
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("point_thresholds must be nonempty and unique")
    device = next(models[0].parameters()).device
    execution_order = model_execution_order(models)
    unet_runner = _fast_unet_runner(cfg.inference_tta_batch_size, cfg.inference_oom_fallback)
    amp_enabled = cfg.use_amp and device.type == "cuda"
    pool_kernel = pool_kernel_from_um(cfg.pool_kernel_um, meta["voxel_size"])
    tta = cfg.tta_mode if cfg.use_tta else "none"
    T = volume.shape[0]
    use_idsim = cfg.idsim["enable"]
    coord_scale = torch.as_tensor(meta["scale"], dtype=torch.float32, device=device)
    outputs = {}
    for threshold in thresholds:
        outputs[threshold] = {
            "prediction": ([], [], [], [], []), "starts": {}, "previous": None,
            "cache": {
                "peak": [], "zyx": [], "probability": [], "frame_offsets": [0],
                "edge_prob": [], "id_dissim": [], "edge_offsets": [0],
                "source_feature": [], "target_feature": [],
                "source_offsets": [0], "target_offsets": [0],
                "selected_peak_indices": [],
            },
        }

    centroid = cfg.centroid
    raw_arr = None
    if centroid["enable"]:
        import zarr
        raw_arr = zarr.open_group(meta["zarr_path"], mode="r")["0"]

    def blend_centroid(t, peak, zyx):
        if raw_arr is None or peak.shape[0] == 0:
            return zyx
        refined = refine_centroid(
            np.asarray(raw_arr[t]), peak.cpu().numpy(), meta["downsample"],
            rz=centroid["rz"], ryx=centroid["ryx"])
        refined_net = image_to_network_coordinates(
            torch.from_numpy(refined).to(zyx.device, torch.float32),
            meta["downsample"])
        blend = centroid["blend"]
        return (1.0 - blend) * zyx.float() + blend * refined_net

    def add_to_out(edge_prob, id_dissim, node_prob0, node_prob1,
                   appearance_prob, division_prob, coord0, coord1, t0, t1):
        if collect_prethreshold:
            flat_edge = np.asarray(edge_prob, np.float32).reshape(-1)
            flat_dissim = (
                np.asarray(id_dissim, np.float32).reshape(-1)
                if id_dissim is not None else np.zeros_like(flat_edge)
            )
            prethreshold["edge_prob"].append(flat_edge)
            prethreshold["id_dissim"].append(flat_dissim)
            prethreshold["edge_offsets"].append(
                prethreshold["edge_offsets"][-1] + len(flat_edge))
        if t0 == 0:
            out_start[t0] = len(out_node)
            for (z, y, x), probability in zip(coord0, node_prob0, strict=True):
                out_node.append([t0, z, y, x])
                out_node_prob.append(float(probability))
                out_appearance.append(1.0)
                out_division.append(0.0)
        source_start = out_start[t0]
        out_division[source_start:source_start + len(coord0)] = division_prob.tolist()

        out_start[t1] = len(out_node)
        for (z, y, x), probability, appearance in zip(
            coord1, node_prob1, appearance_prob, strict=True,
        ):
            out_node.append([t1, z, y, x])
            out_node_prob.append(float(probability))
            out_appearance.append(float(appearance))
            out_division.append(0.0)

        s0, s1 = out_start[t0], out_start[t1]
        ii, jj = np.nonzero(edge_prob > cfg.edge_threshold)
        if ii.size == 0:
            return
        probs = edge_prob[ii, jj]
        dists = np.linalg.norm(coord0[ii] - coord1[jj], axis=1)
        dissim = (
            id_dissim[ii, jj]
            if id_dissim is not None else np.zeros(ii.size, np.float32))
        order = sorted(range(ii.size), key=lambda k: (probs[k], ii[k], jj[k]), reverse=True)
        for k in order:
            out_edge.append([
                int(ii[k]) + s0, int(jj[k]) + s1,
                float(probs[k]), float(dists[k]), float(dissim[k]),
            ])

    def forward_models(image):
        features, logits = [None] * len(models), [None] * len(models)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled,
        ):
            for model_index in execution_order:
                model = models[model_index]
                feature, logit = unet_runner(model, image)
                # Execute larger U-Nets before other models' maps occupy VRAM,
                # but preserve the original model order in every ensemble mean.
                features[model_index] = [[undo_tta(level, tta) for level in frame]
                                         for frame in feature]
                logits[model_index] = [undo_tta(x, tta) for x in logit]
                del feature, logit
        probabilities = [
            torch.sigmoid(torch.stack([
                logit[frame].float().mean(0) for logit in logits
            ]).mean(0))
            for frame in range(2)
        ]
        return features, probabilities

    def encode_pair(t):
        im = torch.from_numpy(volume[t:t + 2]).to(device)
        image = torch.stack(do_tta(im, tta), dim=0)
        return forward_models(image)

    def tta_features(model, levels, peaks, zyx, fuse):
        views = levels if fuse else [level[:1] for level in levels]
        return model.sample_node_features(views, peaks, zyx)

    def embed_identities(features, frame, peaks, zyx):
        # identities for the ILP: full-res map, sampled as the model was trained
        if not use_idsim:
            return None
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled,
        ):
            return [
                model.identity_embedding(tta_features(
                    model, feature[frame][:1], peaks, zyx, cfg.tta_idsim_fusion))
                for model, feature in zip(models, features, strict=True)
            ]

    def detect_frame(probability):
        floor = 0.0 if collect_prethreshold or point_thresholds is not None else cfg.point_threshold
        peaks = prob_to_zyx(probability, floor, pool_kernel)
        zyx = (refine_peaks_parabolic(probability, peaks)
               if cfg.parabolic_refinement else peaks.float())
        scores = probability[0, peaks[:, 0], peaks[:, 1], peaks[:, 2]]
        return peaks, zyx, scores

    def select_frame(detected, features, frame, t, threshold, offset):
        all_peaks, all_zyx, scores = detected
        keep = scores > threshold
        peaks, zyx = all_peaks[keep], all_zyx[keep]
        zyx = blend_centroid(t, peaks, zyx)
        identities = embed_identities(features, frame, peaks, zyx)
        prethreshold["selected_peak_indices"].extend(
            (torch.nonzero(keep).flatten().cpu().numpy() + offset).tolist())
        return peaks, zyx, scores[keep], identities

    def record_frame(peaks, zyx, probabilities):
        if not collect_prethreshold:
            return
        prethreshold["peak"].append(peaks.cpu().numpy())
        prethreshold["zyx"].append(zyx.cpu().numpy())
        prethreshold["probability"].append(probabilities.cpu().numpy())
        prethreshold["frame_offsets"].append(
            prethreshold["frame_offsets"][-1] + len(peaks))

    def record_pair_features(features, peaks0, peaks1):
        if not collect_prethreshold:
            return
        source = torch.stack([
            select_feature(feature[0][0][0], peaks0)
            for feature in features
        ], dim=1)
        target = torch.stack([
            select_feature(feature[1][0][0], peaks1)
            for feature in features
        ], dim=1)
        prethreshold["source_feature"].append(source.half().cpu().numpy())
        prethreshold["target_feature"].append(target.half().cpu().numpy())
        prethreshold["source_offsets"].append(
            prethreshold["source_offsets"][-1] + len(peaks0))
        prethreshold["target_offsets"].append(
            prethreshold["target_offsets"][-1] + len(peaks1))

    def fuse_links(features, peak0, peak1, zyx0, zyx1, coord0, coord1,
                   identities0, identities1):
        pos0 = embed_position(zyx0, t=0, cfg=cfg)
        pos1 = embed_position(zyx1, t=1, cfg=cfg)
        links = []
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled,
        ):
            for model, feature in zip(models, features, strict=True):
                select0 = tta_features(model, feature[0], peak0, zyx0, cfg.tta_head_fusion)
                select1 = tta_features(model, feature[1], peak1, zyx1, cfg.tta_head_fusion)
                views = select0.shape[0]
                links.append(model.predict_links(
                    select0, select1,
                    coord0[None].expand(views, -1, -1),
                    coord1[None].expand(views, -1, -1),
                    pos0[None].expand(views, -1, -1),
                    pos1[None].expand(views, -1, -1),
                    coord_scale=coord_scale))

        edge = torch.stack([link.edge.float() for link in links])
        appearance = torch.stack([link.appearance.float() for link in links])
        division = torch.stack([link.division.float() for link in links])
        edge = edge.mean((0, 1)).softmax(dim=0)
        appearance = appearance.mean((0, 1)).sigmoid()
        division = division.mean((0, 1)).sigmoid()
        dissim = None
        if identities0 is not None:
            dissim = torch.stack([
                1.0 - source.float() @ target.float().transpose(-1, -2)
                for source, target in zip(
                    identities0, identities1, strict=True)
            ]).mean((0, 1)).cpu().numpy()
        return (
            edge.cpu().numpy(),
            appearance.cpu().numpy(),
            division.cpu().numpy(),
            dissim,
        )

    next_pair = None
    detected1 = None
    frame_offset = 0
    drift_um = {}
    with torch.inference_mode(), fast_model_context(models):
        for t in range(T - 1):
            if next_pair is None:
                features, probabilities = encode_pair(t)
            else:
                features, probabilities = next_pair
                next_pair = None
            if cfg.temporal_detection_fusion and t + 1 < T - 1:
                next_pair = encode_pair(t + 1)
                probabilities[1] = (probabilities[1] + next_pair[1][0]) / 2

            detected0 = detect_frame(probabilities[0]) if t == 0 else detected1
            detected1 = detect_frame(probabilities[1])
            next_offset = frame_offset + len(detected0[0])
            for threshold, state in outputs.items():
                out_node, out_edge, out_node_prob, out_appearance, out_division = state["prediction"]
                out_start, prethreshold = state["starts"], state["cache"]
                if t == 0:
                    source = select_frame(detected0, features, 0, 0, threshold, frame_offset)
                    record_frame(*detected0)
                else:
                    source = state["previous"]
                target = select_frame(detected1, features, 1, t + 1, threshold, next_offset)
                state["previous"] = target
                record_frame(*detected1)
                record_pair_features(features, detected0[0], detected1[0])

                peak0, zyx0, prob0, identity0 = source
                peak1, zyx1, prob1, identity1 = target
                coord0 = network_to_image_coordinates(zyx0, meta["downsample"])
                coord1 = network_to_image_coordinates(zyx1, meta["downsample"])
                c0, c1 = coord0.float().cpu().numpy(), coord1.float().cpu().numpy()
                q0, q1 = prob0.float().cpu().numpy(), prob1.float().cpu().numpy()
                if cfg.drift_compensate["enable"]:
                    # the link head sees t+1 shifted back by the whole-frame drift; the
                    # drift is estimated once per frame pair and shared by all thresholds
                    if t not in drift_um:
                        drift_um[t] = frame_drift_um(c0, c1, meta["scale"], cfg.drift_compensate)
                    if np.any(drift_um[t]):
                        shift = torch.as_tensor(drift_um[t] / np.asarray(meta["scale"]), dtype=coord1.dtype, device=coord1.device)
                        coord1 = coord1 - shift
                        zyx1 = image_to_network_coordinates(coord1, meta["downsample"]).to(zyx1.dtype)
                if len(zyx0) == 0 or len(zyx1) == 0:
                    edge_prob = np.zeros((len(zyx0), len(zyx1)), np.float32)
                    id_dissim = None
                    appearance_prob = np.ones(len(zyx1), np.float32)
                    division_prob = np.zeros(len(zyx0), np.float32)
                else:
                    edge_prob, appearance_prob, division_prob, id_dissim = fuse_links(
                        features, peak0, peak1, zyx0, zyx1, coord0, coord1,
                        identity0, identity1)
                add_to_out(edge_prob, id_dissim, q0, q1, appearance_prob,
                           division_prob, c0, c1, t, t + 1)
            frame_offset = next_offset
            # Do not keep this pair's feature maps alive while encoding the
            # next pair. Temporal fusion's next_pair retains its own maps.
            del features, probabilities
    if cfg.drift_compensate["enable"]:
        print(f"drift-compensated frame pairs: {sum(np.any(d) for d in drift_um.values())}/{len(drift_um)}", flush=True)

    results = {}
    for threshold, state in outputs.items():
        prediction, prethreshold = state["prediction"], state["cache"]
        if collect_prethreshold:
            prethreshold.update({
                "node": prediction[0], "node_prob": prediction[2],
                "appearance_prob": prediction[3], "division_prob": prediction[4],
                "selected_frame_offsets": [state["starts"][t] for t in range(T)] + [len(prediction[0])],
                "scale": meta["voxel_size"], "downsample": meta["downsample"],
            })
        results[threshold] = prediction, prethreshold
    if point_thresholds is not None:
        return results
    prediction, prethreshold = results[cfg.point_threshold]
    return (prediction, prethreshold) if collect_prethreshold else prediction


def predict_graph_gpu(cfg, models, sample_id, split=None, collect_prethreshold=False):
    # GPU half of one clip: load, detect centres, score every (t, t+1) pair.
    # Returns numpy only, so solve_graph can run in another thread.
    mode = cfg.track_postprocess["mode"]
    low_threshold = cfg.track_postprocess["low_point_threshold"]
    if mode not in ("none", "no_new_divisions", "anchor_paths"):
        raise ValueError(f"Unknown track_postprocess mode: {mode}")
    if mode != "none" and not 0 <= low_threshold < cfg.point_threshold:
        raise ValueError("low_point_threshold must be below point_threshold")
    volume, meta = load_volume(cfg, sample_id, split=split)
    if mode == "none" or not cfg.ilp1["enable"]:
        # one detection set: point_threshold without post-processing, low_point_threshold without ILP1
        threshold = cfg.point_threshold if mode == "none" else low_threshold
        prediction, prethreshold = predict_clip(
            cfg, models, volume, meta, collect_prethreshold=collect_prethreshold,
            point_thresholds=(threshold,))[threshold]
        return dict(meta=meta, prediction=prediction, prethreshold=prethreshold)
    predictions = predict_clip(
        cfg, models, volume, meta, collect_prethreshold=collect_prethreshold,
        point_thresholds=(cfg.point_threshold, low_threshold))
    reference, reference_cache = predictions[cfg.point_threshold]
    prediction, prethreshold = predictions[low_threshold]
    return dict(meta=meta, prediction=prediction, prethreshold=prethreshold,
                reference=reference, reference_cache=reference_cache)


def solve_graph(cfg, staged):
    # Build tracks with up to three ILP passes, optionally repair links, then smooth and fill gaps.
    prediction, scale = staged["prediction"], np.asarray(staged["meta"]["scale"], dtype=np.float64)
    ilp1 = cfg.ilp1["enable"] and cfg.track_postprocess["mode"] != "none"
    if ilp1:
        # ILP1 on the point_threshold detections, a reference that prunes ILP2 and ILP3
        peaks = np.asarray(staged["prethreshold"]["selected_peak_indices"])
        reference_peaks = staged["reference_cache"]["selected_peak_indices"]
        coord, ref_edges, ref_q, *probs = staged["reference"]
        ref_kept = [e for e in ref_edges if e[2] > cfg.ilp_edge_threshold]
        reference = solve_ilp(cfg, build_graph(coord, ref_kept, ref_q, *probs))

    class ILP2Config(cfg):
        ilp_appearance_weight = cfg.ilp2["appearance_weight"]
        idsim = {**cfg.idsim, "weight": cfg.ilp2["idsim"]}

    coord, edges, *probs = prediction
    low_kept = [e for e in edges if e[2] > cfg.ilp2["edge_threshold"]]
    merged = solve_ilp(ILP2Config, build_graph(coord, low_kept, *probs))
    if ilp1 and cfg.ilp1["filter_ilp2"]:
        merged = filter_track_graph(ILP2Config, reference, merged, reference_peaks, peaks)
    es3 = relink_ilp(cfg, prediction, merged)
    if ilp1 and cfg.ilp1["filter_ilp3"]:
        es3 = filter_track_graph(cfg, reference, es3, reference_peaks, peaks)
    graph = es3
    if cfg.rematch["enable"]:
        graph = rematch_ambiguous(cfg, prediction, merged, es3, cfg.rematch)
    if cfg.gap_close["enable"]:
        graph = close_gaps(graph, prediction, scale, cfg.gap_close)
    if cfg.adjacent_join["enable"]:
        graph = join_adjacent(graph, prediction, scale, cfg.adjacent_join)
    return fill_track_gaps(cfg, smooth_track_coords(cfg, graph), scale)


def predict_graph(cfg, models, sample_id, split=None,
                  collect_prethreshold=False):
    # One clip end to end: detect centres, score every (t, t+1) pair, solve globally.
    # The single definition of "the pipeline", shared by submission, holdout and
    # in-training scoring so the three can never drift apart.
    staged = predict_graph_gpu(cfg, models, sample_id, split=split,
                               collect_prethreshold=collect_prethreshold)
    graph = solve_graph(cfg, staged)
    return (graph, staged["prethreshold"]) if collect_prethreshold else graph


def inference_devices(cfg):
    """Respect an explicit device/list; automatic mode uses only visible GPUs."""
    requested = cfg.inference_devices
    if requested is not None:
        if isinstance(requested, (str, bytes)):
            raise ValueError("inference_devices must be a list, e.g. ['cuda:0']")
        devices = [torch.device(f"cuda:{device}" if isinstance(device, int) else device)
                   for device in requested]
        if not devices:
            raise ValueError("inference_devices must not be empty")
        if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
            raise ValueError("Unavailable inference device: CUDA is not available")
        devices = [torch.device("cuda", torch.cuda.current_device())
                   if device.type == "cuda" and device.index is None else device
                   for device in devices]
    else:
        selected = torch.device(cfg.device)
        if selected.type != "cuda":
            devices = [selected]
        elif not torch.cuda.is_available():
            devices = [torch.device("cpu")]
        elif selected.index is not None:
            devices = [selected]
        else:
            devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    if len(set(map(str, devices))) != len(devices):
        raise ValueError("inference_devices must be unique")
    for device in devices:
        if device.type == "cuda" and (not torch.cuda.is_available()
                                      or device.index >= torch.cuda.device_count()):
            raise ValueError(f"Unavailable inference device: {device}")
    return devices


def run_clips(cfg, ids, devices=None):
    """Notebook-style persistent workers without changing the selected decoder."""
    from inference_fast import clip_costs, run_scheduled

    os.makedirs(cfg.predict_dir, exist_ok=True)
    ids = list(ids)
    devices = inference_devices(cfg) if devices is None else devices

    class Backend:
        @staticmethod
        def load_models(config, device):
            if device.type == "cuda":
                torch.cuda.set_device(device)
            ensemble = load_models(config, device)
            print(f"[fast {device}] loaded {len(ensemble)} checkpoints", flush=True)
            return ensemble

        @staticmethod
        def predict(config, ensemble, sample_id):
            return predict_graph_gpu(config, ensemble, sample_id)

        @staticmethod
        def solve_save(config, sample_id, staged):
            graph = solve_graph(config, staged)
            save_graph(graph, os.path.join(config.predict_dir, f"{sample_id}.geff"))

        @staticmethod
        def model_report(ensemble):
            stats = _fast_unet_runner(
                cfg.inference_tta_batch_size, cfg.inference_oom_fallback).stats(ensemble)
            paths = resolve_checkpoints(cfg)
            for row in stats["models"]:
                row["checkpoint"] = paths[row["ensemble_index"]]
            stats["unet_execution_order"] = model_execution_order(ensemble)
            stats["fusion_model_order"] = list(range(len(ensemble)))
            return stats

        @staticmethod
        def cleanup(device):
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

    schedule = run_scheduled(cfg, ids, devices, Backend(), costs=clip_costs(cfg, ids))
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "inference_schedule.json"), "w") as stream:
        json.dump(schedule, stream, indent=2)
    return schedule


def main(cfg=InferConfig):
    set_seed(cfg.seed)
    quiet_known_warnings()
    os.makedirs(cfg.predict_dir, exist_ok=True)

    ids = clip_ids(cfg)
    print(f"checkpoints={cfg.checkpoint_paths} split={cfg.split} clips={len(ids)} "
          f"tta={cfg.tta_mode if cfg.use_tta else 'none'} "
          f"temporal_fusion={cfg.temporal_detection_fusion} "
          f"thr={cfg.point_threshold} "
          f"disap={cfg.ilp_disappearance_weight}")
    run_clips(cfg, ids)
    write_submission(cfg, ids)


if __name__ == "__main__":
    main()
