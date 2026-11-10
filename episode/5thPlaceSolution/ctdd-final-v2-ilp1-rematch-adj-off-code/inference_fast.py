"""Neural inference accelerations and whole-clip scheduling.

Execution/chunk order changes allocation lifetimes, not ensemble reduction
order. Callers must store results at their original model indices. Larger
batches and need_weights=False may change floating-point rounding.

Scheduling uses one persistent ensemble per device, overlaps CPU decoding with
GPU prediction, and leaves final CSV order in the caller's original input order.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import copy
import math
import os
import queue
import threading
import time
from types import MethodType
import weakref

import torch
import zarr


_context_lock = threading.Lock()
_active_modules = set()


def _attention_without_weights(self, q, kv, kv_mask=None):
    key_padding_mask = ~kv_mask if kv_mask is not None else None
    attn_out, _ = self.cross_attn(
        self.norm1(q), self.norm1(kv), self.norm1(kv),
        key_padding_mask=key_padding_mask, need_weights=False,
    )
    q = q + attn_out
    q = q + self.mlp(self.norm2(q))
    return q


@contextmanager
def fast_model_context(models):
    """Temporarily disable discarded linker-attention weights on these instances.

    Separate worker threads must own separate models and child modules. Prior
    instance overrides are restored on success and exceptions. All modules must
    be in eval mode.
    This context does not change gradient settings, parameters, or model mode.
    """
    from models import CrossAttentionBlock

    modules = {id(module): module for model in models for module in model.modules()}
    if any(module.training for module in modules.values()):
        raise ValueError("Fast model context requires eval-mode models")
    owned = set(modules)
    with _context_lock:
        if _active_modules.intersection(owned):
            raise RuntimeError("Concurrent fast contexts must own separate model/module instances")
        _active_modules.update(owned)
    restored = []
    try:
        for block in modules.values():
            if isinstance(block, CrossAttentionBlock):
                restored.append((block, "forward" in block.__dict__,
                                 block.__dict__.get("forward")))
                block.forward = MethodType(_attention_without_weights, block)
        yield
    finally:
        for block, had_override, previous in reversed(restored):
            if had_override:
                block.forward = previous
            else:
                del block.forward
        with _context_lock:
            _active_modules.difference_update(owned)


def model_execution_order(models):
    """Run larger residual U-Nets first; keep original indices for fusion."""
    from models import _ResidualConvBlock

    def priority(index):
        first_block = models[index].unet.encoder_blocks[0]
        if isinstance(first_block, _ResidualConvBlock):
            return (0, -first_block.conv[0].out_channels)
        return (1, 0)

    return sorted(range(len(models)), key=priority)


def _allocate_outputs(features, logits, views):
    return (
        [[level.new_empty((views, *level.shape[1:])) for level in frame]
         for frame in features],
        [logit.new_empty((views, *logit.shape[1:])) for logit in logits],
    )


def _copy_outputs(outputs, features, logits, start, count):
    target_features, target_logits = outputs
    if (len(target_features) != len(features) or len(target_logits) != len(logits)
            or any(len(target) != len(frame)
                   for target, frame in zip(target_features, features))):
        raise ValueError("forward_unet output structure changed between chunks")
    pairs = [(target, level) for targets, frame in zip(target_features, features)
             for target, level in zip(targets, frame)]
    pairs.extend(zip(target_logits, logits))
    for target, source in pairs:
        if (source.shape != (count, *target.shape[1:])
                or source.dtype != target.dtype or source.device != target.device):
            raise ValueError("forward_unet output shape/dtype/device changed between chunks")
        target[start:start + count].copy_(source)


def _forward_chunks(model, image, batch_size):
    if len(image) <= batch_size:
        return model.forward_unet(image)
    outputs = None
    for start in range(0, len(image), batch_size):
        part = image[start:start + batch_size]
        features, logits = model.forward_unet(part)
        if outputs is None:
            outputs = _allocate_outputs(features, logits, len(image))
        _copy_outputs(outputs, features, logits, start, len(part))
        # Keep the final buffers plus only the current chunk; do not retain all
        # chunk outputs while also allocating their concatenated duplicates.
        del features, logits, part
    return outputs


class InferenceUNetRunner:
    """Eval-only TTA microbatching with persistent per-model OOM recovery.

    All views are retained, in their input order. The default retry sequence is
    8 -> 4 -> 2 -> 1. Only torch.OutOfMemoryError is retried; batch-1 OOM and all
    other exceptions propagate. State uses weak keys so finished model lists
    can be freed. Independent model instances may run concurrently.
    """

    def __init__(self, batch_size=8, oom_fallback=True, log=print):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.oom_fallback = bool(oom_fallback)
        self.log = log
        self._lock = threading.Lock()
        self._states = weakref.WeakKeyDictionary()
        self._next_index = 0
        self._active = set()

    @torch.inference_mode()
    def __call__(self, model, image):
        if model.training:
            raise ValueError("InferenceUNetRunner requires eval-mode models")
        if len(image) == 0:
            raise ValueError("TTA input must contain at least one view")
        with self._lock:
            if id(model) in self._active:
                raise RuntimeError("Concurrent calls must own separate model instances")
            self._active.add(id(model))
            if model not in self._states:
                self._states[model] = {
                    "model_index": self._next_index,
                    "model_type": type(model).__name__,
                    "device": str(image.device),
                    "chunk_limit": self.batch_size,
                    "successful_forwards": 0,
                    "actual_chunks": {}, "tta_views": {}, "oom_events": [],
                }
                self._next_index += 1
            state = self._states[model]
            batch = state["chunk_limit"]
        try:
            while True:
                try:
                    result = _forward_chunks(model, image, batch)
                except torch.OutOfMemoryError as error:
                    next_batch = max(1, batch // 2) if self.oom_fallback and batch > 1 else None
                    with self._lock:
                        state["oom_events"].append({
                            "device": str(image.device), "tta_views": len(image),
                            "attempted_chunk_limit": batch,
                            "retry_chunk_limit": next_batch, "error": str(error),
                        })
                    if next_batch is None:
                        raise
                else:
                    chunks = "+".join(str(min(batch, len(image) - start))
                                      for start in range(0, len(image), batch))
                    views = str(len(image))
                    with self._lock:
                        state["device"] = str(image.device)
                        state["chunk_limit"] = batch
                        state["successful_forwards"] += 1
                        state["actual_chunks"][chunks] = state["actual_chunks"].get(chunks, 0) + 1
                        state["tta_views"][views] = state["tta_views"].get(views, 0) + 1
                    return result
                # The exception scope has ended, releasing failed-frame tensors
                # and its traceback before logging, empty_cache, and retrying.
                if self.log is not None:
                    self.log(f"[fast neural] OOM {image.device} model {state['model_index']}: "
                             f"TTA chunk {batch}->{next_batch}; retaining all {len(image)} views")
                if image.device.type == "cuda":
                    with torch.cuda.device(image.device):
                        torch.cuda.empty_cache()
                batch = next_batch
        finally:
            with self._lock:
                self._active.remove(id(model))

    def stats(self, models):
        """Copy this ensemble's runtime statistics in original model order."""
        with self._lock:
            records = []
            for index, model in enumerate(models):
                state = self._states.get(model)
                if state is not None:
                    records.append({**state, "ensemble_index": index})
            return {
                "initial_chunk_limit": self.batch_size,
                "oom_fallback": self.oom_fallback,
                "models": copy.deepcopy(records),
            }


def clip_costs(cfg, ids):
    """Read only Zarr metadata; frame pairs are a hint, not a runtime prediction."""
    directory = cfg.train_dir if cfg.split == "train" else cfg.test_dir
    costs = {}
    for sample_id in ids:
        try:
            path = os.path.join(directory, f"{sample_id}.zarr")
            shape = zarr.open_group(path, mode="r")["0"].shape
            if len(shape) != 4 or min(shape) < 1:
                raise ValueError("Expected a positive TZYX image shape")
            costs[sample_id] = max(shape[0] - 1, 1)
        except Exception as error:
            # A metadata hint must never prevent normal input loading/reporting.
            print(f"[schedule] {sample_id}: no frame-count hint ({type(error).__name__}); FIFO fallback",
                  flush=True)
            costs[sample_id] = 0
    return costs


def run_scheduled(cfg, ids, devices, backend, costs):
    """Run a clip queue using the Backend defined in inference.run_clips.

    CPU solve(i) overlaps GPU predict(i+1), with at most one pending CPU future
    and one newly staged clip per worker, as in the original inference loop.
    """
    ids, devices = list(ids), list(devices)
    if len(set(ids)) != len(ids):
        raise ValueError("Clip IDs must be unique")
    if len({str(device) for device in devices}) != len(devices):
        raise ValueError("Inference devices must be unique")
    if ids and not devices:
        raise ValueError("At least one inference device is required")
    jobs = queue.PriorityQueue()
    for index, sample_id in enumerate(ids):
        cost = float(costs[sample_id])
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(f"Invalid scheduling cost for {sample_id}: {cost}")
        jobs.put((-cost, index, sample_id))

    started = time.perf_counter()
    failed = threading.Event()
    lock = threading.Lock()
    report = {"policy": "shared longest-frame-count-first queue", "input_ids": ids,
              "devices": [str(device) for device in devices[:len(ids)]],
              "workers": [], "assignments": []}

    def worker(device):
        models = None
        worker_report = {"device": str(device), "clips": [],
                         "start_seconds": time.perf_counter() - started}
        with lock:
            report["workers"].append(worker_report)
        try:
            load_start = time.perf_counter()
            models = backend.load_models(cfg, device)
            worker_report["model_load_seconds"] = time.perf_counter() - load_start

            def solve_save(sample_id, staged, assignment):
                solve_start = time.perf_counter()
                try:
                    backend.solve_save(cfg, sample_id, staged)
                except BaseException:
                    assignment["status"] = "failed"
                    failed.set()
                    raise
                assignment.update(status="complete",
                                  solve_and_save_seconds=time.perf_counter() - solve_start,
                                  complete_seconds=time.perf_counter() - started)
                print(f"[fast {device}] {sample_id}: complete | "
                      f"predict={assignment['prediction_seconds']:.1f}s "
                      f"post+save={assignment['solve_and_save_seconds']:.1f}s", flush=True)

            pending = None
            with ThreadPoolExecutor(max_workers=1) as cpu:
                while not failed.is_set():
                    with lock:
                        if failed.is_set():
                            break
                        try:
                            negative_cost, index, sample_id = jobs.get_nowait()
                        except queue.Empty:
                            break
                        assignment = {"clip": sample_id, "device": str(device),
                                      "input_index": index, "cost": -negative_cost,
                                      "status": "predicting"}
                        report["assignments"].append(assignment)
                        worker_report["clips"].append(sample_id)
                    print(f"[fast {device}] {sample_id}: predicting", flush=True)
                    predict_start = time.perf_counter()
                    staged = backend.predict(cfg, models, sample_id)
                    assignment["prediction_seconds"] = time.perf_counter() - predict_start
                    if pending is not None:
                        pending.result()
                    assignment["status"] = "postprocessing"
                    pending = cpu.submit(solve_save, sample_id, staged, assignment)
                    # The future owns this clip while CPU decoding runs; do not
                    # retain an extra reference through the next GPU forward.
                    del staged
                if pending is not None:
                    pending.result()
            worker_report["neural"] = backend.model_report(models)
        except BaseException:
            failed.set()
            raise
        finally:
            del models
            backend.cleanup(device)
            worker_report["end_seconds"] = time.perf_counter() - started

    if ids:
        active_devices = devices[:min(len(ids), len(devices))]
        with ThreadPoolExecutor(max_workers=len(active_devices)) as pool:
            futures = [pool.submit(worker, device) for device in active_devices]
            for future in as_completed(futures):
                future.result()
    completed = [row["clip"] for row in report["assignments"] if row["status"] == "complete"]
    if sorted(completed) != sorted(ids):
        raise RuntimeError("Scheduling failed to complete every input clip exactly once")
    report["wall_seconds"] = time.perf_counter() - started
    report["workers"].sort(key=lambda row: row["device"])
    return report
