"""Soon probabilities and occupancy maps for detector-cache coordinates."""

import numpy as np
import torch
import torch.nn.functional as F
from cell_tracking.soon.crops import CROP_SIZE, extract_crop, make_gaussian_channel
from cell_tracking.soon.auxiliary import aux_division_probs, aux_occupancy_probs

TTA_FLIPS = {"none": ((),), "xy": ((), (1,), (2,), (1, 2))}
_SIGMOID_FP16_LUT = {}


def _tta_flips(tta):
    if tta not in TTA_FLIPS:
        raise ValueError(f"unsupported soon TTA: {tta!r}")
    return TTA_FLIPS[tta]


def _autocast_dtype(autocast):
    if autocast not in (None, "none", "fp16"):
        raise ValueError(f"unsupported scoring precision: {autocast!r}")
    return torch.float16 if autocast == "fp16" else None


def _crops_on_device(vol, zyx, *, crop_size, prev, nxt, device):
    """Cut three intensity channels and the query Gaussian on the device."""
    sz, sy, sx = (int(s) for s in crop_size)
    frames = [vol if prev is None else prev, vol, vol if nxt is None else nxt]
    v = torch.stack([torch.from_numpy(np.ascontiguousarray(f, dtype=np.float32)) for f in frames])
    v = F.pad(v.to(device), (sx, sx, sy, sy, sz, sz))
    lo = np.round(np.asarray(zyx, np.float64)).astype(np.int64) - np.array([sz // 2, sy // 2, sx // 2])
    lo = torch.from_numpy(lo + np.array([sz, sy, sx])).to(device)
    ar = [torch.arange(n, device=device) for n in (sz, sy, sx)]
    zi = (lo[:, 0, None] + ar[0])[:, :, None, None]
    yi = (lo[:, 1, None] + ar[1])[:, None, :, None]
    xi = (lo[:, 2, None] + ar[2])[:, None, None, :]
    crops = v[:, zi, yi, xi].transpose(0, 1)
    g = torch.from_numpy(make_gaussian_channel((sz, sy, sx), (sz / 2.0, sy / 2.0, sx / 2.0)))
    return torch.cat([crops, g.to(device).expand(len(crops), -1, -1, -1, -1)], dim=1)


def _crop_tensor(vol, zyx, *, crop_size, prev, nxt, device):
    if torch.device(device).type == "cuda":
        return _crops_on_device(vol, zyx, crop_size=crop_size, prev=prev, nxt=nxt, device=device)
    crops = np.stack([extract_crop(vol, p, size=crop_size, prev=prev, nxt=nxt) for p in zyx])
    return torch.from_numpy(crops.astype(np.float32))


def _coordinates(coords_native):
    zyx = np.asarray(coords_native, dtype=np.float64)
    if zyx.ndim != 2 or zyx.shape[1] != 3:
        raise ValueError(f"coords_native must be (N, 3), got {zyx.shape}")
    return zyx


def _empty_maps(net):
    return np.zeros((0, 3, *net.aux_out_grid), np.float32)


def _forward(net, xb, autocast, return_aux=False, to_cpu=True):
    dt = _autocast_dtype(autocast)
    with torch.autocast(device_type=xb.device.type, dtype=dt, enabled=dt is not None and xb.device.type == "cuda"):
        result = net(xb, return_aux=True) if return_aux else net(xb)
    if return_aux:
        logits, aux = result
        logits, aux = (logits.detach().float(), aux.detach().float())
        return (logits.cpu(), aux.cpu()) if to_cpu else (logits, aux)
    result = result.detach().float()
    return result.cpu() if to_cpu else result


def _unflip(maps, flips):
    return torch.flip(maps, dims=[2 + a for a in flips]) if flips else maps


def _sum_maps(acc, current):
    return current.double() if acc is None else acc + current.double()


def _aux_sigmoid_exact(aux):
    """Preserve CPU sigmoid rounding for fp16 auxiliary logits on CUDA."""
    if aux.dtype != torch.float16:
        return torch.sigmoid(aux.detach().float().cpu()).to(aux.device)
    key = str(aux.device)
    if key not in _SIGMOID_FP16_LUT:
        codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
        _SIGMOID_FP16_LUT[key] = torch.sigmoid(codes.float()).to(aux.device)
    return _SIGMOID_FP16_LUT[key][aux.view(torch.int16).long() & 65535]


@torch.no_grad()
def score_points(
    net,
    vol,
    coords_native,
    *,
    crop_size=CROP_SIZE,
    prev=None,
    nxt=None,
    device="cpu",
    batch=128,
    tta="xy",
    autocast=None,
):
    """Class scores, occupancy maps and division maps from one held-out model."""
    net.eval()
    zyx = _coordinates(coords_native)
    if not len(zyx):
        empty = _empty_maps(net)
        return (np.zeros((0, 3), np.float32), empty, empty)
    x = _crop_tensor(vol, zyx, crop_size=crop_size, prev=prev, nxt=nxt, device=device)
    flips = _tta_flips(tta)
    acc = aux_acc = div_acc = None
    on_device = torch.device(device).type == "cuda"
    for flip in flips:
        xf = torch.flip(x, dims=[2 + a for a in flip]) if flip else x
        logits, aux_chunks, div_chunks = ([], [], [])
        for start in range(0, len(xf), int(batch)):
            xb = xf[start : start + int(batch)].to(device)
            if on_device:
                dt = _autocast_dtype(autocast)
                with torch.autocast(device_type="cuda", dtype=dt, enabled=dt is not None):
                    logit, aux = net(xb, return_aux=True)
                logits.append(logit.detach().float().cpu())
                probabilities = _aux_sigmoid_exact(aux.detach())
                aux_chunks.append(torch.maximum(probabilities[:, :3], probabilities[:, 3:]))
                div_chunks.append(probabilities[:, 3:6])
            else:
                logit, aux = _forward(net, xb, autocast, return_aux=True)
                logits.append(logit)
                aux_chunks.append(aux_occupancy_probs(aux, True))
                div_chunks.append(aux_division_probs(aux, True))
        probabilities = torch.softmax(torch.cat(logits), dim=-1)
        acc = probabilities.double() if acc is None else acc + probabilities.double()
        aux_acc = _sum_maps(aux_acc, _unflip(torch.cat(aux_chunks, 0), flip))
        div_acc = _sum_maps(div_acc, _unflip(torch.cat(div_chunks, 0), flip))
    return (
        (acc / len(flips)).numpy().astype(np.float32),
        (aux_acc / float(len(flips))).cpu().numpy().astype(np.float32),
        (div_acc / float(len(flips))).cpu().numpy().astype(np.float32),
    )


@torch.no_grad()
def score_points_per_net(
    nets, vol, coords_native, *, crop_size=CROP_SIZE, prev=None, nxt=None, device="cpu", batch=64, autocast=None
):
    """Per-model class probabilities and ensemble-mean occupancy/division maps."""
    nets = list(nets)
    if not nets:
        raise ValueError("at least one soon model is required")
    zyx = _coordinates(coords_native)
    if not len(zyx):
        empty = _empty_maps(nets[0])
        return ([np.zeros((0, 3), np.float32) for net in nets], empty, empty)
    x = _crop_tensor(vol, zyx, crop_size=crop_size, prev=prev, nxt=nxt, device=device)
    for net in nets:
        net.eval()
    logits = [[] for net in nets]
    occupancy = [[] for net in nets]
    division = [[] for net in nets]
    for start in range(0, len(x), int(batch)):
        xb = x[start : start + int(batch)].to(device)
        for index, net in enumerate(nets):
            logit, aux = _forward(net, xb, autocast, return_aux=True, to_cpu=False)
            logits[index].append(logit)
            occupancy[index].append(aux_occupancy_probs(aux, True))
            division[index].append(aux_division_probs(aux, True))
    scores = [torch.softmax(torch.cat(parts), dim=-1).cpu().numpy().astype(np.float32) for parts in logits]
    occ = torch.stack([torch.cat(parts, 0) for parts in occupancy], dim=0).mean(dim=0)
    div = torch.stack([torch.cat(parts, 0) for parts in division], dim=0).mean(dim=0)
    return (scores, occ.cpu().numpy().astype(np.float32), div.cpu().numpy().astype(np.float32))


@torch.no_grad()
def score_points_shared(
    nets,
    vol,
    coords_native,
    *,
    crop_size=CROP_SIZE,
    prev=None,
    nxt=None,
    device="cpu",
    batch=128,
    tta="xy",
    autocast=None,
):
    """Average teacher probabilities over models and XY flips with shared crops."""
    nets = list(nets)
    if not nets:
        raise ValueError("at least one soon model is required")
    zyx = _coordinates(coords_native)
    if not len(zyx):
        return np.zeros((0, 3), np.float32)
    x = _crop_tensor(vol, zyx, crop_size=crop_size, prev=prev, nxt=nxt, device=device)
    for net in nets:
        net.eval()
    acc, count = (None, 0)
    for flip in _tta_flips(tta):
        xf = torch.flip(x, dims=[2 + a for a in flip]) if flip else x
        per_net = [[] for net in nets]
        for start in range(0, len(xf), int(batch)):
            xb = xf[start : start + int(batch)].to(device)
            for index, net in enumerate(nets):
                per_net[index].append(_forward(net, xb, autocast, to_cpu=False))
        for chunks in per_net:
            probabilities = torch.softmax(torch.cat(chunks), dim=-1)
            acc = probabilities.double() if acc is None else acc + probabilities.double()
            count += 1
    return (acc / count).float().cpu().numpy().astype(np.float32)
