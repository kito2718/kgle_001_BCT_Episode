"""div3f input geometry: frame windows, flow warping of neighbouring frames onto the centre frame,
and the D4 (rot90 / flip) transform of point coordinates used by the rot4 TTA.
"""
from pathlib import Path

import numpy as np
import torch



def frame_times(t, length, n_frames=3, frame_offset=0):
    """Choose input images without moving the labeled node's time or position."""
    if n_frames < 1 or n_frames % 2 != 1:
        raise ValueError('n_frames must be a positive odd number')
    if not isinstance(frame_offset, (int, np.integer)):
        raise ValueError('frame_offset must be an integer')
    half = n_frames // 2
    return tuple(min(max(t + frame_offset + d, 0), length - 1)
                 for d in range(-half, half + 1))


class FlowWarp:
    """Backward-warp the t-1 / t+1 frames onto frame t with the dense flow
    cache (<sid>.npy = (T-1, 3, Zf, Yf, Xf)
    f16, flow(t -> t+1) of boundary t in FLOW-grid voxels on the fixed [1,4,4]
    grid). A voxel v of frame t (ds grid) samples the next frame at v + f_t(v)
    and the previous frame at v - f_{t-1}(v): the cache is forward-only, so
    the previous frame uses the t-1 -> t field evaluated at the destination
    instead of at its source (exact up to the field's variation over one
    displacement, ~1-2 um on a field smooth over tens of um)."""

    FLOW_DS = np.array([1, 4, 4], np.float64)

    def __init__(self, root, ds, device="cpu", keep_raw=False):
        self.root, self.ds = Path(root), np.asarray(ds, np.float64)
        # native parent: also keep the raw neighbours -> 5 inputs
        # [raw t-1, warped t-1, t, warped t+1, raw t+1] (model.n_frames=5)
        self.keep_raw = bool(keep_raw)
        self.device = torch.device(device)
        self._mm = {}    # sid -> memmap (per worker, lazy)
        self._grid = {}  # (Z, Y, X) -> (ds-grid coords (Z,Y,X,3), flow-grid coords)


    def _coords(self, shape):
        g = self._grid.get(shape)
        if g is None:
            v = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float64) for n in shape],
                                           indexing="ij"), -1)
            orig = v * torch.from_numpy(self.ds) + torch.from_numpy((self.ds - 1) / 2)
            fg = (orig - torch.from_numpy((self.FLOW_DS - 1) / 2)) / torch.from_numpy(self.FLOW_DS)
            g = (v.float().to(self.device), fg.float().to(self.device))
            self._grid[shape] = g
        return g

    def warp(self, sid, t, frames, times):
        """frames (n, Z, Y, X) at `times` around centre t -> the same with every
        frame at t-1 / t+1 warped onto t (replicated boundary frames = t stay)."""
        from biohub_infer.ssl_flow.ops import sample_volume
        mm = self._mm.get(sid)
        if mm is None:
            mm = np.load(self.root / f"{sid}.npy", mmap_mode="r")
            # the cache must sit on the assumed FLOW_DS grid of the same volume:
            # a cache built on another grid would warp silently misaligned
            expect = tuple(int(n) * int(d) // int(f) for n, d, f in
                           zip(frames.shape[-3:], self.ds, self.FLOW_DS))
            if tuple(mm.shape[-3:]) != expect or mm.shape[1] != 3:
                raise ValueError(f"{self.root / sid}.npy: flow grid {mm.shape} does not match "
                                 f"FLOW_DS {self.FLOW_DS.tolist()} for frames {frames.shape[-3:]} "
                                 f"at ds {self.ds.tolist()} (expected (T-1, 3, *{expect}))")
            self._mm[sid] = mm
        out = frames.copy()
        v, fg = self._coords(frames.shape[-3:])
        for i, u in enumerate(times):
            if u == t:
                continue
            assert abs(u - t) == 1, "warp supports the t-1 / t+1 neighbours only"
            b = t if u > t else t - 1
            flow = torch.from_numpy(np.asarray(mm[b], np.float32))[None].to(self.device)
            d = sample_volume(flow, fg.reshape(-1, 3))[0]          # flow-grid voxels
            d = d * torch.from_numpy(self.FLOW_DS / self.ds).float().to(self.device)  # -> ds-grid voxels
            src = (v.reshape(-1, 3) + (d if u > t else -d)).reshape(*v.shape)
            vol = torch.from_numpy(np.ascontiguousarray(frames[i], np.float32))[None, None].to(self.device)
            out[i] = sample_volume(vol, src.reshape(-1, 3))[0, :, 0].reshape(v.shape[:3]).cpu().numpy()
        if self.keep_raw:
            assert len(times) == 3
            return np.stack([frames[0], out[0], out[1], out[2], frames[2]])
        return out


def apply_d4_points(pts, k, flip, shape, zflip=0):
    """Same D4 (+z-flip) element on (N, 3) zyx grid coords; shape = spatial
    (Z, Y, X) of the volume (Y == X for the rot90 to be shape-preserving).
    np.rot90 maps index i to S-1-i, whose continuous extension on the
    block-center grid is v -> S-1-v; one rot90 sends (y, x) -> (S-1-x, y)."""
    z, y, x = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()
    s = shape[-1] - 1
    for _ in range(k % 4):
        y, x = s - x, y
    if flip:
        x = s - x
    if zflip:
        z = (shape[0] - 1) - z
    return np.stack([z, y, x], 1)


