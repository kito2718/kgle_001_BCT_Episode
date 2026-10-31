"""Compositing primitives — donor signal into real movies, additively.

Conventions (one place, stated once):

- Donor patches are NORMALIZED RAW-resolution voxels, shape (13, 49, 49)
  (z, y, x) centered on the donor's integer GT voxel: ±6 z ≈ ±9.75 um,
  ±24 y/x ≈ ±9.75 um. Normalization is the exact cache path
  (`normalise`, lo 0.02 / hi 0.999) — donor units are the training units.
- Placement is INTEGER-voxel. Real GT annotations are integer voxels; a
  trilinear shift would smooth donor noise texture (variance x0.5 per axis
  at half-voxel shifts) — the exact class of artifact a discriminator
  catches. Continuous geometry is rounded at the last step.
- Compositing is ADDITIVE in normalized space (fluorescence adds light);
  background pixels are never modified. `downsample` is mean-pooling, so
  pooling the padded signal patch and adding to the cached grid volume is
  EXACTLY equal to compositing at raw resolution and re-pooling the frame
  (tests/test_synthesis.py pins the identity).
"""

from __future__ import annotations

import numpy as np

from biohub.pipeline.baseline import DOWNSAMPLE, GRID_UM
from biohub.metrics.gt import SCALE_ARR

PATCH_Z, PATCH_Y, PATCH_X = 6, 24, 24  # half-extents, raw voxels
PATCH_SHAPE = (2 * PATCH_Z + 1, 2 * PATCH_Y + 1, 2 * PATCH_X + 1)
BORDER_XY = 3          # xy ring width for the background shell, raw voxels
FEATHER_PX = 3.0       # cosine feather width outside the footprint, raw voxels
FOOT_SMOOTH_UM = 1.5   # smoothing before footprint thresholding
FOOT_REL = 0.05        # footprint threshold as fraction of smoothed peak
FOOT_ABS_MAD = 3.0     # ... and as multiple of border-shell MAD
DILATE_PX = 2


def estimate_background(patch: np.ndarray) -> tuple[float, float]:
    """(median, MAD) of the xy border ring across all z — far from the cell
    (the ring sits >= 8.5 um from center in xy; cell extent ~2 HWHM = 7.6 um).
    z faces are NOT used: the axial PSF tail can still carry signal there."""
    ring = np.concatenate([
        patch[:, :BORDER_XY, :].ravel(), patch[:, -BORDER_XY:, :].ravel(),
        patch[:, BORDER_XY:-BORDER_XY, :BORDER_XY].ravel(),
        patch[:, BORDER_XY:-BORDER_XY, -BORDER_XY:].ravel(),
    ])
    med = float(np.median(ring))
    mad = float(np.median(np.abs(ring - med))) + 1e-9
    return med, mad


def footprint_mask(signal: np.ndarray,
                   feather_px: float = FEATHER_PX) -> np.ndarray:
    """Soft mask confining pasted signal (and its real noise) to the cell.

    Smooth -> threshold (relative to peak AND to border noise) -> connected
    component containing the center -> dilate -> cosine feather. Confinement
    matters: donor signal away from the cell is pure donor noise, and adding
    that across the whole patch would double background noise variance — a
    discriminator gift. Inside the footprint the noise is the donor cell's
    own, which is exactly what a real cell carries.
    """
    from scipy import ndimage

    sig_z = FOOT_SMOOTH_UM / float(SCALE_ARR[0])
    sig_xy = FOOT_SMOOTH_UM / float(SCALE_ARR[1])
    sm = ndimage.gaussian_filter(signal.astype(np.float32),
                                 sigma=(sig_z, sig_xy, sig_xy))
    _, mad = estimate_background(signal)
    thr = max(FOOT_REL * float(sm.max()), FOOT_ABS_MAD * mad)
    core = sm > thr
    center = tuple(s // 2 for s in signal.shape)
    if not core.any():
        return np.zeros_like(sm)
    lab, _ = ndimage.label(core)
    lc = lab[center]
    if lc == 0:
        # center voxel below threshold (very faint donor): take the component
        # nearest the center rather than nothing
        idx = np.argwhere(core)
        d = np.abs(idx - np.array(center)).sum(axis=1)
        lc = lab[tuple(idx[np.argmin(d)])]
    foot = lab == lc
    foot = ndimage.binary_dilation(foot, iterations=DILATE_PX)
    dist = ndimage.distance_transform_edt(~foot)
    t = np.clip(dist / feather_px, 0.0, 1.0)
    return (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)


def donor_signal(patch: np.ndarray,
                 feather_px: float = FEATHER_PX
                 ) -> tuple[np.ndarray, float, float]:
    """Background-subtracted, footprint-masked signal (float32, >= -2 MAD)."""
    med, mad = estimate_background(patch)
    sig = patch.astype(np.float32) - med
    mask = footprint_mask(sig, feather_px=feather_px)
    sig = sig * mask
    np.clip(sig, -2.0 * mad, None, out=sig)
    return sig, med, mad


def blur_signal(sig: np.ndarray, sigma_um_xy: float,
                z_factor: float = 0.5) -> np.ndarray:
    """Depth-scatter blur on a raw-resolution signal patch.

    Scatter at depth is mostly lateral: full sigma in y/x, `z_factor` of it
    axially. Applied BEFORE attenuation calibration, so alpha is fit to the
    blurred cell and the prominence target stays exact.
    """
    if sigma_um_xy <= 1e-3:
        return sig
    from scipy import ndimage

    s = (z_factor * sigma_um_xy / float(SCALE_ARR[0]),
         sigma_um_xy / float(SCALE_ARR[1]),
         sigma_um_xy / float(SCALE_ARR[2]))
    return ndimage.gaussian_filter(sig, sigma=s)


def hwhm_um_grid(crop: np.ndarray) -> float:
    """FWHM/2 (um) along y/x through the peak of the z-max projection of a
    grid-space crop. Shared by the fidelity gate and the faint calibrator."""
    proj = crop.max(axis=0)
    n = proj.shape[0]
    lo, hi = n // 2 - 3, n // 2 + 3
    cy, cx = np.unravel_index(int(np.argmax(proj[lo:hi, lo:hi])), (hi - lo,
                                                                   hi - lo))
    cy, cx = cy + lo, cx + lo
    pk = proj[cy, cx]
    bg = float(np.median(proj))
    half = bg + 0.5 * (pk - bg)
    widths = []
    for line in (proj[cy, :], proj[:, cx]):
        above = np.flatnonzero(line >= half)
        if len(above):
            widths.append((above.max() - above.min() + 1) / 2.0)
    return float(np.mean(widths) * 1.625) if widths else np.nan


def pooled_patch(signal: np.ndarray, center_vox: np.ndarray,
                 vol_shape_raw: tuple[int, int, int]
                 ) -> tuple[tuple[int, int, int], np.ndarray] | None:
    """Place `signal` (raw voxels) at integer `center_vox` and mean-pool.

    Returns `((gz, gy, gx), grid_add)` — origin in grid voxels and the pooled
    float32 patch to ADD there — or None if fully outside the volume. Edges
    are zero-padded to [1,4,4] block boundaries, so partially covered blocks
    receive exactly the mean over the covered voxels / 16, which is what
    raw-space compositing followed by pooling would produce.
    """
    cz, cy, cx = (int(round(c)) for c in center_vox)
    hz, hy, hx = (s // 2 for s in signal.shape)
    z0, z1 = cz - hz, cz + hz + 1
    y0, y1 = cy - hy, cy + hy + 1
    x0, x1 = cx - hx, cx + hx + 1
    # crop to volume
    sz0, sy0, sx0 = max(0, -z0), max(0, -y0), max(0, -x0)
    z0, y0, x0 = max(z0, 0), max(y0, 0), max(x0, 0)
    z1 = min(z1, vol_shape_raw[0])
    y1 = min(y1, vol_shape_raw[1])
    x1 = min(x1, vol_shape_raw[2])
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return None
    crop = signal[sz0:sz0 + (z1 - z0), sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    # align to pooling blocks (z factor is 1)
    dy, dx = int(DOWNSAMPLE[1]), int(DOWNSAMPLE[2])
    y0a, x0a = (y0 // dy) * dy, (x0 // dx) * dx
    y1a, x1a = -(-y1 // dy) * dy, -(-x1 // dx) * dx
    buf = np.zeros((z1 - z0, y1a - y0a, x1a - x0a), dtype=np.float32)
    buf[:, y0 - y0a:y1 - y0a, x0 - x0a:x1 - x0a] = crop
    z, y, x = buf.shape
    pooled = buf.reshape(z, y // dy, dy, x // dx, dx).mean(axis=(2, 4))
    return (z0, y0a // dy, x0a // dx), pooled


def add_patches(canvas: np.ndarray, patches: list[tuple[tuple[int, int, int], np.ndarray]]) -> None:
    """ADD pooled patches onto a (Z, Y, X) float32 grid frame, in place."""
    for (gz, gy, gx), p in patches:
        canvas[gz:gz + p.shape[0], gy:gy + p.shape[1], gx:gx + p.shape[2]] += p


# ---- QC / calibration -------------------------------------------------------

HALF_UM = 12.0
SIGMA_UM = 2.0


def measure_prominence_grid(vol_grid: np.ndarray, center_src_vox: np.ndarray
                            ) -> dict | None:
    """Exact port of `stage6_resolution_proof.measure` at the [1,4,4] grid —
    the estimator whose killed/matched profile (4.2 vs 6.2 MAD) the faint arm
    is calibrated against."""
    from scipy import ndimage

    grid_um = np.asarray(GRID_UM, dtype=np.float64)
    center = np.asarray(center_src_vox, np.float64) / (grid_um / SCALE_ARR)
    half = np.maximum((HALF_UM / grid_um).astype(int), 2)
    lo = np.maximum(center.astype(int) - half, 0)
    hi = np.minimum(center.astype(int) + half + 1, np.array(vol_grid.shape))
    patch = vol_grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    if min(patch.shape) < 3:
        return None
    sigma = SIGMA_UM / grid_um
    resp = -ndimage.gaussian_laplace(patch, sigma=sigma)
    mx = ndimage.maximum_filter(resp, size=3)
    peaks = np.argwhere((resp == mx) & (resp > 0))
    if not len(peaks):
        return None
    peaks_um = (peaks + lo) * grid_um
    center_um = np.asarray(center_src_vox, np.float64) * SCALE_ARR
    d = np.linalg.norm(peaks_um - center_um, axis=1)
    vals = resp[tuple(peaks.T)]
    near = d <= 5.0
    if not near.any():
        best_i = int(np.argmin(d))
    else:
        cand = np.flatnonzero(near)
        best_i = int(cand[np.argmax(vals[cand])])
    med = float(np.median(resp))
    mad = float(np.median(np.abs(resp - med))) + 1e-9
    return {
        "peak_dist_um": float(d[best_i]),
        "prominence": float((vals[best_i] - med) / mad),
        "n_peaks_5um": int(near.sum()),
    }


def sigma_at(curve: dict, mu: float) -> float:
    """Interpolate the mined grid-level noise curve sigma(intensity)."""
    xs, ys = curve["mu"], curve["sigma"]
    return float(np.interp(mu, xs, ys))


def noise_compensation(pooled: np.ndarray, canvas_region: np.ndarray,
                       alpha: float, curve: dict,
                       rng: np.random.Generator,
                       residual_bank: np.ndarray | None = None) -> np.ndarray:
    """Shot-noise deficit filler for attenuated pastes.

    Attenuation by alpha scales donor noise by alpha, but a real cell at that
    brightness carries ~sqrt(alpha)-scaled noise. Per grid voxel where signal
    was added: sigma_target = sigma(mu_after) from the sample's measured
    curve; sigma_present ~ sqrt(sigma(mu_bg)^2 + alpha^2 * dsig^2) with
    dsig = sigma(mu_donor_equiv) - sigma(mu_bg); the deficit is drawn from
    the residual bank (real high-pass residuals) when given, else Gaussian.
    """
    m = pooled > 1e-4
    if not m.any():
        return pooled
    mu_bg = np.clip(canvas_region, 0.0, None)
    mu_after = mu_bg + pooled
    sig_t = np.vectorize(lambda v: sigma_at(curve, v))(mu_after[m])
    sig_bg = np.vectorize(lambda v: sigma_at(curve, v))(mu_bg[m])
    mu_full = mu_bg + pooled / max(alpha, 1e-3)
    sig_full = np.vectorize(lambda v: sigma_at(curve, v))(mu_full[m])
    present = np.sqrt(sig_bg**2 + (alpha * np.clip(sig_full - sig_bg, 0, None))**2)
    deficit = np.sqrt(np.clip(sig_t**2 - present**2, 0.0, None))
    out = pooled.copy()
    if residual_bank is not None and residual_bank.size >= m.sum():
        draw = rng.choice(residual_bank, size=int(m.sum()), replace=True)
        unit = draw / (1.4826 * (np.median(np.abs(draw - np.median(draw))) + 1e-9))
    else:
        unit = rng.standard_normal(int(m.sum()))
    out[m] += (deficit * unit).astype(np.float32)
    return out


def residual_bank_from_frame(frame_grid: np.ndarray, n: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Real high-pass residuals sampled from tissue voxels of a grid frame."""
    from scipy import ndimage

    mu = ndimage.uniform_filter(frame_grid.astype(np.float32), size=3)
    r = frame_grid - mu
    tissue = mu > 0.05
    vals = r[tissue]
    if vals.size == 0:
        vals = r.ravel()
    idx = rng.choice(vals.size, size=min(n, vals.size), replace=False)
    return vals[idx].astype(np.float32)
