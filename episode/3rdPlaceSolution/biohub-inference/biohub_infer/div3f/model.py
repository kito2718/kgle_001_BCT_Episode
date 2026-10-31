"""div3f helpers used at inference (the div3f networks themselves run as TensorRT engines, see trt.py)."""
from biohub_infer.ssl_flow.ops import sample_volume


def sample_logits(logits, points_zyx):
    """Trilinearly sample a (1, 1, Z, Y, X) logit volume at (N, 3) zyx grid
    coords -> (N,) logits. Detection coords are fractional after to_ds, so
    nearest-index sampling would be off by up to ~2 um."""
    return sample_volume(logits, points_zyx)[0, :, 0]
