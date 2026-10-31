"""Frame normalization and mean-pool downsampling for the flow model."""
import numpy as np


def normalize(frame, qlo, qhi):
    return np.clip((frame.astype(np.float32) - qlo) / (qhi - qlo + 1e-6), 0, None)


def downsample_mean(x, factors):
    dz, dy, dx = factors
    z, y, xx = x.shape
    x = x[:z // dz * dz, :y // dy * dy, :xx // dx * dx]
    return x.reshape(z // dz, dz, y // dy, dy, xx // dx, dx).mean((1, 3, 5))


