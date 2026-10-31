"""Constants and the output-channel layout."""

import numpy as np

SCALE = np.array([1.625, 0.40625, 0.40625])
NATIVE_SHAPE = (64, 256, 256)
POOL = 2
FLOW_SCALE = 5.0
TRACK_SCALE = 10.0
COMP_NUCLEUS_VOX = np.array([7.0, 19.0, 21.0])
BG_LABEL = 65535
SOON_CROP_SIZE = (16, 64, 64)
POOL_SCHEDULE = ((1, 2, 2), (1, 2, 2), (2, 2, 2))


def input_offsets(context: int = 0, horizons: int = 0) -> tuple:
    """Which timepoints, relative to the sampled frame, the INPUT channels carry."""
    c, k = (int(context), int(horizons))
    if c < 0:
        raise ValueError(f"context must be >= 0, got {c}")
    if k < 0:
        raise ValueError(f"horizons must be >= 0, got {k}")
    return tuple(sorted(set(range(-c, c + 1)) | set(range(0, k + 1))))


def parse_pool_schedule(s):
    """``"z,y,x;z,y,x;z,y,x"`` -> 3-tuple of (Z,Y,X) strides; None -> POOL_SCHEDULE."""
    if s is None:
        raise ValueError("an explicit pooling schedule is required")
    stages = [st for st in str(s).split(";") if st.strip()]
    if len(stages) != 3:
        raise ValueError(f"pool schedule needs exactly 3 stages, got {len(stages)}: {s!r}")
    out = []
    for st in stages:
        parts = [p.strip() for p in st.split(",")]
        if len(parts) != 3:
            raise ValueError(f"stage {st!r} needs 3 comma-separated strides (Z,Y,X)")
        try:
            vals = tuple((int(p) for p in parts))
        except ValueError:
            raise ValueError(f"stage {st!r} has a non-integer stride") from None
        if any((v < 1 for v in vals)):
            raise ValueError(f"stage {st!r} has a stride < 1")
        out.append(vals)
    return tuple(out)


OTSU_SCALE = 1.0
OTSU_CLASSES = 3
BG_DILATE = 1
BG_MIN_VOX = 128
BG_BORDER_ONLY = True
REPS = ("flow", "stardist")
DEFAULT_N_RAYS = 96
MIN_N_RAYS = 4
DIST_UNIT = "um"


class Channels:
    """Named output-channel slices, discriminated by instance representation `rep`."""

    __slots__ = ("horizons", "rep", "n_rays", "n_inst", "inst", "prob", "ttd", "out_ch", "contact", "_track0")

    def __init__(self, horizons: int = 0, rep: str = "flow", n_rays: int = DEFAULT_N_RAYS, contact: bool = False):
        h = int(horizons)
        if h < 0:
            raise ValueError(f"horizons must be >= 0, got {h}")
        rep = str(rep)
        if rep not in REPS:
            raise ValueError(f"rep must be one of {REPS}, got {rep!r}")
        self.horizons = h
        self.rep = rep
        if rep == "flow":
            self.n_rays = 0
            self.n_inst = 3
        else:
            nr = int(n_rays)
            if nr < MIN_N_RAYS:
                raise ValueError(f"n_rays must be >= {MIN_N_RAYS}, got {nr}")
            self.n_rays = nr
            self.n_inst = nr
        n = self.n_inst
        self.inst = slice(0, n)
        self._track0 = n
        self.prob = n + 3 * h
        self.ttd = n + 1 + 3 * h
        self.contact = n + 2 + 3 * h if contact else None
        self.out_ch = n + (3 if contact else 2) + 3 * h

    @property
    def flow(self) -> slice:
        """The 3 intra-mask flow channels."""
        if self.rep != "flow":
            raise ValueError(
                f"ch.flow is undefined at rep={self.rep!r} -- use ch.dist, or ch.inst for representation-agnostic code"
            )
        return self.inst

    @property
    def dist(self) -> slice:
        """The n_rays star-convex ray-distance channels."""
        if self.rep != "stardist":
            raise ValueError(
                f"ch.dist is undefined at rep={self.rep!r} -- use ch.flow, or ch.inst for representation-agnostic code"
            )
        return self.inst

    def track(self, k: int) -> slice:
        """Channel slice for horizon k (1-based: k=1 is t+1)."""
        k = int(k)
        if not 1 <= k <= self.horizons:
            raise ValueError(f"horizon {k} outside 1..{self.horizons}")
        s = self._track0 + 3 * (k - 1)
        return slice(s, s + 3)

    def aux(self) -> "Channels":
        """The Channels the deep-supervision aux heads emit: same rep, horizons=0."""
        return Channels(0, rep=self.rep, n_rays=self.n_rays or DEFAULT_N_RAYS)
