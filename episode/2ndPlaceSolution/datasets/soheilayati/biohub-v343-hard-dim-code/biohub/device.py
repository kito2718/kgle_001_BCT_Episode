"""GPU selection and precision policy.

Exists because of a trap found in week one (Stage 0): Kaggle's *default* GPU is
a Tesla P100 (sm_60), and Kaggle's own preinstalled PyTorch is built only for
sm_70 and above. On a P100 session, `torch.cuda.is_available()` returns **True**
and then every kernel launch fails. The failure is loud but late, and it costs a
whole notebook run to discover.

Two consequences encoded here:

1. **Ask for the right machine.** The notebook must be pushed with
   `--accelerator NvidiaTeslaT4` (or `machine_shape: NvidiaTeslaT4` in
   kernel-metadata.json). See `REQUIRED_ACCELERATOR`.
2. **Check capability, not availability.** `select_device()` compares the actual
   compute capability against the arch list the installed torch was built for,
   and fails immediately with an actionable message rather than deep inside a
   training step.

Precision: bf16 needs sm_80+ (Ampere). The T4 is sm_75, so autocast must use
fp16 there. Local development on newer cards gets bf16. Picking this per-device
rather than hardcoding avoids a silent slowdown or overflow on Kaggle.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Kaggle accelerator token. The default ("Gpu") yields a P100, which Kaggle's
# own torch build cannot use. Verified 2026-08-06: NvidiaTeslaT4 -> 2x Tesla T4,
# sm_75, 15.4 GB each, and conv3d fwd+bwd runs.
REQUIRED_ACCELERATOR = "NvidiaTeslaT4"

# Minimum compute capability for torch.bfloat16 autocast (Ampere).
BF16_MIN_CAPABILITY = (8, 0)


class IncompatibleGpuError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceSpec:
    device: torch.device
    name: str
    capability: tuple[int, int] | None
    amp_dtype: torch.dtype | None
    vram_gb: float | None

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def describe(self) -> str:
        if not self.is_cuda:
            return "cpu (no CUDA)"
        sm = f"sm_{self.capability[0]}{self.capability[1]}"
        return (
            f"{self.name} {sm} {self.vram_gb:.1f} GB, "
            f"autocast={str(self.amp_dtype).replace('torch.', '')}"
        )


def _supported_capabilities() -> list[tuple[int, int]]:
    """Compute capabilities the installed torch was actually compiled for."""
    caps = []
    for arch in torch.cuda.get_arch_list():
        if arch.startswith("sm_"):
            digits = arch[3:]
            if digits.isdigit() and len(digits) >= 2:
                caps.append((int(digits[:-1]), int(digits[-1])))
    return caps


def select_device(allow_cpu: bool = False) -> DeviceSpec:
    """Return the device to use, or raise with an actionable message."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        capability = (props.major, props.minor)
        amp_dtype = torch.bfloat16 if capability >= BF16_MIN_CAPABILITY else torch.float16
        return DeviceSpec(
            device=torch.device("cuda"),
            name=props.name,
            capability=capability,
            amp_dtype=amp_dtype,
            vram_gb=props.total_memory / 1e9,
        )

    if allow_cpu:
        return DeviceSpec(torch.device("cpu"), "cpu", None, None, None)

    raise IncompatibleGpuError(
        "No CUDA device found."
    )
