"""TensorRT engine runners on torch CUDA tensors (no host copies).

Contexts are created without their own activation memory and share one workspace per device, sized to
the largest engine seen: the engines run strictly one after another, and fifty-five resident contexts with
private memory would not fit a T4 next to the torch models.

  TrtModel     one input / one output (the detector engines: `images` (B,1,Z,Y,X) -> `heatmap`), called as
               model(x) inside common.forward_tta
  TrtIO        several named inputs / outputs (the div3f encoder and decoder engines)
  EngineDiv3F  encode / decode / __call__ of an eval Div3FNet25D, backed by an encoder and a decoder engine
"""
from __future__ import annotations

import json
from pathlib import Path

import tensorrt as trt
import torch

_DTYPE = {trt.float32: torch.float32, trt.float16: torch.float16}
_WORKSPACE = {}   # device -> shared activation buffer (torch.uint8)


class _Engine:
    def __init__(self, engine_path, device='cuda'):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context_without_device_memory()
        self.mem_size = int(getattr(self.engine, 'device_memory_size_v2', None) or self.engine.device_memory_size)
        self._bound_ptr = None
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.device = torch.device(device)
        self.max_batch = self.engine.get_tensor_profile_shape(self.inputs[0], 0)[2][0]

    def eval(self):
        return self

    def _bind_workspace(self):
        buf = _WORKSPACE.get(self.device)
        if buf is None or buf.numel() < self.mem_size:
            buf = torch.empty(self.mem_size, dtype=torch.uint8, device=self.device)
            _WORKSPACE[self.device] = buf
        if self._bound_ptr != buf.data_ptr():
            if hasattr(self.context, 'set_device_memory_v2'):
                self.context.set_device_memory_v2(buf.data_ptr(), buf.numel())
            else:
                self.context.device_memory = buf.data_ptr()
            self._bound_ptr = buf.data_ptr()

    def _execute(self):
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        stream.synchronize()


class TrtModel(_Engine):
    def __init__(self, engine_path, device='cuda'):
        super().__init__(engine_path, device)
        assert len(self.inputs) == 1 and len(self.outputs) == 1, (self.inputs, self.outputs)

    def __call__(self, x):
        name_in, name_out = self.inputs[0], self.outputs[0]
        self._bind_workspace()
        x = x.to(self.device, _DTYPE[self.engine.get_tensor_dtype(name_in)]).contiguous()
        if x.shape[0] > self.max_batch:
            return torch.cat([self(x[i:i + self.max_batch]) for i in range(0, x.shape[0], self.max_batch)])
        if not self.context.set_input_shape(name_in, tuple(x.shape)):
            raise ValueError(f'engine rejects input shape {tuple(x.shape)}')
        out = torch.empty(tuple(self.context.get_tensor_shape(name_out)), device=self.device,
                          dtype=_DTYPE[self.engine.get_tensor_dtype(name_out)])
        self.context.set_tensor_address(name_in, x.data_ptr())
        self.context.set_tensor_address(name_out, out.data_ptr())
        self._execute()
        return out


class TrtIO(_Engine):
    def __call__(self, **tensors):
        """Inputs by name -> list of outputs in engine order; batches beyond the profile are split."""
        missing = [n for n in self.inputs if n not in tensors]
        if missing or len(tensors) != len(self.inputs):
            raise ValueError(f'engine expects inputs {self.inputs}, got {sorted(tensors)}')
        n = tensors[self.inputs[0]].shape[0]
        if n > self.max_batch:
            parts = [self(**{k: v[i:i + self.max_batch] for k, v in tensors.items()})
                     for i in range(0, n, self.max_batch)]
            return [torch.cat(p) for p in zip(*parts)]
        self._bind_workspace()
        keep, outs = [], []
        for name in self.inputs:
            x = tensors[name].to(self.device, _DTYPE[self.engine.get_tensor_dtype(name)]).contiguous()
            if x.shape[0] != n:
                raise ValueError(f'input batches differ: {name} has {x.shape[0]}, expected {n}')
            if not self.context.set_input_shape(name, tuple(x.shape)):
                raise ValueError(f'engine rejects {name} shape {tuple(x.shape)}')
            self.context.set_tensor_address(name, x.data_ptr())
            keep.append(x)
        for name in self.outputs:
            out = torch.empty(tuple(self.context.get_tensor_shape(name)), device=self.device,
                              dtype=_DTYPE[self.engine.get_tensor_dtype(name)])
            self.context.set_tensor_address(name, out.data_ptr())
            outs.append(out)
        self._execute()
        return outs


class EngineDiv3F:
    def __init__(self, enc, dec, n_frames, in_planes=1):
        self.enc, self.dec = enc, dec
        self.n_frames, self.in_planes = int(n_frames), int(in_planes)
        self.stages = list(enc.outputs)
        if enc.inputs != ['frames'] or dec.inputs != self.stages or dec.outputs != ['logits']:
            raise ValueError(f'unexpected engine tensors: enc {enc.inputs}->{enc.outputs}, '
                             f'dec {dec.inputs}->{dec.outputs}')

    def eval(self):
        return self

    def encode(self, frames):
        """(N, in_planes, Z, Y, X) -> [stage1, ..., stageK] (the raw-frame stage 0 is not returned)."""
        return self.enc(frames=frames)

    def decode(self, feats, size):
        """Per-stage features with n_frames frames concatenated item-major -> (B, 1, *size)."""
        feats = list(feats)
        if len(feats) != len(self.stages):
            raise ValueError(f'decode expects {len(self.stages)} stages, got {len(feats)}')
        out, = self.dec(**dict(zip(self.stages, feats)))
        if tuple(out.shape[-3:]) != tuple(int(v) for v in size):
            raise ValueError(f'engine logits {tuple(out.shape)} do not match size {tuple(size)}')
        return out

    def __call__(self, x):
        """(B, n_frames[, in_planes], Z, Y, X) -> (B, 1, Z, Y, X)."""
        b, n = x.shape[0], x.shape[1]
        if n != self.n_frames:
            raise ValueError(f'engine takes {self.n_frames} frames per item, got {n}')
        feats = self.encode(x.reshape(b * n, self.in_planes, *x.shape[-3:]))
        return self.decode([f.reshape(b, -1, *f.shape[-3:]) for f in feats], x.shape[-3:])


def load_div3f_roles(paths, role_stems, manifest_path, device='cuda'):
    """{role: [EngineDiv3F per stem]}; frame count and input planes from the build manifest."""
    manifest = json.loads(Path(manifest_path).read_text())
    roles = {}
    for role, stems in role_stems.items():
        roles[role] = []
        for stem in stems:
            entry = manifest[stem]
            enc, dec = (TrtIO(paths[f'{stem}_{kind}.engine'], device) for kind in ('enc', 'dec'))
            roles[role].append(EngineDiv3F(enc, dec, entry['n_frames'], entry.get('in_planes', 1)))
    return roles
