"""Export exact submitted detector/edge candidates without running a tracker.

The original infer_video is interrupted at build_graph, after candidate creation.
No selected graph or tracking score is emitted by this candidate-only stage.
"""
from __future__ import annotations
import __future__, ast, hashlib, contextlib, importlib.util, json, os, pickle, sys, types
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'legacy'))
DEST=Path(os.environ['BIOHUB_CANDIDATE_DIR']);DEST.mkdir(parents=True,exist_ok=True)

class Captured(Exception): pass
current=None
contract=hashlib.sha256((B/'legacy/biohub_motion_submission.py').read_bytes()+json.dumps(sys.argv[1:]).encode()).hexdigest()
def capture_graph(rows,edges):
    payload=dict(contract=contract,stem=current,node_rows=rows,edges=edges,stage='candidates_only',tracking_evaluated=False)
    p=DEST/f'{current}.pkl';tmp=p.with_suffix('.tmp');tmp.write_bytes(pickle.dumps(payload,protocol=5));tmp.replace(p)
    raise Captured

def forbidden(*args,**kwargs):
    raise RuntimeError('Tracking is prohibited in the detector-capture stage')

# Import only the unchanged pure candidate-edge function, excluding solver imports.
tree=ast.parse((B/'legacy/biohub_motion_ilp_fold.py').read_text())
module=types.ModuleType('biohub_motion_ilp_fold')
module.__dict__.update(np=np,cKDTree=cKDTree,SPACING_ZYX_UM=np.array([1.625,.40625,.40625]),build_graph=capture_graph,save_graph=forbidden,suppress_solver_output=contextlib.nullcontext)
body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='add_candidate_edges']
assert len(body)==1
exec(compile(ast.Module(body=body,type_ignores=[]),'historical_add_candidate_edges','exec',flags=__future__.annotations.compiler_flag),module.__dict__)
sys.modules['biohub_motion_ilp_fold']=module
sys.modules['tracksdata']=types.ModuleType('tracksdata')
import biohub_motion_submission as m
original=m.infer_video

def infer(stem,*args,**kwargs):
    global current
    current=stem;p=DEST/f'{stem}.pkl'
    if not p.exists():
        try: original(stem,*args,**kwargs)
        except Captured: pass
        else: raise RuntimeError('Original inference did not reach candidate capture')
    payload=pickle.loads(p.read_bytes())
    if payload.get('contract')!=contract:raise RuntimeError('Candidate replay contract differs; use a new experiment output directory')
    print('CANDIDATES_EXPORTED',stem,len(payload['node_rows']),len(payload['edges']),flush=True)
    # Keys required by the original notebook summary; these are explicitly not tracking results.
    return dict(dataset=stem,stem=stem,nodes=len(payload['node_rows']),candidate_nodes=len(payload['node_rows']),candidate_edges=len(payload['edges']),selected_nodes=0,selected_edges=0,stage='candidates_only',tracking_evaluated=False,elapsed_seconds=0)

m.infer_video=infer
# Installed after the original capture imports, before m.main().
from functools import wraps
import time as _time
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from two_model_shared_forward import shared_d4_forward,cpu_outputs_once
import torch

_base_capture_infer=infer
_second_network=None
_second_init_seconds=None
_second_model_dir=Path(os.environ['BIOHUB_SECOND_MODEL_DIR'])
_second_sha=os.environ['BIOHUB_SECOND_MODEL_SHA256']
if hashlib.sha256((_second_model_dir/'fold_0/checkpoint_final.pth').read_bytes()).hexdigest()!=_second_sha:
    raise RuntimeError('Second checkpoint SHA mismatch')

def ensemble_joint_forward_batch(network,data,device,amp,required_stride,motion_scale_um,
                                 detector_tta_mode='none'):
    if detector_tta_mode!='d4-detector':
        raise RuntimeError(f'Unplanned detector TTA mode {detector_tta_mode}')
    if data.ndim!=5:raise ValueError(f'Expected B,C,Z,Y,X: {data.shape}')
    original_shape=tuple(int(v) for v in data.shape[-3:])
    tensor=torch.from_numpy(np.ascontiguousarray(data,dtype=np.float32))
    tensor,_=m.pad_full_volume_to_stride(tensor,required_stride)
    tensor=tensor.to(device,non_blocking=True)
    with torch.inference_mode(),m.amp_context(device,amp):
        logit,motion=shared_d4_forward(network,_second_network,tensor)
        probability,vector=cpu_outputs_once(logit,motion,original_shape,motion_scale_um)
    return probability,vector

def ensemble_infer(stem,*args,**kwargs):
    global _second_network,_second_init_seconds
    if _second_network is None:
        started=_time.perf_counter()
        device=args[3]
        predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=False,use_mirroring=False,
            perform_everything_on_device=device.type=='cuda',device=device,
            verbose=False,allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(str(_second_model_dir),
            use_folds=(0,),checkpoint_name='checkpoint_final.pth')
        _second_network=predictor.network.to(device).eval()
        if os.environ.get('BIOHUB_TRT_ENABLE')=='1':
            from trt_backend import TensorRTNetwork
            _second_network=TensorRTNetwork(_second_network,_second_sha)
        assert m.cumulative_stride(predictor.configuration_manager)==args[4]
        _second_init_seconds=_time.perf_counter()-started
        m.joint_forward_batch=ensemble_joint_forward_batch
        print('SECOND_MODEL_READY',_second_sha,_second_init_seconds,flush=True)
    return _base_capture_infer(stem,*args,**kwargs)

m.infer_video=ensemble_infer

if __name__=='__main__':m.main()
