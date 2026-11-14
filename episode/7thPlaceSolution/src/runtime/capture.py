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
if __name__=='__main__':m.main()
