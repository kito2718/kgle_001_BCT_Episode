from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
import numpy as np
from parent import VideoCrops,peaks_from_rows,save
from parent_combined import predict

def score_video(scorer,stem,rows,data_dir,output,eligible_indices=None):
 indices=list(range(len(rows))) if eligible_indices is None else sorted(eligible_indices)
 assert len(set(indices))==len(indices) and all(0<=i<len(rows) for i in indices)
 builder=VideoCrops(Path(data_dir)/(stem+'.zarr'),peaks_from_rows(stem,rows));batches=[indices[i:i+32] for i in range(0,len(indices),32)]
 scored=[dict(node_index=i,**r,parent_prior=0.,seed20260817=None,seed20260818=None,inferred=False) for i,r in enumerate(rows)]
 def crop(ids):return [builder.inputs(int(rows[i]['t']),[rows[i][k] for k in 'zyx']) for i in ids]
 start=time.perf_counter()
 if batches:
  with ThreadPoolExecutor(max_workers=1) as pool:
   future=pool.submit(crop,batches[0])
   for j,ids in enumerate(batches):
    pairs=future.result()
    if j+1<len(batches):future=pool.submit(crop,batches[j+1])
    probs=predict(scorer,[r for r,h in pairs],[h for r,h in pairs]);assert np.isfinite(probs).all()
    for i,v in zip(ids,probs):scored[i]=dict(node_index=i,**rows[i],parent_prior=float(v.mean()),seed20260817=float(v[0]),seed20260818=float(v[1]),inferred=True)
 save(output,dict(stem=stem,nodes=scored,models=scorer.provenance,coverage_all_nodes=len(indices)==len(rows),coverage_all_fork_capable_nodes=True,inferred_nodes=len(indices),total_nodes=len(rows),timings_seconds=dict(total=time.perf_counter()-start),optimization='Full fork fp32 combined transfer one-batch CPU prefetch'))
