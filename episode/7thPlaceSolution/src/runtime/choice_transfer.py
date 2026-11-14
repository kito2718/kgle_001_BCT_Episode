"""Label-blind production transfer on incumbent-fork hypotheses, not GT cohorts."""
from pathlib import Path
import os,sys,json
from parent import save
E=Path(os.environ["BIOHUB_CHOICE_OUT"])
from choice_features import features,SCALE
from choice_net import Net
import pickle,numpy as np,torch,geff,collections
from types import SimpleNamespace
from biohub_division_physical_pair_cpu import Candidate,PointClouds,PhysicalFeatureBuilder
from biohub_division_pair_inference_oof import enumerate_candidate_pairs
P=Path(os.environ['BIOHUB_CHOICE_BASE']);F=P
def main():
 models={}
 for arm in [os.environ['BIOHUB_CHOICE_ARM']]:
  cp=torch.load(Path(os.environ['BIOHUB_CHOICE_MODEL']),map_location='cpu',weights_only=False);model=Net(len(cp['features']));model.load_state_dict(cp['state']);model.eval();models[arm]=(model,cp)
 receipts=[]
 for scope in ['one']:
  for stem in (P/f'config/{scope}.txt').read_text().split():
   receipt=E/f'transfer/{stem}.receipt.json'
   if receipt.exists():receipts.append(json.loads(receipt.read_text()));continue
   data=pickle.loads((P/f'candidates/{stem}.pkl').read_bytes());rows=data['node_rows'];edges=data['edges'];prior=json.loads((P/f'parent_scores/{stem}.json').read_text())['nodes'];cloud=collections.defaultdict(list);indices=collections.defaultdict(list)
   for i,r in enumerate(rows):key=f'{stem}_t{int(r["t"]):03d}';cloud[key].append([r[a] for a in 'zyx']+[r['node_score']]);indices[key].append(i)
   cloud={k:np.asarray(v,float) for k,v in cloud.items()};vox={k:v[:,:3] for k,v in cloud.items()};points={k:v*SCALE for k,v in vox.items()};scores={k:v[:,3] for k,v in cloud.items()};builder=PhysicalFeatureBuilder(PointClouds(points,scores,{k:np.zeros(len(v),np.int8) for k,v in points.items()}),15.,6)
   g=geff.read(str(F/f'pair_final/{stem}.geff'),node_props=['t','z','y','x'],structure_validation=False)[0];pre=geff.read(str(F/f'pair_graphs/{stem}.geff'),node_props=['t','z','y','x'],structure_validation=False)[0];lookup={(int(r['t']),r['z'],r['y'],r['x']):i for i,r in enumerate(rows)}
   allx=[];meta=[];unavailable=0
   for node in g:
    if g.out_degree(node)!=2 or node not in pre:continue
    q=pre.nodes[node];i=lookup.get((int(q['t']),q['z'],q['y'],q['x']))
    if i is None:continue
    r=rows[i];t=int(r['t']);case=f'{stem}_t{t:03d}';parent_index=indices[case].index(i);c=Candidate(str(i),'detector','2',case,stem,t,3,points[case][parent_index],0,'unknown','unknown');pairs,ok=enumerate_candidate_pairs(c,prior[i]['parent_prior'],builder,vox,SimpleNamespace(parent_match_um=.25,max_pair_points=6,pair_radius_um=15.,fold=2))
    for p in pairs:
     child=[indices[f'{stem}_t{t+1:03d}'][p[k]] for k in ['first','second']]
     if not all((i,j) in edges for j in child):unavailable+=1;continue
     fs=features(p,points[case],points[f'{stem}_t{t+1:03d}'],parent_index,builder,stem,t,scores[case][parent_index]);allx.append(fs);meta.append(dict(node_index=i,children=child,parent_t=t,parent_zyx=[r[a] for a in 'zyx']))
   saturation={}
   for arm,(model,cp) in models.items():
    if allx:
     matrix=np.asarray([[f[k] for k in cp['features']] for f in allx],np.float32);z=(matrix-cp['mean'])/cp['std'];saturation[arm]=float(np.mean(np.abs(z)>10))
     with torch.no_grad():logits=model(torch.from_numpy(np.clip(z,-10,10))).flatten().numpy()
     rr=[dict(**m,logit=float(v),reward=float(-.2*np.tanh(v))) for m,v in zip(meta,logits)]
    else:rr=[];saturation[arm]=0.
    save(E/f'transfer/{arm}/{stem}.json',dict(stem=stem,rows=rr,labels_read=False,candidate_contract=data['contract'],strength=.2,policy='Add bounded pair margin only at label-blind incumbent fork parents; evaluate all valid pair alternatives there',calibrated_probability=False))
   r=dict(stem=stem,options=len(meta),parents=len({m['node_index'] for m in meta}),excluded_unavailable_edge_pairs=unavailable,feature_clip_fraction=saturation,labels_read=False);save(receipt,r);receipts.append(r);print(r,flush=True)
 save(E/'transfer/summary.json',dict(videos=receipts,scope='D4 deployment candidate transfer, GT-free; restricted to incumbent fork parents. Full fork discovery not claimed.'));(E/'TRANSFER_DONE').write_text('complete')
if __name__=='__main__':main()
