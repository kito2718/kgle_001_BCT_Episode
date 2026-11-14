from pathlib import Path
import sys,json,pickle,collections,importlib.util,os
import numpy as np,torch,lightgbm as lgb
W=Path(__file__).resolve().parent;sys.path[:0]=[str(W/'legacy'),str(W)]
from parent import ParentScorer,VideoCrops,peaks_from_rows,make_image,save,SPACING_ZYX
from choice_features import features as original_features
from choice_net import Net
from portable_forest import PortableForest
from biohub_division_physical_pair_cpu import Candidate,PointClouds,PhysicalFeatureBuilder
from biohub_division_pair_inference_oof import enumerate_candidate_pairs
from types import SimpleNamespace
sp=importlib.util.spec_from_file_location('addition_features',W/'addition_features.py');af=importlib.util.module_from_spec(sp);sp.loader.exec_module(af)
R=Path(os.environ['BIOHUB_ROOT']);E=Path(os.environ['BIOHUB_CHOICE_OUT'])
def main():
 scorer=ParentScorer('cuda');scorer.models=scorer.models[:1];model=scorer.models[0]
 cp=torch.load(R/'models/nn_h.pt',map_location='cpu',weights_only=False);net=Net(len(cp['mean']));net.load_state_dict(cp['state']);net.eval();forest_meta=json.loads((R/'models/extra_trees_fold2.json').read_text());forest=PortableForest(R/'models/extra_trees_fold2.npz')
 old=torch.load(Path(os.environ['BIOHUB_CHOICE_MODEL']),map_location='cpu',weights_only=False);names=old['features'];assert cp['features']==names+['th_logit','oof_extratrees_pair_score'];assert len(forest_meta['features'])==73
 treepath=R/'models/lgb_hl';tree=lgb.Booster(model_file=str(treepath/'model.txt'));norm=json.loads((treepath/'normalization.json').read_text());mu=np.asarray(norm['mean']);sd=np.asarray(norm['std']);assert len(mu)==93
 receipts=[]
 for v in [dict(stem=stem,base=str(E),source=str(E)) for stem in (E/'config/one.txt').read_text().split()]:
  stem=v['stem'];base=Path(v['base']);source=Path(v['source']);done=E/f'new_division_checks/{stem}.json'
  if done.exists():receipts.append(json.loads(done.read_text()));continue
  data=pickle.loads((base/f'candidates/{stem}.pkl').read_bytes());rows=data['node_rows'];edges=data['edges'];oldterms=json.loads((source/f'transfer/preilp/{stem}.json').read_text());assert oldterms['candidate_contract']==data['contract'];groups=collections.defaultdict(list)
  for x in oldterms['rows']:groups[x['node_index']].append(x)
  clouds=collections.defaultdict(list);indices=collections.defaultdict(list)
  for i,r in enumerate(rows):key=f'{stem}_t{int(r["t"]):03d}';clouds[key].append([r[a] for a in 'zyx']+[r['node_score']]);indices[key].append(i)
  cloud={k:np.asarray(x) for k,x in clouds.items()};vox={k:x[:,:3] for k,x in cloud.items()};points={k:x*SPACING_ZYX for k,x in vox.items()};scores={k:x[:,3] for k,x in cloud.items()};builder=PhysicalFeatureBuilder(PointClouds(points,scores,{k:np.zeros(len(x),np.int8) for k,x in points.items()}),15.,6)
  crops=VideoCrops(Path(os.environ['BIOHUB_DATA_DIR'])/f'{stem}.zarr',peaks_from_rows(stem,rows));out={'nn_h':[],'lgb_hl':[]};audit=[]
  for i,terms in sorted(groups.items(),key=lambda item:rows[item[0]]['t']):
   r=rows[i];t=int(r['t']);center=np.array([r[a] for a in 'zyx']);case=f'{stem}_t{t:03d}';pi=indices[case].index(i);raw,heat=crops.inputs(t,center);im=make_image(raw,heat,np.zeros(3),np.zeros(3),scorer.mean,scorer.std)[None].cuda()
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=os.environ.get('BIOHUB_TH_AUX_AMP','fp32')=='bf16'):pred=model(im)
   vals={k:x.float().cpu().numpy()[0].tolist() for k,x in pred.items()};h=float(vals['binary']);flow=crops.flow.estimate(case,f'{stem}_t{t+1:03d}',center*SPACING_ZYX,'source')[1]
   c=Candidate(str(i),'detector','2',case,stem,t,3,points[case][pi],0,'unknown','unknown');pairs,_=enumerate_candidate_pairs(c,0.,builder,vox,SimpleNamespace(parent_match_um=.25,max_pair_points=6,pair_radius_um=15.,fold=2));lookup={tuple(sorted([indices[f'{stem}_t{t+1:03d}'][p[k]] for k in ['first','second']])):p for p in pairs}
   fs=[];opts=[]
   for term in terms:
    key=tuple(sorted(term['children']))
    if key not in lookup:continue
    p=lookup[key];fs.append(original_features(p,points[case],points[f'{stem}_t{t+1:03d}'],pi,builder,stem,t,scores[case][pi]));opts.append(dict(daughter1_zyx=[rows[term['children'][0]][a] for a in 'zyx'],daughter2_zyx=[rows[term['children'][1]][a] for a in 'zyx']))
   if not fs:continue
   terms=[term for term in terms if tuple(sorted(term['children'])) in lookup]
   x=np.array([[f[k] for k in names] for f in fs],np.float32);q=dict(outputs=vals,next_flow_um=np.asarray(flow).tolist());ev=dict(parent_zyx=center.tolist(),options=opts);extra=af.extra(ev,q);pair_x=np.asarray([[f[key] for key in forest_meta['features']] for f in fs],np.float32);pair_probability=forest.predict_proba(pair_x)[:,1];xx=np.column_stack([x,np.full(len(x),h,dtype=np.float32),pair_probability])
   with torch.no_grad():nnscore=net(torch.tensor(np.clip((xx-cp['mean'])/cp['std'],-10,10),dtype=torch.float32)).flatten().numpy()
   z=np.clip((np.column_stack([x,extra])-mu)/sd,-10,10);null=np.r_[z.mean(0),1.][None];allz=np.vstack([null,np.column_stack([z,np.zeros(len(z))])]);ss=tree.predict(allz,raw_score=True);delta=ss[1:]-ss[0]
   assert np.isfinite(nnscore).all() and np.isfinite(delta).all()
   for arm,sscore in [('nn_h',nnscore),('lgb_hl',delta)]:
    for term,s in zip(terms,sscore):out[arm].append(dict(node_index=i,children=term['children'],parent_t=t,parent_zyx=center.tolist(),logit=float(s),reward=float(-.2*np.tanh(s))))
   audit.append(dict(node_index=i,th_logit=h,options=len(terms),null_lgb_score=float(ss[0])))
  fusion={(int(r['node_index']),tuple(sorted(r['children']))):float(r['pair_score']) for r in oldterms['fusion']}
  th={int(r['node_index']):float(r['th_logit']) for r in audit}
  bridged=0
  for term in out['nn_h']:
   key=(int(term['node_index']),tuple(sorted(term['children'])))
   if th[term['node_index']]>=4. and fusion.get(key,-1.)>=.8 and term['logit']<0.:term['logit']=0.;term['reward']=0.;bridged+=1
  for arm,rr in out.items():save(E/f'transfer/{arm}/{stem}.json',dict(stem=stem,rows=rr,labels_read=False,candidate_contract=data['contract'],strength=.2,policy='Pre-ILP prospective valid pairs; NN-H TH/fusion bridge, all40 surrogate; separate research arm.'))
  receipt=dict(stem=stem,parents=len(groups),options=len(oldterms['rows']),coverage_equal=all(len(rr)==len(out['nn_h']) for rr in out.values()),preilp_options=len(oldterms['rows']),physically_valid_options=len(out['nn_h']),bridged=bridged,th_seed=20260817,precision=os.environ.get('BIOHUB_TH_AUX_AMP','fp32'),research_precision='bf16; target numerical parity pending',labels_read=False);assert receipt['coverage_equal'];save(done,receipt);save(E/f'outputs/th/{stem}.json',audit);receipts.append(receipt);print(receipt,flush=True)
 save(E/'checks/transfer_summary.json',receipts);(E/'TRANSFER_DONE').touch()
if __name__=='__main__':main()
