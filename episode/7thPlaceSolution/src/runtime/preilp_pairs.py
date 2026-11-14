"""Prediction-only, pre-ILP Division pair proposals for a two-model candidate."""
import collections,itertools,json,os,pickle,sys
from pathlib import Path
import numpy as np
R=Path(os.environ['BIOHUB_ROOT']);E=Path(os.environ['BIOHUB_CHOICE_OUT'])
def main():
 for stem in (E/'config/one.txt').read_text().split():
  data=pickle.loads((E/f'candidates/{stem}.pkl').read_bytes());rows=data['node_rows'];edges=data['edges']
  prior=json.loads((E/f'parent_scores/{stem}.json').read_text())['nodes']
  outgoing=collections.defaultdict(list);incoming=collections.defaultdict(list);by_time=collections.defaultdict(list)
  for i,r in enumerate(rows):by_time[int(r['t'])].append(i)
  for (source,target),(prob,distance,*_) in edges.items():
   if int(rows[target]['t'])!=int(rows[source]['t'])+1:continue
   outgoing[source].append((float(prob),float(distance),target));incoming[target].append((float(prob),source))
  features=[];parents=[]
  for parent,ee in sorted(outgoing.items()):
   if len(ee)<2:continue
   ee=sorted(ee,reverse=True);a,b=ee[:2]
   rivals=[max((p for p,source in incoming[target] if source!=parent),default=0.) for _,_,target in ee[:2]]
   features.append([float(prior[parent]['parent_prior']),float(rows[parent]['node_score']),a[0],b[0],a[0]-b[0],a[0]+b[0],b[0],a[1]+b[1],len(ee),max(rivals),int(rows[parent]['t'])/100.])
   parents.append(parent)
  model=json.loads((R/'models/parent_surrogate_all40.json').read_text())
  x=np.asarray(features,np.float64)
  if len(x):
   z=(x-np.asarray(model['mean']))/np.asarray(model['scale'])
   score=z@np.asarray(model['coef'])+model['intercept']
   ranked=[parents[i] for i in np.argsort(-score,kind='stable')[:1536]]
  else:ranked=[]
  keys={(p,tuple(sorted(pair))) for p in ranked for pair in itertools.combinations([target for _,_,target in outgoing[p]],2)}
  fusion={};pair_data=json.loads((E/f'pair_scores_min01/{stem}.json').read_text())
  for r in pair_data['rows']:
   if int(r['pair_rank'])>3:continue
   p=int(r['node_index']);target=by_time[int(rows[p]['t'])+1]
   daughters=tuple(sorted((target[int(r['first'])],target[int(r['second'])])))
   if all(d in {e[2] for e in outgoing[p]} for d in daughters):
    fusion[(p,daughters)]=float(r['pair_score']);keys.add((p,daughters))
  records=[dict(node_index=p,children=list(ds),parent_t=int(rows[p]['t']),parent_zyx=[rows[p][a] for a in 'zyx']) for p,ds in sorted(keys)]
  dest=E/f'transfer/preilp/{stem}.json';dest.parent.mkdir(parents=True,exist_ok=True)
  dest.write_text(json.dumps(dict(stem=stem,candidate_contract=data['contract'],rows=records,fusion=[dict(node_index=p,children=list(ds),pair_score=value) for (p,ds),value in sorted(fusion.items())],labels_read=False,policy='all40 surrogate top1536 union pre-ILP fusion top3')))
  print(json.dumps(dict(stem=stem,parents=len(ranked),pairs=len(records),fusion=len(fusion))),flush=True)
if __name__=='__main__':main()
