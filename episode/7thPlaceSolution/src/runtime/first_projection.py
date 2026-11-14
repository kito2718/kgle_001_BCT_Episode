"""First observed position on image-identical nonbranching chains; no labels read."""
from collections import Counter
import copy,json,time,runpy,argparse
from pathlib import Path
def groups(n,e,f):
 inc=Counter(x['target_id'] for x in e);out=Counter(x['source_id'] for x in e)
 branches={k for k in n if inc[k]>1 or out[k]>1};protect=set(branches)
 for x in e:
  if x['source_id'] in branches or x['target_id'] in branches:protect.update([x['source_id'],x['target_id']])
 nxt={};prev={};frozen=[]
 for x in e:
  u,v=x['source_id'],x['target_id'];t=int(n[u]['t'])
  if t not in f or int(n[v]['t'])!=t+1:continue
  frozen.append((u,v))
  if u in protect or v in protect:continue
  assert out[u]==1 and inc[v]==1
  nxt[u]=v;prev[v]=u
 result=[]
 for u in nxt:
  if u in prev:continue
  g=[u]
  while g[-1] in nxt:g.append(nxt[g[-1]])
  result.append(g)
 assert sum(len(g)-1 for g in result)==len(nxt)
 return result,dict(frozen_edges=len(frozen),protected_nodes=len(protect),eligible_edges=len(nxt),segments=len(result),segment_lengths=dict(Counter(map(len,result)))),frozen

def project(nodes,edges,observed_ids,frozen):
    chains,stats,_=groups(nodes,edges,set(frozen))
    result=copy.deepcopy(nodes);moved=0
    for chain in chains:
        anchors=[k for k in chain if k in observed_ids]
        if not anchors:continue
        q={axis:float(nodes[anchors[0]][axis]) for axis in ('z','y','x')}
        for k in chain:
            moved+=any(result[k][axis]!=q[axis] for axis in q)
            result[k].update(q)
    assert set(result)==set(nodes)
    assert all(result[k]['t']==nodes[k]['t'] for k in nodes)
    return result,dict(stats,moved_nodes=moved)

def image_freezes(array,transitions):
    import numpy as np
    assert len(array.shape)==4
    previous=None;previous_t=None;frozen=[]
    for t in sorted(set(transitions)):
        assert 0<=t<t+1<array.shape[0]
        left=previous if previous_t==t else np.asarray(array[t])
        right=np.asarray(array[t+1])
        if np.array_equal(left,right):frozen.append(t)
        previous=right;previous_t=t+1
    return frozen

def main():
    import zarr
    p=argparse.ArgumentParser();p.add_argument('--post-tool',required=True);p.add_argument('--post-dir',type=Path,required=True);p.add_argument('--raw-dir',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    b=runpy.run_path(a.post_tool);a.out.mkdir(parents=True,exist_ok=True);rows=[]
    for path in sorted(a.post_dir.glob('*.geff')):
        started=time.perf_counter();n,e=b['read_graph'](path);rn,re=b['read_graph'](a.raw_dir/path.name)
        ts={int(n[x['source_id']]['t']) for x in e if int(n[x['target_id']]['t'])==int(n[x['source_id']]['t'])+1}
        ar=zarr.open_group(str(a.data_dir/(path.stem+'.zarr')),mode='r')['0']
        f=image_freezes(ar,ts);nn,s=project(n,e,set(rn),f)
        b['write_graph'](a.out/path.name,nn,e)
        rows.append(dict(stem=path.stem,seconds=time.perf_counter()-started,frozen_transitions=f,nodes=len(n),edges=len(e),**s))
        print(json.dumps(rows[-1]),flush=True)
    assert rows
    (a.out/'first_projection_receipt.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
