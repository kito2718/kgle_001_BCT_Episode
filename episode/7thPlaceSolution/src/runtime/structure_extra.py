"""Five independent structural/position hypotheses; no GT or new detector calls."""
import copy
from collections import Counter,defaultdict
import numpy as np
import new_arms
SCALE=np.array([1.625,.40625,.40625])
DESIGNS={
'x6_restore_connect':'Connect restored endpoints to existing free endpoints using native candidates',
'x7_cut_reconnect':'Replace one internal native edge with a stronger endpoint connection',
'x8_restore_raw_position':'Keep restored fragment coordinates raw before First',
'x9_first_supported_anchor':'Choose earliest above-median raw anchor when frozen first anchor is weak',
'x10_unique_freeze_regression':'Use one earliest observed point per frozen group in local regression before First'}
ARM_SPECS={k:dict(title=v,hypothesis=v,stage='final',first_applied=k in ['x9_first_supported_anchor','x10_unique_freeze_regression']) for k,v in DESIGNS.items()}
def pos(n):return np.array([n[a] for a in ['z','y','x']],float)*SCALE
def put(n,p):n.update({a:float(v) for a,v in zip(['z','y','x'],p/SCALE)})
def base(ctx):
 if 'restore_base' in ctx:
  n,e,r=ctx['restore_base'];return copy.deepcopy(n),copy.deepcopy(e),copy.deepcopy(r)
 if ctx.get('first_only'):return copy.deepcopy(ctx['final']),copy.deepcopy(ctx['fe']),{}
 return new_arms.apply('r3_restore_weaklink',ctx)
def topology(n,e):return new_arms.topology(n,e)
def freeze(ctx):
 f=ctx['freeze'];return set(f[ctx['stem']] if isinstance(f,dict) else f)
def apply(arm,ctx):
 if arm not in ARM_SPECS:raise KeyError(arm)
 n,e,r=base(ctx);before=copy.deepcopy(n);raw=ctx['raw'];cand=ctx['candidates'];inc,out,protected=topology(n,e);stats=Counter();details=[]
 native=set(n)&set(raw);added=set(n)-set(ctx['final'])
 def conf(k):return float(ctx['rows'][k]['node_score'])
 def residual(u,v):
  velocity=pos(n[v])-pos(n[u]);vals=[]
  if len(inc[u])==1:
   p=inc[u][0]
   if n[u]['t']-n[p]['t']==1:vals.append(float(np.linalg.norm(velocity-(pos(n[u])-pos(n[p])))))
  if len(out[v])==1:
   q=out[v][0]
   if n[q]['t']-n[v]['t']==1:vals.append(float(np.linalg.norm((pos(n[q])-pos(n[v]))-velocity)))
  return float(np.mean(vals)) if vals else None
 if arm=='x6_restore_connect':
  probs=[float(cand[u,v][0]) for u in native for v in out[u] if (u,v) in cand and v in native and u not in added and v not in added]
  threshold=float(np.median(probs)) if probs else float('inf');proposals=[]
  for (u,v),value in cand.items():
   if u not in native or v not in native or u in protected or v in protected or out[u] or inc[v]:continue
   if (u in added)==(v in added) or n[v]['t']-n[u]['t']!=1:continue
   stats['native_endpoint_candidates']+=1;p=float(value[0]);distance=float(np.linalg.norm(pos(n[v])-pos(n[u])))
   if p>=threshold and distance<=4.5:proposals.append((-p,distance,u,v))
  used=set()
  for _,d,u,v in sorted(proposals):
   if u in used or v in used:continue
   used.update([u,v]);e.append(dict(source_id=u,target_id=v));details.append(dict(source=u,target=v,p=float(cand[u,v][0]),distance_um=d))
  stats['connections']=len(details)
 elif arm=='x7_cut_reconnect':
  edgekeys={(q['source_id'],q['target_id']) for q in e};proposals=[]
  # One endpoint free, other occupied by exactly one eligible original edge.
  for (u,v),val in cand.items():
   if u not in native or v not in native or u in protected or v in protected or n[v]['t']-n[u]['t']!=1 or (u,v) in edgekeys:continue
   old=None
   if not inc[v] and len(out[u])==1:old=(u,out[u][0])
   elif not out[u] and len(inc[v])==1:old=(inc[v][0],v)
   if old is None or any(k in protected or k not in native for k in old) or old not in cand:continue
   # Internal old edge: support on both its outer sides, so not merely swapping tails.
   a,b=old
   if not inc[a] or not out[b] or n[b]['t']-n[a]['t']!=1:continue
   stats['eligible_alternatives']+=1;gain=float(val[0])-float(cand[old][0]);newres=residual(u,v);oldres=residual(*old)
   if gain>1e-10 and newres is not None and oldres is not None and newres+1e-10<oldres:proposals.append((-gain,newres,old,u,v))
  used=set();remove=set()
  for gain,res,old,u,v in sorted(proposals):
   ids=set(old)|{u,v}
   if ids&used:continue
   used.update(ids);remove.add(old);details.append(dict(old=list(old),new=[u,v],p_gain=-gain,motion_residual_um=res))
  e=[q for q in e if (q['source_id'],q['target_id']) not in remove]+[dict(source_id=q['new'][0],target_id=q['new'][1]) for q in details];stats['reconnections']=len(details)
 elif arm=='x8_restore_raw_position':
  for k in sorted(added):
   assert k in raw
   for a in ['z','y','x']:n[k][a]=raw[k][a]
  stats['restored_raw_nodes']=len(added)
 else:
  p=ctx['first_projection'];groups,_,_=p.groups(n,e,freeze(ctx))
  if arm=='x10_unique_freeze_regression':
   source={k:copy.deepcopy(ctx['gap'][k] if k in ctx['gap'] else raw[k]) for k in n}
   representative={}
   for g in groups:
    observed=[k for k in g if k in raw]
    if observed:
     for k in g:representative[k]=observed[0]
   neighbourhoods=ctx['b'].trajectory_neighbourhoods(n,e)
   for k,hood in neighbourhoods.items():
    if k in protected:continue
    ids=[j for _,j in hood]
    # Only altered repeated-image windows; preserve ordinary baseline fitting.
    if not any(j in representative for j in ids):continue
    unique=[];seen=set()
    for j in ids:
     rep=representative.get(j,j)
     if rep not in seen:seen.add(rep);unique.append(rep)
    if len(unique)==len(ids) and unique==ids:continue
    stats['collapsed_windows']+=1
    if len(unique)<3:stats['insufficient_unique_support']+=1;continue
    dt=np.array([source[j]['t']-source[k]['t'] for j in unique],float);coords=np.array([pos(source[j]) for j in unique]);fit=np.linalg.lstsq(np.c_[dt,np.ones(len(dt))],coords,rcond=None)[0][1];origin=pos(source[k]);delta=float(ctx['b'].LINEFIT_WEIGHT)*(fit-origin);norm=np.linalg.norm(delta)
    if norm>2:delta*=2/norm
    put(n[k],origin+delta);stats['regression_updates']+=1
  fixed=copy.deepcopy(n)
  for g in groups:
   anchors=[k for k in g if k in raw]
   if not anchors:continue
   anchor=anchors[0]
   if arm=='x9_first_supported_anchor':
    scores=np.array([conf(k) for k in anchors]);median=float(np.median(scores))
    if scores[0]<median:
     anchor=next(k for k in anchors if conf(k)>=median);stats['anchor_changes']+=1;details.append(dict(first=anchors[0],anchor=anchor,nodes=g))
   for k in g:
    for a in ['z','y','x']:fixed[k][a]=n[anchor][a]
  n=fixed
 stats['moved_nodes']=sum(any(n[k][a]!=before[k][a] for a in ['z','y','x']) for k in n)
 assert set(n)==set(before);keys=[(q['source_id'],q['target_id']) for q in e];assert len(keys)==len(set(keys));assert all(n[v]['t']>n[u]['t'] for u,v in keys)
 for q in ctx['fe']:
  if q['source_id'] in protected or q['target_id'] in protected:assert (q['source_id'],q['target_id']) in keys
 return n,e,dict(restoration=r,stats=dict(stats),changes=details)
