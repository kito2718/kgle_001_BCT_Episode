"""Second ten hypotheses; final-stage, saved single-model predictions only."""
import copy
from collections import defaultdict, Counter
import numpy as np
from scipy.spatial import cKDTree
SCALE=np.array([1.625,.40625,.40625]); EPS=1e-10
DESIGNS={
'r1_restore_separated':('Separated supported fragments','S5 subset: every restored raw point farther than 3.2um from retained same-frame points.'),
'r2_restore_boundary':('Temporal-boundary supported fragments','S5 subset: raw component starts at first candidate frame or ends at last candidate frame.'),
'r3_restore_weaklink':('Strong weakest-link fragments','S5 subset: minimum node confidence and minimum edge p exceed medians of corresponding minima over retained raw >=9 components.'),
'r4_restore_local_support':('Locally supported fragments','S5 subset: detector and edge medians also exceed medians among retained reference tracks overlapping in time and whose centroid is within 32um; require >=3 local reference tracks.'),
'r5_restore_motion':('Motion-consistent fragments','S5 subset: median physical acceleration <= median over retained reference components and max speed <= retained reference median max speed.'),
'p1_endpoint_extrapolation':('Low-confidence endpoint extrapolation','Whole nonbranch >=9 tracks; endpoint confidence below track lower quartile; extrapolate preceding four points. Require disagreement >2 times median internal acceleration, >0 and <=4um; fixed baseline proposals.'),
'p2_weighted_quadratic':('Confidence-weighted quadratic repair','Seven-point consecutive windows; center raw confidence below window lower quartile; weighted quadratic fit with weights max(confidence,1e-6); replace center only if displacement <=2um. Protected branch neighborhood unchanged.'),
'p3_z_alternating_jitter':('Axial alternating jitter correction','Five-point windows, four z increments strictly alternating signs; median absolute z increment >2 times median transverse step. Replace center z with five-point quadratic fit, max correction one axial voxel; x/y unchanged.'),
't1_unsupported_tail_trim':('Weak unsupported tail trimming','Whole nonbranch >=10 tracks; remove at most one point at each endpoint when raw confidence below track lower quartile and endpoint edge p below track lower quartile and endpoint speed >2 times median internal speed; retain >=9 nodes.'),
't2_parallel_duplicate_drop':('Weaker parallel track suppression','Whole nonbranch 9..30 tracks; remove track only when >=90% points within 1.625um of ONE other track, that track covers its full time range, and both median raw node and native association confidence are strictly lower. Preserve support track; branch-connected tracks excluded.')}
HYPOTHESES={
'r1_restore_separated':'Spatial separation may avoid redundant fragments that increase predicted-node penalty or compete in matching.',
'r2_restore_boundary':'Truncation at the acquisition boundary may explain short length despite a real trajectory.',
'r3_restore_weaklink':'Median support can conceal a weak node or association that marks an unreliable fragment.',
'r4_restore_local_support':'Local confidence references may distinguish useful fragments better than a video-wide median in heterogeneous density or image quality.',
'r5_restore_motion':'Strong confidence plus locally typical motion may reject noisy short fragments while retaining genuine trajectories.',
'p1_endpoint_extrapolation':'A low-confidence endpoint lacks symmetric smoothing context; interior motion can constrain its position.',
'p2_weighted_quadratic':'A curved local trajectory fitted using detector confidence may preserve acceleration while correcting uncertain points.',
'p3_z_alternating_jitter':'Coarse axial sampling may produce oscillation larger than transverse motion that warrants axis-specific correction.',
 't1_unsupported_tail_trim':'A weak fast endpoint may be a false track extension rather than part of the trajectory.',
 't2_parallel_duplicate_drop':'A weaker parallel track closely covered by a stronger track may be duplicate detections.'}
ARM_SPECS={k:dict(title=v[0],hypothesis=HYPOTHESES[k],method=v[1],stage='final') for k,v in DESIGNS.items()}
def point(n):return np.array([n[x] for x in ('z','y','x')],float)*SCALE
def put(n,p):
 for k,v in zip(('z','y','x'),p/SCALE):n[k]=float(v)
def topology(n,e):
 inc=defaultdict(list);out=defaultdict(list)
 for q in e:out[q['source_id']].append(q['target_id']);inc[q['target_id']].append(q['source_id'])
 branch={k for k in n if len(inc[k])>1 or len(out[k])>1};protect=set(branch)
 for k in branch:protect.update(inc[k]);protect.update(out[k])
 return inc,out,protect
def chains(n,e):
 inc,out,protect=topology(n,e);cc=[]
 for k in sorted(n):
  if k in protect:continue
  if len(inc[k])==1 and inc[k][0] not in protect and len(out[inc[k][0]])==1:continue
  ch=[k]
  while len(out[ch[-1]])==1:
   v=out[ch[-1]][0]
   if v in protect or len(inc[v])!=1:break
   ch.append(v)
  cc.append(ch)
 return cc,inc,out,protect
def support(ch,ctx):
 if any(k not in ctx['raw'] or k>=len(ctx['rows']) or 'node_score' not in ctx['rows'][k] for k in ch):return None
 pairs=list(zip(ch,ch[1:]))
 if not pairs or any(q not in ctx['candidates'] for q in pairs):return None
 ns=np.array([float(ctx['rows'][k]['node_score']) for k in ch]);ps=np.array([float(ctx['candidates'][q][0]) for q in pairs])
 if not np.isfinite(ns).all() or not np.isfinite(ps).all():return None
 return ns,ps
def motion(ch,n):
 p=np.array([point(n[k]) for k in ch]);t=np.array([n[k]['t'] for k in ch]);v=np.diff(p,axis=0)/np.diff(t)[:,None]
 return float(np.median(np.linalg.norm(np.diff(v,axis=0),axis=1))),float(np.max(np.linalg.norm(v,axis=1)))
def apply(arm,ctx):
 if arm not in ARM_SPECS:raise KeyError(arm)
 n=copy.deepcopy(ctx['final']);e=copy.deepcopy(ctx['fe']);stats=Counter();details=[]
 cc,inc,out,protect=chains(n,e);raw=ctx['raw'];re=ctx['re'];original=set(n)
 if arm.startswith('r'):
  rc,ri,ro,_=chains(raw,re);refs=[]
  for ch in rc:
   q=support(ch,ctx)
   if len(ch)>=9 and all(k in n for k in ch) and q is not None:refs.append((ch,q))
  stats['reference_components']=len(refs)
  if refs:
   med=np.median([[np.median(q[0]),np.median(q[1])] for ch,q in refs],axis=0);weak=np.median([[min(q[0]),min(q[1])] for ch,q in refs],axis=0);mot=np.median([motion(ch,raw) for ch,q in refs],axis=0)
   frames=defaultdict(list)
   if arm=='r1_restore_separated':
    for k,q in n.items():frames[int(q['t'])].append(point(q))
   trees={t:cKDTree(p) for t,p in frames.items()};edge_lookup={(q['source_id'],q['target_id']):q for q in re}
   times=[int(q['t']) for q in ctx['rows']];bounds=(min(times),max(times)) if times else (None,None)
   refcenters=np.array([np.mean([point(raw[k]) for k in ch],axis=0) for ch,q in refs]);rtree=cKDTree(refcenters)
   for ch in rc:
    if not 5<=len(ch)<=8 or any(k in n for k in ch) or ri[ch[0]] or ro[ch[-1]]:continue
    q=support(ch,ctx);stats['eligible_components']+=1
    if q is None or np.any(np.array([np.median(q[0]),np.median(q[1])])<med):continue
    stats['s5_selected_components']+=1;accept=False
    if arm=='r1_restore_separated':accept=all(int(raw[k]['t']) not in trees or trees[int(raw[k]['t'])].query(point(raw[k]))[0]>3.2 for k in ch)
    elif arm=='r2_restore_boundary':accept=raw[ch[0]]['t']==bounds[0] or raw[ch[-1]]['t']==bounds[1]
    elif arm=='r3_restore_weaklink':accept=min(q[0])>=weak[0] and min(q[1])>=weak[1]
    elif arm=='r4_restore_local_support':
     near=rtree.query_ball_point(np.mean([point(raw[k]) for k in ch],axis=0),32.);near=[i for i in near if raw[refs[i][0][0]]['t']<=raw[ch[-1]]['t'] and raw[refs[i][0][-1]]['t']>=raw[ch[0]]['t']]
     if len(near)>=3:
      local=np.median([[np.median(refs[i][1][0]),np.median(refs[i][1][1])] for i in near],axis=0);accept=np.median(q[0])>=local[0] and np.median(q[1])>=local[1]
     else:stats['insufficient_local_reference']+=1
    else:accept=bool(np.all(np.asarray(motion(ch,raw))<=mot+EPS))
    if not accept:stats['additional_gate_rejected']+=1;continue
    nn={k:copy.deepcopy(raw[k]) for k in ch};ee=[copy.deepcopy(edge_lookup[u,v]) for u,v in zip(ch,ch[1:])];nn=ctx['b'].linefit(nn,ee,defaultdict(int));n.update(nn);e.extend(ee);stats['restored_components']+=1;stats['restored_nodes']+=len(nn);details.append({'ids':ch})
 elif arm.startswith('p'):
  updates={}
  for ch in cc:
   if len(ch)<7 or any(n[v]['t']-n[u]['t']!=1 for u,v in zip(ch,ch[1:])):continue
   p=np.array([point(n[k]) for k in ch]);q=support(ch,ctx)
   if q is None:stats['missing_native_support_chains']+=1
   if arm=='p1_endpoint_extrapolation':
    if len(ch)<9 or inc[ch[0]] or out[ch[-1]] or q is None:continue
    scale=float(np.median(np.linalg.norm(np.diff(p,n=2,axis=0),axis=1)));threshold=np.quantile(q[0],.25)
    for reverse in [False,True]:
     ids=ch[::-1] if reverse else ch;pp=p[::-1] if reverse else p;conf=q[0][::-1] if reverse else q[0]
     if conf[0]>=threshold:continue
     fit=np.linalg.lstsq(np.c_[np.ones(4),np.arange(1,5)],pp[1:5],rcond=None)[0][0];d=float(np.linalg.norm(fit-pp[0]));stats['eligible_endpoints']+=1
     if d>max(EPS,2*scale) and d<=4:updates[ids[0]]=fit
   elif arm=='p2_weighted_quadratic':
    if q is None:continue
    for i in range(3,len(ch)-3):
     w=q[0][i-3:i+4]
     if w[3]>=np.quantile(w,.25):continue
     tt=np.arange(-3,4.);X=np.c_[np.ones(7),tt,tt*tt];sw=np.sqrt(np.maximum(w,1e-6));fit=np.linalg.lstsq(X*sw[:,None],p[i-3:i+4]*sw[:,None],rcond=None)[0][0];d=np.linalg.norm(fit-p[i]);stats['eligible_centers']+=1
     if EPS<d<=2:updates[ch[i]]=fit
   else:
    for i in range(2,len(ch)-2):
     pp=p[i-2:i+3];delta=np.diff(pp,axis=0);zz=delta[:,0]
     if not np.all(zz[:-1]*zz[1:]<0) or np.median(abs(zz))<=2*np.median(np.linalg.norm(delta[:,1:],axis=1)):continue
     tt=np.arange(-2,3.);z=np.linalg.lstsq(np.c_[np.ones(5),tt,tt*tt],pp[:,0],rcond=None)[0][0];stats['eligible_centers']+=1
     if EPS<abs(z-p[i,0])<=SCALE[0]:v=p[i].copy();v[0]=z;updates[ch[i]]=v
  for k,p in updates.items():put(n[k],p)
  stats['updated_nodes']=len(updates)
 else:
  remove=set();whole=[ch for ch in cc if not inc[ch[0]] and not out[ch[-1]]]
  if arm=='t1_unsupported_tail_trim':
   for ch in whole:
    q=support(ch,ctx)
    if q is None:stats['missing_native_support_chains']+=1
    if len(ch)<10 or q is None:continue
    p=np.array([point(n[k]) for k in ch]);dt=np.diff([n[k]['t'] for k in ch]);speed=np.linalg.norm(np.diff(p,axis=0),axis=1)/dt;ref=np.median(speed[1:-1]);ns=np.quantile(q[0],.25);ps=np.quantile(q[1],.25)
    for j,edge in [(0,0),(-1,-1)]:
     if len(ch)-len(set(ch)&remove)<=9:break
     if q[0][j]<ns and q[1][edge]<ps and speed[edge]>max(EPS,2*ref):remove.add(ch[j]);stats['trimmed_endpoints']+=1
  else:
   sup={i:support(ch,ctx) for i,ch in enumerate(whole)};stats['missing_native_support_chains']=sum(q is None for q in sup.values());node_component={k:i for i,ch in enumerate(whole) for k in ch};frames=defaultdict(list)
   for k,v in n.items():
    if k in node_component:frames[int(v['t'])].append(k)
   trees={t:(ids,cKDTree([point(n[k]) for k in ids])) for t,ids in frames.items()};protected_components=set()
   order=sorted(range(len(whole)),key=lambda i:(np.median(sup[i][0]) if sup[i] is not None else float('inf'),i))
   for i in order:
    ch=whole[i];q=sup[i]
    if not 9<=len(ch)<=30 or q is None or i in protected_components:continue
    votes=Counter()
    for k in ch:
     ids,tree=trees[int(n[k]['t'])];near=tree.query_ball_point(point(n[k]),1.625);votes.update(set(node_component[ids[j]] for j in near if node_component[ids[j]]!=i))
    for j,count in sorted(votes.items(),key=lambda x:(-x[1],x[0])):
     other=whole[j];s=sup[j]
     if count/len(ch)<.9 or s is None or set(other)&remove or n[other[0]]['t']>n[ch[0]]['t'] or n[other[-1]]['t']<n[ch[-1]]['t']:continue
     if np.median(q[0])<np.median(s[0]) and np.median(q[1])<np.median(s[1]):remove.update(ch);protected_components.add(j);stats['removed_components']+=1;details.append({'removed':ch,'support_component_ids':other});break
  n={k:v for k,v in n.items() if k not in remove};e=[q for q in e if q['source_id'] not in remove and q['target_id'] not in remove];stats['removed_nodes']=len(remove)
 keys=[(q['source_id'],q['target_id']) for q in e];assert len(keys)==len(set(keys));assert all(u in n and v in n and n[v]['t']>n[u]['t'] for u,v in keys);assert all(np.isfinite(point(q)).all() for q in n.values())
 protected_edges={(q['source_id'],q['target_id']) for q in ctx['fe'] if q['source_id'] in protect or q['target_id'] in protect};assert protected_edges<=set(keys)
 stats.update(final_nodes=len(n),final_edges=len(e));return n,e,dict(stats= dict(stats),components=details)
