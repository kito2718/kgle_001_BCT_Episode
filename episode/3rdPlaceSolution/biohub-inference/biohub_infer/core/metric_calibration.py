"""Cache-only features and weighted logistic count calibration.

No GT, dataset name, absolute position, or legacy fitted p_cal enters features.
Heatmap confidence is a regression value, not treated as a probability.
"""
from itertools import combinations

import numpy as np
from scipy.special import expit, logit

SCALE = np.array([1.625,.40625,.40625])
EDGE_FEATURES = ['distance','distance2','log_distance','p','logit_p','rev','logit_rev',
                 'sym','logit_sym','rank','source_margin','target_margin',
                 'source_conf','target_conf','log_source_conf','log_target_conf',
                 'incoming_support','outgoing_support','r2','parent_div_score']
DIV_FEATURES = ['parent_logit','cosine','midpoint','separation','near_distance','far_distance',
                'low_p','high_p','low_rev','low_sym','parent_conf','daughter_conf',
                'incoming_support','daughter_support']
# the clip's own detection density, log(nodes per frame) of the fixed
# graph, appended as one column to both calibrators. Per clip, no GT, available at inference.


def bounded_logit(p):
    return logit(np.clip(p,1e-5,1-1e-5))


def candidate_features(graph, confidence, division, *, radius_um=12.):
    nodes=graph['nodes'];ci=graph['cand_idx'];cd=graph['cand_dist']
    n,k=ci.shape
    if np.asarray(confidence).shape!=(n,) or np.asarray(division).shape!=(n,):
        raise ValueError('unaligned confidence or division scores')
    if not np.isfinite(confidence).all() or np.any(confidence<0):
        raise ValueError('invalid confidence')
    valid=(ci>=0)&np.isfinite(cd)&(cd<=radius_um)
    src,slot=np.nonzero(valid);dst=ci[src,slot]
    if np.any(dst>=n) or np.any(nodes[dst,0]-nodes[src,0]!=1):
        raise ValueError('invalid temporal candidate')
    p=graph['cand_prob'][src,slot].astype(float)
    rev=graph['cand_rev'][src,slot].astype(float)
    sym=graph['cand_sym'][src,slot].astype(float)
    d=cd[src,slot].astype(float)
    outmax=np.zeros(n);inmax=np.zeros(n)
    np.maximum.at(outmax,src,p);np.maximum.at(inmax,dst,p)
    r2=np.zeros(len(src))
    for col in range(graph['r2_idx'].shape[1]):
        hit=graph['r2_idx'][src,col]==dst
        r2[hit]=graph['r2_prob'][src[hit],col]
    rank=graph['cand_rank'][src,slot] if 'cand_rank' in graph else slot/max(k-1,1)
    x=np.column_stack([d,d*d/10,np.log1p(d),p,bounded_logit(p),rev,bounded_logit(rev),
        sym,bounded_logit(sym),rank,p-outmax[src],p-inmax[dst],
        confidence[src],confidence[dst],np.log1p(confidence[src]),np.log1p(confidence[dst]),
        inmax[src],outmax[dst],r2,division[src]]).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite matcher features')
    return src,dst,slot,x,inmax,outmax


def division_features(graph,confidence,division,src,dst,slot,inmax,outmax):
    """All pairs among cached successors for plausible parents, no fixed wiring.

    Candidate floor 0.5 is a computational restriction, not an acceptance rule.
    Drift uses matcher top choices as evidence, never the old selected edges.
    """
    nodes=graph['nodes'];n=len(nodes);xyz=nodes[:,1:]*SCALE
    by_parent=[[] for _ in range(n)]
    for arc,i in enumerate(src):by_parent[i].append(arc)
    drift={}
    for t in np.unique(nodes[:,0]):
        sources=np.flatnonzero((nodes[:,0]==t)&(outmax>=.5))
        steps=[]
        for i in sources:
            arcs=by_parent[i]
            if arcs:
                arc=max(arcs,key=lambda a:float(graph['cand_prob'][i,slot[a]]))
                steps.append(xyz[dst[arc]]-xyz[i])
        drift[int(t)]=np.median(steps,axis=0) if len(steps)>=5 else np.zeros(3)
    parents=[];daughters=[];arc_pairs=[];features=[]
    for i in np.flatnonzero((division>=.5)&(inmax>0)):
        for a,b in combinations(by_parent[i],2):
            j,l=int(dst[a]),int(dst[b])
            if min(outmax[j],outmax[l])<=0:continue
            v1=xyz[j]-xyz[i]-drift[int(nodes[i,0])]
            v2=xyz[l]-xyz[i]-drift[int(nodes[i,0])]
            d1,d2=np.linalg.norm(v1),np.linalg.norm(v2)
            if min(d1,d2)<1e-6:continue
            cosine=float(v1@v2/(d1*d2))
            # Broad geometry universe; the classifier chooses the operating point.
            if cosine>.5:continue
            ss=[slot[a],slot[b]]
            p=graph['cand_prob'][i,ss];rev=graph['cand_rev'][i,ss];sym=graph['cand_sym'][i,ss]
            features.append([bounded_logit(division[i]),cosine,np.linalg.norm((v1+v2)/2),
                             np.linalg.norm(v1-v2),min(d1,d2),max(d1,d2),min(p),max(p),
                             min(rev),min(sym),confidence[i],min(confidence[j],confidence[l]),
                             inmax[i],min(outmax[j],outmax[l])])
            parents.append(i);daughters.append((j,l));arc_pairs.append((a,b))
    return (np.asarray(parents,int),np.asarray(daughters,int).reshape(-1,2),
            np.asarray(arc_pairs,int).reshape(-1,2),
            np.asarray(features,np.float32).reshape(-1,len(DIV_FEATURES)))


def predict_logistic(model,x):
    z=(np.asarray(x,float)-np.asarray(model['mean']))/np.asarray(model['scale'])
    beta=np.asarray(model['beta'])
    return expit(beta[0]+z@beta[1:])
