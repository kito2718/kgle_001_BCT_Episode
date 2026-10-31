"""Edge / division calibration and the division exclusion rows; no GT in inference."""
import hashlib
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
from biohub_infer.core.metric_calibration import candidate_features,bounded_logit
from biohub_infer.core.division_phase_wiring import phase_division_features,phase_edge_features,PHASE_DIV_FEATURES
from biohub_infer.lineage.div763_constraint import repair_seed

FEATURES=PHASE_DIV_FEATURES+['incoming_count','incoming_mass','incoming_max',
    'pre_parent','pre_support','pre_top1','pre_top2','pre_mean','pre_max']


def read(path):
    with np.load(path,allow_pickle=False) as z:return dict(z)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def incoming_features(pre,src,dst,q):
    n=len(pre);count=np.bincount(dst,minlength=n);mass=np.bincount(dst,weights=q,minlength=n)
    weighted=np.bincount(dst,weights=q*pre[src],minlength=n)
    maxq,maximum,support=[np.zeros(n) for _ in range(3)]
    np.maximum.at(maxq,dst,q);np.maximum.at(maximum,dst,pre[src]);np.maximum.at(support,dst,pre[src]*q)
    order=np.lexsort((np.arange(len(src)),-q,dst));target=dst[order]
    first=np.r_[True,target[1:]!=target[:-1]] if len(order) else np.zeros(0,bool)
    ix=np.flatnonzero(first);top1=np.zeros(n);top2=np.zeros(n)
    top1[target[ix]]=pre[src[order[ix]]]
    second=ix[count[target[ix]]>1]+1;top2[target[second]]=pre[src[order[second]]]
    aux=np.column_stack([np.log1p(count),mass,maxq]).astype(np.float32)
    rich=np.column_stack([pre,support,top1,top2,
        np.divide(weighted,mass,out=np.zeros(n),where=mass>0),maximum]).astype(np.float32)
    return aux,rich


def extract(g,confidence,phase):
    np.testing.assert_array_equal(g['nodes'],phase['nodes'])
    score=phase['score'];src,dst,slot,x,im,om=candidate_features(g,confidence,score,radius_um=12.)
    eligible=np.where(score>=.1,np.maximum(score,.5),score)
    par,dau,pair,dx=phase_division_features(g,confidence,eligible,phase['pre'],phase['post'],src,dst,slot,im,om)
    dx[:,0]=bounded_logit(score[par])
    ex=phase_edge_features(x,src,dst,score,phase['pre'],phase['post'])
    aux,rich=incoming_features(phase['pre'],src,dst,g['cand_prob'][src,slot])
    raw=dx.copy();dx[:,[14,16]]=0
    tree=np.column_stack([dx,aux[par],rich[par]])
    return dict(nodes=g['nodes'],src=src,dst=dst,slot=slot,parents=par,daughters=dau,arc_pairs=pair),ex,raw,dx,tree


def seed_decision(p,edges):
    n=len(p['nodes']);o=len(p['src']);m=o+len(p['parents']);decision=np.zeros(m+n,bool)
    selected=np.isin(p['src']*n+p['dst'],edges[:,0]*n+edges[:,1])
    decision[:o]=selected;decision[m+p['src'][selected]]=True;decision[m+p['dst'][selected]]=True
    incoming=np.full(n,-1,int);outgoing=np.full(n,-1,int);active=np.flatnonzero(selected)
    incoming[p['dst'][active]]=p['src'][active];outgoing[p['src'][active]]=active
    for k in np.argsort(-p['seed_division_conditional'],kind='stable'):
        if p['seed_division_conditional'][k]<.25:break
        j=p['seed_division_indices'][k];parent=p['parents'][j];dau=p['daughters'][j];a=outgoing[parent]
        if incoming[parent]<0 or a<0 or p['dst'][a] not in dau:continue
        if any(incoming[d]>=0 and incoming[d]!=parent for d in dau):continue
        if any(outgoing[d]==-1 for d in dau):continue
        decision[a]=False;decision[o+j]=True;outgoing[parent]=-2;incoming[dau]=parent;decision[m+dau]=True
    return repair_seed(p,decision)
