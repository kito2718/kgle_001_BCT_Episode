"""Persist an ExtraTrees model and score label-blind t+1 daughter pairs."""
from __future__ import annotations
import argparse, csv, hashlib, json, pickle, sys, os, torch
from portable_forest import PortableForest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from parent import ROOT, B, ParentScorer, score_video, save, peaks_from_rows, SPACING_ZYX
from biohub_division_pair_inference_oof import enumerate_candidate_pairs
from biohub_division_physical_pair_cpu import Candidate,PointClouds,PhysicalFeatureBuilder


def run(out,scope,fold):
    model_path=ROOT/'models/extra_trees_fold2.npz'
    fitted=json.loads((ROOT/'models/extra_trees_fold2.json').read_text());fitted['model']=PortableForest(model_path)
    assert fitted['held_out_fold']==fold
    stems=(out/'config'/f'{scope}.txt').read_text().split()
    scorer=None
    for stem in stems:
        candidate=pickle.loads((out/'candidates'/f'{stem}.pkl').read_bytes());rows=candidate['node_rows']
        parent_path=out/'parent_scores'/f'{stem}.json'
        if not parent_path.exists():
            if scorer is None:scorer=ParentScorer()
            score_video(scorer,stem,rows,Path(os.environ['BIOHUB_DATA_DIR']),parent_path)
        parents=json.loads(parent_path.read_text())['nodes']
        assert len(parents)==len(rows)
        dest=out/'pair_scores_min01'/f'{stem}.json'
        if dest.exists():continue
        peaks=peaks_from_rows(stem,rows)
        voxels={k:np.array([p.coordinate for p in v]).reshape(-1,3) for k,v in peaks.items()}
        cloud=PointClouds({k:v*SPACING_ZYX for k,v in voxels.items()},{k:np.array([p.score for p in v]) for k,v in peaks.items()},{k:np.zeros(len(v),dtype=np.int8) for k,v in peaks.items()})
        builder=PhysicalFeatureBuilder(cloud,15.,6)
        args=SimpleNamespace(parent_match_um=.25,max_pair_points=6,pair_radius_um=15.,fold=fold)
        results=[]
        outgoing={}
        for s,t in candidate['edges']:outgoing[s]=outgoing.get(s,0)+1
        for i,row in enumerate(parents):
            # 0.1 is the lowest permitted search threshold. Only existing fork-capable
            # parents need pair ranking; candidate edges themselves remain unchanged.
            if row['parent_prior']<.1 or outgoing.get(i,0)<2:continue
            t=int(row['t']);case=f'{stem}_t{t:03d}'
            c=Candidate(str(i),'detector',str(fold),case,stem,t,3,np.array([row[k] for k in ('z','y','x')])*SPACING_ZYX,0,'unknown','unknown')
            pairs,matched=enumerate_candidate_pairs(c,row['parent_prior'],builder,voxels,args)
            if not pairs:continue
            matrix=np.array([[r[f] for f in fitted['features']] for r in pairs])
            probabilities=fitted['model'].predict_proba(matrix)[:,1]
            ranking=np.argsort(-probabilities,kind='stable')
            for rank,j in enumerate(ranking):
                r=pairs[j];r.update(node_index=i,pair_score=float(probabilities[j]),pair_rank=int(rank+1),pair_rank_norm=1-rank/max(1,len(pairs)-1));results.append(r)
        save(dest,dict(stem=stem,rows=results,model=str(model_path),candidate_policy='unchanged submitted graph',minimum_parent_prior=.1,labels_read_at_inference=False))
        print('PAIRS',stem,len(results),flush=True)

    if scorer is not None and scorer.device.type=='cuda':save(out/'parent_runtime.json',dict(amp=os.environ['BIOHUB_PARENT_AMP'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--scope',default='search');p.add_argument('--fold',type=int,default=2);a=p.parse_args();run(a.out,a.scope,a.fold)
