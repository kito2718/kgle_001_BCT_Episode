"""Division846: fixed797 initialization and823 LGB/XGB probability ensemble."""
import argparse
import json
import os
import sys
from pathlib import Path
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import xgboost as xgb
from biohub_infer.lineage import div797_lineage as original
from biohub_infer.lineage import run_metric_event_pipeline as pipeline
from biohub_infer.core.metric_event_optimizer import save_problem
from biohub_infer.lineage.div763_constraint import constrain
from biohub_infer.lineage.div797_solver import compact
from biohub_infer.core.metric_calibration import bounded_logit,predict_logistic
read,digest=original.read,original.digest
FEATURES=original.FEATURES
step=original.step


class CaptureFeatures:
    def __init__(self, tree):
        self.tree=tree
        self.features=np.empty((0,len(FEATURES)),np.float32)

    def predict(self, x, **kwargs):
        self.features=x
        return self.tree.predict(x,**kwargs)


def predict(g,wide_graph,confidence,phase,old,models,trees,xgb_trees):
    wrapped={k:CaptureFeatures(v) for k,v in trees.items()}
    p=original.predict(g,wide_graph,confidence,phase,old,models,wrapped)
    p['lgb_seed_conditional']=p['division_conditional'].copy()
    values=[]
    for kind in ('native','wide'):
        x=wrapped[kind].features
        if len(x):
            init=bounded_logit(predict_logistic(models[kind]['division_conditional'],x[:,:19]))
            q=xgb_trees[kind].predict(xgb.DMatrix(x,base_margin=init,nthread=1))
        else:q=np.empty(0,np.float32)
        values.append(q)
    p['division_conditional']=(p['division_conditional']+np.concatenate(values))*np.float32(.5)
    return p


def seed_decision(p,edges):
    return original.seed_decision(dict(p,division_conditional=p['lgb_seed_conditional']),edges)


def check_bundle(path):
    path=Path(path)
    meta,trees=original.check_bundle(path)
    extra=json.loads((path/'div846.json').read_text())
    assert extra['fold']==meta['fold'] and extra['training_ids']==meta['training_ids']
    assert extra['features']==FEATURES and extra['xgboost_version']==xgb.__version__
    assert extra['source_div797_sha256']==digest(path/'div797.json')
    xgb_trees={}
    for kind in ('native','wide'):
        file=path/f'{kind}.ubj'
        assert digest(file)==extra['models'][kind]['sha256']
        xgb_trees[kind]=xgb.Booster(params={'nthread':1},model_file=file)
    return meta,trees,xgb_trees


def initialize(sid,args):
    _,edges,p,problem=pipeline.load_problem(sid,args)
    constrain(problem,p);sizes=compact(problem)
    decision,removed=seed_decision(p,edges)
    assert np.all(problem.matrix@decision.astype(float)<=problem.ub+1e-9)
    save_problem(problem,args.out/'problems'/f'{sid}.npz')
    np.savez_compressed(args.out/'round_0'/f'{sid}.npz',decision=decision)
    return dict(sid=sid,counts=pipeline.record(problem,decision),matrix=sizes,seed_demotions=removed)


def export_one(sid,args):
    try:
        import resource
        get_rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        get_rss = lambda: 0
    start=time.monotonic();models,trees,xgb_trees=check_bundle(args.model.parent)
    old=json.loads(args.model.read_text());g=read(args.graphs/f'{sid}.npz');c=read(args.confidence/f'{sid}.npz')
    np.testing.assert_array_equal(g['nodes'],c['nodes'])
    w=read(args.graphs.parent/'division_graphs'/f'{sid}.npz')
    np.testing.assert_array_equal(g['nodes'],w['nodes'])
    p=predict(g,w,c['confidence'],read(args.div_scores/f'{sid}.npz'),old,models,trees,xgb_trees)
    np.savez_compressed(args.out/'predictions'/f'{sid}.npz',**p)
    return dict(sid=sid,nodes=len(p['nodes']),native=int(p['native_count']),candidates=len(p['parents']),
        seconds=time.monotonic()-start,maxrss_kib=get_rss())


def main():
    mode=sys.argv.pop(1)
    if mode=='optimize':
        pipeline.main(initialize,step);return
    assert mode=='predict'
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('graphs','confidence','div-scores','model','out'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=4);args=parser.parse_args()
    (args.out/'predictions').mkdir(parents=True,exist_ok=True)
    from joblib import Parallel,delayed
    ids=sorted(p.stem for p in args.graphs.glob('*.npz'));assert ids
    records=[]
    for r in Parallel(n_jobs=args.jobs,return_as='generator')(delayed(export_one)(s,args) for s in ids):
        records.append(r);print('div846 predict',r,flush=True)
    (args.out/'prediction_manifest.json').write_text(json.dumps(dict(samples=records,model=str(args.model),
        base_sha256=digest(args.model),division_sha256=digest(args.model.parent/'div797.json'),xgb_sha256=digest(args.model.parent/'div846.json')),indent=2)+'\n')


if __name__=='__main__':main()
