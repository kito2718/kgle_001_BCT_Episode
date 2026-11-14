"""One GPU shard: exact detector -> parent/pairs -> certified ILP -> postprocess."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def run(a):
    import torch
    cfg=json.loads(a.config.read_text())
    if cfg['status']!='selected' and not a.smoke:raise RuntimeError('Full inference requires a selected,validated configuration')
    amp=cfg['parent_amp']
    if amp not in ['bf16','fp32']:raise ValueError('Unknown parent precision')
    if amp=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('Selected parent precision requires BF16 support;do not silently change precision')
    if cfg['solver'].get('pair_bonus',0) and cfg['solver'].get('pair_parent_threshold',cfg['solver']['parent_threshold'])<.1:raise ValueError('Pair scoring floor is0.1')
    b=a.addon/'runtime';out=a.out;out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,BIOHUB_ROOT=str(a.addon),BIOHUB_DATA_DIR=str(a.data),BIOHUB_PARENT_AMP=amp,BIOHUB_CANDIDATE_DIR=str(out/'candidates'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',POLARS_MAX_THREADS='2')
    env['PYTHONPATH']=str(b/'vendor')+os.pathsep+env.get('PYTHONPATH','')
    def command(name,args):
        t=time.perf_counter()
        with (out/(name+'.log')).open('w') as f:subprocess.run([sys.executable]+list(map(str,args)),env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        timings[name]=time.perf_counter()-t
    timings={};stems=[p.stem for p in sorted(a.data.glob('*.zarr'))][a.shard::a.shards]
    if not stems:raise ValueError('Empty worker shard')
    (out/'config').mkdir(exist_ok=True);(out/'config/assigned.txt').write_text('\n'.join(stems)+'\n')
    detector=[b/'capture.py','--model-dir',a.model,'--test-dir',a.data,'--output-dir',out/'detector','--fold','0','--checkpoint','checkpoint_final.pth','--device','cuda','--amp','fp16','--detector-tta-mode','d4-detector','--det-threshold','.3','--division-weight','.3','--num-shards',a.shards,'--shard-index',a.shard]
    if a.max_frames is not None:detector+=['--max-frames',a.max_frames]
    command('detector',detector)
    command('parent_pairs',[b/'pairs.py','--out',out,'--scope','assigned','--fold','2'])
    solver=out/'config/solver.json';solver.write_text(json.dumps(cfg['solver']))
    for stem in stems:
        env['BIOHUB_SOLVER_RECEIPTS']=str(out/'solver_dispatch'/stem)
        command('solve_'+stem,[b/'biohub_fast_ilp.py',b/'solve.py','--candidate',out/'candidates'/(stem+'.pkl'),'--parent',out/'parent_scores'/(stem+'.json'),'--pairs',out/'pair_scores_min01'/(stem+'.json'),'--config',solver,'--output',out/'graphs'/(stem+'.geff'),'--receipt',out/'solver_receipts'/(stem+'.json')])
    post=out/'config/post.json';post.write_text(json.dumps(cfg['post']))
    command('postprocess',[b/'post.py','--source',out/'graphs','--dest',out/'final_graphs','--stems',out/'config/assigned.txt','--data',a.data,'--config',post])
    report=dict(shard=a.shard,stems=stems,timings_seconds=timings,config=cfg,tracking_fallback=False,detector=json.loads((out/'detector'/f'summary_shard_{a.shard}.json').read_text()))
    parent_stats=out/'parent_runtime.json'
    if parent_stats.exists():report['parent_runtime']=json.loads(parent_stats.read_text())
    (out/'worker_receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(shard=a.shard,stems=stems,timings=timings)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--addon',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--max-frames',type=int);p.add_argument('--smoke',action='store_true');run(p.parse_args())
