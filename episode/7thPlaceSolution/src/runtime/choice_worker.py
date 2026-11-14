"""D4 detector and exact structural-gated parent/ILP pipeline, one GPU shard."""
import argparse,json,os,pickle,subprocess,sys,time,traceback,collections,concurrent.futures
from pathlib import Path


def dynamic_claims(all_stems,candidate_source):
 if candidate_source is None:raise RuntimeError('Dynamic scheduling requires shared candidate source')
 claims=Path(os.environ['BIOHUB_DYNAMIC_CLAIMS']);claims.mkdir(parents=True,exist_ok=True)
 done_marker=Path(os.environ['BIOHUB_DETECTORS_DONE'])
 while True:
  if sum((claims/stem/'done').exists() for stem in all_stems)==len(all_stems):return
  ready=[stem for stem in all_stems if (candidate_source/(stem+'.pkl')).exists() and not (claims/stem).exists()]
  ready.sort(key=lambda stem:(-(candidate_source/(stem+'.pkl')).stat().st_size,stem))
  for stem in ready:
   claim=claims/stem
   try:claim.mkdir()
   except FileExistsError:continue
   (claim/'owner.json').write_text(json.dumps(dict(pid=os.getpid(),shard=os.environ.get('BIOHUB_WORKER_SHARD'),stem=stem)))
   yield stem
   break
  else:
   if done_marker.exists():
    missing=[stem for stem in all_stems if not (candidate_source/(stem+'.pkl')).exists()]
    if missing:raise RuntimeError(f'Detectors ended without candidates: {missing}')
   time.sleep(.2)

def main(a):
 started=time.perf_counter();b=a.addon/'runtime';out=a.out;out.mkdir(parents=True,exist_ok=True)
 cfg=json.loads(a.config.read_text());solver_cfg=cfg['solver']
 assert cfg['parent_amp']=='fp32' and solver_cfg['fusion']=='pair'
 assert solver_cfg.get('pair_bonus',0)>=0 and solver_cfg['pair_parent_threshold']>=.1
 env=dict(os.environ,BIOHUB_ROOT=str(a.addon),BIOHUB_DATA_DIR=str(a.data),BIOHUB_PARENT_AMP='fp32',BIOHUB_CANDIDATE_DIR=str(out/'candidates'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',POLARS_MAX_THREADS='1')
 env['PYTHONPATH']=os.pathsep.join([str(b/'vendor'),str(b),env.get('PYTHONPATH','')]);os.environ.update(env);sys.path[:0]=[str(b/'vendor'),str(b)]
 all_stems=[p.stem for p in sorted(a.data.glob('*.zarr'))];assert all_stems
 dynamic=os.environ.get('BIOHUB_DYNAMIC_SCHEDULING')=='1'
 stems=[] if dynamic else all_stems[a.shard::a.shards]
 assert dynamic or stems
 task_stems=dynamic_claims(all_stems,a.candidate_source) if dynamic else stems
 (out/'config').mkdir(exist_ok=True);(out/'candidates').mkdir(exist_ok=True)
 solver=out/'config/solver.json';solver.write_text(json.dumps(solver_cfg));post=out/'config/post.json';post.write_text(json.dumps(cfg['post']))
 detector=[sys.executable,str(b/'capture.py'),'--model-dir',str(a.model),'--test-dir',str(a.data),'--output-dir',str(out/'detector'),'--fold','0','--checkpoint','checkpoint_final.pth','--device','cuda','--amp','fp16','--detector-tta-mode','d4-detector','--det-threshold','.3','--division-weight','.3','--num-shards',str(a.shards),'--shard-index',str(a.shard)]
 if a.max_frames is not None:detector+=['--max-frames',str(a.max_frames)]
 process=None;log=None
 try:
  if a.candidate_source:
   for stem in (all_stems if dynamic else stems):(out/'candidates'/(stem+'.pkl')).symlink_to(a.candidate_source/(stem+'.pkl'))
  else:
   log=(out/'detector.log').open('w');process=subprocess.Popen(detector,env=env,stdout=log,stderr=subprocess.STDOUT)
   if a.no_overlap:
    if process.wait():raise RuntimeError('Detector failed; inspect detector.log')
  # Persistent frozen models share the GPU only with this shard's detector.
  from parent import ParentScorer,save
  from parent_fast import score_video
  import torch
  scorer=ParentScorer()
  if os.environ.get('BIOHUB_DIVISION_TRT_ENABLE')=='1':
   from division_trt_backend import wrap_scorer
   wrap_scorer(scorer)
  videos=[];staged=[];pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
  for stem in task_stems:
   if dynamic:stems.append(stem)
   vstart=time.perf_counter();candidate=out/'candidates'/(stem+'.pkl')
   while not candidate.exists():
    if process and process.poll() is not None:raise RuntimeError(f'Detector ended without candidate {stem}')
    time.sleep(.2)
   candidate_ready=time.perf_counter();payload=pickle.loads(candidate.read_bytes());rows=payload['node_rows'];outgoing=collections.Counter(s for s,t in payload['edges']);eligible=[i for i in range(len(rows)) if outgoing.get(i,0)>=2]
   parent=out/'parent_scores'/(stem+'.json');score_video(scorer,stem,rows,a.data,parent,eligible_indices=eligible if a.gate=='fork' else None)
   parent_seconds=time.perf_counter()-candidate_ready
   pairs=out/'pair_scores_min01'/(stem+'.json');one=out/'config'/('one_'+stem+'.txt');one.write_text(stem+'\n')
   t=time.perf_counter()
   if solver_cfg.get('pair_bonus',0):
    import pairs as pairs_module
    pairs_module.run(out,'one_'+stem,2)
    from daughter_image_lgb5_runtime import rescore as image_rescore
    image_rescore(out,stem,a.data,a.addon/'models/daughter_image_lgb5',2)
   else:save(pairs,dict(stem=stem,rows=[],reason='pair_bonus zero; pair scores unused in objective',labels_read_at_inference=False))
   pair_seconds=time.perf_counter()-t
   env['BIOHUB_SOLVER_RECEIPTS']=str(out/'solver_dispatch'/stem)
   def command(label,args,command_env=None):
    t=time.perf_counter()
    with (out/(label+'.log')).open('w') as f:subprocess.run([sys.executable,*map(str,args)],env=(command_env or env),stdout=f,stderr=subprocess.STDOUT,check=True)
    return time.perf_counter()-t
   staged.append(dict(stem=stem,candidate=candidate,parent=parent,pairs=pairs,one=one,total_nodes=len(rows),fork_capable_nodes=len(eligible),candidate_wait_seconds=candidate_ready-vstart,parent_seconds=parent_seconds,pair_seconds=pair_seconds,first_future=None))
   item=staged.pop()
   stem=item['stem'];candidate=item['candidate'];parent=item['parent'];pairs=item['pairs'];one=item['one']
   solve_seconds=0.;post_seconds=0.
   # Historical transfer scripts read config/one.txt; this phase is intentionally sequential.
   (out/'config'/'one.txt').write_text(stem+'\n')
   env.update(BIOHUB_CHOICE_OUT=str(out),BIOHUB_CHOICE_BASE=str(out),BIOHUB_CHOICE_ARM=cfg['arm'],BIOHUB_CHOICE_MODEL=str(a.addon/'models'/('learned_'+cfg['arm']+'.pt')))
   choice_seconds=command('preilp_pairs_'+stem,[b/'preilp_pairs.py'])
   choice_seconds+=command('new_division_transfer_'+stem,[b/'division_new_transfer.py'])
   learned=dict(solver_cfg,learned_terms=str(out/'transfer'/'nn_h'/(stem+'.json')))
   learned_cfg=out/'config'/('learned_'+stem+'.json');learned_cfg.write_text(json.dumps(learned))
   solve_args=[b/'biohub_fast_ilp.py',b/'solve_choice.py','--candidate',candidate,'--parent',parent,'--pairs',pairs,'--config',learned_cfg,'--output',out/'graphs'/(stem+'.geff'),'--receipt',out/'learned_solver_receipts'/(stem+'.json')]
   solve_receipt=out/'learned_solver_receipts'/(stem+'.json');graph=out/'graphs'/(stem+'.geff')
   off_eligible=item['total_nodes']<=30000
   solve_env=env.copy();solve_env['BIOHUB_SCIP_SEPARATION_OFF']='1' if off_eligible else '0'
   learned_seconds=0.;off_failed=False;off_started=time.perf_counter()
   try:learned_seconds+=command('choice_solve_'+stem,solve_args,solve_env)
   except subprocess.CalledProcessError:
    if not off_eligible:raise
    off_failed=True;learned_seconds=time.perf_counter()-off_started
   if not solve_receipt.exists():raise RuntimeError(f'Missing solve receipt {stem}')
   solve_result=json.loads(solve_receipt.read_text())
   if off_eligible and solve_result.get('status')=='timelimit':
    archived=out/'learned_solver_receipts'/(stem+'.off_timelimit.json')
    assert not archived.exists();solve_receipt.rename(archived)
    off_log=out/('choice_solve_'+stem+'.log');off_log.rename(out/('choice_solve_'+stem+'.off_timelimit.log'))
    if graph.exists():
     archived_graph=out/'graphs'/(stem+'.off_timelimit.geff');assert not archived_graph.exists();graph.rename(archived_graph)
    learned_seconds+=command('choice_solve_recovery_'+stem,solve_args,env)
    (out/'solver_recovery').mkdir(exist_ok=True)
    (out/'solver_recovery'/(stem+'.json')).write_text(json.dumps(dict(initial='SCIP separation OFF timelimit',recovery='standard SCIP',initial_receipt=str(archived))))
   elif off_failed:raise RuntimeError(f'OFF solver failed without certified timelimit: {stem}: {solve_result.get("status")}')
   result=json.loads(solve_receipt.read_text())
   assert result['status']=='optimal' and result['gap']==0 and graph.exists(),(stem,result)
   post_seconds+=command('choice_post_'+stem,[b/'final_post.py','--source',out/'graphs','--dest',out/'final_graphs','--stems',one,'--data',a.data,'--candidates',out/'candidates','--normal-first'])
   videos.append(dict(stem=stem,total_nodes=item['total_nodes'],fork_capable_nodes=item['fork_capable_nodes'],candidate_wait_seconds=item['candidate_wait_seconds'],parent_seconds=item['parent_seconds'],pair_seconds=item['pair_seconds'],choice_seconds=choice_seconds,learned_seconds=learned_seconds,solve_seconds=solve_seconds,post_seconds=post_seconds,finished_at_seconds=time.perf_counter()-started))
   save(out/'pipeline_progress.json',dict(videos=videos,elapsed_seconds=time.perf_counter()-started));print(json.dumps(videos[-1]),flush=True)
   if dynamic:(Path(os.environ['BIOHUB_DYNAMIC_CLAIMS'])/stem/'done').write_text('ok\n')
  pool.shutdown(wait=True)
  if process and process.wait():raise RuntimeError('Detector failed at final wait')
  report=dict(passed=True,stems=stems,videos=videos,elapsed_seconds=time.perf_counter()-started,config=cfg,gate=a.gate,pipeline_overlap=not a.no_overlap,candidate_reused=bool(a.candidate_source),parent_gpu_peak_bytes=torch.cuda.max_memory_allocated(),tracking_fallback=False)
  if not a.candidate_source:report['detector']=json.loads((out/'detector'/f'summary_shard_{a.shard}.json').read_text())
  save(out/'worker_receipt.json',report);print(json.dumps(report),flush=True)
 except Exception:
  (out/'FAILED').write_text(traceback.format_exc());raise
 finally:
  if process and process.poll() is None:process.terminate();process.wait(timeout=30)
  if log:log.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--addon',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--max-frames',type=int);p.add_argument('--no-overlap',action='store_true');p.add_argument('--candidate-source',type=Path);p.add_argument('--gate',choices=['all','fork'],default='fork');main(p.parse_args())
