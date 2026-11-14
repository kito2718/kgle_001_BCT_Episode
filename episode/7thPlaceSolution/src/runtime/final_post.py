"""Only final choice graph postprocessing changes; first-stage hypotheses stay frozen."""
import argparse,json,pickle,sys,time
from pathlib import Path
from collections import defaultdict
import zarr
sys.path.insert(0,str(Path(__file__).parent/'legacy'))
import postbest as b
import structure_extra, first_projection
import boundary_mean1_post

def run(source,dest,stems,data,candidates,supported,normal_first=False):
    assert not (supported and normal_first)
    settings=dict(GAP_CLOSE_UM=4.5,GAP_REUSE_UM=3.2,GAP_MAX_ADDED_FRAC=.02,GAP_MAX_ADDED_ABS=2000,MIN_TRACK_LEN=9,MIN_TRACK_LEN_BY_DATASET={},LINEFIT_WINDOW=4,LINEFIT_WEIGHT=.6,LINEFIT_CAP_UM=2.,LINEFIT_HUBER_DELTA_UM=1.5)
    for k,v in settings.items():setattr(b,k,v)
    b.linefit=lambda n,e,stats:b.safer_local_linefit(n,e,stats,robust=False,cap_um=2.)
    dest.mkdir(parents=True,exist_ok=True)
    for stem in stems:
        old=dest/f'{stem}.post.json'
        if old.exists() and (dest/f'{stem}.geff').exists():
            receipt=json.loads(old.read_text());assert receipt['settings']==settings and receipt['supported_first']==supported and receipt.get('normal_first',False)==normal_first
            continue
        start=time.perf_counter();raw,re=b.read_graph(source/f'{stem}.geff')
        final,fe,stats=b.apply_arm('gap_short_track_linefit',raw,re,stem,data)
        assert not supported
        final,fe=boundary_mean1_post.apply(raw,re,final,fe,stem,data,candidates,b,stats)
        extra={}
        if supported:
            payload=pickle.loads((candidates/f'{stem}.pkl').read_bytes());rows=payload['node_rows'];edges=payload['edges']
            # apply_arm copies input; ensure native graph coordinates are still aligned.
            assert all(all(float(n[k])==float(rows[i][k]) for k in ('t','z','y','x')) for i,n in raw.items())
            frame_max=max(int(r['t']) for r in rows)
            freezes=first_projection.image_freezes(zarr.open_group(str(data/f'{stem}.zarr'),mode='r')['0'],range(frame_max))
            ctx=dict(raw=raw,re=re,final=final,fe=fe,rows=rows,candidates=edges,b=b,stem=stem,data_dir=data,freeze=set(freezes),first_projection=first_projection)
            final,fe,extra=structure_extra.apply('x9_first_supported_anchor',ctx)
            b.validate_and_clip_graph(final,fe,stem,data,defaultdict(int))
        if normal_first:
            transitions={int(final[e['source_id']]['t']) for e in fe if int(final[e['target_id']]['t'])==int(final[e['source_id']]['t'])+1}
            freezes=first_projection.image_freezes(zarr.open_group(str(data/f'{stem}.zarr'),mode='r')['0'],transitions)
            final,extra=first_projection.project(final,fe,set(raw),freezes)
            extra['frozen_transitions']=freezes
            b.validate_and_clip_graph(final,fe,stem,data,defaultdict(int))
        b.write_graph(dest/f'{stem}.geff',final,fe)
        (dest/f'{stem}.post.json').write_text(json.dumps(dict(settings=settings,supported_first=supported,normal_first=normal_first,stats=stats,extra=extra,nodes=len(final),edges=len(fe),seconds=time.perf_counter()-start),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','dest','stems','data','candidates']:p.add_argument('--'+name,type=Path,required=True)
    g=p.add_mutually_exclusive_group();g.add_argument('--supported',action='store_true');g.add_argument('--normal-first',action='store_true');a=p.parse_args();run(a.source,a.dest,a.stems.read_text().split(),a.data,a.candidates,a.supported,a.normal_first)
