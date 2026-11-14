"""Apply the exact submitted postprocess functions with explicit bounded overrides."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'legacy'))
import biohub_highscore_postprocess_ablation as m

def run(source,dest,stems,data,config):
    m.GAP_CLOSE_UM=config.get('gap',4.)
    m.GAP_REUSE_UM=3.2;m.GAP_MAX_ADDED_FRAC=config.get('cap',.03);m.GAP_MAX_ADDED_ABS=2000
    m.MIN_TRACK_LEN=config.get('length',9);m.MIN_TRACK_LEN_BY_DATASET={}
    m.LINEFIT_WINDOW=config.get('window',2);m.LINEFIT_WEIGHT=config.get('weight',.8)
    dest.mkdir(parents=True,exist_ok=True)
    for stem in stems:
        output=dest/f'{stem}.geff';receipt=dest/f'{stem}.post.json'
        if output.exists() and receipt.exists():continue
        nodes,edges=m.read_graph(source/f'{stem}.geff')
        nodes,edges,stats=m.apply_arm('gap_short_track_linefit',nodes,edges,stem,data)
        m.write_graph(output,nodes,edges)
        receipt.write_text(json.dumps(dict(config=config,stats=stats,nodes=len(nodes),edges=len(edges),fallback=False),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);p.add_argument('--stems',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    run(a.source,a.dest,a.stems.read_text().split(),a.data,json.loads(a.config.read_text()))
