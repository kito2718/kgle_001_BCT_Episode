"""Reuse certified solver/constraints; modify only explicit fusion coefficients."""
import sys,inspect,json,argparse
from pathlib import Path
from terms import parent_cost,incoming_alternatives,pair_reward
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import solve as core
code=inspect.getsource(core.solve)
old="r.update(division_prior=prob,division_cost=config.get('positive_cost',.3) if prob>=config.get('parent_threshold',2.) else .3)"
assert code.count(old)==1
code=code.replace(old,'r.update(division_prior=prob,division_cost=parent_cost(prob,config))')
code=code.replace('    graph,node_ids,edge_map=build_graph(rows,edges,return_mappings=True)','    alternatives=incoming_alternatives(edges)\n    graph,node_ids,edge_map=build_graph(rows,edges,return_mappings=True)')
old="pair_terms.append((edge_map[keys[0]],edge_map[keys[1]],-config['pair_bonus']*p['pair_rank_norm']));audit['attached_pairs']+=1"
new="""reward,detail=pair_reward(p,source,children,edges,alternatives,config)
            pair_terms.append((edge_map[keys[0]],edge_map[keys[1]],reward));audit['attached_pairs']+=1
            audit.setdefault('fusion_terms',[]).append(dict(node_index=source,children=children,reward=reward,**detail))"""
assert code.count(old)==1;code=code.replace(old,new)
core.parent_cost=parent_cost;core.incoming_alternatives=incoming_alternatives;core.pair_reward=pair_reward
exec(compile(code,__file__+'::solve','exec'),core.__dict__)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ['candidate','parent','pairs','config','output','receipt']:p.add_argument('--'+name,type=Path,required=name!='pairs')
 a=p.parse_args();core.solve(a.candidate,a.parent,a.pairs,json.loads(a.config.read_text()),a.output,a.receipt)
