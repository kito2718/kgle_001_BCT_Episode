"""Append learned pair margins to frozen strongest fusion; unchanged ILP constraints."""
from pathlib import Path
import sys,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
import solve_fusion as f
needle='    edge_node_pairs={eid:'
insertion='''    learned=json.loads(Path(config['learned_terms']).read_text())
    assert learned['candidate_contract']==payload['contract'] and not learned['labels_read']
    for term in learned['rows']:
        source=term['node_index'];children=term['children']
        assert int(rows[source]['t'])==term['parent_t'] and [rows[source][a] for a in 'zyx']==term['parent_zyx']
        keys=[(source,c) for c in children]
        assert all(k in edge_map for k in keys)
        assert -.20000001<=term['reward']<=.20000001
        reward=term['reward']
        if term['logit']<0:
            pair_ids={edge_map[keys[0]],edge_map[keys[1]]}
            old_reward=sum(w for a,b,w in pair_terms if {a,b}==pair_ids)
            weak=min(edges[k][0] for k in keys)
            assert abs(rows[source]['division_cost']-.3)<1e-8
            amplitude=max(.2,weak+.1-.3-old_reward+.01)
            reward=amplitude*(-np.tanh(term['logit']))
            assert 0<=reward<=2
        pair_terms.append((edge_map[keys[0]],edge_map[keys[1]],reward))
    audit['learned_pair_terms']=len(learned['rows'])
'''
assert f.code.count(needle)==1
code=f.code.replace(needle,insertion+needle);exec(compile(code,__file__+'::solve','exec'),f.core.__dict__)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['candidate','parent','pairs','config','output','receipt']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();f.core.solve(a.candidate,a.parent,a.pairs,json.loads(a.config.read_text()),a.output,a.receipt)
