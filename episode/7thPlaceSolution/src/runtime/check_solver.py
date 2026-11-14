"""Independent tiny-graph checks: unchanged control and true pair reward."""
from pathlib import Path
import json
import tracksdata as td
from solve import CertifiedPairSolver,build_graph
from tracksdata.constants import DEFAULT_ATTR_KEYS as K

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    rows=[dict(t=0,z=1.,y=1.,x=1.,node_score=.9,division_cost=.3,division_prior=0.)]+[dict(t=1,z=1.,y=1.,x=float(i+2),node_score=.9,division_cost=.3,division_prior=0.) for i in range(3)]
    edges={(0,1):(.8,1.,1),(0,2):(.05,2.,1),(0,3):(.04,3.,1)}
    def solve(pair=False,standard=False):
        g,ids,em=build_graph(rows,edges,return_mappings=True)
        kw=dict(edge_weight=-td.EdgeAttr('edge_prob'),appearance_weight=.1,disappearance_weight=.1,division_weight='division_cost',num_threads=1,gap=0,timeout=30)
        if standard:s=td.solvers.ILPSolver(**kw)
        else:s=CertifiedPairSolver(edge_node_pairs={eid:(ids[a],ids[b],edges[a,b][0]) for (a,b),eid in em.items()},quality_path=out/('pair.json' if pair else 'control.json'),flow_division_costs={n:.3 for n in ids},pair_terms=[(em[0,1],em[0,2],-.5)] if pair else [],**kw)
        r=s.solve(g)
        selected={(int(x[K.EDGE_SOURCE]),int(x[K.EDGE_TARGET])) for x in r.edge_attrs().iter_rows(named=True)}
        return {(ids.index(a),ids.index(b)) for a,b in selected}
    standard=solve(standard=True);control=solve();pair=solve(pair=True)
    assert standard==control=={(0,1)},(standard,control)
    assert pair=={(0,1),(0,2)},pair
    assert pair<=set(edges)
    result=dict(passed=True,standard_edges=sorted(standard),control_edges=sorted(control),rewarded_pair_edges=sorted(pair),candidate_edges_unchanged=True)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    import sys
    run(Path(sys.argv[1]))
