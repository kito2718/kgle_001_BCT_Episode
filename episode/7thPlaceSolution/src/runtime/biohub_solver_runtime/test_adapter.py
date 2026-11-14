import json, os, tempfile
from pathlib import Path
import polars as pl
import tracksdata as td
from adapter import install
install()

def graph(veto=False):
    g=td.graph.InMemoryGraph()
    g.add_node_attr_key("division_cost", pl.Float64, 1.)
    ids=g.bulk_add_nodes([{"t":0,"division_cost":float("inf") if veto else 1.}, {"t":1}, {"t":1}, {"t":2}, {"t":2}])
    g.add_edge_attr_key("p",pl.Float64,0.)
    g.bulk_add_edges([{"source_id":ids[s],"target_id":ids[t],"p":.99} for s,t in [(0,1),(0,2),(1,3),(2,4)]])
    return g
with tempfile.TemporaryDirectory() as tmp:
    os.environ["BIOHUB_SOLVER_RECEIPTS"]=tmp
    def run(**kwargs):
        return td.solvers.ILPSolver(edge_weight=-td.EdgeAttr("p"), appearance_weight=.1, disappearance_weight=.1, division_weight="division_cost", **kwargs).solve(graph())
    assert run().num_edges()==4
    veto=td.solvers.ILPSolver(edge_weight=-td.EdgeAttr("p"),appearance_weight=.1,disappearance_weight=.1,division_weight="division_cost").solve(graph(True))
    assert veto.num_edges()==3
    assert run(return_solution=False) is None
    td.solvers.ILPSolver(edge_weight=-td.EdgeAttr("p"),appearance_weight=.1,disappearance_weight=.1,division_weight=.3).solve(graph())
    class Custom(td.solvers.ILPSolver): pass
    Custom(edge_weight=-td.EdgeAttr("p"), appearance_weight=.1,disappearance_weight=.1,division_weight=1.).solve(graph())
    receipts=[json.loads(p.read_text()) for p in Path(tmp).glob("*.json")]
    assert sum(r["backend"]=="ortools_min_cost_circulation" for r in receipts)==3
    assert sum(r["backend"]=="original_ilp" for r in receipts)==2
print("ADAPTER_TESTS_PASS: regular, veto, return contract, changed weights, custom subclass")
