"""Equivalent min-cost circulation for this constant-cost, zero-node-cost ILP.

One in->out capacity-1 arc per detection represents n. Root->in is appearance;
out->root is disappearance; root->out is division; candidate out->in arcs are
links. Flow conservation is exactly the two ILP flow equalities. The remaining
d<=n constraint is redundant at an optimum: if d=1,n=0, replacing the division
arc by appearance+node arcs is feasible and reduces cost from 1 to .1.
Per-node division costs >=.1 are supported; equality ties are normalized.
No overlaps/merges and the remaining reviewed weights are required by the caller.
Probabilities and division costs are rounded to 1e-6; bounds include both. The returned dual bound accounts for
rounding both the returned flow and a floating-point optimum (at most 2N links).
"""
from pathlib import Path
import math,signal,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'ortools_runtime'))
from ortools.graph.python import min_cost_flow
SCALE=1_000_000
def optimize(node_ids,edges,timeout=300,gap=.05,prior=None,division_forbidden=None,division_costs=None):
    started=time.monotonic();n=len(node_ids);m=len(edges)
    lookup={v:i for i,v in enumerate(node_ids)};idx=np.arange(n,dtype=np.int64)
    dc=np.ones(n) if division_costs is None else np.asarray(division_costs,dtype=float)
    assert dc.shape==(n,) and np.isfinite(dc).all() and (dc>=.1).all()
    dc_int=np.rint(dc*SCALE).astype(np.int64)
    source=np.array([lookup[s] for e,s,t,p in edges],dtype=np.int64)
    target=np.array([lookup[t] for e,s,t,p in edges],dtype=np.int64)
    probability=np.array([p for e,s,t,p in edges]);assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
    starts=np.concatenate([1+2*idx,np.zeros(n,dtype=np.int64),2+2*idx,np.zeros(n,dtype=np.int64),2+2*source])
    ends=np.concatenate([2+2*idx,1+2*idx,np.zeros(n,dtype=np.int64),2+2*idx,1+2*target])
    costs=np.concatenate([np.zeros(n,dtype=np.int64),np.full(2*n,SCALE//10,dtype=np.int64),dc_int,-np.rint(probability*SCALE).astype(np.int64)])
    solver=min_cost_flow.SimpleMinCostFlow()
    capacity=np.ones(4*n+m,dtype=np.int64)
    for node in division_forbidden or []:capacity[3*n+lookup[node]]=0
    solver.add_arcs_with_capacity_and_unit_cost(starts,ends,capacity,costs)
    solver.set_nodes_supplies(np.arange(2*n+1,dtype=np.int64),np.zeros(2*n+1,dtype=np.int64))
    print('FLOW_PHASE optimize',n,m,time.monotonic()-started,flush=True)
    if hasattr(signal,'SIGALRM'):
        previous_handler=signal.signal(signal.SIGALRM,signal.SIG_DFL)
        signal.alarm(max(1,math.ceil(timeout)))
        try:status=solver.solve()
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,previous_handler)
    else:
        status=solver.solve()
    if status!=solver.OPTIMAL:raise RuntimeError(f'Flow solver not optimal: {status}')
    x=np.array(solver.flows(np.arange(4*n+m,dtype=np.int64)))
    used=x[:n];appear=x[n:2*n];disappear=x[2*n:3*n];division=x[3*n:4*n];links=x[4*n:]
    # At cost .1 a division-only unit can tie appearance+node. Normalize
    # that tie to the original d<=n constraint without changing link choices.
    invalid=(division>used)
    used[invalid]=1;appear[invalid]=1;division[invalid]=0
    inc=np.bincount(target,weights=links,minlength=n);out=np.bincount(source,weights=links,minlength=n)
    assert np.array_equal(appear+inc,used) and np.array_equal(disappear+out,used+division)
    assert (division<=used).all() and ((x==0)|(x==1)).all()
    ns={node_ids[i] for i in np.flatnonzero(used)};es={edges[i][0] for i in np.flatnonzero(links)}
    primal=float(.1*(appear.sum()+disappear.sum())+dc@division-probability@links)
    rounding_bound=(len(es)+2*n)*.5/SCALE+2*float(np.abs(dc-dc_int/SCALE).sum())
    dual=primal-rounding_bound
    receipt={'backend':'ortools_min_cost_circulation','status':'optimal_quantized_equivalent_flow','primal_bound':primal,'dual_bound':dual,'gap':rounding_bound/max(abs(primal),1e-12),'feasible':True,'nodes':len(ns),'edges':len(es),'divisions':int(division.sum()),'forbidden_divisions':len(division_forbidden or []),'seeds':[],'wall_seconds':time.monotonic()-started,'solve_seconds':time.monotonic()-started,'gap_target':gap,'timeout':timeout,'cost_scale':SCALE,'rounding_bound':rounding_bound,'constraint_equivalence':'division cost >= appearance0.1, node cost0, no overlaps/merges; equality ties normalized', 'division_cost_min':float(dc.min()) if n else None}
    return ns,es,receipt
