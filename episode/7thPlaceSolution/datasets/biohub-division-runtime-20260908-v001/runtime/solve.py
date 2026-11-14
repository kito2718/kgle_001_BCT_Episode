"""Solve unchanged candidates with optional parent cost and true daughter-pair terms.

Invoke through biohub_fast_ilp.py. Flow incumbent plus exact SCIP certification.
"""
from __future__ import annotations
import argparse,json,os,pickle,sys,time
from pathlib import Path
import numpy as np
import tracksdata as td
from ilpy import Solver,Preference,VariableType,Solution,SolverStatus
from tracksdata.constants import DEFAULT_ATTR_KEYS as K

B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'legacy'))
from biohub_motion_ilp_fold import build_graph,save_graph,DivisionPairILPSolver

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

class CertifiedPairSolver(DivisionPairILPSolver):
    def __init__(self,*,edge_node_pairs,quality_path,flow_division_costs=None,**kw):
        self.edge_node_pairs=edge_node_pairs;self.quality_path=quality_path
        self.flow_division_costs=flow_division_costs
        super().__init__(**kw)

    def _solve(self):
        solver=Solver(self._count,VariableType.Binary,preference=Preference.Scip)
        solver.set_num_threads(self.num_threads);solver.set_objective(self._objective);solver.set_constraints(self._constraints)
        solver.set_optimality_gap(0);solver.set_timeout(self.timeout)
        native=solver.native_model();variables=solver._backend._vars
        # Feasible one-successor forest supplies a nonempty incumbent. All scientific
        # constraints/objective were already constructed by the historical solver.
        selected=[];incoming=set();outgoing=set()
        for edge_id,(source,target,prob) in sorted(self.edge_node_pairs.items(),key=lambda x:-x[1][2]):
            if source not in outgoing and target not in incoming:
                selected.append(edge_id);outgoing.add(source);incoming.add(target)
        values=np.zeros(self._count)
        for node,var in self._node_vars.items():values[var.index]=1
        for node,var in self._appear_vars.items():values[var.index]=int(node not in incoming)
        for node,var in self._disappear_vars.items():values[var.index]=int(node not in outgoing)
        for edge in selected:values[self._edge_vars[edge].index]=1
        initial=native.createSol()
        for var,value in zip(variables,values):native.setSolVal(initial,var,float(value))
        warmstart_feasible=bool(native.checkSol(initial,printreason=False,completely=True,original=True))
        if not warmstart_feasible:raise RuntimeError('Greedy incumbent violates original constraints')
        native.addSol(initial)
        flow_receipt=None
        if self.flow_division_costs is not None and os.environ.get('BIOHUB_DISABLE_FLOW_SEED')!='1':
            # The launcher exposes the reviewed flow implementation and OR-Tools.
            from flow_solver import optimize
            nodes=list(self._node_vars)
            dc=[max(.1,self.flow_division_costs[node]) for node in nodes]
            ns,es,flow_receipt=optimize(nodes,[(e,s,t,p) for e,(s,t,p) in self.edge_node_pairs.items()],timeout=self.timeout,gap=0,division_costs=dc)
            inc={n:0 for n in nodes};out={n:0 for n in nodes}
            for edge in es:
                source,target,_=self.edge_node_pairs[edge];out[source]+=1;inc[target]+=1
            fv=np.zeros(self._count)
            for node,var in self._node_vars.items():fv[var.index]=int(node in ns)
            for node,var in self._appear_vars.items():fv[var.index]=int(node in ns)-inc[node]
            for node,var in self._disappear_vars.items():fv[var.index]=int(node in ns)+int(out[node]==2)-out[node]
            for node,var in self._division_vars.items():fv[var.index]=int(out[node]==2)
            for edge in es:fv[self._edge_vars[edge].index]=1
            usable=[p for p in self.pair_terms if p[0] in self._edge_vars and p[1] in self._edge_vars]
            for var,(a,b,_) in zip(self._pair_vars,usable):fv[var.index]=int(a in es and b in es)
            seed=native.createSol()
            for var,value in zip(variables,fv):native.setSolVal(seed,var,float(value))
            if not native.checkSol(seed,printreason=False,completely=True,original=True):raise RuntimeError('Flow incumbent violates original ILP constraints')
            flow_receipt.update(original_objective=float(native.getSolObjVal(seed)),original_constraints_feasible=True,seed_only=True)
            native.addSol(seed)
        native.hideOutput();native.optimize()
        status=str(native.getStatus());n_solutions=int(native.getNSols())
        receipt=dict(backend='SCIP',status=status,n_solutions=n_solutions,primal_bound=float(native.getPrimalbound()),dual_bound=float(native.getDualbound()),gap=float(native.getGap()),seconds=float(native.getSolvingTime()),warmstart_feasible=warmstart_feasible,variables=self._count,pair_terms=len(self.pair_terms),constraints_preserved=True,flow_seed=flow_receipt)
        write(self.quality_path,receipt)
        if status!='optimal' or not n_solutions:raise RuntimeError(f'Uncertified solve; retained receipt: {receipt}')
        best=native.getBestSol();v=[native.getSolVal(best,var) for var in variables]
        result=Solution(v,float(native.getSolObjVal(best)),SolverStatus.OPTIMAL,float(native.getSolvingTime()),status)
        native.freeTransform()
        return result

def solve(candidate,parent,pairs,config,output,receipt):
    payload=pickle.loads(candidate.read_bytes());rows=[dict(r) for r in payload['node_rows']];edges=payload['edges']
    if not rows or not edges:raise RuntimeError('Empty candidate graph is not a valid tracking result')
    prior=json.loads(parent.read_text())['nodes'] if parent else []
    if prior:
        assert len(prior)==len(rows)
        for i,(r,p) in enumerate(zip(rows,prior)):
            assert p['node_index']==i and all(r[k]==p[k] for k in ['t','z','y','x'])
    for i,r in enumerate(rows):
        prob=prior[i]['parent_prior'] if prior else 0.
        r.update(division_prior=prob,division_cost=config.get('positive_cost',.3) if prob>=config.get('parent_threshold',2.) else .3)
    graph,node_ids,edge_map=build_graph(rows,edges,return_mappings=True)
    pair_terms=[];audit=dict(scored_pairs=0,attached_pairs=0,unavailable_edge_pairs=0)
    if pairs and config.get('pair_bonus',0):
        lookup={(int(r['t']),r['z'],r['y'],r['x']):i for i,r in enumerate(rows)}
        for p in json.loads(pairs.read_text())['rows']:
            if p['parent_prior']<config.get('pair_parent_threshold',config.get('parent_threshold',.99)) or p['pair_rank']>config.get('pair_topk',3):continue
            audit['scored_pairs']+=1;source=int(p['node_index']);t=int(rows[source]['t'])+1
            children=[lookup.get((t,p[f'daughter{k}_z'],p[f'daughter{k}_y'],p[f'daughter{k}_x'])) for k in (1,2)]
            keys=[(source,c) for c in children]
            if any(k not in edge_map for k in keys):audit['unavailable_edge_pairs']+=1;continue
            pair_terms.append((edge_map[keys[0]],edge_map[keys[1]],-config['pair_bonus']*p['pair_rank_norm']));audit['attached_pairs']+=1
    edge_node_pairs={eid:(node_ids[s],node_ids[t],edges[(s,t)][0]) for (s,t),eid in edge_map.items()}
    solver=CertifiedPairSolver(edge_node_pairs=edge_node_pairs,quality_path=receipt,flow_division_costs={node_ids[i]:r['division_cost'] for i,r in enumerate(rows)},pair_terms=pair_terms,edge_weight=-td.EdgeAttr('edge_prob'),appearance_weight=.1,disappearance_weight=.1,division_weight='division_cost',num_threads=1,gap=0,timeout=config.get('solver_timeout',1200))
    result=solver.solve(graph)
    assert result.num_nodes()>0 and result.num_edges()>0
    output.parent.mkdir(parents=True,exist_ok=True);save_graph(result,output)
    write(output.parent/(output.stem+'.audit.json'),dict(config=config,**audit,candidate_nodes=len(rows),candidate_edges=len(edges),selected_nodes=result.num_nodes(),selected_edges=result.num_edges(),empty_graph=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--parent',type=Path);p.add_argument('--pairs',type=Path);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    solve(a.candidate,a.parent,a.pairs,json.loads(a.config.read_text()),a.output,a.receipt)
