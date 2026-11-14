import itertools,json,random
from pathlib import Path
from flow_solver import optimize
random.seed(20260908)
cases=[[(0,0,1,.99),(1,0,2,.99),(2,1,3,.99),(3,2,4,.99)]]
for _ in range(60):
    pairs=[(s,t) for s in range(5) for t in range(s+1,5) if random.random()<.7]
    cases.append([(i,s,t,round(random.uniform(.01,1),7)) for i,(s,t) in enumerate(pairs)])
results=[]
for case_index,edges in enumerate(cases):
    forbidden=[] if case_index==0 else [i for i in range(5) if (i+case_index)%2==0]
    nodes=list(range(5));exact=0.
    dc=[1.,.3,.2,.1,.3] if case_index else [1.]*5
    for bits in itertools.product([0,1],repeat=len(edges)):
        inc=[0]*5;out=[0]*5
        for b,(e,s,t,p) in zip(bits,edges):inc[t]+=b;out[s]+=b
        if max(inc)>1 or max(out)>2 or any(out[i]>1 for i in forbidden):continue
        cost=-sum(b*x[3] for b,x in zip(bits,edges))
        for i in nodes:
            use=int(bool(inc[i] or out[i]));div=int(out[i]==2)
            cost+=.1*(use-inc[i]+use+div-out[i])+dc[i]*div
        exact=min(exact,cost)
    ns,es,r=optimize(nodes,edges,division_forbidden=forbidden,division_costs=dc)
    assert r['dual_bound']-1e-9<=exact<=r['primal_bound']+1e-9,(exact,r)
    results.append({'exact':exact,'flow':r['primal_bound'],'divisions':r['divisions'],'bound':r['rounding_bound']})
assert results[0]['divisions']==1,results[0]
Path(__file__).with_name('flow_tests.json').write_text(json.dumps({'passed':len(results),'includes_positive_division_case':True,'results':results},indent=2))
print('FLOW_EXHAUSTIVE_TESTS_PASS',len(results),flush=True)
