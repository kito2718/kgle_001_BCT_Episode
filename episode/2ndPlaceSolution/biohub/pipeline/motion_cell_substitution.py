"""Select raw cell alternatives using agreement of two temporal motion anchors."""
from collections import Counter, defaultdict
import numpy as np
from scipy.spatial import cKDTree

from biohub.io.submission import DatasetGraph, validate_graph
from biohub.metrics.gt import SCALE_ARR
from biohub.pipeline.contested_pair import neighborhood

RADIUS_UM = 3.0
FORK_CLEARANCE_UM = 14.0


def propose_substitutions(base, motion_ids, motion_nodes, backward, forward, candidates):
    ids = np.asarray(list(base.nodes), np.int64)
    coords = np.asarray(list(base.nodes.values()), np.float64).reshape(-1,4)
    lookup = {int(n):i for i,n in enumerate(ids)}
    n = len(ids)
    back, ahead = np.full((n,3),np.nan), np.full((n,3),np.nan)
    if len(motion_ids)!=len(motion_nodes) or np.shape(backward)!=(len(motion_ids),3) or np.shape(forward)!=(len(motion_ids),3):
        raise ValueError('unaligned motion inputs')
    for j,node in enumerate(motion_ids):
        i=lookup.get(int(node))
        if i is not None:
            if not np.array_equal(coords[i],motion_nodes[j]):raise ValueError('motion source coordinate differs')
            back[i],ahead[i]=backward[j],forward[j]
    pairs=np.asarray([(lookup[p],lookup[c]) for p,c,w in base.edges],np.int64).reshape(-1,2)
    indegree=np.bincount(pairs[:,1],minlength=n);outdegree=np.bincount(pairs[:,0],minlength=n)
    prev,nxt=np.full(n,-1,np.int64),np.full(n,-1,np.int64)
    prev[pairs[:,1]],nxt[pairs[:,0]]=pairs[:,0],pairs[:,1]
    guard=neighborhood(base,ids[outdegree==2],2)
    valid=np.isfinite(back).all(axis=1)&np.isfinite(ahead).all(axis=1)
    xyz=coords[:,1:]*SCALE_ARR
    raw=np.clip(np.rint(candidates[['z','y','x']].to_numpy(np.float64)),0,np.asarray(base.shape[1:])-1)
    if not np.isfinite(raw).all():raise ValueError('nonfinite candidates')
    raw_t=candidates.t.to_numpy(np.int64);raw_xyz=raw*SCALE_ARR
    score=candidates.score.to_numpy(np.float64)
    if not np.isfinite(score).all():raise ValueError('nonfinite candidate score')
    pools,existing,guard_trees={},{},{}
    for t in np.unique(coords[:,0].astype(int)):
        rows=np.flatnonzero(raw_t==t)
        if len(rows):pools[int(t)]=(rows,cKDTree(raw_xyz[rows]))
        rows=np.flatnonzero(coords[:,0]==t)
        existing[int(t)]=(rows,cKDTree(xyz[rows]))
        protected=np.asarray([i for i in rows if int(ids[i]) in guard],np.int64)
        if len(protected):guard_trees[int(t)]=cKDTree(xyz[protected])
    proposals=[];eligible=0;agreement=0
    for i in np.flatnonzero((indegree==1)&(outdegree==1)&valid):
        p,c=int(prev[i]),int(nxt[i]);t=int(coords[i,0])
        if p<0 or c<0 or not valid[p] or not valid[c] or any(int(ids[j]) in guard for j in (p,i,c)):
            continue
        if outdegree[p]!=1 or indegree[c]!=1 or t not in pools:
            continue
        left,right=xyz[p]+ahead[p],xyz[c]+back[c]
        if np.linalg.norm(left-right)>RADIUS_UM:
            continue
        agreement+=1
        old=np.linalg.norm(np.stack([left,right])-xyz[i],axis=1)
        # Ordinary center must disagree with at least one mutually consistent anchor.
        if np.max(old)<=RADIUS_UM:
            continue
        if t in guard_trees and guard_trees[t].query(xyz[i])[0]<=FORK_CLEARANCE_UM:
            continue
        eligible+=1
        rows,tree=pools[t]
        options=[]
        for j in tree.query_ball_point(left,RADIUS_UM):
            r=int(rows[j])
            if np.linalg.norm(raw_xyz[r]-right)>RADIUS_UM or np.linalg.norm(raw_xyz[r]-xyz[i])<=RADIUS_UM:
                continue
            base_rows,base_tree=existing[t]
            close=base_tree.query_ball_point(raw_xyz[r],RADIUS_UM)
            if any(int(base_rows[k])!=int(i) for k in close):
                continue
            if t in guard_trees and guard_trees[t].query(raw_xyz[r])[0]<=FORK_CLEARANCE_UM:
                continue
            cost=float(np.mean(np.sum((np.stack([left,right])-raw_xyz[r])**2,axis=1)))
            old_cost=float(np.mean(old**2))
            if cost>=old_cost:
                continue
            options.append((cost,-float(score[r]),r))
        if options:
            cost,_,r=min(options)
            proposals.append(dict(node=int(ids[i]),parent=int(ids[p]),child=int(ids[c]),candidate_row=r,
                old_node=tuple(map(int,coords[i])),new_node=(t,*map(int,raw[r])),
                old_residual_um2=float(np.mean(old**2)),new_residual_um2=cost,
                improvement_um2=float(np.mean(old**2)-cost)))
    proposals.sort(key=lambda a:(-a['improvement_um2'],a['new_residual_um2'],a['node'],a['candidate_row']))
    actions,used,peaks=[],set(),set()
    added=defaultdict(list)
    for action in proposals:
        touched={action['parent'],action['node'],action['child']}
        if touched&used or action['candidate_row'] in peaks:
            continue
        t,*point=action['new_node'];position=np.asarray(point)*SCALE_ARR
        if added[t] and np.any(np.linalg.norm(np.asarray(added[t])-position,axis=1)<=RADIUS_UM):
            continue
        actions.append(action);used.update(touched);peaks.add(action['candidate_row'])
        added[t].append(position)
    return actions,dict(consistent_anchor_triples=agreement,outlier_centers=eligible,
                        proposals=len(proposals),selected=len(actions),protected_nodes=len(guard))


def materialize_substitutions(base,actions):
    nodes=dict(base.nodes)
    touched=set()
    guard=neighborhood(base,[p for p,count in Counter(p for p,c,w in base.edges).items() if count==2],2)
    for action in actions:
        node=int(action['node'])
        if node not in nodes or node in touched or node in guard:raise ValueError('invalid or protected substitution')
        point=tuple(map(int,action['new_node']))
        if point[0]!=int(nodes[node][0]):raise ValueError('substitution changed the timepoint')
        nodes[node]=point;touched.add(node)
    graph=DatasetGraph(base.dataset,nodes,list(base.edges),base.shape)
    audit=validate_graph(graph)
    if set(graph.nodes)!=set(base.nodes) or graph.edges!=base.edges or any(graph.nodes[n]!=base.nodes[n] for n in guard):
        raise RuntimeError('node identities, edges or fork coordinates changed')
    audit.update(changed_coordinates=sum(graph.nodes[n]!=base.nodes[n] for n in touched),
        node_ids_count_exact=True,edges_exact=True,protected_coordinates_exact=True,
        no_action_exact=not actions and graph.nodes==base.nodes and graph.edges==base.edges,
        max_displacement_um=max((float(np.linalg.norm((np.asarray(graph.nodes[n][1:])-np.asarray(base.nodes[n][1:]))*SCALE_ARR)) for n in touched),default=0.))
    return graph,audit
