"""Bounded diagnostic fusion; scores are not assumed calibrated probabilities."""
def parent_cost(p,cfg):
 mode=cfg.get('fusion','step')
 if mode=='continuous':return .3-cfg.get('strength',.2)*min(1.,max(0.,p))
 if mode in ('pair','competition'):return .3
 return cfg.get('positive_cost',.3) if p>=cfg.get('parent_threshold',2.) else .3

def incoming_alternatives(edges):
 result={}
 for (source,target),value in edges.items():result.setdefault(target,[]).append((float(value[0]),source))
 return {target:sorted(values,reverse=True)[:2] for target,values in result.items()}

def pair_reward(pair,source,children,edges,alternatives,cfg):
 p=min(1.,max(0.,float(pair['parent_prior'])));q=min(1.,max(0.,float(pair['pair_score'])))
 support=[]
 for child in children:
  edge=float(edges[(source,child)][0]);other=max([v for v,s in alternatives.get(child,[]) if s!=source]+[0.])
  support.append(edge-other)
 margin=min(support)
 gate=min(1.,max(0.,.5+margin/(2*cfg.get('competition_tau',.1)))) if cfg['fusion']=='competition' else 1.
 return -cfg.get('strength',.2)*p*q*gate,dict(parent=p,pair_score=q,weakest_edge_margin=margin,competition_gate=gate)
