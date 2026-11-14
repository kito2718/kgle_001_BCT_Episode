"""Candidate-specific TH alignment; label-free, daughter permutation invariant."""
import numpy as np
SCALE=np.array([1.625,.40625,.40625])
def extra(e,q):
 out=q['outputs'];offset=np.asarray(out['offset'],float).reshape(2,3)*10
 count=np.asarray(out['count'],float);count=np.exp(count-count.max());count/=count.sum()
 parent=np.asarray(e['parent_zyx']);flow=np.asarray(q['next_flow_um'])
 children=np.array([[o['daughter1_zyx'],o['daughter2_zyx']] for o in e['options']]);actual=(children-parent)*SCALE-flow
 dist=np.linalg.norm(offset[None,:,None,:]-actual[:,None,:,:],axis=-1)
 assignments=np.stack([np.stack([dist[:,0,0],dist[:,1,1]],1),np.stack([dist[:,0,1],dist[:,1,0]],1)],1);best=assignments[np.arange(len(children)),assignments.sum(2).argmin(1)]
 # One-successor fit uses either slot; useful evidence, not a learned disappearance model.
 single=dist.min((1,2));best_single=float(single.min());sep=np.abs(np.linalg.norm(actual[:,0]-actual[:,1],axis=1)-np.linalg.norm(offset[0]-offset[1]));mid=np.linalg.norm(actual.mean(1)-offset.mean(0),axis=1)
 h=float(out['binary']);mat=np.column_stack([np.full(len(children),h),np.tile(count,(len(children),1)),best.min(1),best.max(1),best.mean(1),mid,sep,single,np.full(len(children),best_single),best.mean(1)-best_single]);assert np.isfinite(mat).all()
 return mat.astype(np.float32)
NAMES=['TH_logit','count0','count1','count2','slot_residual_min','slot_residual_max','slot_residual_mean','midpoint_residual','separation_residual','single_successor_fit','best_continuation_fit','pair_minus_best_continuation_fit']
