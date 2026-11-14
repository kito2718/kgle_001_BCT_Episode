"""NumPy inference for an exported binary sklearn ExtraTrees forest."""
import numpy as np
class PortableForest:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.arrays={k:data[k] for k in data.files}
        self.n_features=int(self.arrays['n_features'])
    def predict_proba(self,X):
        # sklearn trees consume float32 feature values and double thresholds.
        X=np.asarray(X,dtype=np.float32).astype(np.float64)
        if X.ndim!=2 or X.shape[1]!=self.n_features or not np.isfinite(X).all():raise ValueError('Invalid feature matrix')
        a=self.arrays;p=np.zeros(len(X),dtype=np.float64)
        for root in a['roots']:
            nodes=np.full(len(X),root,dtype=np.int64)
            active=np.flatnonzero(a['left'][nodes]>=0)
            while len(active):
                n=nodes[active];go_left=X[active,a['feature'][n]]<=a['threshold'][n]
                nodes[active]=np.where(go_left,a['left'][n],a['right'][n])
                active=active[a['left'][nodes[active]]>=0]
            p+=a['probability'][nodes]
        p/=len(a['roots']);return np.column_stack([1-p,p])

def export(model,path):
    assert model.classes_.tolist()==[0,1]
    arrays={k:[] for k in ['left','right','feature','threshold','probability']};roots=[];offset=0
    for estimator in model.estimators_:
        tree=estimator.tree_;roots.append(offset)
        for name,source in [('left',tree.children_left),('right',tree.children_right)]:arrays[name].append(np.where(source>=0,source+offset,-1))
        arrays['feature'].append(tree.feature);arrays['threshold'].append(tree.threshold)
        values=tree.value[:,0,:];normalizer=values.sum(axis=1);normalizer[normalizer==0]=1;arrays['probability'].append(values[:,1]/normalizer)
        offset+=tree.node_count
    np.savez_compressed(path,**{k:np.concatenate(v) for k,v in arrays.items()},roots=np.asarray(roots,dtype=np.int64),n_features=np.array(model.n_features_in_))
