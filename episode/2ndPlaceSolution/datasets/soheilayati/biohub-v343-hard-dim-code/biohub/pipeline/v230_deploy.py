"""Live V230 correspondence using deployable inputs and fixed learned weights."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from biohub.pipeline.bidirectional_correspondence import inventory,candidates,features,patches,assign,materialize
from biohub.pipeline.supervised_patch_correspondence import PatchCorrespondence
from biohub.pipeline.residual_correspondence import combined_logit


def load_models(tree_path,image_path,device):
    import lightgbm as lgb
    tree=lgb.Booster(model_file=str(tree_path))
    payload=torch.load(image_path,map_location='cpu',weights_only=False)
    if tree.feature_name()!=payload['features']:raise RuntimeError('tree/image feature order differs')
    image=PatchCorrespondence(len(payload['features'])).to(device).eval();image.load_state_dict(payload['state_dict'])
    return tree,image,payload


def feature_inputs(graph,primary,upstream,volume,motion_ids,motion_nodes,backward,forward):
    st=inventory(graph,motion_ids,motion_nodes,backward,forward);pairs=candidates(st)
    ids=st['ids'];up_p=np.asarray(upstream['pi']);up_c=np.asarray(upstream['ci'])
    if not len(pairs):return st,None,None,None
    nkey=max(int(ids.max())+1,int(max(up_p.max(initial=-1),up_c.max(initial=-1)))+1)
    key=up_p.astype(np.int64)*nkey+up_c.astype(np.int64);order=np.argsort(key);ordered=key[order]
    query=ids[pairs[:,0]]*nkey+ids[pairs[:,1]];index=np.searchsorted(ordered,query)
    original=np.full((len(pairs),2),np.nan,np.float32)
    if len(ordered):
        safe=np.minimum(index,len(ordered)-1);valid=(index<len(ordered))&(ordered[safe]==query)
        primary_valid=np.asarray([primary.nodes.get(int(node))==graph.nodes[int(node)] for node in ids])
        valid &= primary_valid[pairs[:,0]] & primary_valid[pairs[:,1]]
        values=np.column_stack([upstream['p'],upstream['p2']]).astype(np.float32)
        original[valid]=values[order[index[valid]]]
    nodes=np.unique(pairs)
    local=patches(volume,st['coords'][nodes])
    wide=patches(gaussian_filter(np.asarray(volume,np.float32),sigma=(0.,.5,.5,.5),mode='nearest'),st['coords'][nodes],pitch=2)
    frame=features(st,pairs,volume,original,image_cache=(local,wide))
    images=np.stack([local,wide],axis=1).reshape(-1,2,9,9,9)
    return st,frame,images,np.searchsorted(nodes,pairs)


def probabilities(frame,images,positions,tree,image,payload,device):
    x=frame[payload['features']].to_numpy(np.float32);missing=~np.isfinite(x)
    filled=np.where(missing,payload['mean'][None],x)
    scalar=np.concatenate([(filled-payload['mean'])/payload['std'],missing.astype(np.float32)],axis=1).astype(np.float32)
    base=tree.predict(x,num_threads=1).astype(np.float32)
    settings=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        enc=[];scores=[]
        with torch.no_grad():
            for start in range(0,len(images),1024):enc.append(image.encode(torch.from_numpy(np.array(images[start:start+1024],copy=True)).to(device)))
            encoded=torch.cat(enc)
            for start in range(0,len(frame),32768):
                stop=min(start+32768,len(frame));p,c=positions[start:stop].T
                residual=image.pair(encoded[torch.from_numpy(p).to(device)],encoded[torch.from_numpy(c).to(device)],torch.from_numpy(scalar[start:stop]).to(device))
                scores.append(combined_logit(torch.from_numpy(base[start:stop]).to(device),residual).sigmoid().float().cpu().numpy())
        q=np.concatenate(scores)
        if not np.isfinite(q).all():raise RuntimeError('nonfinite live correspondence probability')
        return q
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=settings


def apply_correspondence(raw,output,primary,upstream,volume,motion_ids,motion_nodes,backward,forward,models,device):
    if set(raw.nodes)!=set(output.nodes) or raw.edges!=output.edges:raise RuntimeError('raw/output topology differs')
    if not raw.nodes:return raw,output,dict(added_edges=0,removed_edges=0,nodes_coordinates_forks_exact=True,protected_edges_exact=True,candidate_pairs=0)
    st,frame,images,positions=feature_inputs(raw,primary,upstream,volume,motion_ids,motion_nodes,backward,forward)
    if frame is None:return raw,output,dict(added_edges=0,removed_edges=0,nodes_coordinates_forks_exact=True,protected_edges_exact=True,candidate_pairs=0)
    tree,image,payload=models;q=probabilities(frame,images,positions,tree,image,payload,device);chosen=assign(st,frame,q,0)
    final,audit=materialize(raw,st,frame,chosen);out,out_audit=materialize(output,st,frame,chosen)
    invariant_keys=('nodes','edges','forks','births','deaths','edgeless_nodes','added_edges','removed_edges','nodes_coordinates_forks_exact','protected_edges_exact')
    if final.edges!=out.edges or any(audit[k]!=out_audit[k] for k in invariant_keys):raise RuntimeError('raw/output correspondence actions differ')
    audit.update(candidate_pairs=len(frame),fixed_bonus=0)
    return final,out,audit
