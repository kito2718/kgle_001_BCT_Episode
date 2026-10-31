"""Combine tree and directly supervised image correspondence evidence."""
import numpy as np


def stacked_features(frame,tree,image):
    tree,image=np.asarray(tree,np.float32),np.asarray(image,np.float32)
    if tree.shape!=(len(frame),) or image.shape!=tree.shape:raise ValueError('prediction alignment differs')
    if not np.isfinite(tree).all() or not np.isfinite(image).all() or np.any((tree<0)|(tree>1)|(image<0)|(image>1)):raise ValueError('invalid model probability')
    result=frame.copy()
    result['f_tree_probability']=tree;result['f_image_probability']=image
    result['f_model_disagreement']=np.abs(tree-image)
    return result


def assert_training_exclusion(meta,outer,row_fold):
    expected=set(range(5))-{int(outer),int(row_fold)}
    if set(meta['training_folds'])!=expected:raise ValueError('base prediction trained on held-out target fold')
