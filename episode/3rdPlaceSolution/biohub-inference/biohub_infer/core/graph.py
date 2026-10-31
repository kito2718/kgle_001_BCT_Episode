"""Structural check of a lineage graph (node array + parent -> child edges)."""
import numpy as np

def validate_graph(nodes, edges, *, branching=False):
    nodes = np.asarray(nodes)
    raw = np.asarray(edges)
    if nodes.ndim != 2 or nodes.shape[1] != 4:
        raise ValueError("nodes must have shape (N,4), t,z,y,x in native voxels")
    if not np.isfinite(nodes).all() or np.any(nodes < 0) or np.any(nodes != np.rint(nodes)):
        raise ValueError("nodes must be finite, nonnegative integer voxel coordinates")
    if raw.size and (raw.ndim != 2 or raw.shape[1] != 2):
        raise ValueError("edges must have shape (M,2)")
    if not np.isfinite(raw).all() or np.any(raw != np.rint(raw)):
        raise ValueError("edge indices must be finite integers")
    edges = raw.astype(np.int64).reshape(-1, 2)
    n = len(nodes)
    if edges.size:
        if edges.min() < 0 or edges.max() >= n:
            raise ValueError("edge index outside node array")
        if len(np.unique(edges, axis=0)) != len(edges):
            raise ValueError("duplicate edges")
        if np.any(nodes[edges[:, 1], 0] - nodes[edges[:, 0], 0] != 1):
            raise ValueError("links must connect consecutive frames")
        if (np.bincount(edges[:, 1], minlength=n).max() > 1
                or np.bincount(edges[:, 0], minlength=n).max() > (2 if branching else 1)):
            raise ValueError("input must be one-to-one" if not branching else "invalid lineage degree")
    return nodes.astype(np.int64), edges


