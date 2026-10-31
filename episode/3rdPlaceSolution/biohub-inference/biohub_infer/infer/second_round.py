"""Second round of the matcher: for every node, the top-k next-frame candidates with the matcher re-run
after the nodes the first-round Hungarian already linked are REMOVED FROM ITS INPUT (renormalising the
softmax afterwards is too late: the cross-attention has already conditioned on the taken child). These
candidates feed the LP wiring and the division candidate pool.
"""
import numpy as np
import torch

SCALE = np.array([1.625, 0.40625, 0.40625])   # (z, y, x) µm/voxel

@torch.no_grad()
def second_round(per_t, edges, model, cfg, device, drifts, k=1):
    """Re-run the matcher per boundary with the claimed frame-b nodes removed
    from its INPUT; return each node's top-k successor among what remains."""
    pos = [np.asarray(p, np.float64).reshape(-1, 3) for p, _ in per_t]
    offs, n = [0], 0
    for p in pos:
        n += len(p)
        offs.append(n)
    claimed = np.zeros(n, bool)
    for _, j in edges:
        claimed[int(j)] = True

    idx = -np.ones((n, k), np.int64)
    prob = np.zeros((n, k), np.float32)
    radius = float(cfg.loss.embed_radius_um)
    scale = torch.tensor(SCALE, dtype=torch.float32, device=device)
    for t in range(len(pos) - 1):
        pa, ea = per_t[t]
        pb, eb = per_t[t + 1]
        free = np.nonzero(~claimed[offs[t + 1]:offs[t + 2]])[0]
        if not len(pa) or not len(free):
            continue
        fa = torch.from_numpy(ea).to(device)
        fb = torch.from_numpy(eb[free]).to(device)
        ga = torch.from_numpy(pa.astype(np.float32)).to(device) * scale
        gb = torch.from_numpy(pb[free].astype(np.float32)).to(device) * scale
        offset = torch.as_tensor(drifts[t], device=device,
                                 dtype=torch.float32).expand(len(pa), 3)
        logits, null_a, _ = model.matcher(fa, ga, fb, gb, radius,
                                          candidate_offset_a=offset)
        p_ab = torch.softmax(torch.cat([logits, null_a[:, None]], 1), 1)[:, :-1]
        p_ab = p_ab.float().cpu().numpy()
        kk = min(k, p_ab.shape[1])
        order = np.argsort(-p_ab, axis=1)[:, :kk]
        rows = np.arange(len(pa))[:, None]
        idx[offs[t]:offs[t + 1], :kk] = free[order] + offs[t + 1]
        prob[offs[t]:offs[t + 1], :kk] = p_ab[rows, order]
    return idx, prob


