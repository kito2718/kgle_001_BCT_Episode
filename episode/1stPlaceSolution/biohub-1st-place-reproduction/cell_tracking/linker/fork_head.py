"""Fork-configuration head of the edge transformer: scores empty, single and two-child configurations."""

from __future__ import annotations
import torch
from cell_tracking.linker.intensity import INTENSITY_FORK_DIM, intensity_fork_features
from cell_tracking.linker.occupancy import OCC_FORK_DIM, occupancy_fork_features
from cell_tracking.linker.pair_features import CONFIG_EMPTY, CONFIG_FORK, CONFIG_KIND_DIM, CONFIG_SINGLETON, FORK_GEOM_DIM, SOON_DAUGHTER, fork_pair_features


class ForkHeadMixin:
    """Methods only; parameters are created in UnetCellPoseEdgeTransformer.__init__."""

    def _soon_fork_width(self) -> int:
        """Width of the soon block appended to each fork config feature."""
        width = 0
        return width

    @staticmethod
    def _expected_fork_block(block, who_logits, config_mask, config_kind):
        """``(B, n_src, D)`` who-softmax expectation of a per-config block over the valid fork configs; zeros for a parent with no valid fork config."""
        if block is None:
            raise ValueError("fork_kind_expected=True but a per-config block was not built")
        valid = config_mask & (config_kind == CONFIG_FORK)
        has = valid.any(dim=-1, keepdim=True)
        score = who_logits.float().masked_fill(~valid, float("-inf"))
        score = torch.where(has, score, torch.zeros_like(score))
        w = torch.softmax(score, dim=-1).masked_fill(~valid, 0.0)
        out = torch.einsum("bsc,bscd->bsd", w, block.float())
        return out.to(dtype=block.dtype).masked_fill(~has, 0.0)

    @staticmethod
    def _gather_child_soon(log_soon_t1, cfg_child, b_ids):
        """``log_soon_t1`` rows of each config's child; zeros where the child is ``-1``."""
        valid = (cfg_child >= 0).unsqueeze(-1)
        picked = log_soon_t1[b_ids, cfg_child.clamp(min=0)]
        return torch.where(valid, picked, torch.zeros_like(picked))

    def _daughter_soon_stats(self, soon_t1, config_child1, config_child2, dtype):
        """Per-config ``(B, n_src, n_configs, KIND_DAUGHTER_SOON_DIM)`` daughter statistics."""
        if soon_t1 is None:
            raise ValueError("fork_kind_expected=True requires soon_t1")
        batch, n_src, n_configs = config_child1.shape
        log_t1 = soon_t1.to(dtype=dtype).clamp_min(1e-06).log()
        b_ids = torch.arange(batch, device=log_t1.device).view(batch, 1, 1)
        b_ids = b_ids.expand(batch, n_src, n_configs)
        d1 = self._gather_child_soon(log_t1, config_child1, b_ids)[..., SOON_DAUGHTER : SOON_DAUGHTER + 1]
        d2 = self._gather_child_soon(log_t1, config_child2, b_ids)[..., SOON_DAUGHTER : SOON_DAUGHTER + 1]
        return torch.cat([d1 + d2, torch.minimum(d1, d2), (d1 - d2).abs()], dim=-1)

    def _fork_extra_width(self) -> int:
        """Width of every optional block appended to each fork config feature."""
        ctx = 0
        return self._soon_fork_width() + 0 + INTENSITY_FORK_DIM + ctx + OCC_FORK_DIM

    def _fork_config_logits(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        volumes_t: torch.Tensor,
        volumes_t1: torch.Tensor,
        flow_t: torch.Tensor | None,
        logits: torch.Tensor,
        pair_hidden: torch.Tensor | None,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        vel_t: torch.Tensor | None,
        log_v_t: torch.Tensor,
        log_v_t1: torch.Tensor,
        pair_mask: torch.Tensor | None,
        who_pair_hidden: torch.Tensor | None = None,
        soon_t: torch.Tensor | None = None,
        soon_t1: torch.Tensor | None = None,
        soon_tm2: torch.Tensor | None = None,
        soon_tm1: torch.Tensor | None = None,
        soon_tp2: torch.Tensor | None = None,
        soon_has_tm2: torch.Tensor | None = None,
        soon_has_tm1: torch.Tensor | None = None,
        soon_has_tp2: torch.Tensor | None = None,
        fork_children: torch.Tensor | None = None,
        sister_key: torch.Tensor | None = None,
        sister_logit: torch.Tensor | None = None,
        intensity_t: torch.Tensor | None = None,
        intensity_t1: torch.Tensor | None = None,
        intensity_raw_sum_t: torch.Tensor | None = None,
        intensity_raw_sum_t1: torch.Tensor | None = None,
        occ_next: torch.Tensor | None = None,
        has_next: torch.Tensor | None = None,
        next_maps: torch.Tensor | None = None,
        occ_prev: torch.Tensor | None = None,
        has_prev: torch.Tensor | None = None,
        occ_fork_next: torch.Tensor | None = None,
        occ_fork_has_next: torch.Tensor | None = None,
        occ_fork_prev: torch.Tensor | None = None,
        occ_fork_has_prev: torch.Tensor | None = None,
        next_block: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score categorical empty/singleton/fork configurations."""
        if self.fork_config_mlp is None:
            raise ValueError("return_fork requires fork_head=True")
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError(f"fork scoring expects batched q/k, got {q.shape}, {k.shape}")
        batch, n_src, n_dst = (q.shape[0], q.shape[1], k.shape[1])
        if logits.shape != (batch, n_src, n_dst):
            raise ValueError(f"fork pair logits shape {tuple(logits.shape)} does not match {(batch, n_src, n_dst)}")
        if pair_mask is None:
            pair_mask = torch.ones_like(logits, dtype=torch.bool)
        else:
            pair_mask = torch.as_tensor(pair_mask, dtype=torch.bool, device=logits.device)
            if pair_mask.shape != logits.shape:
                raise ValueError(f"fork pair mask shape {tuple(pair_mask.shape)} does not match {tuple(logits.shape)}")
        shortlist_k = self.fork_train_topk if self.training else self.fork_topk
        topk = min(shortlist_k, n_dst)
        if topk <= 0:
            raise ValueError("fork_topk must leave at least one child candidate")
        scored = logits.masked_fill(~pair_mask, float("-inf"))
        if fork_children is None:
            top_scores, top_children = scored.topk(topk, dim=-1)
            top_valid = torch.isfinite(top_scores)
        else:
            if fork_children.ndim != 3 or fork_children.shape[:2] != (batch, n_src):
                raise ValueError(f"fork_children must be (B, n_src, K), got {tuple(fork_children.shape)}")
            topk = int(fork_children.shape[-1])
            if topk < 1:
                raise ValueError("fork_children must list at least one child slot")
            top_children = fork_children.clamp(min=0).long()
            top_valid = (fork_children >= 0) & (fork_children < n_dst) & pair_mask.gather(-1, top_children)
            top_scores = scored.gather(-1, top_children)
            top_valid = top_valid & torch.isfinite(top_scores)
        batch_idx = torch.arange(batch, device=q.device).view(batch, 1, 1)
        batch_idx = batch_idx.expand(batch, n_src, topk)
        source_idx = torch.arange(n_src, device=q.device).view(1, n_src, 1)
        source_idx = source_idx.expand(batch, n_src, topk)
        child_idx = top_children
        b_flat = batch_idx.reshape(-1)
        i_flat = source_idx.reshape(-1)
        j_flat = child_idx.reshape(-1)
        kind_pair_ctx = None
        if pair_hidden is None:
            h_top, _ = self._pair_encode(
                self._pair_parts_at(
                    q, k, coords_t, coords_t1, log_v_t, log_v_t1, vel_t, flow_t, feat_t, feat_t1, b_flat, i_flat, j_flat
                )
            )
            pair_hidden = h_top.view(batch, n_src, topk, -1)
        else:
            pair_hidden = pair_hidden[batch_idx, source_idx, child_idx]
        pair_scores = top_scores.detach()
        valid_f = top_valid.unsqueeze(-1)
        pair_hidden = torch.where(valid_f, pair_hidden, torch.zeros_like(pair_hidden))
        pair_scores = torch.where(top_valid, pair_scores, torch.zeros_like(pair_scores))
        col_max = scored.max(dim=1).values
        safe_col_max = torch.where(torch.isfinite(col_max), col_max, torch.zeros_like(col_max))
        pair_margin = pair_scores - safe_col_max.unsqueeze(1).expand(-1, n_src, -1).gather(-1, top_children)
        pair_margin = torch.where(top_valid, pair_margin, torch.zeros_like(pair_margin))
        n_forks = topk * (topk - 1) // 2
        n_configs = 1 + topk + n_forks
        config_kind = torch.full((batch, n_src, n_configs), CONFIG_EMPTY, dtype=torch.long, device=q.device)
        config_child1 = torch.full((batch, n_src, n_configs), -1, dtype=torch.long, device=q.device)
        config_child2 = torch.full((batch, n_src, n_configs), -1, dtype=torch.long, device=q.device)
        config_mask = torch.zeros((batch, n_src, n_configs), dtype=torch.bool, device=q.device)
        config_mask[..., 0] = True
        config_kind[..., 1 : 1 + topk] = CONFIG_SINGLETON
        config_child1[..., 1 : 1 + topk] = top_children
        config_mask[..., 1 : 1 + topk] = top_valid
        pair_hidden1 = q.new_zeros((batch, n_src, n_configs, pair_hidden.shape[-1]))
        pair_hidden2 = torch.zeros_like(pair_hidden1)
        pair_scores1 = q.new_zeros((batch, n_src, n_configs))
        pair_scores2 = torch.zeros_like(pair_scores1)
        pair_margin1 = torch.zeros_like(pair_scores1)
        pair_margin2 = torch.zeros_like(pair_scores1)
        pair_hidden1[..., 1 : 1 + topk, :] = pair_hidden
        pair_scores1[..., 1 : 1 + topk] = pair_scores
        pair_margin1[..., 1 : 1 + topk] = pair_margin
        geom = q.new_zeros((batch, n_src, n_configs, FORK_GEOM_DIM))
        intensity_fork_block = q.new_zeros((batch, n_src, n_configs, INTENSITY_FORK_DIM))
        occupancy_fork_block = q.new_zeros((batch, n_src, n_configs, OCC_FORK_DIM))
        if n_forks:
            left, right = torch.triu_indices(topk, topk, offset=1, device=q.device)
            fork = slice(1 + topk, n_configs)
            config_kind[..., fork] = CONFIG_FORK
            config_child1[..., fork] = top_children[..., left]
            config_child2[..., fork] = top_children[..., right]
            config_mask[..., fork] = top_valid[..., left] & top_valid[..., right]
            pair_hidden1[..., fork, :] = pair_hidden[..., left, :]
            pair_hidden2[..., fork, :] = pair_hidden[..., right, :]
            pair_scores1[..., fork] = pair_scores[..., left]
            pair_scores2[..., fork] = pair_scores[..., right]
            pair_margin1[..., fork] = pair_margin[..., left]
            pair_margin2[..., fork] = pair_margin[..., right]
            parent_ids = torch.arange(n_src, device=q.device).view(1, n_src, 1)
            fork_pairs = torch.stack(
                (parent_ids.expand(batch, n_src, n_forks), top_children[..., left], top_children[..., right]), dim=-1
            )
            geom_fork = fork_pair_features(
                coords_t,
                coords_t1,
                volumes_t,
                volumes_t1,
                fork_pairs.reshape(batch, n_src * n_forks, 3),
                flow_t,
                feat_t,
                feat_t1,
                use_geom=True,
                vol_conserv=True,
            )
            geom[..., fork, :] = geom_fork.view(batch, n_src, n_forks, FORK_GEOM_DIM)
            fint = intensity_fork_features(
                intensity_raw_sum_t,
                intensity_t,
                intensity_raw_sum_t1,
                intensity_t1,
                fork_pairs.reshape(batch, n_src * n_forks, 3),
            )
            intensity_fork_block[..., fork, :] = fint.view(batch, n_src, n_forks, INTENSITY_FORK_DIM)
            if (
                occ_fork_next is None
                or occ_fork_has_next is None
                or occ_fork_prev is None
                or (occ_fork_has_prev is None)
            ):
                raise ValueError(
                    "occupancy_fork_daughter=True requires occ_fork_next/occ_fork_has_next/occ_fork_prev/occ_fork_has_prev (point --occupancy-division-dir at a --out-division sidecar)"
                )
            focc = occupancy_fork_features(
                occ_fork_next,
                occ_fork_has_next,
                fork_pairs.reshape(batch, n_src * n_forks, 3),
                occ_prev=occ_fork_prev,
                has_prev=occ_fork_has_prev,
                daughter_query=True,
            )
            occupancy_fork_block[..., fork, :] = focc.view(batch, n_src, n_forks, OCC_FORK_DIM)
        kind_one_hot = torch.nn.functional.one_hot(config_kind, num_classes=CONFIG_KIND_DIM).to(dtype=q.dtype)
        blocks = [
            q.unsqueeze(2).expand(-1, -1, n_configs, -1),
            pair_hidden1 + pair_hidden2,
            (pair_hidden1 - pair_hidden2).abs(),
            geom,
            torch.stack(
                [
                    pair_scores1 + pair_scores2,
                    (pair_scores1 - pair_scores2).abs(),
                    (pair_margin1 + pair_margin2) / 2.0,
                    (pair_margin1 - pair_margin2).abs(),
                ],
                dim=-1,
            ),
            kind_one_hot,
        ]
        if intensity_t is None or intensity_t1 is None or intensity_raw_sum_t is None or (intensity_raw_sum_t1 is None):
            raise ValueError(
                "intensity_fork=True requires intensity_t/intensity_t1 and intensity_raw_sum_t/intensity_raw_sum_t1 (point --intensity-dir at a sidecar built with build_intensity_sidecar.py)"
            )
        blocks.append(intensity_fork_block)
        if occupancy_fork_block is None:
            raise ValueError("occupancy_fork=True but occupancy_fork_block was not built")
        blocks.append(occupancy_fork_block)
        features = torch.cat(blocks, dim=-1)
        who_logits = self.fork_config_mlp(features).squeeze(-1)
        if soon_t is None:
            raise ValueError("fork_hierarchical=True requires soon_t")
        log_soon_parent = soon_t.to(dtype=q.dtype).clamp_min(1e-06).log()
        kind_in = log_soon_parent

        def pick(block):
            return self._expected_fork_block(block, who_logits, config_mask, config_kind)

        occ_kind = pick(occupancy_fork_block)
        kind_in = torch.cat([kind_in, occ_kind.to(dtype=kind_in.dtype)], dim=-1)
        geom_kind = pick(geom)
        kind_in = torch.cat([kind_in, geom_kind.to(dtype=kind_in.dtype)], dim=-1)
        daughters = self._daughter_soon_stats(soon_t1, config_child1, config_child2, q.dtype)
        kind_in = torch.cat([kind_in, pick(daughters).to(dtype=kind_in.dtype)], dim=-1)
        kind_logits = self.fork_kind_mlp(kind_in)
        log_p_kind = torch.log_softmax(kind_logits, dim=-1)
        log_p_not_fork = log_p_kind[..., 0:1]
        log_p_fork = log_p_kind[..., 1:2]
        n_nf = 1 + topk
        who_nf = who_logits[..., :n_nf]
        mask_nf = config_mask[..., :n_nf]
        who_f = who_logits[..., n_nf:]
        mask_f = config_mask[..., n_nf:]
        who_nf_masked = who_nf.masked_fill(~mask_nf, float("-inf"))
        log_slot_nf = who_nf_masked - torch.logsumexp(who_nf_masked, dim=-1, keepdim=True)
        log_p_c_nf = log_p_not_fork + log_slot_nf
        if who_f.shape[-1] > 0:
            has_valid_f = mask_f.any(dim=-1, keepdim=True)
            safe_f = torch.where(mask_f, who_f, float("-inf"))
            dummy_f = torch.where(has_valid_f, safe_f, torch.zeros_like(safe_f))
            lse_f = torch.logsumexp(dummy_f, dim=-1, keepdim=True)
            log_slot_f = torch.where(has_valid_f, safe_f - lse_f, torch.full_like(who_f, float("-inf")))
            log_p_c_f = log_p_fork + log_slot_f
        else:
            log_p_c_f = who_f
        log_p = torch.cat([log_p_c_nf, log_p_c_f], dim=-1)
        log_p_masked = torch.where(config_mask, log_p, float("-inf"))
        row_lse = torch.logsumexp(log_p_masked, dim=-1, keepdim=True)
        log_p = log_p - row_lse
        config_logits = log_p.masked_fill(~config_mask, 0.0)
        null_in = k
        null_logits = self.fork_null_head(null_in).squeeze(-1)
        config_parent = torch.arange(n_src, device=q.device).view(1, n_src, 1)
        config_parent = config_parent.expand(batch, n_src, n_configs)
        out = {
            "config_logits": config_logits,
            "config_mask": config_mask,
            "config_kind": config_kind,
            "config_parent": config_parent,
            "config_child1": config_child1,
            "config_child2": config_child2,
            "null_logits": null_logits,
        }
        out["kind_logits"] = kind_logits
        out["who_logits"] = who_logits
        return out
