"""Reviewed automatic fast path; preserve unsupported objectives on original ILP."""
import json
import os
from pathlib import Path
import uuid


def install():
    import tracksdata as td
    import numpy as np
    import polars as pl
    from tracksdata.constants import DEFAULT_ATTR_KEYS as K
    from tracksdata.attrs import NodeAttr, EdgeAttr
    cls = td.solvers.ILPSolver
    if getattr(cls.solve, '_biohub_fast', False):
        return
    original = cls.solve

    def solve(self, graph):
        reason = None
        if type(self) is not cls:
            reason = 'custom solver subclass (additional constraints/objective)'
        elif not self.reset:
            reason = 'incremental solve with reset=False'
        elif graph.has_overlaps() or self.merge_weight_expr is not None:
            reason = 'overlap or merge constraints'
        if reason is None:
            expressions = [self.node_weight_expr, self.appearance_weight_expr,
                           self.disappearance_weight_expr, self.division_weight_expr]
            columns = list(dict.fromkeys([K.NODE_ID] + [c for x in expressions for c in x.columns]))
            nodes = graph.node_attrs(attr_keys=columns)
            values = [np.asarray(self._evaluate_expr(x, nodes)) for x in expressions]
            if not all(np.all(v == w) for v, w in zip(values[:3], [0, .1, .1])):
                reason = 'node/appearance/disappearance weights outside reviewed scope'
            elif not np.all((values[3] == 1) | np.isposinf(values[3])):
                reason = 'division weights outside reviewed scope'
        if reason is None:
            edges = graph.edge_attrs(attr_keys=self.edge_weight_expr.columns)
            probabilities = -np.asarray(self._evaluate_expr(self.edge_weight_expr, edges))
            if not np.all(np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)):
                reason = 'edge weights outside [-1, 0]'
        dest = Path(os.environ.get('BIOHUB_SOLVER_RECEIPTS',
                    str(Path(os.environ.get('BIOHUB_OUTPUTS_DIR', '.')) / 'solver_receipts')))
        dest.mkdir(parents=True, exist_ok=True)
        receipt_path = dest / f'{os.getpid()}_{uuid.uuid4().hex}.json'
        if reason:
            receipt_path.write_text(json.dumps({'backend': 'original_ilp', 'reason': reason}))
            print('BIOHUB_SOLVER original_ilp:', reason, flush=True)
            return original(self, graph)
        # Import errors are fatal: an eligible job must not silently run slow ILP.
        from flow_solver import optimize
        ids = nodes[K.NODE_ID].to_list()
        ee = list(zip(edges[K.EDGE_ID], edges[K.EDGE_SOURCE], edges[K.EDGE_TARGET], probabilities))
        forbidden = [n for n, w in zip(ids, np.broadcast_to(values[3], (len(ids),))) if np.isposinf(w)]
        ns, es, receipt = optimize(ids, ee, self.timeout or 300, self.gap,
                                  division_forbidden=forbidden)
        receipt_path.write_text(json.dumps(receipt, indent=2))
        if ee and not es:
            raise RuntimeError(f'Empty tracking solution; review before scoring: {receipt_path}')
        if self.output_key not in graph.node_attr_keys():
            graph.add_node_attr_key(self.output_key, pl.Boolean)
        if self.output_key not in graph.edge_attr_keys():
            graph.add_edge_attr_key(self.output_key, pl.Boolean)
        graph.update_node_attrs(attrs={self.output_key: False})
        graph.update_edge_attrs(attrs={self.output_key: False})
        if ns:
            graph.update_node_attrs(node_ids=list(ns), attrs={self.output_key: True})
        if es:
            graph.update_edge_attrs(edge_ids=list(es), attrs={self.output_key: True})
        if not self.return_solution:
            return None
        return graph.filter(NodeAttr(self.output_key) == True, EdgeAttr(self.output_key) == True).subgraph()

    solve._biohub_fast = True
    cls.solve = solve
