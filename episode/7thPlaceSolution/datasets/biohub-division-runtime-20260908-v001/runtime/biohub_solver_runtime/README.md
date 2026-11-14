# Shared Biohub tracking solver

All new tracking evaluations on RTX/Vast, including copied experiment bundles, use:

```sh
BIOHUB_SOLVER_RECEIPTS="$BIOHUB_OUTPUTS_DIR/solver_receipts/<arm>" \
python <shared-repo>/tools/biohub_fast_ilp.py <existing-ilp-script.py> <unchanged-arguments>
```

Use the existing Python and vendor PYTHONPATH. Copy this launcher and the sibling
`biohub_solver_runtime/` directory together into isolated bundles. This also works
for motion-only and Transformer tracking entries. Do not reuse the dated Temporal
repair installer as a general entry. Healthy running jobs are not modified.

Dependency: OR-Tools 9.15.6755 (`ortools.graph.python.min_cost_flow`) and the original
tracksdata/numpy/polars environment. Install OR-Tools into an isolated directory
with the experiment's Python using `pip install --no-deps --target <runtime> ortools==9.15.6755`
and set `BIOHUB_ORTOOLS_RUNTIME=<runtime>`. Missing dependency on an eligible case
is fatal; do not silently substitute a long SCIP solve.

Auto fast path: standard ILPSolver, reset=True, no overlaps or merges, node0,
appearance/disappearance0.1, division1 or +infinity veto, edge costs [-1,0].
The same graph objective is solved by circulation; probabilities round to1e-6.
Receipts store backend, feasibility, objective bounds, rounding bound and timing.
An empty tracking result with candidate edges requires review before scoring.

Changed weights, custom subclasses (including daughter-pair objective), overlaps,
merges, or incremental solves explicitly use the original ILP and record why.
Never remove constraints/alter scientific parameters to force the fast path.
For unsupported ILP use the established feasible-start/quality checking protocol;
a timeout with no usable solution must not become a valid score.

Validation: test_flow.py checks61 exhaustive DAGs including true division and veto;
test_adapter.py uses real tracksdata graphs to check dispatch and graph output.

Division0939 additionally uses the flow optimizer as a feasible incumbent for
its custom SCIP solver. The optimizer accepts per-node division costs >=0.1;
equality ties are normalized to retain d<=n. For original costs below0.1 only
the seed problem is clamped; SCIP retains the original costs and all pair terms.
The final output still requires SCIP optimal status and zero gap. This does not
extend the standard adapter dispatch or declare custom pair objectives to be
flow-equivalent. Receipts include flow_seed and original-objective feasibility.
