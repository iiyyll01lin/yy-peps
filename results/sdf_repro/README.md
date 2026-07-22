# SDF reproduction status

This directory is the SDF-only output contract for the updated Phase 6.

- Table 3: Lucy, Thai Statue, and Armadillo; 10 methods; MAPE training.
- Table 6: the published 9-method L1 subset (Hash-PEPS is not in Table 6).
- Aggregate label: `three_shape_aggregate`, never the paper's four-shape
  `Global`.
- Table 4: `deferred_auth_required`. It depends entirely on the authorized
  canonical Pitted Stonefish CT mesh. No substitute mesh or numeric result is
  permitted.
- Armadillo render/FLIP: generated for Table 6 using a checked-in fixed
  orthographic camera and lighting protocol. The paper does not publish its
  camera, so these images retain `render_protocol_assumption` status.

The full matrix contains 57 jobs, 6,840,000 optimizer steps,
410,400,000,000 sampled training points, and 7,650,410,496 streamed inference
queries. Each worker holds one 512 MiB volume plus model, Adam state,
activations, and framework workspace.

Run the complete four-GPU matrix only when the GPUs are free:

```bash
scripts/run_sdf_repro_4gpu.sh
```

CPU-only validation and smoke commands:

```bash
.venv/bin/python -m experiments.sdf_repro validate
.venv/bin/python -m experiments.sdf_repro smoke
.venv/bin/python -m experiments.sdf_repro estimate
```

Raw records and checkpoints live under the git-ignored
`results/work/sdf-repro/`. Completed per-shape and three-shape aggregates are
written here. Checkpoint identity is independent of shard rank, so a failed
rank can be relaunched with the same command.
