# SDF `3-of-4 public subset` status

This directory is the SDF-only output contract for the updated Phase 6.

- Public inputs: provenance-validated Lucy, Thai Statue, and Armadillo 512³
  volumes.
- MAPE: the 10-method Table 3 configuration.
- L1: the published 9-method appendix subset (no invented Hash-PEPS row).
- Every execution/result scope is exactly `3-of-4 public subset`; it is never
  full Table 3 or the paper's four-shape `Global`.
- Table 4: `deferred_auth_required`. It depends entirely on the authorized
  canonical Pitted Stonefish CT mesh. It is not executed by this launcher, and
  no substitute mesh or numeric result is permitted.
- Armadillo render/FLIP: generated for the L1 subset using a checked-in fixed
  orthographic camera and lighting protocol. The paper does not publish its
  camera, so these images retain `render_protocol_assumption` status.

The full matrix contains 57 jobs, 6,840,000 optimizer steps,
410,400,000,000 sampled training points, and 7,650,410,496 streamed inference
queries. Each worker holds one 512 MiB volume plus model, Adam state,
activations, and framework workspace.

Run the manifest-backed matrix on physical GPUs 2 and 3:

```bash
bash scripts/run_sdf_public_subset_2gpu.sh
```

CPU-only validation and smoke commands:

```bash
.venv/bin/python -m experiments.sdf_repro validate
.venv/bin/python -m experiments.sdf_repro smoke
.venv/bin/python -m experiments.sdf_repro estimate
```

Raw records and checkpoints live under the git-ignored
`results/work/sdf-repro/<run-id>/`. Completed per-instance/method/loss rows,
GPU-hours, checkpoint hashes, validation, and residual limitations are indexed
by `public_subset_receipt.json`. Relaunching the same command resumes the same
manifest-backed checkpoints.

`volume_validation.json` records the checksum- and schema-validated public
512³ volumes. Its recovery section confirms that the latest host reboot left
no related process or partial file, so the complete Lucy, Thai Statue, and
Armadillo outputs were accepted without rerunning preprocessing. Pitted
Stonefish remains `deferred_auth_required`, with no substitution.
