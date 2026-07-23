# Kodak image-budget convergence pilot

This is a bounded calibration artifact, not a paper-exact result and not a
verified Table 1 reproduction. The pilot did not resume or consume the 648-job
Table 1 namespace. Separate recovery launchers did restart that matrix during
the session. The incident receipt identifies the source as a prior Cursor
one-shot wake chain followed by the transient user-systemd service
`peps-image-repro-table1.service`. It records exact ancestry and a bounded
zero-process observation. All wakes are exited and unrearmed, the service is
permanently user-masked, and every Table 1 entry point is code-disabled pending
the separate full-reproduction gate.

## Result

**Inconclusive**; no optimizer-step budget is recommended.

a final-interval mean PSNR change exceeded 0.15 dB. No untested budget is inferred.

- Jobs: 18/18
- Optimizer steps: 540000/540000
- Curve points: 90/90
- Latest four-GPU launch wall time: 2670.250 seconds
- Budgets: 1000, 3000, 10000, 20000, 30000
- Methods: Grid (baseline), G-PEPS (PEPS), G-P-PEPS (Pink)
- Seeds: 0, 1

The deterministic three-image subset spans the lowest and highest measured
Kodak luma-gradient images and includes a portrait near the 24-image median.
It is a coverage subset, not a population estimate.

## Evidence and commands

- `receipt.json` records protocol, hardware, runtime, checkpoint hashes,
  stability, the decision, and assumption status.
- `curves.csv` has one runtime/quality row per image/method/seed/budget.
- `external_table1_recovery_incident.json` records the unrelated full-run
  relaunches and the final stopped/checkpoint-valid state.
- Local resumable checkpoints and raw curves live under the receipt's
  `run_manifest` work namespace and are intentionally git-ignored.

```bash
bash scripts/run_image_convergence_4gpu.sh
.venv/bin/python -m experiments.image_convergence validate --no-work
```
