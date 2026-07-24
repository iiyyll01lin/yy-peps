# Kodak image-budget convergence pilot

This is a bounded calibration artifact, not a paper-exact result and not a
verified Table 1 reproduction. It did not resume the 648-job Table 1 run.

## Result

**Inconclusive**; no optimizer-step budget is recommended.

a final-interval mean PSNR change exceeded 0.15 dB. No untested budget is inferred.

- Jobs: 18/18
- Optimizer steps: 2160000/2160000
- Additional optimizer steps: 1620000/1620000
- Curve points: 144/144
- Latest queued two-GPU launch wall time: 10537.163 seconds
- Total/source/extension GPU-hours: 7.741191 / 1.905315 / 5.835876
- Budgets: 1000, 3000, 10000, 20000, 30000, 60000, 90000, 120000
- Methods: Grid (baseline), G-PEPS (PEPS), G-P-PEPS (Pink)
- Seeds: 0, 1

The deterministic three-image subset spans the lowest and highest measured
Kodak luma-gradient images and includes a portrait near the 24-image median.
It is a coverage subset, not a population estimate.

The 30k model, Adam moments, and minibatch stream were hash-validated and
retained. The source cosine had already reached zero, so the continuation
explicitly re-horizons the learning rates to the 120k global-cosine value at
step 30k. Any recommendation is therefore a checkpoint-continuation protocol
assumption, not an uninterrupted paper-exact 120k schedule.

## Evidence and commands

- `receipt.json` records protocol, hardware, runtime, checkpoint hashes,
  stability, the decision, and assumption status.
- `curves.csv` has one runtime/quality row per image/method/seed/budget.
- Local resumable checkpoints and raw curves live under the receipt's
  `run_manifest` work namespace and are intentionally git-ignored.

```bash
bash scripts/run_image_convergence_2gpu.sh
.venv/bin/python -m experiments.image_convergence validate
```
