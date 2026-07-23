# Texture reproduction artifacts

Phase 5 uses `experiments.texture_repro`; the legacy
`results/table2_texture.csv` is not an input. Full GPU jobs have not been
started; the checked-in convergence pilot is a separate, hard-capped artifact.

## Safe validation

```bash
python -m experiments.texture_repro manifest \
  --verify-files --decode-size 8
python -m experiments.texture_repro run \
  --artifact smoke --device cpu
python -m experiments.texture_repro report
```

`dataset_verification.json` currently records 18/18 decoded sets, 78/78
checksum-verified native-4K maps, all eight semantics, and dynamic output
widths from 9 to 18 channels.

## Bounded convergence pilot

`configs/paper/texture/convergence_pilot.toml` freezes two provider-diverse
sets covering all eight semantics, five architecture families, seeds 0/1/2,
and budgets 10/50/200. Its preflight refuses CPU execution, stale dataset
receipts, active full-Table-2 workers, insufficient disk/VRAM, more than 6,000
optimizer steps, or more than 20 minutes per rank.

```bash
.venv/bin/python -m experiments.texture_repro pilot-plan
.venv/bin/python -m experiments.texture_repro pilot-status
.venv/bin/python -m experiments.texture_repro pilot-report
```

The completed pilot has 30/30 trajectories and 90/90 observations. Ranking
agreement is only 0.8 and the final interval still changes by up to 10.69 dB,
so `convergence_pilot.json` is explicitly inconclusive and recommends no Table
2 budget. The 71,280,000-step run remains unauthorized.

## Full four-GPU runs

Run only after the GPUs are free:

```bash
scripts/run_texture_repro_4gpu.sh table2
scripts/run_texture_repro_4gpu.sh sweep
```

Table 2 contains 594 jobs and 71,280,000 optimizer steps. The 3F/4F sweep
contains 432 jobs and 51,840,000 steps. At hypothetical aggregate rates of
50/100/200 steps per second, Table 2 would take 16.5/8.25/4.13 days and the
sweep 12/6/3 days; these are planning scenarios, not measured texture ETAs.
`status.json` will replace them with checkpoint-based observations after launch.

Completed checkpoints are pruned after atomic result creation, except Table 2
seed-0 Paving Stones checkpoints needed by Figure 8. This reduces estimated
Table 2 completed-checkpoint storage from about 115 GB to about 2.14 GB while
retaining every interrupted job checkpoint for resume.

## Outputs

- `protocol.json`: methods, paper values, budgets, code/data receipts, job plans.
- `table2.{json,csv}`: map-weighted overall PSNR/SSIM and eight PSNR categories.
- `table2_instances.csv`: per-set/per-method/per-seed details and compression.
- `frequency_sweep.{json,csv}`: 3F/4F quality, decoder input width, and measured
  Paving Stones PyTorch latency.
- `convergence_pilot.json`: bounded runtime-quality curves and conservative
  budget decision.
- `convergence_pilot_progress.json`: boot-scoped worker/checkpoint validation.
- `convergence_pilot_recovery.json`: reboot/interruption resume receipt.
- `convergence_pilot_observations.csv`: all 90 raw pilot observations.
- `figure8_status.json`: blocked until the required final checkpoints exist;
  the generator then emits `figure8.png` and `figure8_flip.csv`.
