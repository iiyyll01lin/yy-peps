# Texture reproduction artifacts

Phase 5 uses `experiments.texture_repro`; the legacy
`results/table2_texture.csv` is not an input. Table 2 is complete; the 3F/4F
sweep has not been authorized and has not been started.

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
and budgets extended to 1000/2000/5000. Its preflight refuses CPU execution,
stale dataset receipts, active full-Table-2 workers, and insufficient
disk/VRAM.

```bash
.venv/bin/python -m experiments.texture_repro pilot-plan
.venv/bin/python -m experiments.texture_repro pilot-status
.venv/bin/python -m experiments.texture_repro pilot-report
```

The completed pilot has 30/30 trajectories and 180/180 observations. Method
ranking is stable across budgets, but the final interval still gains more than
the plateau threshold, so `convergence_pilot.json` stays explicitly
inconclusive and sets `recommended_table2_steps` to null. The pilot therefore
never justified a budget; Table 2 was authorized separately and directly.

## Full four-GPU runs

```bash
scripts/run_texture_repro_4gpu.sh table2
scripts/run_texture_repro_4gpu.sh sweep
```

Table 2 contains 594 jobs and 71,280,000 optimizer steps; it ran to completion
on two GPUs under `peps-texture-table2.service`. Measured throughput was about
2.3 jobs per hour, roughly 77 aggregate optimizer steps per second, for about
258 hours of active compute. The 3F/4F sweep contains 432 jobs and 51,840,000
steps and is still at 0/432: it has never been authorized.

Completed checkpoints are pruned after atomic result creation, except Table 2
seed-0 Paving Stones checkpoints needed by Figure 8. This reduces estimated
Table 2 completed-checkpoint storage from about 115 GB to about 2.14 GB while
retaining every interrupted job checkpoint for resume.

## Table 2 result status

All eleven methods report `complete`, 54/54 jobs and 234/234 map observations
each, with `verification_status` `complete_protocol_assumption`. The run is
reproduction evidence under declared protocol assumptions, not a paper-exact
claim.

Every method lands below its published PSNR, by 0.67 to 1.77 dB, averaging
about 1.15 dB. The deficit is not uniform: the grid family averages about
0.77 dB low while the NTC family averages about 1.50 dB low. SSIM matches the
published values closely. The PEPS effect reproduces in direction on the NTC
pipeline (`NTC_PEPS` minus `NTC_N` is +1.15 dB here against +1.59 dB
published) and is absent on the grid pipeline in both this run and the paper.
Because the NTC gain falls about 0.44 dB short, `NTC_PEPS` does not overtake
`BI-Grid` here, so the published top-line ordering is not reproduced.

## Outputs

- `protocol.json`: methods, paper values, budgets, code/data receipts, job plans.
- `table2.{json,csv}`: map-weighted overall PSNR/SSIM and eight PSNR categories.
- `table2_instances.csv`: per-set/per-method/per-seed details and compression.
- `status.json`: checkpoint-based progress for every artifact.
- `frequency_sweep.{json,csv}`: 3F/4F quality, decoder input width, and measured
  Paving Stones PyTorch latency; still empty because the sweep is unauthorized.
- `convergence_pilot.json`: bounded runtime-quality curves and conservative
  budget decision.
- `convergence_pilot_progress.json`: boot-scoped worker/checkpoint validation.
- `convergence_pilot_recovery.json`: reboot/interruption resume receipt.
- `convergence_pilot_observations.csv`: all 180 raw pilot observations.
- `figure8_status.json`: `generated`; the generator emitted `figure8.png` and
  `figure8_flip.csv` from the retained Paving Stones seed-0 checkpoints.
