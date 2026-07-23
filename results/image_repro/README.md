# Image/core reproduction artifacts

This directory is the Phase 4 image/core namespace for PEPS Extended
`arXiv:2604.24167v1`. Generated PNG files are ignored; checked-in CSV/JSON
receipts retain the data, protocol assumptions, and blocker state.

## Current evidence classes

- Figures 1, 2, 4, 10, and 11 are analytic regenerations. Figure 11 explicitly
  records that the paper source does not define its third plotted coordinate.
- Figure 3 is computed from all 24 original-size Kodak images and an explicitly
  selected ten-set texture sensitivity. The paper does not identify its ten
  texture sets or PSD normalization.
- Figures 6 and 7 are generated only after all Table 1 jobs finish.
- Figure 5 remains blocked: the paper does not identify its native-4K images,
  image count, batch size, optimizer, or training steps. The existing
  `image-fig5` runner accepts a checksum manifest and explicit sensitivity
  budget, but its output is never labelled exact.
- Smoke rows use one 512-sample optimizer step at native Kodak resolution and
  are always labelled `smoke_not_paper_comparable`.

## Commands

```bash
# Analytic and PSD figures; dependent figures remain blocked until full results.
.venv/bin/python -m experiments.image_figures all --device cuda:0

# Bounded smoke shards do not require full-run authorization.
bash scripts/run_image_repro_4gpu.sh smoke
bash scripts/run_image_repro_4gpu.sh appendix-smoke

# Table 1 currently exits before creating logs or workers. A separate
# full-reproduction gate must deliberately change the code-level interlock.
bash scripts/run_image_repro_4gpu.sh table1

# Queue Table 5 and Appendix matrices only after their own review.
bash scripts/run_image_repro_4gpu.sh table5
bash scripts/run_image_repro_4gpu.sh core-ablations
bash scripts/run_image_repro_4gpu.sh recipe-ablations

# Refresh exact job/optimizer-step completion and paired confidence reports.
.venv/bin/python -m experiments.image_repro status
```

The full Table 1 configuration is 648 jobs
(`24 images × 9 methods × 3 seeds`) and 77,760,000 optimizer steps. Table 5
adds 216 L1 jobs; its L2 rows reuse Table 1. Core and recipe ablations each add
360 jobs. These totals are intentionally not hidden behind an aggregate
percentage.

The Table 1 preflight is code-disabled after the external-recovery incident and
accepts no authorization receipt. The future receipt contract is frozen in
`results/schemas/full_run_authorization.schema.json`, but enabling it requires a
deliberate code review in the separate full-reproduction gate. The bounded
convergence receipt remains explicitly non-authorizing. Direct
`experiments.image_repro`, legacy `experiments.reproduce`, and the user-systemd
manager enforce the same interlock.

Raw checkpoints and worker logs live under ignored `results/work/image-repro/`.
Per-job JSON records contain instance, seed, method, rank, parameter split,
compression factor, training recipe, metric versions, elapsed time, Git state,
and all measured metrics. `../image_repro_instances.csv` is the tracked tidy
projection of every available raw instance/seed/metric row;
`../image_repro_summary.csv` and `../image_repro_paired.json` provide aggregate
and paired-confidence views with explicit completeness labels.

Status generation never treats PID existence alone as worker liveness. New
worker receipts bind the PID to the current boot, process start time, and
command digest; older receipts without that identity are conservatively
unverified. When a worker disappears without updating its receipt, the status
uses validated result/checkpoint evidence to report a stopped state and freezes
the throughput observation at the latest valid output timestamp.

Malformed expected results/checkpoints, unexpected job files, and interrupted
atomic-write temporaries are reported as integrity failures and excluded from
derived summaries. Checkpoints created before embedded job identity was added
remain loadable but are explicitly labelled `legacy-unverified` rather than
silently upgraded.
