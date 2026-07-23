# Real Kodak strong-scaling benchmark

`experiments.real_workload` is the final real-data validation path for one
PEPS job. It loads `kodim01` through the checksum-verifying Kodak manifest,
builds `G-PEPS` from `configs/paper/image_full.toml`, and keeps seed `0` and
global batch `60,000` fixed for 1/2/4 GPU comparisons.

The command has no RCCL fallback mode. The two ROCr settings must be present at
the command layer, `--rccl-p2p on` is mandatory, and each multi-GPU log must
contain RCCL `via P2P/IPC` routes with no alternate channel route:

```bash
HSA_ENABLE_IPC_MODE_LEGACY=0 \
HSA_FORCE_FINE_GRAIN_PCIE=1 \
.venv/bin/python -m experiments.real_workload suite \
  --rccl-p2p on \
  --gpu-counts 1,2,4 \
  --repetitions 5 \
  --warmup-steps 100 \
  --timed-steps 500 \
  --convergence-steps 600 \
  --convergence-log-every 100 \
  --output results/multigpu/benchmark-real-workload.json
```

Before and after every measured process, the launcher requires three
consecutive sysfs observations with each selected GPU at no more than 5% busy
and 512 MiB VRAM used. It fails instead of reporting a run contaminated by
another GPU workload.

The five repetitions measure only the post-warmup training window. Separate
deterministic 600-step prefixes record comparable 1/2/4 GPU loss trajectories;
they are not used for throughput because their loss reductions add
synchronization.

Run one complete configured paper job separately:

```bash
HSA_ENABLE_IPC_MODE_LEGACY=0 \
HSA_FORCE_FINE_GRAIN_PCIE=1 \
.venv/bin/python -m experiments.real_workload full-job \
  --rccl-p2p on \
  --gpus 4 \
  --config configs/paper/image_full.toml \
  --instance kodim01 \
  --method G-PEPS \
  --seed 0
```

The full-job wrapper prepares the same manifest-backed tensor handoff consumed
by `experiments.ddp`, preflights every configured metric, runs all configured
steps, and records the DDP result, runtime, final loss/metrics, command, RCCL
route evidence, and telemetry.

## Telemetry

The host's `/opt/rocm` installation has no aligned `amd-smi` or `rocm-smi`
binary, while `/usr/bin/rocm-smi` 5.7 is incompatible and aborts. The benchmark
therefore samples read-only amdgpu sysfs attributes:

- edge, junction, and memory temperature;
- average board power;
- core and memory clock;
- VRAM used/total and GPU busy percentage.

Every receipt includes raw samples, per-card min/median/max and dispersion, the
exact sysfs source paths, and missing-value counts. Unsupported or unreadable
attributes remain JSON `null`; the collector does not infer values.

Generated files are under `results/multigpu/real-workload/` and are ignored by
Git. `results/multigpu/benchmark-p2p-ab.json` remains the authoritative direct
P2P A/B receipt and is hash-linked from the strong-scaling result.
