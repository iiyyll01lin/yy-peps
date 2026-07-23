# Experiment profiles and run provenance

Use `paper_exact` for the protocol reported by PEPS
`arXiv:2604.24167v1`; use `course_fast` for the downscaled teaching notebooks.
Both use `peps.experiment_profile` schema v1. Profiles are recursively immutable,
so a run cannot silently change a paper constant:

```python
from peps.profiles import get_profile

profile = get_profile("paper_exact")
image_config = profile.image["kodak_table_1"]
# Pass image_config values explicitly to the app builder/training code.
```

`paper_exact` is a protocol declaration, not a claim that every current
notebook already executes that workload. Quantization is marked as excluded
from the paper protocol. Parameters absent from the paper are recorded as
`not_reported`, rather than guessed. The paper also describes L1 as its stable
image protocol while its Table 1 values match the appendix's L2 row; both facts
are retained under `image["kodak_table_1"]["training"]`.

## Recording a run

```python
from peps import report
from peps.profiles import get_profile

profile = get_profile("course_fast")
manifest = report.collect_run_manifest(
    experiment="image.table_1",
    profile=profile,
    config=profile.image,
    seed=0,
    dataset_files={"kodim01": "data/raw/kodak/kodim01.png"},
)
report.write_run(
    manifest,
    [
        report.InstanceRow(
            instance_id="kodim01",
            method="g_peps",
            metric="psnr",
            value=measured_psnr,  # produced by the optimization above
            unit="dB",
        )
    ],
)
```

This writes:

```text
results/runs/<run_id>/manifest.json
results/runs/<run_id>/instances.csv
```

Manifest schema `peps.run_manifest` v1 records the immutable config and its
SHA-256, seed, UTC timestamp, git SHA/branch/dirty counts, SHA-256 and byte size
for every supplied dataset artifact, package/PyTorch/ROCm versions, and visible
GPU properties. Instance schema `peps.instance_metric` v1 is tidy: one
instance/method/metric observation per row. Put experiment-specific fields in
`metadata_json`; adding ad-hoc CSV columns is rejected so downstream aggregation
can rely on a stable schema.

## Course release contract

`results/course_release/receipt.json` is the machine-readable teaching-release
index. It promotes only manifest-backed synthetic smokes, complete bounded
pilots, and validated input provenance. Its three statuses deliberately encode
claim scope:

- `validated-course-smoke-not-paper-comparable`;
- `validated-inconclusive-pilot-not-paper-comparable`;
- `validated-input-provenance-not-numeric-result`.

The receipt and `results/manifest.json` must agree exactly. Every top-level
result CSV remains `legacy-unverified`, both pilots retain null budget
recommendations, and the receipt reports zero paper-comparable results. See
`course/RELEASE_CHECKLIST.md` for the bounded validation procedure.

## Application reproduction CLI

The common image/texture/SDF run-manifest path is:

```bash
python -m experiments.reproduce check --profile paper_exact
python -m experiments.reproduce smoke --task all
# image-table1 is code-disabled pending the separate full-reproduction gate.
python -m experiments.reproduce run --artifact texture-table2
python -m experiments.reproduce run --artifact sdf-table3-mape
python -m experiments.reproduce run --artifact sdf-table3-l1
python -m experiments.reproduce run --artifact sdf-table4
```

Every successful command writes raw rows and a summary beside its manifest.
The specialized `experiments.image_repro`, `experiments.image_convergence`,
`experiments.texture_repro`, and `experiments.sdf_repro` tools use their own
versioned status/recovery contracts. A complete specialized receipt is not
automatically a paper result; the release index controls its claim status.
The prerequisite command emits schema
`peps.reproduction_prerequisites` v1 and exits 2 while any required
data/dependency/GPU condition is missing. Figure 5 additionally requires a
checksum manifest conforming to `results/schemas/fig5_dataset.schema.json`;
because the paper does not identify that dataset or its training budget, such a
run remains a documented sensitivity result rather than verified exact evidence.

Table 1 currently accepts no authorization receipt and exits before loading
data or creating workers. A future independent gate must deliberately change
the code-level interlock; its receipt contract is frozen in
`results/schemas/full_run_authorization.schema.json`. A bounded pilot receipt
or `--allow-protocol-assumptions` cannot authorize launch or recovery.

## Independent job sharding versus one-job DDP

`experiments.run` retains its original job-level sharding. `RANK=1,
WORLD_SIZE=4` there means that process executes every fourth
instance/method/seed job. Records now state `parallelism.mode = "job_shard"` so
that value cannot be mistaken for four GPUs training one model.

`experiments.ddp` is the explicit single-job path. It requires one matching
instance/method/seed and initializes a process group from `RANK`, `WORLD_SIZE`,
and `LOCAL_RANK`. Each process calls `torch.cuda.set_device(LOCAL_RANK)`. ROCm
uses the PyTorch `cuda` device API and the `nccl` backend name; the runtime
library is RCCL.

The input is the existing tensor handoff consumed by `experiments.run`:

```python
torch.save(
    {
        "instances": [
            {
                "name": "kodim01",
                "coords": coords,       # (P, coordinate_dimension), CPU tensor
                "targets": targets,     # (P, output_dimension), CPU tensor
                "shape": (512, 768, 3),
                "metadata": {},
            }
        ]
    },
    "instances.pt",
)
```

Launch one exact job:

```bash
.venv/bin/torchrun --standalone --nproc-per-node=4 -m experiments.ddp \
  --config configs/paper/image_full.toml \
  --input instances.pt \
  --output results/paper/kodim01-g-peps-ddp \
  --instance kodim01 --method G-PEPS --seed 0
```

The configured batch is global. All ranks reproduce one checkpointable CPU
index stream, then consume exhaustive, non-overlapping slices of each global
draw. The existing random-with-replacement protocol is unchanged, so a sampled
dataset index can still occur twice in the same global draw. Uneven local
sizes are sample-weighted before DDP's rank average. Loss logging is globally
sample-weighted; only rank 0 logs, renders metrics, and atomically writes
checkpoints/results.

Checkpoints contain the unwrapped model, optimizer, scheduler, global
minibatch-stream state, step, world size, and local/global batch metadata.
Because model keys have no `module.` prefix and the stream is global, a
checkpoint can resume with `fit_paper` on one device or
`fit_paper_distributed` on a different world size without changing the global
batch protocol.

`experiments.reproduce` remains single-device. Its many paper artifact jobs can
still be distributed with the job-shard runner; that is separate from the
single selected job above.

## Four-GPU topology and performance validation

Run the bounded end-to-end suite with the repository virtual environment:

```bash
.venv/bin/python -m experiments.multigpu suite \
  --output results/multigpu/benchmark.json
```

The JSON records:

- PyTorch device names, gfx architecture, UUID and PCI BDF, plus Linux PCIe
  speed/width and NUMA data when sysfs exposes them;
- the complete directed `torch.cuda.can_device_access_peer` matrix;
- synchronized directed GPU-to-GPU `Tensor.copy_` bandwidth after warmup;
- four-rank RCCL all-reduce algorithmic bandwidth (`message_bytes / time`) and
  standard all-reduce bus bandwidth
  (`algorithmic * 2 * (world_size - 1) / world_size`);
- steady-state 1-GPU and 4-GPU throughput for the real paper Table-1 G-PEPS
  architecture, Adam dual learning rates, and fixed global batch 60,000.

The training data is deterministic synthetic coordinate regression so the
benchmark has no dataset prerequisite. It is not a dummy tensor collective:
timed steps execute the repository projector, shared grid encoder, aggregator,
MLP, loss, backward pass, Adam update, and DDP gradient collectives. Setup,
warmup, and final metric reduction are outside the timed region. Device
synchronization and the slowest rank define elapsed time.

PyTorch exposes P2P capability but not the physical route or XGMI link type.
Accordingly, the tool never infers XGMI from an all-true P2P matrix. On some
PCIe ROCm systems RCCL peer IPC also requires an `iommu=pt` boot parameter.
Default `--rccl-p2p auto` runs a direct-IPC preflight. If RCCL reports
`hipIpcGetMemHandle ... invalid argument`, the suite explicitly records that
failure and retries with `NCCL_P2P_DISABLE=1`; those collective/training numbers
are labelled host-transport fallback, not direct P2P performance. Use
`--rccl-p2p on` to require direct IPC and fail instead. The same explicit
fallback for `experiments.ddp` is `--disable-rccl-p2p`.

The opt-in hardware integration tests are:

```bash
PEPS_RUN_4GPU_TESTS=1 .venv/bin/python -m pytest -q \
  tests/test_distributed.py
```
