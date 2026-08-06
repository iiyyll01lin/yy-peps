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

## When an unreported protocol dominates the result

Texture Table 2 reproduces the paper's qualitative pattern but not its
top-line ordering: our `NTC_PEPS` minus `NTC_N` gain is +1.152 dB against a
published +1.59 dB, and that 0.44 dB shortfall is exactly enough to stop
`NTC_PEPS` from overtaking `BI-Grid`. Two bounded probes localise the cause,
and the second one is the more useful lesson.

The first probe tests compute. `NTC_N` and `NTC_PEPS` were retrained at
240,000 and 480,000 optimizer steps, each with its own full cosine schedule so
every point is a clean run rather than a continuation. Absolute PSNR keeps
rising, so part of the roughly 1.15 dB absolute deficit is under-training at
120k. The PEPS advantage, however, shrinks monotonically on two sets and two
seeds:

| set | seed | 120k | 240k | 480k |
| --- | --- | --- | --- | --- |
| paving-stones-070 | 0 | +0.6588 | +0.5601 | +0.4497 |
| paving-stones-070 | 1 | +0.7861 | | +0.6457 |
| metal-plates-013 | 0 | +2.5832 | +2.3199 | +2.0493 |

Seed spread of the 120k advantage is about 0.07 dB, so the shrinkage is well
outside noise. More compute lifts both methods and lifts the baseline faster,
so the published ordering cannot be recovered by training longer.

The second probe changes how that loss is reduced across maps. The paper does
report the Table 2 recipe, L1 included, and this reproduction uses L1, so the
loss family is not a deviation. What "L1" leaves open is the reduction: a set
carries five to eight maps, each decoded as three channels, and Table 2
reduces one L1 globally over all concatenated channels. That weights every
map's mean absolute error equally. Reducing per map and normalising each by
its own detached magnitude is an equally literal reading, and it hands
proportionally more gradient to maps that are already accurate. On the same
set, seed, architecture and budget:

| budget | gap under global L1 | gap under per-map normalised L1 | ratio |
| --- | --- | --- | --- |
| 240,000 | +0.5601 | +3.1919 | 5.70x |
| 480,000 | +0.4497 | +3.5334 | 7.86x |

The budget trend reverses as well: the advantage shrinks with compute under
the global loss and grows under the per-map loss. The loss also decides where
the advantage appears. At 480k, PEPS minus `NTC_N` moves from +0.89 to +14.20
dB on ambient occlusion and from -0.19 to +3.83 dB on displacement, while
colour moves from +1.82 to -0.69 dB. PEPS is strongest on the smooth maps, and
the loss decides whether those maps receive optimisation pressure.

This matters because Table 2 reports a per-map average of PSNR, a relative
log-domain quantity per map, while the frozen recipe optimises a single
absolute global L1. Aligning the loss with the metric raises both methods on
the probed set, `NTC_N` from 34.16 to 35.69 and `NTC_PEPS` from 34.61 to
39.22, and the published gain of +1.59 dB falls between our two variants.

The reproduction lesson is therefore not "we needed more GPUs". A detail the
paper leaves open, one level below the reported recipe, moves the headline
effect by almost eight times, far more than any budget effect measured here.
This is not a claim that Table 2 is mis-specified: all 594 jobs used the same
global reduction, so the method comparison is internally fair. It is a claim
about where to look first. When a reproduction misses a published margin, rank
the candidate causes by measured sensitivity before spending compute.

Evidence lives in `results/texture_repro/budget_probe/`; `curves.csv` carries
one row per loss, set, seed and budget, and `receipt.json` records the design,
the matched loss contrast, and the limitations. Both probes are labelled
`bounded_budget_probe_not_paper_comparable`: they cover two sets and at most
two seeds, and the per-map normalised loss is our own construction rather than
a recovered recipe, so they demonstrate sensitivity rather than restating the
paper's protocol.
