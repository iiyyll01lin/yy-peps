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
39.22. The published gain of +1.59 dB is a mean over eighteen materials,
so no single-set number here can be compared against it directly.

Extending the contrast to a second material overturns the general reading. On
`metal-plates-013` at the same budget the per-map reduction changes almost
nothing: +2.3199 dB becomes +2.2105 dB, a ratio of 0.95. The sixfold swing is a
property of `paving-stones-070`, not of the reduction. The reading that fits
both is that per-map normalisation only unlocks advantage a global reduction
was hiding in smooth maps, and on `metal-plates-013` PEPS already leads by a
wide margin, so there is nothing left to unlock.

A graded ladder on `paving-stones-070` at 240k steps shows a dose response
rather than a quirk of one implementation. Dividing each map by its own error
raised to 0, 0.5 and 1 gives +0.5601, +1.2399 and +3.1919 dB, monotone in the
normalisation strength. A fourth variant dividing by the target's dynamic range
rather than the current error gives +0.6359, barely above the global reduction,
so the mechanism is adaptive error-dependent weighting and not putting the maps
on a common scale.

The reproduction lesson is therefore not "we needed more GPUs", and not that
the reduction explains the shortfall either. It is that an unreported detail one
level below the published recipe can move a per-set result several fold while
leaving another material untouched. Rank candidate causes by measured
sensitivity, and confirm a mechanism replicates before believing it.

Evidence lives in `results/texture_repro/budget_probe/`; `curves.csv` carries
one row per loss, set, seed and budget, and `receipt.json` records the design,
the matched loss contrast, and the limitations. Both probes are labelled
`bounded_budget_probe_not_paper_comparable`: they cover two sets and at most
two seeds, and the per-map normalised loss is our own construction rather than
a recovered recipe, so they demonstrate sensitivity rather than restating the
paper's protocol.

## The shortfall is a selection effect, not an implementation error

Every one of the eleven reproduced Table 2 methods lands below its published
value, by a mean of 1.154 dB. That looks like a defect until you read what the
number averages. `table2.json` declares its own aggregation as `map_weighted`,
with the unit "individual RGB map, then mean over all maps and seeds", so the
headline figure is a plain mean over the 76 individual maps the frozen
selection carries.

The eight map categories are not comparable to one another. Pooled over all
eleven methods, eighteen sets and three seeds they run from `normal` at 32.640
dB to `Displacement` at 52.085 dB, a spread of 19.445 dB. Our selection puts
47% of its maps in the two lowest-scoring categories: eighteen `normal` and
eighteen `DIFF` out of 76.

Because the mean is taken over maps, composition moves it directly. Swapping a
single `normal` map for a single `Displacement` map shifts the headline number
by 0.2558 dB, so 4.5 such swaps, 5.9% of the selection, close the entire
shortfall. A category-balanced selection would score 41.759 against our 39.643,
a headroom of +2.115 dB, or 1.83 times the gap to be explained.

The paper names eighteen sets and eight map categories but never publishes the
file list, which `table2.json` already carries as
`texture_file_selection_not_published`. A 5.9% difference in which files were
chosen is not a coincidence that needs ruling out; it is the expected state of
affairs.

Two independent observations agree. The offset is nearly uniform across all
eleven methods, which is what differing content produces and what an
algorithmic error usually does not. And SSIM matches the published values
closely, which is what a bounded metric does when the content differs but the
method is right.

What this does not explain is the ordering. Composition shifts every method by
almost the same amount, so it cannot reorder them, and this reproduction still
places the Grid family above the NTC family where the paper does the reverse.
That mismatch is the open question, and a better target than the offset.

Evidence lives in `results/texture_repro/shortfall_analysis/`, labelled
`analysis_of_committed_evidence_no_new_measurement`: it re-reads the committed
Table 2 artifacts and runs nothing. `per_set_gap.csv` also bounds what the
two-set probes above can claim, since `paving-stones-070` ranks 8th of 18 on
the PEPS advantage and `metal-plates-013` ranks 16th.

The process lesson is the sharper one. This decomposition costs no GPU time and
uses data that already existed, yet it was run only after several GPU-hours of
loss probes. Cheap variance decomposition belongs before expensive mechanism
hunting.
