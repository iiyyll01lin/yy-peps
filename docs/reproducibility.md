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

## Application reproduction CLI

The image/texture/SDF notebooks delegate result production to one runner:

```bash
python -m experiments.reproduce check --profile paper_exact
python -m experiments.reproduce smoke --task all
python -m experiments.reproduce run --artifact image-table1 \
  --allow-protocol-assumptions
python -m experiments.reproduce run --artifact texture-table2
python -m experiments.reproduce run --artifact sdf-table3-mape
python -m experiments.reproduce run --artifact sdf-table3-l1
python -m experiments.reproduce run --artifact sdf-table4
```

Every successful command writes raw rows and a summary beside its manifest.
The prerequisite command emits schema
`peps.reproduction_prerequisites` v1 and exits 2 while any required
data/dependency/GPU condition is missing. Figure 5 additionally requires a
checksum manifest conforming to `results/schemas/fig5_dataset.schema.json`;
because the paper does not identify that dataset or its training budget, such a
run remains a documented sensitivity result rather than verified exact evidence.
