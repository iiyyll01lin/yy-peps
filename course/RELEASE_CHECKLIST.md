# Course release checklist

This checklist prepares a teaching release. It does not authorize a
`paper_exact` run or promote any paper-comparable numerical result.

## 1. Protect existing work and hardware

- [ ] Record `git status --short --branch` and `git worktree list --porcelain`.
- [ ] Do not commit, amend, reset, rebase, stash, checkout, or push as part of
  release validation.
- [ ] Confirm that no image/texture/SDF recovery worker, `torchrun`, or full
  training process is active.
- [ ] Do not launch Table 1, Table 2, Figure 5, Table 3/6, or Table 4 training.

## 2. Validate the release evidence

- [ ] Run `.venv/bin/python scripts/validate_course.py`.
- [ ] Confirm the three `course_fast` run directories contain a valid manifest,
  raw `instances.csv`, and `summary.json`.
- [ ] Confirm both convergence pilots are complete but remain explicitly
  inconclusive, with no recommended budget and no full-run authorization.
- [ ] Confirm a bare `scripts/run_image_repro_4gpu.sh table1` preflight exits
  before creating logs or workers; bounded pilot receipts must not satisfy the
  future full-run authorization contract, and
  `peps-image-repro-table1.service` must remain user-masked.
- [ ] Confirm Lucy, Thai Statue, and Armadillo each have a tracked 512³
  provenance receipt and appear in `results/sdf_repro/volume_validation.json`.
- [ ] Confirm Pitted Stonefish remains `deferred_auth_required`; no substitute
  or numeric Table 4 result is allowed.
- [ ] Confirm every top-level `results/*.csv` is still
  `legacy-unverified` in `results/manifest.json`.

## 3. Run bounded checks

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q apps course data experiments hip peps scripts tests
.venv/bin/python scripts/validate_course.py
```

If `npx` is installed, also run `make -C slides validate`. Otherwise record
Marp as explicitly blocked by the missing tool; do not silently claim a build.

The optional four-GPU smoke is safe only when all four devices are idle and no
recovery/full-training process exists:

```bash
PEPS_RUN_4GPU_TESTS=1 .venv/bin/python -m pytest -q tests/test_distributed.py
```

That smoke uses synthetic data and at most two optimizer steps. Do not replace
it with a full benchmark or paper run.

## 4. Review the handoff

- [ ] Check `results/course_release/receipt.json` against
  `results/schemas/course_release_receipt.schema.json`.
- [ ] Verify the receipt reports zero paper-comparable results and
  `paper_exact.ready=false`.
- [ ] Keep Figure 5 dataset/budget omissions, the Table 1 step omission and
  loss conflict, optimizer/seed assumptions, the unreleased SDF converter, and
  the Stonefish authorization blocker visible.
- [ ] Report tests, explicit blockers, process state, Git/worktree state, and
  every changed file. Do not commit or push.
