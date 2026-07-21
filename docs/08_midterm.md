# Midterm — matched image evidence / 期中評量

The midterm covers W01–W07. Rebuild one grid-bottleneck comparison (`fig5`) and
one Kodak method comparison (`table1`) under a single declared profile. The goal
is not to copy historical numbers or force PEPS to win; it is to produce a
matched, auditable result.

期中涵蓋 W01–W07：在同一個明確設定下重做一組 grid 瓶頸比較（`fig5`）與一組
Kodak 方法比較（`table1`）。目標不是抄既有數字，也不要求 PEPS 勝出，而是交付
公平且可稽核的結果。

## Choose one scope / 選擇範圍

- `course_fast` (default): reduced image subset and training budget suitable for
  the course. The conclusion must say that it is teaching evidence, not a paper
  reproduction.
- `paper_exact` (optional): all 24 checksum-verified Kodak images and the exact
  protocol. Budget and hardware must be approved before running.

The tracked CSVs in `results/manifest.json` are `legacy-unverified`; copying a
row from them is not a baseline rerun.

## Required artifacts / 必交項目

1. An executed notebook containing the analysis.
2. `results/midterm_<student>.csv` with raw rows for both `fig5` and `table1`.
3. A run-manifest JSON containing profile, resolved config, seeds, Git revision,
   data checksums, package/hardware versions, runtime, and source notebook.
4. `course/submissions/midterm_<student>.json`, copied from
   `course/templates/midterm_submission.json`.
5. A short conclusion and limitations statement, including negative/null results.

The standard CSV columns are:

```text
task,method,seed,metric,value,profile,status
```

Every required field must be nonblank, every numeric value finite, and every row
submitted for grading must use `status=verified`.

## Matching rules / 公平對照規則

- Same source pixels and checksum set.
- Same seeds, optimizer, step/batch budget, and metric implementations.
- Parameter budgets differ by at most the tolerance declared before the run.
- Report raw per-seed/per-image rows; aggregate only from those rows.
- A subset or shortened run is labelled `course_fast`.
- Do not use “reproduced the paper” unless `paper_exact` passes in full.

There is no PSNR/SSIM target and no “PEPS must win” rule. A correct negative result
can receive full credit.

## Automatic gate / 自動門檻

```bash
mkdir -p course/submissions
cp course/templates/midterm_submission.json \
  course/submissions/midterm_<student>.json
# Fill every field, then:
python3 scripts/validate_submission.py \
  course/submissions/midterm_<student>.json \
  --kind midterm
```

The gate checks structure, nonblank fields, finite CSV values, required `fig5`
and `table1` rows, data hashes, artifact existence, and profile consistency. It
does not decide whether the scientific conclusion is persuasive.

## Rubric / 評分

| Criterion | Points | Full-credit evidence |
|---|---:|---|
| Protocol fidelity | 25 | Correct profile; resolved config and data provenance are complete |
| Matched comparison | 25 | Data, seeds, optimizer/budget, metrics, and parameter tolerance are fair |
| Reproducible artifacts | 20 | Notebook, raw CSV, and run manifest agree and pass the validator |
| Analysis | 20 | Conclusion follows the measurements; uncertainty and negative results are handled honestly |
| Communication | 10 | Figures/tables are readable and limitations are explicit |

Blank placeholders, copied legacy rows, fabricated verification status, or a
missing run manifest fail the automatic gate and must be corrected before grading.
