# Where the Table 2 shortfall comes from

Every one of the eleven reproduced Table 2 methods scores below the published
value, by a mean of **1.154 dB**, and the reproduced table also reverses the
paper's ordering between the NTC and Grid families. This directory shows that
one unpublished choice accounts for both. No GPU work was run; `receipt.json`
and `reordering.json` are derived entirely from `table2.json` and
`table2_instances.csv`.

## The reported score is a mean over maps, not over materials

`table2.json` declares its own aggregation:

    global_weighting : map_weighted
    unit             : individual RGB map, then mean over all maps and seeds

So the headline number is a plain average over the 76 individual maps the
frozen selection carries. That matters because the eight map categories are not
remotely comparable to each other.

| map category | maps | share | mean PSNR |
| --- | ---: | ---: | ---: |
| normal | 18 | 23.7% | 32.640 |
| DIFF | 18 | 23.7% | 37.256 |
| ARM | 13 | 17.1% | 38.365 |
| rough | 5 | 6.6% | 38.488 |
| AO | 5 | 6.6% | 41.194 |
| specular | 3 | 3.9% | 44.510 |
| metal | 2 | 2.6% | 49.533 |
| Displacement | 12 | 15.8% | 52.085 |

The spread from `normal` to `Displacement` is **19.445 dB**, and **47% of our
maps sit in the two lowest-scoring categories**.

## Composition alone covers the level

| quantity | value |
| --- | ---: |
| effect of swapping one `normal` map for one `Displacement` map | 0.2558 dB |
| swaps needed to close the 1.154 dB shortfall | 4.5 |
| as a fraction of the selection | **5.9%** |
| our map-weighted mean | 39.643 |
| mean under a category-balanced selection | 41.759 |
| rebalancing headroom | **+2.115 dB** |
| headroom relative to the shortfall | **1.83x** |

The paper names eighteen sets and eight map categories but does not publish the
file list, which `table2.json` already records as
`texture_file_selection_not_published`. A 5.9% difference in which files were
chosen is not a coincidence to rule out; it is the expected state of affairs.

## Composition also changes the order

An earlier version of this file claimed a selection effect could shift the whole
table but never reorder it. That is wrong, and `reordering.json` shows why: it
would only hold if every method's relative strength were constant across map
categories, and it is not.

`NTC_PEPS` minus `BI-Grid`, computed within each category:

| category | `NTC_PEPS` - `BI-Grid` |
| --- | ---: |
| Displacement | **-1.17** |
| ARM | -0.52 |
| normal | -0.36 |
| AO | -0.17 |
| rough | +0.09 |
| DIFF | +0.34 |
| specular | **+2.00** |
| metal | **+2.06** |

The sign depends on the category. Our selection carries twelve `Displacement`
maps but only two `metal` and three `specular`, which is close to the worst
possible weighting for `NTC_PEPS`. Reweighting the same measurements to equal
categories:

| contrast | our composition | balanced | published |
| --- | ---: | ---: | ---: |
| `NTC_PEPS` - `BI-Grid` | -0.152 | **+0.284** | +0.540 |
| `NTC_PEPS` - `NTC_N` | +1.159 | **+1.544** | +1.590 |

The first flips sign into agreement with the paper. The second, the headline
PEPS gain, moves from 0.43 dB short of the published value to **0.046 dB
short**. Six method pairs swap places between the two weightings, and
`NTC_PEPS` moves from third to first.

## What this means

One unpublished choice, the map-file selection, is enough to account for the
level, for the size of the headline PEPS gain, and for the ordering. That is
more parsimonious than attributing the level and the order to two separate
causes, which is what an earlier version of this analysis did.

It remains a bound rather than a measurement. The paper's file list is
unpublished, so we can show what a reasonable selection does to our own numbers
but not what the authors' selection was. A perfectly balanced composition
overshoots the published absolute values, so the truth lies between our
selection and that one.

`../ordering_probe/` shows the unreported map-reduction can also flip the order
on some materials. That remains true and remains a real degree of freedom, but
it is no longer required to explain the mismatch.

## Per-set variation, for scale

`per_set_gap.csv` records the `NTC_PEPS` minus `NTC_N` advantage per set,
averaged over the three seeds. It ranges from **-0.453 dB**
(fabric-pattern-07, where PEPS loses) to **+4.061 dB** (garden-gnome), with a
standard deviation of 1.089 across the eighteen sets. Seed spread within a set
is small by comparison, mostly under 0.5 dB.

This bounds what the two-material probes in `../budget_probe/` and
`../ordering_probe/` can claim: `paving-stones-070` ranks 8th of 18 on this
advantage and `metal-plates-013` ranks 16th, so that pair averages +1.642
against the full-table +1.249 and is not a representative sample.

## Files

| file | contents |
| --- | --- |
| `receipt.json` | the decomposition, its inputs and its limitations |
| `composition.csv` | map count, share and mean PSNR per category |
| `reordering.json` | per-method category means, both rankings, every pair that swaps |
| `method_by_category.csv` | mean PSNR per method per category |
| `per_set_gap.csv` | per-set `NTC_PEPS` advantage, ranked |

## Status

`analysis_of_committed_evidence_no_new_measurement`. `paper_exact` is false.
The paper's map-file list is unpublished, so the composition difference is
bounded here, not measured.
