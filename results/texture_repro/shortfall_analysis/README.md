# Where the Table 2 shortfall comes from

Every one of the eleven reproduced Table 2 methods scores below the published
value, by a mean of **1.154 dB**. This directory decomposes that shortfall using
only committed evidence. No GPU work was run and no new measurement was taken;
`receipt.json` is derived entirely from `table2.json` and
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

## Composition alone covers the shortfall

Because the mean is taken over maps, one substitution moves it directly:

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
chosen is therefore not a plausible coincidence to rule out; it is the expected
state of affairs. Rebalancing the categories has almost twice the headroom
needed to erase the gap.

This is consistent with two other observations. The offset is nearly uniform
across all eleven methods, which is what a content difference produces and what
an algorithmic error generally does not. And SSIM matches the published values
closely, which is what one expects from a bounded metric that is far less
sensitive to map category than PSNR is.

## What this does not explain

Composition shifts every method by almost the same amount, so it cannot produce
a **reordering**. The ordering mismatch is untouched by this analysis and
remains open: the paper puts the NTC family on top (NTC_PinkPEPS 41.89,
NTC_PEPS 41.79) while this reproduction puts the Grid family on top
(Grid-PinkPEPS4F 40.603, BI-Grid 40.549), demoting the paper's best method to
fifth.

## Per-set variation, for scale

`per_set_gap.csv` records the NTC_PEPS minus NTC_N advantage per set, averaged
over the three seeds. It ranges from **-0.453 dB** (fabric-pattern-07, where
PEPS loses) to **+4.061 dB** (garden-gnome), with a standard deviation of 1.089
across the eighteen sets. The mean is +1.249 against the published +1.59.

Seed spread within a set is small by comparison, mostly under 0.5 dB. The
variation that matters here is between materials, not between seeds.

This also bounds what the two-set budget and loss probes in `../budget_probe/`
can claim: paving-stones-070 ranks 8th of 18 on this advantage and
metal-plates-013 ranks 16th, so that pair averages +1.642 against the
full-table +1.249. It is not a representative sample and no probe run on it
should be read as a Table 2-wide statement.

## Files

| file | contents |
| --- | --- |
| `receipt.json` | the decomposition, its inputs and its limitations |
| `composition.csv` | map count, share and mean PSNR per category |
| `per_set_gap.csv` | per-set NTC_PEPS advantage, ranked |

## Status

`analysis_of_committed_evidence_no_new_measurement`. `paper_exact` is false.
The paper's map-file list is unpublished, so the true composition difference is
bounded here, not measured.
