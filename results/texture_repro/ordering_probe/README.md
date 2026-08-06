# Can the unreported reduction explain the ordering mismatch?

Table 2 reproduces the paper's eleven methods but reverses one of its
conclusions. The paper puts the NTC family above `BI-Grid`; this reproduction
puts `BI-Grid` above the NTC family.

| contrast | paper | this reproduction |
| --- | ---: | ---: |
| `NTC_PEPS` - `BI-Grid` | +0.540 | -0.162 |
| `NTC_PinkPEPS` - `BI-Grid` | +0.640 | -0.424 |

`../shortfall_analysis/` shows that the unpublished map-file selection accounts
for the uniform 1.15 dB offset, but composition shifts every method by almost
the same amount, so it cannot reorder them. This probe tests the other
unreported choice: how the L1 is reduced across a set's maps.

## Method

Data, architecture, seed and budget are held fixed. The only thing that varies
is the exponent applied to each map's own detached error before dividing:
exponent 0 is the frozen global reduction over concatenated channels, exponent
1 is full per-map normalisation.

## Result on paving-stones-070, seed 0, 240k steps

| reduction | exponent | `BI-Grid` | `NTC_N` | `NTC_PEPS` | `NTC_PEPS` - `BI-Grid` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `global_l1` | 0.0 | 34.5411 | 33.9101 | 34.4702 | **-0.0709** |
| `sqrt_map_l1` | 0.5 | 35.7542 | 34.8824 | 36.1223 | **+0.3680** |
| `per_map_normalised_l1` | 1.0 | 36.4066 | 35.2192 | 38.4111 | **+2.0044** |

Three things hold at once. The series is monotone in the normalisation
strength. The sign flips between exponent 0 and exponent 0.5, so the
reproduction's reversal disappears. And the published margin of +0.540 is
bracketed, sitting between exponents 0.5 and 1.0 and much nearer 0.5.

The mechanism is visible in the columns: moving from exponent 0 to 1 lifts
`BI-Grid` by 1.87 dB but `NTC_PEPS` by 3.94 dB. The reduction rewards the two
methods unequally, and it rewards the PEPS side more.

## What this does and does not establish

It establishes **sufficiency**. An unreported protocol detail, one level below
the recipe the paper does publish, is enough on its own to turn the published
ordering into the reproduced one and back again. No implementation difference
is required to explain the mismatch.

It does **not** establish that the paper used such a reduction. Sufficiency is
not necessity, and another unreported choice could produce the same effect.
Nor can a single-material number be compared directly with +0.540, which is a
mean over eighteen materials and three seeds.

## Files

| file | contents |
| --- | --- |
| `receipt.json` | the ladders, the bracketing tests, coverage and limitations |
| `ladder.csv` | one row per instance, reduction, method |

`receipt.json` lists any observation still outstanding under
`coverage.pending`, so a partially populated ladder is visible rather than
silently averaged.

## Status

`bounded_ordering_probe_not_paper_comparable`, `paper_exact` false. One seed,
one budget, and reductions of our own construction. `paving-stones-070` ranks
8th of 18 on the PEPS advantage, so it is not a representative material.
