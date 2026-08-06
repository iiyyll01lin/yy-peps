# Can the unreported reduction explain the ordering mismatch?

Table 2 reproduces the paper's eleven methods but reverses one of its
conclusions. The paper puts the NTC family above `BI-Grid`; the reproduced
table puts `BI-Grid` above the NTC family.

| contrast | paper | this reproduction |
| --- | ---: | ---: |
| `NTC_PEPS` - `BI-Grid` | +0.540 | -0.162 |
| `NTC_PinkPEPS` - `BI-Grid` | +0.640 | -0.424 |

`../shortfall_analysis/` shows the unpublished map-file selection accounts for
the uniform 1.15 dB offset, but composition shifts every method by nearly the
same amount, so it cannot reorder rows. This probe tests the other unreported
choice: how the L1 is reduced across a set's maps.

## Method

Data, architecture, seed and budget are held fixed. Only the exponent applied
to each map's own detached error before dividing changes: exponent 0 is the
frozen global reduction over concatenated channels, exponent 1 is full per-map
normalisation.

## The reversal is not present on every material

| instance | exponent 0.0 | exponent 0.5 | exponent 1.0 | published |
| --- | ---: | ---: | ---: | ---: |
| paving-stones-070 | **-0.0709** | +0.3680 | +2.0044 | +0.540 |
| metal-plates-013 | **+1.9302** | pending | +1.3483 | +0.540 |

On `paving-stones-070` the reproduction does reverse the published order, and
sweeping the reduction flips it back: the series is monotone, the sign changes
between exponent 0 and 0.5, and the published +0.540 is bracketed, nearer 0.5
than 1. `NTC_PinkPEPS` behaves the same way, running -0.0344 to +2.0905 and
bracketing its own published +0.640.

On `metal-plates-013` there is nothing to fix. `NTC_PEPS` already leads
`BI-Grid` by +1.93 dB under the frozen reduction, well past the published
+0.540. So the reproduction's reversal is itself material-specific, and the
Table 2 aggregate reverses because materials like `paving-stones-070` drag it
there.

## The reduction's differential effect changes sign between materials

What each method gains going from exponent 0 to exponent 1:

| instance | `BI-Grid` | `NTC_N` | `NTC_PEPS` | favours |
| --- | ---: | ---: | ---: | --- |
| paving-stones-070 | +1.865 | +1.309 | **+3.941** | the NTC side |
| metal-plates-013 | **+1.768** | +1.295 | +1.186 | `BI-Grid` |

This is the substantive finding, and it is narrower than it first appeared. The
reduction does not uniformly favour PEPS. On `paving-stones-070` it hands
`NTC_PEPS` an extra 2.08 dB over `BI-Grid`; on `metal-plates-013` it hands
`BI-Grid` an extra 0.58 dB over `NTC_PEPS`. **No single exponent fixes the
ordering everywhere.**

The reading consistent with both, and with the loss contrast in
`../budget_probe/`, is that per-map normalisation only helps where a global
reduction was hiding advantage in smooth maps. `metal-plates-013` has no such
hidden advantage: PEPS already leads there by a wide margin.

## What this does and does not establish

It establishes **sufficiency where the reversal occurs**. On a material whose
reproduced ordering contradicts the paper, an unreported detail one level below
the published recipe is enough on its own to restore the published ordering and
to bracket its margin. No implementation difference is required.

It does **not** establish that the paper used such a reduction, and it does not
offer a single reduction that would repair Table 2 as a whole. Sufficiency is
not necessity, the effect reverses sign across the two materials probed, and a
single-material number cannot be compared directly with +0.540, which is a mean
over eighteen materials and three seeds.

## Files

| file | contents |
| --- | --- |
| `receipt.json` | ladders, bracketing tests, differential gains, coverage, limitations |
| `ladder.csv` | one row per instance, reduction and method |

`receipt.json` lists outstanding observations under `coverage.pending`, so a
partially populated ladder stays visible rather than being silently averaged.
The `sqrt_map_l1` row for `metal-plates-013` is among them.

## Status

`bounded_ordering_probe_not_paper_comparable`, `paper_exact` false. Two
materials, one seed, one budget, and reductions of our own construction.
`paving-stones-070` ranks 8th of 18 on the PEPS advantage and
`metal-plates-013` ranks 16th, so neither is representative.
