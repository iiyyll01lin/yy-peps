# Texture budget probe

Bounded diagnostic for one question: is the shortfall between our reproduced
PEPS gain on the NTC pipeline (+1.152 dB) and the published gain (+1.59 dB)
caused by training budget?

The answer is no. Compute makes the gain smaller, not larger. The quantity
that does move it is how the training loss is reduced across a set's maps.

This is not a Table 2 re-run, not paper-exact, and not a method-level claim.

## What the paper does and does not fix

The paper reports the Table 2 recipe: GELU, L1, grid learning rate 0.1, MLP
learning rate 0.001, cosine decay, and 3,000 x 40 batches of 60,000 pixel
locations. Table 2 here uses exactly that, so the loss family is not a
deviation and not a candidate explanation.

What "L1" does not pin down is the reduction across maps. A set carries five
to eight maps, each decoded as three output channels. Table 2 reduces one L1
globally over all concatenated channels, which weights every map's mean
absolute error equally. Reducing per map and normalising each by its own
magnitude is an equally literal reading of "L1" and is what this probe tests.

## Budget is not the explanation

Absolute PSNR is budget-limited: both methods keep improving as the budget
grows, so part of the roughly 1.15 dB absolute deficit against the published
values is under-training at 120k.

The PEPS advantage is not. It shrinks monotonically instead of growing toward
the published gap, on two sets and two seeds:

| set | seed | 120k | 240k | 480k | change |
| --- | --- | --- | --- | --- | --- |
| paving-stones-070 | 0 | +0.6588 | +0.5601 | +0.4497 | -0.2092 |
| paving-stones-070 | 1 | +0.7861 | | +0.6457 | -0.1404 |
| metal-plates-013 | 0 | +2.5832 | +2.3199 | +2.0493 | -0.5339 |

Seed spread of the 120k advantage on `paving-stones-070` is about 0.07 dB, so
the shrinkage is well outside seed noise. Extra compute lifts the baseline and
the PEPS variant together, and the baseline lifts faster, so the published
top-line ordering cannot be recovered by training longer.

## The reduction across maps moves the result by about eight times

Changing only the reduction, on the same set, seed, architecture and budget:

| budget | gap under global L1 | gap under per-map normalised L1 | ratio |
| --- | --- | --- | --- |
| 240,000 | +0.5601 | +3.1919 | 5.70x |
| 480,000 | +0.4497 | +3.5334 | 7.86x |

The budget trend also reverses: under the global reduction the advantage
shrinks with more compute, under the per-map one it grows.

The reduction decides where the advantage appears. At 480k, PEPS minus
`NTC_N` per map:

| map | global L1 | per-map normalised L1 |
| --- | --- | --- |
| AO | +0.89 | +14.20 |
| DIFF | +1.82 | -0.69 |
| Displacement | -0.19 | +3.83 |
| normal | -0.22 | +0.58 |
| rough | -0.06 | -0.26 |

PEPS is far stronger on the smooth maps, and the reduction decides whether
those maps receive optimisation pressure.

## Why this matters for the reproduction

Table 2 reports a per-map average of PSNR, a relative log-domain quantity per
map, while the recipe optimises one absolute global L1. Aligning the reduction
with the metric raises both methods on the probed set, `NTC_N` from 34.16 to
35.69 and `NTC_PEPS` from 34.61 to 39.22, and the published gain of +1.59 dB
falls between our two variants.

This does not prove which reduction the authors used, and it is not a claim
that Table 2 here is mis-specified: every one of the 594 jobs used the same
global L1, so the method comparison is internally fair. It shows that a
detail the paper leaves open is a larger lever on the headline margin than
any budget change measured here.

## Corrections

Two earlier revisions of this file were wrong and are retracted.

The first claimed that under the global L1 "maps with larger absolute error
dominate the gradient". With equal channel counts per map the global L1
weights every map's mean absolute error equally.

The second claimed the paper "does not report" its texture loss. It does
report L1; what it leaves unspecified is the reduction across maps.

## Outputs

- `receipt.json`: design, per-budget curves per reduction, the matched
  contrast, per-map advantages, and limitations.
- `curves.csv`: one row per loss, set, seed and budget with overall and
  per-map advantage.

## Limitations

Two texture sets and at most two seeds. The contrast covers one set and one
seed at two budgets. The per-map normalised L1 is our own construction, so
this demonstrates sensitivity rather than recovering the authors' choice.
