# Texture budget probe

Bounded diagnostic for one question: is the shortfall between our reproduced
PEPS gain on the NTC pipeline (+1.152 dB) and the published gain (+1.59 dB)
caused by training budget?

The answer is no. The gain is dominated by the training loss, which the paper
does not report.

This is not a Table 2 re-run, not paper-exact, and not a method-level claim.

## Design

`NTC_N` and `NTC_PEPS` are trained at 120,000, 240,000 and 480,000 optimizer
steps. The 120k row is read from the committed Table 2 evidence rather than
re-run. Each larger budget uses its own full cosine schedule, so every point is
a clean run at that budget instead of a re-horizoned continuation. One set and
seed is additionally repeated under a per-map normalised L1 loss.

Probes ran in disposable clones. The budget arms changed only the config and
left the texture code digest untouched; the loss arm also patches
`peps/train.py`, which changes the code digest and so lands in its own output
namespace. The committed Table 2 evidence was never affected.

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

## The loss is the explanation

Table 2 trains with a single global L1 over all concatenated output channels.
Because every map occupies exactly three channels, that loss gives each map an
equal coefficient on its mean absolute error; it is blind to how large each
map's error is. A per-map normalised L1 divides each map's term by its own
detached magnitude, so maps with small residual error receive proportionally
larger gradient.

Swapping only the loss, on the same set, seed, architecture and budget:

| budget | gap under global L1 | gap under per-map normalised L1 | ratio |
| --- | --- | --- | --- |
| 240,000 | +0.5601 | +3.1919 | 5.70x |
| 480,000 | +0.4497 | +3.5334 | 7.86x |

The budget trend also reverses: under the global loss the advantage shrinks
with more compute, under the per-map loss it grows.

The loss decides where the advantage appears. At 480k, PEPS minus `NTC_N` per
map:

| map | global L1 | per-map normalised L1 |
| --- | --- | --- |
| AO | +0.89 | +14.20 |
| DIFF | +1.82 | -0.69 |
| Displacement | -0.19 | +3.83 |
| normal | -0.22 | +0.58 |
| rough | -0.06 | -0.26 |

PEPS is far stronger on the smooth maps, and the loss decides whether those
maps receive optimisation pressure.

## Why this matters for the reproduction

Table 2 reports a per-map average of PSNR, which is a relative, log-domain
quantity per map, while the frozen recipe optimises a single absolute global
L1. Aligning the loss with the metric raises both methods on the probed set,
`NTC_N` from 34.16 to 35.69 and `NTC_PEPS` from 34.61 to 39.22, and the
published gain of +1.59 dB falls between our two variants. The reproduction
shortfall in the PEPS effect is therefore best explained as an unreported
loss or normalisation choice, not as insufficient compute.

## Correction

An earlier revision of this file claimed that under the global L1 "maps with
larger absolute error dominate the gradient". That was wrong. With equal
channel counts per map the global L1 weights every map's mean absolute error
equally. The measured per-map shifts above are what corrected it.

## Outputs

- `receipt.json`: design, per-budget curves per loss, the matched loss
  contrast, per-map advantages, and limitations.
- `curves.csv`: one row per loss/set/seed/budget with overall and per-map
  advantage.

## Limitations

Two texture sets and at most two seeds. The loss contrast covers one set and
one seed at two budgets. The per-map normalised loss is our own construction,
not a published recipe, so it demonstrates sensitivity rather than recovering
the paper's protocol.
