# Texture budget probe

Bounded diagnostic for one question: is the shortfall between our reproduced
PEPS gain on the NTC pipeline (+1.152 dB) and the published gain (+1.59 dB)
caused by training budget?

This is not a Table 2 re-run, not paper-exact, and not a method-level claim.
It covers two texture sets and up to two seeds.

## Design

`NTC_N` and `NTC_PEPS` are trained at 120,000, 240,000 and 480,000 optimizer
steps. The 120k row is read from the committed Table 2 evidence rather than
re-run. Each larger budget uses its own full cosine schedule, so every point is
a clean run at that budget instead of a re-horizoned continuation.

The probes ran in disposable clones whose `configs/paper/texture_full.toml` was
replaced. The texture code digest was unchanged, so the committed Table 2
output namespace was never touched.

## Result

Two effects separate cleanly.

Absolute PSNR is budget-limited. Both methods keep improving as the budget
grows, so part of the roughly 1.15 dB absolute deficit against the published
values is under-training at 120k.

The PEPS advantage is not budget-limited. It shrinks monotonically instead of
growing toward the published gap:

| set | 120k | 240k | 480k | change |
| --- | --- | --- | --- | --- |
| paving-stones-070 | +0.6588 | +0.5601 | +0.4497 | -0.2092 |
| metal-plates-013 | +2.5832 | +2.3199 | pending | -0.2633 so far |

Seed spread of the 120k advantage on paving-stones-070 is about 0.07 dB, and
seed 0 happens to have the smallest 120k advantage, so the measured shrinkage
is larger than seed noise.

Extra compute lifts the baseline and the PEPS variant together, and the
baseline lifts faster. The published top-line ordering therefore cannot be
recovered by training longer.

## Mechanism

Per-map values show where the advantage is lost, and it differs by set.

On `paving-stones-070` the whole advantage sits in the colour map. `NTC_PEPS`
colour PSNR moves 30.53 to 31.15 between 120k and 240k and then stops at 31.15
through 480k, while the baseline keeps climbing from 27.96 to 29.33. Every
other map of the same `NTC_PEPS` run continues to improve over that interval,
so the plateau is specific to colour rather than a stalled run.

On `metal-plates-013` the advantage comes from displacement and ambient
occlusion instead, and the colour map is already slightly worse than the
baseline at 120k.

Training uses a single global L1 loss with no per-map weighting, so maps with
larger absolute error dominate the gradient. Once the PEPS colour residual
becomes small, optimisation pressure moves to the harder maps and colour
stalls; the baseline, whose colour error is still large, keeps improving and
closes the gap. The paper does not report its texture loss, so a different
loss or per-map normalisation is a plausible protocol difference and is a
candidate follow-up.

## Outputs

- `receipt.json`: design, per-budget curves, per-map advantages, limitations.
- `curves.csv`: one row per set/seed/budget with overall and per-map advantage.

## Regenerate

The probe clones are disposable and are not part of this repository. The
receipt is regenerated from Table 2 plus whichever probe clones exist.
