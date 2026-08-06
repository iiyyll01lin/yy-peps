# Is the missing PEPS margin a budget problem?

Table 2 reproduces the paper's qualitative pattern but the `NTC_PEPS` minus
`NTC_N` gain lands at +1.152 dB against the published +1.59. These probes test
the two obvious explanations: not enough training, or the wrong loss.

Both probes ran in disposable clones. The budget probes changed only the
config; the loss probes also patched `peps/train.py`, which changes the code
digest and therefore lands their output in a separate namespace. Committed
Table 2 evidence was never touched.

## Budget is not the cause

More compute makes the advantage *smaller*, on both materials and both seeds.

| set | seed | 120k | 240k | 480k |
| --- | ---: | ---: | ---: | ---: |
| paving-stones-070 | 0 | +0.6588 | +0.5601 | +0.4497 |
| paving-stones-070 | 1 | +0.7861 | +0.6677 | +0.6457 |
| metal-plates-013 | 0 | +2.5832 | +2.3199 | +2.0493 |

Every curve decreases monotonically. Quadrupling the schedule from 120k to
480k steps costs `paving-stones-070` seed 0 about 0.21 dB of advantage. No
budget within reach closes a 0.44 dB gap that is moving the wrong way.

## The loss family is not the cause either

The paper reports the Table 2 recipe, L1 included, and this reproduction trains
with L1. `docs/03_applications.md` records that only the optimizer and the
three seeds are local assumptions.

## What the paper does leave open: the reduction across maps

A set carries five to eight maps, each decoded as exactly three channels. Table
2 reduces one L1 globally over all concatenated channels, which weights every
map's mean absolute error equally. Reducing per map and normalising each term
by its own detached magnitude is an equally literal reading of "L1", and it
hands proportionally more gradient to maps that are already accurate.

### The effect is a dose response, not a quirk of one implementation

On `paving-stones-070` seed 0 at 240k steps, varying only the exponent applied
to each map's own error before dividing:

| reduction | exponent | NTC_N | NTC_PEPS | gap |
| --- | ---: | ---: | ---: | ---: |
| `global_l1` | 0.0 | 33.9101 | 34.4702 | +0.5601 |
| `range_map_l1` | static | 33.7950 | 34.4309 | +0.6359 |
| `sqrt_map_l1` | 0.5 | 34.8824 | 36.1223 | +1.2399 |
| `per_map_normalised_l1` | 1.0 | 35.2192 | 38.4111 | +3.1919 |

The gap is monotone in the normalisation strength. `range_map_l1` is the
informative negative: it divides each map by its target's dynamic range rather
than by its current error, a static reweighting, and it barely moves the result.
So the mechanism is adaptive, error-dependent weighting, not putting the maps
on a common scale.

### It is seed-stable within a material and absent on another

| set | seed | steps | global | per-map | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| paving-stones-070 | 0 | 240k | +0.5601 | +3.1919 | 5.70x |
| paving-stones-070 | 1 | 240k | +0.6677 | +3.1784 | 4.76x |
| paving-stones-070 | 0 | 480k | +0.4497 | +3.5334 | 7.86x |
| metal-plates-013 | 0 | 240k | +2.3199 | +2.2105 | **0.95x** |

Two seeds on `paving-stones-070` agree closely, so this is not seed noise. But
on `metal-plates-013` the same change does nothing at all. The reading that
fits both is that per-map normalisation only unlocks advantage a global
reduction was hiding in smooth maps, and on `metal-plates-013` PEPS already
leads by a wide margin under the global reduction, so there is nothing left to
unlock.

**The reduction therefore does not explain the Table 2 shortfall.** It is a
real and large uncontrolled degree of freedom, but its size depends on the
material. See `../shortfall_analysis/` for what does account for the uniform
offset: the unpublished map-file selection.

### Where PEPS's advantage moves

`paving-stones-070` seed 0 at 480k, advantage by map:

| map | global L1 | per-map L1 |
| --- | ---: | ---: |
| AO | +0.89 | **+14.20** |
| DIFF | **+1.82** | -0.69 |
| Displacement | -0.19 | +3.83 |
| normal | -0.22 | +0.58 |
| rough | -0.06 | -0.26 |

The reduction decides *where* PEPS helps. Under a global reduction its benefit
shows up in colour; under per-map normalisation it appears overwhelmingly in
the smooth maps.

## Corrections

Two claims in earlier versions of this file were wrong and are retracted here
rather than deleted.

**"Maps with larger absolute error dominate the gradient."** They do not. Every
map occupies exactly three channels, so a global L1 already gives each map's
mean absolute error an equal coefficient. What per-map normalisation changes is
the *relative* weighting, not a pre-existing imbalance.

**"The paper never reports its texture loss."** It does. `docs/03_applications.md`
records that the paper specifies GELU, L1, the two learning rates, cosine decay
and the batch schedule; only the optimizer and the seeds are assumptions. What
is unreported is the reduction across maps, which is what these probes vary.

## Files

| file | contents |
| --- | --- |
| `receipt.json` | design, curves, matched contrasts, reduction ladder, limitations |
| `curves.csv` | one row per loss, set, seed and budget |

## Status

`bounded_budget_probe_not_paper_comparable`, `paper_exact` false. Two
materials, at most two seeds, and reductions of our own construction. These
probes demonstrate sensitivity; they do not recover the paper's protocol.
