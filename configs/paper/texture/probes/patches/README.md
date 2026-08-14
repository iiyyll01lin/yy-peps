# Probe patches — the code the probe configs need

The configs in the parent directory name losses that `peps/train.py` does not
implement: `per_map_l1`, `sqrt_map_l1`, `range_map_l1`. Running them against
the repository as it stands fails at config validation. These two patches are
the working-tree changes that existed only inside the throwaway probe clones,
recovered on 2026-08-14 before those clones were deleted.

| patch | implements | used by |
|---|---|---|
| `train_map_losses_full.patch` | `_map_groups`, `per_map_l1_loss`, `sqrt_map_l1_loss`, `range_map_l1_loss` | `lr-metal-s0`, `lr-paving-s1`, `lr-range-s0`, `lr-sqrt-s0`, `rank-permap`, `rank-sqrt` |
| `train_per_map_l1_early.patch` | `per_map_l1_loss` only | `lossprobe`, `lossprobe240` |

## The two are not the same function

This is the part that matters for reading the results. Both patches define
`per_map_l1_loss`, and they are **different implementations**, not a refactor:

- the early one divides each map group by its own detached magnitude to stop
  the largest-error maps dominating the gradient;
- the full one frames the same division as equalising *relative* progress,
  which deliberately hands more gradient to maps that are already accurate,
  and states that this is the reduction matching the per-map PSNR metric.

So a number produced by `lossprobe` and a number produced by `rank-permap`
were not produced by the same loss, even though both configs say
`loss = "per_map_l1"`. Do not pool them.

## Status

These are **archived working-tree diffs, not merged code**. Nothing in the
repository applies them. If the map-family losses are wanted for real, the
full patch is the starting point and needs the usual treatment: tests, a
decision about which `per_map_l1` definition survives, and a note in the
receipts about which one produced which published number.
