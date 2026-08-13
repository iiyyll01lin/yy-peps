# Probe configs recovered from the sandbox clones (2026-08-14)

These are the runnable configs behind the probe results already committed
under `results/texture_repro/{budget,ordering}_probe/`. The receipts there
record the *semantics* of each run (loss family, budget, seeds, instances);
these files are what you actually feed the trainer.

Until now they existed only as uncommitted edits inside 14 throwaway clones
(`~/yy/workspace/peps-*`), so a `git checkout --` in any of them would have
destroyed the ability to re-run that point. That is why they are here.

| file | family | loss | epochs | seed | sha256[:12] | recovered from |
|---|---|---|---|---|---|---|
| `gap240_l1_s0.toml` | texture-gap-probe-240k | `l1` | 6000 | 0 | `25dc7db8dce5` | gapprobe240 |
| `gap240_l1_s1.toml` | texture-gap-probe-240k | `l1` | 6000 | 1 | `f80343404745` | lr-paving-s1-global |
| `gap240_per_map_l1_s0.toml` | texture-gap-probe-240k | `per_map_l1` | 6000 | 0 | `1292f13d6e0c` | lossprobe240, lr-metal-s0 |
| `gap240_per_map_l1_s1.toml` | texture-gap-probe-240k | `per_map_l1` | 6000 | 1 | `3ba85d03ae9c` | lr-paving-s1 |
| `gap240_range_map_l1_s0.toml` | texture-gap-probe-240k | `range_map_l1` | 6000 | 0 | `9d2172dbb0ac` | lr-range-s0 |
| `gap240_sqrt_map_l1_s0.toml` | texture-gap-probe-240k | `sqrt_map_l1` | 6000 | 0 | `194b0255584f` | lr-sqrt-s0 |
| `gap480_l1_s0.toml` | texture-gap-probe-480k | `l1` | 12000 | 0 | `76e1fa159d61` | gapprobe--gap_probe_480k, gapprobe, gapprobe480m |
| `gap480_l1_s1.toml` | texture-gap-probe-480k | `l1` | 12000 | 1 | `13413001da67` | gapprobe480s1 |
| `gap480_per_map_l1_s0.toml` | texture-gap-probe-480k | `per_map_l1` | 12000 | 0 | `01ee3de1cb86` | lossprobe |
| `rank240_l1_s0.toml` | texture-ranking-probe-240k | `l1` | 6000 | 0 | `0ee0e7e55da4` | rank-global |
| `rank240_per_map_l1_s0.toml` | texture-ranking-probe-240k | `per_map_l1` | 6000 | 0 | `3a93be30e884` | rank-permap |
| `rank240_sqrt_map_l1_s0.toml` | texture-ranking-probe-240k | `sqrt_map_l1` | 6000 | 0 | `d41286cb2edd` | rank-sqrt |

12 unique configs recovered from 15 files.

Note: duplicates collapsed here were byte-different but line-for-line
identical (CRLF vs LF), so they are the same experiment, not two.
