#!/usr/bin/env python3
"""Occupancy for the fused kernel, derived from the profiler and then checked.

This model has been wrong twice, and both times the counter caught it.

First it divided the 64 KB a compute unit advertises by the per-workgroup
footprint. Then it divided a 128 KB per-WGP pool by the same footprint. The
three models agree on many footprints, which is how the wrong ones survived:
over seven measured footprints the per-CU model matches five and the plain
per-WGP model matches three. Only one matches all seven.

That one rounds the footprint up to a 1024-byte granule before dividing a
128 KB per-WGP pool by it. It is recorded as the model that fits every
measurement taken, not as a claim about hardware internals read from a
specification, and the two it replaced are still computed below so any
future footprint that separates them shows up immediately.

Point it at a rocprofv3 output directory. If that directory also contains a
counter collection with OccupancyPercent or MeanOccupancyPerCU, the derived
figure is checked against it instead of merely asserted.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

PV3 = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pv3")
KERNEL = "integrated_peps_wmma"
GRANULE = 1024


def load(pattern: str) -> list[dict]:
    matches = sorted(PV3.glob(pattern))
    if not matches:
        return []
    return list(csv.DictReader(matches[0].open(encoding="utf-8")))


agents = load("*agent_info.csv")
gpu = next(a for a in agents if a.get("Agent_Type") == "GPU"
           or a.get("Name", "").startswith("gfx"))

lds_kb = int(gpu["Lds_Size_In_Kb"])
cu = int(gpu["Cu_Count"])
simd_per_cu = int(gpu["Simd_Per_Cu"])
waves_per_cu = int(gpu["Max_Waves_Per_Cu"])
wave = int(gpu["Wave_Front_Size"])

print(f"part           : {gpu.get('Name')}")
print(f"compute units  : {cu}")
print(f"LDS advertised : {lds_kb} KB per CU, so {2 * lds_kb} KB per WGP")
print(f"max waves / CU : {waves_per_cu}   wavefront {wave}")

trace = load("*kernel_trace.csv") or load("*counter_collection.csv")
k = next(r for r in trace if r["Kernel_Name"].startswith(KERNEL))
lds_bytes = int(k["LDS_Block_Size"])
if "Workgroup_Size_X" in k:
    wg = (int(k["Workgroup_Size_X"]) * int(k["Workgroup_Size_Y"])
          * int(k["Workgroup_Size_Z"]))
else:
    wg = int(k["Workgroup_Size"])
vgpr, sgpr = int(k["VGPR_Count"]), int(k["SGPR_Count"])
waves_per_wg = -(-wg // wave)

print(f"\nlaunch         : workgroup {wg} threads = {waves_per_wg} waves")
print(f"per workgroup  : LDS {lds_bytes} B, VGPR {vgpr}, SGPR {sgpr}, "
      f"scratch {k['Scratch_Size']}")

pool = 2 * lds_kb * 1024
effective = -(-lds_bytes // GRANULE) * GRANULE
by_lds_wgp = pool // effective
waves_by_lds = by_lds_wgp * waves_per_wg / 2
by_waves = waves_per_cu // waves_per_wg
by_vgpr = (512 // max(vgpr, 1)) * simd_per_cu // waves_per_wg

limits = {"LDS": waves_by_lds,
          "wave slots": by_waves * waves_per_wg,
          "VGPRs": by_vgpr * waves_per_wg}
limiter = min(limits, key=limits.get)
achieved = limits[limiter]
derived = 100 * achieved / waves_per_cu

if effective != lds_bytes:
    print(f"\nLDS {lds_bytes} B rounds up to {effective} B at a "
          f"{GRANULE}-byte granule")
print(f"\nresident waves per CU allowed by LDS        : {waves_by_lds:g} "
      f"({by_lds_wgp} workgroups per WGP)")
print(f"resident waves per CU allowed by wave slots : {by_waves * waves_per_wg}")
print(f"resident waves per CU allowed by VGPRs      : {by_vgpr * waves_per_wg}")
print(f"limiter                                     : {limiter}")
print(f"\nderived occupancy : {derived:.2f}%  "
      f"({achieved:g} of {waves_per_cu} waves)")

superseded = {
    "64 KB per-CU pool": 100 * ((lds_kb * 1024) // lds_bytes)
    * waves_per_wg / waves_per_cu,
    "128 KB per-WGP pool, no granule": 100 * (pool // lds_bytes)
    * waves_per_wg / 2 / waves_per_cu,
}
for label, value in superseded.items():
    if abs(value - derived) > 0.01:
        print(f"a {label} would have said {value:.2f}% -- "
              "this footprint separates it from the model above")

counters = load("*counter_collection.csv")
samples: dict[str, list[float]] = {}
for row in counters:
    if not row["Kernel_Name"].startswith(KERNEL):
        continue
    samples.setdefault(row["Counter_Name"], []).append(float(row["Counter_Value"]))

if not samples:
    print("\nno counter collection here; the figure above is derived, not measured")
    raise SystemExit(0)

print(f"\n{'counter':<22}{'n':>4}{'min':>10}{'mean':>10}{'max':>10}{'derived':>10}")
for name, values in sorted(samples.items()):
    expected = derived if name == "OccupancyPercent" else achieved
    mean = sum(values) / len(values)
    print(f"{name:<22}{len(values):>4}{min(values):>10.2f}{mean:>10.2f}"
          f"{max(values):>10.2f}{expected:>10.2f}")
    # A counter disagreeing with the arithmetic means one of them is wrong,
    # and on this kernel it has so far been the arithmetic.
    if abs(mean - expected) > 0.5:
        print(f"  MISMATCH: {name} is {mean:.2f}, arithmetic says {expected:.2f}")
