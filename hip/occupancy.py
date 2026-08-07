#!/usr/bin/env python3
"""Compute occupancy for the fused kernel from the profiler's own records.

The launch asks for 32 KB of LDS per 64-thread workgroup. Whether that is the
limiter depends on the part's LDS capacity and wave slots, both of which
rocprofv3 records in agent_info.csv, so nothing here is assumed.
"""
from __future__ import annotations

import csv
from pathlib import Path

PV3 = Path("/tmp/pv3")

agents = list(csv.DictReader((PV3 / "run1_agent_info.csv").open(encoding="utf-8")))
gpu = next(a for a in agents if a.get("Agent_Type") == "GPU"
           or a.get("Name", "").startswith("gfx"))

lds_kb = int(gpu["Lds_Size_In_Kb"])
cu = int(gpu["Cu_Count"])
simd_per_cu = int(gpu["Simd_Per_Cu"])
waves_per_simd = int(gpu["Max_Waves_Per_Simd"])
waves_per_cu = int(gpu["Max_Waves_Per_Cu"])
wave = int(gpu["Wave_Front_Size"])

print(f"part           : {gpu.get('Name')}")
print(f"compute units  : {cu}")
print(f"LDS per CU     : {lds_kb} KB")
print(f"SIMDs per CU   : {simd_per_cu}, max waves per SIMD {waves_per_simd}")
print(f"max waves / CU : {waves_per_cu}   wavefront {wave}")

trace = list(csv.DictReader((PV3 / "run1_kernel_trace.csv").open(encoding="utf-8")))
k = next(r for r in trace if r["Kernel_Name"].startswith("integrated_peps_wmma"))
lds_bytes = int(k["LDS_Block_Size"])
wg = int(k["Workgroup_Size_X"]) * int(k["Workgroup_Size_Y"]) * int(k["Workgroup_Size_Z"])
grid = int(k["Grid_Size_X"]) * int(k["Grid_Size_Y"]) * int(k["Grid_Size_Z"])
vgpr, sgpr = int(k["VGPR_Count"]), int(k["SGPR_Count"])

waves_per_wg = -(-wg // wave)
print(f"\nlaunch         : workgroup {wg} threads = {waves_per_wg} waves, "
      f"grid {grid} threads = {grid // wg} workgroups")
print(f"per workgroup  : LDS {lds_bytes} B ({lds_bytes/1024:.0f} KB), "
      f"VGPR {vgpr}, SGPR {sgpr}, scratch {k['Scratch_Size']}")

by_lds = (lds_kb * 1024) // lds_bytes
by_waves = waves_per_cu // waves_per_wg
by_vgpr = (512 // max(vgpr, 1)) * simd_per_cu // waves_per_wg
limit = min(by_lds, by_waves, by_vgpr)
achieved = limit * waves_per_wg

print(f"\nworkgroups per CU allowed by LDS   : {by_lds}")
print(f"workgroups per CU allowed by waves : {by_waves}")
print(f"workgroups per CU allowed by VGPRs : {by_vgpr}")
print(f"limiter                            : "
      f"{'LDS' if by_lds == limit else 'waves' if by_waves == limit else 'VGPRs'}")
print(f"\nwaves resident per CU : {achieved} of {waves_per_cu} "
      f"= {100 * achieved / waves_per_cu:.1f}% occupancy")
print(f"\nHalving LDS to {lds_bytes//2} B would allow {(lds_kb*1024)//(lds_bytes//2)} "
      f"workgroups per CU, i.e. "
      f"{100 * min((lds_kb*1024)//(lds_bytes//2), by_waves, by_vgpr) * waves_per_wg / waves_per_cu:.1f}% "
      "occupancy, if the other limits allow.")
