---
marp: true
theme: default
paginate: true
title: PEPS on AMD — Part V: AMD hardware
---

<!--
繁體中文:本檔為 Part V(AMD 硬體,W11–12)的 Marp 投影片。輸出 PDF:
  npx @marp-team/marp-cli slides/05_amd_hardware.md -o slides/05_amd_hardware.pdf
或於 slides/ 執行 `make 05_amd_hardware.pdf`。每張投影片:英文標題 + 繁中要點。
來源:docs/05_amd_hardware.md、results/hip_latency.csv、hip/README.md。
-->

# Part V — AMD hardware
## PyTorch → HIP → RDNA4 WMMA
### W11–W12 · 在真實 AMD 晶片上落地

The course ends where the paper's hardware story begins.
課程在論文硬體故事開始之處結束。

> Current latency CSV: **legacy-unverified** · 目前延遲 CSV 尚未驗證

---

# Two AMD boxes / 兩台 AMD

| Box | GPU | ISA | Role |
|---|---|---|---|
| **B** | 4× Navi 48 | `gfx1201` / RDNA 4 | the paper's target |
| **A** | Radeon 8060S | `gfx1151` / RDNA 3.5 | cross-gen comparison |

Sources target both ISAs; the gated self-hosted workflow must verify each toolchain.
程式碼針對兩種 ISA;需由 gated self-hosted workflow 驗證。

---

# W11 · PyTorch → HIP / 從 PyTorch 到 HIP

- Parts I–IV run through PyTorch's ROCm backend; real-time decode needs **custom kernels**
- HIP = AMD's CUDA-like C++; `hipcc --offload-arch=gfx1201` (RDNA4) / `gfx1151` (RDNA3.5)
- Integrated kernel: projection → all shared-grid samples → aggregation → full MLP

<br>

- 即時解碼需**自訂 kernel**;HIP 是 AMD 類 CUDA 的 C++
- 整合 projection → 所有 shared-grid samples → aggregation → 完整 MLP

---

# W11 · Three parity modes / 三種對拍 mode

- `baseline`: original coordinate → one grid sample
- `peps`: `(x,S₁..S₃,C₁..C₃)` → concat
- `pink`: same points → paper-exact circular channel slices
- Every mode executes **three hidden layers + output**
- Fixtures pass for both **scalar fp32 and fused fp16/rocWMMA**

<br>

- baseline / concat / Pink 都有 binary fixture
- 每種 mode 都執行**三個 hidden layers + output**

---

# W11 · Current integrated rerun / 最新整合重跑

| mode | implementation | Box B ms/iter |
|---|---|---:|
| baseline | scalar fp32 | **246.3032** |
| concat PEPS | scalar fp32 | **411.7999** |
| Pink PEPS | scalar fp32 | **295.3217** |

ROCm 7.2.3 · 1024² RGB · 20 iterations · parity passed.
These correctness-reference rows are **not paper-comparable**.

正確性參考實測,`comparable_to_paper=false`,不可冒充論文 4–5 ms 數字。

---

# W12 · RDNA4 WMMA / RDNA4 WMMA

- The MLP decoder is a stack of small matmuls — ideal for **WMMA**
  (Wave Matrix Multiply-Accumulate)
- `hip/wmma_mlp.hip`: rocWMMA, **16×16×16 FP16 tiles**, **FP32 accumulate**
- The **same code** compiles for RDNA3.5 & RDNA4; rocWMMA picks the intrinsic per arch

<br>

- MLP 解碼器是一疊小矩陣乘,正適合 **WMMA**;rocWMMA 16×16×16 FP16 tile + FP32 累加
- **同一份程式碼**可為兩世代編譯

---

# W12 · Diagnostics stay diagnostics / 診斷仍是診斷

- Old sample + first-layer timing is retained as `supplementary_microbenchmark`
- Isolated fp16/int8 rocWMMA GEMMs are also supplementary
- Neither includes the complete projection/sampling/aggregation/MLP workload
- Result schema fixes `comparable_to_paper=false`

舊 first-layer 與 isolated WMMA 數字保留,但不可當作 paper-workload latency。

Current Box B 4096×64×64: **15.0896 ms fp16**, **15.4592 ms int8**;
both pass parity. Large 2048³ and Box A rows remain `legacy_reported`.

---

# Cross-generation, not reproduction / 跨世代,非重現

> Legacy RDNA3.5 rows are **not** a reproduction of the paper's RDNA4 figures
> and are not a verified cross-generation claim.

> legacy RDNA3.5 列**不是**論文 RDNA4 重現,也不是已驗證跨世代結論。

---

# The paper comparison gate / 論文比較門檻

- Optimize the integrated fp16/WMMA path (all-mode parity already passes)
- Match precision, activation/bias handling, fusion, and timing boundaries
- Record compiler/warmup/timing provenance
- Measure on target RX 9070 XT (`gfx1201`)

<br>

- integrated fp16/WMMA parity 已通過,下一步是效能最佳化
- precision、activation/bias、fusion、timing boundary 必須配對
- 記錄 compiler、warmup、timing provenance 後才可比較

---

# Honest limitations / 誠實的限制

1. Current measured integrated rows are scalar fp32
2. Current and legacy component rows are diagnostics, not generation throughput
3. Safe fp16 preflight projected **280.6 s** for only 1 warmup + 2 iterations; no row

<br>

1. 目前整合實測是 scalar fp32
2. 現有實測列為 component diagnostics
3. fp16 safety preflight 即預估 **280.6 秒**,因此未寫入結果列

---

# <!-- fit --> 完成 / The full loop

### paper → PyTorch → quantization → HIP/WMMA — on real AMD silicon
### Next: W13–W14 capstone — `docs/06_capstone.md`
