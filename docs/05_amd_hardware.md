# Part V — AMD hardware (W11–W12) / AMD 硬體

## W11 · From PyTorch to HIP / 從 PyTorch 到 HIP

**English.** Parts I–IV run through PyTorch's ROCm backend. Real-time texture
decode needs custom kernels. HIP is AMD's CUDA-like C++ dialect; `hipcc` compiles
for a specific arch (`--offload-arch=gfx1201` RDNA4, `gfx1151` RDNA3.5). We fuse
the PEPS inference inner loop — bilinear grid sample + first MLP layer — into one
kernel (`hip/fused_peps_kernel.hip`). Fusion keeps latents in registers instead of
round-tripping through global memory. Measured on Box A (RDNA3.5): 2.36 ms/iter for
262k points.

**繁體中文.** Part I–IV 走 PyTorch 的 ROCm 後端。即時材質解碼需要自訂 kernel。HIP 是
AMD 類 CUDA 的 C++ 方言;`hipcc` 針對特定架構編譯。我們把 PEPS 推論內迴圈(雙線性
grid 取樣 + MLP 第一層)融合成單一 kernel。融合讓 latent 留在暫存器,不必經全域記憶體
來回。Box A(RDNA3.5)實測:262k 點 2.36 ms/iter。

## W12 · RDNA4 WMMA — matrix acceleration / RDNA4 WMMA 矩陣加速

**English.** The MLP decoder is a stack of small matmuls — ideal for **WMMA** (Wave
Matrix Multiply-Accumulate). `hip/wmma_mlp.hip` uses rocWMMA with 16×16×16 FP16
tiles and FP32 accumulate. The **same code** compiles for RDNA3.5 (`gfx1151`) and
RDNA4 (`gfx1201`); rocWMMA selects the intrinsic per arch. Correctness is checked
against a CPU reference (matches). The paper's ms figures target RDNA4, so the true
comparison runs on **Box B**; RDNA3.5 (Box A) is a second data point.

Measured MLP-layer latency (16×16×16 WMMA, 4096×64×64):
- **RDNA3.5 (gfx1151, Box A): ~0.01 ms/iter** (verified end-to-end)
- **RDNA4 (gfx1201, Box B): run `bash hip/bench_latency.sh` on Box B**

**繁體中文.** MLP 解碼器是一疊小矩陣乘,正適合 **WMMA**。`hip/wmma_mlp.hip` 用 rocWMMA
的 16×16×16 FP16 tile + FP32 累加。**同一份程式碼**可為 RDNA3.5 與 RDNA4 編譯;rocWMMA
依架構選 intrinsic。正確性對 CPU 參考驗證(相符)。論文 ms 數字針對 RDNA4,故真正對照
在 **Box B**;RDNA3.5(Box A)是第二資料點。

## Honest hardware note / 誠實硬體註記

RDNA3.5 and RDNA4 both expose WMMA, but fragment shapes and int8 paths differ; RDNA4
adds the faster path RTXNTC uses. The RDNA3.5 numbers are **not** a reproduction of
the paper's RDNA4 figures — they are a genuine cross-generation comparison the
course can make because it has both boxes.

RDNA3.5 與 RDNA4 都提供 WMMA,但 fragment 形狀與 int8 路徑不同;RDNA4 多了 RTXNTC 用的
快路徑。RDNA3.5 數字**不是**論文 RDNA4 數字的重現,而是本課程因擁有兩台而能做的真正
跨世代對照。
