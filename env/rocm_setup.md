# ROCm setup notes / ROCm 安裝筆記

Both course machines run **Ubuntu 24.04 + ROCm 7.x**. PyTorch is installed from
the ROCm wheel index, which bundles the ROCm runtime libraries — a full system
ROCm install is **not** strictly required for Parts I–IV, only the `amdgpu`
kernel driver and `/dev/kfd`.

兩台課程機器皆為 **Ubuntu 24.04 + ROCm 7.x**。PyTorch 由 ROCm wheel index 安裝,
wheel 自帶 ROCm runtime,Parts I–IV 只需 `amdgpu` 驅動與 `/dev/kfd`,不必完整系統 ROCm。

## GPU access / GPU 存取
Your user must be in the `render` and `video` groups:
使用者需加入 `render` 與 `video` 群組:
```bash
sudo usermod -aG render,video $USER   # re-login for it to take effect
```

## PyTorch install / 安裝
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0
```

## Verify / 驗證
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
rocm-smi         # list GPUs / 列出 GPU
```

## Per-box hardware / 各機硬體
| Box | GPU | `gfx_target_version` | ISA |
|---|---|---|---|
| A `Box A` | Radeon 8060S (Strix Halo) | 115100? (gfx1151) | RDNA 3.5 |
| B `Box B` | 4× Navi 48 | 120001 (gfx1201) | RDNA 4 |

## HIP / hipcc (Part V)
Box A has system ROCm 7.2.3 with `hipcc`. On Box B, install ROCm HIP dev tools if
building kernels there:
```bash
hipcc --version
# Build for a specific arch:
hipcc --offload-arch=gfx1201 hip/wmma_mlp.hip -o wmma_mlp     # RDNA4 (Box B)
hipcc --offload-arch=gfx1151 hip/wmma_mlp.hip -o wmma_mlp     # RDNA3.5 (Box A)
```

**Note / 注意**: WMMA intrinsics differ between RDNA 3.5 (`gfx1151`) and RDNA 4
(`gfx1201`). The paper's ms latency numbers target RDNA4; RDNA3.5 results are a
separate comparison point, not a reproduction.

WMMA intrinsics 在 RDNA 3.5 與 RDNA 4 間不同。論文延遲數字針對 RDNA4;RDNA3.5
的結果是獨立對照點,而非重現。
