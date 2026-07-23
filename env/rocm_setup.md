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
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r env/torch-rocm70.txt \
  --index-url https://download.pytorch.org/whl/rocm7.0
python -m pip install -r env/requirements.txt -c env/constraints.txt
python -m pip install -e . --no-deps
```

The checked-in ROCm wheel file currently selects PyTorch `2.10.0+rocm7.0` and
torchvision `0.25.0+rocm7.0`. Do not mix CPU and ROCm wheel indexes in one
environment. Capture `python -m pip freeze --all` with every experiment.

## Version policy / 版本策略

Keep the Python and native HIP stacks isolated:

- `.venv` uses the checked-in PyTorch `2.10.0+rocm7.0` wheel. The wheel bundles
  its ROCm user-space runtime, so do not point `LD_LIBRARY_PATH` at system ROCm
  when running Python.
- Native kernels use the compiler, headers, and libraries from one
  `ROCM_HOME` (normally `/opt/rocm`). Do not fall back to an older distro
  `/usr/bin/hipcc` when an active ROCm tree is present.
- Keep a working distro/kernel `amdgpu` driver unless the selected ROCm release
  explicitly requires a driver change. Matching a kernel module's package
  version to a bundled PyTorch runtime is neither required nor safe.

This policy deliberately preserves the validated PyTorch environment while
preventing 5.x and 7.x native tools from being mixed in one build.

## Four-GPU PCIe RCCL / 四卡 PCIe RCCL

On the four-card `gfx1201` host with a mainline kernel, scope these variables to
the multi-GPU command:

```bash
HSA_ENABLE_IPC_MODE_LEGACY=0 \
HSA_FORCE_FINE_GRAIN_PCIE=1 \
python -m experiments.multigpu suite --gpus 4 --rccl-p2p on
```

`HSA_ENABLE_IPC_MODE_LEGACY=0` selects ROCr's dma-buf IPC implementation;
without it, RCCL can fail in `hipIpcGetMemHandle` even when `iommu=pt` is active.
Keep `--rccl-p2p on` during validation so a direct-transport failure exits
instead of silently falling back. `NCCL_DMABUF_ENABLE` configures a different
RCCL network path and does not replace the ROCr IPC setting above. Do not export
the experimental IPC mode globally; keep it attached to the tested command.

## Verify / 驗證
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
"${ROCM_HOME:-/opt/rocm}/bin/offload-arch"   # list GPU ISAs / 列出 GPU ISA
```

PyTorch intentionally exposes ROCm devices through the `torch.cuda` API.

## Per-box hardware / 各機硬體
| Box | GPU | `gfx_target_version` | ISA |
|---|---|---|---|
| A | Radeon 8060S (Strix Halo) | gfx1151 | RDNA 3.5 |
| B | 4× Navi 48 | gfx1201 | RDNA 4 |

## HIP / hipcc (Part V)
Native HIP builds require one complete ROCm development tree. Some ROCm 7.2
installs expose `amdclang++` without a `hipcc` wrapper, so select the compiler
from the active tree rather than whichever `hipcc` appears first on `PATH`:
```bash
ROCM_HOME=${ROCM_HOME:-/opt/rocm}
if [[ -x "$ROCM_HOME/bin/hipcc" ]]; then
  HIP_CXX=("$ROCM_HOME/bin/hipcc")
  HIP_LINK=()
elif [[ -x "$ROCM_HOME/bin/amdclang++" ]]; then
  HIP_CXX=("$ROCM_HOME/bin/amdclang++" -x hip)
  HIP_LINK=(-L"$ROCM_HOME/lib" -lamdhip64)
else
  echo "No HIP compiler under $ROCM_HOME" >&2
  exit 1
fi
export PATH="$ROCM_HOME/bin:$PATH"
export CPLUS_INCLUDE_PATH="$ROCM_HOME/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
"${HIP_CXX[@]}" --version

# Build for a specific arch:
"${HIP_CXX[@]}" --offload-arch=gfx1201 hip/wmma_mlp.hip "${HIP_LINK[@]}" -o wmma_mlp     # RDNA4 (Box B)
"${HIP_CXX[@]}" --offload-arch=gfx1151 hip/wmma_mlp.hip "${HIP_LINK[@]}" -o wmma_mlp     # RDNA3.5 (Box A)
```

**Note / 注意**: WMMA intrinsics differ between RDNA 3.5 (`gfx1151`) and RDNA 4
(`gfx1201`). The paper's ms latency numbers target RDNA4; RDNA3.5 results are a
separate comparison point, not a reproduction.

WMMA intrinsics 在 RDNA 3.5 與 RDNA 4 間不同。論文延遲數字針對 RDNA4;RDNA3.5
的結果是獨立對照點,而非重現。

## Optional self-hosted CI / 選配自架 CI

`.github/workflows/amd-gpu.yml` is manual-only. Its job runs only when:

1. the workflow-dispatch input `run_gpu_checks` is enabled;
2. repository variable `ENABLE_AMD_GPU_CI` equals `true`; and
3. a runner has labels `self-hosted`, `linux`, `x64`, `amd-gpu`, and `rocm`.

The gate prevents ordinary pull requests from waiting for private hardware.
Runner registration, credentials, and raw datasets are machine administration
concerns and must not be committed.
