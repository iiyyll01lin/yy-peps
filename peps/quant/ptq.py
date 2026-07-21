"""int8 post-training quantization for latents + MLP weights (W10).

繁體中文:int8 事後量化(PTQ)。對 grid latent 與 MLP 權重做對稱 int8 量化,
用來檢驗 PEPS 的參數優勢在量化後是否仍成立。純 PyTorch、可在 ROCm 上重現。
提供:
- quantize_tensor / dequantize_tensor:對稱線性量化。
- quantize_grid_encoder / quantize_mlp:就地量化模型元件並回傳「有效位元率」。
- effective_bits:估計每參數平均位元(latent+weights 混合)。
量化以「模擬」方式進行(存 int8 + scale,前向時反量化),量測品質退化而非真加速。
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class QuantResult:
    bits: int
    n_params: int


def quantize_tensor(x: torch.Tensor, bits: int = 8):
    """Symmetric per-tensor linear quantization. Returns (q_int, scale)."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max().clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q.to(torch.int8 if bits <= 8 else torch.int16), scale


def dequantize_tensor(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale


def fake_quantize(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Quantize then dequantize (simulated quantization) in one call."""
    q, s = quantize_tensor(x, bits)
    return dequantize_tensor(q, s)


@torch.no_grad()
def quantize_module_params(module: nn.Module, bits: int = 8,
                           param_filter=lambda name: True) -> QuantResult:
    """In-place fake-quantize all parameters of ``module`` matching ``param_filter``."""
    n = 0
    for name, p in module.named_parameters():
        if param_filter(name):
            p.copy_(fake_quantize(p.data, bits))
            n += p.numel()
    return QuantResult(bits=bits, n_params=n)


@torch.no_grad()
def quantize_model(model: nn.Module, weight_bits: int = 8, latent_bits: int = 8):
    """Fake-quantize a PEPS/grid model: grid/table latents and Linear weights.

    Returns a dict with quantized parameter counts and the effective bitrate.
    """
    latent_n = 0
    weight_n = 0
    for name, p in model.named_parameters():
        lname = name.lower()
        if "grid" in lname or "table" in lname:      # latent codes
            p.copy_(fake_quantize(p.data, latent_bits)); latent_n += p.numel()
        elif "weight" in lname or "bias" in lname:   # MLP params
            p.copy_(fake_quantize(p.data, weight_bits)); weight_n += p.numel()
    total = latent_n + weight_n
    eff = (latent_n * latent_bits + weight_n * weight_bits) / max(total, 1)
    return {"latent_params": latent_n, "weight_params": weight_n,
            "total_params": total, "effective_bits": eff}


def model_bitrate(model: nn.Module, latent_bits: int, weight_bits: int) -> float:
    """Effective average bits/param if quantized at the given widths (no mutation)."""
    latent_n = weight_n = 0
    for name, p in model.named_parameters():
        lname = name.lower()
        if "grid" in lname or "table" in lname:
            latent_n += p.numel()
        else:
            weight_n += p.numel()
    total = latent_n + weight_n
    return (latent_n * latent_bits + weight_n * weight_bits) / max(total, 1)
