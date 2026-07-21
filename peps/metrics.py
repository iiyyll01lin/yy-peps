"""Metrics — PSNR, SSIM, (optional) LPIPS, LSD spectral metric, IoU.

繁體中文:評估指標。PSNR/SSIM 為主;LPIPS 需額外套件(延後載入);LSD 是論文用的
頻譜指標(log 頻譜距離),量高頻重建品質;IoU 給 SDF 佔用率比較。
"""

from __future__ import annotations

import math
import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB. Inputs in the same range as data_range."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Global SSIM (single-window approximation) on ``(H, W)`` or ``(H, W, C)``.

    For teaching we use the simple global-statistics SSIM; swap in
    ``skimage.metrics.structural_similarity`` for the windowed version.
    """
    x = pred.reshape(-1).double()
    y = target.reshape(-1).double()
    mu_x, mu_y = x.mean(), y.mean()
    vx, vy = x.var(unbiased=False), y.var(unbiased=False)
    cxy = ((x - mu_x) * (y - mu_y)).mean()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * cxy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (vx + vy + c2)
    )
    return float(s.item())


def lsd(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Log-Spectral Distance — L2 distance of log power spectra.

    Captures how well high-frequency content is reproduced (PEPS's selling
    point). Lower is better. Operates on ``(H, W)`` grayscale or per-channel.
    """
    if pred.dim() == 3:  # (H, W, C) -> average over channels
        return sum(
            lsd(pred[..., c], target[..., c], eps) for c in range(pred.shape[-1])
        ) / pred.shape[-1]
    Pp = torch.fft.rfft2(pred.double())
    Pt = torch.fft.rfft2(target.double())
    sp = torch.log(Pp.abs() ** 2 + eps)
    st = torch.log(Pt.abs() ** 2 + eps)
    return float(torch.sqrt(torch.mean((sp - st) ** 2)).item())


def iou(pred_occupancy: torch.Tensor, target_occupancy: torch.Tensor) -> float:
    """Intersection-over-Union of two boolean occupancy grids (SDF, W09)."""
    p = pred_occupancy.bool()
    t = target_occupancy.bool()
    inter = (p & t).sum().item()
    union = (p | t).sum().item()
    return inter / union if union > 0 else 1.0


_lpips_model = None


def lpips(pred: torch.Tensor, target: torch.Tensor, net: str = "alex") -> float:
    """LPIPS perceptual distance. Lazily imports the ``lpips`` package.

    Inputs ``(H, W, 3)`` in ``[0, 1]``. Returns a scalar (lower is better).
    """
    global _lpips_model
    import lpips as _lpips_pkg  # deferred: optional dependency

    if _lpips_model is None:
        _lpips_model = _lpips_pkg.LPIPS(net=net)
    device = pred.device
    _lpips_model = _lpips_model.to(device)

    def to_nchw(t):
        return (t.permute(2, 0, 1).unsqueeze(0) * 2 - 1).float()

    with torch.no_grad():
        d = _lpips_model(to_nchw(pred), to_nchw(target))
    return float(d.item())
