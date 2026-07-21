"""Spectral analysis helpers — radial PSD and 1/f slope (paper Fig. 3).

繁體中文:頻譜分析工具。計算影像的徑向平均功率譜密度(PSD),並用 log-log 直線
擬合估計 1/f^alpha 的斜率 alpha。用於 W06 驗證自然影像/材質的 PSD 呈 1/f,
支撐 Pink 聚合器「依頻率反比分配容量」的動機。
"""

from __future__ import annotations

import numpy as np
import torch


def radial_psd(image: torch.Tensor, nbins: int = 64):
    """Radially-averaged power spectral density of a 2D signal.

    Args:
        image: ``(H, W)`` or ``(H, W, C)`` tensor; channels are averaged first.
        nbins: number of radial frequency bins.
    Returns:
        ``(freqs, psd)`` numpy arrays, both length ``nbins`` (DC bin dropped).
    """
    if image.dim() == 3:
        image = image.mean(-1)
    g = image.double()
    F = torch.fft.fftshift(torch.fft.fft2(g))
    power = (F.abs() ** 2).cpu().numpy()

    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = r.max()
    bins = np.linspace(0, r_max, nbins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    psd = np.zeros(nbins)
    for b in range(nbins):
        sel = idx == b
        if sel.any():
            psd[b] = power.ravel()[sel].mean()
    freqs = 0.5 * (bins[:-1] + bins[1:])
    # drop DC (first bin) to keep log-log fit stable
    return freqs[1:], psd[1:]


def fit_one_over_f(freqs, psd):
    """Fit ``log psd = -alpha * log freq + c``; return ``alpha`` (the 1/f slope)."""
    mask = (freqs > 0) & (psd > 0)
    lf = np.log(freqs[mask])
    lp = np.log(psd[mask])
    a, _c = np.polyfit(lf, lp, 1)
    return float(-a)
