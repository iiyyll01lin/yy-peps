"""Shared training loop for coordinate-based fitting.

繁體中文:共用訓練迴圈。給定一個 model(接受座標、輸出訊號)、座標張量與目標張量,
做小批次的座標回歸(MSE)。影像、材質、SDF 三個應用都共用這個迴圈。
支援 ROCm/CUDA GPU;device 由呼叫端決定,預設自動偵測。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn


def auto_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 65536
    lr: float = 1e-2
    weight_decay: float = 0.0
    log_every: int = 200
    device: Optional[torch.device] = None
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = field(
        default_factory=lambda: nn.functional.mse_loss
    )


def fit(
    model: nn.Module,
    coords: torch.Tensor,
    targets: torch.Tensor,
    cfg: TrainConfig = TrainConfig(),
    on_log: Optional[Callable[[int, float], None]] = None,
) -> nn.Module:
    """Fit ``model(coords) -> targets`` with random minibatches of points.

    Args:
        coords: ``(P, dim)`` coordinates in ``[0, 1]``.
        targets: ``(P, out_dim)`` target signal values.
        cfg: training config.
        on_log: optional callback ``(step, loss)`` for notebooks to record curves.
    """
    device = cfg.device or auto_device()
    model = model.to(device)
    coords = coords.to(device)
    targets = targets.to(device)
    n = coords.shape[0]

    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    for step in range(cfg.steps):
        if cfg.batch_size >= n:
            idx = torch.arange(n, device=device)
        else:
            idx = torch.randint(0, n, (cfg.batch_size,), device=device)
        pred = model(coords[idx])
        loss = cfg.loss_fn(pred, targets[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % cfg.log_every == 0 or step == 0:
            lv = float(loss.item())
            if on_log:
                on_log(step + 1, lv)
    return model


@torch.no_grad()
def render_full(
    model: nn.Module,
    coords: torch.Tensor,
    chunk: int = 262144,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Evaluate ``model`` over all coords in chunks; returns predictions on CPU."""
    device = device or auto_device()
    model = model.to(device).eval()
    outs = []
    for i in range(0, coords.shape[0], chunk):
        c = coords[i : i + chunk].to(device)
        outs.append(model(c).cpu())
    return torch.cat(outs, dim=0)
