"""Small MLP decoder — paper config: 3 layers, 64 neurons.

繁體中文:小型 MLP 解碼器。論文設定為 3 層、每層 64 神經元。輸入是聚合器輸出的
向量(可再串接額外座標 delta),輸出訊號值(影像 RGB / SDF 距離等)。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        layers = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), act()]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
