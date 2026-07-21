"""PEPS wrapper — Project -> Encode -> Aggregate -> Model (paper Eq. 8).

繁體中文:PEPS 主組裝。實作
    M( A( E(P_1), ..., E(P_{2L+1}) ), delta )
流程:
    1. Projector 把座標 x 投影成 2L+1 個興趣點 P。
    2. 對每個點,用「共享」的 encoder E 取樣得到 latent。
    3. Aggregator A 把這些 latent 併成一個向量。
    4. MLP M 解碼成輸出訊號;delta 為可選的附加座標特徵(直接串接)。

若把 encoder 換成 IdentityEncoder 且 aggregator 用 concat,PEPS 會退化回純 APE。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .projector import Projector


class PEPS(nn.Module):
    """Assemble projector + shared encoder + aggregator + decoder.

    Args:
        projector: a :class:`Projector`.
        encoder: a shared encoder module mapping ``(N, dim) -> (N, k)``. The SAME
            module instance is applied to every projected point, so all points
            share (and send gradients to) one grid.
        aggregator: combines ``(N, num_points, k) -> (N, agg_out)``.
        model: decoder MLP mapping ``(N, agg_out + delta_dim) -> (N, out_dim)``.
        append_input_delta: if True, concatenate the raw input coords to the
            aggregated vector before the MLP (the ``delta`` in Eq. 8).
    """

    def __init__(
        self,
        projector: Projector,
        encoder: nn.Module,
        aggregator: nn.Module,
        model: nn.Module,
        append_input_delta: bool = False,
    ) -> None:
        super().__init__()
        self.projector = projector
        self.encoder = encoder
        self.aggregator = aggregator
        self.model = model
        self.append_input_delta = append_input_delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, dim)
        n, dim = x.shape
        pts = self.projector(x)                 # (N, P, dim)
        p = pts.shape[1]
        flat = pts.reshape(n * p, dim)          # (N*P, dim)
        lat = self.encoder(flat)                # (N*P, k)
        k = lat.shape[-1]
        lat = lat.reshape(n, p, k)              # (N, P, k)
        vec = self.aggregator(lat)              # (N, agg_out)
        if self.append_input_delta:
            vec = torch.cat([vec, x], dim=1)
        return self.model(vec)

    @property
    def num_points(self) -> int:
        return self.projector.num_points
