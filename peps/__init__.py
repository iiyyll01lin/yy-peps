"""PEPS — Positional Encoding Projected Sampling (faithful reimplementation).

繁體中文:PEPS 函式庫。將座標經 Lissajous 投影成多個「興趣點」,對共享的
grid encoder 取樣,再以聚合器(concat / pink / brownian)組合後送入小型 MLP。
本檔匯出常用類別,方便 `from peps import PEPS, Projector, GridEncoder`。
"""

from .projector import Projector
from .aggregate import ConcatAggregator, PinkAggregator, BrownianAggregator, make_aggregator
from .wrapper import PEPS
from .encoders.grid import GridEncoder
from .encoders.positional import AbsolutePositionalEncoding, IdentityEncoder
from .models.mlp import MLP

__all__ = [
    "Projector",
    "ConcatAggregator",
    "PinkAggregator",
    "BrownianAggregator",
    "make_aggregator",
    "PEPS",
    "GridEncoder",
    "AbsolutePositionalEncoding",
    "IdentityEncoder",
    "MLP",
]
