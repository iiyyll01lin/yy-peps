"""Encoders: learned grid encoders and analytic positional encodings.

繁體中文:編碼器模組。包含可學習的 grid encoder(雙/三線性內插、hash、
multi-res)與解析式位置編碼(APE、Identity)。
"""

from .grid import GridEncoder
from .lpe import LocalPositionalEncoding
from .multires import HashGridEncoder, MultiResGridEncoder
from .ntc import (
    FourNeighborGridEncoder,
    NTCNEncoder,
    NTCPEPSEncoder,
    TiledTriangularEncoding,
)
from .positional import AbsolutePositionalEncoding, IdentityEncoder

__all__ = [
    "GridEncoder",
    "LocalPositionalEncoding",
    "MultiResGridEncoder",
    "HashGridEncoder",
    "FourNeighborGridEncoder",
    "TiledTriangularEncoding",
    "NTCNEncoder",
    "NTCPEPSEncoder",
    "AbsolutePositionalEncoding",
    "IdentityEncoder",
]
