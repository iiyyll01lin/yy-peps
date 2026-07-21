"""HIP <-> PyTorch numerical parity reference (Phase 0 harness).

繁體中文:HIP kernel 與 PyTorch 的數值對拍地基。這裡先把 fused_peps_kernel.hip
的數學(border-padding 雙線性取樣 + 第一層 Linear + ReLU)用 PyTorch 精確重寫成
參考實作,並用「相同輸入」驗證兩種寫法一致。W11(Phase 4)會把訓練好的權重
dump 成 .npz,讓真正的 HIP 二進位讀同一份輸入、輸出與此參考對拍(<1e-3)。

本檔在純 CPU 上跑,不需 GPU;它保證「參考數學」正確,HIP 只要對上參考即對上 PyTorch。
"""

import numpy as np
import torch
import torch.nn.functional as F


def bilinear_sample_border_ref(grid_chw: np.ndarray, u: float, v: float) -> np.ndarray:
    """Reference bilinear sample matching hip/fused_peps_kernel.hip exactly.

    grid_chw: (C, H, W). (u, v) in [0,1] map to (W-1, H-1) with clamp-to-border.
    Returns latent of shape (C,).
    """
    C, H, W = grid_chw.shape
    fx = u * (W - 1)
    fy = v * (H - 1)
    x0 = int(np.floor(fx))
    y0 = int(np.floor(fy))
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    wx = fx - x0
    wy = fy - y0
    out = np.empty(C, dtype=np.float64)
    for c in range(C):
        g = grid_chw[c]
        v00, v01 = g[y0, x0], g[y0, x1]
        v10, v11 = g[y1, x0], g[y1, x1]
        top = v00 * (1 - wx) + v01 * wx
        bot = v10 * (1 - wx) + v11 * wx
        out[c] = top * (1 - wy) + bot * wy
    return out


def fused_sample_mlp_ref(grid_chw, coords, W1, b1):
    """Full reference for the fused kernel: sample -> Linear -> ReLU.

    grid_chw: (C,H,W); coords: (N,2) in [0,1]; W1: (C, Hn); b1: (Hn,).
    Returns (N, Hn).
    """
    N = coords.shape[0]
    C, H, W = grid_chw.shape
    Hn = b1.shape[0]
    out = np.empty((N, Hn), dtype=np.float64)
    for i in range(N):
        lat = bilinear_sample_border_ref(grid_chw, coords[i, 0], coords[i, 1])
        acc = b1.astype(np.float64).copy()
        for j in range(Hn):
            acc[j] += float(lat @ W1[:, j])
        out[i] = np.maximum(acc, 0.0)
    return out


def test_border_sample_matches_grid_sample_interior():
    """Our border-clamp bilinear must match torch.grid_sample on interior points.

    (Corners differ by convention; we check strictly-interior coords where the
    2x2 stencil stays in-bounds, so both implementations agree.)
    """
    rng = np.random.default_rng(0)
    C, H, W = 3, 8, 8
    grid = rng.standard_normal((C, H, W)).astype(np.float64)
    # interior coords keep fx,fy within [1, size-2] so no clamping happens
    coords = rng.uniform(0.2, 0.8, size=(16, 2))

    ref = np.stack([bilinear_sample_border_ref(grid, u, v) for u, v in coords])

    # torch grid_sample: align_corners=True maps [-1,1] to pixel centers.
    g = torch.from_numpy(grid).unsqueeze(0)  # (1,C,H,W)
    xy = torch.from_numpy(coords * 2 - 1).view(1, -1, 1, 2)  # (1,N,1,2) order (x=u,y=v)
    ts = F.grid_sample(g, xy, mode="bilinear", align_corners=True,
                       padding_mode="border").view(C, -1).t().numpy()
    assert np.allclose(ref, ts, atol=1e-10), np.abs(ref - ts).max()


def test_fused_ref_matches_torch_linear():
    """The sample->Linear->ReLU reference matches a torch nn.Linear composition."""
    rng = np.random.default_rng(1)
    C, H, W, Hn, N = 4, 16, 16, 6, 32
    grid = rng.standard_normal((C, H, W)).astype(np.float64)
    W1 = rng.standard_normal((C, Hn)).astype(np.float64)
    b1 = rng.standard_normal((Hn,)).astype(np.float64)
    coords = rng.uniform(0.15, 0.85, size=(N, 2))

    ref = fused_sample_mlp_ref(grid, coords, W1, b1)

    # torch path: sample each point, then y = relu(lat @ W1 + b1)
    lat = np.stack([bilinear_sample_border_ref(grid, u, v) for u, v in coords])
    lat_t = torch.from_numpy(lat)
    y = torch.relu(lat_t @ torch.from_numpy(W1) + torch.from_numpy(b1)).numpy()
    assert np.allclose(ref, y, atol=1e-10), np.abs(ref - y).max()


def test_parity_fixture_roundtrips(tmp_path):
    """A saved .npz fixture (grid, coords, W1, b1, expected) reloads bit-exact.

    This is the contract the real HIP binary will consume in W11: read the same
    arrays, compute on GPU, and compare its output to ``expected`` at <1e-3.
    """
    rng = np.random.default_rng(7)
    C, H, W, Hn, N = 8, 32, 32, 16, 64
    grid = rng.standard_normal((C, H, W)).astype(np.float32)
    coords = rng.uniform(0, 1, size=(N, 2)).astype(np.float32)
    W1 = rng.standard_normal((C, Hn)).astype(np.float32)
    b1 = np.zeros((Hn,), dtype=np.float32)
    expected = fused_sample_mlp_ref(grid, coords, W1, b1).astype(np.float32)

    f = tmp_path / "fused_fixture.npz"
    np.savez(f, grid=grid, coords=coords, W1=W1, b1=b1, expected=expected)
    d = np.load(f)
    assert d["grid"].shape == (C, H, W)
    assert d["expected"].shape == (N, Hn)
    # recompute from reloaded arrays -> same
    again = fused_sample_mlp_ref(d["grid"], d["coords"], d["W1"], d["b1"])
    assert np.allclose(again, d["expected"], atol=1e-5)
