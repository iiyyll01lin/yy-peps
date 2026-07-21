"""CPU and real-GPU parity for the integrated HIP workload and WMMA diagnostics.

The integrated fixture covers projection, every shared-grid point, concat and
paper-exact Pink aggregation, and all four Linear layers (three hidden layers).
Real HIP tests skip cleanly when hipcc or an AMD device is unavailable.
"""

import functools
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hip import export_fixture as hip_fixture


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


MODE_BASELINE = 0
MODE_CONCAT = 1
MODE_PINK = 2
ACT_GELU = 1


def projected_points_ref(coords: np.ndarray, frequencies: int) -> np.ndarray:
    """Paper layout ``(x, S_1..S_L, C_1..C_L)``."""

    points = [coords]
    for trig in (np.sin, np.cos):
        for index in range(1, frequencies + 1):
            points.append((1.0 + trig(coords * (2**index * np.pi))) * 0.5)
    return np.stack(points, axis=1).astype(coords.dtype, copy=False)


def pink_channel_indices(channels: int, frequencies: int) -> tuple[tuple[int, ...], ...]:
    """Exact Algorithm 1 circular slices in point-layout order."""

    widths = [max(1, channels // (2**index)) for index in range(1, frequencies + 1)]
    cumulative = [0]
    for width in widths:
        cumulative.append(cumulative[-1] + width)
    sine = [
        tuple(index % channels for index in range(-cumulative[i], -cumulative[i - 1]))
        for i in range(1, frequencies + 1)
    ]
    cosine = [
        tuple(index % channels for index in range(cumulative[i - 1], cumulative[i]))
        for i in range(1, frequencies + 1)
    ]
    return (tuple(range(channels)), *sine, *cosine)


def integrated_features_ref(grid_chw, coords, mode, frequencies):
    """Projection, every shared-grid sample, and selected aggregation."""

    if mode == MODE_BASELINE:
        return np.stack(
            [bilinear_sample_border_ref(grid_chw, u, v) for u, v in coords]
        ).astype(np.float32)
    points = projected_points_ref(coords, frequencies)
    sampled = np.stack(
        [
            np.stack(
                [bilinear_sample_border_ref(grid_chw, u, v) for u, v in point_set]
            )
            for point_set in points
        ]
    ).astype(np.float32)
    if mode == MODE_CONCAT:
        return sampled.reshape(sampled.shape[0], -1)
    pieces = [
        sampled[:, point, list(indices)]
        for point, indices in enumerate(pink_channel_indices(grid_chw.shape[0], frequencies))
    ]
    return np.concatenate(pieces, axis=1)


def complete_mlp_ref(features, weights, biases):
    """Four Linear layers: three GELU hidden layers and one output layer."""

    value = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    for layer in range(3):
        value = F.gelu(
            value @ torch.from_numpy(weights[layer])
            + torch.from_numpy(biases[layer]),
            approximate="none",
        )
    value = (
        value @ torch.from_numpy(weights[3])
        + torch.from_numpy(biases[3])
    )
    return value.numpy()


def integrated_workload_ref(grid, coords, mode, frequencies, weights, biases):
    features = integrated_features_ref(grid, coords, mode, frequencies)
    return complete_mlp_ref(features, weights, biases)


def integrated_workload_fp16_ref(
    grid, coords, mode, frequencies, weights, biases
):
    """Reference the fused WMMA contract: fp16 values, fp32 accumulation."""

    half_grid = torch.from_numpy(grid).half().float().numpy()
    value = torch.from_numpy(
        integrated_features_ref(
            half_grid, coords, mode, frequencies
        )
    ).half()
    for layer in range(3):
        half_weight = torch.from_numpy(weights[layer]).half()
        accumulated = (
            value.float() @ half_weight.float()
            + torch.from_numpy(biases[layer])
        )
        value = F.gelu(accumulated, approximate="none").half()
    half_weight = torch.from_numpy(weights[3]).half()
    return (
        value.float() @ half_weight.float()
        + torch.from_numpy(biases[3])
    ).numpy()


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


@pytest.mark.parametrize("mode", [MODE_BASELINE, MODE_CONCAT, MODE_PINK])
def test_integrated_features_match_projector_grid_and_aggregator(mode):
    from peps import ConcatAggregator, GridEncoder, PinkAggregator, Projector

    rng = np.random.default_rng(1)
    channels, height, width, frequencies, points = 8, 9, 7, 3, 12
    grid = rng.standard_normal((channels, height, width)).astype(np.float32)
    coords = rng.uniform(0.0, 1.0, size=(points, 2)).astype(np.float32)
    got = integrated_features_ref(grid, coords, mode, frequencies)

    encoder = GridEncoder(2, (height, width), channels)
    with torch.no_grad():
        encoder.grid.copy_(torch.from_numpy(grid).unsqueeze(0))
    coordinates = torch.from_numpy(coords)
    if mode == MODE_BASELINE:
        expected = encoder(coordinates)
    else:
        projector = Projector(frequencies)
        projected = projector(coordinates)
        latents = torch.stack(
            [encoder(projected[:, point]) for point in range(projector.num_points)],
            dim=1,
        )
        aggregator = (
            ConcatAggregator(projector.num_points, channels)
            if mode == MODE_CONCAT
            else PinkAggregator(projector.num_points, channels)
        )
        expected = aggregator(latents)
    assert np.allclose(got, expected.detach().numpy(), atol=1e-5)


def test_pink_fixture_indices_match_paper_example():
    assert pink_channel_indices(8, 3) == (
        (0, 1, 2, 3, 4, 5, 6, 7),
        (4, 5, 6, 7),
        (2, 3),
        (1,),
        (0, 1, 2, 3),
        (4, 5),
        (6,),
    )


def test_complete_three_hidden_layer_reference_matches_mlp_module():
    from peps import MLP

    rng = np.random.default_rng(5)
    features = rng.standard_normal((11, 14)).astype(np.float32) * 0.1
    hidden, output = 16, 3
    shapes = [(14, hidden), (hidden, hidden), (hidden, hidden), (hidden, output)]
    weights = [rng.standard_normal(shape).astype(np.float32) * 0.1 for shape in shapes]
    biases = [
        rng.standard_normal(shape[1]).astype(np.float32) * 0.01 for shape in shapes
    ]
    reference = complete_mlp_ref(features, weights, biases)

    model = MLP(14, output, hidden_dim=hidden, num_layers=4, activation="gelu")
    linear_layers = [module for module in model.net if isinstance(module, torch.nn.Linear)]
    with torch.no_grad():
        for layer, weight, bias in zip(linear_layers, weights, biases):
            layer.weight.copy_(torch.from_numpy(weight.T))
            layer.bias.copy_(torch.from_numpy(bias))
    expected = model(torch.from_numpy(features)).detach().numpy()
    assert np.allclose(reference, expected, atol=1e-6)


# =============================================================================
# WMMA references (fp16 matmul + int8 GEMM), used by both CPU and GPU tests.
# =============================================================================

def wmma_fp16_matmul_ref(a_f32: np.ndarray, b_f32: np.ndarray) -> np.ndarray:
    """Reference for wmma_mlp.hip fp16 path: fp16 inputs, fp32/higher accumulate.

    ``a_f32`` / ``b_f32`` must already be fp16-representable float32 arrays (round
    them with ``torch.half`` first), so the reference and the kernel consume the
    *identical* inputs and only the accumulation precision can differ. Computed as
    a torch matmul in float64 of the fp16-rounded operands.
    """
    a16 = torch.from_numpy(np.ascontiguousarray(a_f32)).half()
    b16 = torch.from_numpy(np.ascontiguousarray(b_f32)).half()
    return (a16.double() @ b16.double()).numpy()


def wmma_int8_gemm_ref(a_f32: np.ndarray, b_f32: np.ndarray):
    """Reference for wmma_mlp.hip int8 path, consistent with peps.quant.ptq.

    Quantizes both operands with the same symmetric per-tensor scheme used in
    ``peps/quant/ptq.py`` (``quantize_tensor``), then does an *exact* int32 GEMM
    of the int8 codes. Returns (qA int8, qB int8, C int32, scaleA, scaleB).
    """
    from peps.quant.ptq import quantize_tensor

    qa, sa = quantize_tensor(torch.from_numpy(np.ascontiguousarray(a_f32)), bits=8)
    qb, sb = quantize_tensor(torch.from_numpy(np.ascontiguousarray(b_f32)), bits=8)
    qa_i8 = qa.numpy().astype(np.int8)
    qb_i8 = qb.numpy().astype(np.int8)
    c_i32 = qa_i8.astype(np.int32) @ qb_i8.astype(np.int32)   # exact integer GEMM
    return qa_i8, qb_i8, c_i32, float(sa), float(sb)


def test_wmma_fp16_ref_matches_torch_half_matmul():
    """The fp16 reference equals torch's own fp16 matmul upcast to float."""
    rng = np.random.default_rng(2)
    a = (rng.standard_normal((32, 48)).astype(np.float32) * 0.25)
    b = (rng.standard_normal((48, 16)).astype(np.float32) * 0.25)
    a16 = torch.from_numpy(a).half().float().numpy()
    b16 = torch.from_numpy(b).half().float().numpy()
    ref = wmma_fp16_matmul_ref(a16, b16)
    torch_half = (torch.from_numpy(a16).half().float()
                  @ torch.from_numpy(b16).half().float()).numpy()
    assert np.abs(ref - torch_half).max() < 1e-2, np.abs(ref - torch_half).max()


def test_wmma_int8_ref_is_exact_and_bounded():
    """int8 GEMM ref is an exact integer product; dequant tracks full precision."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal((32, 64)).astype(np.float32)
    b = rng.standard_normal((64, 16)).astype(np.float32)
    qa, qb, c_i32, sa, sb = wmma_int8_gemm_ref(a, b)
    # exact integer recompute
    assert np.array_equal(c_i32, qa.astype(np.int64) @ qb.astype(np.int64))
    # dequantized result is a reasonable approximation of the fp32 GEMM
    deq = c_i32.astype(np.float64) * sa * sb
    full = a.astype(np.float64) @ b.astype(np.float64)
    denom = np.abs(full).mean() + 1e-8
    assert np.abs(deq - full).mean() / denom < 0.1


# =============================================================================
# Real GPU parity: compile the HIP kernels with hipcc and compare their output
# to the PyTorch references above. Skips gracefully without hipcc / an AMD GPU.
# =============================================================================

ROOT = _REPO_ROOT
HIP_DIR = ROOT / "hip"
WORKLOAD_MAGIC = 0x50505332  # 'PPS2'
WORKLOAD_SCHEMA = 2
WMMA_MAGIC = 0x574D4D31   # 'WMM1'


def _hipcc() -> str | None:
    """Return the newest usable HIP compiler entry point.

    Some ROCm 7.2 packages ship ``amdclang++`` but no ``hipcc`` wrapper, while
    the OS may still expose an older ``/usr/bin/hipcc`` that cannot target
    gfx1201. Prefer the active ROCm tree before falling back to ``PATH``.
    """

    candidates = (
        "/opt/rocm/bin/hipcc",
        "/opt/rocm/bin/amdclang++",
        shutil.which("hipcc"),
    )
    return next(
        (candidate for candidate in candidates if candidate and os.path.exists(candidate)),
        None,
    )


def _compiler_command(
    compiler: str, arch: str, source: str, output: str
) -> list[str]:
    command = [compiler]
    if os.path.basename(compiler).startswith("amdclang"):
        command.extend(["-x", "hip"])
    command.extend([f"--offload-arch={arch}", source])
    if os.path.basename(compiler).startswith("amdclang"):
        command.extend(["-L/opt/rocm/lib", "-lamdhip64"])
    command.extend(["-o", output])
    return command


def _detect_arch() -> str | None:
    """Detect the local GPU's gfx arch robustly.

    ``offload-arch`` queries the device directly and works even where ``rocminfo``
    does not emit a gfx line (observed on the RDNA4 box). Falls back to rocminfo.
    """
    for tool in ("offload-arch", "/opt/rocm/bin/offload-arch"):
        exe = tool if (os.path.isabs(tool) and os.path.exists(tool)) else shutil.which(tool)
        if not exe:
            continue
        try:
            out = subprocess.run([exe], capture_output=True, text=True, timeout=20).stdout
            toks = [t for t in out.split() if t.startswith("gfx")]
            if toks:
                return toks[0].strip()
        except Exception:
            pass
    exe = shutil.which("rocminfo")
    if not exe and os.path.exists("/usr/bin/rocminfo"):
        exe = "/usr/bin/rocminfo"
    if exe:
        try:
            out = subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout
            m = re.search(r"gfx[0-9a-f]+", out)
            if m:
                return m.group(0)
        except Exception:
            pass
    return None


def _skip_reason() -> str | None:
    if _hipcc() is None:
        return "HIP compiler not found (CPU-only environment)"
    if _detect_arch() is None:
        return "no AMD GPU arch detected"
    return None


requires_gpu = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


@functools.lru_cache(maxsize=None)
def _build(src_name: str) -> str:
    """Compile a HIP source for the detected arch; cached across tests."""
    hipcc, arch = _hipcc(), _detect_arch()
    assert hipcc and arch, "guarded by requires_gpu"
    outdir = tempfile.mkdtemp(prefix="peps_hip_")
    out = os.path.join(outdir, Path(src_name).stem)
    env = dict(os.environ)
    env["PATH"] = "/opt/rocm/bin:" + env.get("PATH", "")
    include_path = env.get("CPLUS_INCLUDE_PATH")
    env["CPLUS_INCLUDE_PATH"] = (
        "/opt/rocm/include"
        if not include_path
        else "/opt/rocm/include" + os.pathsep + include_path
    )
    r = subprocess.run(
        _compiler_command(hipcc, arch, str(HIP_DIR / src_name), out),
        capture_output=True, text=True, timeout=600, env=env,
    )
    if r.returncode != 0:
        raise RuntimeError(f"hipcc failed for {src_name}:\n{r.stderr[-3000:]}")
    return out


def _run(binary: str, args, timeout: int = 180) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("HIP_VISIBLE_DEVICES", "0")
    return subprocess.run([binary, *map(str, args)], capture_output=True,
                          text=True, timeout=timeout, env=env)


def _write_integrated_fixture(
    path, mode, grid, coords, frequencies, weights, biases
):
    c, h, w = grid.shape
    n = coords.shape[0]
    hidden = biases[0].shape[0]
    output = biases[3].shape[0]
    with open(path, "wb") as f:
        f.write(
            np.array(
                [
                    WORKLOAD_MAGIC,
                    WORKLOAD_SCHEMA,
                    mode,
                    c,
                    h,
                    w,
                    n,
                    frequencies,
                    hidden,
                    output,
                    ACT_GELU,
                ],
                dtype="<i4",
            ).tobytes()
        )
        f.write(np.ascontiguousarray(grid, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(coords, dtype="<f4").tobytes())
        for weight, bias in zip(weights, biases):
            f.write(np.ascontiguousarray(weight, dtype="<f4").tobytes())
            f.write(np.ascontiguousarray(bias, dtype="<f4").tobytes())


def _read_integrated_output(path):
    with open(path, "rb") as f:
        magic, schema, mode, n, output = struct.unpack("<5i", f.read(20))
        assert magic == WORKLOAD_MAGIC and schema == WORKLOAD_SCHEMA
        data = np.frombuffer(f.read(), dtype="<f4")
    return mode, data.reshape(n, output)


def _write_wmma_fixture(path, a, b, dtype_code):
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    np_dtype = "<f4" if dtype_code == 0 else "<i1"
    with open(path, "wb") as f:
        f.write(np.array([WMMA_MAGIC, dtype_code, m, k, n], dtype="<i4").tobytes())
        f.write(np.ascontiguousarray(a, dtype=np_dtype).tobytes())
        f.write(np.ascontiguousarray(b, dtype=np_dtype).tobytes())


def _read_wmma_output(path, dtype_code):
    with open(path, "rb") as f:
        magic, dt, m, n = struct.unpack("<4i", f.read(16))
        assert magic == WMMA_MAGIC and dt == dtype_code, "bad wmma output header"
        out_dtype = "<f4" if dtype_code == 0 else "<i4"
        data = np.frombuffer(f.read(), dtype=out_dtype)
    return data.reshape(m, n)


@pytest.mark.parametrize("mode", [MODE_BASELINE, MODE_CONCAT, MODE_PINK])
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
@requires_gpu
def test_hip_integrated_modes_match_pytorch(tmp_path, mode, precision):
    """Scalar and fused-WMMA paths cover all three modes end to end."""

    rng = np.random.default_rng(11)
    c, h, w, frequencies, hidden, output, n = 8, 11, 9, 3, 16, 3, 37
    grid = (rng.standard_normal((c, h, w)) * 0.2).astype(np.float32)
    coords = rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float32)
    input_dim = integrated_features_ref(grid, coords, mode, frequencies).shape[1]
    shapes = [
        (input_dim, hidden),
        (hidden, hidden),
        (hidden, hidden),
        (hidden, output),
    ]
    weights = [
        (rng.standard_normal(shape) * 0.08).astype(np.float32)
        for shape in shapes
    ]
    biases = [
        (rng.standard_normal(shape[1]) * 0.01).astype(np.float32)
        for shape in shapes
    ]
    reference = (
        integrated_workload_ref
        if precision == "fp32"
        else integrated_workload_fp16_ref
    )
    expected = reference(grid, coords, mode, frequencies, weights, biases)

    binary = _build("fused_peps_kernel.hip")
    fin = tmp_path / f"integrated_{precision}_{mode}_in.bin"
    fout = tmp_path / f"integrated_{precision}_{mode}_out.bin"
    _write_integrated_fixture(
        fin, mode, grid, coords, frequencies, weights, biases
    )
    r = _run(binary, ["fixture", precision, fin, fout])
    assert r.returncode == 0, (
        f"integrated fixture run failed:\n{r.stdout}\n{r.stderr}"
    )

    output_mode, got = _read_integrated_output(fout)
    assert output_mode == mode
    assert got.shape == (n, output)
    max_abs = float(np.abs(got.astype(np.float64) - expected).max())
    tolerance = 1e-3 if precision == "fp32" else 4e-3
    assert max_abs < tolerance, (
        f"{precision} integrated kernel max abs err {max_abs:.2e}"
    )


@pytest.mark.parametrize("method", tuple(hip_fixture.METHOD_SPECS))
@requires_gpu
def test_hip_fused_wmma_paper_methods_match_exporter(tmp_path, method):
    """All four paper methods execute the fused path through every RGB value."""

    fixture = hip_fixture.make_random_fixture(
        method,
        channels=16,
        grid_height=11,
        grid_width=9,
        points=37,  # deliberately not a 16-query WMMA tile multiple
        hidden=64,
        output=3,
        seed=41,
    )
    expected = fixture.reference("fp16")
    fixture_path = tmp_path / f"{method}.fixture.bin"
    output_path = tmp_path / f"{method}.output.bin"
    metadata = hip_fixture.write_fixture(
        fixture_path,
        fixture,
        manifest_path=tmp_path / f"{method}.manifest.json",
    )
    assert metadata["selective_channel_sampling"] == ("pink" in method)

    result = _run(
        _build("fused_peps_kernel.hip"),
        ["fixture", "fp16", fixture_path, output_path],
    )
    assert result.returncode == 0, (
        f"fused WMMA fixture failed:\n{result.stdout}\n{result.stderr}"
    )
    output_mode, got = hip_fixture.read_output(output_path)
    assert output_mode == fixture.method.mode
    assert got.shape == expected.shape == (37, 3)
    error = np.abs(got.astype(np.float64) - expected.astype(np.float64))
    assert float(error.max()) < 4e-3
    assert float(error.mean()) < 5e-4


@pytest.mark.parametrize("points", [1, 15, 16, 17, 33])
@requires_gpu
def test_hip_wmma_odd_hidden_border_and_tail_shapes(tmp_path, points):
    """Exercise border clamp and query/hidden/output matrix tile tails."""

    border = np.array(
        [[-0.25, 0.5], [0.0, 0.0], [1.0, 1.0], [0.5, 1.25]],
        dtype=np.float32,
    )
    fixture = hip_fixture.make_random_fixture(
        "grid-pink-peps-4f",
        channels=7,
        grid_height=5,
        grid_width=9,
        points=points,
        hidden=17,
        output=5,
        coords=np.resize(border, (points, 2)).astype(np.float32),
        seed=53,
    )
    expected = fixture.reference("fp16")
    fixture_path = tmp_path / f"tail_{points}.fixture.bin"
    output_path = tmp_path / f"tail_{points}.output.bin"
    hip_fixture.write_fixture(fixture_path, fixture)
    result = _run(
        _build("fused_peps_kernel.hip"),
        ["fixture", "fp16", fixture_path, output_path],
    )
    assert result.returncode == 0, result.stderr
    _, got = hip_fixture.read_output(output_path)
    error = np.abs(got.astype(np.float64) - expected.astype(np.float64))
    assert got.shape == (points, 5)
    assert float(error.max()) < 4e-3


@requires_gpu
def test_hip_wmma_fp16_matches_pytorch(tmp_path):
    """Real wmma_mlp fp16 binary output matches the torch matmul reference (<1e-3)."""
    rng = np.random.default_rng(21)
    m, k, n = 64, 64, 32  # all multiples of the 16x16x16 WMMA tile
    a = (rng.standard_normal((m, k)).astype(np.float32) * 0.25)
    b = (rng.standard_normal((k, n)).astype(np.float32) * 0.25)
    a16 = torch.from_numpy(a).half().float().numpy()  # fp16-exact operands
    b16 = torch.from_numpy(b).half().float().numpy()

    expected = wmma_fp16_matmul_ref(a16, b16)

    binary = _build("wmma_mlp.hip")
    fin, fout = tmp_path / "wmma_fp16_in.bin", tmp_path / "wmma_fp16_out.bin"
    _write_wmma_fixture(fin, a16, b16, dtype_code=0)
    r = _run(binary, ["fixture", "fp16", fin, fout])
    assert r.returncode == 0, f"wmma fp16 run failed:\n{r.stdout}\n{r.stderr}"

    got = _read_wmma_output(fout, dtype_code=0)
    assert got.shape == (m, n)
    max_abs = float(np.abs(got.astype(np.float64) - expected).max())
    assert max_abs < 1e-3, f"wmma fp16 max abs err {max_abs:.2e}"


@requires_gpu
def test_hip_wmma_int8_matches_pytorch(tmp_path):
    """Real wmma_mlp int8 binary output matches the ptq int8 GEMM exactly."""
    rng = np.random.default_rng(31)
    m, k, n = 64, 64, 32
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)

    qa, qb, expected_i32, sa, sb = wmma_int8_gemm_ref(a, b)

    binary = _build("wmma_mlp.hip")
    fin, fout = tmp_path / "wmma_i8_in.bin", tmp_path / "wmma_i8_out.bin"
    _write_wmma_fixture(fin, qa, qb, dtype_code=1)
    r = _run(binary, ["fixture", "int8", fin, fout])
    assert r.returncode == 0, f"wmma int8 run failed:\n{r.stdout}\n{r.stderr}"

    got = _read_wmma_output(fout, dtype_code=1)
    assert got.shape == (m, n)
    # int8 GEMM must be bit-exact against the integer reference.
    assert np.array_equal(got, expected_i32), (
        f"int8 mismatch: max |diff| = {int(np.abs(got - expected_i32).max())}"
    )
