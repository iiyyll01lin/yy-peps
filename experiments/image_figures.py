"""Regenerate PEPS Figures 1--7 and 10--11 without invented results.

Analytic figures are always available.  Data-dependent figures are emitted only
when their checksum-verified inputs or complete training records exist; blocked
figures are represented in ``figure_status.json`` rather than by placeholder
numbers.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from apps.image.data import coords_grid, image_to_coords_targets, load_paper_kodak
from apps.texture.data import load_paper_texture_set
from data.manifest import hash_file
from experiments.config import load_experiment_config
from experiments.image_repro import (
    ARTIFACT_CONFIGS,
    DEFAULT_OUTPUT_ROOT,
    ROOT,
    _artifact_output,
    _job_paths,
    artifact_progress,
)
from experiments.runner import (
    TensorInstance,
    _build_model,
    atomic_write_json,
    collect_raw_records,
)
from peps import PinkAggregator, Projector
from peps.train import render_full


FIGURE_SCHEMA_VERSION = 1
PAPER_TEXTURE_PSD_IDS = (
    "bench-vice-01",
    "cardboard-box-01",
    "cannon-01",
    "clay-roof-tiles-02",
    "fabric-pattern-07",
    "garden-gnome",
    "garden-sprinkler-01",
    "wood-planks",
    "treasure-chest",
    "paving-stones-070",
)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    try:
        fig.savefig(temporary, dpi=180, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def figure1_workflow(path: Path) -> dict[str, object]:
    """Draw the paper's Project -> Encode -> Aggregate -> Model pipeline."""

    plt = _pyplot()
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(15, 3.2),
        gridspec_kw={"width_ratios": [1.1, 1.4, 1.2, 0.8, 1.0]},
    )
    coordinate = torch.tensor([[0.20, 0.35]])
    points = Projector(3)(coordinate)[0].numpy()
    colors = ["#1597a5"] + ["#8f4bb8"] * 3 + ["#e8892d"] * 3

    axes[0].scatter([coordinate[0, 0]], [coordinate[0, 1]], s=55, color=colors[0])
    axes[0].set_title("(a) input $x$")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), aspect="equal")

    axes[1].plot(points[1:4, 0], points[1:4, 1], color=colors[1], alpha=0.5)
    axes[1].plot(points[4:, 0], points[4:, 1], color=colors[4], alpha=0.5)
    for point, color in zip(points, colors):
        axes[1].scatter(point[0], point[1], color=color, s=35)
    axes[1].set_title("(b) project $P^x$")
    axes[1].set(xlim=(0, 1), ylim=(0, 1), aspect="equal")

    for position in range(6):
        axes[2].axhline(position / 5, color="#cccccc", linewidth=0.6)
        axes[2].axvline(position / 5, color="#cccccc", linewidth=0.6)
    axes[2].scatter(points[:, 0], points[:, 1], c=colors, s=35)
    axes[2].set_title("(c) shared grid $G$")
    axes[2].set(xlim=(0, 1), ylim=(0, 1), aspect="equal")

    latent = np.arange(7)[:, None] + np.linspace(0, 1, 8)[None, :]
    axes[3].imshow(latent, aspect="auto", cmap="viridis")
    axes[3].set_title("(d) concat")
    axes[3].set_xlabel("latent channel")
    axes[3].set_ylabel("point")

    axes[4].axis("off")
    axes[4].text(
        0.5,
        0.65,
        "3 × 64\nMLP",
        ha="center",
        va="center",
        fontsize=14,
        bbox={"boxstyle": "round", "facecolor": "#eeeeee"},
    )
    axes[4].text(0.5, 0.25, "$T(x)\\in\\mathbb{R}^{3k}$", ha="center", fontsize=13)
    axes[4].set_title("(e) model/output")
    fig.suptitle("Figure 1 — Grid-PEPS data flow (schematic reproduction)")
    for axis in axes[:3]:
        axis.set_xticks([])
        axis.set_yticks([])
    _save_figure(fig, path)
    plt.close(fig)
    return {
        "status": "generated",
        "path": str(path),
        "verification": "analytic_schematic",
        "frequencies": [2, 4, 8],
        "coordinate": coordinate[0].tolist(),
    }


def figure2_rotation(path: Path) -> dict[str, object]:
    plt = _pyplot()
    parameter = torch.linspace(0.05, 0.95, 80)
    line = torch.stack((parameter, 0.25 + 0.5 * parameter), dim=1)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    axes[0].plot(line[:, 0], line[:, 1], color="#1597a5")
    axes[0].set_title("original $(x,y)$")
    axes[0].set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), aspect="equal")
    for panel, frequency_index in enumerate((1, 2, 3), start=1):
        phi = 2**frequency_index * math.pi
        for axis_index, label, color in (
            (0, "x axis", "#8f4bb8"),
            (1, "y axis", "#e8892d"),
        ):
            value = line[:, axis_index]
            axes[panel].plot(
                torch.sin(phi * value),
                torch.cos(phi * value),
                color=color,
                label=label,
            )
        axes[panel].set_title(f"$\\phi_{frequency_index}=2^{frequency_index}\\pi$")
        axes[panel].set(
            xlim=(-1.05, 1.05),
            ylim=(-1.05, 1.05),
            aspect="equal",
        )
    axes[-1].legend(loc="lower left", fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Figure 2 — axis-wise APE rotations")
    _save_figure(fig, path)
    plt.close(fig)
    return {
        "status": "generated",
        "path": str(path),
        "verification": "analytic_from_phi_i_equals_2_pow_i_pi",
        "frequency_indices": [1, 2, 3],
        "line_definition": "(t, 0.25 + 0.5t), t in [0.05,0.95]",
    }


def figure4_pink_allocation(path: Path, data_path: Path) -> dict[str, object]:
    plt = _pyplot()
    aggregator = PinkAggregator(
        7,
        8,
        num_frequencies=3,
        include_input=True,
        frequency_scales=(2, 4, 8),
    )
    selected = np.zeros((8, 7), dtype=float)
    rows = []
    for point, indices in enumerate(aggregator.point_channel_indices):
        for channel in indices:
            selected[channel, point] = 1.0
            rows.append(
                {
                    "point": aggregator.point_layout[point],
                    "channel": channel,
                    "selected": 1,
                }
            )
    _write_csv(data_path, rows, ("point", "channel", "selected"))
    fig, axis = plt.subplots(figsize=(7.3, 4.1))
    axis.imshow(selected, origin="upper", cmap="RdPu", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(7), aggregator.point_layout, rotation=30, ha="right")
    axis.set_yticks(range(8))
    axis.set_xlabel("projected point")
    axis.set_ylabel("latent channel")
    axis.set_title(
        "Figure 4 — Pink allocation: d=8, L=3\n"
        f"56 concat channels → {aggregator.out_dim} selected channels"
    )
    _save_figure(fig, path)
    plt.close(fig)
    return {
        "status": "generated",
        "path": str(path),
        "data": str(data_path),
        "verification": "paper_algorithm_1_exact",
        "concat_width": 56,
        "pink_width": aggregator.out_dim,
        "frequency_widths": aggregator.frequency_widths,
        "cumulative_allocations": list(aggregator.cumulative_allocations),
    }


def figure10_lissajous(path: Path, data_path: Path) -> dict[str, object]:
    plt = _pyplot()
    phi = torch.linspace(0.0, 40.0 * math.pi, 8000)
    marked_phi = torch.tensor([2**index * math.pi for index in range(1, 5)])
    points = ((0.35, 0.20), (0.20, 0.30))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))
    rows = []
    for axis, (x_value, y_value) in zip(axes, points):
        x_curve = (1.0 + torch.sin(x_value * phi)) / 2.0
        y_curve = (1.0 + torch.sin(y_value * phi)) / 2.0
        axis.plot(x_curve, y_curve, color="#333333", linewidth=0.7)
        marked_x = (1.0 + torch.sin(x_value * marked_phi)) / 2.0
        marked_y = (1.0 + torch.sin(y_value * marked_phi)) / 2.0
        axis.scatter(marked_x, marked_y, color="#d73027", zorder=3)
        axis.set(
            xlim=(-0.02, 1.02),
            ylim=(-0.02, 1.02),
            aspect="equal",
            title=f"point ({x_value}, {y_value})",
            xlabel="$S_\\phi(x)$",
            ylabel="$S_\\phi(y)$",
        )
        for frequency_index, x_mark, y_mark in zip(
            range(1, 5), marked_x, marked_y
        ):
            rows.append(
                {
                    "point_x": x_value,
                    "point_y": y_value,
                    "frequency_index": frequency_index,
                    "phi": 2**frequency_index * math.pi,
                    "projected_x": float(x_mark),
                    "projected_y": float(y_mark),
                }
            )
    fig.suptitle("Figure 10 — normalized Lissajous curves")
    _save_figure(fig, path)
    plt.close(fig)
    _write_csv(
        data_path,
        rows,
        (
            "point_x",
            "point_y",
            "frequency_index",
            "phi",
            "projected_x",
            "projected_y",
        ),
    )
    return {
        "status": "generated",
        "path": str(path),
        "data": str(data_path),
        "verification": "analytic_from_appendix_points",
        "continuous_phi_interval": [0.0, 40.0 * math.pi],
    }


def figure11_unique_curves(path: Path, data_path: Path) -> dict[str, object]:
    plt = _pyplot()
    phi = torch.linspace(0.0, 20.0 * math.pi, 6000)
    phi_normalized = phi / phi[-1]
    marked_phi = torch.tensor([2**index * math.pi for index in range(1, 5)])
    points = (
        (0.20, 0.30, "#d73027"),
        (0.30, 0.45, "#1a9850"),
        (0.40, 0.60, "#4575b4"),
    )
    fig = plt.figure(figsize=(7.0, 5.5))
    axis = fig.add_subplot(111, projection="3d")
    rows = []
    for x_value, y_value, color in points:
        projected_x = (1.0 + torch.cos(x_value * phi)) / 2.0
        projected_y = (1.0 + torch.cos(y_value * phi)) / 2.0
        axis.plot(
            projected_x,
            projected_y,
            phi_normalized,
            color=color,
            linewidth=1.0,
            label=f"({x_value}, {y_value})",
        )
        marked_x = (1.0 + torch.cos(x_value * marked_phi)) / 2.0
        marked_y = (1.0 + torch.cos(y_value * marked_phi)) / 2.0
        marked_z = marked_phi / phi[-1]
        axis.scatter(marked_x, marked_y, marked_z, marker="+", color=color)
        for frequency_index, x_mark, y_mark, z_mark in zip(
            range(1, 5), marked_x, marked_y, marked_z
        ):
            rows.append(
                {
                    "point_x": x_value,
                    "point_y": y_value,
                    "frequency_index": frequency_index,
                    "projected_x": float(x_mark),
                    "projected_y": float(y_mark),
                    "normalized_phi": float(z_mark),
                }
            )
    axis.set(
        xlabel="$C_\\phi(x)$",
        ylabel="$C_\\phi(y)$",
        zlabel="normalized $\\phi$",
        title="Figure 11 — frequency-parameterized curves",
    )
    axis.legend()
    _save_figure(fig, path)
    plt.close(fig)
    _write_csv(
        data_path,
        rows,
        (
            "point_x",
            "point_y",
            "frequency_index",
            "projected_x",
            "projected_y",
            "normalized_phi",
        ),
    )
    return {
        "status": "generated",
        "path": str(path),
        "data": str(data_path),
        "verification": "analytic_with_explicit_third_coordinate_assumption",
        "paper_ambiguity": "The Appendix source names a 3D figure but does not define its third coordinate; normalized phi is frozen here.",
        "continuous_phi_interval": [0.0, 20.0 * math.pi],
    }


def odd_alignment_report() -> dict[str, object]:
    """Validate exact vertex alignment on an odd 9x7 synthetic signal."""

    from peps import GridEncoder

    height, width = 7, 9
    encoder = GridEncoder(
        dim=2,
        resolution=(height, width),
        feature_dim=1,
        align_corners=True,
    )
    target = torch.arange(height * width, dtype=torch.float32).reshape(
        1,
        1,
        height,
        width,
    )
    with torch.no_grad():
        encoder.grid.copy_(target)
    aligned = coords_grid(height, width)
    with torch.no_grad():
        aligned_prediction = encoder(aligned).reshape(height, width)
    expected = target[0, 0]
    half_x = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    half_y = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    shifted_y, shifted_x = torch.meshgrid(half_y, half_x, indexing="ij")
    shifted = torch.stack((shifted_x.reshape(-1), shifted_y.reshape(-1)), dim=1)
    with torch.no_grad():
        shifted_prediction = encoder(shifted).reshape(height, width)
    return {
        "schema": "peps.odd_alignment_validation",
        "schema_version": 1,
        "resolution_xy": [width, height],
        "align_corners": True,
        "canonical_coordinate_formula": ["x/(W-1)", "y/(H-1)"],
        "half_texel_formula": ["(x+0.5)/W", "(y+0.5)/H"],
        "aligned_max_abs_error": float(
            (aligned_prediction - expected).abs().max()
        ),
        "half_texel_mean_abs_error": float(
            (shifted_prediction - expected).abs().mean()
        ),
        "status": (
            "passed"
            if float((aligned_prediction - expected).abs().max()) <= 1e-5
            else "failed"
        ),
        "scope": "synthetic coordinate-contract validation, not a paper texture metric",
    }


def _radial_psd(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return channel-averaged radial power using native image samples."""

    if image.ndim != 3:
        raise ValueError("PSD input must have shape HWC")
    channels = image.permute(2, 0, 1).to(device=device, dtype=torch.float32)
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(channels, norm="ortho"),
        dim=(-2, -1),
    )
    power = spectrum.abs().square().mean(dim=0)
    height, width = power.shape
    y = torch.arange(height, device=device, dtype=torch.float32) - height // 2
    x = torch.arange(width, device=device, dtype=torch.float32) - width // 2
    radius = torch.floor(torch.sqrt(y[:, None].square() + x[None, :].square())).long()
    bins = int(radius.max().item()) + 1
    radial_sum = torch.zeros(bins, device=device, dtype=torch.float64)
    radial_sum.scatter_add_(0, radius.reshape(-1), power.double().reshape(-1))
    counts = torch.bincount(radius.reshape(-1), minlength=bins).clamp_min(1)
    # Bins beyond min(H, W)/2 contain only diagonal/corner frequencies and
    # develop a sparse-count tail that is not an isotropic radial statistic.
    isotropic_nyquist_bins = min(height, width) // 2 + 1
    return (radial_sum / counts)[:isotropic_nyquist_bins].cpu()


def _normalized_shape(psd: torch.Tensor) -> torch.Tensor:
    if psd.numel() < 3:
        raise ValueError("PSD needs at least three radial bins")
    reference = psd[1].clamp_min(torch.finfo(psd.dtype).tiny)
    return psd / reference


def figure3_psd(
    path: Path,
    data_path: Path,
    *,
    device: torch.device,
    include_textures: bool,
) -> dict[str, object]:
    plt = _pyplot()
    kodak = load_paper_kodak()
    kodak_shapes = []
    for image in kodak:
        kodak_shapes.append(_normalized_shape(_radial_psd(image.tensor, device)))
    common_bins = min(value.numel() for value in kodak_shapes)
    kodak_mean = torch.stack(
        [value[:common_bins] for value in kodak_shapes]
    ).mean(dim=0)

    texture_shapes = []
    texture_signals = 0
    texture_receipts = []
    texture_error = None
    if include_textures:
        try:
            for set_id in PAPER_TEXTURE_PSD_IDS:
                loaded = load_paper_texture_set(set_id)
                for texture_map in loaded.maps:
                    map_tensor = loaded.tensor[..., texture_map.channel_slice]
                    texture_shapes.append(
                        _normalized_shape(_radial_psd(map_tensor, device))
                    )
                    texture_signals += 1
                    texture_receipts.append(
                        {
                            "set_id": set_id,
                            "map_id": texture_map.map_id,
                            "semantic": texture_map.semantic,
                            "path": str(texture_map.source_path),
                            "sha256": hash_file(texture_map.source_path, "sha256"),
                        }
                    )
                del loaded
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        except Exception as exc:
            texture_error = f"{type(exc).__name__}: {exc}"
            texture_shapes = []
            texture_signals = 0
            texture_receipts = []
    texture_mean = None
    if texture_shapes:
        texture_bins = min(value.numel() for value in texture_shapes)
        texture_mean = torch.stack(
            [value[:texture_bins] for value in texture_shapes]
        ).mean(dim=0)

    rows = [
        {
            "dataset": "Kodak-24",
            "frequency_bin": frequency,
            "normalized_power": float(kodak_mean[frequency]),
            "signal_count": len(kodak),
            "normalization": "per-signal radial PSD divided by bin 1",
        }
        for frequency in range(1, kodak_mean.numel())
    ]
    if texture_mean is not None:
        rows.extend(
            {
                "dataset": "paper-texture-10-assumption",
                "frequency_bin": frequency,
                "normalized_power": float(texture_mean[frequency]),
                "signal_count": texture_signals,
                "normalization": "per-map radial PSD divided by bin 1",
            }
            for frequency in range(1, texture_mean.numel())
        )
    _write_csv(
        data_path,
        rows,
        (
            "dataset",
            "frequency_bin",
            "normalized_power",
            "signal_count",
            "normalization",
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    if texture_mean is None:
        axes[0].axis("off")
        axes[0].text(
            0.5,
            0.5,
            "Texture PSD blocked\n" + (texture_error or "not requested"),
            ha="center",
            va="center",
        )
    else:
        frequencies = np.arange(1, texture_mean.numel())
        axes[0].loglog(frequencies, texture_mean[1:], label="10 texture sets")
        axes[0].loglog(
            frequencies,
            frequencies.astype(float) ** -1,
            linestyle="--",
            label="$1/f$ reference",
        )
        axes[0].set_title("texture maps")
        axes[0].legend()
    frequencies = np.arange(1, kodak_mean.numel())
    axes[1].loglog(frequencies, kodak_mean[1:], label="Kodak 24")
    axes[1].loglog(
        frequencies,
        frequencies.astype(float) ** -1,
        linestyle="--",
        label="$1/f$ reference",
    )
    axes[1].loglog(
        frequencies,
        frequencies.astype(float) ** -2,
        linestyle=":",
        label="$1/f^2$ reference",
    )
    axes[1].set_title("Kodak images")
    axes[1].legend()
    for axis in axes:
        if axis.axison:
            axis.set_xlabel("radial frequency bin")
            axis.set_ylabel("normalized power")
            axis.grid(alpha=0.2)
    fig.suptitle("Figure 3 — radial power spectral density")
    _save_figure(fig, path)
    plt.close(fig)
    return {
        "status": (
            "generated"
            if texture_mean is not None
            else "partial_kodak_generated_texture_blocked"
        ),
        "path": str(path),
        "data": str(data_path),
        "kodak_images": 24,
        "kodak_original_resolution": True,
        "texture_set_ids": list(PAPER_TEXTURE_PSD_IDS),
        "texture_map_signals": texture_signals,
        "texture_receipts": texture_receipts,
        "texture_error": texture_error,
        "protocol_assumption": "The paper says 10 textures but does not identify them or PSD normalization; the first ten paper-listed sets, per-signal bin-1 normalization, and the isotropic min(H,W)/2 Nyquist disk are frozen here.",
    }


def figure7_scatter(
    path: Path,
    data_path: Path,
    *,
    output_root: Path,
) -> dict[str, object]:
    progress = artifact_progress("table1", output_root=output_root)
    if not progress["complete"]:
        return {
            "status": "blocked",
            "reason": "Figure 7 requires all 648 Table 1 instance/method/seed records.",
            "table1_progress": progress,
        }
    records = collect_raw_records(_artifact_output(output_root, "table1"))
    values: dict[tuple[str, str], list[float]] = {}
    for record in records:
        values.setdefault(
            (str(record["instance"]), str(record["method"])),
            [],
        ).append(float(record["metrics"]["psnr"]))
    means = {
        key: sum(items) / len(items) for key, items in values.items()
    }
    comparisons = (
        ("Grid", "G-PEPS"),
        ("Grid", "G-P-PEPS"),
        ("NTC_N", "NTC_PinkPEPS"),
        ("LPE", "Grid"),
        ("Grid", "G-P-PEPS-25"),
        ("G-PEPS", "G-P-PEPS"),
    )
    rows = []
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    instances = sorted({record["instance"] for record in records})
    for axis, (left, right) in zip(axes.reshape(-1), comparisons):
        x_values = [means[(instance, left)] for instance in instances]
        y_values = [means[(instance, right)] for instance in instances]
        lower = min(*x_values, *y_values)
        upper = max(*x_values, *y_values)
        axis.scatter(x_values, y_values, s=20)
        axis.plot([lower, upper], [lower, upper], color="black", linestyle="--")
        axis.set(xlabel=left, ylabel=right, title=f"{left} vs {right}")
        axis.grid(alpha=0.2)
        for instance, left_value, right_value in zip(
            instances,
            x_values,
            y_values,
        ):
            rows.append(
                {
                    "instance": instance,
                    "seed_reduction": "mean_of_3",
                    "left_method": left,
                    "right_method": right,
                    "left_psnr": left_value,
                    "right_psnr": right_value,
                    "delta_right_minus_left": right_value - left_value,
                }
            )
    fig.suptitle("Figure 7 — 24-image dual PSNR scatters")
    _save_figure(fig, path)
    plt.close(fig)
    _write_csv(
        data_path,
        rows,
        (
            "instance",
            "seed_reduction",
            "left_method",
            "right_method",
            "left_psnr",
            "right_psnr",
            "delta_right_minus_left",
        ),
    )
    return {
        "status": "generated",
        "path": str(path),
        "data": str(data_path),
        "records": len(records),
        "instances": 24,
        "seeds": 3,
    }


def figure6_qualitative(
    path: Path,
    *,
    output_root: Path,
    device: torch.device,
    instance_id: str = "kodim24",
    seed: int = 0,
) -> dict[str, object]:
    progress = artifact_progress("table1", output_root=output_root)
    if not progress["complete"]:
        return {
            "status": "blocked",
            "reason": "Figure 6 is not emitted until Table 1 is complete; partial checkpoints must not be presented as the paper qualitative result.",
            "table1_progress": progress,
        }
    config = load_experiment_config(ARTIFACT_CONFIGS["table1"])
    loaded = load_paper_kodak(instance_ids=(instance_id,))[0]
    coords, targets, (height, width) = image_to_coords_targets(loaded.tensor)
    instance = TensorInstance(
        instance_id,
        coords,
        targets,
        shape=(height, width, 3),
    )
    methods = ("Grid", "G-PEPS", "G-P-PEPS", "NTC_N", "NTC_PinkPEPS")
    predictions = []
    checkpoint_receipts = []
    output_dir = _artifact_output(output_root, "table1")
    for method_name in methods:
        method = next(item for item in config.methods if item.name == method_name)
        model, _ = _build_model(config, method, instance)
        _, checkpoint_path = _job_paths(
            output_dir,
            instance_id,
            method_name,
            seed,
        )
        state = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(state["model"])
        prediction = render_full(
            model,
            coords,
            chunk=int(config.runner["render_chunk"]),
            device=device,
        ).reshape(height, width, 3)
        predictions.append(prediction)
        checkpoint_receipts.append(
            {
                "method": method_name,
                "path": str(checkpoint_path),
                "sha256": hash_file(checkpoint_path, "sha256"),
                "step": int(state["step"]),
            }
        )
        del model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    import flip_evaluator

    crop_size = 100
    top = max(0, (height - crop_size) // 2)
    left = max(0, (width - crop_size) // 2)
    reference = loaded.tensor.numpy()
    images = [loaded.tensor, *predictions]
    labels = ["reference", *methods]
    flip_maps = [None]
    for prediction in predictions:
        error_map, _, _ = flip_evaluator.evaluate(
            reference,
            prediction.numpy(),
            "LDR",
        )
        flip_maps.append(np.asarray(error_map))
    plt = _pyplot()
    fig, axes = plt.subplots(2, len(images), figsize=(14, 5.2))
    for column, (label, image, error) in enumerate(
        zip(labels, images, flip_maps)
    ):
        crop = image[top : top + crop_size, left : left + crop_size].clamp(0, 1)
        axes[0, column].imshow(crop.numpy())
        axes[0, column].set_title(label, fontsize=9)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
        if error is not None:
            if error.ndim == 3:
                error = error.mean(axis=-1)
            axes[1, column].imshow(
                error[top : top + crop_size, left : left + crop_size],
                cmap="magma",
                vmin=0,
                vmax=1,
            )
    fig.suptitle(
        f"Figure 6 — {instance_id} center {crop_size}×{crop_size} crop and FLIP map"
    )
    _save_figure(fig, path)
    plt.close(fig)
    return {
        "status": "generated",
        "path": str(path),
        "instance": instance_id,
        "seed": seed,
        "crop_xywh": [left, top, crop_size, crop_size],
        "checkpoints": checkpoint_receipts,
    }


def generate_analytic(output_dir: Path) -> dict[str, object]:
    figure_dir = output_dir / "figures"
    data_dir = output_dir / "figure_data"
    status = {
        "1": figure1_workflow(figure_dir / "figure01_workflow.png"),
        "2": figure2_rotation(figure_dir / "figure02_rotation.png"),
        "4": figure4_pink_allocation(
            figure_dir / "figure04_pink_allocation.png",
            data_dir / "figure04_pink_allocation.csv",
        ),
        "10": figure10_lissajous(
            figure_dir / "figure10_lissajous.png",
            data_dir / "figure10_lissajous.csv",
        ),
        "11": figure11_unique_curves(
            figure_dir / "figure11_unique_curves.png",
            data_dir / "figure11_unique_curves.csv",
        ),
    }
    atomic_write_json(output_dir / "odd_alignment_validation.json", odd_alignment_report())
    return status


def generate_dependent(
    output_dir: Path,
    *,
    output_root: Path,
    device: torch.device,
) -> dict[str, object]:
    figure_dir = output_dir / "figures"
    data_dir = output_dir / "figure_data"
    return {
        "5": {
            "status": "blocked",
            "reason": "The paper does not identify the native-4K image suite or optimizer steps. No exact Figure 5 numbers are generated.",
            "available_runner": "python -m experiments.reproduce run --artifact image-fig5 --fig5-manifest ... --assumed-steps ... --allow-protocol-assumptions",
        },
        "6": figure6_qualitative(
            figure_dir / "figure06_kodak_qualitative.png",
            output_root=output_root,
            device=device,
        ),
        "7": figure7_scatter(
            figure_dir / "figure07_dual_scatter.png",
            data_dir / "figure07_dual_scatter.csv",
            output_root=output_root,
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("analytic", "psd", "dependent", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "image_repro",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-textures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    device = torch.device(
        "cuda:0"
        if arguments.device == "auto" and torch.cuda.is_available()
        else ("cpu" if arguments.device == "auto" else arguments.device)
    )
    status_path = arguments.output_dir / "figure_status.json"
    statuses: dict[str, object] = {}
    if arguments.mode != "all" and status_path.is_file():
        try:
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            statuses.update(previous.get("figures", {}))
        except (OSError, json.JSONDecodeError):
            statuses = {}
    if arguments.mode in {"analytic", "all"}:
        statuses.update(generate_analytic(arguments.output_dir))
    if arguments.mode in {"psd", "all"}:
        statuses["3"] = figure3_psd(
            arguments.output_dir / "figures/figure03_psd.png",
            arguments.output_dir / "figure_data/figure03_psd.csv",
            device=device,
            include_textures=not arguments.skip_textures,
        )
    if arguments.mode in {"dependent", "all"}:
        statuses.update(
            generate_dependent(
                arguments.output_dir,
                output_root=arguments.output_root,
                device=device,
            )
        )
    payload = {
        "schema": "peps.image_figure_status",
        "schema_version": FIGURE_SCHEMA_VERSION,
        "paper": "PEPS Extended arXiv:2604.24167v1",
        "generated_mode": arguments.mode,
        "device": str(device),
        "figures": statuses,
        "no_placeholder_numbers": True,
    }
    atomic_write_json(status_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
