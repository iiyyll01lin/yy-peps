"""Paper evaluation metrics with lazy optional dependencies."""

from __future__ import annotations

import importlib.metadata
import math
from dataclasses import dataclass

import torch


PAPER_METRIC_ORACLES = {
    "psnr": {
        "equation": "10*log10(data_range^2/mean((prediction-target)^2))",
        "data_range": 1.0,
    },
    "ssim": {
        "implementation": "torchmetrics.functional.image.structural_similarity_index_measure",
        "gaussian_kernel": True,
        "kernel_size": 11,
        "sigma": 1.5,
        "k1": 0.01,
        "k2": 0.03,
    },
    "flip": {
        "implementation": "flip_evaluator.evaluate",
        "mode": "LDR",
        "reduction": "mean_error",
    },
    "lpips": {
        "implementation": "lpips.LPIPS",
        "network": "alex",
        "input_range": [-1.0, 1.0],
    },
    "lsd": {
        "implementation": "peps.metrics.lsd",
        "equation": "mean_channel(rms(log1p(|FFT2_ortho(pred)|)-log1p(|FFT2_ortho(target)|)))",
        "ambiguity": "The PEPS paper names LSD but does not publish its normalization; this frozen oracle is an explicit reproduction assumption.",
    },
    "iou": {
        "equation": "count(prediction<0 and target<0)/count(prediction<0 or target<0)",
    },
}


@dataclass
class IoUAccumulator:
    """Exact integer accumulator for streamed 512^3 occupancy evaluation."""

    intersection: int = 0
    union: int = 0

    def update(
        self,
        pred_occupancy: torch.Tensor,
        target_occupancy: torch.Tensor,
    ) -> None:
        prediction = pred_occupancy.bool()
        target = target_occupancy.bool()
        if prediction.shape != target.shape:
            raise ValueError("occupancy tensors must have matching shapes")
        self.intersection += int((prediction & target).sum().item())
        self.union += int((prediction | target).sum().item())

    def compute(self) -> float:
        return self.intersection / self.union if self.union else 1.0


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB. Inputs in the same range as data_range."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def _as_nchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        return image.unsqueeze(0).unsqueeze(0)
    if image.ndim == 3:
        if image.shape[-1] in {1, 3, 4}:
            return image.permute(2, 0, 1).unsqueeze(0)
        if image.shape[0] in {1, 3, 4}:
            return image.unsqueeze(0)
        raise ValueError("rank-3 images must have an explicit 1/3/4 channel axis")
    if image.ndim == 4:
        if image.shape[-1] in {1, 3, 4}:
            return image.permute(0, 3, 1, 2)
        if image.shape[1] in {1, 3, 4}:
            return image
    raise ValueError("image must have shape HW, HWC, CHW, NHWC, or NCHW")


def _check_pair(
    pred: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = _as_nchw(pred)
    reference = _as_nchw(target)
    if prediction.shape != reference.shape:
        raise ValueError(
            f"metric inputs must have matching shapes, got "
            f"{tuple(prediction.shape)} and {tuple(reference.shape)}"
        )
    return prediction, reference


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Windowed SSIM using the implementation named by the paper."""

    try:
        from torchmetrics.functional.image import (
            structural_similarity_index_measure,
        )
    except ImportError as exc:
        raise ImportError(
            "paper SSIM requires the 'torchmetrics' optional dependency"
        ) from exc
    prediction, reference = _check_pair(pred, target)
    with torch.no_grad():
        value = structural_similarity_index_measure(
            prediction.float(),
            reference.float(),
            data_range=data_range,
            gaussian_kernel=True,
            sigma=1.5,
            kernel_size=11,
            k1=0.01,
            k2=0.03,
            reduction="elementwise_mean",
        )
    return float(value.item())


def lsd(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Paper-scale log-spectral distance on the 2D Fourier amplitude.

    ``log1p`` avoids allowing almost-zero Fourier bins to dominate the score,
    which was the scale error in the earlier implementation.
    """

    prediction, reference = _check_pair(pred, target)
    prediction_spectrum = torch.log1p(
        torch.fft.fft2(prediction.double(), norm="ortho").abs()
    )
    reference_spectrum = torch.log1p(
        torch.fft.fft2(reference.double(), norm="ortho").abs()
    )
    per_image_channel = (
        (prediction_spectrum - reference_spectrum)
        .square()
        .mean(dim=(-2, -1))
        .sqrt()
    )
    return float(per_image_channel.mean().item())


def _radial_power_spectrum(image: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(image.double(), norm="ortho"),
        dim=(-2, -1),
    )
    power = spectrum.abs().square()
    height, width = power.shape[-2:]
    y = torch.arange(height, device=image.device, dtype=torch.float64)
    x = torch.arange(width, device=image.device, dtype=torch.float64)
    radius = torch.floor(
        torch.sqrt(
            (y[:, None] - height // 2).square()
            + (x[None, :] - width // 2).square()
        )
    ).long()
    bins = int(radius.max().item()) + 1
    indices = radius.reshape(1, -1).expand(power.shape[0] * power.shape[1], -1)
    flattened = power.reshape(power.shape[0] * power.shape[1], -1)
    radial_sum = power.new_zeros((flattened.shape[0], bins))
    radial_sum.scatter_add_(1, indices, flattened)
    counts = torch.bincount(radius.reshape(-1), minlength=bins).clamp_min(1)
    return radial_sum / counts.to(dtype=radial_sum.dtype).unsqueeze(0)


def lpsd(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Log-power spectral distance after radial PSD aggregation.

    The paper defines LPSD as the L1 distance between the two log radial power
    spectra.  Circular translations therefore leave this metric unchanged.
    """

    prediction, reference = _check_pair(pred, target)
    prediction_psd = torch.log1p(_radial_power_spectrum(prediction))
    reference_psd = torch.log1p(_radial_power_spectrum(reference))
    return float((prediction_psd - reference_psd).abs().mean().item())


def iou(pred_occupancy: torch.Tensor, target_occupancy: torch.Tensor) -> float:
    """Intersection-over-Union of two boolean occupancy grids (SDF, W09)."""
    accumulator = IoUAccumulator()
    accumulator.update(pred_occupancy, target_occupancy)
    return accumulator.compute()


def mean_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error used to audit L1 SDF runs."""

    if pred.shape != target.shape:
        raise ValueError("metric inputs must have matching shapes")
    return float((pred - target).abs().mean().item())


def mean_absolute_percentage_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    percentage_scale: float = 100.0,
) -> float:
    """Explicit finite MAPE oracle matching :func:`peps.train.mape_loss`."""

    if pred.shape != target.shape:
        raise ValueError("metric inputs must have matching shapes")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    value = (
        percentage_scale
        * ((target - pred).abs() / target.abs().clamp_min(epsilon)).mean()
    )
    return float(value.item())


def metric_oracles() -> dict[str, dict[str, object]]:
    """Detached metric definitions suitable for run manifests."""

    return {
        name: dict(definition)
        for name, definition in PAPER_METRIC_ORACLES.items()
    }


_lpips_models = {}


def lpips(pred: torch.Tensor, target: torch.Tensor, net: str = "alex") -> float:
    """LPIPS perceptual distance. Lazily imports the ``lpips`` package.

    Inputs ``(H, W, 3)`` in ``[0, 1]``. Returns a scalar (lower is better).
    """
    import lpips as _lpips_pkg  # deferred: optional dependency

    prediction, reference = _check_pair(pred, target)
    if prediction.shape[1] != 3:
        raise ValueError("LPIPS requires three-channel RGB images")
    device = prediction.device
    key = (net, str(device))
    if key not in _lpips_models:
        _lpips_models[key] = _lpips_pkg.LPIPS(net=net).to(device).eval()

    with torch.no_grad():
        d = _lpips_models[key](
            prediction.float() * 2.0 - 1.0,
            reference.float() * 2.0 - 1.0,
        )
    return float(d.item())


def flip(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean official LDR-FLIP error, with target passed as the reference."""

    try:
        import flip_evaluator
    except ImportError as exc:
        raise ImportError(
            "paper FLIP requires the 'flip-evaluator' optional dependency"
        ) from exc
    prediction, reference = _check_pair(pred, target)
    if prediction.shape[0] != 1 or prediction.shape[1] != 3:
        raise ValueError("FLIP requires one three-channel RGB image")
    prediction_hwc = (
        prediction[0].permute(1, 2, 0).detach().cpu().float().numpy()
    )
    reference_hwc = reference[0].permute(1, 2, 0).detach().cpu().float().numpy()
    _, mean_error, _ = flip_evaluator.evaluate(
        reference_hwc,
        prediction_hwc,
        "LDR",
    )
    return float(mean_error)


def metric_versions() -> dict[str, str | None]:
    """Versions recorded alongside every raw paper metric row."""

    versions = {"torch": torch.__version__}
    for distribution in ("torchmetrics", "lpips", "flip-evaluator"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions
