"""Shared training loop for coordinate-based fitting.

繁體中文:共用訓練迴圈。給定一個 model(接受座標、輸出訊號)、座標張量與目標張量,
做小批次的座標回歸(MSE)。影像、材質、SDF 三個應用都共用這個迴圈。
支援 ROCm/CUDA GPU;device 由呼叫端決定,預設自動偵測。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
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


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(prediction, target)


def l2_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(prediction, target)


def mape_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    percentage_scale: float = 100.0,
) -> torch.Tensor:
    """Finite version of the paper's mean absolute percentage error.

    The source paper does not state its zero-distance convention.  ``epsilon``
    is therefore explicit and recorded in every config rather than hidden in
    the implementation.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    denominator = target.abs().clamp_min(epsilon)
    return percentage_scale * ((target - prediction).abs() / denominator).mean()


@dataclass(frozen=True)
class PaperTrainConfig:
    """Training protocol shared by the paper experiment runner."""

    task: str
    loss: str
    batch_size: int
    model_lr: float
    encoder_lr: float | None = None
    steps: int | None = None
    epochs: int = 1
    batches_per_epoch: int = 1
    cosine: bool = False
    weight_decay: float = 0.0
    log_every: int = 200
    checkpoint_every: int = 0
    seed: int = 0
    mape_epsilon: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        if self.task not in {"image", "texture", "sdf"}:
            raise ValueError("task must be image, texture, or sdf")
        if self.loss not in {"l1", "l2", "mape"}:
            raise ValueError("loss must be l1, l2, or mape")
        integer_fields = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "batches_per_epoch": self.batches_per_epoch,
            "log_every": self.log_every,
            "checkpoint_every": self.checkpoint_every,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.steps is not None:
            if isinstance(self.steps, bool) or not isinstance(self.steps, int):
                raise TypeError("steps must be an integer or None")
            if self.steps < 1:
                raise ValueError("steps must be positive")
        elif self.epochs < 1 or self.batches_per_epoch < 1:
            raise ValueError("epochs and batches_per_epoch must be positive")
        if self.log_every < 1:
            raise ValueError("log_every must be positive")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be non-negative")
        if self.model_lr <= 0:
            raise ValueError("model_lr must be positive")
        if self.encoder_lr is not None and self.encoder_lr <= 0:
            raise ValueError("encoder_lr must be positive")
        if self.mape_epsilon <= 0:
            raise ValueError("mape_epsilon must be positive")

    @property
    def total_steps(self) -> int:
        if self.steps is not None:
            return self.steps
        return self.epochs * self.batches_per_epoch


def paper_image_recipe(
    *,
    ablation: str = "main",
    steps: int = 120_000,
    batch_size: int = 60_000,
    seed: int = 0,
) -> PaperTrainConfig:
    """Table 1 main recipe or the Table 5 adaptive L2 ablation."""

    if ablation == "main":
        return PaperTrainConfig(
            task="image",
            loss="l1",
            steps=steps,
            batch_size=batch_size,
            model_lr=0.01,
            encoder_lr=0.01,
            cosine=False,
            seed=seed,
        )
    if ablation in {"l2_dual_cosine", "table5"}:
        return PaperTrainConfig(
            task="image",
            loss="l2",
            steps=steps,
            batch_size=batch_size,
            model_lr=0.001,
            encoder_lr=0.1,
            cosine=True,
            seed=seed,
        )
    raise ValueError(f"unknown image ablation: {ablation!r}")


def paper_texture_recipe(
    *,
    epochs: int = 3000,
    batches_per_epoch: int = 40,
    batch_size: int = 60_000,
    seed: int = 0,
) -> PaperTrainConfig:
    return PaperTrainConfig(
        task="texture",
        loss="l1",
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        batch_size=batch_size,
        model_lr=0.001,
        encoder_lr=0.1,
        cosine=True,
        seed=seed,
    )


def paper_sdf_recipe(
    *,
    loss: str = "mape",
    epochs: int = 3000,
    batches_per_epoch: int = 40,
    batch_size: int = 60_000,
    seed: int = 0,
    mape_epsilon: float = 1e-6,
) -> PaperTrainConfig:
    if loss not in {"mape", "l1"}:
        raise ValueError("paper SDF loss must be mape or l1")
    return PaperTrainConfig(
        task="sdf",
        loss=loss,
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        batch_size=batch_size,
        model_lr=0.001 if loss == "mape" else 0.01,
        encoder_lr=0.001 if loss == "mape" else 0.01,
        cosine=False,
        seed=seed,
        mape_epsilon=mape_epsilon,
    )


def paper_recipe_from_mapping(values: Mapping) -> PaperTrainConfig:
    """Construct a validated recipe from a TOML ``[training]`` table."""

    allowed = set(PaperTrainConfig.__dataclass_fields__)
    unexpected = set(values) - allowed
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(f"unknown training config fields: {names}")
    return PaperTrainConfig(**dict(values))


class MinibatchStream:
    """Serializable deterministic random-index stream shared across methods."""

    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        if size < 1:
            raise ValueError("size must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)
        self.draws = 0

    def next(self) -> torch.Tensor:
        self.draws += 1
        if self.batch_size >= self.size:
            return torch.arange(self.size)
        return torch.randint(
            self.size,
            (self.batch_size,),
            generator=self.generator,
        )

    def state_dict(self) -> dict:
        return {
            "size": self.size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "draws": self.draws,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping) -> None:
        for field_name in ("size", "batch_size", "seed"):
            if int(state[field_name]) != getattr(self, field_name):
                raise ValueError(f"minibatch stream {field_name} does not match")
        self.draws = int(state["draws"])
        self.generator.set_state(state["generator_state"])


def split_encoder_decoder_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return disjoint encoder and decoder parameter lists."""

    decoder = None
    if isinstance(getattr(model, "model", None), nn.Module):
        decoder = model.model
    elif isinstance(getattr(model, "decoder", None), nn.Module):
        decoder = model.decoder
    elif isinstance(model, nn.Sequential) and len(model) > 1:
        decoder = model[-1]

    decoder_parameters = (
        list(decoder.parameters()) if decoder is not None else list(model.parameters())
    )
    decoder_ids = {id(parameter) for parameter in decoder_parameters}
    encoder_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in decoder_ids
    ]
    return encoder_parameters, decoder_parameters


def make_paper_optimizer(
    model: nn.Module,
    config: PaperTrainConfig,
) -> torch.optim.Optimizer:
    encoder_parameters, decoder_parameters = split_encoder_decoder_parameters(model)
    groups = []
    if encoder_parameters:
        groups.append(
            {
                "params": encoder_parameters,
                "lr": (
                    config.encoder_lr
                    if config.encoder_lr is not None
                    else config.model_lr
                ),
                "group_name": "encoder",
            }
        )
    if decoder_parameters:
        groups.append(
            {
                "params": decoder_parameters,
                "lr": config.model_lr,
                "group_name": "model",
            }
        )
    return torch.optim.Adam(
        groups,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )


def _paper_loss(config: PaperTrainConfig):
    if config.loss == "l1":
        return l1_loss
    if config.loss == "l2":
        return l2_loss
    return lambda prediction, target: mape_loss(
        prediction,
        target,
        epsilon=config.mape_epsilon,
    )


def _training_state(
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    stream: MinibatchStream,
) -> dict:
    return {
        "schema_version": 1,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "minibatch_stream": stream.state_dict(),
    }


def fit_paper(
    model: nn.Module,
    coords: torch.Tensor,
    targets: torch.Tensor,
    config: PaperTrainConfig,
    *,
    on_log: Optional[Callable[[int, float], None]] = None,
    on_checkpoint: Optional[Callable[[int, Mapping], None]] = None,
    resume_state: Mapping | None = None,
    return_state: bool = False,
):
    """Run a paper recipe without SDF extensions such as eikonal loss."""

    if coords.ndim != 2 or targets.ndim != 2:
        raise ValueError("coords and targets must both be rank-2 tensors")
    if coords.shape[0] != targets.shape[0]:
        raise ValueError("coords and targets must contain the same number of rows")

    device = config.device or auto_device()
    model = model.to(device)
    coords = coords.to(device)
    targets = targets.to(device)
    optimizer = make_paper_optimizer(model, config)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.total_steps
        )
        if config.cosine
        else None
    )
    stream = MinibatchStream(
        coords.shape[0], config.batch_size, config.seed
    )
    start_step = 0
    if resume_state is not None:
        if int(resume_state.get("schema_version", 0)) != 1:
            raise ValueError("unsupported training checkpoint schema")
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        if scheduler is not None:
            if resume_state["scheduler"] is None:
                raise ValueError("checkpoint is missing cosine scheduler state")
            scheduler.load_state_dict(resume_state["scheduler"])
        stream.load_state_dict(resume_state["minibatch_stream"])
        start_step = int(resume_state["step"])
        if start_step > config.total_steps:
            raise ValueError("checkpoint step exceeds configured total steps")

    loss_function = _paper_loss(config)
    final_state = _training_state(
        step=start_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        stream=stream,
    )
    for step_index in range(start_step, config.total_steps):
        indices = stream.next().to(device=device)
        prediction = model(coords.index_select(0, indices))
        loss = loss_function(prediction, targets.index_select(0, indices))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        completed_step = step_index + 1
        if completed_step == 1 or completed_step % config.log_every == 0:
            if on_log is not None:
                on_log(completed_step, float(loss.item()))
        should_checkpoint = (
            config.checkpoint_every > 0
            and completed_step % config.checkpoint_every == 0
        )
        if should_checkpoint or completed_step == config.total_steps:
            final_state = _training_state(
                step=completed_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                stream=stream,
            )
            if on_checkpoint is not None:
                on_checkpoint(completed_step, final_state)

    if return_state:
        return model, final_state
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
