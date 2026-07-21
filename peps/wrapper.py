"""PEPS wrapper — Project, Encode, Aggregate, Model (paper Eq. 8)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .projector import Projector


class PEPS(nn.Module):
    """Assemble a projector, one or more encoders, aggregator, and decoder.

    Args:
        projector: a :class:`Projector`.
        encoder: either one shared encoder or a sequence containing one encoder
            per projected point (the general ``E_1, ..., E_(2L+1)`` in Eq. 8).
        aggregator: combines ``(N, num_points, k) -> (N, agg_out)``.
        model: decoder MLP mapping ``(N, agg_out + delta_dim) -> (N, out_dim)``.
        append_input_delta: if True, concatenate the raw input coords to the
            aggregated vector before the MLP (the ``delta`` in Eq. 8).
        selective_sampling: ask encoders for only the channel slices required by
            the aggregator. Encoders may implement
            ``sample_channels(coords, channel_indices)``; otherwise a
            semantically equivalent full-encode fallback is used.
    """

    def __init__(
        self,
        projector: Projector,
        encoder,
        aggregator: nn.Module,
        model: nn.Module,
        append_input_delta: bool = False,
        selective_sampling: bool = False,
    ) -> None:
        super().__init__()
        self.projector = projector
        self.aggregator = aggregator
        self.model = model
        self.append_input_delta = bool(append_input_delta)
        self.selective_sampling = bool(selective_sampling)

        if isinstance(encoder, nn.Module):
            self.encoder = encoder
            self.encoders = None
            self.shared_encoder = True
        else:
            try:
                encoders = list(encoder)
            except TypeError as exc:
                raise TypeError(
                    "encoder must be an nn.Module or a sequence of modules"
                ) from exc
            if len(encoders) != projector.num_points:
                raise ValueError(
                    f"expected {projector.num_points} per-point encoders, "
                    f"got {len(encoders)}"
                )
            if not all(isinstance(item, nn.Module) for item in encoders):
                raise TypeError("every per-point encoder must be an nn.Module")
            self.encoder = None
            self.encoders = nn.ModuleList(encoders)
            self.shared_encoder = False

        aggregator_points = getattr(aggregator, "num_points", None)
        if (
            aggregator_points is not None
            and aggregator_points != projector.num_points
        ):
            raise ValueError(
                f"projector returns {projector.num_points} points but aggregator "
                f"expects {aggregator_points}"
            )
        if self.selective_sampling:
            if not callable(
                getattr(aggregator, "channel_indices_for_point", None)
            ) or not callable(getattr(aggregator, "aggregate_selected", None)):
                raise TypeError(
                    "selective_sampling requires an aggregator exposing "
                    "channel_indices_for_point and aggregate_selected"
                )

    def _encoder_for_point(self, point: int) -> nn.Module:
        if self.shared_encoder:
            assert self.encoder is not None
            return self.encoder
        assert self.encoders is not None
        return self.encoders[point]

    @staticmethod
    def _validate_encoded(
        latent: torch.Tensor, expected_batch: int, point: int | None = None
    ) -> torch.Tensor:
        label = "shared encoder" if point is None else f"encoder for point {point}"
        if not isinstance(latent, torch.Tensor) or latent.ndim != 2:
            raise ValueError(f"{label} must return a tensor with shape (N, k)")
        if latent.shape[0] != expected_batch:
            raise ValueError(
                f"{label} returned batch {latent.shape[0]}, expected "
                f"{expected_batch}"
            )
        return latent

    def _encode_selected(
        self, encoder: nn.Module, coords: torch.Tensor, point: int
    ) -> torch.Tensor:
        indices = self.aggregator.channel_indices_for_point(
            point, device=coords.device
        )
        sampler = getattr(encoder, "sample_channels", None)
        if callable(sampler):
            latent = sampler(coords, indices)
        else:
            full = self._validate_encoded(
                encoder(coords), coords.shape[0], point
            )
            latent = full.index_select(1, indices)
        return self._validate_encoded(latent, coords.shape[0], point)

    def forward(
        self, x: torch.Tensor, delta: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must have shape (N, dim), got {tuple(x.shape)}")
        n, dim = x.shape
        pts = self.projector(x)
        p = pts.shape[1]
        if p == 0:
            raise ValueError("PEPS requires at least one projected point")

        if self.selective_sampling:
            selected = [
                self._encode_selected(
                    self._encoder_for_point(point), pts[:, point, :], point
                )
                for point in range(p)
            ]
            vec = self.aggregator.aggregate_selected(selected)
        elif self.shared_encoder:
            flat = pts.reshape(n * p, dim)
            assert self.encoder is not None
            lat = self._validate_encoded(self.encoder(flat), n * p)
            lat = lat.reshape(n, p, lat.shape[-1])
            vec = self.aggregator(lat)
        else:
            point_latents = [
                self._validate_encoded(
                    self._encoder_for_point(point)(pts[:, point, :]),
                    n,
                    point,
                )
                for point in range(p)
            ]
            aggregate_points = getattr(
                self.aggregator, "forward_points", None
            )
            if callable(aggregate_points):
                vec = aggregate_points(point_latents)
            else:
                try:
                    stacked = torch.stack(point_latents, dim=1)
                except RuntimeError as exc:
                    raise ValueError(
                        "per-point encoder outputs must have equal widths when "
                        "the aggregator has no forward_points method"
                    ) from exc
                vec = self.aggregator(stacked)

        if self.append_input_delta:
            vec = torch.cat([vec, x], dim=1)
        if delta is not None:
            if delta.ndim != 2 or delta.shape[0] != n:
                raise ValueError(
                    f"delta must have shape ({n}, q), got {tuple(delta.shape)}"
                )
            vec = torch.cat([vec, delta], dim=1)
        return self.model(vec)

    @property
    def num_points(self) -> int:
        return self.projector.num_points
