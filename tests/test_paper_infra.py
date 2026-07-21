"""Focused CPU tests for the paper experiment infrastructure."""

from __future__ import annotations

import io
import math
from pathlib import Path

import pytest
import torch

from experiments.config import load_experiment_config
from experiments.runner import (
    ExperimentRunner,
    TensorInstance,
    enumerate_jobs,
    paired_delta,
    shard_jobs,
)
from peps.encoders.lpe import LocalPositionalEncoding
from peps.encoders.multires import HashGridEncoder, MultiResGridEncoder
from peps.encoders.ntc import (
    FourNeighborGridEncoder,
    NTCNEncoder,
    NTCPEPSEncoder,
    TiledTriangularEncoding,
)
from peps.metrics import flip, lpsd, lsd, metric_versions, ssim
from peps.train import (
    MinibatchStream,
    PaperTrainConfig,
    fit_paper,
    make_paper_optimizer,
    mape_loss,
    paper_sdf_recipe,
    paper_texture_recipe,
)


ROOT = Path(__file__).resolve().parents[1]


def test_lpe_matches_eq_6_7_basis_and_corner_interpolation():
    encoder = LocalPositionalEncoding(
        dim=2,
        resolution=(2, 2),
        num_frequencies=2,
        init_bound=0,
    )
    with torch.no_grad():
        encoder.coefficients.copy_(
            torch.arange(4, dtype=torch.float32).unsqueeze(1).expand(-1, 8)
        )

    coords = torch.tensor([[0.25, 0.25]])
    coefficients = encoder.interpolate_coefficients(coords)
    assert torch.allclose(coefficients, torch.full((1, 8), 1.5))

    expected_basis = torch.tensor(
        [[1.0, 1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0]]
    )
    assert torch.allclose(encoder.local_basis(coords), expected_basis, atol=1e-6)
    assert torch.allclose(encoder(coords), 1.5 * expected_basis, atol=1e-6)
    assert encoder.num_params == 2 * 2 * 2 * 2 * 2


def test_ntc_n_corner_order_tiling_and_width():
    corners = FourNeighborGridEncoder((2, 2), feature_dim=1, init_std=0)
    with torch.no_grad():
        corners.grid[:, 0].copy_(torch.arange(4, dtype=torch.float32))
    assert torch.equal(
        corners(torch.tensor([[0.5, 0.5]])),
        torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
    )

    tiled = TiledTriangularEncoding((17, 17))
    origin = tiled(torch.tensor([[0.0, 0.0]]))
    one_tile = tiled(torch.tensor([[0.5, 0.5]]))
    assert origin.shape == (1, 12)
    assert torch.allclose(origin, one_tile)
    integer_texels = torch.arange(17, dtype=torch.float32) / 16
    horizontal = tiled(
        torch.stack([integer_texels, torch.zeros_like(integer_texels)], dim=1)
    )
    assert torch.allclose(horizontal[:, 5], torch.zeros(17), atol=1e-6)

    encoder = NTCNEncoder(
        16,
        g0_resolution=4,
        g0_feature_dim=12,
        g1_resolution=2,
        g1_feature_dim=20,
    )
    assert encoder.feature_dim == 80
    assert encoder.num_params == 4 * 4 * 12 + 2 * 2 * 20
    assert encoder(torch.rand(3, 2)).shape == (3, 80)

    peps_encoder = NTCPEPSEncoder(
        16,
        g0_resolution=4,
        g0_feature_dim=2,
        g1_resolution=2,
        g1_feature_dim=3,
        num_frequencies=2,
    )
    # Five projected samples of 8+3 learned grid values, plus one 12D tiled PE.
    assert peps_encoder.feature_dim == 5 * 11 + 12
    assert peps_encoder(torch.rand(3, 2)).shape == (3, 67)


def test_paper_dense_and_hash_budgets_are_exact():
    multi_grid = MultiResGridEncoder(
        dim=3,
        resolutions=(16, 32, 64),
        feature_dim=2,
    )
    multi_hash = HashGridEncoder(
        dim=3,
        resolutions=(16, 32, 64, 128),
        feature_dim=2,
        log2_hashmap_size=17,
    )
    single_hash = HashGridEncoder(
        dim=3,
        resolutions=(64,),
        feature_dim=18,
        log2_hashmap_size=15,
    )
    assert multi_grid.num_params == 598_016
    assert multi_hash.table_sizes == (4096, 32768, 131072, 131072)
    assert multi_hash.num_params == 598_016
    assert single_hash.num_params == 589_824
    assert multi_hash(torch.rand(2, 3)).shape == (2, 8)


def test_texture_builder_has_dynamic_three_k_output():
    from apps.texture.build import build_paper_texture

    model, count = build_paper_texture(
        "ntc_n",
        num_textures=5,
        signal_resolution=16,
        g0_resolution=4,
        g0_feature_dim=2,
        g1_resolution=2,
        g1_feature_dim=3,
        hidden_dim=8,
        num_layers=2,
    )
    assert model(torch.rand(4, 2)).shape == (4, 15)
    assert count == sum(parameter.numel() for parameter in model.parameters())


def test_paper_losses_recipes_and_dual_optimizer_groups():
    prediction = torch.tensor([[0.0], [0.0], [1.0]])
    target = torch.tensor([[1.0], [-2.0], [0.0]])
    value = mape_loss(prediction, target, epsilon=0.5)
    expected = 100 * (1.0 + 1.0 + 2.0) / 3
    assert value.item() == pytest.approx(expected)

    texture = paper_texture_recipe()
    assert texture.total_steps == 120_000
    assert texture.model_lr == 0.001
    assert texture.encoder_lr == 0.1
    assert texture.cosine
    sdf = paper_sdf_recipe(loss="l1")
    assert sdf.model_lr == 0.01

    from apps.image.build import build_grid

    model, _ = build_grid(
        resolution=4,
        feature_dim=2,
        hidden_dim=4,
        num_layers=2,
        out_dim=1,
    )
    optimizer = make_paper_optimizer(
        model,
        PaperTrainConfig(
            task="image",
            loss="l2",
            steps=1,
            batch_size=2,
            model_lr=0.001,
            encoder_lr=0.1,
        ),
    )
    assert {
        group["group_name"]: group["lr"] for group in optimizer.param_groups
    } == {"encoder": 0.1, "model": 0.001}


def test_minibatch_stream_and_training_resume_are_deterministic():
    first = MinibatchStream(20, 5, seed=7)
    first.next()
    state = first.state_dict()
    expected_next = first.next()
    resumed = MinibatchStream(20, 5, seed=7)
    resumed.load_state_dict(state)
    assert torch.equal(resumed.next(), expected_next)

    coords = torch.linspace(0, 1, 16).unsqueeze(1)
    targets = coords.square()
    config = PaperTrainConfig(
        task="image",
        loss="l2",
        steps=4,
        batch_size=5,
        model_lr=0.01,
        seed=3,
        checkpoint_every=2,
        device=torch.device("cpu"),
    )
    torch.manual_seed(11)
    initial = torch.nn.Linear(1, 1)
    uninterrupted = torch.nn.Linear(1, 1)
    uninterrupted.load_state_dict(initial.state_dict())
    resumed_model = torch.nn.Linear(1, 1)
    resumed_model.load_state_dict(initial.state_dict())

    saved = {}

    def capture(step, checkpoint):
        if step == 2:
            buffer = io.BytesIO()
            torch.save(dict(checkpoint), buffer)
            buffer.seek(0)
            saved["state"] = torch.load(buffer, weights_only=False)

    fit_paper(
        uninterrupted,
        coords,
        targets,
        config,
        on_checkpoint=capture,
    )
    fit_paper(
        resumed_model,
        coords,
        targets,
        config,
        resume_state=saved["state"],
    )
    for expected, actual in zip(
        uninterrupted.parameters(), resumed_model.parameters()
    ):
        assert torch.equal(expected, actual)


def test_spectral_metrics_identity_and_phase_invariance():
    image = torch.arange(16 * 16, dtype=torch.float32).reshape(16, 16)
    shifted = torch.roll(image, shifts=(3, 5), dims=(0, 1))
    assert lsd(image, image) == 0.0
    assert lpsd(image, image) == 0.0
    assert lsd(image, shifted) > 0
    assert lpsd(image, shifted) == pytest.approx(0.0, abs=1e-12)


def test_optional_paper_metrics_identity_and_version_record():
    pytest.importorskip("torchmetrics")
    pytest.importorskip("flip_evaluator")
    image = torch.rand(16, 16, 3)
    assert ssim(image, image) == pytest.approx(1.0)
    assert flip(image, image) == pytest.approx(0.0)
    versions = metric_versions()
    assert versions["torchmetrics"] is not None
    assert versions["flip-evaluator"] is not None


def test_all_paper_configs_are_valid_and_frozen():
    paths = sorted((ROOT / "configs" / "paper").glob("*.toml"))
    assert {path.name for path in paths} == {
        "image_full.toml",
        "image_smoke.toml",
        "sdf_full.toml",
        "sdf_smoke.toml",
        "texture_full.toml",
        "texture_smoke.toml",
    }
    configs = [load_experiment_config(path) for path in paths]
    assert all(config.paper.endswith("2604.24167v1") for config in configs)
    assert all(
        any(
            len(method.seeds or config.seeds) >= 3
            for method in config.methods
        )
        for config in configs
        if config.profile == "full"
    )
    with pytest.raises(TypeError):
        configs[0].training["loss"] = "l2"


def _synthetic_image_instance() -> TensorInstance:
    y, x = torch.meshgrid(
        torch.linspace(0, 1, 8),
        torch.linspace(0, 1, 8),
        indexing="ij",
    )
    coords = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
    targets = torch.stack(
        [coords[:, 0], coords[:, 1], (coords[:, 0] + coords[:, 1]) / 2],
        dim=1,
    )
    return TensorInstance(
        "synthetic",
        coords,
        targets,
        shape=(8, 8, 3),
    )


def test_runner_sharding_atomic_results_and_resume(tmp_path):
    config = load_experiment_config(ROOT / "configs/paper/image_smoke.toml")
    instance = _synthetic_image_instance()
    jobs = enumerate_jobs(config, [instance])
    assert len(jobs) == 3
    assert {
        job.index for job in shard_jobs(jobs, rank=0, world_size=2)
    }.isdisjoint(
        {job.index for job in shard_jobs(jobs, rank=1, world_size=2)}
    )

    runner = ExperimentRunner(
        config,
        tmp_path,
        device=torch.device("cpu"),
    )
    records = runner.run([instance])
    assert len(records) == 3
    assert len(list((tmp_path / "raw").glob("**/*.json"))) == 3
    assert len(runner.run([instance])) == 3

    paired = [
        {
            "instance": "a",
            "seed": seed,
            "method": method,
            "metrics": {"psnr": value},
        }
        for seed, baseline, candidate in ((0, 1.0, 2.0), (1, 2.0, 4.0))
        for method, value in (("baseline", baseline), ("candidate", candidate))
    ]
    delta = paired_delta(
        paired,
        baseline="baseline",
        candidate="candidate",
        metric="psnr",
    )
    assert delta["count"] == 2
    assert delta["mean"] == pytest.approx(1.5)
