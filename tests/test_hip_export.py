"""CPU-side regression tests for HIP fixture and weight export."""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hip.export_fixture import (
    METHOD_SPECS,
    WORKLOAD_MAGIC,
    WORKLOAD_SCHEMA,
    aggregate_dim,
    export_pytorch_fixture,
    make_random_fixture,
    pink_channel_indices,
    write_fixture,
    write_weight_archive,
)


def test_paper_method_dimensions_and_selective_indices():
    assert aggregate_dim("bi-grid", 16) == 16
    assert aggregate_dim("grid-peps-3f", 16) == 112
    assert aggregate_dim("grid-pink-peps-3f", 16) == 44
    assert aggregate_dim("grid-pink-peps-4f", 16) == 46
    assert pink_channel_indices(8, 3) == (
        (0, 1, 2, 3, 4, 5, 6, 7),
        (4, 5, 6, 7),
        (2, 3),
        (1,),
        (0, 1, 2, 3),
        (4, 5),
        (6,),
    )


def test_fixture_and_weight_archive_are_self_describing(tmp_path):
    fixture = make_random_fixture(
        "grid-pink-peps-4f",
        channels=7,
        grid_height=5,
        grid_width=9,
        points=17,
        hidden=19,
        output=3,
        seed=7,
    )
    fixture_path = tmp_path / "pink4.bin"
    manifest_path = tmp_path / "pink4.json"
    weights_path = tmp_path / "pink4.npz"
    metadata = write_fixture(
        fixture_path, fixture, manifest_path=manifest_path
    )
    archive = write_weight_archive(weights_path, fixture)

    magic, schema = struct.unpack("<2i", fixture_path.read_bytes()[:8])
    assert (magic, schema) == (WORKLOAD_MAGIC, WORKLOAD_SCHEMA)
    assert metadata["method"] == "grid-pink-peps-4f"
    assert metadata["selective_channel_sampling"] is True
    assert len(metadata["sha256"]) == len(archive["sha256"]) == 64
    assert json.loads(manifest_path.read_text())["weight_layout"] == "input_major"
    with np.load(weights_path) as values:
        assert values["weight_layout"].item() == "input_major"
        assert values["weight_1"].shape == fixture.weights[0].shape
        assert np.array_equal(values["grid"], fixture.grid)


def test_export_actual_pytorch_peps_transposes_all_weights(tmp_path):
    from peps import GridEncoder, MLP, PEPS, PinkAggregator, Projector

    torch.manual_seed(11)
    projector = Projector(4)
    encoder = GridEncoder(2, (5, 9), 7)
    aggregator = PinkAggregator(projector.num_points, 7)
    decoder = MLP(
        aggregator.out_dim,
        3,
        hidden_dim=17,
        num_layers=4,
        activation="gelu",
    )
    model = PEPS(
        projector,
        encoder,
        aggregator,
        decoder,
        selective_sampling=True,
    )
    coords = torch.rand(13, 4)[:, ::2]  # non-contiguous source view
    fixture = export_pytorch_fixture(
        tmp_path / "model.bin",
        model=model,
        coords=coords,
        method="grid-pink-peps-4f",
        weights_path=tmp_path / "weights.npz",
        manifest_path=tmp_path / "manifest.json",
    )

    linear = [
        module for module in decoder.modules() if isinstance(module, torch.nn.Linear)
    ]
    assert len(linear) == 4
    for exported, layer in zip(fixture.weights, linear):
        expected = layer.weight.detach().t().contiguous().numpy()
        assert np.array_equal(exported, expected)
    expected_output = model(coords).detach().numpy()
    assert np.allclose(fixture.reference("fp32"), expected_output, atol=2e-6)
    assert set(METHOD_SPECS) == {
        "bi-grid",
        "grid-peps-3f",
        "grid-pink-peps-3f",
        "grid-pink-peps-4f",
    }
