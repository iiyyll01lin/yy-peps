"""Focused tests for the manifest-driven paper data pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from data.download import (
    AccessRequiredError,
    _fetch_stonefish,
    _validate_archive_member,
)
from data.manifest import (
    DataIntegrityError,
    MissingDataError,
    hash_file,
    load_manifest,
    load_texture_set,
)
from data.preprocess_sdf import (
    load_sdf_volume,
    normalize_mesh,
    sdf_asset_spec,
)


def test_checked_in_manifests_cover_paper_datasets() -> None:
    kodak = load_manifest("kodak")
    assert [image["id"] for image in kodak["images"]] == [
        f"kodim{i:02d}" for i in range(1, 25)
    ]
    assert sum(image["width"] == 512 for image in kodak["images"]) == 6
    assert all(image["checksum"]["algorithm"] == "sha256" for image in kodak["images"])
    assert all(image["credit"]["photographer"] for image in kodak["images"])

    textures = load_manifest("textures")
    expected_names = {
        "bench vice 01",
        "cardboard box 01",
        "cannon 01",
        "clay roof tiles 02",
        "fabric pattern 07",
        "garden gnome",
        "garden sprinkler 01",
        "wood planks",
        "treasure chest",
        "paving stones 070",
        "rails 001",
        "red dirt mud 01",
        "aerial rocks 02",
        "bricks 090",
        "forest sand 01",
        "metal plates 013",
        "roof 09",
        "wood 063",
    }
    assert len(textures["sets"]) == 18
    assert {item["paper_name"].lower() for item in textures["sets"]} == expected_names
    assert sum(len(item["maps"]) for item in textures["sets"]) == 78
    assert {item["source"]["provider"] for item in textures["sets"]} == {
        "polyhaven",
        "ambientcg",
    }
    assert all(
        (entry["width"], entry["height"], entry["channels"]) == (4096, 4096, 3)
        for item in textures["sets"]
        for entry in item["maps"]
    )

    sdf = load_manifest("sdf")
    assert [item["id"] for item in sdf["assets"]] == [
        "lucy",
        "pitted-stonefish",
        "thai-statue",
        "armadillo",
    ]
    assert sdf["preprocessing"]["resolution"] == 512
    assert sdf["preprocessing"]["sign_convention"] == "negative_inside"
    stonefish = sdf_asset_spec("pitted-stonefish", sdf)
    assert stonefish["source"]["uid"] == "0cdc3d1419384fd78fd952dc251a3169"
    assert stonefish["source"]["face_count"] == 10_548_062
    assert stonefish["substitutions"][0]["canonical"] is False


def test_source_verification_references_current_manifests() -> None:
    root = Path(__file__).resolve().parents[1]
    record = json.loads(
        (root / "data/provenance/source-verification.json").read_text(
            encoding="utf-8"
        )
    )
    for filename, expected in record["manifests"].items():
        assert hash_file(root / "data/manifests" / filename) == expected["sha256"]
    assert record["secrets_or_raw_data_committed"] is False


def test_dynamic_loader_is_strict_and_renormalizes_normals(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    texture_dir = raw / "textures" / "synthetic"
    texture_dir.mkdir(parents=True)

    diffuse = np.zeros((4, 4, 3), dtype=np.uint8)
    diffuse[..., 0] = 64
    diffuse[..., 1] = 128
    diffuse[..., 2] = 255
    normal = np.empty((4, 4, 3), dtype=np.uint8)
    normal[...] = (255, 191, 128)  # deliberately not unit length after decode
    diffuse_path = texture_dir / "diffuse.png"
    normal_path = texture_dir / "normal.png"
    Image.fromarray(diffuse).save(diffuse_path)
    Image.fromarray(normal).save(normal_path)

    manifest_path = tmp_path / "textures.json"
    manifest_path.write_text(
        json.dumps(
            _synthetic_texture_manifest(
                [
                    _map_spec(diffuse_path, raw, "diffuse", "DIFF"),
                    _map_spec(normal_path, raw, "normal", "normal"),
                ]
            )
        ),
        encoding="utf-8",
    )

    loaded = load_texture_set(
        "synthetic",
        raw_root=raw,
        output_size=2,
        manifest_path=manifest_path,
    )
    assert loaded.tensor.shape == (2, 2, 6)
    assert loaded.channel_count == 6
    normal_slice = loaded.maps[1].channel_slice
    vectors = loaded.tensor[..., normal_slice] * 2.0 - 1.0
    lengths = torch.linalg.vector_norm(vectors, dim=-1)
    torch.testing.assert_close(lengths, torch.ones_like(lengths), atol=2e-6, rtol=0)

    normal_path.unlink()
    with pytest.raises(MissingDataError):
        load_texture_set(
            "synthetic",
            raw_root=raw,
            manifest_path=manifest_path,
        )


def test_loader_preserves_16_bit_scalar_and_rejects_tampering(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    texture_dir = raw / "textures" / "synthetic"
    texture_dir.mkdir(parents=True)
    values = np.array(
        [[0, 65535], [16384, 32768]],
        dtype=np.uint16,
    )
    path = texture_dir / "displacement.png"
    Image.fromarray(values).save(path)
    manifest_path = tmp_path / "textures.json"
    manifest_path.write_text(
        json.dumps(
            _synthetic_texture_manifest(
                [
                    _map_spec(
                        path,
                        raw,
                        "displacement",
                        "Displacement",
                        encoded_bit_depth=16,
                    )
                ]
            )
        ),
        encoding="utf-8",
    )

    loaded = load_texture_set(
        "synthetic",
        raw_root=raw,
        manifest_path=manifest_path,
    )
    assert loaded.tensor.shape == (2, 2, 3)
    torch.testing.assert_close(
        loaded.tensor[..., 0],
        torch.tensor(values.astype(np.float32) / 65535.0),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(loaded.tensor[..., 0], loaded.tensor[..., 2])

    Image.fromarray(np.zeros_like(values)).save(path)
    with pytest.raises(DataIntegrityError):
        load_texture_set(
            "synthetic",
            raw_root=raw,
            manifest_path=manifest_path,
        )


def test_archive_member_traversal_is_rejected() -> None:
    _validate_archive_member("asset/maps/normal.png")
    with pytest.raises(DataIntegrityError):
        _validate_archive_member("../outside")
    with pytest.raises(DataIntegrityError):
        _validate_archive_member("/absolute/path")


def test_mesh_normalization_is_centered_and_isotropic() -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(2.0, 4.0, 8.0))
    mesh.apply_translation((10.0, -3.0, 5.0))
    normalized, transform = normalize_mesh(mesh)
    np.testing.assert_allclose(normalized.bounds.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(normalized.extents, (0.5, 1.0, 2.0), atol=1e-12)
    assert transform["isotropic_scale"] == pytest.approx(0.25)


def test_sdf_loader_requires_matching_provenance_checksum(tmp_path: Path) -> None:
    volume_path = tmp_path / "sdf.npy"
    provenance_path = tmp_path / "sdf.provenance.json"
    volume = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(4, 4, 4)
    np.save(volume_path, volume)
    provenance = {
        "schema_version": 1,
        "canonical_paper_protocol": False,
        "grid": {
            "shape": [4, 4, 4],
            "axis_order": "zyx",
            "sign_convention": "negative_inside",
            "dtype": "float32",
        },
        "output": {
            "bytes": volume_path.stat().st_size,
            "checksum": {
                "algorithm": "sha256",
                "value": hash_file(volume_path),
            },
        },
    }
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    loaded = load_sdf_volume(
        volume_path,
        provenance_path,
        require_paper_protocol=False,
    )
    np.testing.assert_array_equal(loaded, volume)
    with volume_path.open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 0x01]))
    with pytest.raises(DataIntegrityError):
        load_sdf_volume(
            volume_path,
            provenance_path,
            require_paper_protocol=False,
        )


def test_stonefish_requires_environment_token_or_manual_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SKETCHFAB_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("SKETCHFAB_API_TOKEN", raising=False)
    item = sdf_asset_spec("pitted-stonefish")
    with pytest.raises(AccessRequiredError, match="SKETCHFAB_OAUTH_TOKEN"):
        _fetch_stonefish(item, tmp_path)


def _synthetic_texture_manifest(maps: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "texture_set_dataset",
        "dataset_id": "synthetic-test",
        "sets": [
            {
                "id": "synthetic",
                "paper_name": "synthetic",
                "source": {"provider": "test"},
                "maps": maps,
            }
        ],
    }


def _map_spec(
    path: Path,
    raw_root: Path,
    map_id: str,
    semantic: str,
    *,
    encoded_bit_depth: int = 8,
) -> dict[str, object]:
    with Image.open(path) as image:
        width, height = image.size
    with path.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    result: dict[str, object] = {
        "id": map_id,
        "semantic": semantic,
        "local_path": path.relative_to(raw_root).as_posix(),
        "bytes": path.stat().st_size,
        "checksum": {"algorithm": "sha256", "value": checksum},
        "format": "PNG",
        "width": width,
        "height": height,
        "channels": 3,
        "color_space": "sRGB" if semantic == "DIFF" else "linear",
        "encoded_bit_depth": encoded_bit_depth,
    }
    if semantic == "normal":
        result["normal_convention"] = "OpenGL"
    return result
