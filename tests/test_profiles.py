"""Schema and immutability tests for experiment profiles."""

import pytest

from peps.profiles import (
    COURSE_FAST,
    PAPER_EXACT,
    PROFILE_NAMES,
    PROFILE_SCHEMA,
    PROFILE_SCHEMA_VERSION,
    PROFILES,
    get_profile,
)


def test_profile_registry_has_only_stable_public_names():
    assert PAPER_EXACT.schema == PROFILE_SCHEMA == "peps.experiment_profile"
    assert PAPER_EXACT.schema_version == PROFILE_SCHEMA_VERSION == 1
    assert PROFILE_NAMES == ("paper_exact", "course_fast")
    assert tuple(PROFILES) == PROFILE_NAMES
    assert get_profile("paper_exact") is PAPER_EXACT
    assert get_profile("course_fast") is COURSE_FAST
    with pytest.raises(KeyError, match="unknown profile"):
        get_profile("full")


def test_paper_exact_image_protocol_is_inspectable():
    sweep = PAPER_EXACT.image["capacity_sweep"]
    assert sweep["dataset"]["resolution"] == "native_4k"
    assert sweep["grid_resolutions"] == (16, 32, 64, 128)
    assert sweep["feature_dimensions"] == (8, 16, 32, 64)
    assert sweep["peps_frequencies"] == 3
    assert sweep["loss"] == "l1"

    kodak = PAPER_EXACT.image["kodak_table_1"]
    assert len(kodak["dataset"]["instance_ids"]) == 24
    assert kodak["dataset"]["resolution_xy"] == (768, 512)
    assert kodak["models"]["grid"]["grid_resolution_xy"] == (196, 128)
    assert kodak["models"]["g_peps"]["frequencies"] == 3
    assert kodak["models"]["pe"]["frequencies"] == 10
    assert kodak["network"]["hidden_layers"] == 3
    assert "Table 1" in kodak["paper_ambiguity"]


def test_paper_exact_texture_sdf_and_runtime_protocols():
    texture = PAPER_EXACT.texture
    assert texture["dataset"]["instance_count"] == 18
    assert texture["dataset"]["resolution_xy"] == (4096, 4096)
    assert texture["models"]["grid_peps"]["frequencies"] == 4
    assert texture["training"]["batch_size"] == 60_000
    assert texture["training"]["optimizer_steps"] == 120_000
    assert texture["training"]["scheduler"] == "cosine"

    sdf = PAPER_EXACT.sdf
    assert sdf["dataset"]["volume_resolution"] == (512, 512, 512)
    assert sdf["models"]["grid_peps"]["frequencies"] == 3
    assert sdf["models"]["multi_hash"]["resolutions"] == (16, 32, 64, 128)
    assert sdf["training"]["primary"]["loss"] == "mape"
    assert sdf["training"]["appendix"]["loss"] == "l1"
    assert sdf["eight_x_ablation"]["encoder_parameter_multiplier"] == 8

    assert PAPER_EXACT.quantization["enabled"] is False
    assert PAPER_EXACT.runtime["target"]["architecture"] == "gfx1201"
    reported = PAPER_EXACT.runtime["benchmark"]["reported_ms"]
    assert reported["bi_grid_0_frequencies"] == 4.32
    assert reported["grid_pink_peps_3_frequencies"] == 4.86


def test_course_fast_profile_keeps_teaching_budgets_explicit():
    assert COURSE_FAST.image["dataset"]["instance_ids"] == (
        "kodim01",
        "kodim05",
        "kodim19",
    )
    assert COURSE_FAST.image["training"]["optimizer_steps"] == 2_000
    assert COURSE_FAST.texture["dataset"]["output_resolution"] == (512, 512)
    assert COURSE_FAST.sdf["dataset"]["name"] == "analytic_torus"
    assert COURSE_FAST.quantization["bit_widths"] == (8, 6, 4)
    assert "not_paper_comparable" in COURSE_FAST.runtime["status"]


def test_profiles_are_recursively_immutable_and_export_detached_json():
    with pytest.raises(TypeError):
        PAPER_EXACT.image["capacity_sweep"] = {}
    with pytest.raises(TypeError):
        PAPER_EXACT.image["capacity_sweep"]["loss"] = "mse"
    with pytest.raises(TypeError):
        PAPER_EXACT.sdf["dataset"]["volume_resolution"][0] = 64

    exported = PAPER_EXACT.to_dict()
    assert exported["schema"] == "peps.experiment_profile"
    assert exported["schema_version"] == 1
    exported["image"]["capacity_sweep"]["grid_resolutions"][0] = 8
    assert PAPER_EXACT.image["capacity_sweep"]["grid_resolutions"][0] == 16
