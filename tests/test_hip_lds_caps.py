"""Guards for the LDS tile caps recorded in results/hip_lds_ab.json.

The fused kernel sizes its four __shared__ tiles from compile-time caps, so
those caps set the per-workgroup LDS footprint and therefore occupancy,
independently of the dimensions any run actually uses. Narrowing them from
512/128 to 128/64 cut measured latency by 1.70x to 2.13x on two parts.

That speedup is only safe while the narrowed caps still cover every method
the benchmark can select. check_config in the kernel rejects an oversized
input at runtime, so a violation fails closed rather than corrupting output
-- but it fails on the GPU, in a build that has to be asked for explicitly,
which is a slow and easily-missed way to find out. These tests catch it in
CI instead: adding a fifth method with a wider aggregate breaks them here.

The second half covers the texture reproduction's method set, which is
wider than the benchmark's and was not covered when the caps were chosen.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hip.export_fixture import (  # noqa: E402
    MODE_CONCAT,
    MODE_PINK,
    METHOD_SPECS,
    MethodSpec,
    aggregate_dim,
)

# The four values below are duplicated from the kernel on purpose. Importing
# them would let a change to the source silently change the test.
STOCK_INPUT_CAP = 512
STOCK_HIDDEN_CAP = 128
TUNED_INPUT_CAP = 128
TUNED_HIDDEN_CAP = 64
TEXTURE_INPUT_CAP = 160
MAX_CHANNELS = 32

WMMA_TILE = 16
CHANNELS = 16
TEXTURE_CHANNELS = 17
HIDDEN = 64
# The allocation pool behaves as 128 KB shared by the two compute units of a
# WGP, and the footprint is rounded up to a 1024-byte granule before it is
# divided. Both parts were established by counter, not assumed; see
# test_only_the_granular_model_matches_every_measurement.
LDS_POOL_PER_WGP = 128 * 1024
LDS_ADVERTISED_PER_CU = 64 * 1024
LDS_GRANULE = 1024
WAVES_PER_WORKGROUP = 2
CUS_PER_WGP = 2
MAX_WAVES_PER_CU = 32

SWEEP = ROOT / "results" / "texture_repro" / "frequency_sweep.json"


def lds_footprint(input_cap: int, hidden_cap: int) -> int:
    """Bytes of LDS integrated_peps_wmma reserves per workgroup.

    feature_tile and the two hidden buffers are float16_t; the accumulator
    is float. This mirrors the four __shared__ declarations exactly.
    """
    return WMMA_TILE * (input_cap * 2 + hidden_cap * 2 + hidden_cap * 2 + hidden_cap * 4)


def effective_footprint(footprint: int) -> int:
    return -(-footprint // LDS_GRANULE) * LDS_GRANULE


def occupancy(footprint: int) -> float:
    workgroups_per_wgp = LDS_POOL_PER_WGP // effective_footprint(footprint)
    waves_per_cu = workgroups_per_wgp * WAVES_PER_WORKGROUP / CUS_PER_WGP
    return waves_per_cu / MAX_WAVES_PER_CU


def occupancy_plain_wgp(footprint: int) -> float:
    """Superseded: a per-WGP pool with no allocation granule."""
    workgroups_per_wgp = LDS_POOL_PER_WGP // footprint
    return workgroups_per_wgp * WAVES_PER_WORKGROUP / CUS_PER_WGP / MAX_WAVES_PER_CU


def occupancy_per_cu_pool(footprint: int) -> float:
    """Superseded: a 64 KB pool per compute unit."""
    workgroups = LDS_ADVERTISED_PER_CU // footprint
    return workgroups * WAVES_PER_WORKGROUP / MAX_WAVES_PER_CU


def test_footprints_match_the_code_object_metadata():
    # Read back from amdclang++ -S as .group_segment_fixed_size.
    assert lds_footprint(STOCK_INPUT_CAP, STOCK_HIDDEN_CAP) == 32768
    assert lds_footprint(TUNED_INPUT_CAP, TUNED_HIDDEN_CAP) == 12288


def test_narrowed_caps_cover_every_benchmarked_method():
    widest = max(aggregate_dim(name, CHANNELS) for name in METHOD_SPECS)
    assert widest == 112, f"method set changed: widest aggregate is now {widest}"
    assert widest <= TUNED_INPUT_CAP
    assert HIDDEN <= TUNED_HIDDEN_CAP


def test_narrowed_caps_are_not_slack_enough_to_be_pointless():
    # If the narrowed cap were close to the stock one the change would not
    # have moved occupancy, and the recorded speedup would need another
    # explanation.
    stock = lds_footprint(STOCK_INPUT_CAP, STOCK_HIDDEN_CAP)
    tuned = lds_footprint(TUNED_INPUT_CAP, TUNED_HIDDEN_CAP)
    assert stock / tuned > 2.0


def test_occupancy_gain_is_the_one_recorded_in_the_receipt():
    stock = occupancy(lds_footprint(STOCK_INPUT_CAP, STOCK_HIDDEN_CAP))
    tuned = occupancy(lds_footprint(TUNED_INPUT_CAP, TUNED_HIDDEN_CAP))
    assert stock == 0.125
    assert tuned == 0.3125
    # The measured speedup was 1.70x-2.13x against a predicted 2.5x, so the
    # receipt must not claim the prediction as the result.
    assert tuned / stock == 2.5


def test_mutation_a_wider_method_would_be_caught():
    # A hypothetical five-frequency concat method needs 176 channels, which
    # the narrowed build would refuse. The guard above must reject it.
    hypothetical = (2 * 5 + 1) * CHANNELS
    assert hypothetical == 176
    assert hypothetical > TUNED_INPUT_CAP
    assert hypothetical <= STOCK_INPUT_CAP


# --- the texture reproduction's own method set -------------------------


def sweep_input_dims() -> dict[str, int]:
    rows = json.loads(SWEEP.read_text())["rows"]
    return {row["method"]: row["decoder_input_dim"] for row in rows}


def texture_geometry(mode: int, frequencies: int) -> int:
    spec = MethodSpec("probe", mode, frequencies, 0.0)
    return aggregate_dim(spec, TEXTURE_CHANNELS)


def test_kernel_aggregation_reproduces_the_texture_grid_family():
    # If these drift apart, the fused kernel is no longer timing the
    # geometry the headline reproduction trains.
    dims = sweep_input_dims()
    assert texture_geometry(MODE_CONCAT, 3) == dims["Grid-PEPS3F"] == 119
    assert texture_geometry(MODE_CONCAT, 4) == dims["Grid-PEPS4F"] == 153
    assert texture_geometry(MODE_PINK, 3) == dims["Grid-PinkPEPS3F"] == 45
    assert texture_geometry(MODE_PINK, 4) == dims["Grid-PinkPEPS4F"] == 47


def test_texture_grid_family_needs_a_wider_cap_than_the_paper_methods():
    # This is why results/hip_texture_geometry.json uses 160 and not the
    # 128 that serves the paper's 16-channel methods.
    widest = max(
        texture_geometry(mode, frequencies)
        for mode, frequencies in (
            (MODE_CONCAT, 3), (MODE_CONCAT, 4), (MODE_PINK, 3), (MODE_PINK, 4)
        )
    )
    assert widest == 153
    assert widest > TUNED_INPUT_CAP
    assert widest <= TEXTURE_INPUT_CAP
    assert TEXTURE_INPUT_CAP % WMMA_TILE == 0


def test_texture_cap_still_beats_the_stock_footprint():
    stock = lds_footprint(STOCK_INPUT_CAP, STOCK_HIDDEN_CAP)
    texture = lds_footprint(TEXTURE_INPUT_CAP, TUNED_HIDDEN_CAP)
    assert texture == 13312
    assert occupancy(stock) == 0.125
    assert occupancy(texture) == 0.28125


def test_ntc_family_is_out_of_reach_of_the_fused_kernel():
    # NTC aggregates 4*12 + 20 = 68 channels against the kernel's cap of 32,
    # and adds a 12-dimension tiled encoding aggregate_dim does not model.
    # Raising the input cap alone would not help.
    dims = sweep_input_dims()
    ntc_channels = 4 * 12 + 20
    assert ntc_channels == 68
    assert ntc_channels > MAX_CHANNELS
    assert dims["NTC_PEPS4F"] == 624
    assert dims["NTC_PEPS4F"] > STOCK_INPUT_CAP
    for name in ("NTC_PEPS3F", "NTC_PEPS4F", "NTC_PinkPEPS3F", "NTC_PinkPEPS4F"):
        assert dims[name] > TEXTURE_INPUT_CAP


def test_widening_for_ntc_would_give_back_the_occupancy_gain():
    # Sizing the tiles for NTC_PEPS4F returns occupancy to the stock 12.5%.
    # The footprint is smaller than stock, because stock also carries a
    # 128-wide hidden cap -- so the footprint alone does not tell the story.
    for_ntc = lds_footprint(624, TUNED_HIDDEN_CAP)
    assert for_ntc == 28160
    assert for_ntc < lds_footprint(STOCK_INPUT_CAP, STOCK_HIDDEN_CAP)
    assert occupancy(for_ntc) == 0.125
    texture = occupancy(lds_footprint(TEXTURE_INPUT_CAP, TUNED_HIDDEN_CAP))
    assert occupancy(for_ntc) < texture


# --- the model the counters corrected ----------------------------------


def test_the_two_pool_models_disagree_at_13312():
    # This is the footprint that caught the error. Nine workgroups per WGP
    # is an odd number, and a per-CU model can only ever produce an even
    # number of workgroups per WGP, so it cannot express this case.
    texture = lds_footprint(TEXTURE_INPUT_CAP, TUNED_HIDDEN_CAP)
    assert LDS_POOL_PER_WGP // texture == 9
    assert occupancy(texture) == 0.28125
    assert occupancy_per_cu_pool(texture) == 0.25
    # rocprofv3 OccupancyPercent measured 28.04% mean over 15 dispatches,
    # and MeanOccupancyPerCU measured 8.97 of the derived 9 waves.
    assert abs(occupancy(texture) * 100 - 28.04) < 0.5
    assert abs(occupancy_per_cu_pool(texture) * 100 - 28.04) > 0.5


def test_the_earlier_footprints_could_not_have_caught_it():
    # Both models agree on 32768 and 12288, which is why the wrong one
    # survived two rounds of measurement before a third footprint exposed it.
    for footprint in (32768, 12288):
        assert occupancy(footprint) == occupancy_per_cu_pool(footprint)


# --- what seven measured footprints say about the model ----------------

# gfx1151, rocprofv3 MeanOccupancyPerCU, 15 dispatches each.
# results/hip_profile/gfx1151_occupancy.csv
MEASURED_WAVES = {
    32768: 4.00,
    13312: 8.97,
    12288: 9.98,
    11776: 9.97,
    10752: 10.97,
    9728: 11.96,
    8704: 13.94,
}


def waves(footprint: int) -> float:
    return occupancy(footprint) * MAX_WAVES_PER_CU


def test_only_the_granular_model_matches_every_measurement():
    def hits(model) -> int:
        return sum(
            abs(measured - model(f) * MAX_WAVES_PER_CU) < 0.5
            for f, measured in MEASURED_WAVES.items()
        )

    assert hits(occupancy) == len(MEASURED_WAVES)
    # Kept as an assertion rather than a comment so that a future footprint
    # which rehabilitates a superseded model cannot pass unnoticed.
    assert hits(occupancy_per_cu_pool) == 5
    assert hits(occupancy_plain_wgp) == 3


def test_granularity_is_1024_and_not_512_or_2048():
    # 8704 is a multiple of 512, so a 512-byte granule would leave it alone
    # and predict 15 waves against the measured 13.94.
    assert 8704 % 512 == 0
    assert LDS_POOL_PER_WGP // 8704 == 15
    assert waves(8704) == 14
    # A 2048-byte granule would round 8704 up to 10240 and predict 12.
    assert -(-8704 // 2048) * 2048 == 10240
    assert LDS_POOL_PER_WGP // 10240 == 12


def test_narrowing_below_128_cannot_help_grid_peps_3f():
    # The one method that gained nothing. 112 is the narrowest cap it can
    # take, and its footprint rounds up to the same granule as 128, so the
    # occupancy is identical and the measured speedup was 1.000x.
    peps = lds_footprint(112, TUNED_HIDDEN_CAP)
    shared = lds_footprint(TUNED_INPUT_CAP, TUNED_HIDDEN_CAP)
    assert peps == 11776 and shared == 12288
    assert effective_footprint(peps) == effective_footprint(shared) == 12288
    assert occupancy(peps) == occupancy(shared)
    # Reaching the next step would need a footprint at or below 11264,
    # i.e. a cap of 96, which is narrower than the 112 the method needs.
    assert lds_footprint(96, TUNED_HIDDEN_CAP) == 11264
    assert aggregate_dim("grid-peps-3f", CHANNELS) == 112


def test_specialised_caps_reach_the_occupancy_they_were_chosen_for():
    assert waves(lds_footprint(16, TUNED_HIDDEN_CAP)) == 14   # bi-grid
    assert waves(lds_footprint(48, TUNED_HIDDEN_CAP)) == 12   # pink 3f and 4f
    assert waves(lds_footprint(TUNED_INPUT_CAP, TUNED_HIDDEN_CAP)) == 10
    for cap, method in ((16, "bi-grid"), (48, "grid-pink-peps-3f"),
                        (48, "grid-pink-peps-4f")):
        assert aggregate_dim(method, CHANNELS) <= cap


def test_cap_narrowing_has_a_provable_ceiling():
    # Only the feature tile scales with the cap. The other three are fixed by
    # the hidden width, and they alone cost what the next occupancy step needs.
    fixed = WMMA_TILE * TUNED_HIDDEN_CAP * (2 + 2 + 4)
    assert fixed == 8192
    assert lds_footprint(0, TUNED_HIDDEN_CAP) == fixed
    assert LDS_POOL_PER_WGP // fixed == 16

    # So sixteen workgroups per WGP is out of reach before the feature tile is
    # counted at all, and fifteen rounds to the same impossible target.
    assert LDS_POOL_PER_WGP / 15 < fixed + LDS_GRANULE
    assert all(waves(lds_footprint(cap, TUNED_HIDDEN_CAP)) <= 14
               for cap in range(1, 513))
    assert waves(lds_footprint(16, TUNED_HIDDEN_CAP)) == 14


def test_the_receipt_records_the_same_ceiling():
    # The proof lives in the test; the receipt states it. They must agree, or
    # a reader takes the ceiling on trust from a file nothing checks.
    receipt = json.loads(
        (ROOT / "results" / "hip_specialised_caps.json").read_text(encoding="utf-8")
    )["ceiling_of_this_technique"]
    assert receipt["fixed_tiles_bytes"] == WMMA_TILE * TUNED_HIDDEN_CAP * 8
    assert receipt["ceiling"]["waves_per_cu"] == 14
    assert receipt["ceiling"]["occupancy_fraction"] == 0.4375
    assert occupancy(lds_footprint(16, TUNED_HIDDEN_CAP)) == 0.4375
    # bi-grid is already there, so this is a reached ceiling, not a bound.
    assert abs(receipt["ceiling"]["measured_waves_per_cu"] - 14) < 0.1
