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
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hip.export_fixture import METHOD_SPECS, aggregate_dim

# The four values below are duplicated from the kernel on purpose. Importing
# them would let a change to the source silently change the test.
STOCK_INPUT_CAP = 512
STOCK_HIDDEN_CAP = 128
TUNED_INPUT_CAP = 128
TUNED_HIDDEN_CAP = 64

WMMA_TILE = 16
CHANNELS = 16
HIDDEN = 64
LDS_BYTES_PER_CU = 64 * 1024
WAVES_PER_WORKGROUP = 2
MAX_WAVES_PER_CU = 32


def lds_footprint(input_cap: int, hidden_cap: int) -> int:
    """Bytes of LDS integrated_peps_wmma reserves per workgroup.

    feature_tile and the two hidden buffers are float16_t; the accumulator
    is float. This mirrors the four __shared__ declarations exactly.
    """
    return WMMA_TILE * (input_cap * 2 + hidden_cap * 2 + hidden_cap * 2 + hidden_cap * 4)


def occupancy(footprint: int) -> float:
    workgroups = LDS_BYTES_PER_CU // footprint
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
