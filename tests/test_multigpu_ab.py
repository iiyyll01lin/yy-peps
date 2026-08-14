"""The multi-GPU A/B receipt must keep its two arms apart.

Both arms were run on the same four cards, on the same day, by the same
harness, with one variable toggled. One of them settles its question and the
other cannot, and the difference is not the effect size but the noise floor:
the collective benchmark repeats to within seven per cent and shows a factor
of 2.5, while the training benchmark varies by up to 2.7x between rounds of
the same arm and shows a difference of two per cent between arms.

The receipt is therefore only useful if it keeps refusing the training claim.
These tests pin that refusal, pin the hashes of the captures it aggregates,
and pin the lower bound as the supported number rather than the median.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = json.loads(
    (ROOT / "results" / "multigpu_ab_receipt.json").read_text(encoding="utf-8")
)


def test_every_aggregated_capture_matches_its_recorded_hash():
    assert RECEIPT["raw_capture_sha256"], "the receipt must name what it aggregates"
    for rel, expected in RECEIPT["raw_capture_sha256"].items():
        blob = (ROOT / rel).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == expected, rel


def test_the_training_arm_refuses_to_claim_a_peer_to_peer_effect():
    finding = RECEIPT["findings"]["training_p2p_effect"]
    assert finding["conclusive"] is False
    # The refusal has to stay quantitative: the within-arm spread must still be
    # the larger number, or the refusal has become a matter of opinion.
    between = abs(finding["median_ratio"] - 1.0)
    within = max(
        finding["within_arm_spread_p2p_enabled"] - 1.0,
        finding["within_arm_spread_p2p_disabled"] - 1.0,
    )
    assert within > between * 10


def test_the_collective_arm_is_conclusive_because_the_noise_is_small():
    finding = RECEIPT["findings"]["collective_p2p_effect"]
    assert finding["conclusive"] is True
    low, high = finding["median_ratio_range"]
    assert 2.5 < low <= high < 2.7
    assert finding["worst_within_arm_spread"] < 1.07
    assert (low - 1.0) > (finding["worst_within_arm_spread"] - 1.0) * 20


def test_the_multi_gpu_speedup_is_reported_as_a_bound_not_a_median():
    finding = RECEIPT["findings"]["four_gpu_beats_one_gpu"]
    assert finding["conclusive"] is True
    # The ranges do not overlap, which is what carries the claim; the median is
    # recorded but is not the supported number.
    assert finding["guaranteed_lower_bound"] > 1.0
    assert finding["guaranteed_lower_bound"] < finding["median_ratio"]


def test_the_single_round_trap_stays_recorded():
    trap = RECEIPT["what_a_single_round_would_have_claimed"]
    assert "3.685" in trap["detail"] and "3.156" in trap["detail"]


def test_the_unanalysed_captures_are_declared_rather_than_implied():
    listed = RECEIPT["not_yet_analysed"]
    assert listed, "captures that carry no claim must still be declared"
    for rel in listed:
        assert (ROOT / rel).exists(), rel
        assert rel not in RECEIPT["raw_capture_sha256"]


def test_the_logs_are_not_dragged_into_the_repository():
    assert not list((ROOT / "results" / "multigpu").glob("*.log.tracked"))
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "results/multigpu/*" in ignore
    assert "!results/multigpu/*.json" in ignore
