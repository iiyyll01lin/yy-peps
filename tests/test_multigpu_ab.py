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

The refusal was not the end of it. A later harness put the same workload on the
same cards behind an idle precondition and answered the scaling question with
non-overlapping ranges, so the tests also pin why one run could be read and the
other could not.
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


def test_every_capture_is_either_analysed_or_declared():
    hashed = set(RECEIPT["raw_capture_sha256"])
    listed = RECEIPT["not_yet_analysed"]
    for rel in listed:
        assert (ROOT / rel).exists(), rel
        assert rel not in hashed
    present = {
        f"results/multigpu/{path.name}"
        for path in (ROOT / "results" / "multigpu").glob("*.json")
    }
    # A capture that is neither hashed nor declared has gone quiet. Requiring the
    # two sets to partition the directory is what stops one being added and
    # carrying no claim and no admission that it carries none.
    assert present == hashed | set(listed)


def test_the_strong_scaling_speedup_survives_as_a_floor():
    scaling = RECEIPT["analysed_supporting_captures"]["strong_scaling_under_an_idle_gate"]
    assert scaling["ranges_disjoint"]["1_vs_2"] is True
    assert scaling["ranges_disjoint"]["2_vs_4"] is True
    for count in ("2", "4"):
        floor = scaling["guaranteed_speedup_worst_case_over_best_single_gpu"][count]
        assert 1.0 < floor < scaling["speedup_vs_1gpu_median"][count]
    # The worst four-card round must still beat the best single-card round, or
    # the disjointness above is being asserted rather than measured.
    assert scaling["steps_per_second"]["4"]["min"] > scaling["steps_per_second"]["1"]["max"]


def test_the_readable_and_refused_runs_are_separated_by_the_idle_gate():
    scaling = RECEIPT["analysed_supporting_captures"]["strong_scaling_under_an_idle_gate"]
    refused = RECEIPT["analysed_supporting_captures"]["why_the_refused_arm_could_not_be_read"]
    worst_readable = max(
        scaling["steps_per_second"][count]["within_arm_spread"] for count in ("1", "2", "4")
    )
    worst_refused = max(
        RECEIPT["training"][arm]["within_arm_spread"] for arm in RECEIPT["training"]
    )
    assert worst_readable < 1.02 < worst_refused
    assert "not identified" in refused["interference_source"]
    # The texture job was the obvious suspect and was cleared. Losing that keeps
    # the reader free to assume it was never checked.
    assert "2026-07-22T03:41:28Z" in refused["refuted_hypothesis"]


def test_the_parallel_split_does_not_move_the_result():
    fidelity = RECEIPT["analysed_supporting_captures"]["numerical_fidelity_across_gpu_counts"]
    ratio = fidelity["max_relative_loss_delta_vs_1gpu"] / fidelity["float32_epsilon"]
    assert abs(ratio - fidelity["ratio_to_float32_epsilon"]) < 1e-9
    assert ratio < 2.0
    psnr = fidelity["reconstruction_psnr_db_median"]
    assert max(psnr.values()) - min(psnr.values()) < 1e-5


def test_the_logs_are_not_dragged_into_the_repository():
    assert not list((ROOT / "results" / "multigpu").glob("*.log.tracked"))
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "results/multigpu/*" in ignore
    assert "!results/multigpu/*.json" in ignore


def test_the_peer_to_peer_path_is_recorded_as_not_working_by_default():
    # The 2.5x result is only reachable after the environment is set. If this
    # ever reads as though peer-to-peer worked out of the box, the headline
    # becomes unreproducible for anyone following the chapter.
    found = RECEIPT["analysed_supporting_captures"][
        "peer_to_peer_does_not_work_by_default"
    ]
    assert found["default_ipc_result"] == "failed"
    assert "hipIpcGetMemHandle" in found["default_ipc_error"]
    assert found["all_six_pairs_failed"] is True
    assert found["required_environment"]["HSA_ENABLE_IPC_MODE_LEGACY"] == "0"
    assert found["required_environment"]["HSA_FORCE_FINE_GRAIN_PCIE"] == "1"


def test_the_silent_fallback_keeps_its_transport_field():
    # A capture that asked for peer-to-peer, got host staging, and recorded a
    # number anyway. The transport field is the only thing separating it from a
    # false headline, so it has to stay.
    found = RECEIPT["analysed_supporting_captures"][
        "a_benchmark_that_silently_measured_the_other_arm"
    ]
    assert found["effective"] == "p2p_disabled_host_transport"
    assert found["fallback_reason"]
    assert found["effective"] != found["requested"]


def test_the_fabric_is_reported_as_uniform():
    found = RECEIPT["analysed_supporting_captures"]["topology"]
    assert found["ordered_pairs"] == 12
    assert found["spread"] < 1.05
    assert found["numa_node"] == -1


def test_the_iommu_comparison_stays_refused_for_a_stated_reason():
    found = RECEIPT["analysed_supporting_captures"]["the_iommu_comparison_is_refused"]
    assert found["conclusive"] is False
    # Two variables moved together; the refusal is only checkable while the
    # protocols are recorded as actually differing.
    assert found["early_timed_iterations"] != found["stable_timed_iterations"]
    assert found["early_warmup_iterations"] != found["stable_warmup_iterations"]
