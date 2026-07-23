"""CPU tests for the real-data multi-GPU benchmark helpers."""

from __future__ import annotations

import pytest

import experiments.real_workload as real_workload
from experiments.real_workload import (
    _active_dpm_mhz,
    _summarize_telemetry_samples,
    _wait_for_idle_gpus,
    summary_statistics,
    validate_direct_transport_log,
)


def test_summary_statistics_reports_robust_dispersion_and_missing_values():
    result = summary_statistics(
        [1.0, 2.0, 3.0, 4.0, 100.0, None],
        include_values=True,
    )
    assert result["count"] == 5
    assert result["missing_count"] == 1
    assert result["min"] == 1.0
    assert result["median"] == 3.0
    assert result["max"] == 100.0
    assert result["median_absolute_deviation"] == 1.0
    assert result["iqr"] == 2.0
    assert result["values"] == [1.0, 2.0, 3.0, 4.0, 100.0]


def test_active_dpm_clock_parser_uses_only_starred_level():
    assert _active_dpm_mhz("0: 96Mhz *\n1: 456Mhz") == 96.0
    assert _active_dpm_mhz("S: 0Mhz *\n1: 500Mhz") == 0.0
    assert _active_dpm_mhz("0: 96Mhz\n1: 456Mhz") is None
    assert _active_dpm_mhz(None) is None


def test_direct_transport_log_requires_p2p_ipc_routes():
    log = "\n".join(
        [
            "NCCL INFO Connected all rings, use ring PXN 0 GDR 1",
            "NCCL INFO Channel 00/0 : 0[f3000] -> 1[d3000] via P2P/IPC",
            "NCCL INFO Channel 00/0 : 1[d3000] -> 0[f3000] via P2P/IPC",
        ]
    )
    result = validate_direct_transport_log(log, world_size=2)
    assert result["verified"] is True
    assert result["effective"] == "peer_ipc"
    assert result["p2p_ipc_route_count"] == 2

    with pytest.raises(RuntimeError, match="verification failed"):
        validate_direct_transport_log(
            log
            + "\nNCCL INFO Channel 01/0 : "
            "0[f3000] -> 1[d3000] via NET/Socket",
            world_size=2,
        )
    with pytest.raises(RuntimeError, match="verification failed"):
        validate_direct_transport_log("", world_size=4)


def test_single_gpu_transport_is_explicitly_not_applicable():
    result = validate_direct_transport_log("", world_size=1)
    assert result["verified"] is True
    assert result["effective"] == "not_applicable_single_gpu"


def test_telemetry_summary_preserves_nulls_instead_of_fabricating_values():
    samples = [
        {
            "elapsed_seconds": 0.0,
            "devices": [
                {
                    "torch_index": 0,
                    "pci": "0000:01:00.0",
                    "temperature_c": {
                        "edge": 50.0,
                        "junction": None,
                        "memory": 60.0,
                    },
                    "power_w": 100.0,
                    "clock_mhz": {"core": 2000.0, "memory": None},
                    "vram_bytes": {"used": 1024, "total": 2048},
                    "gpu_busy_percent": 90.0,
                }
            ],
        },
        {
            "elapsed_seconds": 0.2,
            "devices": [
                {
                    "torch_index": 0,
                    "pci": "0000:01:00.0",
                    "temperature_c": {
                        "edge": 52.0,
                        "junction": None,
                        "memory": 62.0,
                    },
                    "power_w": None,
                    "clock_mhz": {"core": 2100.0, "memory": None},
                    "vram_bytes": {"used": 1536, "total": 2048},
                    "gpu_busy_percent": 95.0,
                }
            ],
        },
    ]
    summary = _summarize_telemetry_samples(samples)["0000:01:00.0"]
    assert summary["sample_count"] == 2
    assert summary["metrics"]["temperature_edge_c"]["median"] == 51.0
    assert summary["metrics"]["temperature_junction_c"]["count"] == 0
    assert summary["metrics"]["temperature_junction_c"]["missing_count"] == 2
    assert summary["metrics"]["power_w"]["count"] == 1
    assert summary["metrics"]["power_w"]["missing_count"] == 1


def test_idle_precondition_rejects_competing_gpu_work(monkeypatch):
    monkeypatch.setattr(
        real_workload,
        "_telemetry_sources",
        lambda index: {"torch_index": index},
    )
    monkeypatch.setattr(
        real_workload,
        "_sample_device",
        lambda source: {
            "torch_index": source["torch_index"],
            "pci": "0000:01:00.0",
            "gpu_busy_percent": 100.0,
            "vram_bytes": {"used": 1024**3, "total": 32 * 1024**3},
        },
    )
    with pytest.raises(RuntimeError, match="not idle"):
        _wait_for_idle_gpus(device_count=1, timeout_seconds=0.0)


def test_idle_precondition_requires_three_clean_observations(monkeypatch):
    calls = []
    monkeypatch.setattr(
        real_workload,
        "_telemetry_sources",
        lambda index: {"torch_index": index},
    )

    def sample(source):
        calls.append(source)
        return {
            "torch_index": source["torch_index"],
            "pci": "0000:01:00.0",
            "gpu_busy_percent": 0.0,
            "vram_bytes": {"used": 0, "total": 32 * 1024**3},
        }

    monkeypatch.setattr(real_workload, "_sample_device", sample)
    monkeypatch.setattr(real_workload.time, "sleep", lambda _: None)
    result = _wait_for_idle_gpus(device_count=1, timeout_seconds=1.0)
    assert result["verified"] is True
    assert len(result["observations"]) == 3
    assert len(calls) == 3
