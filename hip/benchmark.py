"""Clean-build, validate, and benchmark the fused PEPS HIP decoder.

This runner refuses guessed architectures, verifies the embedded code object,
runs full-output fp32/fp16 parity, measures all four paper configurations with
HIP events, and atomically emits JSON plus CSV artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(SCRIPT_DIR))

from export_fixture import (  # noqa: E402
    METHOD_SPECS,
    make_random_fixture,
    read_output,
    write_fixture,
)


class BenchmarkError(RuntimeError):
    """A build, provenance, parity, or benchmark invariant failed."""


def _run(
    command: Sequence[str | Path],
    *,
    env: dict[str, str] | None = None,
    timeout: int = 600,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(value) for value in command],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        rendered = " ".join(str(value) for value in command)
        raise BenchmarkError(
            f"command failed ({result.returncode}): {rendered}\n"
            f"{result.stdout[-3000:]}\n{result.stderr[-5000:]}"
        )
    return result


def _tool(candidates: Sequence[str | Path]) -> str | None:
    for candidate in candidates:
        value = str(candidate)
        if os.path.isabs(value) and os.path.isfile(value) and os.access(value, os.X_OK):
            return value
        resolved = shutil.which(value)
        if resolved:
            return resolved
    return None


def detect_compiler() -> str:
    compiler = _tool(
        ("/opt/rocm/bin/hipcc", "/opt/rocm/bin/amdclang++", "hipcc")
    )
    if compiler is None:
        raise BenchmarkError("no compatible HIP compiler found")
    return compiler


def detect_arch(device: int) -> str:
    environment = dict(os.environ)
    environment["HIP_VISIBLE_DEVICES"] = str(device)
    detector = _tool(("offload-arch", "/opt/rocm/bin/offload-arch"))
    if detector:
        output = _run([detector], env=environment, timeout=30).stdout
        arches = re.findall(r"(?m)^gfx[0-9a-f]+$", output)
        if arches:
            return arches[0]
    rocminfo = _tool(("rocminfo", "/usr/bin/rocminfo"))
    if rocminfo:
        output = _run([rocminfo], env=environment, timeout=60).stdout
        match = re.search(r"\bgfx[0-9a-f]+\b", output)
        if match:
            return match.group(0)
    raise BenchmarkError("no local AMD GPU architecture detected; refusing a guess")


def _git(args: Sequence[str]) -> str:
    return _run(["git", *args], timeout=60).stdout.strip()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def compile_clean(arch: str, compiler: str) -> dict[str, Any]:
    source = SCRIPT_DIR / "fused_peps_kernel.hip"
    git_sha = _git(["rev-parse", "--short=8", "HEAD"])
    source_sha = source_fingerprint(
        [
            source,
            SCRIPT_DIR / "export_fixture.py",
            SCRIPT_DIR / "benchmark.py",
        ]
    )
    build_dir = SCRIPT_DIR / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    binary = build_dir / f"fused_peps_{arch}_{git_sha}"

    command = [compiler]
    if Path(compiler).name.startswith("amdclang"):
        command.extend(["-x", "hip"])
    command.extend(
        [
            "-O3",
            "-DNDEBUG",
            f"--offload-arch={arch}",
            f'-DPEPS_GIT_SHA="{git_sha}"',
            f'-DPEPS_TARGET_ISA="{arch}"',
            str(source),
        ]
    )
    if Path(compiler).name.startswith("amdclang"):
        command.extend(["-L/opt/rocm/lib", "-lamdhip64"])
    command.extend(["-o", str(binary)])
    environment = dict(os.environ)
    environment["PATH"] = "/opt/rocm/bin:" + environment.get("PATH", "")
    include = environment.get("CPLUS_INCLUDE_PATH")
    environment["CPLUS_INCLUDE_PATH"] = (
        "/opt/rocm/include"
        if not include
        else "/opt/rocm/include" + os.pathsep + include
    )
    compiled = _run(command, env=environment, timeout=900)
    if not binary.is_file():
        raise BenchmarkError(f"compiler succeeded without creating {binary}")

    inspector = _tool(
        ("roc-obj-ls", "/opt/rocm/bin/roc-obj-ls", "/usr/bin/roc-obj-ls")
    )
    if inspector is None:
        raise BenchmarkError("roc-obj-ls is required for ISA provenance")
    inspection = _run([inspector, binary], timeout=60).stdout.strip()
    targets = sorted(set(re.findall(r"hipv4-[^\s]*--(gfx[0-9a-f]+)", inspection)))
    if targets != [arch]:
        raise BenchmarkError(
            f"code object targets {targets!r}, expected exactly [{arch!r}]"
        )

    compiler_version = _run([compiler, "--version"], timeout=60).stdout.strip()
    return {
        "binary": str(binary.relative_to(ROOT)),
        "binary_name": binary.name,
        "binary_sha256": sha256_path(binary),
        "source_sha256": source_sha,
        "git_sha": git_sha,
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain", "--", "hip", "tests/test_hip_parity.py", "tests/test_hip_export.py", "results/hip_latency.csv", "results/hip_latency.schema.json"])),
        "compiler": compiler,
        "compiler_version": compiler_version,
        "command": command,
        "compile_stdout": compiled.stdout,
        "compile_stderr": compiled.stderr,
        "code_object_inspector": inspector,
        "code_object_inspection": inspection,
        "code_object_targets": targets,
        "target_isa": arch,
    }


def run_parity(binary: Path, device: int) -> list[dict[str, Any]]:
    environment = dict(os.environ)
    environment["HIP_VISIBLE_DEVICES"] = str(device)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="peps_hip_parity_") as directory:
        temporary = Path(directory)
        cases = [
            (
                name,
                make_random_fixture(
                    name,
                    channels=16,
                    grid_height=11,
                    grid_width=9,
                    points=37,
                    hidden=64,
                    output=3,
                    seed=71,
                ),
            )
            for name in METHOD_SPECS
        ]
        border = np.array(
            [[-0.25, 0.5], [0.0, 0.0], [1.0, 1.0], [0.5, 1.25]],
            dtype=np.float32,
        )
        cases.append(
            (
                "tail-border-stride",
                make_random_fixture(
                    "grid-pink-peps-4f",
                    channels=7,
                    grid_height=5,
                    grid_width=9,
                    points=33,
                    hidden=17,
                    output=5,
                    coords=np.resize(border, (33, 2)).astype(np.float32),
                    seed=73,
                ),
            )
        )
        for case_name, fixture in cases:
            fixture_path = temporary / f"{case_name}.bin"
            write_fixture(fixture_path, fixture)
            for precision, tolerance in (("fp32", 1e-3), ("fp16", 4e-3)):
                output_path = temporary / f"{case_name}.{precision}.out"
                result = _run(
                    [binary, "fixture", precision, fixture_path, output_path],
                    env=environment,
                    timeout=300,
                )
                output_mode, actual = read_output(output_path)
                expected = fixture.reference(precision)
                difference = np.abs(
                    actual.astype(np.float64) - expected.astype(np.float64)
                )
                record = {
                    "case": case_name,
                    "method": fixture.method.name,
                    "mode": output_mode,
                    "precision": precision,
                    "shape": list(actual.shape),
                    "max_abs_error": float(difference.max()),
                    "mean_abs_error": float(difference.mean()),
                    "rmse": float(np.sqrt(np.mean(difference**2))),
                    "tolerance": tolerance,
                    "passed": bool(float(difference.max()) < tolerance),
                    "binary_stdout": result.stdout.strip(),
                }
                records.append(record)
                if not record["passed"]:
                    raise BenchmarkError(
                        f"{case_name} {precision} parity max error "
                        f"{record['max_abs_error']:.6g} >= {tolerance}"
                    )
    return records


def _json_lines(output: str) -> list[dict[str, Any]]:
    records = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            records.append(json.loads(stripped))
    return records


def rocm_smi_snapshot() -> dict[str, Any] | None:
    tool = _tool(("rocm-smi", "/usr/bin/rocm-smi"))
    if tool is None:
        return None
    result = _run(
        [
            tool,
            "--showproductname",
            "--showdriverversion",
            "--showclocks",
            "--showpower",
            "--json",
        ],
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip(), "stdout": result.stdout.strip()}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"raw": result.stdout.strip()}


def run_benchmark(
    binary: Path,
    *,
    device: int,
    side: int,
    warmup: int,
    iters: int,
    arch: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    environment = dict(os.environ)
    environment["HIP_VISIBLE_DEVICES"] = str(device)
    before = rocm_smi_snapshot()
    result = _run(
        [binary, "benchmark", "all", side, warmup, iters],
        env=environment,
        timeout=max(900, iters * 30),
    )
    after = rocm_smi_snapshot()
    records = _json_lines(result.stdout)
    if set(record.get("method") for record in records) != set(METHOD_SPECS):
        raise BenchmarkError(
            f"benchmark emitted methods {[r.get('method') for r in records]!r}"
        )
    for record in records:
        runtime_arch = str(record["isa"]).split(":", 1)[0]
        if runtime_arch != arch or record["compiled_target"] != arch:
            raise BenchmarkError(
                f"runtime/compiled ISA mismatch: {runtime_arch}, "
                f"{record['compiled_target']}, expected {arch}"
            )
    return records, before, after


def preflight_benchmark(
    binary: Path,
    *,
    device: int,
    side: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """Estimate full protocol time from a safe 64² baseline smoke run."""

    environment = dict(os.environ)
    environment["HIP_VISIBLE_DEVICES"] = str(device)
    smoke_side, smoke_warmup, smoke_iters = 64, 1, 3
    result = _run(
        [
            binary,
            "benchmark",
            "bi-grid",
            smoke_side,
            smoke_warmup,
            smoke_iters,
        ],
        env=environment,
        timeout=180,
    )
    records = _json_lines(result.stdout)
    if len(records) != 1:
        raise BenchmarkError("preflight did not emit exactly one JSON receipt")
    measurement = records[0]
    point_ratio = (side * side) / float(smoke_side * smoke_side)
    estimated_iteration_ms = float(measurement["median_ms"]) * point_ratio
    estimated_seconds = (
        estimated_iteration_ms
        * (warmup + iters)
        * len(METHOD_SPECS)
        / 1000.0
    )
    return {
        "smoke": measurement,
        "estimated_full_protocol_seconds": estimated_seconds,
        "estimate_model": "linear_in_output_points_times_four_methods",
    }


def rocm_version() -> str:
    for path in (
        Path("/opt/rocm/.info/version"),
        Path("/opt/rocm/.info/version-dev"),
    ):
        if path.is_file():
            return path.read_text().strip()
    return "unknown"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


CSV_FIELDS = [
    "schema_version",
    "benchmark_kind",
    "kernel",
    "method",
    "mode",
    "implementation",
    "isa",
    "box",
    "rocm_version",
    "dtype",
    "output_width",
    "output_height",
    "grid_width",
    "grid_height",
    "feature_dim",
    "num_frequencies",
    "selected_feature_dim",
    "hidden_dim",
    "hidden_layers",
    "out_dim",
    "activation",
    "workload",
    "warmup",
    "iters",
    "ms_per_iter",
    "median_ms",
    "p95_ms",
    "mean_ms",
    "stddev_ms",
    "min_ms",
    "max_ms",
    "paper_reference_ms",
    "paper_reference_source",
    "parity_status",
    "max_abs_error",
    "mean_abs_error",
    "git_sha",
    "source_sha256",
    "binary_sha256",
    "binary",
    "code_object_target",
    "provenance",
    "comparable_to_paper",
]


def update_latency_csv(
    path: Path,
    measurements: Sequence[dict[str, Any]],
    parity: Sequence[dict[str, Any]],
    build: dict[str, Any],
    *,
    hostname: str,
    version: str,
) -> None:
    rows: list[dict[str, str]] = []
    previous_fields: list[str] = []
    if path.is_file():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            previous_fields = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    rows = [
        row
        for row in rows
        if not (
            row.get("schema_version") == "3"
            and row.get("implementation") == "fused_rocwmma_fp16"
            and row.get("code_object_target") == build["target_isa"]
        )
    ]
    parity_by_method = {
        record["method"]: record
        for record in parity
        if record["precision"] == "fp16" and record["case"] in METHOD_SPECS
    }
    for measurement in measurements:
        method = measurement["method"]
        correctness = parity_by_method[method]
        rows.append(
            {
                "schema_version": "3",
                "benchmark_kind": "integrated_paper_workload",
                "kernel": "fused_peps",
                "method": method,
                "mode": str(measurement["mode"]),
                "implementation": "fused_rocwmma_fp16",
                "isa": str(measurement["isa"]).split(":", 1)[0],
                "box": hostname,
                "rocm_version": version,
                "dtype": "fp16",
                "output_width": str(measurement["output_width"]),
                "output_height": str(measurement["output_height"]),
                "grid_width": str(measurement["grid_width"]),
                "grid_height": str(measurement["grid_height"]),
                "feature_dim": str(measurement["feature_dim"]),
                "num_frequencies": str(measurement["num_frequencies"]),
                "selected_feature_dim": str(measurement["selected_feature_dim"]),
                "hidden_dim": str(measurement["hidden_dim"]),
                "hidden_layers": str(measurement["hidden_layers"]),
                "out_dim": str(measurement["out_dim"]),
                "activation": str(measurement["activation"]),
                "workload": "1024x1024_rgb_grid1024_c16_h64x3",
                "warmup": str(measurement["warmup"]),
                "iters": str(measurement["iters"]),
                "ms_per_iter": f"{measurement['median_ms']:.6f}",
                "median_ms": f"{measurement['median_ms']:.6f}",
                "p95_ms": f"{measurement['p95_ms']:.6f}",
                "mean_ms": f"{measurement['mean_ms']:.6f}",
                "stddev_ms": f"{measurement['stddev_ms']:.6f}",
                "min_ms": f"{measurement['min_ms']:.6f}",
                "max_ms": f"{measurement['max_ms']:.6f}",
                "paper_reference_ms": (
                    f"{measurement['paper_reference_ms']:.6f}"
                ),
                "paper_reference_source": measurement[
                    "paper_reference_source"
                ],
                "parity_status": "passed",
                "max_abs_error": f"{correctness['max_abs_error']:.9g}",
                "mean_abs_error": f"{correctness['mean_abs_error']:.9g}",
                "git_sha": build["git_sha"],
                "source_sha256": build["source_sha256"],
                "binary_sha256": build["binary_sha256"],
                "binary": build["binary"],
                "code_object_target": build["target_isa"],
                "provenance": (
                    f"clean_build:{build['binary_name']};"
                    f"roc_obj_ls:{build['target_isa']}"
                ),
                "comparable_to_paper": "false",
            }
        )
    fields = CSV_FIELDS + [
        field for field in previous_fields if field not in CSV_FIELDS
    ]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", help="target ISA; defaults to detected local ISA")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--side", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--max-estimated-seconds",
        type=float,
        default=300.0,
        help="abort before paper-scale timing when the 64² preflight exceeds this budget",
    )
    parser.add_argument(
        "--force-slow",
        action="store_true",
        help="run even when the preflight exceeds --max-estimated-seconds",
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--json",
        type=Path,
        help="output bundle (default results/hip_benchmark_<isa>.json)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    detected = None if args.build_only and args.arch else detect_arch(args.device)
    arch = args.arch or detected
    if arch is None or not re.fullmatch(r"gfx[0-9a-f]+", arch):
        raise BenchmarkError(f"invalid or unknown target ISA {arch!r}")
    if not args.build_only and detected != arch:
        raise BenchmarkError(
            f"requested {arch}, but selected local device reports {detected}"
        )
    compiler = detect_compiler()
    build = compile_clean(arch, compiler)
    binary = ROOT / build["binary"]
    output = args.json or RESULTS_DIR / f"hip_benchmark_{arch}.json"
    if not output.is_absolute():
        output = ROOT / output
    if args.build_only:
        print(json.dumps({"build": build}, indent=2, sort_keys=True))
        return 0

    parity = run_parity(binary, args.device)
    preflight = preflight_benchmark(
        binary,
        device=args.device,
        side=args.side,
        warmup=args.warmup,
        iters=args.iters,
    )
    if (
        preflight["estimated_full_protocol_seconds"]
        > args.max_estimated_seconds
        and not args.force_slow
    ):
        blocked = {
            "schema_version": 3,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "blocked-performance",
            "build": build,
            "hardware": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "device_index": args.device,
                "detected_isa": detected,
                "rocm_version": rocm_version(),
            },
            "parity": parity,
            "preflight": preflight,
            "requested_protocol": {
                "side": args.side,
                "warmup": args.warmup,
                "iters": args.iters,
                "max_estimated_seconds": args.max_estimated_seconds,
            },
            "reason": (
                "safe preflight predicts the repeated paper-scale benchmark "
                "will exceed the configured runtime budget; no latency row "
                "was written"
            ),
        }
        _atomic_json(output, blocked)
        print(json.dumps({
            "status": blocked["status"],
            "result": str(output.relative_to(ROOT)),
            "estimated_full_protocol_seconds": preflight[
                "estimated_full_protocol_seconds"
            ],
        }, sort_keys=True))
        return 2
    measurements, power_before, power_after = run_benchmark(
        binary,
        device=args.device,
        side=args.side,
        warmup=args.warmup,
        iters=args.iters,
        arch=arch,
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    bundle = {
        "schema_version": 3,
        "generated_at": generated_at,
        "status": "passed",
        "build": build,
        "hardware": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "device_index": args.device,
            "detected_isa": detected,
            "rocm_version": rocm_version(),
            "power_clock_before": power_before,
            "power_clock_after": power_after,
        },
        "protocol": {
            "timing": "HIP events, one start/stop pair per iteration",
            "output_resolution": [args.side, args.side],
            "grid_resolution": [1024, 1024],
            "feature_dim": 16,
            "hidden_layers": 3,
            "hidden_dim": 64,
            "output_dim": 3,
            "warmup": args.warmup,
            "timed_iterations": args.iters,
            "summary": ["median_ms", "p95_ms"],
            "preflight": preflight,
            "method_order": "each method run to completion in turn",
            "clock_settling": "none",
            "latency_caveat": (
                "This protocol does not spin the card to steady clocks and "
                "does not interleave methods, so whichever method runs first "
                "absorbs the clock ramp. On gfx1201 that inflated the first "
                "method 5.7x and made the reported ordering an artefact of "
                "measurement order. Use hip/stable_latency.py for any latency "
                "claim. The build, parity and code-object provenance in this "
                "receipt do not depend on the timing protocol and stand."
            ),
        },
        "parity": parity,
        "measurements": measurements,
        "paper_comparison": {
            "reference_gpu": "AMD Radeon RX 9070 XT",
            "precision_and_timing_protocol_reported": False,
            "directly_comparable": False,
            "reason": (
                "The paper does not report precision, warmup, iteration, or "
                "synchronization details and does not release its HIP kernel."
            ),
        },
        "rdna35_validation": {
            "status": "deferred",
            "target_isa": "gfx1151",
        },
    }
    _atomic_json(output, bundle)
    update_latency_csv(
        RESULTS_DIR / "hip_latency.csv",
        measurements,
        parity,
        build,
        hostname=socket.gethostname(),
        version=rocm_version(),
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "result": str(output.relative_to(ROOT)),
                "binary": build["binary"],
                "isa": arch,
                "methods": {
                    value["method"]: {
                        "median_ms": value["median_ms"],
                        "p95_ms": value["p95_ms"],
                    }
                    for value in measurements
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
