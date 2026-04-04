# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import json
import sys
from pathlib import Path

import pytest


PLOTTER_PATH = (Path(__file__).resolve().parents[2] / "offline_bench" /
                "manual_multinode" / "plot_manual_multinode_metrics.py")


def load_plotter_module():
    spec = importlib.util.spec_from_file_location(
        "plot_manual_multinode_metrics",
        PLOTTER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_summary(
    root: Path,
    *,
    model: str,
    dataset: str,
    case_name: str,
    timestamp: str,
    tpot_by_e2e: float,
    tpot_without_queue_ms: float,
    ttft_ms: float,
) -> None:
    benchmark_dir = root / model / dataset / case_name / timestamp / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "tpot_by_e2e": {
            "mean": tpot_by_e2e,
            "p50": tpot_by_e2e,
            "p99": tpot_by_e2e,
        },
        "tpot_without_queue_ms": {
            "mean": tpot_without_queue_ms,
            "p50": tpot_without_queue_ms,
            "p99": tpot_without_queue_ms,
        },
        "ttft_ms": {
            "mean": ttft_ms,
            "p50": ttft_ms,
            "p99": ttft_ms,
        },
    }
    (benchmark_dir / "summary.json").write_text(
        json.dumps(summary) + "\n",
        encoding="utf-8",
    )


@pytest.mark.benchmark
def test_main_renders_separate_figures_per_model_and_dataset(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plotter = load_plotter_module()
    root = tmp_path / "manual_multinode"
    output_dir = tmp_path / "plots"

    write_summary(
        root,
        model="KIMI",
        dataset="issue01_random",
        case_name="dp16cp2-mem85-bs384-rate10-dur600",
        timestamp="20260404-010000",
        tpot_by_e2e=10.0,
        tpot_without_queue_ms=9.0,
        ttft_ms=100.0,
    )
    write_summary(
        root,
        model="KIMI",
        dataset="issue01_random",
        case_name="dp16cp2-mem85-bs384-rate20-dur600",
        timestamp="20260404-010001",
        tpot_by_e2e=11.0,
        tpot_without_queue_ms=10.0,
        ttft_ms=110.0,
    )
    write_summary(
        root,
        model="KIMI",
        dataset="issue03_random",
        case_name="dp16cp2-mem85-bs384-rate10-dur600",
        timestamp="20260404-010002",
        tpot_by_e2e=12.0,
        tpot_without_queue_ms=11.0,
        ttft_ms=120.0,
    )
    write_summary(
        root,
        model="DPSK",
        dataset="issue01_random",
        case_name="dp32-mem90-bs256-rate10-dur600",
        timestamp="20260404-010003",
        tpot_by_e2e=13.0,
        tpot_without_queue_ms=12.0,
        ttft_ms=130.0,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PLOTTER_PATH),
            "--root",
            str(root),
            "--output-dir",
            str(output_dir),
            "--models",
            "KIMI",
            "DPSK",
        ],
    )

    plotter.main()

    generated_files = sorted(path.name for path in output_dir.iterdir())
    assert generated_files == [
        "DPSK_issue01_random_manual_multinode_metrics.png",
        "KIMI_issue01_random_manual_multinode_metrics.png",
        "KIMI_issue03_random_manual_multinode_metrics.png",
    ]
