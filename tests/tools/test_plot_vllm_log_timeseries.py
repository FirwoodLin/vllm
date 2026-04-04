# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


PLOTTER_PATH = Path(__file__).resolve().parents[2] / "tools" / (
    "plot_vllm_log_timeseries.py")


def load_plotter_module():
    spec = importlib.util.spec_from_file_location(
        "plot_vllm_log_timeseries",
        PLOTTER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_frontend_log(path: Path) -> None:
    path.write_text(
        "\n".join([
            "INFO 04-04 13:19:55 [loggers.py:265] Engine 000: "
            "Avg prompt throughput: 0.0 tokens/s, "
            "Avg generation throughput: 10.0 tokens/s, "
            "Running: 2 reqs, Waiting: 3 reqs, Waiting tokens: 100, "
            "Waiting head tokens: 40, GPU KV cache usage: 10.0%, "
            "Prefix cache hit rate: 0.0%, External prefix cache hit rate: 100.0%",
            "INFO 04-04 13:19:55 [loggers.py:265] Engine 001: "
            "Avg prompt throughput: 0.0 tokens/s, "
            "Avg generation throughput: 20.0 tokens/s, "
            "Running: 4 reqs, Waiting: 1 reqs, Waiting tokens: 50, "
            "Waiting head tokens: 20, GPU KV cache usage: 25.0%, "
            "Prefix cache hit rate: 0.0%, External prefix cache hit rate: 100.0%",
            "INFO 04-04 13:19:56 [core.py:465] unrelated line",
            "INFO 04-04 13:19:56 [loggers.py:265] Engine 000: "
            "Avg prompt throughput: 0.0 tokens/s, "
            "Avg generation throughput: 12.0 tokens/s, "
            "Running: 1 reqs, Waiting: 4 reqs, Waiting tokens: 80, "
            "Waiting head tokens: 30, GPU KV cache usage: 12.5%, "
            "Prefix cache hit rate: 0.0%, External prefix cache hit rate: 100.0%",
            "INFO 04-04 13:19:56 [loggers.py:265] Engine 001: "
            "Avg prompt throughput: 0.0 tokens/s, "
            "Avg generation throughput: 18.0 tokens/s, "
            "Running: 3 reqs, Waiting: 0 reqs, Waiting tokens: 0, "
            "Waiting head tokens: 0, GPU KV cache usage: 30.0%, "
            "Prefix cache hit rate: 0.0%, External prefix cache hit rate: 100.0%",
        ]) + "\n",
        encoding="utf-8",
    )


def write_multi_engine_frontend_log(path: Path, engine_count: int) -> None:
    lines: list[str] = []
    for second_offset, timestamp in enumerate(("04-04 13:19:55",
                                               "04-04 13:19:56")):
        for engine_id in range(engine_count):
            waiting_tokens = 100 + engine_id * 3 - second_offset * 10
            waiting_head_tokens = 40 + engine_id - second_offset * 5
            gpu_kv_cache_usage = 10.0 + engine_id * 1.5 + second_offset * 2.5
            lines.append(
                f"INFO {timestamp} [loggers.py:265] Engine {engine_id:03d}: "
                "Avg prompt throughput: 0.0 tokens/s, "
                f"Avg generation throughput: {10.0 + engine_id:.1f} tokens/s, "
                f"Running: {1 + (engine_id % 4)} reqs, "
                f"Waiting: {engine_id % 3} reqs, "
                f"Waiting tokens: {waiting_tokens}, "
                f"Waiting head tokens: {waiting_head_tokens}, "
                f"GPU KV cache usage: {gpu_kv_cache_usage:.1f}%, "
                "Prefix cache hit rate: 0.0%, "
                "External prefix cache hit rate: 100.0%")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_log_extracts_expected_rows(tmp_path: Path) -> None:
    plotter = load_plotter_module()
    log_path = tmp_path / "frontend.log"
    write_frontend_log(log_path)

    df = plotter.parse_log(log_path, year=2026)

    assert list(df["engine_id"]) == [0, 1, 0, 1]
    assert list(df["waiting_tokens"]) == [100, 50, 80, 0]
    assert list(df["waiting_head_tokens"]) == [40, 20, 30, 0]
    assert list(df["gpu_kv_cache_usage_pct"]) == [10.0, 25.0, 12.5, 30.0]
    assert str(df["timestamp"].iloc[0]) == "2026-04-04 13:19:55"


def test_parse_engine_ids_accepts_ranges_and_lists() -> None:
    plotter = load_plotter_module()
    assert plotter.parse_engine_ids("0-2,5,7-8") == [0, 1, 2, 5, 7, 8]


def test_main_writes_csv_and_plots_for_run_dir(tmp_path: Path,
                                               monkeypatch:
                                               pytest.MonkeyPatch) -> None:
    plotter = load_plotter_module()
    run_dir = tmp_path / "20260404-131817"
    run_dir.mkdir()
    write_frontend_log(run_dir / "frontend.log")
    out_dir = tmp_path / "plots"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PLOTTER_PATH),
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--year",
            "2026",
            "--engine-ids",
            "1",
            "--dump-csv",
        ],
    )

    plotter.main()

    aggregate_csv = out_dir / "20260404-131817.aggregate.csv"
    parsed_csv = out_dir / "20260404-131817.parsed.csv"
    aggregate_png = out_dir / "20260404-131817.aggregate.png"
    per_engine_png = out_dir / "20260404-131817.per_engine.png"

    assert aggregate_csv.is_file()
    assert parsed_csv.is_file()
    assert aggregate_png.is_file()
    assert per_engine_png.is_file()
    assert aggregate_png.stat().st_size > 0
    assert per_engine_png.stat().st_size > 0

    with aggregate_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["waiting_tokens_total"] == "50"
    assert rows[0]["waiting_head_tokens_total"] == "20"
    assert rows[0]["gpu_kv_cache_usage_mean"] == "25.0"
    assert rows[0]["gpu_kv_cache_usage_max"] == "25.0"


def test_main_batch_root_writes_outputs_next_to_each_frontend_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plotter = load_plotter_module()
    batch_root = tmp_path / "DPSK" / "issue05_random"

    valid_run_a = batch_root / "dp32-mem90-bs256-rate10-dur600" / "20260404-075808"
    valid_run_b = batch_root / "dp32-mem90-bs256-rate15-dur600" / "20260404-081242"
    invalid_run = batch_root / "dp32-mem90-bs256-rate20-dur600" / "20260404-082730"

    valid_run_a.mkdir(parents=True)
    valid_run_b.mkdir(parents=True)
    invalid_run.mkdir(parents=True)

    write_frontend_log(valid_run_a / "frontend.log")
    write_frontend_log(valid_run_b / "frontend.log")
    (invalid_run / "frontend.log").write_text(
        "INFO 04-04 13:19:56 [core.py:465] unrelated line\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PLOTTER_PATH),
            "--batch-root",
            str(batch_root),
            "--year",
            "2026",
            "--plot-mode",
            "aggregate",
            "--dump-csv",
        ],
    )

    plotter.main()

    assert (valid_run_a / "20260404-075808.aggregate.png").is_file()
    assert (valid_run_a / "20260404-075808.aggregate.csv").is_file()
    assert (valid_run_a / "20260404-075808.parsed.csv").is_file()

    assert (valid_run_b / "20260404-081242.aggregate.png").is_file()
    assert (valid_run_b / "20260404-081242.aggregate.csv").is_file()
    assert (valid_run_b / "20260404-081242.parsed.csv").is_file()

    assert not (invalid_run / "20260404-082730.aggregate.png").exists()


def test_main_splits_per_engine_plots_into_groups_of_eight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plotter = load_plotter_module()
    run_dir = tmp_path / "20260404-131817"
    run_dir.mkdir()
    write_multi_engine_frontend_log(run_dir / "frontend.log", engine_count=17)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PLOTTER_PATH),
            "--run-dir",
            str(run_dir),
            "--year",
            "2026",
            "--plot-mode",
            "per-engine",
        ],
    )

    plotter.main()

    expected_plots = [
        run_dir / "20260404-131817.per_engine.part1.png",
        run_dir / "20260404-131817.per_engine.part2.png",
        run_dir / "20260404-131817.per_engine.part3.png",
    ]

    assert not (run_dir / "20260404-131817.per_engine.png").exists()
    for plot_path in expected_plots:
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 0
