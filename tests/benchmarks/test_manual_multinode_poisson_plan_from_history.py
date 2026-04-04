# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import json
import sys
from pathlib import Path

import pytest


BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
RUNNER_PATH = BENCHMARKS_DIR / "manual_multinode_poisson_runner.py"
PLANNER_PATH = BENCHMARKS_DIR / "manual_multinode_poisson_plan_from_history.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "manual_multinode_poisson_runner",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_planner_module():
    load_runner_module()
    spec = importlib.util.spec_from_file_location(
        "manual_multinode_poisson_plan_from_history",
        PLANNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_case(runner, *, strategy: str, request_rate: float):
    for case in runner.build_experiment_matrix("coarse10_then_mid5"):
        if (case.model == "kimi_k2_instruct_0905"
                and case.dataset == "issue01_random"
                and case.strategy == strategy
                and case.request_rate == request_rate):
            return case
    raise AssertionError(
        f"Unable to find template case for strategy={strategy}, rate={request_rate}"
    )


def write_history_case(case_dir: Path, *, status: str,
                       summary: dict[str, object] | None = None) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_manifest.json").write_text(
        json.dumps({"status": status}) + "\n",
        encoding="utf-8",
    )
    if summary is not None:
        benchmark_dir = case_dir / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        (benchmark_dir / "summary.json").write_text(
            json.dumps(summary) + "\n",
            encoding="utf-8",
        )


@pytest.mark.benchmark
def test_template_cases_for_plan_supports_non_default_dataset_and_model() -> None:
    planner = load_planner_module()

    cases = planner.template_cases_for_plan(
        model="deepseek_v3_1024k",
        dataset="issue05_random",
        strategies=("dp32", ),
        rate_plan="coarse10_then_mid5",
        bench_duration_sec=None,
    )

    assert len(cases) == 17
    assert {case.model for case in cases} == {"deepseek_v3_1024k"}
    assert {case.dataset for case in cases} == {"issue05_random"}
    assert {case.strategy for case in cases} == {"dp32"}


@pytest.mark.benchmark
def test_build_plan_rows_starts_below_first_tpot_threshold_hit(
        tmp_path: Path) -> None:
    runner = load_runner_module()
    planner = load_planner_module()
    artifact_root = tmp_path / "artifacts"

    case40 = find_case(runner, strategy="dp4dcp8", request_rate=40.0)
    history_dir = (runner.case_artifact_group_dir(artifact_root, case40) /
                   "20260404-120000")
    write_history_case(
        history_dir,
        status="ok",
        summary={
            "tpot_by_e2e": {
                "mean": 120.0,
            },
        },
    )

    rows = planner.build_plan_rows(
        artifact_root=artifact_root,
        model="kimi_k2_instruct_0905",
        dataset="issue01_random",
        strategies=("dp4dcp8", ),
        rate_plan="coarse10_then_mid5",
        ignore_bs=True,
        bench_duration_sec=None,
    )

    assert [row.case.request_rate for row in rows] == [35.0, 30.0, 25.0, 20.0,
                                                       15.0, 10.0]
    assert {row.reason for row in rows} == {"start_below_tpot100_at_rate40"}
    assert {row.historical_reference for row in rows} == {str(history_dir)}

    csv_path = tmp_path / "plan.csv"
    planner.write_plan_csv(csv_path, rows)
    loaded_cases = runner.load_cases_from_csv(csv_path)
    assert [case.request_rate for case in loaded_cases] == [35.0, 30.0, 25.0,
                                                            20.0, 15.0, 10.0]


@pytest.mark.benchmark
def test_build_plan_rows_falls_back_to_timeout_rate_when_no_threshold_hit(
        tmp_path: Path) -> None:
    runner = load_runner_module()
    planner = load_planner_module()
    artifact_root = tmp_path / "artifacts"

    case40 = find_case(runner, strategy="dp8dcp4", request_rate=40.0)
    history_dir = (runner.case_artifact_group_dir(artifact_root, case40) /
                   "20260404-130000")
    write_history_case(
        history_dir,
        status="timed_out",
        summary=None,
    )

    rows = planner.build_plan_rows(
        artifact_root=artifact_root,
        model="kimi_k2_instruct_0905",
        dataset="issue01_random",
        strategies=("dp8dcp4", ),
        rate_plan="coarse10_then_mid5",
        ignore_bs=True,
        bench_duration_sec=None,
    )

    assert [row.case.request_rate for row in rows] == [35.0, 30.0, 25.0, 20.0,
                                                       15.0, 10.0]
    assert {row.reason for row in rows} == {"start_from_timeout_rate35"}
    assert {row.historical_reference for row in rows} == {str(history_dir)}


@pytest.mark.benchmark
def test_build_plan_rows_filters_exact_cases_runner_would_skip(
        tmp_path: Path) -> None:
    runner = load_runner_module()
    planner = load_planner_module()
    artifact_root = tmp_path / "artifacts"

    case30 = find_case(runner, strategy="dp4dcp8", request_rate=30.0)
    history_dir = (runner.case_artifact_group_dir(artifact_root, case30) /
                   "20260404-140000")
    write_history_case(
        history_dir,
        status="ok",
        summary={
            "tpot_by_e2e": {
                "mean": 80.0,
            },
        },
    )

    rows = planner.build_plan_rows(
        artifact_root=artifact_root,
        model="kimi_k2_instruct_0905",
        dataset="issue01_random",
        strategies=("dp4dcp8", ),
        rate_plan="coarse10_then_mid5",
        ignore_bs=True,
        bench_duration_sec=None,
    )

    assert 30.0 not in [row.case.request_rate for row in rows]
    assert [row.case.request_rate for row in rows[:5]] == [90.0, 85.0, 80.0,
                                                           75.0, 70.0]


@pytest.mark.benchmark
def test_build_plan_rows_filters_group_rates_runner_would_skip(
        tmp_path: Path) -> None:
    runner = load_runner_module()
    planner = load_planner_module()
    artifact_root = tmp_path / "artifacts"

    case30 = find_case(runner, strategy="dp4dcp8", request_rate=30.0)
    history_dir = (runner.case_artifact_group_dir(artifact_root, case30) /
                   "20260404-150000")
    write_history_case(
        history_dir,
        status="failed",
        summary=None,
    )

    rows = planner.build_plan_rows(
        artifact_root=artifact_root,
        model="kimi_k2_instruct_0905",
        dataset="issue01_random",
        strategies=("dp4dcp8", ),
        rate_plan="coarse10_then_mid5",
        ignore_bs=True,
        bench_duration_sec=None,
    )

    assert [row.case.request_rate for row in rows] == [25.0, 20.0, 15.0, 10.0]
