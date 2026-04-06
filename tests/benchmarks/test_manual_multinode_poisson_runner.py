# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import io
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = (Path(__file__).resolve().parents[2] / "benchmarks" /
               "manual_multinode_poisson_runner.py")


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "manual_multinode_poisson_runner",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SequencedPollProcess:

    def __init__(self, poll_results: list[int | None]) -> None:
        self._poll_results = list(poll_results)
        self._last_result: int | None = None

    def poll(self) -> int | None:
        if self._poll_results:
            self._last_result = self._poll_results.pop(0)
        return self._last_result


def make_node_runtime(
    runner,
    tmp_path: Path,
    *,
    node_rank: int,
    host: str,
    poll_results: list[int | None],
    launch_log: str = "",
    runtime_log: str = "",
):
    launch_log_path = tmp_path / f"rank{node_rank}.launch.log"
    runtime_log_path = tmp_path / f"rank{node_rank}.log"
    launch_log_path.write_text(launch_log, encoding="utf-8")
    runtime_log_path.write_text(runtime_log, encoding="utf-8")
    return runner.NodeRuntime(
        node_rank=node_rank,
        host=host,
        process=SequencedPollProcess(poll_results),
        launch_log_path=launch_log_path,
        launch_log_handle=io.StringIO(),
        runtime_log_path=runtime_log_path,
    )


def write_case_benchmark_artifacts(case_dir: Path, *, manifest_status: str,
                                   summary: dict[str, object] | None,
                                   requests_present: bool = True,
                                   requests: list[dict[str, object]] | None = None,
                                   run_meta: dict[str, object] | None = None) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_manifest.json").write_text(
        json.dumps({"status": manifest_status}) + "\n",
        encoding="utf-8",
    )
    benchmark_dir = case_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    if summary is not None:
        (benchmark_dir / "summary.json").write_text(
            json.dumps(summary) + "\n",
            encoding="utf-8",
        )
    if run_meta is not None:
        (benchmark_dir / "run_meta.json").write_text(
            json.dumps(run_meta) + "\n",
            encoding="utf-8",
        )
    if requests_present:
        (benchmark_dir / "requests.jsonl").write_text(
            "".join(
                json.dumps(record) + "\n" for record in (requests or [{
                    "request_id": "r0",
                    "is_error": False,
                }])
            ),
            encoding="utf-8",
        )


@pytest.mark.benchmark
def test_resolve_case_uses_overridden_remote_hosts(tmp_path: Path) -> None:
    runner = load_runner_module()
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("prompt_len,output_len\n4,7\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    case = runner.ExperimentCase(
        name="manual_case",
        cluster="cluster_a",
        strategy="strategy_a",
        dataset="dataset_alias",
        model="model_alias",
        request_rate=10.0,
        remote_hosts_override=("node-b", "node-c", "node-d"),
    )
    resolved = runner.resolve_case(
        case,
        clusters={
            "cluster_a":
            runner.ClusterSpec(
                master_addr="10.0.0.1",
                master_port=29579,
                remote_hosts=("node-a", ),
            ),
        },
        strategies={
            "strategy_a":
            runner.StrategySpec(
                data_parallel_size=16,
                data_parallel_size_local=4,
                tensor_parallel_size=1,
            ),
        },
        datasets={"dataset_alias": str(dataset_path)},
        models={"model_alias": str(model_dir)},
    )

    assert resolved.cluster.nnodes == 4
    assert resolved.cluster.remote_hosts == ("node-b", "node-c", "node-d")
    assert resolved.dataset_path == dataset_path.resolve()
    assert resolved.model_path == model_dir.resolve()


@pytest.mark.benchmark
def test_build_parser_keeps_going_by_default() -> None:
    runner = load_runner_module()
    parser = runner.build_parser()

    assert parser.parse_args(["--list"]).keep_going is True
    assert parser.parse_args(["--list", "--no-keep-going"]).keep_going is False


@pytest.mark.benchmark
def test_build_experiment_matrix_filters_model_dataset_and_strategy() -> None:
    runner = load_runner_module()

    cases = runner.build_experiment_matrix(
        "coarse10_then_mid5",
        models=("deepseek_v3_1024k", ),
        datasets=("issue05_random", ),
        strategies=("dp32", ),
    )

    assert len(cases) == 17
    assert {case.model for case in cases} == {"deepseek_v3_1024k"}
    assert {case.dataset for case in cases} == {"issue05_random"}
    assert {case.strategy for case in cases} == {"dp32"}
    assert all(case.gpu_memory_utilization == pytest.approx(0.87)
               for case in cases)
    assert [case.request_rate for case in cases[:4]] == [10.0, 20.0, 30.0, 40.0]
    assert [case.request_rate for case in cases[-4:]] == [55.0, 65.0, 75.0,
                                                           85.0]


@pytest.mark.benchmark
def test_build_frontend_and_headless_argv_include_required_flags(
        tmp_path: Path) -> None:
    runner = load_runner_module()
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("prompt_len,output_len\n4,7\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output_dir = tmp_path / "benchmark"

    resolved = runner.resolve_case(
        runner.ExperimentCase(
            name="case_a",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_alias",
            model="model_alias",
            dispatch_policy="least_cache",
            request_rate=100.0,
            max_requests=9000,
            warmup_requests=32,
            max_num_seqs=512,
            max_model_len=32768,
            max_num_batched_tokens=16384,
            data_parallel_rpc_port=29550,
            request_id_prefix="req-",
            save_merged_parquet=True,
            frontend_extra_args=("--logging-step-timing-interval", "10"),
            headless_extra_args=("--logging-step-timing-interval", "10"),
        ),
        clusters={
            "cluster_a":
            runner.ClusterSpec(
                master_addr="10.0.0.1",
                master_port=29579,
                remote_hosts=("node-a", ),
            ),
        },
        strategies={
            "strategy_a":
            runner.StrategySpec(
                data_parallel_size=16,
                data_parallel_size_local=8,
                tensor_parallel_size=1,
                decode_context_parallel_size=1,
                enable_expert_parallel=True,
                attention_backend="FLASHMLA",
                all2all_backend="deepep_low_latency",
            ),
        },
        datasets={"dataset_alias": str(dataset_path)},
        models={"model_alias": str(model_dir)},
    )

    frontend_argv = runner.build_frontend_argv(resolved, output_dir)
    headless_argv = runner.build_headless_argv(resolved, node_rank=1)

    assert frontend_argv[:3] == [
        "python3",
        runner.HARNESS_ENTRYPOINT,
        "frontend",
    ]
    assert headless_argv[:3] == [
        "python3",
        runner.HARNESS_ENTRYPOINT,
        "headless-engine",
    ]
    assert "--input-csv" in frontend_argv
    assert str(dataset_path.resolve()) in frontend_argv
    assert "--request-rate" in frontend_argv
    assert "100" in frontend_argv
    assert "--output-dir" in frontend_argv
    assert str(output_dir) in frontend_argv
    assert "--save-merged-parquet" in frontend_argv
    assert "--request-id-prefix" in frontend_argv
    assert "--max-num-seqs" in frontend_argv
    assert "--max-model-len" in frontend_argv
    assert "--max-num-batched-tokens" in frontend_argv
    assert "--enable-expert-parallel" in frontend_argv
    assert "--attention-backend" in frontend_argv
    assert "--all2all-backend" in frontend_argv
    assert "--data-parallel-dispatch-policy" in frontend_argv
    assert "least_cache" in frontend_argv
    assert "--input-csv" not in headless_argv
    assert "--request-rate" not in headless_argv


@pytest.mark.benchmark
def test_launch_and_cleanup_commands_use_shared_artifact_paths(
        tmp_path: Path) -> None:
    runner = load_runner_module()
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("prompt_len,output_len\n4,7\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    resolved = runner.resolve_case(
        runner.ExperimentCase(
            name="case_a",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_alias",
            model="model_alias",
            request_rate=20.0,
            max_requests=100,
            warmup_requests=8,
            max_num_seqs=64,
            max_model_len=8192,
            data_parallel_rpc_port=29550,
            cleanup_worker_processes=True,
        ),
        clusters={
            "cluster_a":
            runner.ClusterSpec(
                master_addr="10.0.0.1",
                master_port=29579,
                remote_hosts=("node-a", ),
            ),
        },
        strategies={
            "strategy_a":
            runner.StrategySpec(
                data_parallel_size=8,
                data_parallel_size_local=4,
                tensor_parallel_size=1,
            ),
        },
        datasets={"dataset_alias": str(dataset_path)},
        models={"model_alias": str(model_dir)},
    )
    artifacts = runner.prepare_artifact_paths(
        tmp_path / "run",
        resolved.cluster.remote_hosts,
        run_tag=resolved.case.name,
    )

    frontend_command = runner.build_local_launch_command(
        resolved,
        artifacts,
        runner.build_frontend_argv(resolved, artifacts.benchmark_dir),
    )
    cleanup_command = runner.build_node_cleanup_command(
        resolved=resolved,
        pid_path=artifacts.frontend_pid_path,
        pgid_path=artifacts.frontend_pgid_path,
        include_frontend_pattern=True,
    )

    assert "child_pid=$!" in frontend_command
    assert "printf '%s\\n' \"$$\"" in frontend_command
    assert str(artifacts.frontend_log_path) in frontend_command
    assert str(artifacts.frontend_pid_path) in frontend_command
    assert str(artifacts.frontend_pgid_path) in frontend_command
    assert "wait \"$child_pid\"" in frontend_command
    assert "child_exit_code=$?" in frontend_command
    assert "status=$?" not in frontend_command
    assert str(artifacts.frontend_pgid_path) in cleanup_command
    assert "kill -TERM -- \"-$pgid\"" in cleanup_command
    assert "offline_poisson_harness.py frontend" in cleanup_command
    assert "--data-parallel-rpc-port 29550" in cleanup_command
    assert "--master-port 29579" in cleanup_command
    assert ("pgrep -f -- '^python3 /vllm/offline_poisson_harness.py frontend"
            "( |$)'" in cleanup_command)
    assert ("pkill -KILL -f -- '^python3 "
            "/vllm/offline_poisson_harness.py frontend( |$)'"
            in cleanup_command)
    assert "VLLM::Worker_" in cleanup_command


@pytest.mark.benchmark
def test_write_case_manifest_includes_commands_and_paths(tmp_path: Path) -> None:
    runner = load_runner_module()
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("prompt_len,output_len\n4,7\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    resolved = runner.resolve_case(
        runner.ExperimentCase(
            name="case_manifest",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_alias",
            model="model_alias",
            dispatch_policy="least_batch",
            request_rate=5.0,
        ),
        clusters={
            "cluster_a":
            runner.ClusterSpec(
                master_addr="10.0.0.1",
                master_port=29579,
                remote_hosts=("node-a", ),
            ),
        },
        strategies={
            "strategy_a":
            runner.StrategySpec(
                data_parallel_size=8,
                data_parallel_size_local=4,
                tensor_parallel_size=1,
            ),
        },
        datasets={"dataset_alias": str(dataset_path)},
        models={"model_alias": str(model_dir)},
    )
    artifacts = runner.prepare_artifact_paths(
        tmp_path / "run",
        resolved.cluster.remote_hosts,
        run_tag=resolved.case.name,
    )
    artifacts.case_dir.mkdir(parents=True, exist_ok=True)
    runner.write_case_manifest(
        resolved,
        artifacts,
        frontend_launch_command="frontend command",
        remote_launch_commands={1: "rank1 command"},
        local_cleanup_command="local cleanup",
        remote_cleanup_commands={1: "rank1 cleanup"},
        status="prepared",
        started_at="2026-04-02T20:00:00+08:00",
        finished_at=None,
        exit_code=None,
        detail=None,
    )

    payload = json.loads(artifacts.case_manifest_path.read_text(encoding="utf-8"))
    assert payload["case_name"] == "case_manifest"
    assert payload["status"] == "prepared"
    assert payload["shared_artifact_mount"] is True
    assert payload["case"]["dispatch_policy"] == "least_batch"
    assert payload["commands"]["frontend_launch_command"] == "frontend command"
    assert payload["commands"]["remote_launch_commands"]["1"] == "rank1 command"
    assert payload["paths"]["benchmark_dir"] == str(artifacts.benchmark_dir)


@pytest.mark.benchmark
def test_non_default_dispatch_policy_affects_case_grouping() -> None:
    runner = load_runner_module()
    default_case = runner.ExperimentCase(
        name="default_policy",
        cluster="cluster_a",
        strategy="strategy_a",
        dataset="dataset_a",
        model="model_a",
    )
    least_cache_case = runner.ExperimentCase(
        name="least_cache_policy",
        cluster="cluster_a",
        strategy="strategy_a",
        dataset="dataset_a",
        model="model_a",
        dispatch_policy="least_cache",
    )

    assert runner.case_group_key(default_case) == "MODEL_A/dataset_a/strategy_a"
    assert runner.case_group_key(least_cache_case) == (
        "MODEL_A/dataset_a/strategy_a/dispatch_least_cache"
    )
    assert (
        runner.case_artifact_scenario_prefix(default_case)
        == "strategy_a-mem85"
    )
    assert (
        runner.case_artifact_scenario_prefix(least_cache_case)
        == "strategy_a-dispatch_least_cache-mem85"
    )


@pytest.mark.benchmark
def test_augment_benchmark_outputs_adds_tpot_metrics(tmp_path: Path) -> None:
    runner = load_runner_module()
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir(parents=True)
    requests_path = benchmark_dir / "requests.jsonl"
    summary_path = benchmark_dir / "summary.json"

    requests_path.write_text(
        "\n".join([
            json.dumps({
                "request_id": "r0",
                "is_error": False,
                "actual_output_tokens": 5,
                "decode_time_ms": 40.0,
                "e2e_ms": 60.0,
                "tpot_with_initial_queue_ms": 999.0,
            }),
            json.dumps({
                "request_id": "r1",
                "is_error": False,
                "actual_output_tokens": 4,
                "decode_time_ms": 12.0,
                "e2e_ms": 18.0,
            }),
            json.dumps({
                "request_id": "r2",
                "is_error": False,
                "actual_output_tokens": 1,
                "decode_time_ms": 0.0,
                "e2e_ms": 50.0,
            }),
            json.dumps({
                "request_id": "r3",
                "is_error": True,
                "actual_output_tokens": 0,
                "e2e_ms": 100.0,
                "decode_time_ms": None,
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps({
            "total_requests": 4,
            "tpot_with_initial_queue_ms": {
                "mean": 123.0,
            },
        }) + "\n",
        encoding="utf-8",
    )

    runner.augment_benchmark_outputs(benchmark_dir)

    records = [
        json.loads(line)
        for line in requests_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert records[0]["tpot_without_queue_ms"] == pytest.approx(10.0)
    assert records[0]["tpot_by_e2e"] == pytest.approx(12.0)
    assert "tpot_with_initial_queue_ms" not in records[0]
    assert records[1]["tpot_without_queue_ms"] == pytest.approx(4.0)
    assert records[1]["tpot_by_e2e"] == pytest.approx(4.5)
    assert "tpot_with_initial_queue_ms" not in records[1]
    assert records[2]["tpot_without_queue_ms"] is None
    assert records[2]["tpot_by_e2e"] == pytest.approx(50.0)
    assert "tpot_with_initial_queue_ms" not in records[2]
    assert records[3]["tpot_without_queue_ms"] is None
    assert records[3]["tpot_by_e2e"] is None
    assert "tpot_with_initial_queue_ms" not in records[3]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["tpot_without_queue_ms"]["mean"] == pytest.approx(7.0)
    assert summary["tpot_without_queue_ms"]["p50"] == pytest.approx(7.0)
    assert summary["tpot_without_queue_ms"]["p90"] == pytest.approx(9.4)
    assert summary["tpot_without_queue_ms"]["p95"] == pytest.approx(9.7)
    assert summary["tpot_without_queue_ms"]["p99"] == pytest.approx(9.94)
    assert summary["tpot_by_e2e"]["mean"] == pytest.approx(22.1666666667)
    assert summary["tpot_by_e2e"]["p50"] == pytest.approx(12.0)
    assert summary["tpot_by_e2e"]["p90"] == pytest.approx(42.4)
    assert summary["tpot_by_e2e"]["p95"] == pytest.approx(46.2)
    assert summary["tpot_by_e2e"]["p99"] == pytest.approx(49.24)
    assert "tpot_with_initial_queue_ms" not in summary


@pytest.mark.benchmark
def test_latest_successful_case_dir_falls_back_to_complete_benchmark_when_manifest_not_ok(
        tmp_path: Path) -> None:
    runner = load_runner_module()
    case_group_dir = tmp_path / "case_group"
    older_ok = case_group_dir / "20260403-150000"
    newer_prepared = case_group_dir / "20260403-160000"

    write_case_benchmark_artifacts(
        older_ok,
        manifest_status="ok",
        summary=None,
        requests_present=False,
    )
    write_case_benchmark_artifacts(
        newer_prepared,
        manifest_status="prepared",
        summary={
            "total_requests": 2,
            "successful_requests": 2,
            "failed_requests": 0,
            "failure_ratio": 0.0,
        },
        requests=[
            {
                "request_id": "r0",
                "is_error": False,
                "actual_output_tokens": 5,
                "decode_time_ms": 40.0,
                "e2e_ms": 60.0,
            },
            {
                "request_id": "r1",
                "is_error": False,
                "actual_output_tokens": 4,
                "decode_time_ms": 12.0,
                "e2e_ms": 18.0,
            },
        ],
        run_meta={
            "benchmark_start_time": "2026-04-03T16:37:06+08:00",
        },
    )

    assert runner.latest_successful_case_dir(case_group_dir) == newer_prepared
    assert runner.latest_successful_case_tpot_by_e2e_mean(case_group_dir) == (
        newer_prepared,
        pytest.approx(8.25),
    )

    repaired_manifest = json.loads(
        (newer_prepared / "case_manifest.json").read_text(encoding="utf-8"))
    assert repaired_manifest["status"] == "ok"
    assert repaired_manifest["exit_code"] == 0
    assert repaired_manifest["detail"] == (
        "completed (recovered from successful benchmark artifacts)")
    assert repaired_manifest["started_at"] == "2026-04-03T16:37:06+08:00"
    assert repaired_manifest["finished_at"] is not None

    repaired_summary = json.loads(
        (newer_prepared / "benchmark" / "summary.json").read_text(
            encoding="utf-8"))
    assert repaired_summary["tpot_by_e2e"]["mean"] == pytest.approx(8.25)


@pytest.mark.benchmark
def test_latest_successful_case_dir_ignores_incomplete_benchmark_fallback(
        tmp_path: Path) -> None:
    runner = load_runner_module()
    case_group_dir = tmp_path / "case_group"
    older_ok = case_group_dir / "20260403-150000"
    newer_failed = case_group_dir / "20260403-160000"

    write_case_benchmark_artifacts(
        older_ok,
        manifest_status="ok",
        summary=None,
        requests_present=False,
    )
    write_case_benchmark_artifacts(
        newer_failed,
        manifest_status="failed",
        summary={
            "total_requests": 3,
            "successful_requests": 2,
            "failed_requests": 1,
            "failure_ratio": 1.0 / 3.0,
        },
    )

    assert runner.latest_successful_case_dir(case_group_dir) == older_ok


@pytest.mark.benchmark
def test_maybe_finalize_interrupted_case_manifest_recovers_success(
        tmp_path: Path) -> None:
    runner = load_runner_module()
    case_dir = tmp_path / "case"

    write_case_benchmark_artifacts(
        case_dir,
        manifest_status="prepared",
        summary={
            "total_requests": 2,
            "successful_requests": 2,
            "failed_requests": 0,
            "failure_ratio": 0.0,
        },
        requests=[
            {
                "request_id": "r0",
                "is_error": False,
                "actual_output_tokens": 5,
                "decode_time_ms": 40.0,
                "e2e_ms": 60.0,
            },
            {
                "request_id": "r1",
                "is_error": False,
                "actual_output_tokens": 4,
                "decode_time_ms": 12.0,
                "e2e_ms": 18.0,
            },
        ],
        run_meta={
            "benchmark_start_time": "2026-04-03T16:37:06+08:00",
        },
    )

    manifest = runner.load_json(case_dir / "case_manifest.json")
    runner.maybe_finalize_interrupted_case_manifest(
        case_dir,
        manifest=manifest,
        signal_name="SIGTERM",
        exit_code=143,
        finished_at="2026-04-03T16:48:00+08:00",
    )

    repaired_manifest = json.loads(
        (case_dir / "case_manifest.json").read_text(encoding="utf-8"))
    assert repaired_manifest["status"] == "ok"
    assert repaired_manifest["exit_code"] == 0
    assert repaired_manifest["detail"] == (
        "completed (recovered from successful benchmark artifacts)")
    assert repaired_manifest["started_at"] == "2026-04-03T16:37:06+08:00"
    assert repaired_manifest["finished_at"] is not None


@pytest.mark.benchmark
def test_maybe_finalize_interrupted_case_manifest_marks_interrupted(
        tmp_path: Path) -> None:
    runner = load_runner_module()
    case_dir = tmp_path / "case"

    write_case_benchmark_artifacts(
        case_dir,
        manifest_status="running",
        summary=None,
        requests_present=False,
    )

    manifest = runner.load_json(case_dir / "case_manifest.json")
    runner.maybe_finalize_interrupted_case_manifest(
        case_dir,
        manifest=manifest,
        signal_name="SIGHUP",
        exit_code=129,
        finished_at="2026-04-03T16:48:00+08:00",
    )

    interrupted_manifest = json.loads(
        (case_dir / "case_manifest.json").read_text(encoding="utf-8"))
    assert interrupted_manifest["status"] == "interrupted"
    assert interrupted_manifest["exit_code"] == 129
    assert interrupted_manifest["detail"] == "interrupted by SIGHUP"
    assert interrupted_manifest["started_at"] is not None
    assert interrupted_manifest["finished_at"] == "2026-04-03T16:48:00+08:00"


@pytest.mark.benchmark
def test_wait_for_frontend_allows_clean_headless_shutdown(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(tmp_path,
                                            dry_run=False,
                                            keep_going=False)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    frontend = make_node_runtime(
        runner,
        tmp_path,
        node_rank=0,
        host="local",
        poll_results=[None, None, 0],
    )
    headless = make_node_runtime(
        runner,
        tmp_path,
        node_rank=1,
        host="node-a",
        poll_results=[None, 0],
    )
    runtime = SimpleNamespace(
        frontend=frontend,
        headless_nodes=[headless],
        resolved=SimpleNamespace(
            case=SimpleNamespace(max_bench_duration_sec=60.0)),
    )

    assert launcher.wait_for_frontend(runtime) == 0


@pytest.mark.benchmark
def test_detect_group_trend_outlier_candidates_flags_local_spike(
        tmp_path: Path) -> None:
    runner = load_runner_module()

    cases = [
        runner.ExperimentCase(
            name="group_a_rate10",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_a",
            model="model_a",
            request_rate=10.0,
        ),
        runner.ExperimentCase(
            name="group_a_rate20",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_a",
            model="model_a",
            request_rate=20.0,
        ),
        runner.ExperimentCase(
            name="group_a_rate30",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_a",
            model="model_a",
            request_rate=30.0,
        ),
    ]

    latest_results_by_case_name = {}
    for case, metric in zip(cases, (40.0, 90.0, 50.0), strict=True):
        benchmark_dir = tmp_path / case.name / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        summary_json = benchmark_dir / "summary.json"
        summary_json.write_text(
            json.dumps({"tpot_by_e2e": {
                "mean": metric,
            }}) + "\n",
            encoding="utf-8",
        )
        latest_results_by_case_name[case.name] = runner.CaseResult(
            case_name=case.name,
            status="ok",
            exit_code=0,
            started_at="2026-04-03T00:00:00+08:00",
            finished_at="2026-04-03T00:10:00+08:00",
            case_dir=benchmark_dir.parent,
            benchmark_dir=benchmark_dir,
            summary_json=summary_json,
            detail="completed",
        )

    candidates = runner.detect_group_trend_outlier_candidates(
        cases,
        latest_results_by_case_name,
        relative_threshold_pct=5.0,
        absolute_threshold_ms=5.0,
    )

    assert [candidate.case.name for candidate in candidates] == ["group_a_rate20"]
    assert candidates[0].expected_value == pytest.approx(45.0)
    assert candidates[0].delta_ms == pytest.approx(45.0)


@pytest.mark.benchmark
def test_find_fatal_signal_in_log_detects_worker_exit_pair(
        tmp_path: Path) -> None:
    runner = load_runner_module()
    log_path = tmp_path / "rank1.log"
    log_path.write_text(
        "\n".join([
            "wrapper still running",
            "Worker proc 1 died unexpectedly, shutting down executor.",
            "tearing down",
        ]) + "\n",
        encoding="utf-8",
    )

    excerpt = runner.find_fatal_signal_in_log(
        log_path, lines=runner.FATAL_LOG_SCAN_LINES)

    assert excerpt is not None
    assert "Worker proc 1 died unexpectedly, shutting down executor." in excerpt


@pytest.mark.benchmark
def test_wait_for_headless_startup_fails_on_fatal_rank_log(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(tmp_path,
                                            dry_run=False,
                                            keep_going=False)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    headless = make_node_runtime(
        runner,
        tmp_path,
        node_rank=1,
        host="node-a",
        poll_results=[None],
        runtime_log="EngineCore encountered a fatal error.\n",
    )
    runtime = SimpleNamespace(
        headless_nodes=[headless],
        resolved=SimpleNamespace(
            case=SimpleNamespace(start_grace_sec=60.0)),
    )

    with pytest.raises(RuntimeError,
                       match="detected fatal engine failure") as exc_info:
        launcher.wait_for_headless_startup(runtime)

    assert str(headless.runtime_log_path) in str(exc_info.value)
    assert "EngineCore encountered a fatal error." in str(exc_info.value)


@pytest.mark.benchmark
def test_wait_for_headless_shutdown_checks_remote_pidfiles(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(tmp_path,
                                            dry_run=False,
                                            keep_going=False)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    headless = make_node_runtime(
        runner,
        tmp_path,
        node_rank=1,
        host="node-a",
        poll_results=[0, 0],
    )
    pid_states = iter([True, False])
    monkeypatch.setattr(
        launcher,
        "remote_pidfile_is_running",
        lambda *_args, **_kwargs: next(pid_states),
    )
    runtime = SimpleNamespace(
        headless_nodes=[headless],
        resolved=SimpleNamespace(
            case=SimpleNamespace(remote_shutdown_grace_sec=30.0),
            cluster=SimpleNamespace(),
        ),
        artifacts=SimpleNamespace(rank_pid_paths={1: tmp_path / "rank1.pid"}),
    )

    launcher.wait_for_headless_shutdown(runtime)


@pytest.mark.benchmark
def test_wait_for_frontend_still_fails_for_nonzero_headless_exit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(tmp_path,
                                            dry_run=False,
                                            keep_going=False)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    frontend = make_node_runtime(
        runner,
        tmp_path,
        node_rank=0,
        host="local",
        poll_results=[None],
    )
    headless = make_node_runtime(
        runner,
        tmp_path,
        node_rank=1,
        host="node-a",
        poll_results=[7],
        runtime_log="remote rank failed\n",
    )
    runtime = SimpleNamespace(
        frontend=frontend,
        headless_nodes=[headless],
        resolved=SimpleNamespace(
            case=SimpleNamespace(max_bench_duration_sec=60.0)),
    )

    with pytest.raises(RuntimeError, match="rank 1 on node-a exited early"):
        launcher.wait_for_frontend(runtime)


@pytest.mark.benchmark
def test_wait_for_frontend_fails_on_fatal_frontend_log(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(tmp_path,
                                            dry_run=False,
                                            keep_going=False)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    frontend = make_node_runtime(
        runner,
        tmp_path,
        node_rank=0,
        host="local",
        poll_results=[None],
        runtime_log="AsyncLLM output_handler failed.\n",
    )
    runtime = SimpleNamespace(
        frontend=frontend,
        headless_nodes=[],
        resolved=SimpleNamespace(
            case=SimpleNamespace(max_bench_duration_sec=60.0)),
    )

    with pytest.raises(RuntimeError,
                       match="detected fatal engine failure") as exc_info:
        launcher.wait_for_frontend(runtime)

    assert str(frontend.runtime_log_path) in str(exc_info.value)
    assert "AsyncLLM output_handler failed." in str(exc_info.value)


@pytest.mark.benchmark
def test_wait_for_frontend_times_out_after_case_limit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(tmp_path,
                                            dry_run=False,
                                            keep_going=False)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monotonic_values = iter([100.0, 103.0])
    monkeypatch.setattr(runner.time, "monotonic",
                        lambda: next(monotonic_values))

    frontend = make_node_runtime(
        runner,
        tmp_path,
        node_rank=0,
        host="local",
        poll_results=[None],
    )
    headless = make_node_runtime(
        runner,
        tmp_path,
        node_rank=1,
        host="node-a",
        poll_results=[None],
    )
    runtime = SimpleNamespace(
        frontend=frontend,
        headless_nodes=[headless],
        resolved=SimpleNamespace(
            case=SimpleNamespace(max_bench_duration_sec=2.0)),
    )

    with pytest.raises(runner.BenchTimeoutError,
                       match="max_bench_duration_sec=2s"):
        launcher.wait_for_frontend(runtime)


@pytest.mark.benchmark
def test_run_preclean_retries_until_local_ports_clear(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(tmp_path,
                                            dry_run=False,
                                            keep_going=True)
    frontend_cleanup = tmp_path / "frontend.cleanup.command.sh"
    frontend_cleanup.write_text("frontend cleanup\n", encoding="utf-8")
    rank1_cleanup = tmp_path / "rank1.cleanup.command.sh"
    rank1_cleanup.write_text("rank1 cleanup\n", encoding="utf-8")

    calls: list[tuple[str, str]] = []
    waits: list[tuple[int, ...]] = []
    busy_states = iter([(29550, ), ()])

    monkeypatch.setattr(
        launcher,
        "run_local_shell",
        lambda _cluster, command: calls.append(("local", command)),
    )
    monkeypatch.setattr(
        launcher,
        "run_remote_shell",
        lambda _cluster, host, command: calls.append((host, command)),
    )

    def fake_wait(ports, *, timeout_sec, poll_interval_sec):
        waits.append(ports)
        assert timeout_sec == runner.PRESTART_CLEANUP_WAIT_SEC
        assert poll_interval_sec == runner.PRESTART_CLEANUP_POLL_INTERVAL_SEC
        return next(busy_states)

    monkeypatch.setattr(runner, "wait_for_local_ports_to_clear", fake_wait)

    resolved = SimpleNamespace(
        cluster=SimpleNamespace(remote_hosts=("node-a", ), master_port=29579),
        case=SimpleNamespace(data_parallel_rpc_port=29550),
    )
    artifacts = SimpleNamespace(
        frontend_cleanup_command_path=frontend_cleanup,
        rank_cleanup_command_paths={1: rank1_cleanup},
    )

    launcher.run_preclean(resolved, artifacts)

    assert calls == [
        ("local", "frontend cleanup\n"),
        ("node-a", "rank1 cleanup\n"),
        ("local", "frontend cleanup\n"),
        ("node-a", "rank1 cleanup\n"),
    ]
    assert waits == [(29550, 29579), (29550, 29579)]


@pytest.mark.benchmark
def test_run_preclean_fails_if_local_ports_remain_busy(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(tmp_path,
                                            dry_run=False,
                                            keep_going=True)
    frontend_cleanup = tmp_path / "frontend.cleanup.command.sh"
    frontend_cleanup.write_text("frontend cleanup\n", encoding="utf-8")

    calls: list[str] = []
    monkeypatch.setattr(
        launcher,
        "run_local_shell",
        lambda _cluster, command: calls.append(command),
    )
    monkeypatch.setattr(launcher, "run_remote_shell", lambda *_args: None)
    monkeypatch.setattr(runner, "PRESTART_CLEANUP_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(
        runner,
        "wait_for_local_ports_to_clear",
        lambda *_args, **_kwargs: (29550, ),
    )

    resolved = SimpleNamespace(
        cluster=SimpleNamespace(remote_hosts=(), master_port=29579),
        case=SimpleNamespace(data_parallel_rpc_port=29550),
    )
    artifacts = SimpleNamespace(
        frontend_cleanup_command_path=frontend_cleanup,
        rank_cleanup_command_paths={},
    )

    with pytest.raises(RuntimeError,
                       match="preclean could not free local ports"):
        launcher.run_preclean(resolved, artifacts)

    assert calls == ["frontend cleanup\n", "frontend cleanup\n"]


@pytest.mark.benchmark
def test_wait_for_local_shutdown_times_out_with_local_diagnostics(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(tmp_path,
                                            dry_run=False,
                                            keep_going=True)
    frontend_pid_path = tmp_path / "frontend.pid"
    frontend_pgid_path = tmp_path / "frontend.pgid"
    frontend_pid_path.write_text("123\n", encoding="utf-8")
    frontend_pgid_path.write_text("456\n", encoding="utf-8")

    monotonic_values = iter([100.0, 131.0])
    monkeypatch.setattr(runner.time, "monotonic",
                        lambda: next(monotonic_values))
    monkeypatch.setattr(runner.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(runner, "local_pid_is_running", lambda _pid: True)
    monkeypatch.setattr(runner, "local_process_group_is_running",
                        lambda _pgid: True)
    monkeypatch.setattr(runner, "can_bind_local_tcp_port", lambda _port: False)
    monkeypatch.setattr(runner, "describe_local_port_diagnostics",
                        lambda port: f"{port} [busy]")

    runtime = SimpleNamespace(
        artifacts=SimpleNamespace(
            frontend_pid_path=frontend_pid_path,
            frontend_pgid_path=frontend_pgid_path,
        ),
        resolved=SimpleNamespace(
            case=SimpleNamespace(
                data_parallel_rpc_port=29550,
                local_shutdown_grace_sec=30.0,
            ),
            cluster=SimpleNamespace(master_port=29579),
        ),
    )

    with pytest.raises(RuntimeError,
                       match="node 0 local shutdown did not complete") as exc_info:
        launcher.wait_for_local_shutdown(runtime)

    message = str(exc_info.value)
    assert "frontend pidfile still points to a live pid" in message
    assert "frontend process group still has live processes" in message
    assert "local ports still busy" in message


@pytest.mark.benchmark
def test_verify_case_cleanup_checks_remote_pidfiles(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(tmp_path,
                                            dry_run=False,
                                            keep_going=True)
    monotonic_values = iter([10.0, 41.0])
    monkeypatch.setattr(runner.time, "monotonic",
                        lambda: next(monotonic_values))
    monkeypatch.setattr(runner.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(launcher, "local_shutdown_failures", lambda _runtime: [])
    monkeypatch.setattr(launcher, "remote_pidfile_is_running",
                        lambda *_args, **_kwargs: True)

    runtime = SimpleNamespace(
        headless_nodes=[
            SimpleNamespace(node_rank=1, host="node-a"),
        ],
        artifacts=SimpleNamespace(rank_pid_paths={1: tmp_path / "rank1.pid"}),
        resolved=SimpleNamespace(
            case=SimpleNamespace(local_shutdown_grace_sec=30.0),
            cluster=SimpleNamespace(),
        ),
    )

    with pytest.raises(RuntimeError,
                       match="post-cleanup verification failed") as exc_info:
        launcher.verify_case_cleanup(runtime)

    assert "rank 1 on node-a pidfile still points to a live pid" in str(
        exc_info.value)


@pytest.mark.benchmark
def test_run_case_fails_if_cleanup_verification_fails(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(tmp_path,
                                            dry_run=False,
                                            keep_going=True)
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("prompt_len,output_len\n4,7\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    resolved = runner.resolve_case(
        runner.ExperimentCase(
            name="case_a",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_alias",
            model="model_alias",
            request_rate=10.0,
        ),
        clusters={
            "cluster_a":
            runner.ClusterSpec(
                master_addr="10.0.0.1",
                master_port=29579,
                remote_hosts=(),
            ),
        },
        strategies={
            "strategy_a":
            runner.StrategySpec(
                data_parallel_size=1,
                data_parallel_size_local=1,
                tensor_parallel_size=1,
            ),
        },
        datasets={"dataset_alias": str(dataset_path)},
        models={"model_alias": str(model_dir)},
    )

    monkeypatch.setattr(runner, "resolve_case", lambda _case: resolved)
    monkeypatch.setattr(launcher, "run_preclean", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "launch_headless_nodes",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "wait_for_headless_startup",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "launch_frontend",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "wait_for_frontend",
                        lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(launcher, "wait_for_headless_shutdown",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "wait_for_local_shutdown",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "cleanup_case_runtime",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "verify_case_cleanup",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            RuntimeError("cleanup dirty")))
    monkeypatch.setattr(runner, "augment_benchmark_outputs",
                        lambda benchmark_dir: runner.write_json(
                            benchmark_dir / "summary.json",
                            {
                                "total_requests": 1,
                                "successful_requests": 1,
                                "failed_requests": 0,
                                "failure_ratio": 0.0,
                            },
                        ))

    result = launcher.run_case(
        runner.ExperimentCase(
            name="case_a",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_alias",
            model="model_alias",
            request_rate=10.0,
        ),
        tmp_path / "run",
    )

    assert result.status == "failed"
    assert result.exit_code == 1
    assert result.detail == "cleanup dirty"


@pytest.mark.benchmark
def test_run_case_fails_when_summary_indicates_failed_requests(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(tmp_path,
                                            dry_run=False,
                                            keep_going=True)
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("prompt_len,output_len\n4,7\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    resolved = runner.resolve_case(
        runner.ExperimentCase(
            name="case_a",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_alias",
            model="model_alias",
            request_rate=10.0,
        ),
        clusters={
            "cluster_a":
            runner.ClusterSpec(
                master_addr="10.0.0.1",
                master_port=29579,
                remote_hosts=(),
            ),
        },
        strategies={
            "strategy_a":
            runner.StrategySpec(
                data_parallel_size=1,
                data_parallel_size_local=1,
                tensor_parallel_size=1,
            ),
        },
        datasets={"dataset_alias": str(dataset_path)},
        models={"model_alias": str(model_dir)},
    )

    monkeypatch.setattr(runner, "resolve_case", lambda _case: resolved)
    monkeypatch.setattr(launcher, "run_preclean", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "launch_headless_nodes",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "wait_for_headless_startup",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "launch_frontend",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "wait_for_frontend",
                        lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(launcher, "wait_for_headless_shutdown",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "wait_for_local_shutdown",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "cleanup_case_runtime",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "verify_case_cleanup",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "augment_benchmark_outputs",
        lambda benchmark_dir: runner.write_json(
            benchmark_dir / "summary.json",
            {
                "total_requests": 21000,
                "successful_requests": 7492,
                "failed_requests": 13508,
                "failure_ratio": 0.643238,
            },
        ),
    )

    result = launcher.run_case(
        runner.ExperimentCase(
            name="case_a",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_alias",
            model="model_alias",
            request_rate=10.0,
        ),
        tmp_path / "run",
    )

    assert result.status == "failed"
    assert result.exit_code == 1
    assert "benchmark summary indicates failure" in result.detail
    assert "total_requests=21000" in result.detail
    assert "successful_requests=7492" in result.detail
    assert "failed_requests=13508" in result.detail
    assert "failure_ratio=0.643238" in result.detail


@pytest.mark.benchmark
def test_run_continues_after_failed_case_when_keep_going_enabled(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(tmp_path,
                                            dry_run=False,
                                            keep_going=True)
    monkeypatch.setattr(runner, "build_historical_skip_state",
                        lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(runner, "current_run_tag",
                        lambda: "20260403-000000")
    monkeypatch.setattr(runner, "current_iso_timestamp",
                        lambda: "2026-04-03T00:00:00+08:00")
    monkeypatch.setattr(launcher, "install_signal_handlers", lambda: None)
    monkeypatch.setattr(launcher, "restore_signal_handlers", lambda: None)

    executed_cases: list[str] = []

    def fake_run_case(case, _run_dir):
        executed_cases.append(case.name)
        case_dir = tmp_path / case.name
        benchmark_dir = case_dir / "benchmark"
        summary_json = benchmark_dir / "summary.json"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        status = "failed" if case.name == "group_a_rate10" else "ok"
        if status == "ok":
            summary_json.write_text(
                json.dumps({
                    "tpot_by_e2e": {
                        "mean": 10.0,
                    },
                }) + "\n",
                encoding="utf-8",
            )

        return runner.CaseResult(
            case_name=case.name,
            status=status,
            exit_code=1 if status == "failed" else 0,
            started_at="2026-04-03T00:00:00+08:00",
            finished_at="2026-04-03T00:10:00+08:00",
            case_dir=case_dir,
            benchmark_dir=benchmark_dir,
            summary_json=summary_json,
            detail=status,
        )

    monkeypatch.setattr(launcher, "run_case", fake_run_case)

    cases = [
        runner.ExperimentCase(
            name="group_a_rate10",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_a",
            model="model_a",
            request_rate=10.0,
        ),
        runner.ExperimentCase(
            name="group_a_rate20",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_a",
            model="model_a",
            request_rate=20.0,
        ),
        runner.ExperimentCase(
            name="group_b_rate10",
            cluster="cluster_a",
            strategy="strategy_b",
            dataset="dataset_a",
            model="model_a",
            request_rate=10.0,
        ),
    ]

    results = launcher.run(cases,
                           run_label="failed_sweep",
                           rate_plan="coarse10")

    assert executed_cases == ["group_a_rate10", "group_b_rate10"]
    assert [result.case_name for result in results] == [
        "group_a_rate10",
        "group_b_rate10",
    ]

    run_manifest_path = (tmp_path / "_runs" / "20260403-000000__failed_sweep" /
                         "run_manifest.json")
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    blocked = manifest["runtime_blocked_groups"]["MODEL_A/dataset_a/strategy_a"]
    assert blocked["blocked_from_rate"] == 10.0
    assert blocked["reason"] == "case status=failed"


@pytest.mark.benchmark
def test_run_auto_reruns_trend_outlier_case(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(
        tmp_path,
        dry_run=False,
        keep_going=True,
        auto_rerun_trend_outliers=True,
        trend_rerun_max_attempts=1,
        trend_rerun_relative_threshold_pct=5.0,
        trend_rerun_absolute_threshold_ms=5.0,
    )
    monkeypatch.setattr(runner, "build_historical_skip_state",
                        lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(runner, "current_run_tag",
                        lambda: "20260403-000000")
    monkeypatch.setattr(runner, "current_iso_timestamp",
                        lambda: "2026-04-03T00:00:00+08:00")
    monkeypatch.setattr(launcher, "install_signal_handlers", lambda: None)
    monkeypatch.setattr(launcher, "restore_signal_handlers", lambda: None)

    executed_cases: list[tuple[str, int]] = []
    attempts: dict[str, int] = {}
    metric_by_attempt = {
        ("group_a_rate10", 1): 40.0,
        ("group_a_rate20", 1): 90.0,
        ("group_a_rate20", 2): 45.0,
        ("group_a_rate30", 1): 50.0,
    }

    def fake_run_case(case, _run_dir):
        attempt = attempts.get(case.name, 0) + 1
        attempts[case.name] = attempt
        executed_cases.append((case.name, attempt))

        benchmark_dir = tmp_path / f"{case.name}_attempt{attempt}" / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        summary_json = benchmark_dir / "summary.json"
        summary_json.write_text(
            json.dumps({"tpot_by_e2e": {
                "mean": metric_by_attempt[(case.name, attempt)],
            }}) + "\n",
            encoding="utf-8",
        )

        return runner.CaseResult(
            case_name=case.name,
            status="ok",
            exit_code=0,
            started_at="2026-04-03T00:00:00+08:00",
            finished_at="2026-04-03T00:10:00+08:00",
            case_dir=benchmark_dir.parent,
            benchmark_dir=benchmark_dir,
            summary_json=summary_json,
            detail="completed",
        )

    monkeypatch.setattr(launcher, "run_case", fake_run_case)

    cases = [
        runner.ExperimentCase(
            name="group_a_rate10",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_a",
            model="model_a",
            request_rate=10.0,
        ),
        runner.ExperimentCase(
            name="group_a_rate20",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_a",
            model="model_a",
            request_rate=20.0,
        ),
        runner.ExperimentCase(
            name="group_a_rate30",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_a",
            model="model_a",
            request_rate=30.0,
        ),
    ]

    results = launcher.run(cases, run_label="trend_rerun", rate_plan="coarse10")

    assert executed_cases == [
        ("group_a_rate10", 1),
        ("group_a_rate20", 1),
        ("group_a_rate30", 1),
        ("group_a_rate20", 2),
    ]
    assert [result.case_name for result in results] == [
        "group_a_rate10",
        "group_a_rate20",
        "group_a_rate30",
        "group_a_rate20",
    ]
    rerun_metric = runner.load_case_result_summary_metric(results[-1])
    assert rerun_metric == pytest.approx(45.0)


@pytest.mark.benchmark
def test_run_continues_after_timeout_and_skips_higher_rates_in_group(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(tmp_path,
                                            dry_run=False,
                                            keep_going=False)
    monkeypatch.setattr(runner, "build_historical_skip_state",
                        lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(runner, "current_run_tag",
                        lambda: "20260403-000000")
    monkeypatch.setattr(runner, "current_iso_timestamp",
                        lambda: "2026-04-03T00:00:00+08:00")
    monkeypatch.setattr(launcher, "install_signal_handlers", lambda: None)
    monkeypatch.setattr(launcher, "restore_signal_handlers", lambda: None)

    executed_cases: list[str] = []

    def fake_run_case(case, _run_dir):
        executed_cases.append(case.name)
        case_dir = tmp_path / case.name
        benchmark_dir = case_dir / "benchmark"
        summary_json = benchmark_dir / "summary.json"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        status = "timed_out" if case.name == "group_a_rate10" else "ok"
        if status == "ok":
            summary_json.write_text(
                json.dumps({
                    "tpot_by_e2e": {
                        "mean": 10.0,
                    },
                }) + "\n",
                encoding="utf-8",
            )

        return runner.CaseResult(
            case_name=case.name,
            status=status,
            exit_code=124 if status == "timed_out" else 0,
            started_at="2026-04-03T00:00:00+08:00",
            finished_at="2026-04-03T00:10:00+08:00",
            case_dir=case_dir,
            benchmark_dir=benchmark_dir,
            summary_json=summary_json,
            detail=status,
        )

    monkeypatch.setattr(launcher, "run_case", fake_run_case)

    cases = [
        runner.ExperimentCase(
            name="group_a_rate10",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_a",
            model="model_a",
            request_rate=10.0,
        ),
        runner.ExperimentCase(
            name="group_a_rate20",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_a",
            model="model_a",
            request_rate=20.0,
        ),
        runner.ExperimentCase(
            name="group_b_rate10",
            cluster="cluster_a",
            strategy="strategy_b",
            dataset="dataset_a",
            model="model_a",
            request_rate=10.0,
        ),
    ]

    results = launcher.run(cases,
                           run_label="timeout_sweep",
                           rate_plan="coarse10")

    assert executed_cases == ["group_a_rate10", "group_b_rate10"]
    assert [result.case_name for result in results] == [
        "group_a_rate10",
        "group_b_rate10",
    ]

    run_manifest_path = (tmp_path / "_runs" / "20260403-000000__timeout_sweep" /
                         "run_manifest.json")
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    blocked = manifest["runtime_blocked_groups"]["MODEL_A/dataset_a/strategy_a"]
    assert blocked["blocked_from_rate"] == 10.0
    assert blocked["reason"] == "case status=timed_out"


@pytest.mark.benchmark
def test_run_writes_partial_manifest_when_aborted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    launcher = runner.ManualMultinodeRunner(tmp_path,
                                            dry_run=False,
                                            keep_going=True)
    monkeypatch.setattr(runner, "build_historical_skip_state",
                        lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(runner, "current_run_tag",
                        lambda: "20260403-000000")
    timestamps = iter([
        "2026-04-03T00:00:00+08:00",
        "2026-04-03T00:20:00+08:00",
    ])
    monkeypatch.setattr(runner, "current_iso_timestamp",
                        lambda: next(timestamps))
    monkeypatch.setattr(launcher, "install_signal_handlers", lambda: None)
    monkeypatch.setattr(launcher, "restore_signal_handlers", lambda: None)

    def fake_run_case(case, _run_dir):
        case_dir = tmp_path / case.name
        benchmark_dir = case_dir / "benchmark"
        summary_json = benchmark_dir / "summary.json"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        if case.name == "group_a_rate10":
            summary_json.write_text(
                json.dumps({
                    "tpot_by_e2e": {
                        "mean": 10.0,
                    },
                }) + "\n",
                encoding="utf-8",
            )
            return runner.CaseResult(
                case_name=case.name,
                status="ok",
                exit_code=0,
                started_at="2026-04-03T00:00:00+08:00",
                finished_at="2026-04-03T00:10:00+08:00",
                case_dir=case_dir,
                benchmark_dir=benchmark_dir,
                summary_json=summary_json,
                detail="ok",
            )

        raise SystemExit(143)

    monkeypatch.setattr(launcher, "run_case", fake_run_case)

    cases = [
        runner.ExperimentCase(
            name="group_a_rate10",
            cluster="cluster_a",
            strategy="strategy_a",
            dataset="dataset_a",
            model="model_a",
            request_rate=10.0,
        ),
        runner.ExperimentCase(
            name="group_b_rate10",
            cluster="cluster_a",
            strategy="strategy_b",
            dataset="dataset_a",
            model="model_a",
            request_rate=10.0,
        ),
    ]

    with pytest.raises(SystemExit) as exc_info:
        launcher.run(cases, run_label="aborted_sweep", rate_plan="coarse10")

    assert exc_info.value.code == 143

    run_manifest_path = (tmp_path / "_runs" / "20260403-000000__aborted_sweep" /
                         "run_manifest.json")
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["finished_at"] == "2026-04-03T00:20:00+08:00"
    assert [result["case_name"] for result in manifest["results"]] == [
        "group_a_rate10",
    ]
    assert manifest["aborted"]["type"] == "SystemExit"
    assert manifest["aborted"]["exit_code"] == 143
    assert manifest["aborted"]["detail"] == "143"


@pytest.mark.benchmark
def test_load_cases_from_csv_skips_disabled_rows_and_preserves_order(
        tmp_path: Path) -> None:
    runner = load_runner_module()
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text(
        "\n".join([
            ",".join(runner.CASE_CSV_FIELDNAMES),
            ("1,case_b,cluster_b,model_b,dataset_b,strategy_b,least_batch,"
             "20,manual,64,0.9,200,8,4096,29551,planned,ref_b"),
            ("0,case_skip,cluster_b,model_b,dataset_b,strategy_b,least_cache,"
             "25,manual,64,0.9,250,8,4096,29551,planned,ref_skip"),
            ("1,case_a,cluster_a,model_a,dataset_a,strategy_a,,10,,32,0.85,"
             "100,0,,29550,planned,ref_a"),
        ]) + "\n",
        encoding="utf-8",
    )

    cases = runner.load_cases_from_csv(csv_path)

    assert [case.name for case in cases] == ["case_b", "case_a"]
    assert cases[0].dispatch_policy == "least_batch"
    assert cases[0].request_rate == pytest.approx(20.0)
    assert cases[0].rate_phase == "manual"
    assert cases[0].max_num_seqs == 64
    assert cases[0].gpu_memory_utilization == pytest.approx(0.9)
    assert cases[0].max_requests == 200
    assert cases[0].warmup_requests == 8
    assert cases[0].max_model_len == 4096
    assert cases[0].data_parallel_rpc_port == 29551
    assert cases[1].rate_phase == runner.DEFAULT_RATE_PLAN
    assert cases[1].dispatch_policy == runner.DEFAULT_DISPATCH_POLICY
    assert cases[1].warmup_requests == 0
    assert cases[1].max_model_len is None


@pytest.mark.benchmark
def test_select_cases_rejects_case_csv_with_all(tmp_path: Path) -> None:
    runner = load_runner_module()
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text(
        "name,cluster,model,dataset,strategy,request_rate\n"
        "case_a,cluster_a,model_a,dataset_a,strategy_a,10\n",
        encoding="utf-8",
    )
    args = runner.build_parser().parse_args(
        ["--case-csv", str(csv_path), "--all"])

    with pytest.raises(SystemExit,
                       match="Use either --case-csv or --all/--case, not both."):
        runner.select_cases(args, [])


@pytest.mark.benchmark
def test_select_cases_rejects_case_csv_with_filter_flags(tmp_path: Path) -> None:
    runner = load_runner_module()
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text(
        "name,cluster,model,dataset,strategy,request_rate\n"
        "case_a,cluster_a,model_a,dataset_a,strategy_a,10\n",
        encoding="utf-8",
    )
    args = runner.build_parser().parse_args(
        ["--case-csv", str(csv_path), "--model", "deepseek_v3_1024k"])

    with pytest.raises(
            SystemExit,
            match=("Use either --case-csv or --model/--dataset/--strategy "
                   "filters, not both.")):
        runner.select_cases(args, [])


@pytest.mark.benchmark
def test_build_historical_skip_state_blocks_higher_rates_after_historical_timeout(
        tmp_path: Path) -> None:
    runner = load_runner_module()
    artifact_root = tmp_path / "artifacts"
    shared = dict(
        cluster="cluster_a",
        strategy="strategy_a",
        dataset="dataset_a",
        model="model_a",
        max_num_seqs=64,
        gpu_memory_utilization=0.85,
        warmup_requests=0,
    )
    cases = [
        runner.ExperimentCase(
            name="group_rate30",
            request_rate=30.0,
            max_requests=300,
            **shared,
        ),
        runner.ExperimentCase(
            name="group_rate40",
            request_rate=40.0,
            max_requests=400,
            **shared,
        ),
        runner.ExperimentCase(
            name="group_rate50",
            request_rate=50.0,
            max_requests=500,
            **shared,
        ),
    ]
    timed_out_dir = (runner.case_artifact_group_dir(artifact_root, cases[1]) /
                     "20260403-160000")
    timed_out_dir.mkdir(parents=True, exist_ok=True)
    (timed_out_dir / "case_manifest.json").write_text(
        json.dumps({"status": "timed_out"}) + "\n",
        encoding="utf-8",
    )

    exact_case_skips, blocked_group_rates = runner.build_historical_skip_state(
        artifact_root,
        cases,
        ignore_bs=False,
    )

    group_key = runner.case_group_key(cases[1])
    assert exact_case_skips == {}
    assert blocked_group_rates[group_key][0] == pytest.approx(40.0)
    assert "status=timed_out" in blocked_group_rates[group_key][1]
    assert runner.block_reason_for_rate(blocked_group_rates, cases[2]) is not None
    assert runner.block_reason_for_rate(blocked_group_rates, cases[1]) is not None
    assert runner.block_reason_for_rate(blocked_group_rates, cases[0]) is None


@pytest.mark.benchmark
def test_build_historical_skip_state_matches_history_with_different_mem(
        tmp_path: Path) -> None:
    runner = load_runner_module()
    artifact_root = tmp_path / "artifacts"
    historical_case = runner.ExperimentCase(
        name="group_rate40_old_mem",
        cluster="cluster_a",
        strategy="strategy_a",
        dataset="dataset_a",
        model="model_a",
        request_rate=40.0,
        max_requests=400,
        max_num_seqs=64,
        gpu_memory_utilization=0.9,
        warmup_requests=0,
    )
    current_case = runner.ExperimentCase(
        name="group_rate40_new_mem",
        cluster="cluster_a",
        strategy="strategy_a",
        dataset="dataset_a",
        model="model_a",
        request_rate=40.0,
        max_requests=400,
        max_num_seqs=64,
        gpu_memory_utilization=0.87,
        warmup_requests=0,
    )
    history_dir = (runner.case_artifact_group_dir(artifact_root,
                                                  historical_case) /
                   "20260403-160000")
    write_case_benchmark_artifacts(
        history_dir,
        manifest_status="ok",
        summary={
            "tpot_by_e2e": {
                "mean": 10.0,
            },
        },
        requests_present=False,
    )

    exact_case_skips, blocked_group_rates = runner.build_historical_skip_state(
        artifact_root,
        [current_case],
        ignore_bs=False,
    )

    assert blocked_group_rates == {}
    assert current_case.name in exact_case_skips
    assert "matched with different mem" in exact_case_skips[current_case.name]
    assert "mem90" in exact_case_skips[current_case.name]
