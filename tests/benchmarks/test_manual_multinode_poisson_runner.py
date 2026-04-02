# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import json
import sys
from pathlib import Path

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
    artifacts = runner.prepare_artifact_paths(tmp_path / "run", resolved.case.name,
                                              resolved.cluster.remote_hosts)

    frontend_command = runner.build_local_launch_command(
        resolved,
        artifacts,
        runner.build_frontend_argv(resolved, artifacts.benchmark_dir),
    )
    cleanup_command = runner.build_node_cleanup_command(
        resolved=resolved,
        pid_path=artifacts.frontend_pid_path,
        include_frontend_pattern=True,
    )

    assert "set -o pipefail" in frontend_command
    assert str(artifacts.frontend_log_path) in frontend_command
    assert str(artifacts.frontend_pid_path) in frontend_command
    assert "offline_poisson_harness.py frontend" in cleanup_command
    assert "--data-parallel-rpc-port 29550" in cleanup_command
    assert "--master-port 29579" in cleanup_command
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
    artifacts = runner.prepare_artifact_paths(tmp_path / "run", resolved.case.name,
                                              resolved.cluster.remote_hosts)
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
    assert payload["commands"]["frontend_launch_command"] == "frontend command"
    assert payload["commands"]["remote_launch_commands"]["1"] == "rank1 command"
    assert payload["paths"]["benchmark_dir"] == str(artifacts.benchmark_dir)


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
