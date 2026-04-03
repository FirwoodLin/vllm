#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual multi-node launcher for offline_poisson_harness experiments.

Edit the config block in this file directly to add or tweak experiment cases.
The CLI intentionally stays thin: list cases, run one case, or run them all.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, TextIO


HARNESS_ENTRYPOINT = "/vllm/offline_poisson_harness.py"
DEFAULT_WORKDIR = "/vllm"
DEFAULT_ARTIFACT_ROOT = Path(
    "/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench/manual_multinode"
)
DEFAULT_CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
DEFAULT_SSH_OPTS = (
    "-F",
    "/root/.ssh/config",
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "UpdateHostKeys=no",
)
FORWARDED_ENV_KEYS = (
    "PATH",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_TC",
    "NCCL_SOCKET_IFNAME",
    "NVSHMEM_HCA_LIST",
    "NVSHMEM_IB_GID_INDEX",
    "NVSHMEM_IBGDA_NUM_RC_PER_PE",
    "NVSHMEM_IB_TRAFFIC_CLASS",
    "NVSHMEM_DISABLE_NVLS",
    "VLLM_DEEP_GEMM_WARMUP",
    "VLLM_MOE_ROUTING_SIMULATION_STRATEGY",
    "VLLM_RANDOMIZE_DP_DUMMY_INPUTS",
)
DEFAULT_ENV_OVERRIDES = {
    "VLLM_DEEP_GEMM_WARMUP": "skip",
    "VLLM_LOG_STATS_INTERVAL": "1",
    "VLLM_MOE_ROUTING_SIMULATION_STRATEGY": "uniform_random",
    "VLLM_RANDOMIZE_DP_DUMMY_INPUTS": "1",
}
DEFAULT_SHARED_CLI_ARGS = (
    "--no-enable-prefix-caching",
    "--trust-remote-code",
)
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_BENCH_TIMEOUT_SEC = 35 * 60
DEFAULT_SWEEP_BENCH_DURATION_SEC = 600.0
PRESTART_CLEANUP_MAX_ATTEMPTS = 3
PRESTART_CLEANUP_WAIT_SEC = 10.0
PRESTART_CLEANUP_POLL_INTERVAL_SEC = 1.0
TPOT_BY_E2E_EARLY_STOP_MS = 100.0
RATE_SWEEP_START = 10
RATE_SWEEP_STOP = 90
RATE_SWEEP_STEP = 10
MID_RATE_SWEEP_OFFSET = RATE_SWEEP_STEP // 2
SWEEP_REQUEST_RATES: tuple[float, ...] = tuple(
    float(rate) for rate in range(RATE_SWEEP_START, RATE_SWEEP_STOP + 1,
                                  RATE_SWEEP_STEP))
MID_SWEEP_REQUEST_RATES: tuple[float, ...] = tuple(
    float(rate)
    for rate in range(RATE_SWEEP_START + MID_RATE_SWEEP_OFFSET,
                      RATE_SWEEP_STOP, RATE_SWEEP_STEP))
DEFAULT_RATE_PLAN = "coarse10"
RATE_PLAN_PHASES: dict[str, tuple[tuple[str, tuple[float, ...]], ...]] = {
    "coarse10": (("coarse10", SWEEP_REQUEST_RATES), ),
    "coarse10_then_mid5": (
        ("coarse10", SWEEP_REQUEST_RATES),
        ("mid5", MID_SWEEP_REQUEST_RATES),
    ),
}
MODEL_SHORT_NAMES: dict[str, str] = {
    "deepseek_v3_1024k": "DPSK",
    "kimi_k2_instruct_0905": "KIMI",
}
DATASET_SHORT_NAMES: dict[str, str] = {
    "issue01_random": "issue01_random",
}


@dataclass(frozen=True)
class ClusterSpec:
    master_addr: str
    master_port: int
    remote_hosts: tuple[str, ...]
    workdir: str = DEFAULT_WORKDIR
    ssh_opts: tuple[str, ...] = DEFAULT_SSH_OPTS
    local_shell: str = "zsh"
    local_shell_flags: tuple[str, ...] = ("-lc",)
    remote_shell: str = "zsh"
    remote_shell_flags: tuple[str, ...] = ("-lc",)
    local_env_script: str | None = None
    remote_env_script: str | None = None

    @property
    def nnodes(self) -> int:
        return 1 + len(self.remote_hosts)


@dataclass(frozen=True)
class StrategySpec:
    data_parallel_size: int
    data_parallel_size_local: int
    tensor_parallel_size: int
    decode_context_parallel_size: int = 1
    data_parallel_backend: str = "mp"
    enable_expert_parallel: bool = True
    attention_backend: str | None = "FLASHMLA"
    all2all_backend: str | None = "deepep_low_latency"
    dcp_comm_backend: str | None = None
    max_num_batched_tokens: int | None = None


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    cluster: str
    strategy: str
    dataset: str
    model: str = "deepseek_v3_1024k"
    request_rate: float = 100.0
    rate_phase: str = DEFAULT_RATE_PLAN
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    max_requests: int | None = None
    warmup_requests: int = 0
    max_num_seqs: int | None = None
    max_model_len: int | None = None
    max_num_batched_tokens: int | None = None
    data_parallel_rpc_port: int = 29550
    cuda_visible_devices: str = DEFAULT_CUDA_VISIBLE_DEVICES
    env: Mapping[str, str] = field(default_factory=dict)
    frontend_env: Mapping[str, str] = field(default_factory=dict)
    headless_env: Mapping[str, str] = field(default_factory=dict)
    shared_cli_args: tuple[str, ...] = DEFAULT_SHARED_CLI_ARGS
    frontend_extra_args: tuple[str, ...] = ()
    headless_extra_args: tuple[str, ...] = ()
    request_id_prefix: str | None = None
    seed: int = 0
    progress_log_interval: int = 100
    save_merged_parquet: bool = False
    remote_hosts_override: tuple[str, ...] | None = None
    start_grace_sec: float = 8.0
    remote_shutdown_grace_sec: float = 30.0
    cleanup_worker_processes: bool = True
    max_bench_duration_sec: float | None = DEFAULT_BENCH_TIMEOUT_SEC


@dataclass(frozen=True)
class ResolvedCase:
    case: ExperimentCase
    cluster_name: str
    cluster: ClusterSpec
    strategy_name: str
    strategy: StrategySpec
    dataset_path: Path
    model_path: Path


@dataclass(frozen=True)
class ArtifactPaths:
    run_dir: Path
    case_dir: Path
    benchmark_dir: Path
    case_manifest_path: Path
    frontend_command_path: Path
    frontend_cleanup_command_path: Path
    frontend_pid_path: Path
    frontend_log_path: Path
    frontend_launch_log_path: Path
    rank_command_paths: dict[int, Path]
    rank_cleanup_command_paths: dict[int, Path]
    rank_pid_paths: dict[int, Path]
    rank_log_paths: dict[int, Path]
    rank_launch_log_paths: dict[int, Path]


@dataclass
class NodeRuntime:
    node_rank: int
    host: str
    process: subprocess.Popen[str]
    launch_log_path: Path
    launch_log_handle: TextIO
    runtime_log_path: Path


@dataclass
class ActiveCaseRuntime:
    resolved: ResolvedCase
    artifacts: ArtifactPaths
    frontend: NodeRuntime | None = None
    headless_nodes: list[NodeRuntime] = field(default_factory=list)


@dataclass(frozen=True)
class CaseResult:
    case_name: str
    status: str
    exit_code: int | None
    started_at: str
    finished_at: str
    case_dir: Path
    benchmark_dir: Path
    summary_json: Path
    detail: str


class BenchTimeoutError(TimeoutError):
    pass


# -----------------------------------------------------------------------------
# Manual config block: edit these directly for your cluster and experiments.
# -----------------------------------------------------------------------------

MODELS: dict[str, str] = {
    "deepseek_v3_1024k":
    "/mnt/nvme1n1/ml_research/models/deepseek-v3-1024k/",
    "kimi_k2_instruct_0905":
    "/mnt/nvme1n1/ml_research/models/models--moonshotai--Kimi-K2-Instruct-0905/"
    "snapshots/7152993552508c9f22042b3bb93b5e6acd06ce73",
}

DATASETS: dict[str, str] = {
    "1k1k":
    "/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset-0110/1024-1024.csv",
    # "issue01_halfhalf":
    # "/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/"
    # "sharegpt-4o-mixlong-0326/sharegpt4o-halfhalf_geminiissue_r0.01_n60000_60k.csv",
    "issue01_random":
    "/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/"
    "sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv",
    "issue03_random":
    "/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/"
    "sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.03_n60000.csv",
    # "issue05_halfhalf":
    # "/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/"
    # "sharegpt-4o-mixlong-0326/sharegpt4o-halfhalf_geminiissue_r0.05_n60000_60k.csv",
    "issue05_random":
    "/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/"
    "sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv",
    # "short_halfhalf":
    # "/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o/"
    # "sharegpt4o-mixed-half-half-60k.csv",
    "short_random":
    "/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o/"
    "sharegpt4o-mixed-random-60k.csv",
}

CLUSTERS: dict[str, ClusterSpec] = {
    "2node_h200":
    ClusterSpec(
        master_addr="10.102.98.166",
        master_port=29579,
        remote_hosts=("h200-rjob1", ),
    ),
    "4node_h200":
    ClusterSpec(
        master_addr="10.102.97.179",
        master_port=29579,
        remote_hosts=("h200-rjob1", "h200-rjob2", "h200-rjob3"),
    ),
}

STRATEGIES: dict[str, StrategySpec] = {
    "dp4dcp8":
    StrategySpec(
        data_parallel_size=4,
        data_parallel_size_local=1,
        tensor_parallel_size=8,
        decode_context_parallel_size=8,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASHMLA",
        all2all_backend="deepep_low_latency",
        dcp_comm_backend="a2a",
    ),
    "dp8dcp4":
    StrategySpec(
        data_parallel_size=8,
        data_parallel_size_local=2,
        tensor_parallel_size=4,
        decode_context_parallel_size=4,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASHMLA",
        all2all_backend="deepep_low_latency",
        dcp_comm_backend="a2a",
    ),
    "dp16cp2":
    StrategySpec(
        data_parallel_size=16,
        data_parallel_size_local=4,
        tensor_parallel_size=2,
        decode_context_parallel_size=2,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASHMLA",
        all2all_backend="deepep_low_latency",
        dcp_comm_backend="a2a",
    ),
    "dp32":
    StrategySpec(
        data_parallel_size=32,
        data_parallel_size_local=8,
        tensor_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASHMLA",
        all2all_backend="deepep_low_latency",
    ),
}

SWEEP_CLUSTER = "4node_h200"
SWEEP_DATASET = "issue01_random"
SWEEP_MODELS: tuple[str, ...] = (
    "kimi_k2_instruct_0905",
    "deepseek_v3_1024k",
)
SWEEP_STRATEGIES: tuple[str, ...] = (
    "dp4dcp8",
    "dp8dcp4",
    "dp16cp2",
    "dp32",
)
STRATEGY_MAX_NUM_SEQS: dict[str, int] = {
    "dp4dcp8": 768,
    "dp8dcp4": 512,
    "dp16cp2": 384,
    "dp32": 256,
}
STRATEGY_GPU_MEMORY_UTILIZATION: dict[str, float] = {
    strategy_name: DEFAULT_GPU_MEMORY_UTILIZATION
    for strategy_name in SWEEP_STRATEGIES
}
STRATEGY_GPU_MEMORY_UTILIZATION["dp32"] = 0.9


def bench_duration_to_max_requests(request_rate: float,
                                   bench_duration_sec: float) -> int:
    if math.isinf(request_rate):
        raise ValueError(
            "bench_duration_to_max_requests does not support inf request_rate."
        )
    if bench_duration_sec <= 0.0:
        raise ValueError("bench_duration_sec must be > 0.")
    return max(1, int(round(request_rate * bench_duration_sec)))


def build_experiment_matrix(
    rate_plan: str = DEFAULT_RATE_PLAN,
) -> list[ExperimentCase]:
    try:
        rate_plan_phases = RATE_PLAN_PHASES[rate_plan]
    except KeyError as exc:
        supported = ", ".join(sorted(RATE_PLAN_PHASES))
        raise ValueError(
            f"Unknown rate plan '{rate_plan}'. Supported values: {supported}"
        ) from exc

    experiments: list[ExperimentCase] = []
    dataset_tag = DATASET_SHORT_NAMES.get(SWEEP_DATASET, SWEEP_DATASET)
    for rate_phase, request_rates in rate_plan_phases:
        for model_name in SWEEP_MODELS:
            model_tag = MODEL_SHORT_NAMES.get(model_name, model_name)
            for strategy_name in SWEEP_STRATEGIES:
                max_num_seqs = STRATEGY_MAX_NUM_SEQS[strategy_name]
                gpu_memory_utilization = STRATEGY_GPU_MEMORY_UTILIZATION[
                    strategy_name]
                for request_rate in request_rates:
                    rate_tag = f"{request_rate:g}"
                    max_requests = bench_duration_to_max_requests(
                        request_rate,
                        DEFAULT_SWEEP_BENCH_DURATION_SEC,
                    )
                    experiments.append(
                        ExperimentCase(
                            name=(f"{model_tag}__{dataset_tag}__{strategy_name}"
                                  f"__rate{rate_tag}__bs{max_num_seqs}"),
                            cluster=SWEEP_CLUSTER,
                            strategy=strategy_name,
                            dataset=SWEEP_DATASET,
                            model=model_name,
                            request_rate=request_rate,
                            rate_phase=rate_phase,
                            gpu_memory_utilization=gpu_memory_utilization,
                            max_requests=max_requests,
                            warmup_requests=32,
                            max_num_seqs=max_num_seqs,
                            max_model_len=1000000,
                            data_parallel_rpc_port=29550,
                        ))
    return experiments


EXPERIMENTS: list[ExperimentCase] = build_experiment_matrix()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def current_iso_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def current_run_tag() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def shell_join(args: list[str] | tuple[str, ...]) -> str:
    return shlex.join(list(args))


def shell_command_argv(shell: str, flags: tuple[str, ...], command: str) -> list[str]:
    return [shell, *flags, command]


def sanitize_tag(raw_value: str) -> str:
    lowered = raw_value.strip().lower()
    output: list[str] = []
    last_was_sep = False
    for ch in lowered:
        if ch.isalnum():
            output.append(ch)
            last_was_sep = False
            continue
        if not last_was_sep:
            output.append("_")
            last_was_sep = True
    return "".join(output).strip("_") or "case"


def stringify_request_rate(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:g}"


def infer_bench_duration_sec(case: ExperimentCase) -> float:
    if case.max_requests is not None and not math.isinf(case.request_rate):
        return case.max_requests / case.request_rate
    return DEFAULT_SWEEP_BENCH_DURATION_SEC


def stringify_bench_duration_sec(value: float) -> str:
    if math.isclose(value, round(value)):
        return f"{int(round(value))}"
    return f"{value:g}"


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("Expected a positive float.")
    return parsed


def model_short_name(model: str) -> str:
    return MODEL_SHORT_NAMES.get(model, sanitize_tag(model).upper())


def dataset_short_name(dataset: str) -> str:
    return DATASET_SHORT_NAMES.get(dataset, sanitize_tag(dataset))


def stringify_gpu_memory_utilization(value: float) -> str:
    percentage = value * 100.0
    if math.isclose(percentage, round(percentage)):
        return f"mem{int(round(percentage))}"
    return f"mem{percentage:g}".replace(".", "_")


def case_group_key(case: ExperimentCase) -> str:
    return "/".join(
        (model_short_name(case.model), dataset_short_name(case.dataset),
         sanitize_tag(case.strategy)))


def case_artifact_group_dir(artifact_root: Path, case: ExperimentCase) -> Path:
    batch_size_tag = (f"bs{case.max_num_seqs}"
                      if case.max_num_seqs is not None else "bsauto")
    duration_tag = f"dur{stringify_bench_duration_sec(infer_bench_duration_sec(case))}"
    rate_tag = f"rate{stringify_request_rate(case.request_rate)}-{duration_tag}"
    scenario_tag = (
        f"{sanitize_tag(case.strategy)}-"
        f"{stringify_gpu_memory_utilization(case.gpu_memory_utilization)}-"
        f"{batch_size_tag}-{rate_tag}")
    return (artifact_root / model_short_name(case.model) /
            dataset_short_name(case.dataset) / scenario_tag)


def bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def tail_file(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def jsonify(value: Any) -> Any:
    if is_dataclass(value):
        return jsonify(asdict(value))
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonify(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(
                    f"Expected JSON object line in {path}, got {type(payload).__name__}"
                )
            records.append(payload)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(jsonify(record), sort_keys=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def write_shell_script(path: Path, command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + command + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def resolve_alias_path(
    raw_value: str,
    aliases: Mapping[str, str],
    *,
    label: str,
    expect_file: bool,
) -> Path:
    resolved = aliases.get(raw_value, raw_value)
    path = Path(resolved).expanduser()
    if expect_file:
        if not path.is_file():
            raise SystemExit(f"{label} file not found: {path}")
    elif not path.exists():
        raise SystemExit(f"{label} path not found: {path}")
    return path.resolve()


def experiments_by_name(
    experiments: list[ExperimentCase],
) -> dict[str, ExperimentCase]:
    return {case.name: case for case in experiments}


def resolve_case(
    case: ExperimentCase,
    *,
    clusters: Mapping[str, ClusterSpec] = CLUSTERS,
    strategies: Mapping[str, StrategySpec] = STRATEGIES,
    datasets: Mapping[str, str] = DATASETS,
    models: Mapping[str, str] = MODELS,
) -> ResolvedCase:
    try:
        cluster = clusters[case.cluster]
    except KeyError as exc:
        supported = ", ".join(sorted(clusters))
        raise SystemExit(
            f"Unknown cluster '{case.cluster}'. Supported values: {supported}"
        ) from exc

    try:
        strategy = strategies[case.strategy]
    except KeyError as exc:
        supported = ", ".join(sorted(strategies))
        raise SystemExit(
            f"Unknown strategy '{case.strategy}'. Supported values: {supported}"
        ) from exc

    if case.request_rate <= 0.0 and not math.isinf(case.request_rate):
        raise SystemExit(
            f"request_rate must be > 0 or inf for case '{case.name}', got {case.request_rate}"
        )

    resolved_remote_hosts = tuple(case.remote_hosts_override or cluster.remote_hosts)
    resolved_cluster = replace(cluster, remote_hosts=resolved_remote_hosts)

    if resolved_cluster.nnodes <= 0:
        raise SystemExit(f"Resolved cluster for case '{case.name}' has no nodes.")

    expected_dp = resolved_cluster.nnodes * strategy.data_parallel_size_local
    if strategy.data_parallel_size != expected_dp:
        raise SystemExit(
            "Strategy / cluster mismatch for "
            f"'{case.name}': dp={strategy.data_parallel_size}, "
            f"nnodes={resolved_cluster.nnodes}, "
            f"dp_local={strategy.data_parallel_size_local}, "
            f"expected dp={expected_dp}"
        )

    dataset_path = resolve_alias_path(
        case.dataset,
        datasets,
        label=f"Dataset for case '{case.name}'",
        expect_file=True,
    )
    model_path = resolve_alias_path(
        case.model,
        models,
        label=f"Model for case '{case.name}'",
        expect_file=False,
    )

    return ResolvedCase(
        case=case,
        cluster_name=case.cluster,
        cluster=resolved_cluster,
        strategy_name=case.strategy,
        strategy=strategy,
        dataset_path=dataset_path,
        model_path=model_path,
    )


def prepare_artifact_paths(case_group_dir: Path,
                           remote_hosts: tuple[str, ...],
                           *,
                           run_tag: str) -> ArtifactPaths:
    case_dir = case_group_dir / run_tag
    benchmark_dir = case_dir / "benchmark"
    rank_command_paths: dict[int, Path] = {}
    rank_cleanup_command_paths: dict[int, Path] = {}
    rank_pid_paths: dict[int, Path] = {}
    rank_log_paths: dict[int, Path] = {}
    rank_launch_log_paths: dict[int, Path] = {}

    for offset, _host in enumerate(remote_hosts, start=1):
        rank_command_paths[offset] = case_dir / f"rank{offset}.command.sh"
        rank_cleanup_command_paths[offset] = (
            case_dir / f"rank{offset}.cleanup.command.sh")
        rank_pid_paths[offset] = case_dir / f"rank{offset}.pid"
        rank_log_paths[offset] = case_dir / f"rank{offset}.log"
        rank_launch_log_paths[offset] = case_dir / f"rank{offset}.launch.log"

    return ArtifactPaths(
        run_dir=case_group_dir,
        case_dir=case_dir,
        benchmark_dir=benchmark_dir,
        case_manifest_path=case_dir / "case_manifest.json",
        frontend_command_path=case_dir / "frontend.command.sh",
        frontend_cleanup_command_path=case_dir / "frontend.cleanup.command.sh",
        frontend_pid_path=case_dir / "frontend.pid",
        frontend_log_path=case_dir / "frontend.log",
        frontend_launch_log_path=case_dir / "frontend.launch.log",
        rank_command_paths=rank_command_paths,
        rank_cleanup_command_paths=rank_cleanup_command_paths,
        rank_pid_paths=rank_pid_paths,
        rank_log_paths=rank_log_paths,
        rank_launch_log_paths=rank_launch_log_paths,
    )


def effective_max_num_batched_tokens(resolved: ResolvedCase) -> int | None:
    if resolved.case.max_num_batched_tokens is not None:
        return resolved.case.max_num_batched_tokens
    return resolved.strategy.max_num_batched_tokens


def build_common_harness_argv(
    resolved: ResolvedCase,
    *,
    role: str,
    node_rank: int,
) -> list[str]:
    argv = [
        "python3",
        HARNESS_ENTRYPOINT,
        role,
        "--model",
        str(resolved.model_path),
        "--master-addr",
        resolved.cluster.master_addr,
        "--master-port",
        str(resolved.cluster.master_port),
        "--nnodes",
        str(resolved.cluster.nnodes),
        "--node-rank",
        str(node_rank),
        "--data-parallel-size",
        str(resolved.strategy.data_parallel_size),
        "--data-parallel-size-local",
        str(resolved.strategy.data_parallel_size_local),
        "--data-parallel-rpc-port",
        str(resolved.case.data_parallel_rpc_port),
        "--data-parallel-backend",
        resolved.strategy.data_parallel_backend,
        "--tensor-parallel-size",
        str(resolved.strategy.tensor_parallel_size),
        "--decode-context-parallel-size",
        str(resolved.strategy.decode_context_parallel_size),
        bool_flag("enable-expert-parallel",
                  resolved.strategy.enable_expert_parallel),
    ]

    if resolved.strategy.attention_backend is not None:
        argv.extend(["--attention-backend", resolved.strategy.attention_backend])
    if resolved.strategy.all2all_backend is not None:
        argv.extend(["--all2all-backend", resolved.strategy.all2all_backend])
    if resolved.strategy.dcp_comm_backend is not None:
        argv.extend(["--dcp-comm-backend", resolved.strategy.dcp_comm_backend])
    argv.extend([
        "--gpu-memory-utilization",
        f"{resolved.case.gpu_memory_utilization:g}",
    ])
    if resolved.case.max_num_seqs is not None:
        argv.extend(["--max-num-seqs", str(resolved.case.max_num_seqs)])
    if resolved.case.max_model_len is not None:
        argv.extend(["--max-model-len", str(resolved.case.max_model_len)])
    max_num_batched_tokens = effective_max_num_batched_tokens(resolved)
    if max_num_batched_tokens is not None:
        argv.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])

    argv.extend(resolved.case.shared_cli_args)
    return argv


def build_frontend_argv(resolved: ResolvedCase, output_dir: Path) -> list[str]:
    argv = build_common_harness_argv(
        resolved,
        role="frontend",
        node_rank=0,
    )
    argv.extend([
        "--input-csv",
        str(resolved.dataset_path),
        "--request-rate",
        stringify_request_rate(resolved.case.request_rate),
        "--warmup-requests",
        str(resolved.case.warmup_requests),
        "--output-dir",
        str(output_dir),
        "--progress-log-interval",
        str(resolved.case.progress_log_interval),
        "--seed",
        str(resolved.case.seed),
    ])
    if resolved.case.max_requests is not None:
        argv.extend(["--max-requests", str(resolved.case.max_requests)])
    if resolved.case.request_id_prefix is not None:
        argv.extend(["--request-id-prefix", resolved.case.request_id_prefix])
    if resolved.case.save_merged_parquet:
        argv.append("--save-merged-parquet")

    argv.extend(resolved.case.frontend_extra_args)
    return argv


def build_headless_argv(resolved: ResolvedCase, node_rank: int) -> list[str]:
    argv = build_common_harness_argv(
        resolved,
        role="headless-engine",
        node_rank=node_rank,
    )
    argv.extend(resolved.case.headless_extra_args)
    return argv


def collect_role_env(resolved: ResolvedCase, *, role: str) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for key in FORWARDED_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            forwarded[key] = value
    forwarded.update(DEFAULT_ENV_OVERRIDES)
    forwarded["CUDA_VISIBLE_DEVICES"] = resolved.case.cuda_visible_devices
    forwarded.update({key: str(value) for key, value in resolved.case.env.items()})
    role_overrides = (resolved.case.frontend_env if role == "frontend" else
                      resolved.case.headless_env)
    forwarded.update({key: str(value) for key, value in role_overrides.items()})
    return forwarded


def build_runtime_shell_command(
    *,
    cwd: str,
    argv: list[str],
    env: Mapping[str, str],
    env_script: str | None,
    log_path: Path,
    pid_path: Path,
) -> str:
    env_prefix = " ".join(
        f"{key}={shlex.quote(str(value))}" for key, value in env.items())
    command = shell_join(argv)
    if env_prefix:
        command = f"env {env_prefix} {command}"
    run_command = (
        f"echo $$ > {shlex.quote(str(pid_path))} && exec {command} "
        f"2>&1 | tee {shlex.quote(str(log_path))} >/dev/null")

    steps = [f"cd {shlex.quote(cwd)}"]
    if env_script:
        steps.append(f"source {shlex.quote(env_script)}")
    steps.append(f"set -o pipefail && {run_command}")
    return " && ".join(steps)


def build_node_cleanup_command(
    *,
    resolved: ResolvedCase,
    pid_path: Path,
    include_frontend_pattern: bool,
) -> str:
    def append_pattern_cleanup(
        pattern: str,
        *,
        wait_pattern: str | None = None,
    ) -> None:
        wait_pattern = wait_pattern or pattern
        lines.append(
            f"pkill -TERM -f -- {shlex.quote(pattern)} >/dev/null 2>&1 || true")
        lines.extend([
            "for _ in 1 2 3 4 5; do",
            f"  if ! pgrep -f -- {shlex.quote(wait_pattern)} >/dev/null 2>&1; then",
            "    break",
            "  fi",
            "  sleep 1",
            "done",
            f"pkill -KILL -f -- {shlex.quote(wait_pattern)} >/dev/null 2>&1 || true",
        ])

    lines = [
        "set +e",
        f"if [ -f {shlex.quote(str(pid_path))} ]; then",
        f"  pid=$(cat {shlex.quote(str(pid_path))} 2>/dev/null || true)",
        "  if [ -n \"$pid\" ] && kill -0 \"$pid\" >/dev/null 2>&1; then",
        "    kill -TERM \"$pid\" >/dev/null 2>&1 || true",
        "    sleep 5",
        "    if kill -0 \"$pid\" >/dev/null 2>&1; then",
        "      kill -KILL \"$pid\" >/dev/null 2>&1 || true",
        "    fi",
        "  fi",
        "fi",
        f"rm -f {shlex.quote(str(pid_path))} >/dev/null 2>&1 || true",
    ]

    append_pattern_cleanup(
        f"python3 {HARNESS_ENTRYPOINT} headless-engine",
        wait_pattern=f"^python3 {HARNESS_ENTRYPOINT} headless-engine( |$)",
    )
    if include_frontend_pattern:
        append_pattern_cleanup(
            f"python3 {HARNESS_ENTRYPOINT} frontend",
            wait_pattern=f"^python3 {HARNESS_ENTRYPOINT} frontend( |$)",
        )

    for pattern in (
            f"--master-port {resolved.cluster.master_port}",
            f"--data-parallel-rpc-port {resolved.case.data_parallel_rpc_port}"):
        lines.append(
            f"pkill -f -- {shlex.quote(pattern)} >/dev/null 2>&1 || true")
    if resolved.case.cleanup_worker_processes:
        append_pattern_cleanup("^VLLM::Worker_")
    return "\n".join(lines)


def build_local_launch_command(resolved: ResolvedCase, artifacts: ArtifactPaths,
                               frontend_argv: list[str]) -> str:
    return build_runtime_shell_command(
        cwd=resolved.cluster.workdir,
        argv=frontend_argv,
        env=collect_role_env(resolved, role="frontend"),
        env_script=resolved.cluster.local_env_script,
        log_path=artifacts.frontend_log_path,
        pid_path=artifacts.frontend_pid_path,
    )


def build_remote_launch_command(
    resolved: ResolvedCase,
    artifacts: ArtifactPaths,
    *,
    node_rank: int,
) -> str:
    return build_runtime_shell_command(
        cwd=resolved.cluster.workdir,
        argv=build_headless_argv(resolved, node_rank),
        env=collect_role_env(resolved, role="headless-engine"),
        env_script=resolved.cluster.remote_env_script,
        log_path=artifacts.rank_log_paths[node_rank],
        pid_path=artifacts.rank_pid_paths[node_rank],
    )


def build_remote_ssh_command(cluster: ClusterSpec, host: str,
                             remote_command: str) -> list[str]:
    remote_shell_argv = shell_command_argv(
        cluster.remote_shell,
        cluster.remote_shell_flags,
        remote_command,
    )
    return [
        "ssh",
        *cluster.ssh_opts,
        host,
        shell_join(remote_shell_argv),
    ]


def write_case_manifest(
    resolved: ResolvedCase,
    artifacts: ArtifactPaths,
    *,
    frontend_launch_command: str,
    remote_launch_commands: dict[int, str],
    local_cleanup_command: str,
    remote_cleanup_commands: dict[int, str],
    status: str,
    started_at: str | None,
    finished_at: str | None,
    exit_code: int | None,
    detail: str | None,
) -> None:
    payload = {
        "case_name": resolved.case.name,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": exit_code,
        "detail": detail,
        "shared_artifact_mount": True,
        "cluster_name": resolved.cluster_name,
        "strategy_name": resolved.strategy_name,
        "cluster": resolved.cluster,
        "strategy": resolved.strategy,
        "case": resolved.case,
        "model_path": resolved.model_path,
        "dataset_path": resolved.dataset_path,
        "paths": artifacts,
        "commands": {
            "frontend_launch_command": frontend_launch_command,
            "remote_launch_commands": remote_launch_commands,
            "frontend_cleanup_command": local_cleanup_command,
            "remote_cleanup_commands": remote_cleanup_commands,
        },
    }
    write_json(artifacts.case_manifest_path, payload)


def write_case_command_files(
    resolved: ResolvedCase,
    artifacts: ArtifactPaths,
    *,
    frontend_launch_command: str,
    remote_launch_commands: dict[int, str],
    local_cleanup_command: str,
    remote_cleanup_commands: dict[int, str],
) -> None:
    write_shell_script(artifacts.frontend_command_path, frontend_launch_command)
    write_shell_script(artifacts.frontend_cleanup_command_path,
                       local_cleanup_command)
    for node_rank, command in remote_launch_commands.items():
        write_shell_script(artifacts.rank_command_paths[node_rank], command)
    for node_rank, command in remote_cleanup_commands.items():
        write_shell_script(artifacts.rank_cleanup_command_paths[node_rank], command)


def select_cases(args: argparse.Namespace,
                 experiments: list[ExperimentCase]) -> list[ExperimentCase]:
    available = experiments_by_name(experiments)
    if args.all and args.case:
        raise SystemExit("Use either --all or --case, not both.")
    if not args.all and not args.case and not args.list:
        raise SystemExit("Select at least one case via --case or use --all.")
    if args.all:
        return list(experiments)
    selected: list[ExperimentCase] = []
    seen: set[str] = set()
    for case_name in args.case or []:
        if case_name in seen:
            continue
        try:
            selected.append(available[case_name])
        except KeyError as exc:
            supported = ", ".join(sorted(available))
            raise SystemExit(
                f"Unknown case '{case_name}'. Supported values: {supported}"
            ) from exc
        seen.add(case_name)
    return selected


def apply_bench_duration_override(
    cases: list[ExperimentCase],
    bench_duration_sec: float | None,
) -> list[ExperimentCase]:
    if bench_duration_sec is None:
        return cases

    overridden: list[ExperimentCase] = []
    for case in cases:
        if math.isinf(case.request_rate):
            raise SystemExit(
                "--bench-duration-sec does not support cases with request_rate=inf."
            )
        overridden.append(
            replace(
                case,
                max_requests=bench_duration_to_max_requests(
                    case.request_rate,
                    bench_duration_sec,
                ),
            ))
    return overridden


def describe_case(case: ExperimentCase) -> str:
    return (
        f"{case.name}: model={model_short_name(case.model)}, "
        f"cluster={case.cluster}, strategy={case.strategy}, "
        f"dataset={case.dataset}, phase={case.rate_phase}, "
        f"rate={stringify_request_rate(case.request_rate)}, "
        f"mem={stringify_gpu_memory_utilization(case.gpu_memory_utilization)}, "
        f"bs={case.max_num_seqs}"
    )


def _metric_summary(records: list[dict[str, Any]],
                    key: str) -> dict[str, float] | None:
    import numpy as np

    values = [
        float(record[key]) for record in records
        if not record.get("is_error") and record.get(key) is not None
    ]
    if not values:
        return None

    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


def _compute_tpot_metrics(
    record: Mapping[str, Any]
) -> tuple[float | None, float | None]:
    if record.get("is_error"):
        return None, None

    actual_output_tokens = int(record.get("actual_output_tokens") or 0)
    if actual_output_tokens <= 0:
        return None, None

    decode_time_ms = record.get("decode_time_ms")
    e2e_ms = record.get("e2e_ms")

    tpot_without_queue_ms = None
    decode_output_tokens = actual_output_tokens - 1
    if decode_time_ms is not None and decode_output_tokens > 0:
        tpot_without_queue_ms = float(decode_time_ms) / decode_output_tokens

    tpot_by_e2e = None
    if e2e_ms is not None:
        tpot_by_e2e = float(e2e_ms) / actual_output_tokens

    return tpot_without_queue_ms, tpot_by_e2e


def augment_benchmark_outputs(benchmark_dir: Path) -> None:
    requests_path = benchmark_dir / "requests.jsonl"
    summary_path = benchmark_dir / "summary.json"

    if not requests_path.is_file():
        raise FileNotFoundError(f"Benchmark requests file not found: {requests_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Benchmark summary file not found: {summary_path}")

    records = load_jsonl(requests_path)
    for record in records:
        (record["tpot_without_queue_ms"],
         record["tpot_by_e2e"]) = _compute_tpot_metrics(record)
        record.pop("tpot_with_initial_queue_ms", None)

    summary = load_json(summary_path)
    summary["tpot_without_queue_ms"] = _metric_summary(records,
                                                       "tpot_without_queue_ms")
    summary["tpot_by_e2e"] = _metric_summary(records, "tpot_by_e2e")
    summary.pop("tpot_with_initial_queue_ms", None)

    write_jsonl(requests_path, records)
    write_json(summary_path, summary)


def extract_summary_tpot_by_e2e_mean(summary: Mapping[str, Any]) -> float | None:
    metric = summary.get("tpot_by_e2e")
    if not isinstance(metric, Mapping):
        return None

    mean_value = metric.get("mean")
    if mean_value is None:
        return None

    try:
        parsed = float(mean_value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed):
        return None
    return parsed


def should_stop_followup_rates(result: CaseResult) -> tuple[bool, str | None]:
    if result.status == "dry_run":
        return False, None

    if result.status != "ok":
        return True, f"case status={result.status}"

    try:
        summary = load_json(result.summary_json)
    except Exception as exc:
        return True, f"unable to read summary.json: {exc}"

    tpot_by_e2e_mean = extract_summary_tpot_by_e2e_mean(summary)
    if tpot_by_e2e_mean is None:
        return True, "missing tpot_by_e2e.mean"

    if tpot_by_e2e_mean > TPOT_BY_E2E_EARLY_STOP_MS:
        return (
            True,
            ("tpot_by_e2e.mean="
             f"{tpot_by_e2e_mean:.3f}ms > {TPOT_BY_E2E_EARLY_STOP_MS:g}ms"),
        )

    return False, None


def latest_successful_case_dir(case_group_dir: Path) -> Path | None:
    if not case_group_dir.is_dir():
        return None

    candidates = sorted((path for path in case_group_dir.iterdir() if path.is_dir()),
                        reverse=True)
    for case_dir in candidates:
        manifest_path = case_dir / "case_manifest.json"
        if not manifest_path.is_file():
            continue

        try:
            manifest = load_json(manifest_path)
        except Exception:
            continue

        if manifest.get("status") == "ok":
            return case_dir
    return None


def latest_successful_case_tpot_by_e2e_mean(case_group_dir: Path) -> tuple[
        Path, float | None] | None:
    case_dir = latest_successful_case_dir(case_group_dir)
    if case_dir is None:
        return None

    summary_path = case_dir / "benchmark" / "summary.json"
    if not summary_path.is_file():
        return case_dir, None

    try:
        summary = load_json(summary_path)
    except Exception:
        return case_dir, None

    return case_dir, extract_summary_tpot_by_e2e_mean(summary)


def build_historical_skip_state(
    artifact_root: Path,
    selected_cases: list[ExperimentCase],
) -> tuple[dict[str, str], dict[str, tuple[float, str]]]:
    exact_case_skips: dict[str, str] = {}
    blocked_group_rates: dict[str, tuple[float, str]] = {}

    for case in selected_cases:
        historical = latest_successful_case_tpot_by_e2e_mean(
            case_artifact_group_dir(artifact_root, case))
        if historical is None:
            continue

        case_dir, tpot_by_e2e_mean = historical
        case_result_tag = case_dir.name
        exact_reason = (
            f"latest successful result already exists at {case_result_tag}")
        if tpot_by_e2e_mean is not None:
            exact_reason += f" (tpot_by_e2e.mean={tpot_by_e2e_mean:.3f}ms)"
        exact_case_skips[case.name] = exact_reason

        if (tpot_by_e2e_mean is None
                or tpot_by_e2e_mean <= TPOT_BY_E2E_EARLY_STOP_MS):
            continue

        group_key = case_group_key(case)
        existing = blocked_group_rates.get(group_key)
        reason = (
            "historical latest successful result for "
            f"rate={stringify_request_rate(case.request_rate)} at "
            f"{case_result_tag} has tpot_by_e2e.mean={tpot_by_e2e_mean:.3f}ms "
            f"> {TPOT_BY_E2E_EARLY_STOP_MS:g}ms")
        if existing is None or case.request_rate < existing[0]:
            blocked_group_rates[group_key] = (case.request_rate, reason)

    return exact_case_skips, blocked_group_rates


def block_reason_for_rate(
    blocked_group_rates: Mapping[str, tuple[float, str]],
    case: ExperimentCase,
) -> str | None:
    blocked = blocked_group_rates.get(case_group_key(case))
    if blocked is None:
        return None

    blocked_rate, reason = blocked
    if case.request_rate >= blocked_rate:
        return reason
    return None


def local_ports_for_case(resolved: ResolvedCase) -> tuple[int, ...]:
    return tuple(
        sorted({
            resolved.cluster.master_port,
            resolved.case.data_parallel_rpc_port,
        }))


def can_bind_local_tcp_port(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Previous cases can leave sockets in TIME_WAIT after a clean
        # shutdown. Treat the port as reusable if we can bind with
        # SO_REUSEADDR, which matches the next listener startup path more
        # closely than a bare bind probe.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
        except OSError:
            return False
    return True


def wait_for_local_ports_to_clear(
    ports: tuple[int, ...],
    *,
    timeout_sec: float,
    poll_interval_sec: float,
) -> tuple[int, ...]:
    if not ports:
        return ()

    deadline = time.monotonic() + timeout_sec
    while True:
        busy_ports = tuple(
            port for port in ports if not can_bind_local_tcp_port(port))
        if not busy_ports:
            return ()
        if time.monotonic() >= deadline:
            return busy_ports
        time.sleep(poll_interval_sec)


class ManualMultinodeRunner:

    def __init__(self, artifact_root: Path, *, dry_run: bool,
                 keep_going: bool = True) -> None:
        self.artifact_root = artifact_root.expanduser().resolve()
        self.dry_run = dry_run
        self.keep_going = keep_going
        self._active_runtime: ActiveCaseRuntime | None = None
        self._previous_handlers: dict[int, Any] = {}
        self._handling_signal = False

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)

    def restore_signal_handlers(self) -> None:
        for sig, handler in self._previous_handlers.items():
            signal.signal(sig, handler)
        self._previous_handlers.clear()

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        if self._handling_signal:
            raise SystemExit(128 + signum)
        self._handling_signal = True
        try:
            if self._active_runtime is not None:
                print(
                    f"[signal] received {signal.Signals(signum).name}, cleaning up active case...",
                    file=sys.stderr,
                )
                self.cleanup_case_runtime(self._active_runtime)
        finally:
            raise SystemExit(128 + signum)

    def run(self, selected_cases: list[ExperimentCase], run_label: str | None,
            *, rate_plan: str) -> list[CaseResult]:
        run_name = f"{current_run_tag()}__{sanitize_tag(run_label or 'manual_poisson')}"
        run_dir = self.artifact_root / "_runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        run_manifest_path = run_dir / "run_manifest.json"
        write_json(
            run_manifest_path,
            {
                "run_name": run_name,
                "created_at": current_iso_timestamp(),
                "dry_run": self.dry_run,
                "keep_going": self.keep_going,
                "rate_plan": rate_plan,
                "cases": [case.name for case in selected_cases],
            },
        )

        results: list[CaseResult] = []
        exact_case_skips, historical_blocked_groups = build_historical_skip_state(
            self.artifact_root, selected_cases)
        runtime_blocked_groups: dict[str, tuple[float, str]] = {}
        self.install_signal_handlers()
        try:
            for case in selected_cases:
                historical_group_reason = block_reason_for_rate(
                    historical_blocked_groups, case)
                if historical_group_reason is not None:
                    print(f"[skip] {case.name}: {historical_group_reason}")
                    continue

                runtime_group_reason = block_reason_for_rate(
                    runtime_blocked_groups, case)
                if runtime_group_reason is not None:
                    print(f"[skip] {case.name}: {runtime_group_reason}")
                    continue

                exact_skip_reason = exact_case_skips.get(case.name)
                if exact_skip_reason is not None:
                    print(f"[skip] {case.name}: {exact_skip_reason}")
                    continue

                result = self.run_case(case, run_dir)
                results.append(result)
                should_stop, stop_reason = should_stop_followup_rates(result)
                if should_stop:
                    reason = stop_reason or "blocked by previous case result"
                    group_key = case_group_key(case)
                    runtime_blocked_groups[group_key] = (case.request_rate,
                                                         reason)
                    print(f"[sweep] stop higher rates for {group_key}: {reason}")
                if (result.status not in {"ok", "dry_run", "timed_out"}
                        and not self.keep_going):
                    break
        finally:
            self.restore_signal_handlers()

        write_json(
            run_manifest_path,
            {
                "run_name": run_name,
                "created_at": current_iso_timestamp(),
                "dry_run": self.dry_run,
                "keep_going": self.keep_going,
                "rate_plan": rate_plan,
                "cases": [case.name for case in selected_cases],
                "historical_exact_case_skips": exact_case_skips,
                "historical_blocked_groups": {
                    key: {
                        "blocked_from_rate": rate,
                        "reason": reason,
                    }
                    for key, (rate, reason) in historical_blocked_groups.items()
                },
                "runtime_blocked_groups": {
                    key: {
                        "blocked_from_rate": rate,
                        "reason": reason,
                    }
                    for key, (rate, reason) in runtime_blocked_groups.items()
                },
                "results": [jsonify(result) for result in results],
            },
        )
        return results

    def run_case(self, case: ExperimentCase, _run_dir: Path) -> CaseResult:
        resolved = resolve_case(case)
        artifacts = prepare_artifact_paths(
            case_artifact_group_dir(self.artifact_root, case),
            resolved.cluster.remote_hosts,
            run_tag=current_run_tag(),
        )
        artifacts.case_dir.mkdir(parents=True, exist_ok=True)
        artifacts.benchmark_dir.mkdir(parents=True, exist_ok=True)

        frontend_argv = build_frontend_argv(resolved, artifacts.benchmark_dir)
        frontend_launch_command = build_local_launch_command(
            resolved,
            artifacts,
            frontend_argv,
        )
        remote_launch_commands = {
            node_rank: build_remote_launch_command(
                resolved,
                artifacts,
                node_rank=node_rank,
            )
            for node_rank in range(1, resolved.cluster.nnodes)
        }
        local_cleanup_command = build_node_cleanup_command(
            resolved=resolved,
            pid_path=artifacts.frontend_pid_path,
            include_frontend_pattern=True,
        )
        remote_cleanup_commands = {
            node_rank: build_node_cleanup_command(
                resolved=resolved,
                pid_path=artifacts.rank_pid_paths[node_rank],
                include_frontend_pattern=False,
            )
            for node_rank in range(1, resolved.cluster.nnodes)
        }

        write_case_command_files(
            resolved,
            artifacts,
            frontend_launch_command=frontend_launch_command,
            remote_launch_commands=remote_launch_commands,
            local_cleanup_command=local_cleanup_command,
            remote_cleanup_commands=remote_cleanup_commands,
        )
        write_case_manifest(
            resolved,
            artifacts,
            frontend_launch_command=frontend_launch_command,
            remote_launch_commands=remote_launch_commands,
            local_cleanup_command=local_cleanup_command,
            remote_cleanup_commands=remote_cleanup_commands,
            status="prepared",
            started_at=None,
            finished_at=None,
            exit_code=None,
            detail=None,
        )

        started_at = current_iso_timestamp()
        print(f"[case] starting {case.name}")
        if self.dry_run:
            finished_at = current_iso_timestamp()
            write_case_manifest(
                resolved,
                artifacts,
                frontend_launch_command=frontend_launch_command,
                remote_launch_commands=remote_launch_commands,
                local_cleanup_command=local_cleanup_command,
                remote_cleanup_commands=remote_cleanup_commands,
                status="dry_run",
                started_at=started_at,
                finished_at=finished_at,
                exit_code=0,
                detail="dry run: commands were written, nothing was executed",
            )
            return CaseResult(
                case_name=case.name,
                status="dry_run",
                exit_code=0,
                started_at=started_at,
                finished_at=finished_at,
                case_dir=artifacts.case_dir,
                benchmark_dir=artifacts.benchmark_dir,
                summary_json=artifacts.benchmark_dir / "summary.json",
                detail="dry run",
            )

        runtime = ActiveCaseRuntime(resolved=resolved, artifacts=artifacts)
        self._active_runtime = runtime

        exit_code: int | None = None
        detail = ""
        status = "ok"
        try:
            self.run_preclean(resolved, artifacts)
            self.launch_headless_nodes(runtime, remote_launch_commands)
            self.wait_for_headless_startup(runtime)
            self.launch_frontend(runtime, frontend_launch_command)
            exit_code = self.wait_for_frontend(runtime)
            self.wait_for_headless_shutdown(runtime)
            augment_benchmark_outputs(artifacts.benchmark_dir)
            detail = "completed"
        except BenchTimeoutError as exc:
            status = "timed_out"
            detail = str(exc)
            exit_code = 124
        except Exception as exc:
            status = "failed"
            detail = str(exc)
            exit_code = exit_code if exit_code is not None else 1
        finally:
            self.cleanup_case_runtime(runtime)
            self._active_runtime = None

        finished_at = current_iso_timestamp()
        write_case_manifest(
            resolved,
            artifacts,
            frontend_launch_command=frontend_launch_command,
            remote_launch_commands=remote_launch_commands,
            local_cleanup_command=local_cleanup_command,
            remote_cleanup_commands=remote_cleanup_commands,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            exit_code=exit_code,
            detail=detail,
        )

        print(f"[case] {case.name}: {status}")
        return CaseResult(
            case_name=case.name,
            status=status,
            exit_code=exit_code,
            started_at=started_at,
            finished_at=finished_at,
            case_dir=artifacts.case_dir,
            benchmark_dir=artifacts.benchmark_dir,
            summary_json=artifacts.benchmark_dir / "summary.json",
            detail=detail,
        )

    def run_preclean(self, resolved: ResolvedCase,
                     artifacts: ArtifactPaths) -> None:
        ports = local_ports_for_case(resolved)
        busy_ports: tuple[int, ...] = ()

        for attempt in range(1, PRESTART_CLEANUP_MAX_ATTEMPTS + 1):
            self.run_local_shell(
                resolved.cluster,
                artifacts.frontend_cleanup_command_path.read_text(
                    encoding="utf-8"),
            )
            for node_rank, host in enumerate(resolved.cluster.remote_hosts,
                                             start=1):
                self.run_remote_shell(
                    resolved.cluster,
                    host,
                    artifacts.rank_cleanup_command_paths[node_rank].read_text(
                        encoding="utf-8"),
                )

            busy_ports = wait_for_local_ports_to_clear(
                ports,
                timeout_sec=PRESTART_CLEANUP_WAIT_SEC,
                poll_interval_sec=PRESTART_CLEANUP_POLL_INTERVAL_SEC,
            )
            if not busy_ports:
                return

            ports_text = ", ".join(str(port) for port in busy_ports)
            print(
                ("[preclean] local ports still busy after cleanup "
                 f"attempt {attempt}/{PRESTART_CLEANUP_MAX_ATTEMPTS}: "
                 f"{ports_text}"),
                file=sys.stderr,
            )

        ports_text = ", ".join(str(port) for port in busy_ports)
        raise RuntimeError("preclean could not free local ports before launch: "
                           f"{ports_text}")

    def launch_headless_nodes(self, runtime: ActiveCaseRuntime,
                              remote_launch_commands: dict[int, str]) -> None:
        for node_rank, host in enumerate(runtime.resolved.cluster.remote_hosts,
                                         start=1):
            launch_log_path = runtime.artifacts.rank_launch_log_paths[node_rank]
            launch_log_path.parent.mkdir(parents=True, exist_ok=True)
            launch_log_handle = launch_log_path.open("w", encoding="utf-8")
            ssh_command = build_remote_ssh_command(
                runtime.resolved.cluster,
                host,
                remote_launch_commands[node_rank],
            )
            process = subprocess.Popen(
                ssh_command,
                stdin=subprocess.DEVNULL,
                stdout=launch_log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            runtime.headless_nodes.append(
                NodeRuntime(
                    node_rank=node_rank,
                    host=host,
                    process=process,
                    launch_log_path=launch_log_path,
                    launch_log_handle=launch_log_handle,
                    runtime_log_path=runtime.artifacts.rank_log_paths[node_rank],
                ))

    def wait_for_headless_startup(self, runtime: ActiveCaseRuntime) -> None:
        deadline = time.monotonic() + runtime.resolved.case.start_grace_sec
        while time.monotonic() < deadline:
            for node in runtime.headless_nodes:
                if node.process.poll() is not None:
                    detail = tail_file(node.launch_log_path) or tail_file(
                        node.runtime_log_path)
                    raise RuntimeError(
                        f"rank {node.node_rank} on {node.host} exited during startup.\n{detail}"
                    )
            time.sleep(0.5)

    def launch_frontend(self, runtime: ActiveCaseRuntime,
                        frontend_launch_command: str) -> None:
        launch_log_handle = runtime.artifacts.frontend_launch_log_path.open(
            "w", encoding="utf-8")
        command_argv = shell_command_argv(
            runtime.resolved.cluster.local_shell,
            runtime.resolved.cluster.local_shell_flags,
            frontend_launch_command,
        )
        process = subprocess.Popen(
            command_argv,
            cwd=runtime.resolved.cluster.workdir,
            stdin=subprocess.DEVNULL,
            stdout=launch_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        runtime.frontend = NodeRuntime(
            node_rank=0,
            host="local",
            process=process,
            launch_log_path=runtime.artifacts.frontend_launch_log_path,
            launch_log_handle=launch_log_handle,
            runtime_log_path=runtime.artifacts.frontend_log_path,
        )

    def wait_for_frontend(self, runtime: ActiveCaseRuntime) -> int:
        assert runtime.frontend is not None
        timeout_sec = runtime.resolved.case.max_bench_duration_sec
        deadline = None
        if timeout_sec is not None:
            deadline = time.monotonic() + timeout_sec
        pending_headless_nodes = list(runtime.headless_nodes)
        while True:
            exit_code = runtime.frontend.process.poll()
            if exit_code is not None:
                if exit_code != 0:
                    detail = tail_file(
                        runtime.frontend.runtime_log_path) or tail_file(
                            runtime.frontend.launch_log_path)
                    raise RuntimeError(
                        f"frontend exited with code {exit_code}.\n{detail}")
                return exit_code

            if deadline is not None and time.monotonic() >= deadline:
                detail = tail_file(runtime.frontend.runtime_log_path) or tail_file(
                    runtime.frontend.launch_log_path)
                message = ("frontend exceeded "
                           f"max_bench_duration_sec={timeout_sec:g}s "
                           f"({timeout_sec / 60.0:g}min)")
                if detail:
                    message = f"{message}.\n{detail}"
                raise BenchTimeoutError(message)

            still_running_headless_nodes = []
            for node in pending_headless_nodes:
                node_exit_code = node.process.poll()
                if node_exit_code is None:
                    still_running_headless_nodes.append(node)
                    continue
                if node_exit_code != 0:
                    detail = tail_file(node.launch_log_path) or tail_file(
                        node.runtime_log_path)
                    raise RuntimeError(
                        f"rank {node.node_rank} on {node.host} exited early "
                        f"with code {node_exit_code}.\n{detail}")
            pending_headless_nodes = still_running_headless_nodes
            time.sleep(1.0)

    def wait_for_headless_shutdown(self, runtime: ActiveCaseRuntime) -> None:
        deadline = (time.monotonic() +
                    runtime.resolved.case.remote_shutdown_grace_sec)
        pending = list(runtime.headless_nodes)
        while pending and time.monotonic() < deadline:
            pending = [node for node in pending if node.process.poll() is None]
            if pending:
                time.sleep(1.0)

        if pending:
            details = []
            for node in pending:
                details.append(
                    f"rank {node.node_rank} on {node.host} did not exit in time.")
            raise RuntimeError("\n".join(details))

    def cleanup_case_runtime(self, runtime: ActiveCaseRuntime) -> None:
        if runtime.frontend is not None:
            self.terminate_process(runtime.frontend.process, "frontend")
            runtime.frontend.launch_log_handle.close()
        for node in runtime.headless_nodes:
            self.terminate_process(node.process, f"rank {node.node_rank}")
            node.launch_log_handle.close()

        self.run_local_shell(
            runtime.resolved.cluster,
            runtime.artifacts.frontend_cleanup_command_path.read_text(
                encoding="utf-8"),
        )
        for node_rank, host in enumerate(runtime.resolved.cluster.remote_hosts,
                                         start=1):
            self.run_remote_shell(
                runtime.resolved.cluster,
                host,
                runtime.artifacts.rank_cleanup_command_paths[node_rank].read_text(
                    encoding="utf-8"),
            )

    def terminate_process(self, proc: subprocess.Popen[str], label: str) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            with contextlib.suppress(Exception):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
        finally:
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.kill()
                with contextlib.suppress(Exception):
                    proc.wait(timeout=5)
            print(f"[cleanup] terminated {label}", file=sys.stderr)

    def run_local_shell(self, cluster: ClusterSpec, command: str) -> None:
        subprocess.run(
            shell_command_argv(cluster.local_shell, cluster.local_shell_flags,
                               command),
            check=False,
            cwd=cluster.workdir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def run_remote_shell(self, cluster: ClusterSpec, host: str,
                         command: str) -> None:
        ssh_command = build_remote_ssh_command(cluster, host, command)
        subprocess.run(
            ssh_command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run handwritten multi-node offline_poisson_harness cases.")
    parser.add_argument("--list",
                        action="store_true",
                        help="List the configured case names.")
    parser.add_argument(
        "--case",
        action="append",
        help="Run a specific case by name. May be repeated.",
    )
    parser.add_argument("--all",
                        action="store_true",
                        help="Run all configured cases sequentially.")
    parser.add_argument(
        "--rate-plan",
        default=DEFAULT_RATE_PLAN,
        choices=sorted(RATE_PLAN_PHASES),
        help=(
            "Request-rate sweep plan to use for --list, --all, and --case "
            "name resolution. 'coarse10' runs only 10-point spacing; "
            "'coarse10_then_mid5' appends 15/25/.../85 after the coarse pass."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="Shared artifact root for run directories.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional suffix for the timestamped run directory.",
    )
    parser.add_argument(
        "--bench-duration-sec",
        type=positive_float,
        default=None,
        help=(
            "Override measured benchmark duration in seconds for selected "
            "cases. Default configured case duration is "
            f"{DEFAULT_SWEEP_BENCH_DURATION_SEC:g}s."
        ),
    )
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Write commands and manifests without execution.")
    parser.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Continue to later cases even if one case fails. "
              "Use --no-keep-going to stop after the first failed case."),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    experiments = build_experiment_matrix(args.rate_plan)

    if args.list:
        for case in experiments:
            print(describe_case(case))
        return

    selected_cases = apply_bench_duration_override(
        select_cases(args, experiments),
        args.bench_duration_sec,
    )
    runner = ManualMultinodeRunner(
        Path(args.artifact_root),
        dry_run=args.dry_run,
        keep_going=args.keep_going,
    )
    results = runner.run(selected_cases, args.run_label, rate_plan=args.rate_plan)

    failures = [result for result in results if result.status not in {"ok", "dry_run"}]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
