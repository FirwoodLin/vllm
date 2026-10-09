#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the fixed two-node KIMI issue-1% Poisson benchmark.

This launcher is intentionally independent from manual_multinode_poisson_runner.py:
it has no experiment matrix, sweep phases, resume logic, or historical skips.
Both engine nodes are launched over SSH, so the launcher itself may run from any
host that can reach h200-rjob0 and h200-rjob2 and see the shared artifact mount.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, TextIO


HARNESS_ENTRYPOINT = "/vllm/offline_poisson_harness.py"
DEFAULT_WORKDIR = "/vllm"
DEFAULT_MODEL = (
    "/mnt/nvme1n1/ml_research/linbinbin1/Kimi-K2-Instruct-0905"
)
DEFAULT_INPUT_CSV = (
    "/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/"
    "sharegpt-4o-mixlong-0326/"
    "sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv"
)
DEFAULT_ARTIFACT_ROOT = (
    "/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/"
    "offline_bench/manual_kimi_2node"
)
DEFAULT_RUN_LABEL = "issue01-random-kimi-2node-dp2tp8dcp8-rate5"
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
    "CUDA_LAUNCH_BLOCKING",
)
DEFAULT_ENV = {
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_DEEP_GEMM_WARMUP": "skip",
    "VLLM_LOG_STATS_INTERVAL": "1",
    "VLLM_MOE_ROUTING_SIMULATION_STRATEGY": "uniform_random",
    "VLLM_RANDOMIZE_DP_DUMMY_INPUTS": "1",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
}


@dataclass(frozen=True)
class ArtifactPaths:
    case_dir: Path
    benchmark_dir: Path
    frontend_log: Path
    headless_log: Path
    frontend_launch_log: Path
    headless_launch_log: Path
    frontend_pid: Path
    headless_pid: Path
    frontend_command: Path
    headless_command: Path
    manifest: Path


@dataclass
class RemoteProcess:
    host: str
    role: str
    process: subprocess.Popen[str]
    launch_log_handle: TextIO
    launch_log_path: Path
    runtime_log_path: Path
    pid_path: Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected an integer greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a number greater than zero")
    return parsed


def sanitize_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower() or "run"


def current_run_tag() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def stringify_number(value: float) -> str:
    return str(int(value)) if math.isclose(value, round(value)) else f"{value:g}"


def shell_join(argv: list[str] | tuple[str, ...]) -> str:
    return shlex.join(list(argv))


def build_ssh_argv(host: str, remote_command: str) -> list[str]:
    remote_shell = shell_join(("zsh", "-lc", remote_command))
    return ["ssh", *DEFAULT_SSH_OPTS, host, remote_shell]


def collect_runtime_env() -> dict[str, str]:
    runtime_env = {
        key: value
        for key in FORWARDED_ENV_KEYS
        if (value := os.environ.get(key))
    }
    runtime_env.update(DEFAULT_ENV)
    return runtime_env


def build_harness_argv(
    args: argparse.Namespace,
    *,
    role: str,
    node_rank: int,
    master_addr: str,
    output_dir: Path,
    max_requests: int,
) -> list[str]:
    argv = [
        "python3",
        HARNESS_ENTRYPOINT,
        role,
        "--model",
        args.model,
        "--master-addr",
        master_addr,
        "--master-port",
        str(args.master_port),
        "--nnodes",
        "2",
        "--node-rank",
        str(node_rank),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--data-parallel-size-local",
        str(args.data_parallel_size_local),
        "--data-parallel-rpc-port",
        str(args.data_parallel_rpc_port),
        "--data-parallel-backend",
        "mp",
        "--data-parallel-dispatch-policy",
        "waiting_x4_plus_running",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--decode-context-parallel-size",
        str(args.decode_context_parallel_size),
        "--enable-expert-parallel",
        "--attention-backend",
        "FLASHMLA",
        "--all2all-backend",
        "deepep_low_latency",
        "--gpu-memory-utilization",
        f"{args.gpu_memory_utilization:g}",
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(args.max_model_len),
        "--no-enable-prefix-caching",
        "--trust-remote-code",
    ]
    if args.decode_context_parallel_size > 1:
        argv.extend(["--dcp-comm-backend", "a2a"])
    if role == "frontend":
        argv.extend(
            [
                "--input-csv",
                args.input_csv,
                "--request-rate",
                f"{args.request_rate:g}",
                "--warmup-requests",
                str(args.warmup_requests),
                "--output-dir",
                str(output_dir),
                "--progress-log-interval",
                str(args.progress_log_interval),
                "--seed",
                str(args.seed),
                "--max-requests",
                str(max_requests),
            ]
        )
    return argv


def build_remote_runtime_command(
    *,
    argv: list[str],
    env: Mapping[str, str],
    env_script: str | None,
    runtime_log: Path,
    pid_path: Path,
    benchmark_dir: Path,
) -> str:
    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in env.items()
    )
    command = shell_join(argv)
    if env_prefix:
        command = f"env {env_prefix} {command}"

    steps = [
        "set -e",
        f"cd {shlex.quote(DEFAULT_WORKDIR)}",
        f"mkdir -p {shlex.quote(str(benchmark_dir))}",
        f"rm -f {shlex.quote(str(pid_path))}",
    ]
    if env_script:
        quoted_script = shlex.quote(env_script)
        # Interactive shell setup (for example, Oh My Zsh's env_default)
        # may use nonzero return codes for non-error control flow. Do not let
        # errexit terminate the remote launcher while sourcing it.
        steps.extend(
            [
                "set +e",
                f"[[ ! -f {quoted_script} ]] || source {quoted_script}",
                "set -e",
            ]
        )
    steps.extend(
        [
            "set +e",
            f"setsid {command} > {shlex.quote(str(runtime_log))} 2>&1 &",
            "child_pid=$!",
            f"printf '%s\\n' \"$child_pid\" > {shlex.quote(str(pid_path))}",
            "wait \"$child_pid\"",
            "child_exit_code=$?",
            "exit \"$child_exit_code\"",
        ]
    )
    return "\n".join(steps)


def build_cleanup_command(pid_path: Path) -> str:
    quoted_pid_path = shlex.quote(str(pid_path))
    return "\n".join(
        [
            "set +e",
            f"if [[ -s {quoted_pid_path} ]]; then",
            f"  child_pid=$(cat {quoted_pid_path})",
            "  if [[ \"$child_pid\" == <-> ]] && "
            "kill -0 -- \"-$child_pid\" >/dev/null 2>&1; then",
            "    kill -TERM -- \"-$child_pid\" >/dev/null 2>&1 || true",
            "    for _ in {1..20}; do",
            "      kill -0 -- \"-$child_pid\" >/dev/null 2>&1 || break",
            "      sleep 0.5",
            "    done",
            "    kill -KILL -- \"-$child_pid\" >/dev/null 2>&1 || true",
            "  fi",
            "fi",
        ]
    )


def prepare_artifacts(args: argparse.Namespace) -> ArtifactPaths:
    rate = stringify_number(args.request_rate)
    duration = stringify_number(args.bench_duration_sec)
    mem = stringify_number(args.gpu_memory_utilization * 100).replace(".", "_")
    group_dir = (
        Path(args.artifact_root)
        / "KIMI"
        / "issue01_random"
        / (
            f"dp{args.data_parallel_size}tp{args.tensor_parallel_size}"
            f"dcp{args.decode_context_parallel_size}-mem{mem}-"
            f"bs{args.max_num_seqs}-"
            f"rate{rate}-dur{duration}"
        )
    )
    case_dir = group_dir / (
        f"{current_run_tag()}__{sanitize_tag(args.run_label)}"
    )
    benchmark_dir = case_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=False)
    return ArtifactPaths(
        case_dir=case_dir,
        benchmark_dir=benchmark_dir,
        frontend_log=case_dir / "frontend.log",
        headless_log=case_dir / "rank1.log",
        frontend_launch_log=case_dir / "frontend.launch.log",
        headless_launch_log=case_dir / "rank1.launch.log",
        frontend_pid=case_dir / "frontend.pid",
        headless_pid=case_dir / "rank1.pid",
        frontend_command=case_dir / "frontend.command.sh",
        headless_command=case_dir / "rank1.command.sh",
        manifest=case_dir / "case_manifest.json",
    )


def write_shell_script(path: Path, command: str) -> None:
    path.write_text(
        "#!/usr/bin/env zsh\nset -euo pipefail\n\n" + command + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    master_addr: str,
    max_requests: int,
    status: str,
    commands: Mapping[str, str],
    detail: str | None = None,
) -> None:
    payload = {
        "status": status,
        "detail": detail,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "frontend_host": args.frontend_host,
        "headless_host": args.headless_host,
        "master_addr": master_addr,
        "model": args.model,
        "input_csv": args.input_csv,
        "request_rate": args.request_rate,
        "max_requests": max_requests,
        "bench_duration_sec": args.bench_duration_sec,
        "parallelism": {
            "dp": args.data_parallel_size,
            "tp": args.tensor_parallel_size,
            "dcp": args.decode_context_parallel_size,
        },
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "commands": dict(commands),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def launch_remote(
    *,
    host: str,
    role: str,
    command: str,
    launch_log_path: Path,
    runtime_log_path: Path,
    pid_path: Path,
) -> RemoteProcess:
    launch_log_handle = launch_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        build_ssh_argv(host, command),
        stdin=subprocess.DEVNULL,
        stdout=launch_log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return RemoteProcess(
        host=host,
        role=role,
        process=process,
        launch_log_handle=launch_log_handle,
        launch_log_path=launch_log_path,
        runtime_log_path=runtime_log_path,
        pid_path=pid_path,
    )


def tail_file(path: Path, lines: int = 60) -> str:
    if not path.is_file():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def stop_remote(remote: RemoteProcess) -> None:
    subprocess.run(
        build_ssh_argv(remote.host, build_cleanup_command(remote.pid_path)),
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    try:
        remote.process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(remote.process.pid, signal.SIGTERM)
        try:
            remote.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(remote.process.pid, signal.SIGKILL)
            remote.process.wait(timeout=5)
    remote.launch_log_handle.close()


def wait_for_startup(remote: RemoteProcess, grace_sec: float) -> None:
    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        exit_code = remote.process.poll()
        if exit_code is not None:
            detail = tail_file(remote.runtime_log_path) or tail_file(
                remote.launch_log_path
            )
            raise RuntimeError(
                f"{remote.role} on {remote.host} exited during startup "
                f"with code {exit_code}\n{detail}"
            )
        time.sleep(0.5)


def wait_for_benchmark(
    frontend: RemoteProcess,
    headless: RemoteProcess,
    timeout_sec: float,
) -> None:
    deadline = time.monotonic() + timeout_sec
    next_status = time.monotonic() + 30
    while True:
        frontend_exit = frontend.process.poll()
        if frontend_exit is not None:
            if frontend_exit != 0:
                detail = tail_file(frontend.runtime_log_path) or tail_file(
                    frontend.launch_log_path
                )
                raise RuntimeError(
                    f"frontend on {frontend.host} exited with code "
                    f"{frontend_exit}\n{detail}"
                )
            return

        headless_exit = headless.process.poll()
        if headless_exit is not None and headless_exit != 0:
            detail = tail_file(headless.runtime_log_path) or tail_file(
                headless.launch_log_path
            )
            raise RuntimeError(
                f"headless engine on {headless.host} exited early with code "
                f"{headless_exit}\n{detail}"
            )

        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(
                f"frontend exceeded --timeout-sec={stringify_number(timeout_sec)}"
            )
        if now >= next_status:
            remaining = max(0, deadline - now)
            print(f"[wait] benchmark is running; timeout in about {remaining:.0f}s")
            next_status = now + 30
        time.sleep(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run KIMI issue-1% at rate=5 on two H200 nodes."
        )
    )
    parser.add_argument("--frontend-host", default="h200-rjob0")
    parser.add_argument("--headless-host", default="h200-rjob2")
    parser.add_argument(
        "--master-addr",
        default="10.102.252.174",
        help="DP master IPv4 address on the frontend host.",
    )
    parser.add_argument("--master-port", type=positive_int, default=29579)
    parser.add_argument(
        "--data-parallel-rpc-port", type=positive_int, default=29550
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--request-rate", type=positive_float, default=5.0)
    parser.add_argument(
        "--bench-duration-sec", type=positive_float, default=600.0
    )
    parser.add_argument(
        "--max-requests",
        type=positive_int,
        default=None,
        help="Default: round(request-rate * bench-duration-sec), i.e. 3000.",
    )
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument("--data-parallel-size", type=positive_int, default=2)
    parser.add_argument(
        "--data-parallel-size-local", type=positive_int, default=1
    )
    parser.add_argument("--tensor-parallel-size", type=positive_int, default=8)
    parser.add_argument(
        "--decode-context-parallel-size", type=positive_int, default=8
    )
    parser.add_argument("--max-num-seqs", type=positive_int, default=1024)
    parser.add_argument("--max-model-len", type=positive_int, default=1_000_000)
    parser.add_argument(
        "--gpu-memory-utilization", type=positive_float, default=0.85
    )
    parser.add_argument("--progress-log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--startup-grace-sec", type=positive_float, default=8.0)
    parser.add_argument("--timeout-sec", type=positive_float, default=3600.0)
    parser.add_argument("--env-script", default="/root/.zshrc")
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write commands and manifest without SSH or benchmark execution.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frontend_host == args.headless_host:
        raise SystemExit("frontend and headless hosts must be different")
    if args.warmup_requests < 0:
        raise SystemExit("--warmup-requests must be >= 0")
    if args.progress_log_interval < 0:
        raise SystemExit("--progress-log-interval must be >= 0")
    if args.gpu_memory_utilization > 1:
        raise SystemExit("--gpu-memory-utilization must be <= 1")
    if args.data_parallel_size != 2 * args.data_parallel_size_local:
        raise SystemExit(
            "two-node runs require --data-parallel-size to equal "
            "2 * --data-parallel-size-local"
        )
    if args.data_parallel_size_local * args.tensor_parallel_size != 8:
        raise SystemExit(
            "each node has 8 GPUs, so --data-parallel-size-local * "
            "--tensor-parallel-size must equal 8"
        )
    if args.tensor_parallel_size % args.decode_context_parallel_size != 0:
        raise SystemExit(
            "--tensor-parallel-size must be divisible by "
            "--decode-context-parallel-size"
        )

    max_requests = args.max_requests or max(
        1, int(round(args.request_rate * args.bench_duration_sec))
    )
    master_addr = args.master_addr

    if not args.dry_run:
        if not Path(args.model).exists():
            raise SystemExit(f"model path not found: {args.model}")
        if not Path(args.input_csv).is_file():
            raise SystemExit(f"input CSV not found: {args.input_csv}")

    artifacts = prepare_artifacts(args)
    runtime_env = collect_runtime_env()
    frontend_argv = build_harness_argv(
        args,
        role="frontend",
        node_rank=0,
        master_addr=master_addr,
        output_dir=artifacts.benchmark_dir,
        max_requests=max_requests,
    )
    headless_argv = build_harness_argv(
        args,
        role="headless-engine",
        node_rank=1,
        master_addr=master_addr,
        output_dir=artifacts.benchmark_dir,
        max_requests=max_requests,
    )
    frontend_command = build_remote_runtime_command(
        argv=frontend_argv,
        env=runtime_env,
        env_script=args.env_script or None,
        runtime_log=artifacts.frontend_log,
        pid_path=artifacts.frontend_pid,
        benchmark_dir=artifacts.benchmark_dir,
    )
    headless_command = build_remote_runtime_command(
        argv=headless_argv,
        env=runtime_env,
        env_script=args.env_script or None,
        runtime_log=artifacts.headless_log,
        pid_path=artifacts.headless_pid,
        benchmark_dir=artifacts.benchmark_dir,
    )
    commands = {
        "frontend": shell_join(build_ssh_argv(args.frontend_host, frontend_command)),
        "headless": shell_join(build_ssh_argv(args.headless_host, headless_command)),
    }
    write_shell_script(artifacts.frontend_command, frontend_command)
    write_shell_script(artifacts.headless_command, headless_command)
    write_manifest(
        artifacts.manifest,
        args=args,
        master_addr=master_addr,
        max_requests=max_requests,
        status="dry_run" if args.dry_run else "prepared",
        commands=commands,
    )

    print(f"[config] nodes: {args.frontend_host}, {args.headless_host}")
    print(f"[config] master_addr: {master_addr}")
    print(
        f"[config] parallelism: DP{args.data_parallel_size} / "
        f"TP{args.tensor_parallel_size} / "
        f"DCP{args.decode_context_parallel_size}"
    )
    print(
        f"[config] issue01_random rate={args.request_rate:g}, "
        f"max_requests={max_requests}"
    )
    print(f"[artifacts] {artifacts.case_dir}")
    if args.dry_run:
        print("[dry-run] commands written; no SSH command was executed")
        return 0

    remotes: list[RemoteProcess] = []
    status = "failed"
    detail: str | None = None
    try:
        print(f"[launch] headless rank 1 on {args.headless_host}")
        headless = launch_remote(
            host=args.headless_host,
            role="headless",
            command=headless_command,
            launch_log_path=artifacts.headless_launch_log,
            runtime_log_path=artifacts.headless_log,
            pid_path=artifacts.headless_pid,
        )
        remotes.append(headless)
        wait_for_startup(headless, args.startup_grace_sec)

        print(f"[launch] frontend rank 0 on {args.frontend_host}")
        frontend = launch_remote(
            host=args.frontend_host,
            role="frontend",
            command=frontend_command,
            launch_log_path=artifacts.frontend_launch_log,
            runtime_log_path=artifacts.frontend_log,
            pid_path=artifacts.frontend_pid,
        )
        remotes.append(frontend)
        write_manifest(
            artifacts.manifest,
            args=args,
            master_addr=master_addr,
            max_requests=max_requests,
            status="running",
            commands=commands,
        )
        wait_for_benchmark(frontend, headless, args.timeout_sec)
        summary_path = artifacts.benchmark_dir / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(
                f"frontend exited successfully but missing {summary_path}"
            )
        status = "ok"
        print(f"[done] summary: {summary_path}")
        return 0
    except KeyboardInterrupt:
        detail = "interrupted; cleaning up both nodes"
        print(f"[error] {detail}", file=sys.stderr)
        return 130
    except Exception as exc:
        detail = str(exc)
        print(f"[error] {detail}", file=sys.stderr)
        return 1
    finally:
        for remote in reversed(remotes):
            try:
                stop_remote(remote)
            except Exception as exc:
                print(
                    f"[cleanup] failed on {remote.host}: {exc}", file=sys.stderr
                )
        write_manifest(
            artifacts.manifest,
            args=args,
            master_addr=master_addr,
            max_requests=max_requests,
            status=status,
            commands=commands,
            detail=detail,
        )


if __name__ == "__main__":
    raise SystemExit(main())
