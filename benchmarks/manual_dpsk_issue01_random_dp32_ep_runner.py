#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch DPSK issue01 random DP32+EP manual multinode experiments.

This is a small wrapper around manual_multinode_poisson_runner.py. It keeps the
requested experiment set fixed and delegates launch, cleanup, manifests,
historical skip logic, and output augmentation to the shared runner.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
RUNNER_PATH = BENCHMARKS_DIR / "manual_multinode_poisson_runner.py"
MODEL_ALIAS = "deepseek_v3_1024k"
MODEL_PATH = Path("/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3")
DEFAULT_MASTER_ADDR = "10.102.237.21"
DEFAULT_ARTIFACT_ROOT = Path(
    "/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/"
    "offline_bench/manual_multinode"
)
DEFAULT_CASE_CSV_NAME = "dpsk_issue01_random_dp32_ep_bs196_rate30_60_cases.csv"
DEFAULT_BENCH_DURATION_SEC = 600.0
DEFAULT_RUN_LABEL = "dpsk-issue01-random-dp32-ep-bs196-rate30-60"
DEFAULT_CLUSTER = "4node_h200"
DEFAULT_DATASET = "issue01_random"
DEFAULT_STRATEGY = "dp32"
DEFAULT_DISPATCH_POLICY = "waiting_x4_plus_running"
DEFAULT_REQUEST_RATES = (30.0, 60.0)
DEFAULT_MAX_NUM_SEQS = 196
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_WARMUP_REQUESTS = 32
DEFAULT_MAX_MODEL_LEN = 1000000
DEFAULT_DATA_PARALLEL_RPC_PORT = 29550
CSV_FIELDNAMES = (
    "enabled",
    "name",
    "cluster",
    "model",
    "dataset",
    "strategy",
    "dispatch_policy",
    "request_rate",
    "rate_phase",
    "max_num_seqs",
    "gpu_memory_utilization",
    "max_requests",
    "warmup_requests",
    "max_model_len",
    "data_parallel_rpc_port",
    "reason",
    "historical_reference",
)


@dataclass(frozen=True)
class CaseRow:
    enabled: int
    name: str
    cluster: str
    model: str
    dataset: str
    strategy: str
    dispatch_policy: str
    request_rate: float
    rate_phase: str
    max_num_seqs: int
    gpu_memory_utilization: float
    max_requests: int
    warmup_requests: int
    max_model_len: int
    data_parallel_rpc_port: int
    reason: str = ""
    historical_reference: str = ""


def stringify_rate(value: float) -> str:
    return f"{value:g}"


def max_requests_for_rate(rate: float,
                          bench_duration_sec: float =
                          DEFAULT_BENCH_DURATION_SEC) -> int:
    return max(1, int(round(rate * bench_duration_sec)))


def build_case_rows(
    *,
    bench_duration_sec: float = DEFAULT_BENCH_DURATION_SEC,
) -> list[CaseRow]:
    rows: list[CaseRow] = []
    for rate in DEFAULT_REQUEST_RATES:
        rows.append(
            CaseRow(
                enabled=1,
                name=(
                    "DPSK__"
                    f"{DEFAULT_DATASET}__dp32_ep__"
                    f"rate{stringify_rate(rate)}__bs{DEFAULT_MAX_NUM_SEQS}"
                ),
                cluster=DEFAULT_CLUSTER,
                model=MODEL_ALIAS,
                dataset=DEFAULT_DATASET,
                strategy=DEFAULT_STRATEGY,
                dispatch_policy=DEFAULT_DISPATCH_POLICY,
                request_rate=rate,
                rate_phase="issue01_random_rate30_60",
                max_num_seqs=DEFAULT_MAX_NUM_SEQS,
                gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
                max_requests=max_requests_for_rate(rate, bench_duration_sec),
                warmup_requests=DEFAULT_WARMUP_REQUESTS,
                max_model_len=DEFAULT_MAX_MODEL_LEN,
                data_parallel_rpc_port=DEFAULT_DATA_PARALLEL_RPC_PORT,
                reason="DPSK issue01_random DP32+EP mem90 bs196 rate30_60",
            ))
    return rows


def write_case_csv(path: Path, rows: list[CaseRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["request_rate"] = stringify_rate(row.request_rate)
            payload["gpu_memory_utilization"] = (
                f"{row.gpu_memory_utilization:g}")
            writer.writerow(payload)


def build_runner_argv(
    *,
    case_csv: Path,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    run_label: str = DEFAULT_RUN_LABEL,
    dry_run: bool = False,
    ignore_historical_skips: bool = False,
    no_keep_going: bool = False,
) -> list[str]:
    argv = [
        sys.executable,
        str(RUNNER_PATH),
        "--case-csv",
        str(case_csv),
        "--artifact-root",
        str(artifact_root),
        "--run-label",
        run_label,
    ]
    if dry_run:
        argv.append("--dry-run")
    if ignore_historical_skips:
        argv.append("--ignore-historical-skips")
    if no_keep_going:
        argv.append("--no-keep-going")
    return argv


def apply_runner_overrides(runner_module: object, *, model_path: Path,
                           master_addr: str) -> None:
    models = getattr(runner_module, "MODELS")
    models[MODEL_ALIAS] = str(model_path)

    clusters = getattr(runner_module, "CLUSTERS")
    clusters[DEFAULT_CLUSTER] = replace(
        clusters[DEFAULT_CLUSTER],
        master_addr=master_addr,
    )


def run_manual_runner(runner_args: list[str], *, model_path: Path,
                      master_addr: str) -> int:
    if str(BENCHMARKS_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS_DIR))

    os.environ["VLLM_4NODE_H200_MASTER_ADDR"] = master_addr

    import manual_multinode_poisson_runner as runner

    apply_runner_overrides(
        runner,
        model_path=model_path,
        master_addr=master_addr,
    )
    try:
        runner.main(runner_args)
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(exc.code, file=sys.stderr)
        return 1
    return 0


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("Expected a positive float.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DPSK issue01_random DP32+EP mem90 bs196 experiments.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help=(
            "Shared artifact root passed to manual_multinode_poisson_runner.py. "
            "The default places outputs under <root>/DPSK/issue01_random."
        ),
    )
    parser.add_argument(
        "--run-label",
        default=DEFAULT_RUN_LABEL,
        help="Run label suffix passed to manual_multinode_poisson_runner.py.",
    )
    parser.add_argument(
        "--case-csv-output",
        type=Path,
        default=None,
        help=(
            "Where to write the generated case CSV. Defaults to "
            "<artifact-root>/_case_csv/"
            "dpsk_issue01_random_dp32_ep_bs196_cases.csv."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=(
            "DeepSeek model path to bind to the deepseek_v3_1024k alias before "
            "calling the underlying manual runner."
        ),
    )
    parser.add_argument(
        "--master-addr",
        default=os.environ.get("VLLM_4NODE_H200_MASTER_ADDR",
                               DEFAULT_MASTER_ADDR),
        help=(
            "Master address for the 4node_h200 cluster. Defaults to "
            "VLLM_4NODE_H200_MASTER_ADDR or 10.102.237.21."
        ),
    )
    parser.add_argument(
        "--bench-duration-sec",
        type=positive_float,
        default=DEFAULT_BENCH_DURATION_SEC,
        help="Measured benchmark duration used to compute max_requests.",
    )
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Write commands/manifests without execution.")
    parser.add_argument(
        "--ignore-historical-skips",
        action="store_true",
        help="Pass through to the underlying manual runner.",
    )
    parser.add_argument(
        "--no-keep-going",
        action="store_true",
        help="Stop after the first failed case.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    case_csv = args.case_csv_output
    if case_csv is None:
        case_csv = (args.artifact_root / "_case_csv" /
                    DEFAULT_CASE_CSV_NAME)

    write_case_csv(
        case_csv,
        build_case_rows(bench_duration_sec=args.bench_duration_sec),
    )
    runner_argv = build_runner_argv(
        case_csv=case_csv,
        artifact_root=args.artifact_root,
        run_label=args.run_label,
        dry_run=args.dry_run,
        ignore_historical_skips=args.ignore_historical_skips,
        no_keep_going=args.no_keep_going,
    )
    print("Generated case CSV:", case_csv, flush=True)
    print("Expected dataset artifact dir:",
          args.artifact_root / "DPSK" / DEFAULT_DATASET,
          flush=True)
    print(f"Using {MODEL_ALIAS} model path:", args.model_path, flush=True)
    print("Using 4node_h200 master addr:", args.master_addr, flush=True)
    print("Launching:", " ".join(runner_argv), flush=True)
    return run_manual_runner(
        runner_argv[2:],
        model_path=args.model_path,
        master_addr=args.master_addr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
