#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch DPSK DBO pure-DP least-batch issue01 manual multinode experiments.

This is a small wrapper around manual_multinode_poisson_runner.py. It keeps the
experiment set fixed and delegates launch, cleanup, manifests, historical skip
logic, and output augmentation to the shared runner.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
RUNNER_PATH = BENCHMARKS_DIR / "manual_multinode_poisson_runner.py"
MODEL_ALIAS = "deepseek_v3_1024k"
MODEL_PATH = Path("/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3")
DEFAULT_ARTIFACT_ROOT = Path(
    "/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/"
    "offline_bench/manual_multinode/DPSK-dbo"
)
DEFAULT_CASE_CSV_NAME = "dpsk_dbo_least_batch_rate30_cases.csv"
DEFAULT_BENCH_DURATION_SEC = 600.0
DEFAULT_RUN_LABEL = "dpsk-dbo-least-batch-rate30"
DEFAULT_DATASET = "issue01_random"
DEFAULT_REQUEST_RATES = (30.0, )
DEFAULT_DBO_DECODE_TOKEN_THRESHOLD = 2
DEFAULT_MAX_NUM_SEQS = 192
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


def stringify_rates(values: tuple[float, ...]) -> str:
    return "_".join(stringify_rate(value) for value in values)


def build_default_case_csv_name(
    dataset: str,
    request_rates: tuple[float, ...],
) -> str:
    if dataset == DEFAULT_DATASET and request_rates == DEFAULT_REQUEST_RATES:
        return DEFAULT_CASE_CSV_NAME
    return (
        f"dpsk_dbo_least_batch_{dataset}_rate"
        f"{stringify_rates(request_rates)}_cases.csv"
    )


def build_default_run_label(
    dataset: str,
    request_rates: tuple[float, ...],
) -> str:
    if dataset == DEFAULT_DATASET and request_rates == DEFAULT_REQUEST_RATES:
        return DEFAULT_RUN_LABEL
    return f"dpsk-dbo-least-batch-{dataset}-rate{stringify_rates(request_rates)}"


def max_requests_for_rate(rate: float,
                          bench_duration_sec: float =
                          DEFAULT_BENCH_DURATION_SEC) -> int:
    return max(1, int(round(rate * bench_duration_sec)))


def build_case_rows(
    *,
    bench_duration_sec: float = DEFAULT_BENCH_DURATION_SEC,
    dataset: str = DEFAULT_DATASET,
    request_rates: tuple[float, ...] = DEFAULT_REQUEST_RATES,
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
) -> list[CaseRow]:
    rows: list[CaseRow] = []
    rate_phase = (
        "dpsk_dbo_least_batch_"
        f"{dataset}_rate{stringify_rates(request_rates)}"
    )
    for rate in request_rates:
        rows.append(
            CaseRow(
                enabled=1,
                name=(
                    "DPSK_DBO__"
                    f"{dataset}__dp32_least_batch__"
                    f"rate{stringify_rate(rate)}__bs{max_num_seqs}"
                ),
                cluster="4node_h200",
                model=MODEL_ALIAS,
                dataset=dataset,
                strategy="dp32",
                dispatch_policy="least_batch",
                request_rate=rate,
                rate_phase=rate_phase,
                max_num_seqs=max_num_seqs,
                gpu_memory_utilization=0.9,
                max_requests=max_requests_for_rate(rate, bench_duration_sec),
                warmup_requests=32,
                max_model_len=1000000,
                data_parallel_rpc_port=29550,
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
    dbo_decode_token_threshold: int = DEFAULT_DBO_DECODE_TOKEN_THRESHOLD,
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
        "--frontend-extra-arg=--enable-dbo",
        "--frontend-extra-arg=--dbo-decode-token-threshold",
        f"--frontend-extra-arg={dbo_decode_token_threshold}",
        "--headless-extra-arg=--enable-dbo",
        "--headless-extra-arg=--dbo-decode-token-threshold",
        f"--headless-extra-arg={dbo_decode_token_threshold}",
    ]
    if dry_run:
        argv.append("--dry-run")
    if ignore_historical_skips:
        argv.append("--ignore-historical-skips")
    if no_keep_going:
        argv.append("--no-keep-going")
    return argv


def apply_model_path_override(runner_module: object, model_path: Path) -> None:
    models = getattr(runner_module, "MODELS")
    models[MODEL_ALIAS] = str(model_path)


def run_manual_runner(runner_args: list[str], *, model_path: Path) -> int:
    if str(BENCHMARKS_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS_DIR))

    import manual_multinode_poisson_runner as runner

    apply_model_path_override(runner, model_path)
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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def parse_request_rates(value: str) -> tuple[float, ...]:
    rates: list[float] = []
    for raw_part in value.replace(",", " ").split():
        rates.append(positive_float(raw_part))
    if not rates:
        raise argparse.ArgumentTypeError("Expected at least one request rate.")
    return tuple(rates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run DPSK ShareGPT4o/Gemini pure DP+EP least_batch DBO experiments."
        ))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Artifact root passed to manual_multinode_poisson_runner.py.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help=(
            "Run label suffix passed to manual_multinode_poisson_runner.py. "
            "Defaults to a dataset/rate-specific label."
        ),
    )
    parser.add_argument(
        "--case-csv-output",
        type=Path,
        default=None,
        help=(
            "Where to write the generated case CSV. Defaults to "
            "<artifact-root>/_case_csv/dpsk_dbo_least_batch_cases.csv."
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
        "--bench-duration-sec",
        type=positive_float,
        default=DEFAULT_BENCH_DURATION_SEC,
        help="Measured benchmark duration used to compute max_requests.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset key to put in the generated case CSV.",
    )
    parser.add_argument(
        "--request-rates",
        type=parse_request_rates,
        default=DEFAULT_REQUEST_RATES,
        help=(
            "Comma- or space-separated request rates. "
            "Example: --request-rates 20,40,60,80,100,120"
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=positive_int,
        default=DEFAULT_MAX_NUM_SEQS,
        help="max_num_seqs value for generated cases.",
    )
    parser.add_argument(
        "--dbo-decode-token-threshold",
        type=positive_int,
        default=DEFAULT_DBO_DECODE_TOKEN_THRESHOLD,
        help="Value passed to --dbo-decode-token-threshold.",
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
                    build_default_case_csv_name(args.dataset,
                                                args.request_rates))
    run_label = args.run_label
    if run_label is None:
        run_label = build_default_run_label(args.dataset, args.request_rates)

    write_case_csv(
        case_csv,
        build_case_rows(
            bench_duration_sec=args.bench_duration_sec,
            dataset=args.dataset,
            request_rates=args.request_rates,
            max_num_seqs=args.max_num_seqs,
        ),
    )
    runner_argv = build_runner_argv(
        case_csv=case_csv,
        artifact_root=args.artifact_root,
        run_label=run_label,
        dbo_decode_token_threshold=args.dbo_decode_token_threshold,
        dry_run=args.dry_run,
        ignore_historical_skips=args.ignore_historical_skips,
        no_keep_going=args.no_keep_going,
    )
    print("Generated case CSV:", case_csv, flush=True)
    print(f"Using {MODEL_ALIAS} model path:", args.model_path, flush=True)
    print("Launching:", " ".join(runner_argv), flush=True)
    return run_manual_runner(runner_argv[2:], model_path=args.model_path)


if __name__ == "__main__":
    raise SystemExit(main())
