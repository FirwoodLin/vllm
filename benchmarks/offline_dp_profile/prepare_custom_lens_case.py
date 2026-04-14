#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Literal

BENCHMARKS_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from offline_profile_strategy_defaults import (  # noqa: E402
    get_strategy_profile_defaults,
)

DISPATCH_POLICIES = (
    "waiting_x4_plus_running",
    "least_cache",
    "least_batch",
)
DEFAULT_DISPATCH_POLICY = "waiting_x4_plus_running"
DEFAULT_LENGTH_CSV_STEM = "custom_lens"
DEFAULT_CASE_CSV_NAME = "custom_lens.casecsv"
MAX_REQUESTS_CSV_ROWS = "csv_rows"
MaxRequestsValue = int | Literal["csv_rows"]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer.")
    return parsed


def max_requests_value(value: str) -> MaxRequestsValue:
    lowered = value.strip().lower()
    if lowered in {"csv", "csv_rows", "all_csv_rows"}:
        return MAX_REQUESTS_CSV_ROWS
    return non_negative_int(value)


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("Expected a positive float.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an offline_dp_profile case from either a nested "
            "input-length JSON file or a synthetic fixed-length workload by "
            "generating a prompt_len/output_len CSV and a derived casecsv. "
            "Strategies with no built-in offline-profile defaults must pass "
            "--max-num-seqs and --gpu-memory-utilization explicitly."
        ))
    parser.add_argument(
        "--base-case-csv",
        required=True,
        help="Template casecsv to copy and override.",
    )
    input_source_group = parser.add_mutually_exclusive_group(required=True)
    input_source_group.add_argument(
        "--lens-json",
        default=None,
        help="JSON file containing integers or nested integer lists.",
    )
    input_source_group.add_argument(
        "--uniform-prompt-len",
        type=positive_int,
        default=None,
        help="Generate a synthetic workload with this prompt_len on every row.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the generated CSV and casecsv will be written.",
    )
    parser.add_argument(
        "--output-len",
        type=positive_int,
        default=64,
        help="output_len written to every generated CSV row.",
    )
    parser.add_argument(
        "--repeat-count",
        type=positive_int,
        default=None,
        help=(
            "When --uniform-prompt-len is used, generate this many CSV rows "
            "with the same prompt_len."
        ),
    )
    parser.add_argument(
        "--cluster",
        default=None,
        help="Override cluster in the derived casecsv.",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help=(
            "Override strategy in the derived casecsv. If the strategy has no "
            "built-in offline-profile defaults, also pass --max-num-seqs and "
            "--gpu-memory-utilization."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model in the derived casecsv.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=non_negative_int,
        default=None,
        help="Override warmup_requests in the derived casecsv.",
    )
    parser.add_argument(
        "--max-requests",
        type=max_requests_value,
        default=None,
        help=(
            "Override max_requests in the derived casecsv. Use "
            f"'{MAX_REQUESTS_CSV_ROWS}' to make measured requests follow the "
            "CSV row count."
        ),
    )
    parser.add_argument(
        "--request-rate",
        type=positive_float,
        default=None,
        help="Override request_rate in the derived casecsv.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=positive_int,
        default=None,
        help=(
            "Override max_num_seqs in the derived casecsv. Required for "
            "strategies without built-in offline-profile defaults."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=positive_float,
        default=None,
        help=(
            "Override gpu_memory_utilization in the derived casecsv. Required "
            "for strategies without built-in offline-profile defaults."
        ),
    )
    parser.add_argument(
        "--data-parallel-rpc-port",
        type=positive_int,
        default=None,
        help="Override data_parallel_rpc_port in the derived casecsv.",
    )
    parser.add_argument(
        "--dispatch-policy",
        choices=DISPATCH_POLICIES,
        default=None,
        help="Override dispatch_policy in the derived casecsv.",
    )
    parser.add_argument(
        "--case-name",
        default=None,
        help=(
            "Optional derived case name. Defaults to "
            "<base_name>__<strategy>__<lens_stem>[__dispatch_<policy>]."
        ),
    )
    args = parser.parse_args()

    if args.uniform_prompt_len is not None and args.repeat_count is None:
        parser.error("--repeat-count is required with --uniform-prompt-len.")
    if args.uniform_prompt_len is None and args.repeat_count is not None:
        parser.error("--repeat-count requires --uniform-prompt-len.")

    return args


def _flatten_nested_lengths(node: object) -> list[int]:
    values: list[int] = []

    def walk(item: object) -> None:
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if isinstance(item, int):
            if item <= 0:
                raise ValueError(f"All input lengths must be >= 1, got {item}")
            values.append(item)
            return
        raise TypeError(
            "Input JSON must contain only integers or nested lists, "
            f"got {type(item)!r}.")

    walk(node)
    if not values:
        raise ValueError("No valid input lengths found in --lens-json.")
    return values


def load_input_lengths(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _flatten_nested_lengths(payload)


def build_uniform_input_lengths(prompt_len: int, repeat_count: int) -> list[int]:
    return [prompt_len] * repeat_count


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


def read_base_case(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Case CSV has no header row: {path}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Case CSV has no data rows: {path}")

    for row in rows:
        enabled = (row.get("enabled") or "").strip().lower()
        if enabled not in {"0", "false", "no", "off"}:
            return list(reader.fieldnames), dict(row)

    return list(reader.fieldnames), dict(rows[0])


def write_length_csv(path: Path, input_lengths: list[int], output_len: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt_len", "output_len"])
        writer.writeheader()
        for prompt_len in input_lengths:
            writer.writerow({
                "prompt_len": str(prompt_len),
                "output_len": str(output_len),
            })


def write_case_csv(path: Path, fieldnames: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    base_case_csv = Path(args.base_case_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    fieldnames, base_row = read_base_case(base_case_csv)
    if args.lens_json is not None:
        lens_json = Path(args.lens_json).expanduser().resolve()
        input_lengths = load_input_lengths(lens_json)
        input_source_tag = lens_json.stem
    else:
        input_lengths = build_uniform_input_lengths(
            args.uniform_prompt_len,
            args.repeat_count,
        )
        input_source_tag = (
            f"uniform_prompt{args.uniform_prompt_len}"
            f"_x{args.repeat_count}"
        )

    derived_row = dict(base_row)
    effective_strategy = (
        args.strategy
        or (derived_row.get("strategy") or "").strip()
    )
    effective_dispatch_policy = (
        args.dispatch_policy
        or (derived_row.get("dispatch_policy") or "").strip()
        or "unknown_dispatch"
    )
    dispatch_tag = f"dispatch_{sanitize_tag(effective_dispatch_policy)}"
    length_csv_name = f"{DEFAULT_LENGTH_CSV_STEM}.{dispatch_tag}.lengths.csv"
    length_csv_path = output_dir / length_csv_name
    case_csv_path = output_dir / DEFAULT_CASE_CSV_NAME

    base_case_name = (derived_row.get("name") or "offline_profile_case").strip()
    derived_case_name_parts = [base_case_name]
    if effective_strategy:
        derived_case_name_parts.append(effective_strategy)
    derived_case_name_parts.append(input_source_tag)
    if effective_dispatch_policy != DEFAULT_DISPATCH_POLICY:
        derived_case_name_parts.append(dispatch_tag)
    derived_case_name = args.case_name or "__".join(derived_case_name_parts)

    derived_row["name"] = derived_case_name
    derived_row["dataset"] = str(length_csv_path)
    if args.cluster is not None:
        derived_row["cluster"] = args.cluster
    if args.strategy is not None:
        derived_row["strategy"] = args.strategy
    if args.model is not None:
        derived_row["model"] = args.model
    if args.warmup_requests is not None:
        derived_row["warmup_requests"] = str(args.warmup_requests)
    if args.max_requests is not None:
        derived_row["max_requests"] = str(args.max_requests)
    if args.request_rate is not None:
        derived_row["request_rate"] = f"{args.request_rate:g}"
    if args.max_num_seqs is not None:
        derived_row["max_num_seqs"] = str(args.max_num_seqs)
    elif args.strategy is not None:
        derived_row["max_num_seqs"] = str(
            get_strategy_profile_defaults(effective_strategy).max_num_seqs)
    if args.gpu_memory_utilization is not None:
        derived_row["gpu_memory_utilization"] = (
            f"{args.gpu_memory_utilization:g}"
        )
    elif args.strategy is not None:
        derived_row["gpu_memory_utilization"] = (
            f"{get_strategy_profile_defaults(effective_strategy).gpu_memory_utilization:g}"
        )
    if args.data_parallel_rpc_port is not None:
        derived_row["data_parallel_rpc_port"] = str(args.data_parallel_rpc_port)
    if args.dispatch_policy is not None:
        derived_row["dispatch_policy"] = args.dispatch_policy

    write_length_csv(length_csv_path, input_lengths, args.output_len)
    write_case_csv(case_csv_path, fieldnames, derived_row)

    print("Prepared offline profile inputs")
    print(f"  base_case_csv: {base_case_csv}")
    if args.lens_json is not None:
        print(f"  lens_json: {lens_json}")
    else:
        print(f"  uniform_prompt_len: {args.uniform_prompt_len}")
        print(f"  repeat_count: {args.repeat_count}")
    print(f"  flattened_lengths: {len(input_lengths)}")
    print(f"  output_len: {args.output_len}")
    print(f"  length_csv: {length_csv_path}")
    print(f"  case_csv: {case_csv_path}")
    print(f"  case_name: {derived_row['name']}")
    print(f"  cluster: {derived_row.get('cluster', '')}")
    print(f"  strategy: {derived_row.get('strategy', '')}")
    print(f"  model: {derived_row.get('model', '')}")
    print(f"  dispatch_policy: {derived_row.get('dispatch_policy', '')}")
    print(f"  request_rate: {derived_row.get('request_rate', '')}")
    print(f"  warmup_requests: {derived_row.get('warmup_requests', '')}")
    print(f"  max_requests: {derived_row.get('max_requests', '')}")
    print(f"  max_num_seqs: {derived_row.get('max_num_seqs', '')}")
    print(
        "  gpu_memory_utilization: "
        f"{derived_row.get('gpu_memory_utilization', '')}"
    )
    print(
        "  data_parallel_rpc_port: "
        f"{derived_row.get('data_parallel_rpc_port', '')}"
    )


if __name__ == "__main__":
    main()
