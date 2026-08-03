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
ROUTING_MODES = (
    "internal_dplb",
    "explicit_rank_replay",
)
DEFAULT_ROUTING_MODE = "internal_dplb"
DEFAULT_LENGTH_CSV_STEM = "custom_lens"
DEFAULT_CASE_CSV_NAME = "custom_lens.casecsv"
GREEDY_EXPLICIT_RANK_POLICIES = frozenset({"least_cache", "least_batch"})
DEFAULT_KV_CACHE_TOKENS_PER_RANK = 1_050_000
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
        "--max-model-len",
        type=positive_int,
        default=None,
        help="Override max_model_len in the derived casecsv.",
    )
    parser.add_argument(
        "--dispatch-policy",
        choices=DISPATCH_POLICIES,
        default=None,
        help="Override dispatch_policy in the derived casecsv.",
    )
    parser.add_argument(
        "--routing-mode",
        choices=ROUTING_MODES,
        default=DEFAULT_ROUTING_MODE,
        help=(
            "How the frontend should route requests. explicit_rank_replay "
            "writes a data_parallel_rank column into the generated CSV."
        ),
    )
    parser.add_argument(
        "--data-parallel-size",
        type=positive_int,
        default=None,
        help=(
            "Global DP size used when --routing-mode=explicit_rank_replay."
        ),
    )
    parser.add_argument(
        "--data-parallel-size-local",
        type=positive_int,
        default=None,
        help=(
            "Local DP size per node used when --routing-mode=explicit_rank_replay."
        ),
    )
    parser.add_argument(
        "--warmup-short-rows",
        type=non_negative_int,
        default=0,
        help=(
            "Reserve this many leading shortest-prompt rows for warmup before "
            "building the explicit rank replay sequence."
        ),
    )
    parser.add_argument(
        "--kv-cache-tokens-per-rank",
        type=positive_int,
        default=DEFAULT_KV_CACHE_TOKENS_PER_RANK,
        help=(
            "Single-rank KV cache token cap used by greedy explicit-rank "
            "pre-allocation for least_cache/least_batch."
        ),
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
    if args.routing_mode == "explicit_rank_replay":
        if args.data_parallel_size is None:
            parser.error(
                "--data-parallel-size is required with "
                "--routing-mode=explicit_rank_replay."
            )
        if args.data_parallel_size_local is None:
            parser.error(
                "--data-parallel-size-local is required with "
                "--routing-mode=explicit_rank_replay."
            )
        if args.data_parallel_size % args.data_parallel_size_local != 0:
            parser.error(
                "--data-parallel-size must be divisible by "
                "--data-parallel-size-local."
            )
    else:
        if args.data_parallel_size is not None:
            parser.error(
                "--data-parallel-size requires "
                "--routing-mode=explicit_rank_replay."
            )
        if args.data_parallel_size_local is not None:
            parser.error(
                "--data-parallel-size-local requires "
                "--routing-mode=explicit_rank_replay."
            )
        if args.warmup_short_rows != 0:
            parser.error(
                "--warmup-short-rows requires "
                "--routing-mode=explicit_rank_replay."
            )

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


def split_counts_evenly(total_count: int, buckets: int) -> list[int]:
    base, remainder = divmod(total_count, buckets)
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def evenly_insert_values(base: list[int], value: int, count: int) -> list[int]:
    if count <= 0:
        return list(base)

    total = len(base) + count
    positions = {(index * total) // count for index in range(count)}
    result: list[int] = []
    base_index = 0
    remaining = count

    for position in range(total):
        if position in positions and remaining > 0:
            result.append(value)
            remaining -= 1
            continue

        result.append(base[base_index])
        base_index += 1

    return result


def build_rank_prompt_sequence(prompt_counts: dict[int, int]) -> list[int]:
    if not prompt_counts:
        return []

    ordered_lengths = sorted(prompt_counts)
    shortest_prompt_len = ordered_lengths[0]
    sequence = [shortest_prompt_len] * prompt_counts[shortest_prompt_len]
    for prompt_len in reversed(ordered_lengths[1:]):
        sequence = evenly_insert_values(sequence, prompt_len,
                                        prompt_counts[prompt_len])
    return sequence


def build_legacy_explicit_rank_rows(
    *,
    input_lengths: list[int],
    output_len: int,
    data_parallel_size: int,
    data_parallel_size_local: int,
    warmup_short_rows: int,
) -> list[dict[str, str]]:
    if data_parallel_size <= 0:
        raise ValueError("data_parallel_size must be >= 1.")
    if data_parallel_size_local <= 0:
        raise ValueError("data_parallel_size_local must be >= 1.")
    if data_parallel_size % data_parallel_size_local != 0:
        raise ValueError(
            "data_parallel_size must be divisible by data_parallel_size_local."
        )

    num_nodes = data_parallel_size // data_parallel_size_local
    prompt_counts: dict[int, int] = {}
    for prompt_len in input_lengths:
        prompt_counts[prompt_len] = prompt_counts.get(prompt_len, 0) + 1

    per_rank_prompt_counts: list[dict[int, int]] = [
        {} for _ in range(data_parallel_size)
    ]
    for prompt_len, total_count in sorted(prompt_counts.items()):
        per_node_counts = split_counts_evenly(total_count, num_nodes)
        for node_index, node_count in enumerate(per_node_counts):
            per_local_rank_counts = split_counts_evenly(
                node_count, data_parallel_size_local)
            for local_rank, local_count in enumerate(per_local_rank_counts):
                if local_count == 0:
                    continue
                global_rank = node_index * data_parallel_size_local + local_rank
                per_rank_prompt_counts[global_rank][prompt_len] = local_count

    shortest_prompt_len = min(prompt_counts)
    rows: list[dict[str, str]] = []
    for warmup_index in range(warmup_short_rows):
        start_rank = warmup_index % data_parallel_size
        assigned_rank: int | None = None
        for offset in range(data_parallel_size):
            rank = (start_rank + offset) % data_parallel_size
            remaining = per_rank_prompt_counts[rank].get(shortest_prompt_len, 0)
            if remaining <= 0:
                continue
            per_rank_prompt_counts[rank][shortest_prompt_len] = remaining - 1
            if per_rank_prompt_counts[rank][shortest_prompt_len] == 0:
                del per_rank_prompt_counts[rank][shortest_prompt_len]
            assigned_rank = rank
            break

        if assigned_rank is None:
            raise ValueError(
                "warmup_short_rows exceeds the available count of shortest "
                "requests."
            )

        rows.append({
            "prompt_len": str(shortest_prompt_len),
            "output_len": str(output_len),
            "data_parallel_rank": str(assigned_rank),
        })

    per_rank_sequences = [
        build_rank_prompt_sequence(prompt_counts)
        for prompt_counts in per_rank_prompt_counts
    ]
    max_rank_sequence_len = max((len(sequence)
                                 for sequence in per_rank_sequences),
                                default=0)
    for round_index in range(max_rank_sequence_len):
        for rank, prompt_sequence in enumerate(per_rank_sequences):
            if round_index >= len(prompt_sequence):
                continue
            rows.append({
                "prompt_len": str(prompt_sequence[round_index]),
                "output_len": str(output_len),
                "data_parallel_rank": str(rank),
            })

    return rows


def extract_shortest_warmup_lengths(
    input_lengths: list[int],
    warmup_short_rows: int,
) -> tuple[list[int], list[int]]:
    if warmup_short_rows <= 0:
        return [], list(input_lengths)
    if warmup_short_rows > len(input_lengths):
        raise ValueError(
            "warmup_short_rows exceeds the available count of requests."
        )

    ranked_indices = sorted(
        range(len(input_lengths)),
        key=lambda index: (input_lengths[index], index),
    )
    warmup_indices = set(ranked_indices[:warmup_short_rows])
    warmup_lengths = [
        input_lengths[index]
        for index in ranked_indices[:warmup_short_rows]
    ]
    remaining_lengths = [
        prompt_len
        for index, prompt_len in enumerate(input_lengths)
        if index not in warmup_indices
    ]
    return warmup_lengths, remaining_lengths


def preallocate_dp_ranks(
    input_lengths: list[int],
    *,
    dp_size: int,
    output_len: int,
    dispatch_policy: str,
    kv_cache_tokens_per_rank: int,
    waiting_requests: list[int] | None = None,
    waiting_tokens: list[int] | None = None,
    free_kv_tokens: list[int] | None = None,
) -> tuple[list[int], list[int], list[int], list[int]]:
    if waiting_requests is None:
        waiting_requests = [0] * dp_size
    else:
        waiting_requests = waiting_requests.copy()
    if waiting_tokens is None:
        waiting_tokens = [0] * dp_size
    else:
        waiting_tokens = waiting_tokens.copy()
    if free_kv_tokens is None:
        free_kv_tokens = [kv_cache_tokens_per_rank] * dp_size
    else:
        free_kv_tokens = free_kv_tokens.copy()

    assigned_ranks: list[int] = []
    for input_len in input_lengths:
        kv_tokens_needed = input_len + output_len
        if dispatch_policy == "least_batch":
            ranks_with_capacity = [
                rank
                for rank in range(dp_size)
                if free_kv_tokens[rank] >= kv_tokens_needed
            ]
            if ranks_with_capacity:
                dp_rank = min(
                    ranks_with_capacity,
                    key=lambda rank: (waiting_requests[rank], rank),
                )
            else:
                dp_rank = min(
                    range(dp_size),
                    key=lambda rank: (
                        kv_tokens_needed - free_kv_tokens[rank],
                        waiting_requests[rank],
                        rank,
                    ),
                )
        elif dispatch_policy == "least_cache":
            dp_rank = min(
                range(dp_size),
                key=lambda rank: (
                    waiting_tokens[rank] - free_kv_tokens[rank],
                    waiting_requests[rank],
                    rank,
                ),
            )
        else:
            raise ValueError(f"Unsupported dispatch policy: {dispatch_policy}")

        assigned_ranks.append(dp_rank)
        waiting_requests[dp_rank] += 1
        waiting_tokens[dp_rank] += input_len
        free_kv_tokens[dp_rank] -= kv_tokens_needed

    return assigned_ranks, waiting_requests, waiting_tokens, free_kv_tokens


def build_greedy_explicit_rank_rows(
    *,
    input_lengths: list[int],
    output_len: int,
    data_parallel_size: int,
    dispatch_policy: str,
    kv_cache_tokens_per_rank: int,
    warmup_short_rows: int,
) -> list[dict[str, str]]:
    warmup_lengths, remaining_lengths = extract_shortest_warmup_lengths(
        input_lengths,
        warmup_short_rows,
    )
    warmup_ranks, waiting_requests, waiting_tokens, free_kv_tokens = (
        preallocate_dp_ranks(
            warmup_lengths,
            dp_size=data_parallel_size,
            output_len=output_len,
            dispatch_policy=dispatch_policy,
            kv_cache_tokens_per_rank=kv_cache_tokens_per_rank,
        )
    )
    measured_ranks, _, _, _ = preallocate_dp_ranks(
        remaining_lengths,
        dp_size=data_parallel_size,
        output_len=output_len,
        dispatch_policy=dispatch_policy,
        kv_cache_tokens_per_rank=kv_cache_tokens_per_rank,
        waiting_requests=waiting_requests,
        waiting_tokens=waiting_tokens,
        free_kv_tokens=free_kv_tokens,
    )

    rows: list[dict[str, str]] = []
    for prompt_len, rank in zip(warmup_lengths, warmup_ranks):
        rows.append({
            "prompt_len": str(prompt_len),
            "output_len": str(output_len),
            "data_parallel_rank": str(rank),
        })
    for prompt_len, rank in zip(remaining_lengths, measured_ranks):
        rows.append({
            "prompt_len": str(prompt_len),
            "output_len": str(output_len),
            "data_parallel_rank": str(rank),
        })
    return rows


def build_explicit_rank_rows(
    *,
    input_lengths: list[int],
    output_len: int,
    dispatch_policy: str,
    data_parallel_size: int,
    data_parallel_size_local: int,
    kv_cache_tokens_per_rank: int,
    warmup_short_rows: int,
) -> list[dict[str, str]]:
    if dispatch_policy in GREEDY_EXPLICIT_RANK_POLICIES:
        return build_greedy_explicit_rank_rows(
            input_lengths=input_lengths,
            output_len=output_len,
            data_parallel_size=data_parallel_size,
            dispatch_policy=dispatch_policy,
            kv_cache_tokens_per_rank=kv_cache_tokens_per_rank,
            warmup_short_rows=warmup_short_rows,
        )
    return build_legacy_explicit_rank_rows(
        input_lengths=input_lengths,
        output_len=output_len,
        data_parallel_size=data_parallel_size,
        data_parallel_size_local=data_parallel_size_local,
        warmup_short_rows=warmup_short_rows,
    )


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


def write_length_csv(
    path: Path,
    input_lengths: list[int],
    output_len: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt_len", "output_len"])
        writer.writeheader()
        for prompt_len in input_lengths:
            writer.writerow({
                "prompt_len": str(prompt_len),
                "output_len": str(output_len),
            })


def write_explicit_rank_length_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["prompt_len", "output_len", "data_parallel_rank"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_case_csv(path: Path, fieldnames: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_non_negative_case_int(value: str) -> int:
    stripped = value.strip()
    if stripped == "":
        return 0
    parsed = int(stripped)
    if parsed < 0:
        raise ValueError(f"Expected a non-negative integer, got {value!r}")
    return parsed


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
    routing_tag = None
    if args.routing_mode != DEFAULT_ROUTING_MODE:
        routing_tag = f"routing_{sanitize_tag(args.routing_mode)}"
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
    if routing_tag is not None:
        derived_case_name_parts.append(routing_tag)
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
    if args.max_model_len is not None:
        derived_row["max_model_len"] = str(args.max_model_len)
    if args.dispatch_policy is not None:
        derived_row["dispatch_policy"] = args.dispatch_policy

    if args.routing_mode == "explicit_rank_replay":
        assert args.data_parallel_size is not None
        assert args.data_parallel_size_local is not None
        effective_warmup_short_rows = args.warmup_short_rows
        if effective_warmup_short_rows == 0:
            effective_warmup_short_rows = parse_non_negative_case_int(
                derived_row.get("warmup_requests", "")
            )
        explicit_rows = build_explicit_rank_rows(
            input_lengths=input_lengths,
            output_len=args.output_len,
            dispatch_policy=effective_dispatch_policy,
            data_parallel_size=args.data_parallel_size,
            data_parallel_size_local=args.data_parallel_size_local,
            kv_cache_tokens_per_rank=args.kv_cache_tokens_per_rank,
            warmup_short_rows=effective_warmup_short_rows,
        )
        write_explicit_rank_length_csv(length_csv_path, explicit_rows)
    else:
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
    print(f"  routing_mode: {args.routing_mode}")
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
