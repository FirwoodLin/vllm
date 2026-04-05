#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from statistics import fmean


DEFAULT_REQUESTS = Path(
    "/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench/"
    "manual_multinode/20260403-002158__manual_poisson/"
    "async_dp8_tp4_1k1k_r30_bs384/benchmark/requests.jsonl"
)
DEFAULT_PERCENTILES = [50.0, 90.0, 95.0, 99.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse requests.jsonl and compare e2e_ms / actual_output_tokens "
            "against TPOT metrics."
        )
    )
    parser.add_argument(
        "requests_jsonl",
        nargs="?",
        default=str(DEFAULT_REQUESTS),
        help=f"Path to requests.jsonl (default: {DEFAULT_REQUESTS})",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=DEFAULT_PERCENTILES,
        help="Percentiles to print, for example: --percentiles 50 90 95 99",
    )
    return parser.parse_args()


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty data")
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * (pct / 100.0)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    if lower_index == upper_index:
        return lower_value
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def summarize(values: list[float], percentiles: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    summary = {
        "count": float(len(sorted_values)),
        "mean": fmean(sorted_values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
    }
    for pct in percentiles:
        summary[f"p{pct:g}"] = percentile(sorted_values, pct)
    return summary


def format_summary_table(
    summaries: list[tuple[str, dict[str, float]]],
    percentiles: list[float],
    unit: str,
) -> str:
    headers = ["metric", "count", "mean", *[f"p{pct:g}" for pct in percentiles], "min", "max"]
    rows: list[list[str]] = []
    for metric_name, summary in summaries:
        row = [metric_name, str(int(summary["count"]))]
        row.append(f"{summary['mean']:.6f}")
        for pct in percentiles:
            row.append(f"{summary[f'p{pct:g}']:.6f}")
        row.extend([f"{summary['min']:.6f}", f"{summary['max']:.6f}"])
        rows.append(row)

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(row: list[str]) -> str:
        return "  ".join(
            cell.ljust(widths[idx]) if idx == 0 else cell.rjust(widths[idx])
            for idx, cell in enumerate(row)
        )

    lines = [f"unit: {unit}", render_row(headers)]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def load_metrics(path: Path) -> tuple[dict[str, list[float]], dict[str, int]]:
    metrics = {
        "e2e_ms_per_actual_token": [],
        "tpot_without_queue_ms": [],
        "tpot_with_initial_queue_ms": [],
        "delta_vs_tpot_without_queue_ms": [],
        "delta_vs_tpot_with_initial_queue_ms": [],
        "ratio_vs_tpot_without_queue_pct": [],
        "ratio_vs_tpot_with_initial_queue_pct": [],
    }
    counters = {
        "total_rows": 0,
        "blank_rows": 0,
        "error_rows": 0,
        "zero_or_missing_actual_tokens": 0,
        "missing_required_fields": 0,
        "used_rows": 0,
    }

    required_fields = {
        "e2e_ms",
        "actual_output_tokens",
        "tpot_without_queue_ms",
        "tpot_with_initial_queue_ms",
    }

    with path.open() as f:
        for line_number, raw_line in enumerate(f, start=1):
            counters["total_rows"] += 1
            line = raw_line.strip()
            if not line:
                counters["blank_rows"] += 1
                continue

            obj = json.loads(line)

            if obj.get("is_error"):
                counters["error_rows"] += 1
                continue

            if not required_fields.issubset(obj):
                counters["missing_required_fields"] += 1
                continue

            actual_output_tokens = obj.get("actual_output_tokens")
            if not actual_output_tokens or actual_output_tokens <= 0:
                counters["zero_or_missing_actual_tokens"] += 1
                continue

            try:
                e2e_ms = float(obj["e2e_ms"])
                actual_output_tokens = float(actual_output_tokens)
                tpot_without_queue_ms = float(obj["tpot_without_queue_ms"])
                tpot_with_initial_queue_ms = float(obj["tpot_with_initial_queue_ms"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"line {line_number}: invalid numeric field: {exc}"
                ) from exc

            derived = e2e_ms / actual_output_tokens

            metrics["e2e_ms_per_actual_token"].append(derived)
            metrics["tpot_without_queue_ms"].append(tpot_without_queue_ms)
            metrics["tpot_with_initial_queue_ms"].append(tpot_with_initial_queue_ms)
            metrics["delta_vs_tpot_without_queue_ms"].append(
                derived - tpot_without_queue_ms
            )
            metrics["delta_vs_tpot_with_initial_queue_ms"].append(
                derived - tpot_with_initial_queue_ms
            )

            if tpot_without_queue_ms != 0:
                metrics["ratio_vs_tpot_without_queue_pct"].append(
                    (derived / tpot_without_queue_ms - 1.0) * 100.0
                )
            if tpot_with_initial_queue_ms != 0:
                metrics["ratio_vs_tpot_with_initial_queue_pct"].append(
                    (derived / tpot_with_initial_queue_ms - 1.0) * 100.0
                )

            counters["used_rows"] += 1

    if counters["used_rows"] == 0:
        raise ValueError(f"no usable rows found in {path}")

    return metrics, counters


def main() -> None:
    args = parse_args()
    requests_path = Path(args.requests_jsonl)
    if not requests_path.exists():
        raise FileNotFoundError(f"file not found: {requests_path}")

    percentiles = sorted(dict.fromkeys(args.percentiles))
    metrics, counters = load_metrics(requests_path)

    metric_summaries = [
        ("e2e_ms/actual_output_tokens", summarize(metrics["e2e_ms_per_actual_token"], percentiles)),
        ("tpot_without_queue_ms", summarize(metrics["tpot_without_queue_ms"], percentiles)),
        ("tpot_with_initial_queue_ms", summarize(metrics["tpot_with_initial_queue_ms"], percentiles)),
    ]
    delta_summaries = [
        (
            "e2e/token - tpot_without_queue_ms",
            summarize(metrics["delta_vs_tpot_without_queue_ms"], percentiles),
        ),
        (
            "e2e/token - tpot_with_initial_queue_ms",
            summarize(metrics["delta_vs_tpot_with_initial_queue_ms"], percentiles),
        ),
    ]
    ratio_summaries = [
        (
            "(e2e/token vs tpot_without_queue_ms) %",
            summarize(metrics["ratio_vs_tpot_without_queue_pct"], percentiles),
        ),
        (
            "(e2e/token vs tpot_with_initial_queue_ms) %",
            summarize(metrics["ratio_vs_tpot_with_initial_queue_pct"], percentiles),
        ),
    ]

    print(f"requests_jsonl: {requests_path}")
    print(
        "rows: "
        f"total={counters['total_rows']} "
        f"used={counters['used_rows']} "
        f"blank={counters['blank_rows']} "
        f"errors={counters['error_rows']} "
        f"zero_or_missing_actual_tokens={counters['zero_or_missing_actual_tokens']} "
        f"missing_required_fields={counters['missing_required_fields']}"
    )
    print()
    print("Metric summaries")
    print(format_summary_table(metric_summaries, percentiles, "ms/token"))
    print()
    print("Difference summaries")
    print(format_summary_table(delta_summaries, percentiles, "ms/token"))
    print()
    print("Relative difference summaries")
    print(format_summary_table(ratio_summaries, percentiles, "%"))


if __name__ == "__main__":
    main()
