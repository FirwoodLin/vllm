#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


DEFAULT_DISPATCH_POLICIES = (
    "waiting_x4_plus_running",
    "least_cache",
)
DATASET_SPECS = (
    ("short_random", "ShareGPT4o"),
    ("issue01_random", "Issue1%"),
    ("issue03_random", "Issue3%"),
    ("issue05_random", "Issue5%"),
    ("long_full", "Gemini Issues"),
)
DATASET_ORDER = [key for key, _ in DATASET_SPECS]
DATASET_LABEL_MAP = dict(DATASET_SPECS)
METRICS = (
    ("slo_attainment", "SLO\nAttainment (%)"),
    ("normlat_mean_ms", "Avg Norm\nLatency (ms)"),
    ("normlat_p99_ms", "P99 Norm\nLatency (ms)"),
)
STRATEGY_ORDER = (
    "dp32",
    "dp16cp2",
    "dp8dcp4",
    "dp4dcp8",
)
CASE_NAME_PATTERN = re.compile(
    r"^(?P<prefix>.+)-rate(?P<rate>\d+(?:\.\d+)?)(?P<suffix>(?:-.+)*)$"
)
MEMORY_TAG_PATTERN = re.compile(r"-mem[^-]+")
BATCH_SIZE_TAG_PATTERN = re.compile(r"-bs[^-]+")
DISPATCH_TAG_PATTERN = re.compile(r"(?:^|-)dispatch_(?P<policy>[^-]+)")
LATENCY_SLO_TARGET_DEFAULT_MS = 50.0
LATENCY_X_CROSS_TARGET_MS = 120.0
LATENCY_Y_MAX_MS = 120.0
LATENCY_Y_TICKS = [0.0, 25.0, 50.0, 75.0, 100.0, 120.0]


@dataclass(frozen=True)
class RunCandidate:
    model: str
    dataset: str
    case_name: str
    timestamp: str
    benchmark_dir: Path
    group_key: str
    rate: float
    dispatch_policy: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_root = repo_root / "offline_bench" / "manual_multinode"
    default_output_dir = repo_root / "offline_bench" / "plots"
    parser = argparse.ArgumentParser(
        description=(
            "Plot manual_multinode vLLM request-level normalized latency and "
            "SLO attainment as a 3x5 dataset matrix."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"manual_multinode root directory (default: {default_root})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"directory to save generated figures (default: {default_output_dir})",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="optional model names to plot, for example: --models DPSK",
    )
    parser.add_argument(
        "--dispatch-policy-filter",
        nargs="+",
        default=list(DEFAULT_DISPATCH_POLICIES),
        help=(
            "only plot the specified dispatch policies. "
            f"Default: {', '.join(DEFAULT_DISPATCH_POLICIES)}. "
            "Use --dispatch-policy-filter all to disable filtering."
        ),
    )
    parser.add_argument(
        "--ignore-bs",
        action="store_true",
        help=(
            "ignore '-bs*' in case names when grouping lines. "
            "Runs with different bs values will be merged into one series."
        ),
    )
    parser.add_argument(
        "--slo-target",
        type=float,
        default=LATENCY_SLO_TARGET_DEFAULT_MS,
        help=f"normalized latency SLO target in ms (default: {LATENCY_SLO_TARGET_DEFAULT_MS})",
    )
    return parser.parse_args()


def parse_case_name(
    case_name: str,
    *,
    ignore_bs_in_group: bool = False,
) -> tuple[str, str, float, str | None] | None:
    match = CASE_NAME_PATTERN.match(case_name)
    if match is None:
        return None
    prefix = match.group("prefix")
    suffix = match.group("suffix")
    rate = float(match.group("rate"))
    raw_group_key = MEMORY_TAG_PATTERN.sub("", f"{prefix}{suffix}")
    if ignore_bs_in_group:
        raw_group_key = BATCH_SIZE_TAG_PATTERN.sub("", raw_group_key)
    dispatch_match = DISPATCH_TAG_PATTERN.search(raw_group_key)
    dispatch_policy = (
        dispatch_match.group("policy") if dispatch_match is not None else None
    )
    group_key = DISPATCH_TAG_PATTERN.sub("", raw_group_key).strip("-")
    strategy_name = prefix.split("-", 1)[0]
    return group_key, strategy_name, rate, dispatch_policy


def load_dispatch_policy(
    benchmark_dir: Path,
    case_dispatch_policy: str | None,
) -> str:
    if case_dispatch_policy:
        return case_dispatch_policy

    run_meta_path = benchmark_dir / "run_meta.json"
    if run_meta_path.exists():
        with run_meta_path.open() as handle:
            payload = json.load(handle)
        dispatch_policy = payload.get("dispatch_policy")
        if isinstance(dispatch_policy, str) and dispatch_policy:
            return dispatch_policy

    return DEFAULT_DISPATCH_POLICIES[0]


def iter_run_candidates(root: Path, ignore_bs_in_group: bool) -> list[RunCandidate]:
    candidates: list[RunCandidate] = []
    for benchmark_dir in root.glob("*/*/*/*/benchmark"):
        if not benchmark_dir.is_dir():
            continue
        rel_parts = benchmark_dir.relative_to(root).parts
        if len(rel_parts) != 5:
            continue
        model, dataset, case_name, timestamp, _ = rel_parts
        if model.startswith("_") or dataset not in DATASET_LABEL_MAP:
            continue
        parsed = parse_case_name(
            case_name,
            ignore_bs_in_group=ignore_bs_in_group,
        )
        if parsed is None:
            continue
        group_key, _, rate, case_dispatch_policy = parsed
        candidates.append(
            RunCandidate(
                model=model,
                dataset=dataset,
                case_name=case_name,
                timestamp=timestamp,
                benchmark_dir=benchmark_dir,
                group_key=group_key,
                rate=rate,
                dispatch_policy=load_dispatch_policy(
                    benchmark_dir,
                    case_dispatch_policy,
                ),
            )
        )
    return candidates


def to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_request_level_metrics(
    requests_path: Path,
    slo_target_ms: float,
) -> dict[str, float | int | None] | None:
    if not requests_path.exists():
        return None

    normalized_latencies: list[float] = []
    total_requests = 0
    slo_success_count = 0

    with requests_path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            row = json.loads(line)
            if row.get("is_error"):
                continue

            normlat_ms = to_float(row.get("tpot_by_e2e"))
            if normlat_ms is None:
                e2e_ms = to_float(row.get("e2e_ms"))
                output_tokens = to_float(row.get("actual_output_tokens"))
                if e2e_ms is not None and output_tokens and output_tokens > 0:
                    normlat_ms = e2e_ms / output_tokens

            if normlat_ms is None:
                raise ValueError(
                    f"{requests_path}: line {line_number} missing tpot_by_e2e "
                    "and cannot derive normalized latency"
                )

            total_requests += 1
            normalized_latencies.append(normlat_ms)
            if normlat_ms <= slo_target_ms:
                slo_success_count += 1

    if not normalized_latencies:
        return None

    values = np.asarray(normalized_latencies, dtype=np.float64)
    return {
        "total_requests": total_requests,
        "slo_success_count": slo_success_count,
        "slo_attainment": 100.0 * slo_success_count / total_requests,
        "normlat_mean_ms": float(np.mean(values)),
        "normlat_p50_ms": float(np.percentile(values, 50)),
        "normlat_p90_ms": float(np.percentile(values, 90)),
        "normlat_p95_ms": float(np.percentile(values, 95)),
        "normlat_p99_ms": float(np.percentile(values, 99)),
    }


def load_summary_context(summary_path: Path) -> dict[str, float | int | None]:
    if not summary_path.exists():
        return {
            "benchmark_runtime_s": None,
            "request_throughput_rps": None,
            "summary_total_requests": None,
        }

    with summary_path.open() as handle:
        payload = json.load(handle)

    return {
        "benchmark_runtime_s": to_float(payload.get("benchmark_runtime_s")),
        "request_throughput_rps": to_float(
            payload.get("achieved_request_throughput_rps")
        ),
        "summary_total_requests": payload.get("total_requests"),
    }


def load_point_metrics(
    benchmark_dir: Path,
    slo_target_ms: float,
) -> dict[str, object] | None:
    requests_path = benchmark_dir / "requests.jsonl"
    summary_path = benchmark_dir / "summary.json"
    request_metrics = compute_request_level_metrics(requests_path, slo_target_ms)
    if request_metrics is None:
        return None

    summary_context = load_summary_context(summary_path)
    runtime_s = summary_context["benchmark_runtime_s"]
    total_requests = int(request_metrics["total_requests"])
    slo_success_count = int(request_metrics["slo_success_count"])
    goodput_rps = None
    request_throughput_rps = summary_context["request_throughput_rps"]
    if runtime_s and runtime_s > 0:
        if request_throughput_rps is None:
            request_throughput_rps = total_requests / runtime_s
        goodput_rps = slo_success_count / runtime_s

    return {
        **request_metrics,
        **summary_context,
        "goodput_rps": goodput_rps,
        "requests_path": str(requests_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "has_request_metrics": True,
    }


def strategy_name_from_group_key(group_key: str) -> str:
    return group_key.split("-", 1)[0]


def make_series_key(group_key: str, dispatch_policy: str) -> str:
    return f"{group_key}@@dispatch={dispatch_policy}"


def split_series_key(series_key: str) -> tuple[str, str]:
    group_key, dispatch_policy = series_key.rsplit("@@dispatch=", 1)
    return group_key, dispatch_policy


def build_display_names(series_keys: list[str]) -> dict[str, str]:
    strategy_to_series: dict[str, list[str]] = defaultdict(list)
    strategy_to_policies: dict[str, set[str]] = defaultdict(set)
    for series_key in series_keys:
        group_key, dispatch_policy = split_series_key(series_key)
        strategy_name = strategy_name_from_group_key(group_key)
        strategy_to_series[strategy_name].append(series_key)
        strategy_to_policies[strategy_name].add(dispatch_policy)

    display_names: dict[str, str] = {}
    for strategy_name, keys in strategy_to_series.items():
        include_dispatch = len(strategy_to_policies[strategy_name]) > 1
        for series_key in keys:
            _, dispatch_policy = split_series_key(series_key)
            label = strategy_name.upper()
            if (
                include_dispatch
                or dispatch_policy != DEFAULT_DISPATCH_POLICIES[0]
            ):
                label = f"{label} [{dispatch_policy}]"
            display_names[series_key] = label
    return display_names


def dispatch_sort_rank(dispatch_policy: str) -> int:
    if dispatch_policy == "waiting_x4_plus_running":
        return 0
    if dispatch_policy == "least_cache":
        return 1
    return 2


def series_sort_key(series_key: str) -> tuple[int, int, str]:
    group_key, dispatch_policy = split_series_key(series_key)
    strategy_name = strategy_name_from_group_key(group_key)
    try:
        strategy_rank = STRATEGY_ORDER.index(strategy_name)
    except ValueError:
        strategy_rank = len(STRATEGY_ORDER)
    return (strategy_rank, dispatch_sort_rank(dispatch_policy), series_key)


def series_style(series_key: str) -> dict[str, object]:
    group_key, dispatch_policy = split_series_key(series_key)
    strategy_name = strategy_name_from_group_key(group_key)
    style_map = {
        ("dp32", "waiting_x4_plus_running"): {
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "--",
        },
        ("dp32", "least_cache"): {
            "color": "#4c78a8",
            "marker": "s",
            "linestyle": "-",
        },
        ("dp16cp2", "waiting_x4_plus_running"): {
            "color": "#f28e2b",
            "marker": "^",
            "linestyle": "-",
        },
        ("dp8dcp4", "waiting_x4_plus_running"): {
            "color": "#59a14f",
            "marker": "D",
            "linestyle": "-",
        },
        ("dp4dcp8", "waiting_x4_plus_running"): {
            "color": "#e15759",
            "marker": "v",
            "linestyle": "-",
        },
    }
    return style_map.get(
        (strategy_name, dispatch_policy),
        {
            "color": "#444444",
            "marker": "o",
            "linestyle": "-",
        },
    )


def collect_plot_rows(
    root: Path,
    selected_models: set[str] | None,
    dispatch_policy_filter: set[str] | None,
    ignore_bs_in_group: bool,
    slo_target_ms: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates_by_case: dict[
        tuple[str, str, str, str, float], list[RunCandidate]
    ] = defaultdict(list)
    for candidate in iter_run_candidates(root, ignore_bs_in_group):
        if selected_models is not None and candidate.model not in selected_models:
            continue
        if (
            dispatch_policy_filter is not None
            and candidate.dispatch_policy not in dispatch_policy_filter
        ):
            continue
        candidate_key = (
            candidate.model,
            candidate.dataset,
            candidate.group_key,
            candidate.dispatch_policy,
            candidate.rate,
        )
        candidates_by_case[candidate_key].append(candidate)

    metrics_cache: dict[Path, dict[str, object] | None] = {}
    rows: list[dict[str, object]] = []
    dropped_rows: list[dict[str, object]] = []

    for candidate_key, case_candidates in candidates_by_case.items():
        chosen_row: dict[str, object] | None = None
        sorted_candidates = sorted(
            case_candidates,
            key=lambda item: item.timestamp,
            reverse=True,
        )
        for candidate in sorted_candidates:
            if candidate.benchmark_dir not in metrics_cache:
                metrics_cache[candidate.benchmark_dir] = load_point_metrics(
                    candidate.benchmark_dir,
                    slo_target_ms=slo_target_ms,
                )
            metrics = metrics_cache[candidate.benchmark_dir]
            if metrics is None:
                continue

            row = {
                "model": candidate.model,
                "dataset_key": candidate.dataset,
                "dataset_label": DATASET_LABEL_MAP[candidate.dataset],
                "case_name": candidate.case_name,
                "timestamp": candidate.timestamp,
                "benchmark_dir": str(candidate.benchmark_dir.resolve()),
                "group_key": candidate.group_key,
                "dispatch_policy": candidate.dispatch_policy,
                "series_key": make_series_key(
                    candidate.group_key,
                    candidate.dispatch_policy,
                ),
                "rate": candidate.rate,
                **metrics,
            }
            if chosen_row is None:
                chosen_row = row
            else:
                dropped_rows.append(row)

        if chosen_row is not None:
            rows.append(chosen_row)

    display_names_by_model: dict[str, dict[str, str]] = {}
    for model in {str(row["model"]) for row in rows + dropped_rows}:
        series_keys = sorted(
            {
                str(row["series_key"])
                for row in rows + dropped_rows
                if str(row["model"]) == model
            },
            key=series_sort_key,
        )
        display_names_by_model[model] = build_display_names(series_keys)

    for row in rows + dropped_rows:
        display_names = display_names_by_model[str(row["model"])]
        row["series_label"] = display_names[str(row["series_key"])]

    rows.sort(
        key=lambda item: (
            item["model"],
            DATASET_ORDER.index(item["dataset_key"]),
            series_sort_key(str(item["series_key"])),
            float(item["rate"]),
            str(item["timestamp"]),
        )
    )
    dropped_rows.sort(
        key=lambda item: (
            item["model"],
            DATASET_ORDER.index(item["dataset_key"]),
            series_sort_key(str(item["series_key"])),
            float(item["rate"]),
            str(item["timestamp"]),
        )
    )
    return rows, dropped_rows


def calculate_x_at_y(
    x_values: np.ndarray,
    y_values: np.ndarray,
    target_y: float,
) -> float | None:
    if len(x_values) < 2 or len(y_values) < 2:
        return None

    for idx in range(len(y_values) - 1):
        x1 = float(x_values[idx])
        x2 = float(x_values[idx + 1])
        y1 = float(y_values[idx])
        y2 = float(y_values[idx + 1])
        if math.isnan(y1) or math.isnan(y2):
            continue
        if y1 == y2 == target_y:
            return x1
        if (y1 >= target_y > y2) or (y1 <= target_y < y2):
            ratio = (target_y - y1) / (y2 - y1)
            return x1 + ratio * (x2 - x1)
    return None


def calculate_last_x_at_y(
    x_values: np.ndarray,
    y_values: np.ndarray,
    target_y: float,
) -> float | None:
    if len(x_values) < 1 or len(y_values) < 1:
        return None

    last_crossing = None
    for idx in range(len(y_values) - 1):
        x1 = float(x_values[idx])
        x2 = float(x_values[idx + 1])
        y1 = float(y_values[idx])
        y2 = float(y_values[idx + 1])
        if math.isnan(y1) or math.isnan(y2):
            continue
        if y1 == target_y:
            last_crossing = x1
        if y2 == target_y:
            last_crossing = x2
        if (y1 - target_y) * (y2 - target_y) < 0:
            ratio = (target_y - y1) / (y2 - y1)
            last_crossing = x1 + ratio * (x2 - x1)

    if last_crossing is None and len(y_values) == 1 and float(y_values[0]) == target_y:
        return float(x_values[0])
    return last_crossing


def build_model_dataset_series(
    rows: list[dict[str, object]],
    model: str,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    nested: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["model"] != model:
            continue
        nested[str(row["dataset_key"])][str(row["series_key"])].append(row)

    for dataset_series in nested.values():
        for points in dataset_series.values():
            points.sort(key=lambda item: float(item["rate"]))
    return nested


def compute_dataset_x_max(
    dataset_series: dict[str, list[dict[str, object]]],
) -> float | None:
    candidate_crossings: list[float] = []
    for metric_key in ("normlat_mean_ms", "normlat_p99_ms"):
        for points in dataset_series.values():
            x_values = np.asarray([float(point["rate"]) for point in points], dtype=float)
            y_values = np.asarray(
                [float(point[metric_key]) for point in points],
                dtype=float,
            )
            if len(x_values) == 0:
                continue
            crossing = calculate_last_x_at_y(
                x_values,
                y_values,
                LATENCY_X_CROSS_TARGET_MS,
            )
            if crossing is not None:
                candidate_crossings.append(crossing)

    if candidate_crossings:
        return max(candidate_crossings)

    max_rate = None
    for points in dataset_series.values():
        for point in points:
            rate = float(point["rate"])
            max_rate = rate if max_rate is None else max(max_rate, rate)
    return max_rate


def build_crossing_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    crossing_rows: list[dict[str, object]] = []
    for model in sorted({str(row["model"]) for row in rows}):
        model_dataset_series = build_model_dataset_series(rows, model)
        display_names = build_display_names(
            sorted(
                {
                    series_key
                    for dataset_series in model_dataset_series.values()
                    for series_key in dataset_series
                },
                key=series_sort_key,
            )
        )
        for dataset_key in DATASET_ORDER:
            dataset_series = model_dataset_series.get(dataset_key, {})
            for series_key in sorted(dataset_series, key=series_sort_key):
                points = dataset_series[series_key]
                x_values = np.asarray([float(point["rate"]) for point in points], dtype=float)
                y_values = np.asarray(
                    [float(point["slo_attainment"]) for point in points],
                    dtype=float,
                )
                crossing_rows.append(
                    {
                        "model": model,
                        "dataset_key": dataset_key,
                        "dataset_label": DATASET_LABEL_MAP[dataset_key],
                        "series_key": series_key,
                        "series_label": display_names[series_key],
                        "slo90_crossing_rate": calculate_x_at_y(
                            x_values,
                            y_values,
                            90.0,
                        ),
                        "min_rate": float(np.min(x_values)),
                        "max_rate": float(np.max(x_values)),
                    }
                )
    return crossing_rows


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def plot_matrix_for_model(
    rows: list[dict[str, object]],
    model: str,
    output_dir: Path,
    slo_target_ms: float,
) -> list[Path]:
    model_dataset_series = build_model_dataset_series(rows, model)
    model_series_keys = sorted(
        {
            series_key
            for dataset_series in model_dataset_series.values()
            for series_key in dataset_series
        },
        key=series_sort_key,
    )
    display_names = build_display_names(model_series_keys)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=len(DATASET_ORDER),
        figsize=(3.0 * len(DATASET_ORDER), 2.2 * len(METRICS) + 0.4),
        squeeze=False,
    )

    dataset_x_max_map = {
        dataset_key: compute_dataset_x_max(model_dataset_series.get(dataset_key, {}))
        for dataset_key in DATASET_ORDER
    }

    for col_idx, dataset_key in enumerate(DATASET_ORDER):
        dataset_series = model_dataset_series.get(dataset_key, {})
        for row_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                ax.set_title(
                    DATASET_LABEL_MAP[dataset_key],
                    fontsize=12,
                    fontweight="bold",
                )
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
            if row_idx == len(METRICS) - 1:
                ax.set_xlabel("Request Rate", fontsize=11)

            if not dataset_series:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#777777",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                continue

            for series_key in model_series_keys:
                points = dataset_series.get(series_key)
                if not points:
                    continue
                style = series_style(series_key)
                ax.plot(
                    [float(point["rate"]) for point in points],
                    [float(point[metric_key]) for point in points],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    markersize=5,
                    label=display_names[series_key],
                )
                if metric_key == "slo_attainment":
                    crossing_rate = calculate_x_at_y(
                        np.asarray([float(point["rate"]) for point in points], dtype=float),
                        np.asarray(
                            [float(point["slo_attainment"]) for point in points],
                            dtype=float,
                        ),
                        90.0,
                    )
                    if crossing_rate is not None:
                        ax.axvline(
                            x=crossing_rate,
                            color=style["color"],
                            linestyle=":",
                            linewidth=1.0,
                            alpha=0.75,
                        )

            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

            dataset_x_max = dataset_x_max_map.get(dataset_key)
            if dataset_x_max is not None:
                ax.set_xlim(left=0, right=dataset_x_max)

            if metric_key == "slo_attainment":
                ax.set_ylim(0, 105)
                ax.axhline(
                    90.0,
                    color="#777777",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.8,
                )
                ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _: f"{value:g}")
                )
            else:
                ax.set_ylim(0, LATENCY_Y_MAX_MS)
                ax.set_yticks(LATENCY_Y_TICKS)
                ax.axhline(
                    slo_target_ms,
                    color="#777777",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.8,
                )
                ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _: f"{int(value)}")
                )
                ax.minorticks_off()

    legend_handles: list[plt.Line2D] = []
    legend_labels: list[str] = []
    for series_key in model_series_keys:
        style = series_style(series_key)
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                markersize=5,
            )
        )
        legend_labels.append(display_names[series_key])

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=min(5, len(legend_labels)),
            frameon=False,
            fontsize=10,
        )

    fig.suptitle(f"{model} manual_multinode matrix", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18, top=0.88, wspace=0.25, hspace=0.28)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / f"{sanitize_filename(model)}_manual_multinode_matrix"
    output_paths = [
        output_prefix.with_suffix(".png"),
        output_prefix.with_suffix(".pdf"),
    ]
    for output_path in output_paths:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_paths


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    selected_models = set(args.models) if args.models else None
    dispatch_policy_filter = (
        None
        if "all" in args.dispatch_policy_filter
        else set(args.dispatch_policy_filter)
    )

    if not root.exists():
        raise FileNotFoundError(f"root directory does not exist: {root}")

    rows, dropped_rows = collect_plot_rows(
        root,
        selected_models,
        dispatch_policy_filter,
        ignore_bs_in_group=args.ignore_bs,
        slo_target_ms=args.slo_target,
    )
    if not rows:
        raise RuntimeError(f"no usable benchmark data found under: {root}")

    data_dir = output_dir / "data"
    write_tsv(data_dir / "manual_multinode_matrix_metrics.tsv", rows)
    write_tsv(data_dir / "manual_multinode_matrix_duplicates_dropped.tsv", dropped_rows)
    write_tsv(data_dir / "manual_multinode_matrix_slo90_crossings.tsv", build_crossing_rows(rows))
    write_tsv(
        data_dir / "manual_multinode_matrix_metadata.tsv",
        [
            {
                "root": str(root),
                "output_dir": str(output_dir),
                "slo_target_ms": args.slo_target,
                "dispatch_policy_filter": (
                    ",".join(sorted(dispatch_policy_filter))
                    if dispatch_policy_filter is not None
                    else "all"
                ),
                "ignore_bs": args.ignore_bs,
                "dataset_order": ",".join(DATASET_ORDER),
            }
        ],
    )

    models = sorted({str(row["model"]) for row in rows})
    generated_paths: list[Path] = []
    for model in models:
        generated_paths.extend(
            plot_matrix_for_model(
                rows,
                model=model,
                output_dir=output_dir,
                slo_target_ms=args.slo_target,
            )
        )

    print(f"input_root: {root}")
    print(f"output_dir: {output_dir}")
    print(f"metrics_tsv: {data_dir / 'manual_multinode_matrix_metrics.tsv'}")
    print(
        "duplicates_tsv: "
        f"{data_dir / 'manual_multinode_matrix_duplicates_dropped.tsv'}"
    )
    print(
        "crossings_tsv: "
        f"{data_dir / 'manual_multinode_matrix_slo90_crossings.tsv'}"
    )
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
