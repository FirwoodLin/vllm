#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

# os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


METRIC_SPECS = (
    ("tpot_by_e2e", "tpot_by_e2e", 150.0),
    ("tpot_without_queue_ms", "tpot_without_queue_ms", 150.0),
    ("ttft_ms", "ttft_ms", 1000.0),
)
SUBMETRICS = ("mean", "p50", "p99")
CASE_NAME_PATTERN = re.compile(
    r"^(?P<prefix>.+)-rate(?P<rate>\d+(?:\.\d+)?)(?P<suffix>(?:-.+)*)$"
)
MEMORY_TAG_PATTERN = re.compile(r"-mem[^-]+")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")


@dataclass(frozen=True)
class RunCandidate:
    model: str
    dataset: str
    case_name: str
    timestamp: str
    benchmark_dir: Path


@dataclass(frozen=True)
class SeriesPoint:
    rate: float
    stats: dict[str, dict[str, float]]
    source_dir: Path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Plot manual_multinode benchmark metrics per model and dataset. "
            "Each figure has metric rows and submetric columns; each line "
            "represents one parallel strategy/configuration."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=script_dir,
        help=f"manual_multinode root directory (default: {script_dir})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "plots",
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional model names to plot, for example: --models KIMI DPSK",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Output figure format.",
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


def summarize(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    return {
        "mean": fmean(sorted_values),
        "p50": percentile(sorted_values, 50.0),
        "p99": percentile(sorted_values, 99.0),
    }


def iter_run_candidates(root: Path) -> list[RunCandidate]:
    candidates: list[RunCandidate] = []
    for benchmark_dir in root.glob("*/*/*/*/benchmark"):
        if not benchmark_dir.is_dir():
            continue
        rel_parts = benchmark_dir.relative_to(root).parts
        if len(rel_parts) != 5:
            continue
        model, dataset, case_name, timestamp, _ = rel_parts
        if model.startswith("_"):
            continue
        candidates.append(
            RunCandidate(
                model=model,
                dataset=dataset,
                case_name=case_name,
                timestamp=timestamp,
                benchmark_dir=benchmark_dir,
            )
        )
    return candidates


def parse_case_name(case_name: str) -> tuple[str, str, float] | None:
    match = CASE_NAME_PATTERN.match(case_name)
    if match is None:
        return None
    prefix = match.group("prefix")
    suffix = match.group("suffix")
    rate = float(match.group("rate"))
    group_key = MEMORY_TAG_PATTERN.sub("", f"{prefix}{suffix}")
    strategy_name = prefix.split("-", 1)[0]
    return group_key, strategy_name, rate


def load_summary_json(path: Path) -> dict[str, dict[str, float]] | None:
    if not path.exists():
        return None

    with path.open() as f:
        payload = json.load(f)

    stats: dict[str, dict[str, float]] = {}
    for metric_key, _, _ in METRIC_SPECS:
        metric_payload = payload.get(metric_key)
        if not isinstance(metric_payload, dict):
            return None

        metric_stats: dict[str, float] = {}
        for submetric in SUBMETRICS:
            value = metric_payload.get(submetric)
            if value is None:
                return None
            metric_stats[submetric] = float(value)
        stats[metric_key] = metric_stats
    return stats


def load_requests_jsonl(path: Path) -> dict[str, dict[str, float]] | None:
    if not path.exists():
        return None

    metric_values: dict[str, list[float]] = {key: [] for key, _, _ in METRIC_SPECS}
    with path.open() as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("is_error"):
                continue

            valid_row = True
            numeric_values: dict[str, float] = {}
            for metric_key, _, _ in METRIC_SPECS:
                value = row.get(metric_key)
                if value is None:
                    valid_row = False
                    break
                try:
                    numeric_values[metric_key] = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}: line {line_number} has invalid {metric_key}: {exc}"
                    ) from exc

            if not valid_row:
                continue

            for metric_key, value in numeric_values.items():
                metric_values[metric_key].append(value)

    if not all(metric_values.values()):
        return None

    return {
        metric_key: summarize(values)
        for metric_key, values in metric_values.items()
    }


def load_point_stats(benchmark_dir: Path) -> dict[str, dict[str, float]] | None:
    summary_path = benchmark_dir / "summary.json"
    stats = load_summary_json(summary_path)
    if stats is not None:
        return stats

    requests_path = benchmark_dir / "requests.jsonl"
    return load_requests_jsonl(requests_path)


def collect_plot_data(
    root: Path,
    selected_models: set[str] | None,
) -> dict[str, dict[str, dict[str, list[SeriesPoint]]]]:
    candidates_by_case: dict[
        tuple[str, str, str, float], list[RunCandidate]
    ] = defaultdict(list)
    for candidate in iter_run_candidates(root):
        if selected_models is not None and candidate.model not in selected_models:
            continue
        parsed = parse_case_name(candidate.case_name)
        if parsed is None:
            continue
        group_key, _, rate = parsed
        candidate_key = (candidate.model, candidate.dataset, group_key, rate)
        candidates_by_case[candidate_key].append(candidate)

    plot_data: dict[str, dict[str, dict[str, list[SeriesPoint]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    selected_candidates: list[tuple[RunCandidate, dict[str, dict[str, float]]]] = []
    for case_candidates in candidates_by_case.values():
        chosen_candidate: RunCandidate | None = None
        chosen_stats: dict[str, dict[str, float]] | None = None
        for candidate in sorted(
            case_candidates,
            key=lambda item: item.timestamp,
            reverse=True,
        ):
            stats = load_point_stats(candidate.benchmark_dir)
            if stats is None:
                continue
            chosen_candidate = candidate
            chosen_stats = stats
            break
        if chosen_candidate is not None and chosen_stats is not None:
            selected_candidates.append((chosen_candidate, chosen_stats))

    for candidate, stats in sorted(
        selected_candidates,
        key=lambda item: (item[0].model, item[0].dataset, item[0].case_name,
                          item[0].timestamp),
    ):
        parsed = parse_case_name(candidate.case_name)
        if parsed is None:
            continue
        group_key, _, rate = parsed
        plot_data[candidate.model][candidate.dataset][group_key].append(
            SeriesPoint(rate=rate, stats=stats, source_dir=candidate.benchmark_dir)
        )

    for model_datasets in plot_data.values():
        for dataset_points in model_datasets.values():
            for points in dataset_points.values():
                points.sort(key=lambda point: point.rate)

    return plot_data


def strategy_name_from_group_key(group_key: str) -> str:
    return group_key.split("-", 1)[0]


def build_display_names(
    model_series: dict[str, list[SeriesPoint]],
) -> dict[str, str]:
    strategy_to_groups: dict[str, list[str]] = defaultdict(list)
    for group_key in model_series:
        strategy_name = strategy_name_from_group_key(group_key)
        strategy_to_groups[strategy_name].append(group_key)

    display_names: dict[str, str] = {}
    for strategy_name, group_keys in strategy_to_groups.items():
        if len(group_keys) == 1:
            display_names[group_keys[0]] = strategy_name
            continue
        for group_key in group_keys:
            display_names[group_key] = group_key
    return display_names


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def render_dataset_figure(
    model: str,
    dataset: str,
    dataset_series: dict[str, list[SeriesPoint]],
    output_dir: Path,
    output_format: str,
) -> Path:
    display_names = build_display_names(dataset_series)
    all_rates = sorted(
        {
            point.rate
            for points in dataset_series.values()
            for point in points
        }
    )

    plt.style.use("bmh")
    fig, axes = plt.subplots(
        nrows=len(METRIC_SPECS),
        ncols=len(SUBMETRICS),
        figsize=(18, 12),
        sharex=True,
        constrained_layout=False,
    )
    if len(METRIC_SPECS) == 1:
        axes = [axes]
    fig.suptitle(
        f"{model} {dataset} manual_multinode metrics",
        fontsize=18,
        y=0.98,
    )

    for strategy_index, group_key in enumerate(sorted(dataset_series)):
        points = dataset_series[group_key]
        marker = MARKERS[strategy_index % len(MARKERS)]
        label = display_names[group_key]

        for row_index, (metric_key, metric_title, y_max) in enumerate(METRIC_SPECS):
            for col_index, submetric in enumerate(SUBMETRICS):
                ax = axes[row_index][col_index]
                x_values = [point.rate for point in points]
                y_values = [point.stats[metric_key][submetric] for point in points]
                ax.plot(
                    x_values,
                    y_values,
                    marker=marker,
                    linewidth=2,
                    markersize=6,
                    label=label,
                )
                ax.set_ylim(0, y_max)
                ax.grid(True, alpha=0.4)
                if row_index == 0:
                    ax.set_title(submetric, fontsize=12)
                if col_index == 0:
                    ax.set_ylabel(metric_title)
                if row_index == len(METRIC_SPECS) - 1:
                    ax.set_xlabel("request rate")
                if all_rates:
                    ax.set_xticks(all_rates)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=min(4, len(labels)),
            frameon=True,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output_path = output_dir / (
        f"{sanitize_filename(model)}_"
        f"{sanitize_filename(dataset)}_"
        f"manual_multinode_metrics.{output_format}"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    selected_models = set(args.models) if args.models else None

    if not root.exists():
        raise FileNotFoundError(f"root directory does not exist: {root}")

    plot_data = collect_plot_data(root, selected_models)
    if not plot_data:
        raise RuntimeError(f"no usable benchmark data found under: {root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []
    for model in sorted(plot_data):
        for dataset in sorted(plot_data[model]):
            dataset_series = plot_data[model][dataset]
            if not dataset_series:
                continue
            generated_paths.append(
                render_dataset_figure(
                    model=model,
                    dataset=dataset,
                    dataset_series=dataset_series,
                    output_dir=output_dir,
                    output_format=args.format,
                )
            )

    if not generated_paths:
        raise RuntimeError("no figures were generated")

    print(f"input_root: {root}")
    print(f"output_dir: {output_dir}")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
