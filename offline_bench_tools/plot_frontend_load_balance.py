#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
KV_CACHE_SIZE_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
ENGINE_STATS_RE = re.compile(
    r"Engine\s+(\d+):.*?Running:\s+(\d+)\s+reqs.*?"
    r"(?:GPU KV cache usage:\s+([0-9.]+)%|Free KV blocks:\s+(\d+))"
)


@dataclass(frozen=True)
class EnginePoint:
    running: float
    kv_usage_pct: float


@dataclass(frozen=True)
class ParsedLog:
    frontend_log: Path
    run_dir: Path
    label: str
    num_engines: int
    num_steps: int
    kv_usage_by_step: list[list[float]]
    running_by_step: list[list[float]]


@dataclass(frozen=True)
class MetricStats:
    mins: list[float]
    maxs: list[float]
    p25s: list[float]
    medians: list[float]
    means: list[float]
    p75s: list[float]
    cvs: list[float]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Parse frontend.log files and plot per-timestep load-balance "
            "statistics for GPU KV cache usage and running requests."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help=(
            "Input frontend.log, run directory, or a higher-level root such as "
            "offline_bench/archive/DPSK/issue05_random."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to save generated figures. By default, figures are "
            "saved to a 'frontend_load_balance' subdirectory under the "
            "corresponding dataset directory."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Output figure format.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help=(
            "When an input resolves to multiple timestamps for the same run "
            "configuration, plot all of them instead of only the latest one."
        ),
    )
    parser.add_argument(
        "--block-size-tokens",
        type=int,
        default=64,
        help=(
            "Token count per KV block when falling back to old logs that only "
            "contain 'Free KV blocks'."
        ),
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


def compute_metric_stats(values_by_step: list[list[float]]) -> MetricStats:
    mins: list[float] = []
    maxs: list[float] = []
    p25s: list[float] = []
    medians: list[float] = []
    means: list[float] = []
    p75s: list[float] = []
    cvs: list[float] = []

    for step_values in values_by_step:
        sorted_values = sorted(step_values)
        mins.append(sorted_values[0])
        maxs.append(sorted_values[-1])
        p25s.append(percentile(sorted_values, 25.0))
        medians.append(percentile(sorted_values, 50.0))
        mean_value = fmean(sorted_values)
        means.append(mean_value)
        p75s.append(percentile(sorted_values, 75.0))
        std_value = pstdev(sorted_values)
        cvs.append(0.0 if mean_value <= 0 else (std_value / mean_value) * 100.0)

    return MetricStats(
        mins=mins,
        maxs=maxs,
        p25s=p25s,
        medians=medians,
        means=means,
        p75s=p75s,
        cvs=cvs,
    )


def clean_text(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def pick_latest_logs(logs: list[Path], all_runs: bool) -> list[Path]:
    resolved_logs = sorted({log.resolve() for log in logs})
    if all_runs:
        return resolved_logs

    latest_by_run_dir: dict[Path, Path] = {}
    for log in resolved_logs:
        run_dir = log.parent.parent
        existing = latest_by_run_dir.get(run_dir)
        if existing is None or log.parent.name > existing.parent.name:
            latest_by_run_dir[run_dir] = log
    return sorted(latest_by_run_dir.values())


def resolve_manifest_frontend_log(
    manifest_path: Path,
    all_runs: bool,
) -> list[Path]:
    try:
        with manifest_path.open() as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    paths_payload = payload.get("paths")
    if not isinstance(paths_payload, dict):
        return []

    candidates: list[Path] = []

    frontend_log_path = paths_payload.get("frontend_log_path")
    if isinstance(frontend_log_path, str):
        frontend_log = Path(frontend_log_path)
        if frontend_log.is_file():
            candidates.append(frontend_log)

    run_dir = paths_payload.get("run_dir")
    if isinstance(run_dir, str):
        run_dir_path = Path(run_dir)
        if run_dir_path.is_dir():
            candidates.extend(sorted(run_dir_path.glob("*/frontend.log")))

    if candidates:
        return pick_latest_logs(candidates, all_runs=all_runs)
    return []


def discover_frontend_logs(input_path: Path, all_runs: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.name == "frontend.log":
            return [input_path.resolve()]
        if input_path.name == "case_manifest.json":
            return resolve_manifest_frontend_log(input_path, all_runs=all_runs)
        return []

    if not input_path.is_dir():
        return []

    direct_logs = [path for path in input_path.rglob("frontend.log") if path.is_file()]
    if direct_logs:
        return pick_latest_logs(direct_logs, all_runs=all_runs)

    manifest_logs: list[Path] = []
    for manifest_path in input_path.rglob("case_manifest.json"):
        manifest_logs.extend(
            resolve_manifest_frontend_log(manifest_path, all_runs=all_runs)
        )
    return pick_latest_logs(manifest_logs, all_runs=all_runs)


def infer_default_output_dir(input_path: Path) -> Path:
    if input_path.is_file():
        if input_path.name in {"frontend.log", "case_manifest.json"}:
            return input_path.parent.parent.parent / "frontend_load_balance"
        return input_path.parent / "frontend_load_balance"

    if not input_path.is_dir():
        return input_path.parent / "frontend_load_balance"

    if (input_path / "frontend.log").is_file() or (input_path
                                                   / "case_manifest.json").is_file():
        return input_path.parent.parent / "frontend_load_balance"

    if any(input_path.glob("*/frontend.log")) or any(
            input_path.glob("*/case_manifest.json")):
        return input_path.parent / "frontend_load_balance"

    if any(input_path.glob("*/*/frontend.log")) or any(
            input_path.glob("*/*/case_manifest.json")):
        return input_path / "frontend_load_balance"

    return input_path / "frontend_load_balance"


def infer_expected_engine_ids(
    snapshots: list[dict[int, EnginePoint]],
) -> tuple[int, tuple[int, ...]] | None:
    if not snapshots:
        return None

    size_counter = Counter(len(snapshot) for snapshot in snapshots if snapshot)
    if not size_counter:
        return None
    expected_size = size_counter.most_common(1)[0][0]

    id_counter = Counter(
        tuple(sorted(snapshot.keys()))
        for snapshot in snapshots
        if len(snapshot) == expected_size
    )
    if not id_counter:
        return None
    expected_ids = id_counter.most_common(1)[0][0]
    return expected_size, expected_ids


def parse_frontend_log(
    frontend_log: Path,
    block_size_tokens: int,
) -> ParsedLog:
    raw_text = frontend_log.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(raw_text)

    max_kv_tokens: list[int] = []
    snapshots: list[dict[int, EnginePoint]] = []
    current_snapshot: dict[int, EnginePoint] = {}
    last_engine_id: int | None = None

    for line in text.splitlines():
        size_match = KV_CACHE_SIZE_RE.search(line)
        if size_match is not None:
            max_kv_tokens.append(int(size_match.group(1).replace(",", "")))

        stats_match = ENGINE_STATS_RE.search(line)
        if stats_match is None:
            continue

        engine_id = int(stats_match.group(1))
        running = float(stats_match.group(2))

        kv_usage_pct_raw = stats_match.group(3)
        free_kv_blocks_raw = stats_match.group(4)

        kv_usage_pct: float | None = None
        if kv_usage_pct_raw is not None:
            kv_usage_pct = float(kv_usage_pct_raw)
        elif free_kv_blocks_raw is not None and max_kv_tokens:
            max_blocks = min(max_kv_tokens) // block_size_tokens
            if max_blocks > 0:
                free_blocks = int(free_kv_blocks_raw)
                kv_usage_pct = (1.0 - (free_blocks / max_blocks)) * 100.0

        if kv_usage_pct is None:
            continue

        if last_engine_id is not None and engine_id <= last_engine_id:
            if current_snapshot:
                snapshots.append(current_snapshot)
            current_snapshot = {}

        current_snapshot[engine_id] = EnginePoint(
            running=running,
            kv_usage_pct=kv_usage_pct,
        )
        last_engine_id = engine_id

    if current_snapshot:
        snapshots.append(current_snapshot)

    inferred = infer_expected_engine_ids(snapshots)
    if inferred is None:
        raise ValueError(f"{frontend_log}: no complete engine snapshots found")
    expected_size, expected_ids = inferred

    complete_snapshots = [
        snapshot
        for snapshot in snapshots
        if len(snapshot) == expected_size and tuple(sorted(snapshot.keys())) == expected_ids
    ]
    if not complete_snapshots:
        raise ValueError(f"{frontend_log}: no complete engine snapshots found")

    kv_usage_by_step = [
        [snapshot[engine_id].kv_usage_pct for engine_id in expected_ids]
        for snapshot in complete_snapshots
    ]
    running_by_step = [
        [snapshot[engine_id].running for engine_id in expected_ids]
        for snapshot in complete_snapshots
    ]

    run_dir = frontend_log.parent.parent
    label_parts = run_dir.parts[-4:] if len(run_dir.parts) >= 4 else run_dir.parts
    label = "/".join(label_parts)

    return ParsedLog(
        frontend_log=frontend_log,
        run_dir=run_dir,
        label=label,
        num_engines=expected_size,
        num_steps=len(complete_snapshots),
        kv_usage_by_step=kv_usage_by_step,
        running_by_step=running_by_step,
    )


def plot_metric(
    ax: plt.Axes,
    steps: list[int],
    stats: MetricStats,
    *,
    title: str,
    ylabel: str,
    ribbon_color: str,
    cv_color: str,
    y_max: float | None = None,
) -> plt.Axes:
    ax.fill_between(
        steps,
        stats.mins,
        stats.maxs,
        color=ribbon_color,
        alpha=0.15,
        label="Min-Max",
    )
    ax.fill_between(
        steps,
        stats.p25s,
        stats.p75s,
        color=ribbon_color,
        alpha=0.35,
        label="P25-P75",
    )
    ax.plot(
        steps,
        stats.medians,
        color=ribbon_color,
        linewidth=1.4,
        label="Median",
    )
    ax.plot(
        steps,
        stats.means,
        color=ribbon_color,
        linewidth=1.2,
        linestyle=":",
        label="Mean",
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(bottom=0)
    if y_max is not None:
        ax.set_ylim(0, y_max)

    cv_ax = ax.twinx()
    cv_ax.plot(
        steps,
        stats.cvs,
        color=cv_color,
        linewidth=1.1,
        linestyle="--",
        label="CV",
    )
    cv_ax.set_ylabel("CV (%)")
    cv_ax.set_ylim(0, 150.0)
    cv_ax.spines["top"].set_visible(False)
    cv_ax.spines["left"].set_visible(False)

    handles = ax.get_legend_handles_labels()
    cv_handles = cv_ax.get_legend_handles_labels()
    ax.legend(
        handles[0] + cv_handles[0],
        handles[1] + cv_handles[1],
        loc="upper left",
        frameon=True,
    )
    return cv_ax


def output_name_for(parsed_log: ParsedLog, fmt: str) -> str:
    timestamp = parsed_log.frontend_log.parent.name
    case_name = parsed_log.run_dir.name
    return f"{case_name}__{timestamp}__frontend_load_balance.{fmt}"


def plot_parsed_log(parsed_log: ParsedLog, output_dir: Path, fmt: str) -> Path:
    kv_stats = compute_metric_stats(parsed_log.kv_usage_by_step)
    running_stats = compute_metric_stats(parsed_log.running_by_step)
    steps = list(range(1, parsed_log.num_steps + 1))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        constrained_layout=True,
    )

    fig.suptitle(
        f"{parsed_log.label}\n"
        f"{parsed_log.num_engines} engines, {parsed_log.num_steps} complete snapshots",
        fontsize=12,
    )

    plot_metric(
        axes[0],
        steps,
        kv_stats,
        title="GPU KV Cache Usage",
        ylabel="Usage (%)",
        ribbon_color="#1f77b4",
        cv_color="#7a3db8",
        y_max=100.0,
    )
    plot_metric(
        axes[1],
        steps,
        running_stats,
        title="Running Requests",
        ylabel="Running Reqs",
        ribbon_color="#2a9d55",
        cv_color="#b85c00",
    )
    axes[1].set_xlabel("Timestep")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name_for(parsed_log, fmt)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()

    log_to_output_dir: dict[Path, Path] = {}
    for input_path in args.inputs:
        discovered_logs = discover_frontend_logs(input_path, all_runs=args.all_runs)
        default_output_dir = infer_default_output_dir(input_path.resolve())
        for discovered_log in discovered_logs:
            resolved_log = discovered_log.resolve()
            if args.output_dir is not None:
                log_to_output_dir[resolved_log] = args.output_dir.resolve()
            else:
                log_to_output_dir.setdefault(resolved_log, default_output_dir)

    frontend_logs = sorted(log_to_output_dir)
    if not frontend_logs:
        raise SystemExit("No frontend.log files could be resolved from the given inputs.")

    saved_paths: list[Path] = []
    skipped_logs: list[tuple[Path, str]] = []
    for frontend_log in frontend_logs:
        try:
            parsed_log = parse_frontend_log(
                frontend_log,
                block_size_tokens=args.block_size_tokens,
            )
        except ValueError as exc:
            skipped_logs.append((frontend_log, str(exc)))
            continue

        output_dir = log_to_output_dir[frontend_log].resolve()
        saved_paths.append(plot_parsed_log(parsed_log, output_dir, args.format))

    if not saved_paths:
        if skipped_logs:
            error_details = "\n".join(
                f"- {log_path}: {reason}" for log_path, reason in skipped_logs)
            raise SystemExit(
                "No plottable frontend logs were found.\n"
                f"Skipped logs:\n{error_details}"
            )
        raise SystemExit("No plottable frontend logs were found.")

    print(f"Resolved {len(frontend_logs)} frontend logs.")
    if skipped_logs:
        print(f"Skipped {len(skipped_logs)} frontend logs:", file=sys.stderr)
        for log_path, reason in skipped_logs:
            print(f"- {log_path}: {reason}", file=sys.stderr)
    for saved_path in saved_paths:
        print(saved_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
