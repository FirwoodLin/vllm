#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter


HEADER_RE = re.compile(
    r"(?P<timestamp>\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?"
    r"Engine\s+(?P<engine_id>\d+):"
)
WAITING_TOKENS_RE = re.compile(r"Waiting tokens:\s+(?P<value>\d+)")
WAITING_HEAD_TOKENS_RE = re.compile(r"Waiting head tokens:\s+(?P<value>\d+)")
GPU_KV_USAGE_RE = re.compile(r"GPU KV cache usage:\s+(?P<value>\d+(?:\.\d+)?)%")
RUN_DIR_YEAR_RE = re.compile(r"^(?P<year>\d{4})\d{4}-\d{6}$")
PER_ENGINE_PLOT_GROUP_SIZE = 8


@dataclass(frozen=True)
class JobSpec:
    log_path: Path
    base_name: str
    out_dir: Path
    year: int
    title: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse vLLM frontend.log engine stats and render waiting token / "
            "GPU KV cache usage time-series plots."
        ))
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--log",
        type=Path,
        help="Path to frontend.log.",
    )
    input_group.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory that contains frontend.log.",
    )
    input_group.add_argument(
        "--batch-root",
        type=Path,
        help=(
            "Directory like .../MODEL/DATASET. Recursively process each "
            "parseable run directory under it and write outputs next to the "
            "local frontend.log."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "Directory to write plots and optional CSV outputs in single-input "
            "mode. Defaults to the frontend.log directory."
        ),
    )
    parser.add_argument(
        "--year",
        type=int,
        help=(
            "Year to prepend to MM-DD HH:MM:SS timestamps. Defaults to the "
            "run directory year when available, otherwise the current year."
        ),
    )
    parser.add_argument(
        "--engine-ids",
        type=str,
        help="Engine subset, for example: 0,1,2 or 0-7 or 0-3,7,9-10.",
    )
    parser.add_argument(
        "--time-axis",
        choices=("relative", "absolute"),
        default="relative",
        help="Use relative seconds or absolute wall clock time on the x-axis.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=("aggregate", "per-engine", "both"),
        default="both",
        help="Which figures to render.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Output figure format.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Custom figure title prefix.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a matched stats line is missing required metrics.",
    )
    parser.add_argument(
        "--dump-csv",
        action="store_true",
        help="Write parsed and aggregate CSV files.",
    )
    args = parser.parse_args()
    if args.batch_root is not None and args.out_dir is not None:
        parser.error(
            "--out-dir is not supported with --batch-root; batch mode writes "
            "next to each frontend.log")
    return args


def infer_year_from_run_dir(run_dir: Path) -> int | None:
    match = RUN_DIR_YEAR_RE.match(run_dir.name)
    if match is None:
        return None
    return int(match.group("year"))


def _resolve_year(args: argparse.Namespace, inferred_year: int | None) -> int:
    if args.year is not None:
        return args.year
    return inferred_year or datetime.now().year


def _validate_log_path(log_path: Path) -> None:
    if log_path.name != "frontend.log":
        raise ValueError(f"expected a frontend.log input, got: {log_path}")
    if not log_path.is_file():
        raise FileNotFoundError(f"frontend.log not found: {log_path}")


def resolve_single_input(args: argparse.Namespace) -> JobSpec:
    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
        log_path = run_dir / "frontend.log"
        base_name = run_dir.name
        inferred_year = infer_year_from_run_dir(run_dir)
        default_out_dir = run_dir
        default_title = run_dir.name
    else:
        log_path = args.log.expanduser().resolve()
        base_name = log_path.stem
        inferred_year = infer_year_from_run_dir(log_path.parent)
        default_out_dir = log_path.parent
        default_title = log_path.stem

    _validate_log_path(log_path)
    out_dir = (args.out_dir.expanduser().resolve()
               if args.out_dir is not None else default_out_dir)
    year = _resolve_year(args, inferred_year)
    title = args.title or default_title

    return JobSpec(
        log_path=log_path,
        base_name=base_name,
        out_dir=out_dir,
        year=year,
        title=title,
    )


def discover_batch_inputs(args: argparse.Namespace) -> list[JobSpec]:
    batch_root = args.batch_root.expanduser().resolve()
    if not batch_root.is_dir():
        raise FileNotFoundError(f"batch root not found: {batch_root}")

    jobs: list[JobSpec] = []
    for log_path in sorted(batch_root.rglob("frontend.log")):
        run_dir = log_path.parent.resolve()
        relative_run_dir = run_dir.relative_to(batch_root)
        if len(relative_run_dir.parts) != 2:
            continue
        inferred_year = infer_year_from_run_dir(run_dir)
        if inferred_year is None:
            continue
        title_suffix = str(relative_run_dir)
        title = (f"{args.title} - {title_suffix}"
                 if args.title is not None else title_suffix)
        jobs.append(
            JobSpec(
                log_path=log_path.resolve(),
                base_name=run_dir.name,
                out_dir=run_dir,
                year=_resolve_year(args, inferred_year),
                title=title,
            ))
    if not jobs:
        raise ValueError(f"no frontend.log run directories found under {batch_root}")
    return jobs


def parse_engine_ids(spec: str | None) -> list[int] | None:
    if spec is None:
        return None

    engine_ids: set[int] = set()
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start_str, end_str = item.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"invalid engine range: {item}")
            engine_ids.update(range(start, end + 1))
            continue
        engine_ids.add(int(item))

    if not engine_ids:
        raise ValueError("engine id selection is empty")
    return sorted(engine_ids)


def _extract_int(pattern: re.Pattern[str], line: str) -> int | None:
    match = pattern.search(line)
    if match is None:
        return None
    return int(match.group("value"))


def _extract_float(pattern: re.Pattern[str], line: str) -> float | None:
    match = pattern.search(line)
    if match is None:
        return None
    return float(match.group("value"))


def parse_log(path: Path, year: int, strict: bool = False) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    malformed_lines: list[int] = []

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if "Engine " not in line or "GPU KV cache usage:" not in line:
                continue

            header = HEADER_RE.search(line)
            if header is None:
                continue

            waiting_tokens = _extract_int(WAITING_TOKENS_RE, line)
            waiting_head_tokens = _extract_int(WAITING_HEAD_TOKENS_RE, line)
            gpu_kv_usage = _extract_float(GPU_KV_USAGE_RE, line)

            if (waiting_tokens is None or waiting_head_tokens is None
                    or gpu_kv_usage is None):
                malformed_lines.append(line_number)
                if strict:
                    raise ValueError(
                        f"{path}: line {line_number} is missing required "
                        "waiting/GPU KV metrics")
                continue

            timestamp = datetime.strptime(
                f"{year}-{header.group('timestamp')}",
                "%Y-%m-%d %H:%M:%S",
            )
            records.append({
                "timestamp": timestamp,
                "engine_id": int(header.group("engine_id")),
                "waiting_tokens": waiting_tokens,
                "waiting_head_tokens": waiting_head_tokens,
                "gpu_kv_cache_usage_pct": gpu_kv_usage,
                "line_number": line_number,
            })

    if malformed_lines:
        warnings.warn(
            f"Skipped {len(malformed_lines)} malformed stats lines in {path}.",
            stacklevel=2,
        )

    if not records:
        raise ValueError(f"no engine stats were parsed from {path}")

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["timestamp", "engine_id", "line_number"]).reset_index(
        drop=True)
    return df


def filter_engine_ids(df: pd.DataFrame,
                      engine_ids: Iterable[int] | None) -> pd.DataFrame:
    if engine_ids is None:
        return df.copy()
    filtered = df[df["engine_id"].isin(list(engine_ids))].copy()
    if filtered.empty:
        raise ValueError("no rows remain after applying --engine-ids")
    return filtered


def add_relative_seconds(df: pd.DataFrame) -> pd.DataFrame:
    first_timestamp = df["timestamp"].min()
    df = df.copy()
    df["relative_seconds"] = (
        df["timestamp"] - first_timestamp).dt.total_seconds()
    return df


def build_metric_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="timestamp",
        columns="engine_id",
        values=metric,
        aggfunc="last",
    )
    return pivot.sort_index().sort_index(axis=1)


def build_aggregate_table(df: pd.DataFrame) -> pd.DataFrame:
    aggregate = (df.groupby("timestamp", as_index=False).agg(
        waiting_tokens_total=("waiting_tokens", "sum"),
        waiting_head_tokens_total=("waiting_head_tokens", "sum"),
        gpu_kv_cache_usage_mean=("gpu_kv_cache_usage_pct", "mean"),
        gpu_kv_cache_usage_max=("gpu_kv_cache_usage_pct", "max"),
        engine_count=("engine_id", "nunique"),
    ))
    return add_relative_seconds(aggregate)


def format_engine_summary(engine_ids: list[int]) -> str:
    if not engine_ids:
        return "No engines"
    if len(engine_ids) == 1:
        return f"Engine {engine_ids[0]}"
    return f"Engines {engine_ids[0]}-{engine_ids[-1]} ({len(engine_ids)})"


def split_engine_ids_for_plots(engine_ids: Iterable[int],
                               group_size: int =
                               PER_ENGINE_PLOT_GROUP_SIZE) -> list[list[int]]:
    ordered_engine_ids = list(map(int, engine_ids))
    return [
        ordered_engine_ids[index:index + group_size]
        for index in range(0, len(ordered_engine_ids), group_size)
    ]


def build_per_engine_output_path(output_path: Path, group_index: int,
                                 group_count: int) -> Path:
    if group_count == 1:
        return output_path
    width = len(str(group_count))
    return output_path.with_name(
        f"{output_path.stem}.part{group_index:0{width}d}{output_path.suffix}")


def configure_time_axis(ax: plt.Axes, x_values: pd.Series | pd.Index,
                        time_axis: str) -> None:
    if time_axis == "absolute":
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S"))
    ax.set_xlim(min(x_values), max(x_values))


def get_x_values(index: pd.Index, time_axis: str,
                 start_timestamp: datetime) -> pd.Index | pd.Series:
    if time_axis == "absolute":
        return index
    relative = (index - start_timestamp).total_seconds()
    return pd.Series(relative, index=index)


def apply_token_axis_style(ax: plt.Axes) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)


def finalize_figure(fig: plt.Figure, axes: list[plt.Axes], x_values,
                    time_axis: str, dpi: int, output_path: Path) -> None:
    xlabel = "Relative time (s)" if time_axis == "relative" else "Timestamp"
    axes[-1].set_xlabel(xlabel)
    if time_axis == "absolute":
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_aggregate_plot(
    aggregate_df: pd.DataFrame,
    title: str,
    time_axis: str,
    dpi: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"{title} - Aggregate")

    x_values = (aggregate_df["timestamp"] if time_axis == "absolute" else
                aggregate_df["relative_seconds"])

    axes[0].plot(x_values,
                 aggregate_df["waiting_tokens_total"],
                 color="#0b6e4f",
                 linewidth=1.8)
    axes[0].set_ylabel("Waiting tokens")
    axes[0].set_title("Waiting tokens total")
    axes[0].grid(alpha=0.3)
    apply_token_axis_style(axes[0])

    axes[1].plot(x_values,
                 aggregate_df["waiting_head_tokens_total"],
                 color="#c44900",
                 linewidth=1.8)
    axes[1].set_ylabel("Waiting head")
    axes[1].set_title("Waiting head tokens total")
    axes[1].grid(alpha=0.3)
    apply_token_axis_style(axes[1])

    axes[2].plot(x_values,
                 aggregate_df["gpu_kv_cache_usage_mean"],
                 color="#1f77b4",
                 linewidth=1.8,
                 label="Mean")
    axes[2].plot(x_values,
                 aggregate_df["gpu_kv_cache_usage_max"],
                 color="#d62728",
                 linewidth=1.8,
                 label="Max")
    axes[2].set_ylabel("GPU KV (%)")
    axes[2].set_title("GPU KV cache usage")
    axes[2].set_ylim(0, 100)
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right")

    for axis in axes:
        configure_time_axis(axis, x_values, time_axis)

    finalize_figure(fig, list(axes), x_values, time_axis, dpi, output_path)


def render_per_engine_plot(
    waiting_tokens_pivot: pd.DataFrame,
    waiting_head_tokens_pivot: pd.DataFrame,
    gpu_kv_usage_pivot: pd.DataFrame,
    engine_ids: list[int],
    title: str,
    time_axis: str,
    dpi: int,
    output_path: Path,
) -> None:
    waiting_tokens_pivot = waiting_tokens_pivot.loc[:, engine_ids]
    waiting_head_tokens_pivot = waiting_head_tokens_pivot.loc[:, engine_ids]
    gpu_kv_usage_pivot = gpu_kv_usage_pivot.loc[:, engine_ids]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    engine_summary = format_engine_summary(engine_ids)
    fig.suptitle(f"{title} - Per Engine - {engine_summary}")

    start_timestamp = min(waiting_tokens_pivot.index.min(),
                          waiting_head_tokens_pivot.index.min(),
                          gpu_kv_usage_pivot.index.min())
    x_values = get_x_values(waiting_tokens_pivot.index, time_axis,
                            start_timestamp)

    waiting_tokens_plot = waiting_tokens_pivot.fillna(0.0)
    waiting_head_plot = waiting_head_tokens_pivot.fillna(0.0)

    axes[0].stackplot(
        x_values,
        waiting_tokens_plot.to_numpy().T,
        alpha=0.9,
    )
    axes[0].set_ylabel("Waiting tokens")
    axes[0].set_title("Waiting tokens by engine")
    axes[0].grid(alpha=0.3)
    apply_token_axis_style(axes[0])

    axes[1].stackplot(
        x_values,
        waiting_head_plot.to_numpy().T,
        alpha=0.9,
    )
    axes[1].set_ylabel("Waiting head")
    axes[1].set_title("Waiting head tokens by engine")
    axes[1].grid(alpha=0.3)
    apply_token_axis_style(axes[1])

    gpu_x_values = get_x_values(gpu_kv_usage_pivot.index, time_axis,
                                start_timestamp)
    for engine_id in gpu_kv_usage_pivot.columns:
        axes[2].plot(
            gpu_x_values,
            gpu_kv_usage_pivot[engine_id],
            linewidth=0.9,
            alpha=0.3,
            color="#4c72b0",
        )
    axes[2].plot(
        gpu_x_values,
        gpu_kv_usage_pivot.mean(axis=1),
        linewidth=2.0,
        color="#d62728",
        label="Mean",
    )
    axes[2].set_ylabel("GPU KV (%)")
    axes[2].set_title("GPU KV cache usage by engine")
    axes[2].set_ylim(0, 100)
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right")
    axes[2].text(
        0.01,
        0.93,
        engine_summary,
        transform=axes[2].transAxes,
        fontsize=9,
        verticalalignment="top",
    )

    configure_time_axis(axes[0], x_values, time_axis)
    configure_time_axis(axes[1], x_values, time_axis)
    configure_time_axis(axes[2], gpu_x_values, time_axis)

    finalize_figure(fig, list(axes), x_values, time_axis, dpi, output_path)


def render_per_engine_plots(
    waiting_tokens_pivot: pd.DataFrame,
    waiting_head_tokens_pivot: pd.DataFrame,
    gpu_kv_usage_pivot: pd.DataFrame,
    title: str,
    time_axis: str,
    dpi: int,
    output_path: Path,
) -> None:
    engine_groups = split_engine_ids_for_plots(waiting_tokens_pivot.columns)
    for group_index, engine_ids in enumerate(engine_groups, start=1):
        render_per_engine_plot(
            waiting_tokens_pivot=waiting_tokens_pivot,
            waiting_head_tokens_pivot=waiting_head_tokens_pivot,
            gpu_kv_usage_pivot=gpu_kv_usage_pivot,
            engine_ids=engine_ids,
            title=title,
            time_axis=time_axis,
            dpi=dpi,
            output_path=build_per_engine_output_path(output_path, group_index,
                                                     len(engine_groups)),
        )


def dump_csv_outputs(parsed_df: pd.DataFrame, aggregate_df: pd.DataFrame,
                     out_dir: Path, base_name: str) -> None:
    parsed_path = out_dir / f"{base_name}.parsed.csv"
    aggregate_path = out_dir / f"{base_name}.aggregate.csv"
    parsed_df.to_csv(parsed_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)


def run_job(job: JobSpec, args: argparse.Namespace,
            selected_engine_ids: list[int] | None) -> None:
    parsed_df = parse_log(job.log_path, year=job.year, strict=args.strict)
    parsed_df = filter_engine_ids(parsed_df, selected_engine_ids)
    parsed_df = add_relative_seconds(parsed_df)
    aggregate_df = build_aggregate_table(parsed_df)

    waiting_tokens_pivot = build_metric_pivot(parsed_df, "waiting_tokens")
    waiting_head_tokens_pivot = build_metric_pivot(parsed_df,
                                                   "waiting_head_tokens")
    gpu_kv_usage_pivot = build_metric_pivot(parsed_df, "gpu_kv_cache_usage_pct")

    out_dir = job.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dump_csv:
        dump_csv_outputs(parsed_df, aggregate_df, out_dir, job.base_name)

    if args.plot_mode in ("aggregate", "both"):
        render_aggregate_plot(
            aggregate_df=aggregate_df,
            title=job.title,
            time_axis=args.time_axis,
            dpi=args.dpi,
            output_path=out_dir / f"{job.base_name}.aggregate.{args.format}",
        )
    if args.plot_mode in ("per-engine", "both"):
        render_per_engine_plots(
            waiting_tokens_pivot=waiting_tokens_pivot,
            waiting_head_tokens_pivot=waiting_head_tokens_pivot,
            gpu_kv_usage_pivot=gpu_kv_usage_pivot,
            title=job.title,
            time_axis=args.time_axis,
            dpi=args.dpi,
            output_path=out_dir / f"{job.base_name}.per_engine.{args.format}",
        )


def main() -> None:
    args = parse_args()
    selected_engine_ids = parse_engine_ids(args.engine_ids)

    if args.batch_root is None:
        run_job(
            job=resolve_single_input(args),
            args=args,
            selected_engine_ids=selected_engine_ids,
        )
        return

    jobs = discover_batch_inputs(args)
    processed = 0
    skipped: list[tuple[Path, str]] = []
    for job in jobs:
        try:
            run_job(job=job, args=args, selected_engine_ids=selected_engine_ids)
            processed += 1
        except ValueError as exc:
            if args.strict:
                raise
            skipped.append((job.log_path, str(exc)))

    if processed == 0:
        raise ValueError(
            f"no valid frontend.log runs were processed under {args.batch_root}")

    print(f"Processed {processed} run(s) under {args.batch_root}.")
    if skipped:
        print(f"Skipped {len(skipped)} run(s) with unparsable frontend.log files.")


if __name__ == "__main__":
    main()
