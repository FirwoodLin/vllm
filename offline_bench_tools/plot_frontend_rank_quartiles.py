#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
KV_CACHE_SIZE_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
ENGINE_STATS_RE = re.compile(
    r"Engine\s+(\d+):.*?Running:\s+(\d+)\s+reqs.*?"
    r"(?:GPU KV cache usage:\s+([0-9.]+)%|Free KV blocks:\s+(\d+))"
)
TIME_POSITION_PCTS = (25, 50, 60, 66, 75)


@dataclass(frozen=True)
class EnginePoint:
    running: float
    kv_usage_pct: float


@dataclass(frozen=True)
class ParsedSnapshots:
    running_by_step: list[list[float]]
    kv_usage_by_step: list[list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse frontend.log and plot per-engine rank bar charts at "
            "25% / 50% / 60% / 66% / 75% time positions for Running and "
            "GPU KV usage."
        )
    )
    parser.add_argument(
        "frontend_log",
        type=Path,
        help="Path to frontend.log",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots. Default: same directory as frontend.log",
    )
    parser.add_argument(
        "--num-ranks",
        type=int,
        default=32,
        help="Expected number of engine ranks (default: 32)",
    )
    parser.add_argument(
        "--block-size-tokens",
        type=int,
        default=64,
        help="Token count per KV block for old logs with only Free KV blocks.",
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
        default=220,
        help="Figure DPI (default: 220).",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def parse_frontend_snapshots(
    frontend_log: Path,
    *,
    num_ranks: int,
    block_size_tokens: int,
) -> ParsedSnapshots:
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

    expected_ids = tuple(range(num_ranks))
    complete_snapshots = [
        snapshot
        for snapshot in snapshots
        if tuple(sorted(snapshot.keys())) == expected_ids
    ]

    if not complete_snapshots:
        raise ValueError(
            f"{frontend_log}: no complete snapshots with rank IDs "
            f"{expected_ids[0]}..{expected_ids[-1]}"
        )

    running_by_step = [
        [snapshot[rank].running for rank in expected_ids]
        for snapshot in complete_snapshots
    ]
    kv_usage_by_step = [
        [snapshot[rank].kv_usage_pct for rank in expected_ids]
        for snapshot in complete_snapshots
    ]
    return ParsedSnapshots(
        running_by_step=running_by_step,
        kv_usage_by_step=kv_usage_by_step,
    )


def quartile_step_indices(num_steps: int) -> list[tuple[int, int]]:
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")
    return [
        (pct, int(round((num_steps - 1) * (pct / 100.0))))
        for pct in TIME_POSITION_PCTS
    ]


def plot_metric_quartiles(
    values_by_step: list[list[float]],
    *,
    metric_name: str,
    ylabel: str,
    output_path: Path,
    y_max: float | None,
    dpi: int,
) -> Path:
    if not values_by_step:
        raise ValueError(f"{metric_name}: empty values")

    num_steps = len(values_by_step)
    num_ranks = len(values_by_step[0])
    rank_ids = np.arange(num_ranks)
    target_steps = quartile_step_indices(num_steps)

    fig, axes = plt.subplots(
        1,
        len(target_steps),
        figsize=(7 * len(target_steps), 5.5),
        sharey=True,
        constrained_layout=True,
    )
    fig.suptitle(
        f"{metric_name} @ {' / '.join(f'{pct}%' for pct in TIME_POSITION_PCTS)} Time Position",
        fontsize=14,
    )

    bar_color = "#4C78A8"
    mean_color = "#E45756"

    for ax, (pct, step_idx) in zip(axes, target_steps):
        values = values_by_step[step_idx]
        mean_value = fmean(values)

        ax.bar(rank_ids, values, color=bar_color, alpha=0.85)
        ax.axhline(
            mean_value,
            color=mean_color,
            linestyle="--",
            linewidth=1.6,
            label=f"mean={mean_value:.2f}",
        )
        ax.set_title(f"T={pct}% (step {step_idx + 1}/{num_steps})")
        ax.set_xlabel("engine rank")
        ax.set_xticks(rank_ids)
        ax.set_xticklabels([str(i) for i in rank_ids], rotation=90)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="upper right")

    axes[0].set_ylabel(ylabel)
    if y_max is not None:
        axes[0].set_ylim(0, y_max)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_quartile_report(
    *,
    running_by_step: list[list[float]],
    kv_usage_by_step: list[list[float]],
    output_path: Path,
) -> Path:
    if not running_by_step or not kv_usage_by_step:
        raise ValueError("cannot export quartile data from empty snapshots")
    if len(running_by_step) != len(kv_usage_by_step):
        raise ValueError("running and kv usage snapshots have mismatched length")

    num_steps = len(running_by_step)
    if len(running_by_step[0]) != len(kv_usage_by_step[0]):
        raise ValueError("running and kv usage rank count mismatch")

    target_steps = quartile_step_indices(num_steps)
    lines: list[str] = []
    lines.append(f"num_steps={num_steps}")
    lines.append(f"num_ranks={len(running_by_step[0])}")
    lines.append("")

    for pct, step_idx in target_steps:
        running_values = running_by_step[step_idx]
        kv_values = kv_usage_by_step[step_idx]
        running_mean = fmean(running_values)
        kv_mean = fmean(kv_values)

        lines.append(f"[time={pct}% step={step_idx + 1}/{num_steps}]")
        lines.append(f"running_mean={running_mean:.6f}")
        lines.append(f"gpu_kv_usage_mean={kv_mean:.6f}")
        lines.append("rank\trunning\tgpu_kv_usage_pct")
        for rank, (running, kv_usage) in enumerate(zip(running_values, kv_values)):
            lines.append(f"{rank}\t{running:.6f}\t{kv_usage:.6f}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    frontend_log = args.frontend_log.resolve()
    if not frontend_log.is_file():
        raise FileNotFoundError(f"frontend.log does not exist: {frontend_log}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else frontend_log.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_frontend_snapshots(
        frontend_log,
        num_ranks=args.num_ranks,
        block_size_tokens=args.block_size_tokens,
    )

    running_path = output_dir / f"{frontend_log.stem}_running_rank_quartiles.{args.format}"
    kv_path = output_dir / f"{frontend_log.stem}_gpu_kv_usage_rank_quartiles.{args.format}"
    txt_path = output_dir / f"{frontend_log.stem}_rank_quartiles_data.txt"

    plot_metric_quartiles(
        parsed.running_by_step,
        metric_name="Running Requests",
        ylabel="Running reqs",
        output_path=running_path,
        y_max=None,
        dpi=args.dpi,
    )
    plot_metric_quartiles(
        parsed.kv_usage_by_step,
        metric_name="GPU KV Cache Usage",
        ylabel="Usage (%)",
        output_path=kv_path,
        y_max=100.0,
        dpi=args.dpi,
    )
    write_quartile_report(
        running_by_step=parsed.running_by_step,
        kv_usage_by_step=parsed.kv_usage_by_step,
        output_path=txt_path,
    )

    print(running_path)
    print(kv_path)
    print(txt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
