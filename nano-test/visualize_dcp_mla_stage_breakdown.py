#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BACKEND_ORDER = ["a2a", "ag_rs"]
TIME_COLUMNS_US = [
    "pre_comm_us",
    "attention_compute_us",
    "post_dcp_comm_us",
    "post_tp_all_reduce_us",
    "post_comm_us",
    "comm_total_us",
    "dcp_stage_total_us",
]
POST_COMPONENT_BASES = {
    "a2a": [
        ("post_sendrecv_output", "sendrecv_output"),
        ("post_sendrecv_lse", "sendrecv_lse"),
        ("post_dcp_lse_combine", "lse_combine"),
        ("post_tp_all_reduce", "tp_all_reduce"),
    ],
    "ag_rs": [
        ("post_lse_all_gather", "lse_all_gather"),
        ("post_correct_attn_cp_out", "correct_attn_cp_out"),
        ("post_reduce_scatter", "reduce_scatter"),
        ("post_tp_all_reduce", "tp_all_reduce"),
    ],
}
STAGE_COLORS = {
    "pre": "#E67E22",
    "attention": "#2E8B57",
    "post": "#1F77B4",
    "tp_all_reduce": "#7D3C98",
    "comm_total": "#C0392B",
    "total": "#4D4D4D",
}
POST_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize DCP MLA stage timing trends from the exported CSV."
    )
    parser.add_argument(
        "--csv",
        default="nano-test/dcp_mla_stage_breakdown_nvtx.csv",
        help="Input CSV exported by analyze_dcp_nsys.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="nano-test/plots",
        help="Directory for generated plots and derived CSV.",
    )
    parser.add_argument(
        "--mode",
        choices=["aggregate", "per_attention"],
        default="aggregate",
        help="Plot aggregate totals or single-attention averages.",
    )
    return parser.parse_args()


def column_for_mode(base_name: str, mode: str) -> str:
    if mode == "per_attention":
        return f"{base_name}_per_attention_us"
    return f"{base_name}_us"


def validate_columns(df: pd.DataFrame, mode: str) -> None:
    required = {
        "backend",
        "bs",
        column_for_mode("pre_query_all_gather", mode),
        column_for_mode("mla_core", mode),
        column_for_mode("post_total", mode),
        column_for_mode("dcp_stage_total", mode),
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")


def enrich(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    df["backend"] = pd.Categorical(df["backend"], categories=BACKEND_ORDER, ordered=True)
    df["bs"] = df["bs"].astype(int)
    post_tp_col = column_for_mode("post_tp_all_reduce", mode)
    if post_tp_col not in df.columns:
        df[post_tp_col] = 0.0
    df["pre_comm_us"] = df[column_for_mode("pre_query_all_gather", mode)]
    df["attention_compute_us"] = df[column_for_mode("mla_core", mode)]
    df["post_tp_all_reduce_us"] = df[post_tp_col].astype(float)
    df["post_comm_us"] = df[column_for_mode("post_total", mode)]
    df["post_dcp_comm_us"] = (
        df["post_comm_us"] - df["post_tp_all_reduce_us"]
    ).clip(lower=0.0)
    df["comm_total_us"] = df["pre_comm_us"] + df["post_comm_us"]
    df["dcp_stage_total_us"] = df[column_for_mode("dcp_stage_total", mode)]
    df["pre_comm_share_pct"] = df["pre_comm_us"] / df["dcp_stage_total_us"] * 100.0
    df["attention_compute_share_pct"] = (
        df["attention_compute_us"] / df["dcp_stage_total_us"] * 100.0
    )
    df["post_dcp_comm_share_pct"] = (
        df["post_dcp_comm_us"] / df["dcp_stage_total_us"] * 100.0
    )
    df["post_tp_all_reduce_share_pct"] = (
        df["post_tp_all_reduce_us"] / df["dcp_stage_total_us"] * 100.0
    )
    df["post_comm_share_pct"] = df["post_comm_us"] / df["dcp_stage_total_us"] * 100.0
    df["comm_total_share_pct"] = df["comm_total_us"] / df["dcp_stage_total_us"] * 100.0

    unit_label = "us" if mode == "per_attention" else "ms"
    unit_scale = 1.0 if mode == "per_attention" else 1.0 / 1000.0
    for col in TIME_COLUMNS_US:
        df[f"{col}_plot"] = df[col] * unit_scale

    return df.sort_values(["backend", "bs"]).reset_index(drop=True), unit_label


def setup_style() -> None:
    plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def plot_stage_trends(df: pd.DataFrame, output_path: Path, unit_label: str, mode: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    for row, backend in enumerate(BACKEND_ORDER):
        sub = df[df["backend"] == backend].sort_values("bs")
        x = sub["bs"]

        abs_ax = axes[row, 0]
        abs_ax.plot(
            x,
            sub["pre_comm_us_plot"],
            marker="o",
            color=STAGE_COLORS["pre"],
            label="pre_comm",
        )
        abs_ax.plot(
            x,
            sub["attention_compute_us_plot"],
            marker="o",
            color=STAGE_COLORS["attention"],
            label="attention_compute",
        )
        abs_ax.plot(
            x,
            sub["post_comm_us_plot"],
            marker="o",
            color=STAGE_COLORS["post"],
            label="post_comm",
        )
        abs_ax.plot(
            x,
            sub["post_tp_all_reduce_us_plot"],
            marker="o",
            linestyle=":",
            color=STAGE_COLORS["tp_all_reduce"],
            label="tp_all_reduce",
        )
        abs_ax.plot(
            x,
            sub["comm_total_us_plot"],
            marker="o",
            linestyle="--",
            color=STAGE_COLORS["comm_total"],
            label="comm_total",
        )
        abs_ax.plot(
            x,
            sub["dcp_stage_total_us_plot"],
            marker="o",
            linewidth=2.0,
            color=STAGE_COLORS["total"],
            label="stage_total",
        )
        title_suffix = "single-attention time trend" if mode == "per_attention" else "stage time trend"
        abs_ax.set_title(f"{backend}: {title_suffix}")
        abs_ax.set_xlabel("Batch size")
        abs_ax.set_ylabel(f"Time ({unit_label})")
        abs_ax.legend(loc="upper left")

        share_ax = axes[row, 1]
        share_ax.plot(
            x,
            sub["pre_comm_share_pct"],
            marker="o",
            color=STAGE_COLORS["pre"],
            label="pre_comm_pct",
        )
        share_ax.plot(
            x,
            sub["attention_compute_share_pct"],
            marker="o",
            color=STAGE_COLORS["attention"],
            label="attention_pct",
        )
        share_ax.plot(
            x,
            sub["post_comm_share_pct"],
            marker="o",
            color=STAGE_COLORS["post"],
            label="post_comm_pct",
        )
        share_ax.plot(
            x,
            sub["post_tp_all_reduce_share_pct"],
            marker="o",
            linestyle=":",
            color=STAGE_COLORS["tp_all_reduce"],
            label="tp_all_reduce_pct",
        )
        share_ax.plot(
            x,
            sub["comm_total_share_pct"],
            marker="o",
            linestyle="--",
            color=STAGE_COLORS["comm_total"],
            label="comm_total_pct",
        )
        share_ax.set_title(f"{backend}: time share trend")
        share_ax.set_xlabel("Batch size")
        share_ax.set_ylabel("Share of stage total (%)")
        share_ax.set_ylim(0, 100)
        share_ax.legend(loc="upper left")

    overview = "DCP MLA single-attention overview" if mode == "per_attention" else "DCP MLA stage trend overview"
    fig.suptitle(overview, fontsize=14)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_stage_stacked(df: pd.DataFrame, output_path: Path, unit_label: str, mode: str) -> None:
    ordered = df.sort_values(["backend", "bs"]).copy()
    labels = [f"{backend}\nbs={bs}" for backend, bs in zip(ordered["backend"], ordered["bs"])]
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    ax.bar(labels, ordered["pre_comm_us_plot"], color=STAGE_COLORS["pre"], label="pre_comm")
    ax.bar(
        labels,
        ordered["attention_compute_us_plot"],
        bottom=ordered["pre_comm_us_plot"],
        color=STAGE_COLORS["attention"],
        label="attention_compute",
    )
    ax.bar(
        labels,
        ordered["post_dcp_comm_us_plot"],
        bottom=ordered["pre_comm_us_plot"] + ordered["attention_compute_us_plot"],
        color=STAGE_COLORS["post"],
        label="post_dcp_comm",
    )
    ax.bar(
        labels,
        ordered["post_tp_all_reduce_us_plot"],
        bottom=(
            ordered["pre_comm_us_plot"]
            + ordered["attention_compute_us_plot"]
            + ordered["post_dcp_comm_us_plot"]
        ),
        color=STAGE_COLORS["tp_all_reduce"],
        label="tp_all_reduce",
    )

    title = "DCP MLA single-attention breakdown" if mode == "per_attention" else "DCP MLA stage breakdown"
    ax.set_title(title)
    ax.set_xlabel("Backend and batch size")
    ax.set_ylabel(f"Time ({unit_label})")
    ax.legend(loc="upper left")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_post_breakdown(df: pd.DataFrame, output_path: Path, unit_label: str, mode: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

    for ax, backend in zip(axes, BACKEND_ORDER):
        sub = df[df["backend"] == backend].sort_values("bs")
        x = [str(bs) for bs in sub["bs"]]
        bottom = pd.Series([0.0] * len(sub), index=sub.index)

        for color, (base_name, label) in zip(POST_COLORS, POST_COMPONENT_BASES[backend]):
            values = sub[column_for_mode(base_name, mode)] * (
                1.0 if mode == "per_attention" else 1.0 / 1000.0
            )
            ax.bar(x, values, bottom=bottom, color=color, label=label)
            bottom = bottom + values

        title = "post breakdown per attention" if mode == "per_attention" else "post communication breakdown"
        ax.set_title(f"{backend}: {title}")
        ax.set_xlabel("Batch size")
        ax.set_ylabel(f"Time ({unit_label})")
        ax.legend(loc="upper left")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_derived_csv(df: pd.DataFrame, output_path: Path) -> None:
    selected = [
        "backend",
        "bs",
        "dcp_stage_total_us",
        "pre_comm_us",
        "attention_compute_us",
        "post_dcp_comm_us",
        "post_tp_all_reduce_us",
        "post_comm_us",
        "comm_total_us",
        "pre_comm_us_plot",
        "attention_compute_us_plot",
        "post_dcp_comm_us_plot",
        "post_tp_all_reduce_us_plot",
        "post_comm_us_plot",
        "comm_total_us_plot",
        "dcp_stage_total_us_plot",
        "pre_comm_share_pct",
        "attention_compute_share_pct",
        "post_dcp_comm_share_pct",
        "post_tp_all_reduce_share_pct",
        "post_comm_share_pct",
        "comm_total_share_pct",
    ]
    df[selected].to_csv(output_path, index=False)


def print_summary(df: pd.DataFrame) -> None:
    summary = df[
        [
            "backend",
            "bs",
            "pre_comm_us_plot",
            "attention_compute_us_plot",
            "post_dcp_comm_us_plot",
            "post_tp_all_reduce_us_plot",
            "post_comm_us_plot",
            "comm_total_us_plot",
            "dcp_stage_total_us_plot",
            "pre_comm_share_pct",
            "attention_compute_share_pct",
            "post_dcp_comm_share_pct",
            "post_tp_all_reduce_share_pct",
            "post_comm_share_pct",
            "comm_total_share_pct",
        ]
    ].copy()
    print(summary.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    validate_columns(df, args.mode)
    df, unit_label = enrich(df, args.mode)
    setup_style()

    export_derived_csv(df, output_dir / "dcp_mla_stage_breakdown_derived.csv")
    plot_stage_trends(df, output_dir / "dcp_mla_stage_trends.png", unit_label, args.mode)
    plot_stage_stacked(
        df,
        output_dir / "dcp_mla_stage_breakdown_stacked.png",
        unit_label,
        args.mode,
    )
    plot_post_breakdown(
        df,
        output_dir / "dcp_mla_post_comm_breakdown.png",
        unit_label,
        args.mode,
    )
    print_summary(df)
    print(f"\nWrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
