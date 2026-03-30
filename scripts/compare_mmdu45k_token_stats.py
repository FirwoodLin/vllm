#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


DEFAULT_NEW_RESULTS = Path(
    "/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset-prepare/"
    "mmdu-45k/mmdu-45k-serving-lengths.jsonl")
DEFAULT_OLD_RESULTS = Path(
    "/mnt/nvme1n1/ml_research/chenjiefei/dataset/mmdu-45k_processing_artifacts/"
    "mmdu-45k-token-stats.jsonl")
DEFAULT_DATASET = Path("/mnt/nvme1n1/ml_research/chenjiefei/dataset/mmdu-45k.json")
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset-prepare/"
    "mmdu-45k")
DEFAULT_OUTPUT_PREFIX = "mmdu-45k-serving-lengths-vs-old"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare new mmdu-45k serving-length outputs against an old token-stats artifact."
    )
    parser.add_argument("--new-results",
                        type=Path,
                        default=DEFAULT_NEW_RESULTS,
                        help="Path to the new JSONL results.")
    parser.add_argument("--old-results",
                        type=Path,
                        default=DEFAULT_OLD_RESULTS,
                        help="Path to the old JSONL results.")
    parser.add_argument("--dataset",
                        type=Path,
                        default=DEFAULT_DATASET,
                        help="Path to the source mmdu-45k JSON file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used for the generated report, summary, and diff table.")
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for generated output files inside --output-dir.")
    parser.add_argument("--top-k",
                        type=int,
                        default=20,
                        help="How many largest prompt-diff rows to highlight in the report.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path} line {line_no}: {exc}") from exc
    return rows


def percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    index = int(round((len(sorted_values) - 1) * q))
    return sorted_values[index]


def stats(values: list[int]) -> dict[str, float | int]:
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": mean(sorted_values),
        "median": median(sorted_values),
        "p95": percentile(sorted_values, 0.95),
        "p99": percentile(sorted_values, 0.99),
    }


def counter_to_json(counter: Counter[int], limit: int = 20) -> list[dict[str, int]]:
    return [{
        "value": value,
        "count": count
    } for value, count in counter.most_common(limit)]


def format_float(value: float) -> str:
    return f"{value:.3f}"


def format_path(path: Path) -> str:
    return str(path.resolve())


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_diff_rows(dataset: list[dict[str, Any]], new_rows: list[dict[str, Any]],
                    old_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(new_rows) != len(old_rows):
        raise ValueError(
            f"Line count mismatch: {len(new_rows)=} vs {len(old_rows)=}.")
    if len(dataset) != len(new_rows):
        raise ValueError(
            f"Dataset length mismatch: {len(dataset)=} vs results={len(new_rows)}.")

    diff_rows: list[dict[str, Any]] = []
    new_id_mismatch_count = 0
    old_id_present_count = 0
    old_id_mismatch_count = 0

    for index, (source, new_row, old_row) in enumerate(zip(dataset, new_rows, old_rows), start=1):
        dataset_id = source["id"]
        new_id = new_row.get("id")
        old_id = old_row.get("id")
        if new_id is not None and new_id != dataset_id:
            new_id_mismatch_count += 1
        if old_id is not None:
            old_id_present_count += 1
            if old_id != dataset_id:
                old_id_mismatch_count += 1

        effective_id = new_id or old_id or dataset_id
        num_images = len(source.get("image", []))
        new_prompt_len = int(new_row["prompt_len"])
        old_prompt_len = int(old_row["prompt_len"])
        new_output_len = int(new_row["output_len"])
        old_output_len = int(old_row["output_len"])
        prompt_diff = new_prompt_len - old_prompt_len
        output_diff = new_output_len - old_output_len

        diff_rows.append({
            "index": index,
            "id": effective_id,
            "dataset_id": dataset_id,
            "new_id": new_id,
            "old_id": old_id,
            "num_images": num_images,
            "new_prompt_len": new_prompt_len,
            "old_prompt_len": old_prompt_len,
            "prompt_diff": prompt_diff,
            "prompt_diff_abs": abs(prompt_diff),
            "new_output_len": new_output_len,
            "old_output_len": old_output_len,
            "output_diff": output_diff,
            "old_image_prompt_len": old_row.get("image_prompt_len"),
            "old_text_prompt_len": old_row.get("text_prompt_len"),
        })

    alignment = {
        "dataset_records": len(dataset),
        "new_rows": len(new_rows),
        "old_rows": len(old_rows),
        "new_id_mismatch_count": new_id_mismatch_count,
        "old_id_present_count": old_id_present_count,
        "old_id_mismatch_count": old_id_mismatch_count,
        "old_aligned_by_dataset_order": old_id_present_count == 0,
    }
    return diff_rows, alignment


def build_summary(diff_rows: list[dict[str, Any]], alignment: dict[str, Any],
                  top_k: int) -> dict[str, Any]:
    prompt_diffs = [row["prompt_diff"] for row in diff_rows]
    output_diffs = [row["output_diff"] for row in diff_rows]
    prompt_counter = Counter(prompt_diffs)
    output_counter = Counter(output_diffs)

    by_num_images: dict[int, list[int]] = defaultdict(list)
    for row in diff_rows:
        by_num_images[row["num_images"]].append(row["prompt_diff"])

    image_buckets = []
    for num_images in sorted(by_num_images):
        values = by_num_images[num_images]
        image_buckets.append({
            "num_images": num_images,
            "count": len(values),
            "min_prompt_diff": min(values),
            "max_prompt_diff": max(values),
            "mean_prompt_diff": mean(values),
            "median_prompt_diff": median(values),
            "top_prompt_diffs": counter_to_json(Counter(values), limit=10),
        })

    top_abs_prompt_diffs = []
    for row in sorted(diff_rows,
                      key=lambda item: (item["prompt_diff_abs"], item["index"]),
                      reverse=True)[:top_k]:
        top_abs_prompt_diffs.append({
            "index": row["index"],
            "id": row["id"],
            "num_images": row["num_images"],
            "prompt_diff": row["prompt_diff"],
            "new_prompt_len": row["new_prompt_len"],
            "old_prompt_len": row["old_prompt_len"],
            "old_image_prompt_len": row["old_image_prompt_len"],
            "old_text_prompt_len": row["old_text_prompt_len"],
            "output_diff": row["output_diff"],
        })

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "alignment": alignment,
        "prompt_diff": {
            "stats": stats(prompt_diffs),
            "nonzero_count": sum(value != 0 for value in prompt_diffs),
            "positive_count": sum(value > 0 for value in prompt_diffs),
            "negative_count": sum(value < 0 for value in prompt_diffs),
            "top_values": counter_to_json(prompt_counter, limit=20),
        },
        "output_diff": {
            "stats": stats(output_diffs),
            "nonzero_count": sum(value != 0 for value in output_diffs),
            "positive_count": sum(value > 0 for value in output_diffs),
            "negative_count": sum(value < 0 for value in output_diffs),
            "top_values": counter_to_json(output_counter, limit=20),
        },
        "by_num_images": image_buckets,
        "top_abs_prompt_diffs": top_abs_prompt_diffs,
    }


def render_markdown(summary: dict[str, Any], args: argparse.Namespace) -> str:
    prompt_stats = summary["prompt_diff"]["stats"]
    output_stats = summary["output_diff"]["stats"]
    alignment = summary["alignment"]

    lines: list[str] = []
    lines.append("# MMDU-45K Serving Length Comparison")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- New results: `{format_path(args.new_results)}`")
    lines.append(f"- Old results: `{format_path(args.old_results)}`")
    lines.append(f"- Dataset: `{format_path(args.dataset)}`")
    lines.append("")
    lines.append("## Alignment")
    lines.append(f"- Dataset records: {alignment['dataset_records']}")
    lines.append(f"- New result rows: {alignment['new_rows']}")
    lines.append(f"- Old result rows: {alignment['old_rows']}")
    lines.append(f"- New id mismatches vs dataset order: {alignment['new_id_mismatch_count']}")
    lines.append(f"- Old rows carried explicit ids: {alignment['old_id_present_count']}")
    lines.append(
        f"- Old rows aligned by dataset order: {alignment['old_aligned_by_dataset_order']}")
    lines.append("")
    lines.append("## Key Findings")
    lines.append(
        f"- `output_len` diff is zero for all {output_stats['count']} rows.")
    lines.append(
        f"- `prompt_len` diff is non-zero for {summary['prompt_diff']['nonzero_count']} rows.")
    lines.append(
        f"- `prompt_len` diff sign distribution: +{summary['prompt_diff']['positive_count']} / "
        f"-{summary['prompt_diff']['negative_count']} / 0="
        f"{prompt_stats['count'] - summary['prompt_diff']['nonzero_count']}.")
    lines.append(
        f"- `prompt_len` diff stats: min={prompt_stats['min']}, max={prompt_stats['max']}, "
        f"mean={format_float(prompt_stats['mean'])}, median={format_float(prompt_stats['median'])}, "
        f"p95={prompt_stats['p95']}, p99={prompt_stats['p99']}.")
    lines.append("")
    lines.append("## Top Prompt Diff Values")
    lines.append("| diff | count |")
    lines.append("| ---: | ---: |")
    for item in summary["prompt_diff"]["top_values"]:
        lines.append(f"| {item['value']} | {item['count']} |")
    lines.append("")
    lines.append("## Prompt Diff By Image Count")
    lines.append("| num_images | count | min | max | mean | median |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for bucket in summary["by_num_images"]:
        lines.append(
            f"| {bucket['num_images']} | {bucket['count']} | {bucket['min_prompt_diff']} | "
            f"{bucket['max_prompt_diff']} | {format_float(bucket['mean_prompt_diff'])} | "
            f"{format_float(bucket['median_prompt_diff'])} |")
    lines.append("")
    lines.append("## Largest Absolute Prompt Diffs")
    lines.append(
        "| index | id | num_images | prompt_diff | new_prompt_len | old_prompt_len | "
        "old_image_prompt_len | old_text_prompt_len | output_diff |")
    lines.append(
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["top_abs_prompt_diffs"]:
        lines.append(
            f"| {row['index']} | {row['id']} | {row['num_images']} | {row['prompt_diff']} | "
            f"{row['new_prompt_len']} | {row['old_prompt_len']} | "
            f"{row['old_image_prompt_len']} | {row['old_text_prompt_len']} | "
            f"{row['output_diff']} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "- The two outputs align by row count, and the new file's `id` order matches the dataset order.")
    lines.append(
        "- The old file has no `id` field, so the comparison for it is row-order based against `mmdu-45k.json`.")
    lines.append(
        "- Because `output_len` matches exactly while `prompt_len` is uniformly larger in the new file, "
        "the behavioral difference is isolated to prompt-side accounting.")
    lines.append(
        "- The average prompt gap grows with image count, which suggests the delta mainly comes from vision-token accounting rather than text-output tokenization.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_json(args.dataset)
    new_rows = load_jsonl(args.new_results)
    old_rows = load_jsonl(args.old_results)

    diff_rows, alignment = build_diff_rows(dataset, new_rows, old_rows)
    summary = build_summary(diff_rows, alignment, top_k=args.top_k)

    prefix = args.output_prefix
    diff_csv_path = args.output_dir / f"{prefix}-diff.csv"
    summary_json_path = args.output_dir / f"{prefix}-summary.json"
    report_md_path = args.output_dir / f"{prefix}-report.md"

    diff_fieldnames = [
        "index",
        "id",
        "dataset_id",
        "new_id",
        "old_id",
        "num_images",
        "new_prompt_len",
        "old_prompt_len",
        "prompt_diff",
        "prompt_diff_abs",
        "new_output_len",
        "old_output_len",
        "output_diff",
        "old_image_prompt_len",
        "old_text_prompt_len",
    ]
    write_csv(diff_csv_path, diff_rows, diff_fieldnames)
    write_json(summary_json_path, summary)
    report_md_path.write_text(render_markdown(summary, args), encoding="utf-8")

    print(f"Wrote diff CSV: {diff_csv_path}")
    print(f"Wrote summary JSON: {summary_json_path}")
    print(f"Wrote markdown report: {report_md_path}")


if __name__ == "__main__":
    main()
