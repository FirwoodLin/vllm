#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

DISPATCH_POLICIES = (
    "waiting_x4_plus_running",
    "least_cache",
    "least_batch",
)
DEFAULT_LENGTH_CSV_STEM = "custom_lens"
DEFAULT_CASE_CSV_NAME = "custom_lens.casecsv"


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


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("Expected a positive float.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an offline_dp_profile case from a nested input-length JSON "
            "file by generating a prompt_len/output_len CSV and a derived casecsv."
        ))
    parser.add_argument(
        "--base-case-csv",
        required=True,
        help="Template casecsv to copy and override.",
    )
    parser.add_argument(
        "--lens-json",
        required=True,
        help="JSON file containing integers or nested integer lists.",
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
        "--warmup-requests",
        type=non_negative_int,
        default=None,
        help="Override warmup_requests in the derived casecsv.",
    )
    parser.add_argument(
        "--max-requests",
        type=non_negative_int,
        default=None,
        help="Override max_requests in the derived casecsv.",
    )
    parser.add_argument(
        "--request-rate",
        type=positive_float,
        default=None,
        help="Override request_rate in the derived casecsv.",
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
        help="Optional derived case name. Defaults to <base_name>__<lens_stem>.",
    )
    return parser.parse_args()


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
    lens_json = Path(args.lens_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    fieldnames, base_row = read_base_case(base_case_csv)
    input_lengths = load_input_lengths(lens_json)

    derived_row = dict(base_row)
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
    derived_case_name = args.case_name or (
        f"{base_case_name}__{lens_json.stem}__{dispatch_tag}"
    )
    derived_row["name"] = derived_case_name
    derived_row["dataset"] = str(length_csv_path)
    if args.warmup_requests is not None:
        derived_row["warmup_requests"] = str(args.warmup_requests)
    if args.max_requests is not None:
        derived_row["max_requests"] = str(args.max_requests)
    if args.request_rate is not None:
        derived_row["request_rate"] = f"{args.request_rate:g}"
    if args.dispatch_policy is not None:
        derived_row["dispatch_policy"] = args.dispatch_policy

    write_length_csv(length_csv_path, input_lengths, args.output_len)
    write_case_csv(case_csv_path, fieldnames, derived_row)

    print("Prepared offline profile inputs from custom lens JSON")
    print(f"  base_case_csv: {base_case_csv}")
    print(f"  lens_json: {lens_json}")
    print(f"  flattened_lengths: {len(input_lengths)}")
    print(f"  output_len: {args.output_len}")
    print(f"  length_csv: {length_csv_path}")
    print(f"  case_csv: {case_csv_path}")
    print(f"  case_name: {derived_row['name']}")
    print(f"  dispatch_policy: {derived_row.get('dispatch_policy', '')}")
    print(f"  request_rate: {derived_row.get('request_rate', '')}")
    print(f"  warmup_requests: {derived_row.get('warmup_requests', '')}")
    print(f"  max_requests: {derived_row.get('max_requests', '')}")


if __name__ == "__main__":
    main()
