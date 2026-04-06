#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate editable CSV plans for manual multi-node Poisson benchmarks."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path

import manual_multinode_poisson_runner as runner


@dataclass(frozen=True)
class RateHistory:
    case: runner.ExperimentCase
    success_reference: Path | None
    success_tpot_by_e2e_mean: float | None
    latest_status: str | None
    status_reference: Path | None


@dataclass(frozen=True)
class PlannedCaseRow:
    case: runner.ExperimentCase
    reason: str
    historical_reference: str


def threshold_tag() -> str:
    threshold = runner.TPOT_BY_E2E_EARLY_STOP_MS
    if math.isclose(threshold, round(threshold)):
        return str(int(round(threshold)))
    return f"{threshold:g}"


def ordered_strategies(values: list[str] | None) -> tuple[str, ...]:
    try:
        return runner.ordered_strategies(values)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def template_cases_for_plan(
    *,
    model: str,
    dataset: str,
    strategies: tuple[str, ...],
    rate_plan: str,
    bench_duration_sec: float | None,
    dispatch_policy: str = runner.DEFAULT_DISPATCH_POLICY,
) -> list[runner.ExperimentCase]:
    try:
        cases = runner.build_experiment_matrix(
            rate_plan,
            models=(model, ),
            datasets=(dataset, ),
            strategies=strategies,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if bench_duration_sec is not None:
        cases = runner.apply_bench_duration_override(cases, bench_duration_sec)
    normalized_dispatch_policy = runner.normalize_dispatch_policy(dispatch_policy)
    cases = [
        replace(case, dispatch_policy=normalized_dispatch_policy) for case in cases
    ]

    if not cases:
        raise SystemExit(
            "No configured sweep cases match "
            f"model='{model}', dataset='{dataset}', strategies={list(strategies)}.")
    return cases


def collect_rate_history(
    artifact_root: Path,
    case: runner.ExperimentCase,
    *,
    ignore_bs: bool,
) -> RateHistory:
    successful = runner.latest_successful_case_tpot_by_e2e_mean_for_case(
        artifact_root,
        case,
        ignore_bs=ignore_bs,
    )
    latest_status = runner.latest_case_status_for_case(
        artifact_root,
        case,
        ignore_bs=ignore_bs,
    )
    return RateHistory(
        case=case,
        success_reference=None if successful is None else successful[0],
        success_tpot_by_e2e_mean=None if successful is None else successful[1],
        latest_status=None if latest_status is None else latest_status[1],
        status_reference=None if latest_status is None else latest_status[0],
    )


def previous_candidate_rate(candidate_rates: tuple[float, ...],
                            blocked_rate: float) -> float | None:
    previous: float | None = None
    for rate in candidate_rates:
        if rate >= blocked_rate:
            return previous
        previous = rate
    return previous


def determine_start_rate(
    histories: list[RateHistory],
) -> tuple[float | None, str, str]:
    candidate_rates = tuple(sorted(history.case.request_rate for history in histories))
    if not candidate_rates:
        return None, "start_from_max_rate_no_history", ""

    threshold = runner.TPOT_BY_E2E_EARLY_STOP_MS
    threshold_hits = [
        history for history in histories if history.success_tpot_by_e2e_mean
        is not None and history.success_tpot_by_e2e_mean >= threshold
    ]
    if threshold_hits:
        first_hit = min(threshold_hits, key=lambda history: history.case.request_rate)
        start_rate = previous_candidate_rate(candidate_rates,
                                             first_hit.case.request_rate)
        return (
            start_rate,
            ("start_below_tpot"
             f"{threshold_tag()}_at_rate"
             f"{runner.stringify_request_rate(first_hit.case.request_rate)}"),
            "" if first_hit.success_reference is None else str(
                first_hit.success_reference),
        )

    timeout_hits = [
        history for history in histories if history.latest_status == "timed_out"
    ]
    if timeout_hits:
        first_timeout = min(timeout_hits,
                            key=lambda history: history.case.request_rate)
        start_rate = previous_candidate_rate(candidate_rates,
                                             first_timeout.case.request_rate)
        reason_rate = (first_timeout.case.request_rate
                       if start_rate is None else start_rate)
        return (
            start_rate,
            ("start_from_timeout_rate"
             f"{runner.stringify_request_rate(reason_rate)}"),
            "" if first_timeout.status_reference is None else str(
                first_timeout.status_reference),
        )

    return candidate_rates[-1], "start_from_max_rate_no_history", ""


def planned_rows_for_group(
    artifact_root: Path,
    cases: list[runner.ExperimentCase],
    *,
    ignore_bs: bool,
) -> list[PlannedCaseRow]:
    templates_by_rate = {case.request_rate: case for case in cases}
    candidate_rates = tuple(sorted(templates_by_rate))
    histories = [
        collect_rate_history(
            artifact_root,
            templates_by_rate[rate],
            ignore_bs=ignore_bs,
        ) for rate in candidate_rates
    ]
    start_rate, reason, historical_reference = determine_start_rate(histories)
    if start_rate is None:
        return []

    planned_rates = sorted(
        (rate for rate in candidate_rates if rate <= start_rate),
        reverse=True,
    )
    return [
        PlannedCaseRow(
            case=templates_by_rate[rate],
            reason=reason,
            historical_reference=historical_reference,
        ) for rate in planned_rates
    ]


def filter_planned_rows_by_historical_skip_state(
    artifact_root: Path,
    rows: list[PlannedCaseRow],
    *,
    ignore_bs: bool,
) -> list[PlannedCaseRow]:
    if not rows:
        return rows

    exact_case_skips, blocked_group_rates = runner.build_historical_skip_state(
        artifact_root,
        [row.case for row in rows],
        ignore_bs=ignore_bs,
    )
    return [
        row for row in rows
        if runner.block_reason_for_rate(blocked_group_rates, row.case) is None
        and exact_case_skips.get(row.case.name) is None
    ]


def build_plan_rows(
    *,
    artifact_root: Path,
    model: str,
    dataset: str,
    strategies: tuple[str, ...],
    rate_plan: str,
    ignore_bs: bool,
    bench_duration_sec: float | None,
    dispatch_policy: str = runner.DEFAULT_DISPATCH_POLICY,
) -> list[PlannedCaseRow]:
    template_cases = template_cases_for_plan(
        model=model,
        dataset=dataset,
        strategies=strategies,
        rate_plan=rate_plan,
        bench_duration_sec=bench_duration_sec,
        dispatch_policy=dispatch_policy,
    )
    grouped: dict[str, list[runner.ExperimentCase]] = {key: [] for key in strategies}
    for case in template_cases:
        grouped.setdefault(case.strategy, []).append(case)

    planned_rows: list[PlannedCaseRow] = []
    for strategy in strategies:
        planned_rows.extend(
            planned_rows_for_group(
                artifact_root,
                grouped.get(strategy, []),
                ignore_bs=ignore_bs,
            ))
    return filter_planned_rows_by_historical_skip_state(
        artifact_root,
        planned_rows,
        ignore_bs=ignore_bs,
    )


def planned_case_to_csv_row(planned: PlannedCaseRow) -> dict[str, str]:
    case = planned.case
    return {
        "enabled": "1",
        "name": case.name,
        "cluster": case.cluster,
        "model": case.model,
        "dataset": case.dataset,
        "strategy": case.strategy,
        "dispatch_policy": case.dispatch_policy,
        "request_rate": runner.stringify_request_rate(case.request_rate),
        "rate_phase": case.rate_phase,
        "max_num_seqs": "" if case.max_num_seqs is None else str(
            case.max_num_seqs),
        "gpu_memory_utilization": f"{case.gpu_memory_utilization:g}",
        "max_requests": "" if case.max_requests is None else str(
            case.max_requests),
        "warmup_requests": str(case.warmup_requests),
        "max_model_len": "" if case.max_model_len is None else str(
            case.max_model_len),
        "data_parallel_rpc_port": str(case.data_parallel_rpc_port),
        "reason": planned.reason,
        "historical_reference": planned.historical_reference,
    }


def write_plan_csv(path: Path, rows: list[PlannedCaseRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=runner.CASE_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(planned_case_to_csv_row(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Generate an editable CSV execution plan from historical "
                     "manual multi-node Poisson benchmark artifacts."))
    parser.add_argument(
        "--artifact-root",
        default=str(runner.DEFAULT_ARTIFACT_ROOT),
        help="Shared artifact root to scan for historical runs.",
    )
    parser.add_argument("--model", required=True, help="Target model name.")
    parser.add_argument("--dataset", required=True, help="Target dataset name.")
    parser.add_argument(
        "--strategy",
        action="append",
        help=("Target strategy name. May be repeated. Defaults to all sweep "
              "strategies."),
    )
    parser.add_argument(
        "--rate-plan",
        default=runner.DEFAULT_RATE_PLAN,
        choices=sorted(runner.RATE_PLAN_PHASES),
        help="Candidate request-rate plan to generate from.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Where to write the generated plan CSV.",
    )
    parser.add_argument(
        "--historical-skip-ignore-bs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("When matching historical artifacts, treat different "
              "bs/max_num_seqs values as the same scenario."),
    )
    parser.add_argument(
        "--bench-duration-sec",
        type=runner.positive_float,
        default=None,
        help=("Override benchmark duration when deriving max_requests for the "
              "generated CSV rows."),
    )
    parser.add_argument(
        "--dispatch-policy",
        default=runner.DEFAULT_DISPATCH_POLICY,
        choices=sorted(runner.DISPATCH_POLICY_CHOICES),
        help=("Dispatch policy to stamp into generated rows. "
              "Defaults to the runner's default policy."),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    rows = build_plan_rows(
        artifact_root=Path(args.artifact_root).expanduser().resolve(),
        model=args.model,
        dataset=args.dataset,
        strategies=ordered_strategies(args.strategy),
        rate_plan=args.rate_plan,
        ignore_bs=args.historical_skip_ignore_bs,
        bench_duration_sec=args.bench_duration_sec,
        dispatch_policy=args.dispatch_policy,
    )
    output_csv = Path(args.output_csv).expanduser().resolve()
    write_plan_csv(output_csv, rows)
    print(f"Wrote {len(rows)} planned case(s) to {output_csv}")


if __name__ == "__main__":
    main()
