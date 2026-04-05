#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODELS = ("DPSK", "KIMI")


@dataclass(frozen=True)
class ArchiveCandidate:
    run_dir: Path
    relative_run_dir: Path


@dataclass(frozen=True)
class EmptyCaseCandidate:
    case_dir: Path
    relative_case_dir: Path


def parse_args() -> argparse.Namespace:
    # script_dir = Path(__file__).resolve().parent
    script_dir = "/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench"
    parser = argparse.ArgumentParser(
        description=(
            "Move manual_multinode run directories whose benchmark/ folder is "
            "empty into offline_bench/archive while preserving the relative "
            "directory structure."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=f"{script_dir}/manual_multinode",
        help=f"manual_multinode root directory (default: {script_dir})",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=f"{script_dir}/archive",
        help=(
            "Archive root directory. Matching runs are moved below this "
            "directory using their path relative to --root."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model directories to scan below --root (default: DPSK KIMI).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the move. Without this flag the script only prints a preview.",
    )
    return parser.parse_args()


def is_empty_dir(path: Path) -> bool:
    return not any(path.iterdir())


def iter_candidates(root: Path, models: Sequence[str]) -> list[ArchiveCandidate]:
    candidates: list[ArchiveCandidate] = []
    for model in models:
        model_dir = root / model
        if not model_dir.is_dir():
            print(f"skip missing model directory: {model_dir}", file=sys.stderr)
            continue

        for benchmark_dir in sorted(model_dir.rglob("benchmark")):
            if not benchmark_dir.is_dir() or not is_empty_dir(benchmark_dir):
                continue
            run_dir = benchmark_dir.parent
            candidates.append(
                ArchiveCandidate(
                    run_dir=run_dir,
                    relative_run_dir=run_dir.relative_to(root),
                )
            )
    return candidates


def iter_empty_case_candidates(
    root: Path,
    models: Sequence[str],
) -> list[EmptyCaseCandidate]:
    candidates: list[EmptyCaseCandidate] = []
    for model in models:
        model_dir = root / model
        if not model_dir.is_dir():
            continue

        for case_dir in sorted(model_dir.glob("*/*")):
            if not case_dir.is_dir() or not is_empty_dir(case_dir):
                continue
            candidates.append(
                EmptyCaseCandidate(
                    case_dir=case_dir,
                    relative_case_dir=case_dir.relative_to(root),
                )
            )
    return candidates


def make_unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    suffix = 1
    while True:
        candidate = path.with_name(f"{path.name}__archived{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def render_plan_line(root: Path, archive_root: Path, candidate: ArchiveCandidate) -> str:
    source = candidate.run_dir.relative_to(root)
    destination = make_unique_destination(
        archive_root / candidate.relative_run_dir
    ).relative_to(
        archive_root.parent
    )
    return f"{source} -> {destination}"


def render_empty_case_plan_line(root: Path, candidate: EmptyCaseCandidate) -> str:
    return str(candidate.case_dir.relative_to(root))


def maybe_remove_empty_case_dir(case_dir: Path) -> bool:
    if not case_dir.is_dir() or any(case_dir.iterdir()):
        return False

    case_dir.rmdir()
    print(f"removed empty case directory: {case_dir}")
    return True


def execute_moves(
    archive_root: Path,
    candidates: Sequence[ArchiveCandidate],
) -> tuple[int, int]:
    moved = 0
    removed_case_dirs = 0
    for candidate in candidates:
        destination = make_unique_destination(archive_root / candidate.relative_run_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(candidate.run_dir), str(destination))
        print(f"moved: {candidate.run_dir} -> {destination}")
        if maybe_remove_empty_case_dir(candidate.run_dir.parent):
            removed_case_dirs += 1
        moved += 1
    return moved, removed_case_dirs


def execute_remove_empty_case_dirs(
    candidates: Sequence[EmptyCaseCandidate],
) -> int:
    removed = 0
    for candidate in candidates:
        if maybe_remove_empty_case_dir(candidate.case_dir):
            removed += 1
    return removed


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    archive_root = args.archive_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"manual_multinode root does not exist: {root}")

    candidates = iter_candidates(root=root, models=args.models)
    empty_case_candidates = iter_empty_case_candidates(root=root, models=args.models)
    if not candidates and not empty_case_candidates:
        print(
            "No run directories found with an empty benchmark/ directory, "
            "and no empty case directories found."
        )
        return

    mode = "execute" if args.execute else "dry-run"
    print(f"manual_multinode root: {root} ({mode})")
    print(f"Run directories to archive: {len(candidates)}")
    for candidate in candidates:
        print(render_plan_line(root=root, archive_root=archive_root, candidate=candidate))
    print(f"Empty case directories to remove: {len(empty_case_candidates)}")
    for candidate in empty_case_candidates:
        print(render_empty_case_plan_line(root=root, candidate=candidate))

    if not args.execute:
        print("Preview only. Re-run with --execute to move these directories.")
        return

    moved, removed_case_dirs = execute_moves(
        archive_root=archive_root,
        candidates=candidates,
    )
    print(f"Moved {moved} run directories into {archive_root}.")
    removed_preexisting_case_dirs = execute_remove_empty_case_dirs(
        empty_case_candidates,
    )
    total_removed_case_dirs = removed_case_dirs + removed_preexisting_case_dirs
    if total_removed_case_dirs:
        print(f"Removed {total_removed_case_dirs} empty case directories.")


if __name__ == "__main__":
    main()
