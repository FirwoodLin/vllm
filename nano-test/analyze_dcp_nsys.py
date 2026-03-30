#!/usr/bin/env python3

import argparse
import csv
import re
import sqlite3
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SQLITE_TRIGGER_CMD = [
    "nsys",
    "stats",
    "--report",
    "cuda_gpu_kern_sum",
    "--format",
    "csv",
    "--output",
    "-",
    "--timeunit",
    "us",
]
CHUNK_SIZE = 5000
PROFILE_PATTERNS = {
    "dp": r"dp(\d+)",
    "tp": r"tp(\d+)",
    "dcp": r"dcp(\d+)",
    "backend": r"(ag_rs|a2a)",
    "bs": r"bs(\d+)",
    "input_len": r"inputlen(\d+)",
    "node_rank": r"node(\d+)(?:\.\d+)?",
}
BACKEND_EVENT_PATTERNS = {
    "ag_rs": [
        "ncclDevKernel_AllGather_",
        "ncclDevKernel_ReduceScatter_",
        "flash_fwd_splitkv_mla_kernel",
        "flash_fwd_mla_combine_kernel",
        "_correct_attn_cp_out_kernel",
        "all_reduce",
        "allreduce",
    ],
    "a2a": [
        "ncclDevKernel_AllGather_",
        "ncclDevKernel_SendRecv",
        "flash_fwd_splitkv_mla_kernel",
        "flash_fwd_mla_combine_kernel",
        "_dcp_lse_combine_kernel",
        "all_reduce",
        "allreduce",
    ],
}
EXPECTED_EVENT_TYPES = {
    "ag_rs": [
        "all_gather",
        "splitkv",
        "combine",
        "all_gather",
        "correct",
        "reduce_scatter",
    ],
    "a2a": [
        "all_gather",
        "splitkv",
        "combine",
        "sendrecv",
        "sendrecv",
        "lse_combine",
    ],
}
EXPECTED_BUCKET_NAMES = {
    "ag_rs": [
        "pre_query_all_gather",
        "mla_splitkv",
        "mla_combine",
        "post_lse_all_gather",
        "post_correct_attn_cp_out",
        "post_reduce_scatter",
    ],
    "a2a": [
        "pre_query_all_gather",
        "mla_splitkv",
        "mla_combine",
        "post_sendrecv_output",
        "post_sendrecv_lse",
        "post_dcp_lse_combine",
    ],
}


@dataclass(frozen=True)
class ProfileMetadata:
    basename: str
    dp: Optional[str]
    tp: Optional[str]
    dcp: Optional[str]
    backend: Optional[str]
    bs: Optional[str]
    input_len: Optional[str]
    node_rank: Optional[str]


@dataclass
class KernelAggregate:
    count: int = 0
    total_us: float = 0.0

    @property
    def avg_us(self) -> float:
        return self.total_us / self.count if self.count else 0.0


@dataclass
class ReplayScope:
    label: str
    event_count: int
    thread_count: int
    replay_count: Optional[int]


@dataclass(frozen=True)
class KernelEvent:
    device_id: int
    start_ns: int
    end_ns: int
    kernel_name: str

    @property
    def duration_us(self) -> float:
        return (self.end_ns - self.start_ns) / 1000.0


@dataclass
class SequenceParseResult:
    buckets: Dict[str, KernelAggregate] = field(
        default_factory=lambda: defaultdict(KernelAggregate)
    )
    logical_buckets: Dict[str, KernelAggregate] = field(
        default_factory=lambda: defaultdict(KernelAggregate)
    )
    cycle_count: int = 0
    logical_cycle_count: int = 0
    matched_events: int = 0
    discarded_events: int = 0


@dataclass
class ParsedCycle:
    bucket_events: Dict[str, KernelEvent]
    trailing_all_reduce: List[KernelEvent] = field(default_factory=list)


@dataclass
class ProfileAnalysis:
    profile: str
    sqlite_path: Path
    scope: Optional[ReplayScope]
    backend: str
    backend_source: str
    metadata: ProfileMetadata
    sequence: SequenceParseResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze DCP MLA stage timing in Nsight Systems profiles. "
            "Supports single-profile text output and batch CSV export."
        )
    )
    parser.add_argument(
        "profiles",
        nargs="+",
        help="One or more .nsys-rep or .sqlite files.",
    )
    parser.add_argument(
        "--backend",
        choices=["ag_rs", "a2a"],
        help="Override DCP backend. Default: parse from filename, else infer from kernels.",
    )
    parser.add_argument(
        "--nvtx-label",
        help="Only analyze kernels launched inside this exact NVTX replay label.",
    )
    parser.add_argument(
        "--list-nvtx",
        metavar="PATTERN",
        help="List NVTX labels matching this substring or SQL LIKE pattern and exit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Row limit for --list-nvtx output. Default: 50.",
    )
    parser.add_argument(
        "--csv",
        help="Write one CSV row per unique profile basename.",
    )
    return parser.parse_args()


def resolve_sqlite(profile: str) -> Path:
    path = Path(profile)
    if path.suffix == ".sqlite":
        if not path.exists():
            raise SystemExit(f"SQLite file not found: {path}")
        return path
    if not str(path).endswith(".nsys-rep"):
        raise SystemExit("Input must end with .nsys-rep or .sqlite")

    sqlite_path = Path(str(path)[: -len(".nsys-rep")] + ".sqlite")
    if sqlite_path.exists():
        return sqlite_path

    cmd = SQLITE_TRIGGER_CMD + [str(path)]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Failed to export SQLite with nsys.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr:\n{result.stderr}\n"
            "Workaround: run the same `nsys stats ...` command in a shell to "
            "generate the sibling .sqlite, then rerun this script on the .sqlite file."
        )
    if not sqlite_path.exists():
        raise SystemExit(f"Expected SQLite was not created: {sqlite_path}")
    return sqlite_path


def open_db(sqlite_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    return conn


def strip_profile_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".nsys-rep", ".sqlite"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def parse_profile_metadata(profile: str) -> ProfileMetadata:
    basename = strip_profile_suffix(Path(profile))
    parsed = {}
    for key, pattern in PROFILE_PATTERNS.items():
        match = re.search(pattern, basename)
        parsed[key] = match.group(1) if match else None
    return ProfileMetadata(basename=basename, **parsed)


def format_pattern(pattern: str) -> str:
    return pattern if "%" in pattern else f"%{pattern}%"


def list_nvtx_labels(conn: sqlite3.Connection, pattern: str, limit: int) -> None:
    query = """
    WITH labels AS (
      SELECT
        COALESCE(text, (SELECT value FROM StringIds WHERE id=NVTX_EVENTS.textId), jsonText) AS label,
        start,
        end,
        globalTid
      FROM NVTX_EVENTS
    )
    SELECT
      label,
      COUNT(*) AS events,
      COUNT(DISTINCT globalTid) AS tids,
      SUM(CASE WHEN end IS NOT NULL THEN (end - start) / 1000.0 ELSE 0 END) AS total_us
    FROM labels
    WHERE label IS NOT NULL
      AND label LIKE ?
    GROUP BY label
    ORDER BY events DESC, label
    LIMIT ?
    """
    rows = conn.execute(query, (format_pattern(pattern), limit)).fetchall()
    if not rows:
        print("No NVTX labels matched.")
        return

    print("Matching NVTX labels")
    for row in rows:
        print(
            f"- {row['label']}: events={row['events']}, "
            f"tids={row['tids']}, total_us={row['total_us']:.3f}"
        )


def merge_intervals(
    rows: Sequence[sqlite3.Row],
) -> Tuple[Dict[int, List[Tuple[int, int]]], int]:
    by_tid: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for row in rows:
        by_tid[row["globalTid"]].append((row["start"], row["end"]))

    merged_total = 0
    for tid, intervals in by_tid.items():
        intervals.sort()
        merged: List[Tuple[int, int]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        by_tid[tid] = merged
        merged_total += len(merged)
    return by_tid, merged_total


def collect_replay_correlation_ids(
    conn: sqlite3.Connection, label: str
) -> Tuple[List[int], ReplayScope]:
    labels = [part.strip() for part in label.split("|") if part.strip()]
    if not labels:
        labels = [label]

    placeholders = ",".join("?" * len(labels))
    nvtx_query = """
    SELECT start, end, globalTid
    FROM NVTX_EVENTS
    WHERE COALESCE(text, (SELECT value FROM StringIds WHERE id=NVTX_EVENTS.textId), jsonText)
          IN ("""
    nvtx_query += placeholders
    nvtx_query += """)
    ORDER BY globalTid, start
    """
    nvtx_rows = conn.execute(nvtx_query, labels).fetchall()
    if not nvtx_rows:
        raise SystemExit(f"NVTX label not found: {label}")

    by_tid, merged_total = merge_intervals(nvtx_rows)
    tids = sorted(by_tid)
    min_start = min(start for intervals in by_tid.values() for start, _ in intervals)
    max_end = max(end for intervals in by_tid.values() for _, end in intervals)

    placeholders = ",".join("?" * len(tids))
    runtime_query = f"""
    SELECT start, end, globalTid, correlationId
    FROM CUPTI_ACTIVITY_KIND_RUNTIME
    WHERE globalTid IN ({placeholders})
      AND start >= ?
      AND end <= ?
      AND correlationId IS NOT NULL
    """
    runtime_rows = conn.execute(runtime_query, [*tids, min_start, max_end]).fetchall()

    correlation_ids = set()
    for row in runtime_rows:
        for range_start, range_end in by_tid[row["globalTid"]]:
            if row["start"] >= range_start and row["end"] <= range_end:
                correlation_ids.add(row["correlationId"])
                break

    if not correlation_ids:
        raise SystemExit(
            "No CUDA runtime launches were mapped into the requested NVTX label."
        )

    replay_count = (
        merged_total // len(tids) if tids and merged_total % len(tids) == 0 else None
    )
    scope = ReplayScope(
        label=label,
        event_count=len(nvtx_rows),
        thread_count=len(tids),
        replay_count=replay_count,
    )
    return sorted(correlation_ids), scope


def aggregate_all_kernels(conn: sqlite3.Connection) -> Dict[str, KernelAggregate]:
    query = """
    SELECT
      s.value AS kernel_name,
      COUNT(*) AS instances,
      SUM((k.end - k.start) / 1000.0) AS total_us
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id
    GROUP BY s.value
    """
    summary: Dict[str, KernelAggregate] = {}
    for row in conn.execute(query):
        summary[row["kernel_name"]] = KernelAggregate(
            count=row["instances"], total_us=row["total_us"]
        )
    return summary


def infer_backend_from_summary(summary: Dict[str, KernelAggregate]) -> Optional[str]:
    kernel_names = summary.keys()
    if any("ncclDevKernel_SendRecv" in kernel_name for kernel_name in kernel_names):
        return "a2a"
    if any(
        "ncclDevKernel_ReduceScatter_" in kernel_name for kernel_name in kernel_names
    ):
        return "ag_rs"
    return None


def resolve_backend(
    profile: str, summary: Dict[str, KernelAggregate], backend_override: Optional[str]
) -> Tuple[str, str, ProfileMetadata]:
    metadata = parse_profile_metadata(profile)
    if backend_override:
        return backend_override, "cli", metadata
    if metadata.backend:
        return metadata.backend, "filename", metadata
    inferred = infer_backend_from_summary(summary)
    if inferred:
        return inferred, "kernel_inference", metadata
    raise SystemExit(
        "Could not determine DCP backend. Use --backend ag_rs|a2a, or encode it in the filename."
    )


def build_like_predicate(patterns: Sequence[str]) -> Tuple[str, List[str]]:
    clauses = ["s.value LIKE ?" for _ in patterns]
    params = [f"%{pattern}%" for pattern in patterns]
    return " OR ".join(clauses), params


def fetch_relevant_kernel_events(
    conn: sqlite3.Connection,
    backend: str,
    correlation_ids: Optional[Sequence[int]],
) -> List[KernelEvent]:
    predicate, predicate_params = build_like_predicate(BACKEND_EVENT_PATTERNS[backend])
    base_query = f"""
    SELECT
      k.deviceId AS device_id,
      k.start AS start_ns,
      k.end AS end_ns,
      s.value AS kernel_name
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id
    WHERE ({predicate})
    """

    rows: List[KernelEvent] = []
    if correlation_ids is None:
        for row in conn.execute(base_query + " ORDER BY k.deviceId, k.start", predicate_params):
            rows.append(
                KernelEvent(
                    device_id=row["device_id"],
                    start_ns=row["start_ns"],
                    end_ns=row["end_ns"],
                    kernel_name=row["kernel_name"],
                )
            )
        return rows

    for index in range(0, len(correlation_ids), CHUNK_SIZE):
        chunk = correlation_ids[index : index + CHUNK_SIZE]
        placeholders = ",".join("?" * len(chunk))
        query = (
            base_query
            + f" AND k.correlationId IN ({placeholders}) ORDER BY k.deviceId, k.start"
        )
        for row in conn.execute(query, [*predicate_params, *chunk]):
            rows.append(
                KernelEvent(
                    device_id=row["device_id"],
                    start_ns=row["start_ns"],
                    end_ns=row["end_ns"],
                    kernel_name=row["kernel_name"],
                )
            )
    rows.sort(key=lambda event: (event.device_id, event.start_ns, event.end_ns))
    return rows


def classify_event_type(kernel_name: str) -> Optional[str]:
    lowered = kernel_name.casefold()
    if "flash_fwd_splitkv_mla_kernel" in kernel_name:
        return "splitkv"
    if "flash_fwd_mla_combine_kernel" in kernel_name:
        return "combine"
    if "ncclDevKernel_AllGather_" in kernel_name:
        return "all_gather"
    if "ncclDevKernel_ReduceScatter_" in kernel_name:
        return "reduce_scatter"
    if "ncclDevKernel_SendRecv" in kernel_name:
        return "sendrecv"
    if "_correct_attn_cp_out_kernel" in kernel_name:
        return "correct"
    if "_dcp_lse_combine_kernel" in kernel_name:
        return "lse_combine"
    if "all_reduce" in lowered or "allreduce" in lowered:
        return "all_reduce"
    return None


def add_event(aggregate: KernelAggregate, event: KernelEvent) -> None:
    aggregate.count += 1
    aggregate.total_us += event.duration_us


def add_duration(aggregate: KernelAggregate, duration_us: float) -> None:
    aggregate.count += 1
    aggregate.total_us += duration_us


def populate_logical_buckets(
    result: SequenceParseResult,
    cycles_by_device: Dict[int, List[ParsedCycle]],
    backend: str,
) -> None:
    if not cycles_by_device:
        return

    cycle_lists = list(cycles_by_device.values())
    logical_cycle_count = min(len(cycles) for cycles in cycle_lists)
    if logical_cycle_count == 0:
        return

    bucket_names = EXPECTED_BUCKET_NAMES[backend]
    for cycle_index in range(logical_cycle_count):
        logical_cycles = [cycles[cycle_index] for cycles in cycle_lists]
        for bucket_name in bucket_names:
            add_duration(
                result.logical_buckets[bucket_name],
                max(
                    cycle.bucket_events[bucket_name].duration_us
                    for cycle in logical_cycles
                ),
            )

        add_duration(
            result.logical_buckets["post_tp_all_reduce"],
            max(
                sum(event.duration_us for event in cycle.trailing_all_reduce)
                for cycle in logical_cycles
            ),
        )

    result.logical_cycle_count = logical_cycle_count


def parse_sequence(events: Sequence[KernelEvent], backend: str) -> SequenceParseResult:
    expected_types = EXPECTED_EVENT_TYPES[backend]
    bucket_names = EXPECTED_BUCKET_NAMES[backend]
    result = SequenceParseResult()
    by_device: Dict[int, List[KernelEvent]] = defaultdict(list)
    for event in events:
        by_device[event.device_id].append(event)

    start_type = expected_types[0]
    cycles_by_device: Dict[int, List[ParsedCycle]] = defaultdict(list)
    for device_id, device_events in by_device.items():
        current: List[KernelEvent] = []
        completed_cycle: Optional[List[KernelEvent]] = None
        trailing_all_reduce: List[KernelEvent] = []
        parsed_cycles: List[ParsedCycle] = []
        next_index = 0

        def finalize_cycle() -> None:
            nonlocal completed_cycle, trailing_all_reduce
            if completed_cycle is None:
                return
            for bucket_name, cycle_event in zip(bucket_names, completed_cycle):
                add_event(result.buckets[bucket_name], cycle_event)
            for cycle_event in trailing_all_reduce:
                add_event(result.buckets["post_tp_all_reduce"], cycle_event)
            parsed_cycles.append(
                ParsedCycle(
                    bucket_events={
                        bucket_name: cycle_event
                        for bucket_name, cycle_event in zip(bucket_names, completed_cycle)
                    },
                    trailing_all_reduce=list(trailing_all_reduce),
                )
            )
            result.cycle_count += 1
            result.matched_events += len(completed_cycle) + len(trailing_all_reduce)
            completed_cycle = None
            trailing_all_reduce = []

        for event in device_events:
            event_type = classify_event_type(event.kernel_name)

            if completed_cycle is not None:
                if event_type == "all_reduce":
                    trailing_all_reduce.append(event)
                    continue
                finalize_cycle()

            expected_type = expected_types[next_index]
            if event_type == expected_type:
                current.append(event)
                next_index += 1
                if next_index == len(expected_types):
                    completed_cycle = current
                    trailing_all_reduce = []
                    current = []
                    next_index = 0
                continue

            if event_type == start_type:
                result.discarded_events += len(current)
                current = [event]
                next_index = 1
                continue

            result.discarded_events += len(current) + 1
            current = []
            next_index = 0

        finalize_cycle()
        result.discarded_events += len(current)
        cycles_by_device[device_id] = parsed_cycles

    populate_logical_buckets(result, cycles_by_device, backend)

    return result


def metric(result: SequenceParseResult, bucket_name: str) -> KernelAggregate:
    return result.buckets.get(bucket_name, KernelAggregate())


def logical_metric(result: SequenceParseResult, bucket_name: str) -> KernelAggregate:
    return result.logical_buckets.get(bucket_name, KernelAggregate())


def safe_pct(numerator: float, denominator: float) -> float:
    return numerator / denominator * 100 if denominator else 0.0


def dcp_stage_total_us(sequence: SequenceParseResult, backend: str) -> float:
    tp_all_reduce_us = metric(sequence, "post_tp_all_reduce").total_us
    if backend == "ag_rs":
        return (
            metric(sequence, "pre_query_all_gather").total_us
            + metric(sequence, "mla_splitkv").total_us
            + metric(sequence, "mla_combine").total_us
            + metric(sequence, "post_lse_all_gather").total_us
            + metric(sequence, "post_correct_attn_cp_out").total_us
            + metric(sequence, "post_reduce_scatter").total_us
            + tp_all_reduce_us
        )
    return (
        metric(sequence, "pre_query_all_gather").total_us
        + metric(sequence, "mla_splitkv").total_us
        + metric(sequence, "mla_combine").total_us
        + metric(sequence, "post_sendrecv_output").total_us
        + metric(sequence, "post_sendrecv_lse").total_us
        + metric(sequence, "post_dcp_lse_combine").total_us
        + tp_all_reduce_us
    )


def logical_dcp_stage_total_us(sequence: SequenceParseResult, backend: str) -> float:
    tp_all_reduce_us = logical_metric(sequence, "post_tp_all_reduce").total_us
    if backend == "ag_rs":
        return (
            logical_metric(sequence, "pre_query_all_gather").total_us
            + logical_metric(sequence, "mla_splitkv").total_us
            + logical_metric(sequence, "mla_combine").total_us
            + logical_metric(sequence, "post_lse_all_gather").total_us
            + logical_metric(sequence, "post_correct_attn_cp_out").total_us
            + logical_metric(sequence, "post_reduce_scatter").total_us
            + tp_all_reduce_us
        )
    return (
        logical_metric(sequence, "pre_query_all_gather").total_us
        + logical_metric(sequence, "mla_splitkv").total_us
        + logical_metric(sequence, "mla_combine").total_us
        + logical_metric(sequence, "post_sendrecv_output").total_us
        + logical_metric(sequence, "post_sendrecv_lse").total_us
        + logical_metric(sequence, "post_dcp_lse_combine").total_us
        + tp_all_reduce_us
    )


def analyze_profile(
    profile: str,
    backend_override: Optional[str],
    nvtx_label: Optional[str],
) -> ProfileAnalysis:
    sqlite_path = resolve_sqlite(profile)
    conn = open_db(sqlite_path)
    scope: Optional[ReplayScope] = None
    correlation_ids: Optional[List[int]] = None
    if nvtx_label:
        correlation_ids, scope = collect_replay_correlation_ids(conn, nvtx_label)

    summary = aggregate_all_kernels(conn)
    backend, backend_source, metadata = resolve_backend(
        profile, summary, backend_override
    )
    events = fetch_relevant_kernel_events(conn, backend, correlation_ids)
    sequence = parse_sequence(events, backend)

    if sequence.cycle_count == 0:
        raise SystemExit(
            f"No complete {backend} DCP MLA cycles were parsed from {profile}. "
            "Check backend selection and replay scope."
        )

    return ProfileAnalysis(
        profile=profile,
        sqlite_path=sqlite_path,
        scope=scope,
        backend=backend,
        backend_source=backend_source,
        metadata=metadata,
        sequence=sequence,
    )


def print_profile_metadata(metadata: ProfileMetadata) -> None:
    fields = [
        ("dp", metadata.dp),
        ("tp", metadata.tp),
        ("dcp", metadata.dcp),
        ("backend", metadata.backend),
        ("bs", metadata.bs),
        ("input_len", metadata.input_len),
        ("node_rank", metadata.node_rank),
    ]
    present = [(key, value) for key, value in fields if value is not None]
    if not present:
        return
    print("Profile metadata")
    for key, value in present:
        print(f"- {key}={value}")
    print()


def print_stage_breakdown(analysis: ProfileAnalysis) -> None:
    sequence = analysis.sequence
    pre_us = metric(sequence, "pre_query_all_gather").total_us
    splitkv_us = metric(sequence, "mla_splitkv").total_us
    combine_us = metric(sequence, "mla_combine").total_us
    mla_core_us = splitkv_us + combine_us
    tp_all_reduce = metric(sequence, "post_tp_all_reduce")
    total_us = dcp_stage_total_us(sequence, analysis.backend)

    print("Stage breakdown")
    print(
        "- pre_query_all_gather_us="
        f"{pre_us:.3f} "
        f"(count={metric(sequence, 'pre_query_all_gather').count})"
    )
    print(
        f"- mla_splitkv_us={splitkv_us:.3f} "
        f"(count={metric(sequence, 'mla_splitkv').count})"
    )
    print(
        f"- mla_combine_us={combine_us:.3f} "
        f"(count={metric(sequence, 'mla_combine').count})"
    )
    print(f"- mla_core_us={mla_core_us:.3f}")
    if analysis.backend == "ag_rs":
        lse_ag = metric(sequence, "post_lse_all_gather")
        correct = metric(sequence, "post_correct_attn_cp_out")
        rs = metric(sequence, "post_reduce_scatter")
        post_total_us = lse_ag.total_us + correct.total_us + rs.total_us
        print(
            f"- post_lse_all_gather_us={lse_ag.total_us:.3f} "
            f"(count={lse_ag.count})"
        )
        print(
            f"- post_correct_attn_cp_out_us={correct.total_us:.3f} "
            f"(count={correct.count})"
        )
        print(
            f"- post_reduce_scatter_us={rs.total_us:.3f} "
            f"(count={rs.count})"
        )
        print(
            f"- post_tp_all_reduce_us={tp_all_reduce.total_us:.3f} "
            f"(count={tp_all_reduce.count})"
        )
        post_total_us = post_total_us + tp_all_reduce.total_us
    else:
        sendrecv_output = metric(sequence, "post_sendrecv_output")
        sendrecv_lse = metric(sequence, "post_sendrecv_lse")
        lse_combine = metric(sequence, "post_dcp_lse_combine")
        post_total_us = (
            sendrecv_output.total_us + sendrecv_lse.total_us + lse_combine.total_us
        )
        print(
            f"- post_sendrecv_output_us={sendrecv_output.total_us:.3f} "
            f"(count={sendrecv_output.count})"
        )
        print(
            f"- post_sendrecv_lse_us={sendrecv_lse.total_us:.3f} "
            f"(count={sendrecv_lse.count})"
        )
        print(
            f"- post_dcp_lse_combine_us={lse_combine.total_us:.3f} "
            f"(count={lse_combine.count})"
        )
        print(
            f"- post_tp_all_reduce_us={tp_all_reduce.total_us:.3f} "
            f"(count={tp_all_reduce.count})"
        )
        post_total_us = post_total_us + tp_all_reduce.total_us
    print(f"- post_total_us={post_total_us:.3f}")
    print(f"- dcp_stage_total_us={total_us:.3f}")
    print(f"- pre_share_pct={safe_pct(pre_us, total_us):.6f}")
    print(f"- mla_core_share_pct={safe_pct(mla_core_us, total_us):.6f}")
    print(f"- post_share_pct={safe_pct(post_total_us, total_us):.6f}")
    print(f"- parsed_cycle_count={sequence.cycle_count}")
    print(f"- matched_event_count={sequence.matched_events}")
    print(f"- discarded_event_count={sequence.discarded_events}")
    if sequence.logical_cycle_count:
        logical_pre_us = logical_metric(sequence, "pre_query_all_gather").avg_us
        logical_mla_core_us = (
            logical_metric(sequence, "mla_splitkv").avg_us
            + logical_metric(sequence, "mla_combine").avg_us
        )
        logical_post_us = (
            logical_dcp_stage_total_us(sequence, analysis.backend)
            / sequence.logical_cycle_count
            - logical_pre_us
            - logical_mla_core_us
        )
        logical_total_us = (
            logical_dcp_stage_total_us(sequence, analysis.backend)
            / sequence.logical_cycle_count
        )
        print()
        print("Single-attention average")
        print(
            "- semantics=max across devices for each logical cycle, then averaged "
            f"over {sequence.logical_cycle_count} logical cycles"
        )
        print(f"- pre_query_all_gather_per_attention_us={logical_pre_us:.3f}")
        print(f"- mla_core_per_attention_us={logical_mla_core_us:.3f}")
        print(f"- post_total_per_attention_us={logical_post_us:.3f}")
        print(f"- dcp_stage_total_per_attention_us={logical_total_us:.3f}")


def print_profile_analysis(analysis: ProfileAnalysis) -> None:
    print(f"Profile: {analysis.profile}")
    print(f"SQLite: {analysis.sqlite_path}")
    print(f"Scope: {'replay' if analysis.scope else 'global'}")
    print(f"Backend: {analysis.backend}")
    print(f"Backend source: {analysis.backend_source}")
    if analysis.scope:
        print(f"NVTX label: {analysis.scope.label}")
        print(f"Matched NVTX events: {analysis.scope.event_count}")
        print(f"Matched threads: {analysis.scope.thread_count}")
        print(
            "Estimated replay count: "
            + (
                str(analysis.scope.replay_count)
                if analysis.scope.replay_count is not None
                else "unknown"
            )
        )
    print()
    print_profile_metadata(analysis.metadata)
    print_stage_breakdown(analysis)


def csv_fieldnames() -> List[str]:
    return [
        "basename",
        "input_profile",
        "sqlite_path",
        "scope",
        "nvtx_label",
        "dp",
        "tp",
        "dcp",
        "backend",
        "backend_source",
        "bs",
        "input_len",
        "node_rank",
        "pre_query_all_gather_count",
        "pre_query_all_gather_us",
        "mla_splitkv_count",
        "mla_splitkv_us",
        "mla_combine_count",
        "mla_combine_us",
        "mla_core_us",
        "post_lse_all_gather_count",
        "post_lse_all_gather_us",
        "post_correct_attn_cp_out_count",
        "post_correct_attn_cp_out_us",
        "post_reduce_scatter_count",
        "post_reduce_scatter_us",
        "post_tp_all_reduce_count",
        "post_tp_all_reduce_us",
        "post_sendrecv_output_count",
        "post_sendrecv_output_us",
        "post_sendrecv_lse_count",
        "post_sendrecv_lse_us",
        "post_dcp_lse_combine_count",
        "post_dcp_lse_combine_us",
        "post_total_us",
        "dcp_stage_total_us",
        "logical_cycle_count",
        "pre_query_all_gather_per_attention_us",
        "mla_splitkv_per_attention_us",
        "mla_combine_per_attention_us",
        "mla_core_per_attention_us",
        "post_lse_all_gather_per_attention_us",
        "post_correct_attn_cp_out_per_attention_us",
        "post_reduce_scatter_per_attention_us",
        "post_tp_all_reduce_per_attention_us",
        "post_sendrecv_output_per_attention_us",
        "post_sendrecv_lse_per_attention_us",
        "post_dcp_lse_combine_per_attention_us",
        "post_total_per_attention_us",
        "dcp_stage_total_per_attention_us",
        "pre_share_pct",
        "mla_core_share_pct",
        "post_share_pct",
        "parsed_cycle_count",
        "matched_event_count",
        "discarded_event_count",
    ]


def build_csv_row(analysis: ProfileAnalysis) -> Dict[str, object]:
    sequence = analysis.sequence
    pre = metric(sequence, "pre_query_all_gather")
    splitkv = metric(sequence, "mla_splitkv")
    combine = metric(sequence, "mla_combine")
    mla_core_us = splitkv.total_us + combine.total_us
    post_tp_all_reduce = metric(sequence, "post_tp_all_reduce")
    logical_pre = logical_metric(sequence, "pre_query_all_gather")
    logical_splitkv = logical_metric(sequence, "mla_splitkv")
    logical_combine = logical_metric(sequence, "mla_combine")
    logical_mla_core_us = logical_splitkv.avg_us + logical_combine.avg_us
    logical_post_tp_all_reduce = logical_metric(sequence, "post_tp_all_reduce")
    if analysis.backend == "ag_rs":
        post_lse_ag = metric(sequence, "post_lse_all_gather")
        post_correct = metric(sequence, "post_correct_attn_cp_out")
        post_rs = metric(sequence, "post_reduce_scatter")
        post_sendrecv_output = KernelAggregate()
        post_sendrecv_lse = KernelAggregate()
        post_dcp_lse_combine = KernelAggregate()
        logical_post_lse_ag = logical_metric(sequence, "post_lse_all_gather")
        logical_post_correct = logical_metric(sequence, "post_correct_attn_cp_out")
        logical_post_rs = logical_metric(sequence, "post_reduce_scatter")
        logical_post_sendrecv_output = KernelAggregate()
        logical_post_sendrecv_lse = KernelAggregate()
        logical_post_dcp_lse_combine = KernelAggregate()
        post_total_us = (
            post_lse_ag.total_us
            + post_correct.total_us
            + post_rs.total_us
            + post_tp_all_reduce.total_us
        )
        logical_post_total_us = (
            logical_post_lse_ag.avg_us
            + logical_post_correct.avg_us
            + logical_post_rs.avg_us
            + logical_post_tp_all_reduce.avg_us
        )
    else:
        post_lse_ag = KernelAggregate()
        post_correct = KernelAggregate()
        post_rs = KernelAggregate()
        post_sendrecv_output = metric(sequence, "post_sendrecv_output")
        post_sendrecv_lse = metric(sequence, "post_sendrecv_lse")
        post_dcp_lse_combine = metric(sequence, "post_dcp_lse_combine")
        logical_post_lse_ag = KernelAggregate()
        logical_post_correct = KernelAggregate()
        logical_post_rs = KernelAggregate()
        logical_post_sendrecv_output = logical_metric(sequence, "post_sendrecv_output")
        logical_post_sendrecv_lse = logical_metric(sequence, "post_sendrecv_lse")
        logical_post_dcp_lse_combine = logical_metric(sequence, "post_dcp_lse_combine")
        post_total_us = (
            post_sendrecv_output.total_us
            + post_sendrecv_lse.total_us
            + post_dcp_lse_combine.total_us
            + post_tp_all_reduce.total_us
        )
        logical_post_total_us = (
            logical_post_sendrecv_output.avg_us
            + logical_post_sendrecv_lse.avg_us
            + logical_post_dcp_lse_combine.avg_us
            + logical_post_tp_all_reduce.avg_us
        )
    total_us = dcp_stage_total_us(sequence, analysis.backend)
    logical_total_us = (
        logical_dcp_stage_total_us(sequence, analysis.backend)
        / sequence.logical_cycle_count
        if sequence.logical_cycle_count
        else 0.0
    )
    return {
        "basename": analysis.metadata.basename,
        "input_profile": analysis.profile,
        "sqlite_path": str(analysis.sqlite_path),
        "scope": "replay" if analysis.scope else "global",
        "nvtx_label": analysis.scope.label if analysis.scope else "",
        "dp": analysis.metadata.dp or "",
        "tp": analysis.metadata.tp or "",
        "dcp": analysis.metadata.dcp or "",
        "backend": analysis.backend,
        "backend_source": analysis.backend_source,
        "bs": analysis.metadata.bs or "",
        "input_len": analysis.metadata.input_len or "",
        "node_rank": analysis.metadata.node_rank or "",
        "pre_query_all_gather_count": pre.count,
        "pre_query_all_gather_us": f"{pre.total_us:.6f}",
        "mla_splitkv_count": splitkv.count,
        "mla_splitkv_us": f"{splitkv.total_us:.6f}",
        "mla_combine_count": combine.count,
        "mla_combine_us": f"{combine.total_us:.6f}",
        "mla_core_us": f"{mla_core_us:.6f}",
        "post_lse_all_gather_count": post_lse_ag.count,
        "post_lse_all_gather_us": f"{post_lse_ag.total_us:.6f}",
        "post_correct_attn_cp_out_count": post_correct.count,
        "post_correct_attn_cp_out_us": f"{post_correct.total_us:.6f}",
        "post_reduce_scatter_count": post_rs.count,
        "post_reduce_scatter_us": f"{post_rs.total_us:.6f}",
        "post_tp_all_reduce_count": post_tp_all_reduce.count,
        "post_tp_all_reduce_us": f"{post_tp_all_reduce.total_us:.6f}",
        "post_sendrecv_output_count": post_sendrecv_output.count,
        "post_sendrecv_output_us": f"{post_sendrecv_output.total_us:.6f}",
        "post_sendrecv_lse_count": post_sendrecv_lse.count,
        "post_sendrecv_lse_us": f"{post_sendrecv_lse.total_us:.6f}",
        "post_dcp_lse_combine_count": post_dcp_lse_combine.count,
        "post_dcp_lse_combine_us": f"{post_dcp_lse_combine.total_us:.6f}",
        "post_total_us": f"{post_total_us:.6f}",
        "dcp_stage_total_us": f"{total_us:.6f}",
        "logical_cycle_count": sequence.logical_cycle_count,
        "pre_query_all_gather_per_attention_us": f"{logical_pre.avg_us:.6f}",
        "mla_splitkv_per_attention_us": f"{logical_splitkv.avg_us:.6f}",
        "mla_combine_per_attention_us": f"{logical_combine.avg_us:.6f}",
        "mla_core_per_attention_us": f"{logical_mla_core_us:.6f}",
        "post_lse_all_gather_per_attention_us": f"{logical_post_lse_ag.avg_us:.6f}",
        "post_correct_attn_cp_out_per_attention_us": f"{logical_post_correct.avg_us:.6f}",
        "post_reduce_scatter_per_attention_us": f"{logical_post_rs.avg_us:.6f}",
        "post_tp_all_reduce_per_attention_us": f"{logical_post_tp_all_reduce.avg_us:.6f}",
        "post_sendrecv_output_per_attention_us": f"{logical_post_sendrecv_output.avg_us:.6f}",
        "post_sendrecv_lse_per_attention_us": f"{logical_post_sendrecv_lse.avg_us:.6f}",
        "post_dcp_lse_combine_per_attention_us": (
            f"{logical_post_dcp_lse_combine.avg_us:.6f}"
        ),
        "post_total_per_attention_us": f"{logical_post_total_us:.6f}",
        "dcp_stage_total_per_attention_us": f"{logical_total_us:.6f}",
        "pre_share_pct": f"{safe_pct(pre.total_us, total_us):.6f}",
        "mla_core_share_pct": f"{safe_pct(mla_core_us, total_us):.6f}",
        "post_share_pct": f"{safe_pct(post_total_us, total_us):.6f}",
        "parsed_cycle_count": sequence.cycle_count,
        "matched_event_count": sequence.matched_events,
        "discarded_event_count": sequence.discarded_events,
    }


def write_csv(path: str, analyses: Sequence[ProfileAnalysis]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=csv_fieldnames())
        writer.writeheader()
        for analysis in analyses:
            writer.writerow(build_csv_row(analysis))


def dedupe_profiles(profiles: Sequence[str]) -> List[str]:
    selected: Dict[str, str] = {}
    order: List[str] = []
    for profile in profiles:
        path = Path(profile)
        key = str(path.parent / strip_profile_suffix(path))
        previous = selected.get(key)
        if previous is None:
            selected[key] = profile
            order.append(key)
            continue
        if path.suffix == ".sqlite" and Path(previous).suffix != ".sqlite":
            selected[key] = profile
    return [selected[key] for key in order]


def main() -> None:
    args = parse_args()
    profiles = dedupe_profiles(args.profiles)

    if args.list_nvtx and len(profiles) != 1:
        raise SystemExit("--list-nvtx only supports a single input profile.")

    if args.list_nvtx:
        sqlite_path = resolve_sqlite(profiles[0])
        conn = open_db(sqlite_path)
        list_nvtx_labels(conn, args.list_nvtx, args.limit)
        if not args.nvtx_label:
            return
        print()

    analyses = [
        analyze_profile(profile, args.backend, args.nvtx_label) for profile in profiles
    ]

    for index, analysis in enumerate(analyses):
        if index:
            print()
        print_profile_analysis(analysis)

    if args.csv:
        write_csv(args.csv, analyses)
        print()
        print(f"CSV written: {args.csv}")


if __name__ == "__main__":
    main()
