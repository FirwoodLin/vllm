#!/usr/bin/env python3

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

DISPATCH_POLICIES = ("least_batch", "least_cache")
DEFAULT_KV_CACHE_TOKENS_PER_RANK = 1_050_000


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    dp_rank: int
    input_len: int
    output_len: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read custom nested input-length JSON, pre-allocate each request to a "
            "DP rank with least_batch/least_cache, then send "
            "X-data-parallel-rank headers."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="OpenAI-compatible vLLM base URL.",
    )
    parser.add_argument(
        "--model",
        default="/mnt/nvme1n1/ml_research/models/deepseek-v3-1024k",
        help="Model name/path passed to the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--lens-json",
        required=True,
        help="Path to JSON file containing nested input lengths.",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=32,
        help="Total DP size. Requests will be pre-allocated to ranks [0, dp_size).",
    )
    parser.add_argument(
        "--dispatch-policy",
        choices=DISPATCH_POLICIES,
        default="least_batch",
        help="Pre-allocation policy for assigning requests to DP ranks.",
    )
    parser.add_argument(
        "--kv-cache-tokens-per-rank",
        type=int,
        default=DEFAULT_KV_CACHE_TOKENS_PER_RANK,
        help=(
            "Single-rank KV cache token cap used by least_cache pre-allocation. "
            "Default: 1050000."
        ),
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=64,
        help="Output length in tokens for every request.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260327,
        help="Base seed for deterministic prompt token generation.",
    )
    parser.add_argument(
        "--queue-settle-seconds",
        type=float,
        default=5.0,
        help=(
            "Sleep after all HTTP POSTs have been issued, before "
            "/start_profile and /resume."
        ),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=1800.0,
        help="Per-request timeout while waiting for generation to complete.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=0,
        help=(
            "Optional explicit vocab size. If unset, the script reads "
            "<model>/config.json and uses vocab_size."
        ),
    )
    parser.add_argument(
        "--print-first-n",
        type=int,
        default=8,
        help="How many pre-allocated request specs to print before sending.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Only run pre-allocation and save outputs (JSON + plots), "
            "without sending HTTP requests."
        ),
    )
    parser.add_argument(
        "--dry-run-output-dir",
        default="dry_run_allocation",
        help="Output directory for --dry-run artifacts.",
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
            f"got {type(item)!r}."
        )

    walk(node)
    if not values:
        raise ValueError("No valid input lengths found in --lens-json.")
    return values


def load_input_lengths(path: str) -> list[int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _flatten_nested_lengths(payload)


def resolve_vocab_size(model: str, explicit_vocab_size: int) -> int:
    if explicit_vocab_size > 0:
        return explicit_vocab_size

    config_path = Path(model) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Could not find {config_path}. Pass --vocab-size explicitly."
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    vocab_size = int(config["vocab_size"])
    if vocab_size <= 4096:
        raise ValueError(f"Unexpected vocab_size={vocab_size}")
    return vocab_size


def build_prompt_token_ids(length: int, seed: int, vocab_size: int) -> list[int]:
    rng = random.Random(seed)
    low = 1024
    high = max(low + 1, vocab_size - 1024)
    width = high - low
    return [low + rng.randrange(width) for _ in range(length)]


def preallocate_dp_ranks(
    input_lengths: list[int],
    dp_size: int,
    output_len: int,
    dispatch_policy: str,
    kv_cache_tokens_per_rank: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    waiting_requests = [0] * dp_size
    waiting_tokens = [0] * dp_size
    free_kv_tokens = [kv_cache_tokens_per_rank] * dp_size
    assigned_ranks: list[int] = []

    for input_len in input_lengths:
        kv_tokens_needed = input_len + output_len
        if dispatch_policy == "least_batch":
            ranks_with_capacity = [
                rank
                for rank in range(dp_size)
                if free_kv_tokens[rank] >= kv_tokens_needed
            ]
            if ranks_with_capacity:
                # Keep least_batch behavior, but only among ranks that can still
                # fit this request's estimated KV usage.
                dp_rank = min(
                    ranks_with_capacity,
                    key=lambda rank: (waiting_requests[rank], rank),
                )
            else:
                # If every rank is already over budget for this request, pick the
                # one that would overflow by the smallest amount.
                dp_rank = min(
                    range(dp_size),
                    key=lambda rank: (
                        kv_tokens_needed - free_kv_tokens[rank],
                        waiting_requests[rank],
                        rank,
                    ),
                )
        elif dispatch_policy == "least_cache":
            dp_rank = min(
                range(dp_size),
                key=lambda rank: (
                    waiting_tokens[rank] - free_kv_tokens[rank],
                    waiting_requests[rank],
                    rank,
                ),
            )
        else:
            raise ValueError(f"Unsupported dispatch policy: {dispatch_policy}")

        assigned_ranks.append(dp_rank)
        waiting_requests[dp_rank] += 1
        waiting_tokens[dp_rank] += input_len

        # Use prompt + max decode length as a simple KV upper-bound estimate.
        free_kv_tokens[dp_rank] -= kv_tokens_needed

    return assigned_ranks, waiting_requests, waiting_tokens, free_kv_tokens


def build_request_specs(
    input_lengths: list[int],
    assigned_ranks: list[int],
    output_len: int,
    seed: int,
) -> list[RequestSpec]:
    specs: list[RequestSpec] = []
    for idx, (input_len, dp_rank) in enumerate(zip(input_lengths, assigned_ranks)):
        specs.append(
            RequestSpec(
                request_id=f"custom-{idx:06d}",
                dp_rank=dp_rank,
                input_len=input_len,
                output_len=output_len,
                seed=seed + idx,
            )
        )
    return specs


def build_rank_request_arrays(
    specs: list[RequestSpec],
    dp_size: int,
    kv_cache_tokens_per_rank: int,
) -> tuple[list[list[dict]], list[int], list[int], list[int]]:
    rank_requests: list[list[dict]] = [[] for _ in range(dp_size)]
    for spec in specs:
        kv_cache_tokens = spec.input_len + spec.output_len
        rank_requests[spec.dp_rank].append(
            {
                "request_id": spec.request_id,
                "input_len": spec.input_len,
                "output_len": spec.output_len,
                "seed": spec.seed,
                "kv_cache_tokens_estimate": kv_cache_tokens,
            }
        )

    batch_sizes = [len(items) for items in rank_requests]
    kv_cache_total_tokens = [
        sum(item["kv_cache_tokens_estimate"] for item in items)
        for items in rank_requests
    ]
    kv_cache_free_tokens = [
        kv_cache_tokens_per_rank - used for used in kv_cache_total_tokens
    ]
    return rank_requests, batch_sizes, kv_cache_total_tokens, kv_cache_free_tokens


def save_rank_bar_chart(
    values: list[int],
    *,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    ranks = list(range(len(values)))
    fig, ax = plt.subplots(figsize=(max(10.0, len(values) * 0.35), 4.8))
    ax.bar(ranks, values, color="#4C78A8")
    ax.set_xlabel("rank")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(ranks)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_dry_run_outputs(
    *,
    output_dir: Path,
    dp_size: int,
    dispatch_policy: str,
    kv_cache_tokens_per_rank: int,
    output_len: int,
    lens_json: str,
    rank_requests: list[list[dict]],
    batch_sizes: list[int],
    kv_cache_total_tokens: list[int],
    kv_cache_free_tokens: list[int],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "allocation_by_rank.json"
    kv_cache_plot_path = output_dir / "kv_cache_total_tokens_by_rank.png"
    batch_size_plot_path = output_dir / "batch_size_by_rank.png"

    payload = {
        "dispatch_policy": dispatch_policy,
        "dp_size": dp_size,
        "output_len": output_len,
        "lens_json": lens_json,
        "kv_cache_tokens_per_rank": kv_cache_tokens_per_rank,
        "per_rank": [
            {
                "dp_rank": rank,
                "batch_size": batch_sizes[rank],
                "kv_cache_total_tokens": kv_cache_total_tokens[rank],
                "kv_cache_free_tokens": kv_cache_free_tokens[rank],
                "requests": rank_requests[rank],
            }
            for rank in range(dp_size)
        ],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    save_rank_bar_chart(
        kv_cache_total_tokens,
        y_label="KV cache total tokens",
        title="KV cache total tokens by rank",
        output_path=kv_cache_plot_path,
    )
    save_rank_bar_chart(
        batch_sizes,
        y_label="Batch size (request count)",
        title="Batch size by rank",
        output_path=batch_size_plot_path,
    )
    return json_path, kv_cache_plot_path, batch_size_plot_path


def build_payload_bytes(
    spec: RequestSpec,
    model: str,
    vocab_size: int,
) -> bytes:
    prompt_token_ids = build_prompt_token_ids(spec.input_len, spec.seed, vocab_size)
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": spec.output_len,
        "min_tokens": spec.output_len,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    return json.dumps(payload).encode("utf-8")


async def post_control(
    client: httpx.AsyncClient,
    base_url: str,
    path: str,
) -> httpx.Response:
    response = await client.post(f"{base_url}{path}")
    response.raise_for_status()
    return response


async def send_one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    vocab_size: int,
    spec: RequestSpec,
    started_state: dict[str, int],
    started_lock: asyncio.Lock,
    all_started_event: asyncio.Event,
    total_requests: int,
) -> dict:
    payload_bytes = build_payload_bytes(spec, model, vocab_size)

    async with started_lock:
        started_state["count"] += 1
        if started_state["count"] == total_requests:
            all_started_event.set()

    start_time = time.perf_counter()
    response = await client.post(
        f"{base_url}/v1/completions",
        content=payload_bytes,
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": spec.request_id,
            "X-data-parallel-rank": str(spec.dp_rank),
        },
    )
    latency_s = time.perf_counter() - start_time

    response.raise_for_status()
    usage = response.json().get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is not None and int(completion_tokens) != spec.output_len:
        raise RuntimeError(
            f"{spec.request_id}: expected completion_tokens={spec.output_len}, "
            f"got {completion_tokens}"
        )

    return {
        "request_id": spec.request_id,
        "dp_rank": spec.dp_rank,
        "input_len": spec.input_len,
        "output_len": spec.output_len,
        "latency_s": latency_s,
    }


async def main() -> None:
    args = parse_args()
    if args.dp_size <= 0:
        raise ValueError(f"--dp-size must be >= 1, got {args.dp_size}")
    if args.output_len <= 0:
        raise ValueError(f"--output-len must be >= 1, got {args.output_len}")
    if args.kv_cache_tokens_per_rank <= 0:
        raise ValueError(
            "--kv-cache-tokens-per-rank must be >= 1, "
            f"got {args.kv_cache_tokens_per_rank}"
        )
    if args.queue_settle_seconds < 0:
        raise ValueError(
            "--queue-settle-seconds must be >= 0, "
            f"got {args.queue_settle_seconds}"
        )
    if args.request_timeout_seconds <= 0:
        raise ValueError(
            "--request-timeout-seconds must be > 0, "
            f"got {args.request_timeout_seconds}"
        )
    if args.print_first_n < 0:
        raise ValueError(f"--print-first-n must be >= 0, got {args.print_first_n}")

    base_url = args.base_url.rstrip("/")
    input_lengths = load_input_lengths(args.lens_json)
    vocab_size = resolve_vocab_size(args.model, args.vocab_size)
    assigned_ranks, waiting_requests, waiting_tokens, free_kv_tokens = (
        preallocate_dp_ranks(
            input_lengths=input_lengths,
            dp_size=args.dp_size,
            output_len=args.output_len,
            dispatch_policy=args.dispatch_policy,
            kv_cache_tokens_per_rank=args.kv_cache_tokens_per_rank,
        )
    )
    specs = build_request_specs(
        input_lengths=input_lengths,
        assigned_ranks=assigned_ranks,
        output_len=args.output_len,
        seed=args.seed,
    )
    rank_requests, batch_sizes, kv_cache_total_tokens, kv_cache_free_tokens = (
        build_rank_request_arrays(
            specs=specs,
            dp_size=args.dp_size,
            kv_cache_tokens_per_rank=args.kv_cache_tokens_per_rank,
        )
    )

    print("Preparing staged profile run with custom input lengths")
    print(f"  base_url: {base_url}")
    print(f"  model: {args.model}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  lengths_file: {args.lens_json}")
    print(f"  requests: {len(specs)}")
    print(f"  dp_size: {args.dp_size}")
    print(f"  output_len: {args.output_len}")
    print(f"  dispatch_policy: {args.dispatch_policy}")
    print(f"  kv_cache_tokens_per_rank: {args.kv_cache_tokens_per_rank}")
    print("  routing: send X-data-parallel-rank header")
    print("  pre-allocation per rank:")
    for rank in range(args.dp_size):
        print(
            f"    dp_rank={rank}: requests={waiting_requests[rank]} "
            f"waiting_tokens={waiting_tokens[rank]} "
            f"kv_cache_total_tokens={kv_cache_total_tokens[rank]} "
            f"free_kv_tokens={free_kv_tokens[rank]}"
        )

    if args.print_first_n > 0:
        print("  sample_specs:")
        for spec in specs[: args.print_first_n]:
            print(
                "    "
                f"{spec.request_id}: dp={spec.dp_rank} input={spec.input_len} "
                f"output={spec.output_len} seed={spec.seed}"
            )

    if args.dry_run:
        output_dir = Path(args.dry_run_output_dir)
        json_path, kv_cache_plot_path, batch_size_plot_path = save_dry_run_outputs(
            output_dir=output_dir,
            dp_size=args.dp_size,
            dispatch_policy=args.dispatch_policy,
            kv_cache_tokens_per_rank=args.kv_cache_tokens_per_rank,
            output_len=args.output_len,
            lens_json=args.lens_json,
            rank_requests=rank_requests,
            batch_sizes=batch_sizes,
            kv_cache_total_tokens=kv_cache_total_tokens,
            kv_cache_free_tokens=kv_cache_free_tokens,
        )
        print("Dry run complete")
        print(f"  allocation_json: {json_path}")
        print(f"  kv_cache_plot: {kv_cache_plot_path}")
        print(f"  batch_size_plot: {batch_size_plot_path}")
        return

    timeout = httpx.Timeout(
        connect=30.0,
        read=args.request_timeout_seconds,
        write=120.0,
        pool=None,
    )
    limits = httpx.Limits(
        max_connections=len(specs) + 8,
        max_keepalive_connections=0,
    )

    started_state = {"count": 0}
    started_lock = asyncio.Lock()
    all_started_event = asyncio.Event()

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        health = await client.get(f"{base_url}/health")
        health.raise_for_status()

        # print("Pausing generation with mode=keep")
        # await post_control(client, base_url, "/pause?mode=keep")

        request_tasks = [
            asyncio.create_task(
                send_one_request(
                    client=client,
                    base_url=base_url,
                    model=args.model,
                    vocab_size=vocab_size,
                    spec=spec,
                    started_state=started_state,
                    started_lock=started_lock,
                    all_started_event=all_started_event,
                    total_requests=len(specs),
                )
            )
            for spec in specs
        ]

        await all_started_event.wait()
        print("All HTTP POSTs have been issued; waiting for queue to settle")
        await asyncio.sleep(args.queue_settle_seconds)

        print("Starting profiler")
        await post_control(client, base_url, "/start_profile")

        # print("Resuming generation")
        # await post_control(client, base_url, "/resume")

        try:
            results = await asyncio.gather(*request_tasks)
        finally:
            print("Stopping profiler")
            await post_control(client, base_url, "/stop_profile")

    per_rank_counts = {rank: 0 for rank in range(args.dp_size)}
    latencies = []
    for item in results:
        per_rank_counts[item["dp_rank"]] += 1
        latencies.append(item["latency_s"])

    print("Run complete")
    print(f"  requests_completed: {len(results)}")
    print(f"  latency_min_s: {min(latencies):.3f}")
    print(f"  latency_median_s: {sorted(latencies)[len(latencies) // 2]:.3f}")
    print(f"  latency_max_s: {max(latencies):.3f}")
    for dp_rank in range(args.dp_size):
        print(f"  dp_rank={dp_rank}: total={per_rank_counts[dp_rank]}")


if __name__ == "__main__":
    asyncio.run(main())
