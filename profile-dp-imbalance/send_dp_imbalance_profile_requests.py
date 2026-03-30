#!/usr/bin/env python3

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


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
            "Pause vLLM, queue per-DP requests with X-data-parallel-rank, "
            "start profiling, resume generation, and wait for completion."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="OpenAI-compatible vLLM base URL on node 0.",
    )
    parser.add_argument(
        "--model",
        default="/mnt/nvme1n1/ml_research/models/deepseek-v3-1024k",
        help="Model name/path passed to the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=4,
        help="Total DP size. Requests will be sent to ranks [0, dp_size).",
    )
    parser.add_argument(
        "--dp-local-size",
        type=int,
        default=0,
        help=(
            "Number of DP ranks per machine. If unset, infer from the "
            "supported 8-GPU/node topologies: dp=4->1, dp=8->2, dp=16->4, "
            "dp=32->8. Exactly one long request is assigned to the first "
            "DP rank of each machine. With --short-only, this is only "
            "validated and logged when explicitly provided."
        ),
    )
    parser.add_argument(
        "--total-requests-per-dp",
        type=int,
        default=64,
        help=(
            "Total requests per DP rank. Ranks selected for a long request "
            "receive 1 long request plus total_requests_per_dp - 1 short "
            "requests; other ranks receive total_requests_per_dp short "
            "requests."
        ),
    )
    parser.add_argument(
        "--short-input-len",
        type=int,
        default=1024,
        help="Short request prompt length in tokens.",
    )
    parser.add_argument(
        "--long-input-len",
        type=int,
        default=512 * 1024,
        help="Long request prompt length in tokens.",
    )
    parser.add_argument(
        "--short-only",
        action="store_true",
        help="Only send short requests and skip long requests.",
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
        default=4,
        help="How many request specs to print before sending the run.",
    )
    return parser.parse_args()


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


def infer_dp_local_size(dp_size: int) -> int | None:
    return {
        4: 1,
        8: 2,
        16: 4,
        32: 8,
    }.get(dp_size)


def resolve_dp_local_size(dp_size: int, explicit_dp_local_size: int) -> int:
    if explicit_dp_local_size > 0:
        return explicit_dp_local_size
    if explicit_dp_local_size < 0:
        raise ValueError(
            f"--dp-local-size must be >= 0, got {explicit_dp_local_size}"
        )

    inferred_dp_local_size = infer_dp_local_size(dp_size)
    if inferred_dp_local_size is None:
        raise ValueError(
            "Could not infer --dp-local-size from --dp-size. "
            "Pass --dp-local-size explicitly."
        )
    return inferred_dp_local_size


def build_long_request_ranks(dp_size: int, dp_local_size: int) -> list[int]:
    return list(range(0, dp_size, dp_local_size))


def build_request_specs(args: argparse.Namespace) -> list[RequestSpec]:
    if args.total_requests_per_dp <= 0:
        raise ValueError(
            "--total-requests-per-dp must be >= 1, "
            f"got {args.total_requests_per_dp}"
        )

    specs: list[RequestSpec] = []
    long_request_ranks = set()
    if not args.short_only:
        long_request_ranks = set(
            build_long_request_ranks(args.dp_size, args.dp_local_size)
        )
    for dp_rank in range(args.dp_size):
        seed_base = args.seed + dp_rank * 10000
        has_long_request = dp_rank in long_request_ranks
        short_requests_for_rank = args.total_requests_per_dp - int(has_long_request)
        for request_idx in range(short_requests_for_rank):
            specs.append(
                RequestSpec(
                    request_id=f"dp{dp_rank}-short-{request_idx:03d}",
                    dp_rank=dp_rank,
                    input_len=args.short_input_len,
                    output_len=args.output_len,
                    seed=seed_base + request_idx,
                )
            )
        if has_long_request:
            specs.append(
                RequestSpec(
                    request_id=f"dp{dp_rank}-long-000",
                    dp_rank=dp_rank,
                    input_len=args.long_input_len,
                    output_len=args.output_len,
                    seed=seed_base + short_requests_for_rank,
                )
            )
    return specs


def build_prompt_token_ids(length: int, seed: int, vocab_size: int) -> list[int]:
    rng = random.Random(seed)
    low = 1024
    high = max(low + 1, vocab_size - 1024)
    width = high - low
    return [low + rng.randrange(width) for _ in range(length)]


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
    body = response.json()
    usage = body.get("usage") or {}
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
    if args.dp_local_size < 0:
        raise ValueError(
            f"--dp-local-size must be >= 0, got {args.dp_local_size}"
        )
    if args.short_only:
        if args.dp_local_size == 0:
            args.dp_local_size = None
        else:
            args.dp_local_size = resolve_dp_local_size(args.dp_size,
                                                       args.dp_local_size)
    else:
        args.dp_local_size = resolve_dp_local_size(args.dp_size, args.dp_local_size)
    if args.dp_local_size is not None:
        if args.dp_local_size <= 0:
            raise ValueError(
                f"--dp-local-size must be >= 1, got {args.dp_local_size}"
            )
        if args.dp_size % args.dp_local_size != 0:
            raise ValueError(
                "--dp-size must be divisible by --dp-local-size, "
                f"got dp_size={args.dp_size}, dp_local_size={args.dp_local_size}"
            )
    if args.total_requests_per_dp <= 0:
        raise ValueError(
            "--total-requests-per-dp must be >= 1, "
            f"got {args.total_requests_per_dp}"
        )
    if args.short_input_len <= 0:
        raise ValueError(
            f"--short-input-len must be >= 1, got {args.short_input_len}"
        )
    if not args.short_only and args.long_input_len <= 0:
        raise ValueError(f"--long-input-len must be >= 1, got {args.long_input_len}")
    if args.output_len <= 0:
        raise ValueError(f"--output-len must be >= 1, got {args.output_len}")
    if args.print_first_n < 0:
        raise ValueError(f"--print-first-n must be >= 0, got {args.print_first_n}")

    base_url = args.base_url.rstrip("/")
    vocab_size = resolve_vocab_size(args.model, args.vocab_size)
    specs = build_request_specs(args)
    long_request_ranks = []
    if not args.short_only:
        long_request_ranks = build_long_request_ranks(args.dp_size, args.dp_local_size)

    print("Preparing staged profile run")
    print(f"  base_url: {base_url}")
    print(f"  model: {args.model}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  dp_size: {args.dp_size}")
    print(f"  short_only: {args.short_only}")
    if args.dp_local_size is not None:
        print(f"  dp_local_size: {args.dp_local_size}")
    print(f"  total_requests_per_dp: {args.total_requests_per_dp}")
    if args.short_only:
        print(f"  short_requests_per_dp: {args.total_requests_per_dp}")
    else:
        print(
            "  short_requests_per_dp: "
            f"{args.total_requests_per_dp - 1} on long-request ranks, "
            f"{args.total_requests_per_dp} on other ranks"
        )
    print(f"  short_input_len: {args.short_input_len}")
    print(f"  output_len: {args.output_len}")
    if args.short_only:
        print("  per_machine_long: disabled")
    else:
        print(
            "  per_machine_long: "
            f"1x{args.long_input_len} on dp ranks "
            f"{long_request_ranks}"
        )
    print(f"  total_requests: {len(specs)}")
    print(f"  queue_settle_seconds: {args.queue_settle_seconds}")

    if args.print_first_n > 0:
        print("  sample_specs:")
        for spec in specs[: args.print_first_n]:
            print(
                "    "
                f"{spec.request_id}: dp={spec.dp_rank} "
                f"input={spec.input_len} output={spec.output_len} seed={spec.seed}"
            )

    timeout = httpx.Timeout(
        connect=30.0,
        read=args.request_timeout_seconds,
        write=120.0,
        pool=None,
    )
    limits = httpx.Limits(max_connections=len(specs) + 8, max_keepalive_connections=0)

    started_state = {"count": 0}
    started_lock = asyncio.Lock()
    all_started_event = asyncio.Event()

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        health = await client.get(f"{base_url}/health")
        health.raise_for_status()

        print("Pausing generation with mode=keep")
        await post_control(client, base_url, "/pause?mode=keep")

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

        print("Resuming generation")
        await post_control(client, base_url, "/resume")

        try:
            results = await asyncio.gather(*request_tasks)
        finally:
            print("Stopping profiler")
            await post_control(client, base_url, "/stop_profile")

    per_rank_counts = {rank: 0 for rank in range(args.dp_size)}
    per_rank_long = {rank: 0 for rank in range(args.dp_size)}
    latencies = []
    for item in results:
        per_rank_counts[item["dp_rank"]] += 1
        if "-long-" in item["request_id"]:
            per_rank_long[item["dp_rank"]] += 1
        latencies.append(item["latency_s"])

    print("Run complete")
    print(f"  requests_completed: {len(results)}")
    print(f"  latency_min_s: {min(latencies):.3f}")
    print(f"  latency_median_s: {sorted(latencies)[len(latencies) // 2]:.3f}")
    print(f"  latency_max_s: {max(latencies):.3f}")
    for dp_rank in range(args.dp_size):
        print(
            f"  dp_rank={dp_rank}: "
            f"total={per_rank_counts[dp_rank]} "
            f"long={per_rank_long[dp_rank]} "
            f"short={per_rank_counts[dp_rank] - per_rank_long[dp_rank]}"
        )


if __name__ == "__main__":
    asyncio.run(main())
