#!/usr/bin/env python3

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pause vLLM, send requests from custom nested input-length JSON, "
            "start profiler, resume generation, and wait for completion. "
            "This script does not send X-data-parallel-rank."
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
    input_len: int,
    output_len: int,
    seed: int,
    request_id: str,
    started_state: dict[str, int],
    started_lock: asyncio.Lock,
    all_started_event: asyncio.Event,
    total_requests: int,
) -> float:
    prompt_token_ids = build_prompt_token_ids(input_len, seed, vocab_size)
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": output_len,
        "min_tokens": output_len,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 1.0,
    }

    async with started_lock:
        started_state["count"] += 1
        if started_state["count"] == total_requests:
            all_started_event.set()

    start_time = time.perf_counter()
    response = await client.post(
        f"{base_url}/v1/completions",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        },
    )
    latency_s = time.perf_counter() - start_time

    response.raise_for_status()
    usage = (response.json().get("usage") or {})
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is not None and int(completion_tokens) != output_len:
        raise RuntimeError(
            f"{request_id}: expected completion_tokens={output_len}, "
            f"got {completion_tokens}"
        )
    return latency_s


async def main() -> None:
    args = parse_args()
    if args.output_len <= 0:
        raise ValueError(f"--output-len must be >= 1, got {args.output_len}")
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

    base_url = args.base_url.rstrip("/")
    input_lengths = load_input_lengths(args.lens_json)
    vocab_size = resolve_vocab_size(args.model, args.vocab_size)

    print("Preparing staged profile run with custom input lengths")
    print(f"  base_url: {base_url}")
    print(f"  model: {args.model}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  lengths_file: {args.lens_json}")
    print(f"  requests: {len(input_lengths)}")
    print(f"  output_len: {args.output_len}")
    print("  routing: no X-data-parallel-rank header")

    timeout = httpx.Timeout(
        connect=30.0,
        read=args.request_timeout_seconds,
        write=120.0,
        pool=None,
    )
    limits = httpx.Limits(
        max_connections=len(input_lengths) + 8,
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
                    input_len=input_len,
                    output_len=args.output_len,
                    seed=args.seed + idx,
                    request_id=f"custom-{idx:06d}",
                    started_state=started_state,
                    started_lock=started_lock,
                    all_started_event=all_started_event,
                    total_requests=len(input_lengths),
                )
            )
            for idx, input_len in enumerate(input_lengths)
        ]

        await all_started_event.wait()
        print("All HTTP POSTs have been issued; waiting for queue to settle")
        await asyncio.sleep(args.queue_settle_seconds)

        print("Starting profiler")
        await post_control(client, base_url, "/start_profile")

        # print("Resuming generation")
        # await post_control(client, base_url, "/resume")

        try:
            
            latencies = await asyncio.gather(*request_tasks)
        finally:
            print("Stopping profiler")
            await post_control(client, base_url, "/stop_profile")

    print("Run complete")
    print(f"  requests_completed: {len(latencies)}")
    print(f"  latency_min_s: {min(latencies):.3f}")
    print(f"  latency_median_s: {sorted(latencies)[len(latencies) // 2]:.3f}")
    print(f"  latency_max_s: {max(latencies):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
