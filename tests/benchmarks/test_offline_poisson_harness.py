# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from pathlib import Path

import pytest

from vllm.outputs import CompletionOutput, RequestOutput
from vllm.benchmarks.offline_poisson_harness import (
    _request_clean_cluster_shutdown,
    _submit_one_request,
    build_poisson_arrival_deadlines_ns,
    build_summary,
    connector_mode_from_config,
    default_cudagraph_capture_sizes,
    load_length_requests,
    resolve_csv_repeat,
    split_warmup_and_measured_requests,
)
from vllm.benchmarks.datasets import SampleRequest


@pytest.fixture(scope="session")
def fake_tokenizer():
    class _FakeTokenizer:
        vocab_size = 1024
        all_special_ids = [0, 1, 2]

    return _FakeTokenizer()


@pytest.mark.benchmark
def test_load_length_requests_uses_exact_lengths(
    tmp_path: Path,
    fake_tokenizer,
) -> None:
    csv_path = tmp_path / "requests.csv"
    csv_path.write_text(
        "prompt_len,output_len\n4,7\n6,5\n5,9\n",
        encoding="utf-8",
    )

    requests = load_length_requests(
        csv_path=str(csv_path),
        tokenizer=fake_tokenizer,
        seed=123,
        request_id_prefix="r123-",
    )

    assert [request.request_id for request in requests] == [
        "r123-000000",
        "r123-000001",
        "r123-000002",
    ]
    assert [request.prompt_len for request in requests] == [4, 6, 5]
    assert [request.expected_output_len for request in requests] == [7, 5, 9]
    assert [len(request.prompt) for request in requests] == [4, 6, 5]


@pytest.mark.benchmark
def test_load_length_requests_repeats_csv_rows(
    tmp_path: Path,
    fake_tokenizer,
) -> None:
    csv_path = tmp_path / "requests.csv"
    csv_path.write_text(
        "prompt_len,output_len\n4,7\n6,5\n5,9\n",
        encoding="utf-8",
    )

    requests = load_length_requests(
        csv_path=str(csv_path),
        tokenizer=fake_tokenizer,
        seed=123,
        request_id_prefix="r123-",
        csv_repeat=2,
    )

    assert [request.request_id for request in requests] == [
        "r123-000000",
        "r123-000001",
        "r123-000002",
        "r123-000003",
        "r123-000004",
        "r123-000005",
    ]
    assert [request.prompt_len for request in requests] == [4, 6, 5, 4, 6, 5]
    assert [request.expected_output_len for request in requests] == [
        7,
        5,
        9,
        7,
        5,
        9,
    ]
    assert [len(request.prompt) for request in requests] == [4, 6, 5, 4, 6, 5]


@pytest.mark.benchmark
def test_resolve_csv_repeat_uses_warmup_plus_max_requests() -> None:
    assert resolve_csv_repeat(
        total_rows=27,
        warmup_requests=32,
        max_requests=9000,
        csv_repeat=None,
    ) == 335


@pytest.mark.benchmark
def test_resolve_csv_repeat_prefers_explicit_override() -> None:
    assert resolve_csv_repeat(
        total_rows=27,
        warmup_requests=32,
        max_requests=9000,
        csv_repeat=400,
    ) == 400


@pytest.mark.benchmark
def test_split_warmup_and_measured_requests(
    tmp_path: Path,
    fake_tokenizer,
) -> None:
    csv_path = tmp_path / "requests.csv"
    csv_path.write_text(
        "prompt_len,output_len\n4,7\n6,5\n5,9\n8,3\n",
        encoding="utf-8",
    )
    requests = load_length_requests(
        csv_path=str(csv_path),
        tokenizer=fake_tokenizer,
        seed=0,
        request_id_prefix="r0-",
    )

    warmup, measured = split_warmup_and_measured_requests(
        requests,
        warmup_requests=1,
        max_requests=2,
    )

    assert [request.request_id for request in warmup] == ["r0-000000"]
    assert [request.request_id for request in measured] == [
        "r0-000001",
        "r0-000002",
    ]


@pytest.mark.benchmark
def test_build_poisson_arrival_deadlines_is_deterministic() -> None:
    deadlines_a = build_poisson_arrival_deadlines_ns(
        num_requests=5,
        request_rate=4.0,
        seed=42,
    )
    deadlines_b = build_poisson_arrival_deadlines_ns(
        num_requests=5,
        request_rate=4.0,
        seed=42,
    )

    assert deadlines_a == deadlines_b
    assert deadlines_a == sorted(deadlines_a)
    assert all(deadline > 0 for deadline in deadlines_a)


@pytest.mark.benchmark
def test_build_poisson_arrival_deadlines_handles_inf() -> None:
    assert build_poisson_arrival_deadlines_ns(
        num_requests=4,
        request_rate=float("inf"),
        seed=1,
    ) == [0, 0, 0, 0]


@pytest.mark.benchmark
def test_build_summary_counts_failures_and_uses_measured_window() -> None:
    records = [
        {
            "request_id": "r0-000000",
            "submit_ts_ns": 100,
            "finish_ts_ns": 1_100,
            "e2e_ms": 1.0,
            "ttft_ms": 0.4,
            "queued_time_ms": 0.1,
            "kv_fill_ms": 0.0,
            "prefill_time_ms": 0.1,
            "decode_time_ms": 0.5,
            "inference_time_ms": 0.6,
            "prompt_len": 4,
            "expected_output_len": 7,
            "actual_output_tokens": 7,
            "finish_reason": "length",
            "num_cached_tokens": 0,
            "is_error": False,
            "error_message": None,
        },
        {
            "request_id": "r0-000001",
            "submit_ts_ns": 200,
            "finish_ts_ns": 2_200,
            "e2e_ms": 2.0,
            "ttft_ms": None,
            "queued_time_ms": None,
            "kv_fill_ms": 0.0,
            "prefill_time_ms": None,
            "decode_time_ms": None,
            "inference_time_ms": None,
            "prompt_len": 8,
            "expected_output_len": 3,
            "actual_output_tokens": 0,
            "finish_reason": None,
            "num_cached_tokens": 0,
            "is_error": True,
            "error_message": "boom",
        },
    ]

    summary = build_summary(
        records=records,
        connector_mode="none",
        first_submit_ts_ns=100,
        last_finish_ts_ns=2_200,
    )

    assert summary["total_requests"] == 2
    assert summary["successful_requests"] == 1
    assert summary["failed_requests"] == 1
    assert summary["benchmark_runtime_s"] == pytest.approx((2_200 - 100) / 1e9)
    assert summary["e2e_ms"]["mean"] == pytest.approx(1.0)
    assert summary["ttft_ms"]["p50"] == pytest.approx(0.4)


@pytest.mark.benchmark
def test_connector_mode_from_config() -> None:
    assert connector_mode_from_config(None) == "none"
    assert connector_mode_from_config(
        {"kv_connector": "DecodeBenchConnector"}
    ) == "decode_bench"


@pytest.mark.benchmark
def test_default_cudagraph_capture_sizes() -> None:
    assert default_cudagraph_capture_sizes(8) == [1, 2, 4, 8]
    assert default_cudagraph_capture_sizes(16) == [1, 2, 4, 8, 16]
    assert default_cudagraph_capture_sizes(48) == [1, 2, 4, 8, 16, 32, 48]


@pytest.mark.benchmark
def test_submit_one_request_uses_processor_inputs() -> None:
    class _Collector:
        async def get(self) -> RequestOutput:
            return RequestOutput(
                request_id="r0-000000",
                prompt=None,
                prompt_token_ids=[11, 12, 13],
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="",
                        token_ids=[21, 22],
                        cumulative_logprob=None,
                        logprobs=None,
                        finish_reason="length",
                    )
                ],
                finished=True,
            )

    class _Engine:
        def __init__(self) -> None:
            self.prompt = None

        async def add_request(self, **kwargs):
            self.prompt = kwargs["prompt"]
            return _Collector()

    async def _run() -> dict:
        engine = _Engine()
        inflight = set()
        await _submit_one_request(
            engine=engine,
            request=SampleRequest(
                prompt=[11, 12, 13],
                prompt_len=3,
                expected_output_len=2,
                request_id="r0-000000",
            ),
            recorder=None,
            inflight=inflight,
        )
        await asyncio.gather(*tuple(inflight))
        assert engine.prompt is not None
        return engine.prompt

    prompt = asyncio.run(_run())
    assert prompt["type"] == "token"
    assert prompt["prompt_token_ids"] == [11, 12, 13]


@pytest.mark.benchmark
def test_request_clean_cluster_shutdown_targets_remote_dp_engines() -> None:
    class _EngineCore:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def call_utility_async(self, method: str) -> None:
            self.calls.append(method)

    class _ParallelConfig:
        data_parallel_size = 16
        data_parallel_size_local = 8

    class _VllmConfig:
        parallel_config = _ParallelConfig()

    class _AsyncLLM:
        vllm_config = _VllmConfig()

        def __init__(self) -> None:
            self.engine_core = _EngineCore()

    async_llm = _AsyncLLM()
    asyncio.run(_request_clean_cluster_shutdown(async_llm))
    assert async_llm.engine_core.calls == ["request_shutdown"]


@pytest.mark.benchmark
def test_request_clean_cluster_shutdown_skips_local_only_runs() -> None:
    class _EngineCore:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def call_utility_async(self, method: str) -> None:
            self.calls.append(method)

    class _ParallelConfig:
        data_parallel_size = 8
        data_parallel_size_local = 8

    class _VllmConfig:
        parallel_config = _ParallelConfig()

    class _AsyncLLM:
        vllm_config = _VllmConfig()

        def __init__(self) -> None:
            self.engine_core = _EngineCore()

    async_llm = _AsyncLLM()
    asyncio.run(_request_clean_cluster_shutdown(async_llm))
    assert async_llm.engine_core.calls == []
