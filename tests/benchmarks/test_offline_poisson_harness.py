# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import vllm.benchmarks.offline_poisson_harness as harness_mod
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.benchmarks.offline_poisson_harness import (
    MAX_REQUESTS_CSV_ROWS,
    _apply_benchmark_arg_defaults,
    _compute_pre_forward_ttft_ms,
    _enforce_harness_observability,
    _request_clean_cluster_shutdown,
    _resolve_profile_prefix,
    _start_frontend_profile_if_requested,
    _stop_frontend_profile,
    _shutdown_frontend_async_llm,
    _submit_one_request,
    build_success_record,
    build_poisson_arrival_deadlines_ns,
    build_summary,
    connector_mode_from_config,
    default_cudagraph_capture_sizes,
    load_length_requests,
    resolve_csv_repeat,
    resolve_max_requests_against_csv,
    split_warmup_and_measured_requests,
)
from vllm.benchmarks.datasets import SampleRequest
from vllm.v1.ttft_timing import RequestTTFTTrace


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
def test_resolve_max_requests_against_csv_rows() -> None:
    assert resolve_max_requests_against_csv(
        total_rows=2353,
        max_requests=MAX_REQUESTS_CSV_ROWS,
    ) == 2353


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
            "ttft_including_forward_ms": 0.5,
            "ttft_post_schedule_to_first_token_ms": 0.1,
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
            "ttft_including_forward_ms": None,
            "ttft_post_schedule_to_first_token_ms": None,
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
        ttft_semantics="first_real_token_latency",
        first_submit_ts_ns=100,
        last_finish_ts_ns=2_200,
    )

    assert summary["total_requests"] == 2
    assert summary["successful_requests"] == 1
    assert summary["failed_requests"] == 1
    assert summary["benchmark_runtime_s"] == pytest.approx((2_200 - 100) / 1e9)
    assert summary["ttft_semantics"] == "first_real_token_latency"
    assert summary["e2e_ms"]["mean"] == pytest.approx(1.0)
    assert summary["ttft_ms"]["p50"] == pytest.approx(0.4)
    assert summary["ttft_including_forward_ms"]["p50"] == pytest.approx(0.5)
    assert summary["ttft_post_schedule_to_first_token_ms"]["p50"] == pytest.approx(
        0.1
    )


def _build_fake_metrics() -> SimpleNamespace:
    return SimpleNamespace(
        queued_ts=1.0,
        scheduled_ts=1.122,
        first_token_ts=1.1367,
        last_token_ts=1.2400,
        first_token_latency=0.1367,
        ttft_trace=RequestTTFTTrace(
            api_preprocess_ns=1_800_000,
            ipc_in_decode_ns=900_000,
            engine_preprocess_ns=1_100_000,
            first_batch_load_kv_ns=12_000_000,
        ),
    )


@pytest.mark.benchmark
def test_compute_pre_forward_ttft_ms_uses_schedule_boundary() -> None:
    assert _compute_pre_forward_ttft_ms(_build_fake_metrics()) == pytest.approx(125.8)


@pytest.mark.benchmark
def test_build_success_record_uses_pre_forward_ttft_for_dummy_prefill() -> None:
    record = build_success_record(
        request=SampleRequest(
            prompt=[11, 12, 13],
            prompt_len=3,
            expected_output_len=2,
            request_id="r0-000000",
        ),
        output=RequestOutput(
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
            metrics=_build_fake_metrics(),
        ),
        submit_ts_ns=100,
        finish_ts_ns=2_100,
        kv_transfer_config={
            "kv_connector": "DecodeBenchConnector",
            "kv_connector_extra_config": {
                "dummy_prefill": True,
            },
        },
    )

    assert record["ttft_ms"] == pytest.approx(125.8)
    assert record["ttft_including_forward_ms"] == pytest.approx(136.7)
    assert record["ttft_post_schedule_to_first_token_ms"] == pytest.approx(10.9)
    assert record["queued_time_ms"] == pytest.approx(122.0)


@pytest.mark.benchmark
def test_build_success_record_keeps_first_token_ttft_without_dummy_prefill() -> None:
    record = build_success_record(
        request=SampleRequest(
            prompt=[11, 12, 13],
            prompt_len=3,
            expected_output_len=2,
            request_id="r0-000000",
        ),
        output=RequestOutput(
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
            metrics=_build_fake_metrics(),
        ),
        submit_ts_ns=100,
        finish_ts_ns=2_100,
        kv_transfer_config=None,
    )

    assert record["ttft_ms"] == pytest.approx(136.7)
    assert record["ttft_including_forward_ms"] == pytest.approx(136.7)
    assert record["ttft_post_schedule_to_first_token_ms"] is None


@pytest.mark.benchmark
def test_connector_mode_from_config() -> None:
    assert connector_mode_from_config(None) == "none"
    assert connector_mode_from_config(
        {"kv_connector": "DecodeBenchConnector"}
    ) == "decode_bench"


@pytest.mark.benchmark
def test_log_frontend_phase_includes_dispatch_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        harness_mod.logger,
        "info",
        lambda message, *args, **_kwargs: messages.append(
            message % args if args else message),
    )

    harness_mod._log_frontend_phase(
        "frontend_start",
        args=Namespace(
            request_rate=12.5,
            data_parallel_dispatch_policy="least_batch",
            data_parallel_size=16,
            data_parallel_size_local=8,
            save_merged_parquet=False,
        ),
        output_dir=tmp_path,
        frontend_started_at_s=0.0,
    )

    assert any("dispatch_policy=least_batch" in message for message in messages)


@pytest.mark.benchmark
def test_apply_benchmark_arg_defaults_normalizes_legacy_decode_bench_config() -> None:
    args = Namespace(
        kv_transfer_config={
            "kv_connector": "DecodeBenchConnector",
            "kv_role": "kv_both",
            "fill_mean": 0.015,
            "fill_std": 0.0,
            "kv_connector_extra_config": {
                "dummy_prefill": True,
                "dummy_output_token_id": 2,
            },
        },
        cudagraph_capture_sizes=[1],
        compilation_config=None,
        max_num_seqs=None,
    )

    _apply_benchmark_arg_defaults(args)

    assert "fill_mean" not in args.kv_transfer_config
    assert "fill_std" not in args.kv_transfer_config
    assert args.kv_transfer_config["kv_connector_extra_config"] == {
        "fill_mean": 0.015,
        "fill_std": 0.0,
        "dummy_prefill": True,
        "dummy_output_token_id": 2,
    }


@pytest.mark.benchmark
def test_default_cudagraph_capture_sizes() -> None:
    assert default_cudagraph_capture_sizes(8) == [1, 2, 4, 8]
    assert default_cudagraph_capture_sizes(16) == [1, 2, 4, 8, 16]
    assert default_cudagraph_capture_sizes(48) == [1, 2, 4, 8, 16, 32, 48]


@pytest.mark.benchmark
def test_enforce_harness_observability_enables_step_and_ttft_timing() -> None:
    args = Namespace(
        disable_log_stats=False,
        skip_tokenizer_init=False,
        enable_logging_step_timing_details=False,
        enable_graph_replay_timing=False,
        logging_step_timing_interval=1,
        enable_logging_ttft_timing_details=False,
        logging_ttft_timing_interval=99,
    )

    _enforce_harness_observability(args)

    assert args.disable_log_stats is False
    assert args.enable_logging_step_timing_details is True
    assert args.enable_graph_replay_timing is True
    assert args.logging_step_timing_interval == 10
    assert args.enable_logging_ttft_timing_details is True
    assert args.logging_ttft_timing_interval == 1


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
    assert asyncio.run(_request_clean_cluster_shutdown(async_llm)) is True
    assert async_llm.engine_core.calls == ["request_shutdown_and_wait"]


@pytest.mark.benchmark
def test_request_clean_cluster_shutdown_logs_frontend_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _EngineCore:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def call_utility_async(self, method: str) -> None:
            self.calls.append(method)
            await asyncio.sleep(0.01)

    class _ParallelConfig:
        data_parallel_size = 16
        data_parallel_size_local = 8

    class _VllmConfig:
        parallel_config = _ParallelConfig()

    class _AsyncLLM:
        vllm_config = _VllmConfig()

        def __init__(self) -> None:
            self.engine_core = _EngineCore()

    args = Namespace(
        request_rate=12.5,
        data_parallel_size=16,
        data_parallel_size_local=8,
        save_merged_parquet=False,
    )
    async_llm = _AsyncLLM()
    messages: list[str] = []
    monkeypatch.setattr(harness_mod, "FRONTEND_TEARDOWN_HEARTBEAT_SEC", 0.001)
    monkeypatch.setattr(
        harness_mod.logger,
        "info",
        lambda message, *args, **_kwargs: messages.append(
            message % args if args else message),
    )

    asyncio.run(
        harness_mod._request_clean_cluster_shutdown(
            async_llm,
            frontend_args=args,
            output_dir=tmp_path,
            frontend_started_at_s=0.0,
            measured_requests=64,
        ))

    joined = "\n".join(messages)
    assert "phase=clean_cluster_shutdown_request_start" in joined
    assert "phase=clean_cluster_shutdown_waiting" in joined
    assert "phase=clean_cluster_shutdown_request_done" in joined


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
    assert asyncio.run(_request_clean_cluster_shutdown(async_llm)) is True
    assert async_llm.engine_core.calls == []


@pytest.mark.benchmark
def test_request_clean_cluster_shutdown_times_out_and_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _EngineCore:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.cancelled = False

        async def call_utility_async(self, method: str) -> None:
            self.calls.append(method)
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    class _ParallelConfig:
        data_parallel_size = 16
        data_parallel_size_local = 8

    class _VllmConfig:
        parallel_config = _ParallelConfig()

    class _AsyncLLM:
        vllm_config = _VllmConfig()

        def __init__(self) -> None:
            self.engine_core = _EngineCore()

    args = Namespace(
        request_rate=12.5,
        data_parallel_size=16,
        data_parallel_size_local=8,
        save_merged_parquet=False,
    )
    async_llm = _AsyncLLM()
    info_messages: list[str] = []
    warning_messages: list[str] = []
    monkeypatch.setattr(harness_mod, "FRONTEND_TEARDOWN_HEARTBEAT_SEC", 0.001)
    monkeypatch.setattr(harness_mod, "FRONTEND_TEARDOWN_TIMEOUT_SEC", 0.003)
    monkeypatch.setattr(
        harness_mod.logger,
        "info",
        lambda message, *args, **_kwargs: info_messages.append(
            message % args if args else message),
    )
    monkeypatch.setattr(
        harness_mod.logger,
        "warning",
        lambda message, *args, **_kwargs: warning_messages.append(
            message % args if args else message),
    )

    graceful = asyncio.run(
        harness_mod._request_clean_cluster_shutdown(
            async_llm,
            frontend_args=args,
            output_dir=tmp_path,
            frontend_started_at_s=0.0,
            measured_requests=64,
        ))

    assert graceful is False
    assert async_llm.engine_core.calls == ["request_shutdown_and_wait"]
    assert async_llm.engine_core.cancelled is True
    assert "phase=clean_cluster_shutdown_timeout" in "\n".join(info_messages)
    assert any(
        "falling back to forced local cleanup" in message
        for message in warning_messages
    )


@pytest.mark.benchmark
def test_shutdown_frontend_async_llm_uses_forced_cleanup_after_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AsyncLLM:
        def __init__(self) -> None:
            self.begin_shutdown_calls = 0
            self.shutdown_calls: list[float | None] = []

        def begin_shutdown(self) -> None:
            self.begin_shutdown_calls += 1

        def shutdown(self, timeout: float | None = None) -> None:
            self.shutdown_calls.append(timeout)

    args = Namespace(
        request_rate=12.5,
        data_parallel_size=16,
        data_parallel_size_local=8,
        save_merged_parquet=False,
    )
    async_llm = _AsyncLLM()
    messages: list[str] = []

    async def _timeout_shutdown(*_args, **_kwargs) -> bool:
        return False

    monkeypatch.setattr(
        harness_mod,
        "_request_clean_cluster_shutdown",
        _timeout_shutdown,
    )
    monkeypatch.setattr(
        harness_mod.logger,
        "info",
        lambda message, *args, **_kwargs: messages.append(
            message % args if args else message),
    )

    asyncio.run(
        _shutdown_frontend_async_llm(
            async_llm,
            args=args,
            output_dir=tmp_path,
            frontend_started_at_s=0.0,
            measured_requests=64,
            summary_written=True,
            parquet_written=False,
        ))

    joined = "\n".join(messages)
    assert async_llm.begin_shutdown_calls == 1
    assert async_llm.shutdown_calls == [0.0]
    assert "phase=clean_cluster_shutdown_fallback" in joined
    assert "phase=async_llm_shutdown_start" in joined
    assert "timeout_s=0.000" in joined


@pytest.mark.benchmark
def test_resolve_profile_prefix_defaults_to_output_dir_name(tmp_path: Path) -> None:
    args = Namespace(profile_prefix=None)
    output_dir = tmp_path / "case_a"

    assert _resolve_profile_prefix(args, output_dir) == "case_a"


@pytest.mark.benchmark
def test_start_frontend_profile_if_requested_uses_default_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AsyncLLM:
        def __init__(self) -> None:
            self.prefixes: list[str | None] = []

        async def start_profile(self, profile_prefix: str | None = None) -> None:
            self.prefixes.append(profile_prefix)

    args = Namespace(
        profile_after_warmup=True,
        profile_prefix=None,
        request_rate=40.0,
        data_parallel_dispatch_policy="waiting_x4_plus_running",
        data_parallel_size=32,
        data_parallel_size_local=8,
        save_merged_parquet=False,
    )
    async_llm = _AsyncLLM()
    messages: list[str] = []
    monkeypatch.setattr(
        harness_mod.logger,
        "info",
        lambda message, *args, **_kwargs: messages.append(
            message % args if args else message),
    )

    profile_prefix = asyncio.run(
        _start_frontend_profile_if_requested(
            async_llm,
            args=args,
            output_dir=tmp_path / "case_a",
            frontend_started_at_s=0.0,
            measured_requests=256,
        ))

    joined = "\n".join(messages)
    assert profile_prefix == "case_a"
    assert async_llm.prefixes == ["case_a"]
    assert "phase=profile_start_start" in joined
    assert "phase=profile_start_done" in joined
    assert "profile_prefix=case_a" in joined


@pytest.mark.benchmark
def test_stop_frontend_profile_logs_stop_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AsyncLLM:
        def __init__(self) -> None:
            self.stop_calls = 0

        async def stop_profile(self) -> None:
            self.stop_calls += 1

    args = Namespace(
        request_rate=40.0,
        data_parallel_dispatch_policy="waiting_x4_plus_running",
        data_parallel_size=32,
        data_parallel_size_local=8,
        save_merged_parquet=False,
    )
    async_llm = _AsyncLLM()
    messages: list[str] = []
    monkeypatch.setattr(
        harness_mod.logger,
        "info",
        lambda message, *args, **_kwargs: messages.append(
            message % args if args else message),
    )

    asyncio.run(
        _stop_frontend_profile(
            async_llm,
            args=args,
            output_dir=tmp_path / "case_a",
            frontend_started_at_s=0.0,
            measured_requests=256,
            profile_prefix="case_a",
        ))

    joined = "\n".join(messages)
    assert async_llm.stop_calls == 1
    assert "phase=profile_stop_start" in joined
    assert "phase=profile_stop_done" in joined
    assert "profile_prefix=case_a" in joined
