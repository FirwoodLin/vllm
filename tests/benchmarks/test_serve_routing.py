# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import time

import pytest

import vllm.benchmarks.serve as serve_module
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput


def _make_args(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        dataset_name="random",
        random_csv_path="/tmp/random.csv",
        backend="openai",
        endpoint="/v1/completions",
        model="test-model",
        skip_tokenizer_init=False,
        routing_prompt_len_threshold=None,
        routing_base_url_short=None,
        routing_base_url_long=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize(
    ("overrides", "error_fragment"),
    [
        (
            {"routing_base_url_short": "http://127.0.0.1:8000"},
            "--routing-base-url-short",
        ),
        (
            {
                "routing_base_url_short": "http://127.0.0.1:8000",
                "routing_base_url_long": "http://127.0.0.1:8001",
                "model": None,
            },
            "explicit --model",
        ),
        (
            {
                "routing_base_url_short": "http://127.0.0.1:8000",
                "routing_base_url_long": "http://127.0.0.1:8001",
                "random_csv_path": None,
            },
            "--random-csv-path",
        ),
        (
            {
                "routing_base_url_short": "http://127.0.0.1:8000",
                "routing_base_url_long": "http://127.0.0.1:8001",
                "backend": "openai-chat",
                "endpoint": "/v1/chat/completions",
            },
            "chat completion",
        ),
    ],
)
def test_validate_prompt_len_routing_args(overrides, error_fragment):
    args = _make_args(**overrides)
    with pytest.raises(ValueError, match=error_fragment):
        serve_module._validate_prompt_len_routing_args(args)


def test_maybe_get_prompt_len_routing_config_uses_default_threshold():
    args = _make_args(
        routing_base_url_short="http://127.0.0.1:8000",
        routing_base_url_long="http://127.0.0.1:8001",
    )

    routing_config = serve_module._maybe_get_prompt_len_routing_config(args)

    assert routing_config is not None
    assert (
        routing_config.threshold_prompt_len
        == serve_module.DEFAULT_ROUTING_PROMPT_LEN_THRESHOLD
    )
    assert routing_config.short.api_url == "http://127.0.0.1:8000/v1/completions"
    assert routing_config.long.api_url == "http://127.0.0.1:8001/v1/completions"


@pytest.mark.asyncio
async def test_benchmark_routes_prompt_len_requests(monkeypatch):
    short_base_url = "http://127.0.0.1:8000"
    long_base_url = "http://127.0.0.1:8001"
    short_api_url = f"{short_base_url}/v1/completions"
    long_api_url = f"{long_base_url}/v1/completions"

    completion_urls: list[str] = []
    profile_urls: list[str] = []
    ready_check_urls: list[str] = []
    spec_metric_calls: list[str] = []

    async def fake_request_func(request_func_input, session, pbar=None):
        if request_func_input.api_url.endswith("profile"):
            profile_urls.append(request_func_input.api_url)
            return RequestFuncOutput(
                success=True,
                prompt_len=request_func_input.prompt_len,
            )

        completion_urls.append(request_func_input.api_url)
        output_len = request_func_input.output_len
        return RequestFuncOutput(
            success=True,
            prompt_len=request_func_input.prompt_len,
            output_tokens=output_len,
            generated_text="x" * output_len,
            ttft=0.01,
            itl=[0.01] * max(output_len - 1, 0),
            latency=0.01 * max(output_len, 1),
            start_time=time.perf_counter(),
        )

    async def fake_wait_for_endpoint(
        request_func,
        test_input,
        session,
        timeout_seconds=600,
        retry_interval=5,
    ):
        ready_check_urls.append(test_input.api_url)
        return RequestFuncOutput(
            success=True,
            prompt_len=test_input.prompt_len,
            output_tokens=test_input.output_len,
            ttft=0.01,
            latency=0.02,
            start_time=time.perf_counter(),
        )

    spec_metric_snapshots = {
        short_base_url: [
            serve_module.SpecDecodeMetrics(
                num_drafts=1,
                num_draft_tokens=10,
                num_accepted_tokens=5,
                accepted_per_pos={0: 2, 1: 1},
            ),
            serve_module.SpecDecodeMetrics(
                num_drafts=3,
                num_draft_tokens=16,
                num_accepted_tokens=9,
                accepted_per_pos={0: 4, 1: 2},
            ),
        ],
        long_base_url: [
            serve_module.SpecDecodeMetrics(
                num_drafts=2,
                num_draft_tokens=8,
                num_accepted_tokens=3,
                accepted_per_pos={0: 1},
            ),
            serve_module.SpecDecodeMetrics(
                num_drafts=3,
                num_draft_tokens=12,
                num_accepted_tokens=5,
                accepted_per_pos={0: 2, 1: 1},
            ),
        ],
    }

    async def fake_fetch_spec_decode_metrics(base_url, session):
        spec_metric_calls.append(base_url)
        return spec_metric_snapshots[base_url].pop(0)

    monkeypatch.setitem(serve_module.ASYNC_REQUEST_FUNCS, "openai", fake_request_func)
    monkeypatch.setattr(serve_module, "wait_for_endpoint", fake_wait_for_endpoint)
    monkeypatch.setattr(
        serve_module,
        "fetch_spec_decode_metrics",
        fake_fetch_spec_decode_metrics,
    )

    input_requests = [
        SampleRequest(prompt=[1] * 99, prompt_len=99, expected_output_len=2),
        SampleRequest(prompt=[2] * 100, prompt_len=100, expected_output_len=2),
        SampleRequest(prompt=[3] * 101, prompt_len=101, expected_output_len=2),
    ]
    routing_config = serve_module.PromptLenRoutingConfig(
        threshold_prompt_len=100,
        short=serve_module.RouteTarget(
            base_url=short_base_url,
            api_url=short_api_url,
        ),
        long=serve_module.RouteTarget(
            base_url=long_base_url,
            api_url=long_api_url,
        ),
    )

    result = await serve_module.benchmark(
        task_type=serve_module.TaskType.GENERATION,
        endpoint_type="openai",
        api_url="http://unused/v1/completions",
        base_url="http://unused",
        model_id="test-model",
        model_name="test-model",
        tokenizer=None,
        input_requests=input_requests,
        logprobs=None,
        request_rate=float("inf"),
        burstiness=1.0,
        disable_tqdm=True,
        num_warmups=1,
        profile=True,
        selected_percentile_metrics=["ttft", "tpot", "itl", "e2el"],
        selected_percentiles=[99.0],
        ignore_eos=False,
        goodput_config_dict={},
        max_concurrency=None,
        lora_modules=None,
        extra_headers=None,
        extra_body=None,
        ready_check_timeout_sec=1,
        routing_config=routing_config,
    )

    assert ready_check_urls == [short_api_url, long_api_url]
    assert completion_urls.count(short_api_url) == 2
    assert completion_urls.count(long_api_url) == 3
    assert profile_urls == [
        f"{short_base_url}/start_profile",
        f"{long_base_url}/start_profile",
        f"{short_base_url}/stop_profile",
        f"{long_base_url}/stop_profile",
    ]
    assert spec_metric_calls == [
        short_base_url,
        long_base_url,
        short_base_url,
        long_base_url,
    ]

    assert result["completed"] == 3
    assert result["failed"] == 0
    assert result["routing_enabled"] is True
    assert result["routing_threshold_prompt_len"] == 100
    assert result["routing_request_counts"] == {"short": 1, "long": 2}
    assert result["request_routes"] == ["short", "long", "long"]
    assert result["routing_base_url_short"] == short_base_url
    assert result["routing_base_url_long"] == long_base_url
    assert result["spec_decode_num_drafts"] == 3
    assert result["spec_decode_draft_tokens"] == 10
    assert result["spec_decode_accepted_tokens"] == 6
    assert result["spec_decode_acceptance_rate"] == pytest.approx(60.0)
    assert result["spec_decode_acceptance_length"] == pytest.approx(3.0)
    assert result["spec_decode_per_position_acceptance_rates"] == pytest.approx(
        [1.0, 2 / 3]
    )
