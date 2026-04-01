# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import SamplingParams
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreOutput,
    EngineCoreRequest,
)
from vllm.v1.engine.output_processor import (
    OutputProcessor,
    logger as output_processor_logger,
)
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.ttft_timing import RequestTTFTTrace


def test_emit_ttft_timing_log(dummy_test_vectors, monkeypatch, caplog):
    monkeypatch.setattr("vllm.v1.metrics.stats.time.time", lambda: 10.0)

    output_processor = OutputProcessor(
        dummy_test_vectors.tokenizer,
        log_stats=False,
        enable_ttft_timing_details=True,
        connector_name="DecodeBenchConnector",
    )
    request = EngineCoreRequest(
        request_id="request-0-int",
        external_req_id="request-0-ext",
        prompt_token_ids=dummy_test_vectors.prompt_tokens[0],
        mm_features=None,
        arrival_time=9.9,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        ttft_trace=RequestTTFTTrace(api_preprocess_ns=7_000_000),
    )
    output_processor.add_request(request, None)

    engine_core_output = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[dummy_test_vectors.generation_tokens[0][0]],
        events=[
            EngineCoreEvent(EngineCoreEventType.QUEUED, 1.0),
            EngineCoreEvent(EngineCoreEventType.SCHEDULED, 1.02),
        ],
        ttft_trace_update=RequestTTFTTrace(
            ipc_in_decode_ns=2_000_000,
            engine_preprocess_ns=3_000_000,
            first_batch_load_kv_ns=50_000_000,
        ),
    )

    with caplog.at_level("INFO", logger=output_processor_logger.name):
        output_processor.process_outputs(
            [engine_core_output],
            engine_core_timestamp=3.0,
            iteration_stats=IterationStats(),
        )

    assert "TTFT_TIMING" in caplog.text
    assert "request_id=request-0-ext" in caplog.text
    assert "internal_request_id=request-0-int" in caplog.text
    assert "connector=DecodeBenchConnector" in caplog.text
    assert "api_preprocess_ms=7.00" in caplog.text
    assert "ipc_in_decode_ms=2.00" in caplog.text
    assert "engine_preprocess_ms=3.00" in caplog.text
    assert "queue_wait_ms=20.00" in caplog.text
    assert "first_batch_load_kv_ms=50.00" in caplog.text
    assert "unattributed_ms=18.00" in caplog.text
    assert "server_ttft_ms=100.00" in caplog.text
