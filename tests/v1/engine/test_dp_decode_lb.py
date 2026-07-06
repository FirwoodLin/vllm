# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import msgspec

from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine import core_client as core_client_module
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.metrics.stats import DPDecodeLBStats


class RecordingLogger:

    def __init__(self):
        self.debug_calls = []
        self.info_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))

    def info(self, *args, **kwargs):
        self.info_calls.append((args, kwargs))


def make_request(
    request_id: str,
    prompt_len: int,
    data_parallel_rank: int | None = None,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        eos_token_id=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=data_parallel_rank,
    )


def make_client(
    decode_lb_stats: list[DPDecodeLBStats | None],
    lb_engines: list[list[int]] | None = None,
    client_count: int = 1,
) -> DPLBAsyncMPClient:
    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = client_count
    client.reqs_in_flight = {}
    client.core_engines = [
        f"engine-{idx}".encode() for idx in range(len(decode_lb_stats))
    ]
    client.lb_engines = (
        lb_engines
        if lb_engines is not None
        else [
            [stats.num_waiting_reqs, stats.num_running_reqs]
            if stats is not None
            else [0, 0]
            for stats in decode_lb_stats
        ]
    )
    client.decode_lb_stats = decode_lb_stats
    client.eng_start_index = 0
    client.dp_lb_iqr_k = 1.5
    client.dp_lb_min_safe_ranks = 1
    client.dp_lb_block_size = 16
    client.dp_decode_lb_policy = "iqr_lex_decode"
    client._dp_lb_logged_iqr_lex_active = False
    client._dp_lb_logged_iqr_lex_assignment = False
    client._dp_lb_logged_queue_fallback = False
    client._dp_lb_logged_queue_fallback_assignment = False
    return client


def test_decode_lb_stats_payload_decodes_to_dataclass_instances():
    payload = msgspec.msgpack.encode(
        (
            [[0, 1], [2, 3]],
            4,
            True,
            [
                DPDecodeLBStats(1, 0, 0.25, 100, 75, 25),
                None,
            ],
        )
    )

    counts, wave, running, decode_lb_stats = (
        DPLBAsyncMPClient._decode_stats_update_payload(payload)
    )

    assert counts == [[0, 1], [2, 3]]
    assert wave == 4
    assert running is True
    assert isinstance(decode_lb_stats[0], DPDecodeLBStats)
    assert decode_lb_stats[0].num_allocated_blocks == 25
    assert decode_lb_stats[1] is None


def test_add_request_async_batches_concurrent_requests_in_one_routing_round():
    client = make_client(
        [
            DPDecodeLBStats(0, 0, 0.0, 100, 100, 0),
            DPDecodeLBStats(0, 0, 0.0, 100, 100, 0),
        ]
    )
    client.current_wave = 7
    client.client_index = 3
    client.engines_running = True
    client._ensure_stats_update_task = lambda: None
    client._ensure_output_queue_task = lambda: None
    batch_sizes = []
    sent = []

    def get_core_engines_for_requests(requests):
        batch_sizes.append(len(requests))
        return [b"engine-0", b"engine-1"]

    async def send_input(request_type, request, engine):
        sent.append((request_type, request.request_id, engine))

    client.get_core_engines_for_requests = get_core_engines_for_requests
    client._send_input = send_input

    req1 = make_request("req-1", 4)
    req2 = make_request("req-2", 8)

    async def run_concurrent_adds():
        await asyncio.gather(
            client.add_request_async(req1),
            client.add_request_async(req2),
        )

    asyncio.run(run_concurrent_adds())

    assert batch_sizes == [2]
    assert sent == [
        (EngineCoreRequestType.ADD, "req-1", b"engine-0"),
        (EngineCoreRequestType.ADD, "req-2", b"engine-1"),
    ]
    assert req1.current_wave == 7
    assert req2.current_wave == 7
    assert req1.client_index == 3
    assert req2.client_index == 3


def test_iqr_lex_batch_sorts_by_sequence_length_and_updates_virtual_state():
    client = make_client(
        [
            DPDecodeLBStats(
                num_running_reqs=0,
                num_waiting_reqs=0,
                kv_cache_usage=0.0,
                num_total_blocks=100,
                num_free_blocks=100,
                num_allocated_blocks=0,
            ),
            DPDecodeLBStats(
                num_running_reqs=0,
                num_waiting_reqs=0,
                kv_cache_usage=0.01,
                num_total_blocks=100,
                num_free_blocks=99,
                num_allocated_blocks=1,
            ),
        ]
    )

    short_req = make_request("short", prompt_len=1)
    long_req = make_request("long", prompt_len=33)

    assignments = client.get_core_engines_for_requests([short_req, long_req])

    assert assignments == [b"engine-1", b"engine-0"]
    assert client.decode_lb_stats[0].num_waiting_reqs == 1
    assert client.decode_lb_stats[0].num_allocated_blocks == 3
    assert client.decode_lb_stats[0].num_free_blocks == 97
    assert client.decode_lb_stats[1].num_waiting_reqs == 1
    assert client.decode_lb_stats[1].num_allocated_blocks == 2
    assert client.decode_lb_stats[1].num_free_blocks == 98


def test_iqr_mask_excludes_kv_outlier_before_lexicographic_selection():
    client = make_client(
        [
            DPDecodeLBStats(0, 1, 0.05, 200, 190, 10),
            DPDecodeLBStats(0, 1, 0.06, 200, 188, 12),
            DPDecodeLBStats(0, 1, 0.07, 200, 186, 14),
            DPDecodeLBStats(0, 0, 0.50, 200, 100, 100),
        ]
    )

    assignments = client.get_core_engines_for_requests(
        [make_request("masked", prompt_len=1)]
    )

    assert assignments == [b"engine-0"]


def test_iqr_mask_is_skipped_for_fewer_than_four_candidates():
    client = make_client(
        [
            DPDecodeLBStats(0, 1, 0.05, 200, 190, 10),
            DPDecodeLBStats(0, 1, 0.06, 200, 188, 12),
            DPDecodeLBStats(0, 0, 0.50, 200, 100, 100),
        ]
    )

    assignments = client.get_core_engines_for_requests(
        [make_request("small-candidate-set", prompt_len=1)]
    )

    assert assignments == [b"engine-2"]


def test_missing_decode_lb_stats_falls_back_to_queue_count_policy():
    client = make_client(
        [None, None],
        lb_engines=[[0, 2], [0, 1]],
        client_count=3,
    )

    assignment = client.get_core_engine_for_request(make_request("fallback", 1))

    assert assignment == b"engine-1"
    assert client.lb_engines == [[0, 2], [3, 1]]


def test_queue_policy_uses_queue_count_even_when_decode_lb_stats_exist():
    client = make_client(
        [
            DPDecodeLBStats(0, 0, 0.90, 200, 20, 180),
            DPDecodeLBStats(0, 2, 0.01, 200, 198, 2),
        ],
        lb_engines=[[0, 0], [10, 0]],
    )
    client.dp_decode_lb_policy = "queue"

    assignment = client.get_core_engine_for_request(make_request("queue-policy", 1))

    assert assignment == b"engine-0"
    assert client.lb_engines == [[1, 0], [10, 0]]
    assert client.decode_lb_stats[0].num_allocated_blocks == 180


def test_missing_decode_lb_stats_logs_queue_count_fallback_once():
    client = make_client(
        [None, None],
        lb_engines=[[0, 2], [0, 1]],
    )
    recorder = RecordingLogger()
    old_logger = core_client_module.logger
    core_client_module.logger = recorder
    try:
        client.get_core_engine_for_request(make_request("fallback-1", 1))
        client.get_core_engine_for_request(make_request("fallback-2", 1))
    finally:
        core_client_module.logger = old_logger

    fallback_logs = [
        args for args, _ in recorder.info_calls if "queue-count fallback" in args[0]
    ]
    assert len(fallback_logs) == 1


def test_iqr_lex_assignment_logs_decision_details():
    client = make_client(
        [
            DPDecodeLBStats(0, 1, 0.05, 200, 190, 10),
            DPDecodeLBStats(0, 1, 0.06, 200, 188, 12),
            DPDecodeLBStats(0, 1, 0.07, 200, 186, 14),
            DPDecodeLBStats(0, 0, 0.50, 200, 100, 100),
        ]
    )
    recorder = RecordingLogger()
    old_logger = core_client_module.logger
    core_client_module.logger = recorder
    try:
        client.get_core_engine_for_request(make_request("logged", 1))
    finally:
        core_client_module.logger = old_logger

    assert recorder.info_calls[0][0][0].startswith(
        "IQR-Lex decode DP LB active"
    )
    assert recorder.info_calls[1][0][0].startswith(
        "IQR-Lex decode DP LB first assignment"
    )
    assert "selected_engine=%d" in recorder.info_calls[1][0][0]
    debug_formats = [args[0] for args, _ in recorder.debug_calls]
    assert any("IQR-Lex decode DP LB assignment" in fmt for fmt in debug_formats)
    decision_args = next(
        args
        for args, _ in recorder.debug_calls
        if "IQR-Lex decode DP LB assignment" in args[0]
    )
    assert "selected_engine=%d" in decision_args[0]
    assert "iqr_threshold_blocks=%s" in decision_args[0]
    assert "virtual_post_b=%d" in decision_args[0]
    assert 0 in decision_args


def test_explicit_data_parallel_rank_bypasses_iqr_lex_policy():
    client = make_client(
        [
            DPDecodeLBStats(0, 0, 0.0, 100, 100, 0),
            DPDecodeLBStats(0, 0, 0.0, 100, 100, 0),
            DPDecodeLBStats(0, 0, 0.0, 100, 100, 0),
        ],
        lb_engines=[[0, 0], [0, 0], [0, 0]],
    )

    assignment = client.get_core_engine_for_request(
        make_request("explicit", 1, data_parallel_rank=2)
    )

    assert assignment == b"engine-2"
    assert client.lb_engines == [[0, 0], [0, 0], [0, 0]]
    assert client.decode_lb_stats[2].num_waiting_reqs == 0
