# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import vllm.v1.engine.core_client as core_client_mod
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm import SamplingParams
from vllm.pooling_params import LateInteractionParams, PoolingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import DPEngineCoreProc
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.pool.late_interaction import (
    LATE_INTERACTION_MODE_CACHE_QUERY,
    LATE_INTERACTION_MODE_SCORE_DOC,
)


def _make_request(request_id: str = "request-0") -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _make_pooling_request(
    request_id: str,
    *,
    mode: str,
    query_key: str,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=None,
        pooling_params=PoolingParams(
            task="token_embed",
            late_interaction_params=LateInteractionParams(
                mode=mode,
                query_key=query_key,
            ),
        ),
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _make_fake_dplb_client(
    *,
    policy: str = "waiting_x4_plus_running",
    lb_engines: list[list[int]],
    eng_start_index: int = 0,
    client_count: int = 1,
    block_size: int = 16,
    decode_context_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
) -> DPLBAsyncMPClient:
    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = client_count
    client.reqs_in_flight = {}
    client.core_engines = [
        index.to_bytes(2, "little") for index in range(len(lb_engines))
    ]
    client.lb_engines = [stats.copy() for stats in lb_engines]
    client.eng_start_index = eng_start_index
    client.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        parallel_config=SimpleNamespace(
            data_parallel_dispatch_policy=policy,
            decode_context_parallel_size=decode_context_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size,
        )
    )
    return client


def test_dplb_default_policy_matches_waiting_x4_plus_running():
    client = _make_fake_dplb_client(
        lb_engines=[[2, 1, 11, 100], [0, 0, 0, 80], [1, 0, 7, 90]]
    )

    chosen_engine = client.get_core_engine_for_request(_make_request())

    assert chosen_engine == client.core_engines[1]
    assert client.lb_engines[1][0] == 1


def test_dplb_least_batch_prefers_smallest_waiting_plus_running():
    client = _make_fake_dplb_client(
        policy="least_batch",
        lb_engines=[[1, 2, 30, 90], [3, 1, 40, 120], [0, 1, 10, 60]],
    )

    chosen_engine = client.get_core_engine_for_request(_make_request())

    assert chosen_engine == client.core_engines[2]
    assert client.lb_engines[2][0] == 1


def test_dplb_least_cache_prefers_smallest_waiting_tokens_minus_free_tokens():
    client = _make_fake_dplb_client(
        policy="least_cache",
        lb_engines=[[1, 0, 50, 4], [0, 1, 20, 5], [2, 1, 35, 3]],
        block_size=10,
    )

    chosen_engine = client.get_core_engine_for_request(_make_request())

    assert chosen_engine == client.core_engines[1]
    assert client.lb_engines[1][0] == 1
    assert client.lb_engines[1][2] == 23


def test_dplb_tie_break_respects_eng_start_index():
    client = _make_fake_dplb_client(
        policy="least_cache",
        lb_engines=[[0, 0, 20, 2], [0, 0, 20, 2], [0, 0, 20, 2]],
        eng_start_index=2,
        block_size=10,
    )

    chosen_engine = client.get_core_engine_for_request(_make_request())

    assert chosen_engine == client.core_engines[2]
    assert client.lb_engines[2][0] == 1


def test_dplb_late_interaction_routing_remains_sticky():
    client = _make_fake_dplb_client(
        lb_engines=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    query_key = "rerank-abc-query-0"

    query_engine = client.get_core_engine_for_request(
        _make_pooling_request(
            "query-req",
            mode=LATE_INTERACTION_MODE_CACHE_QUERY,
            query_key=query_key,
        )
    )
    doc_engine = client.get_core_engine_for_request(
        _make_pooling_request(
            "doc-req",
            mode=LATE_INTERACTION_MODE_SCORE_DOC,
            query_key=query_key,
        )
    )

    assert query_engine == doc_engine
    assert client.reqs_in_flight["query-req"] == query_engine
    assert client.reqs_in_flight["doc-req"] == doc_engine


def test_dp_engine_core_publishes_lb_stats_when_free_kv_blocks_change():
    engine_core = object.__new__(DPEngineCoreProc)
    engine_core.publish_dp_lb_stats = True
    engine_core.scheduler = MagicMock()
    engine_core.scheduler.get_request_counts.return_value = (3, 2)
    engine_core.scheduler.waiting_total_tokens = 11
    engine_core.scheduler.get_num_free_kv_blocks.side_effect = [100, 90, 90]
    engine_core.last_lb_snapshot = (0, 0, 0, 0)
    engine_core.step_counter = 7
    engine_core.current_wave = 4
    engine_core.output_queue = MagicMock()

    engine_core._maybe_publish_lb_stats()
    first_stats = (
        engine_core.output_queue.put_nowait.call_args.args[0][1].scheduler_stats
    )
    assert first_stats is not None
    assert first_stats.num_running_reqs == 3
    assert first_stats.num_waiting_reqs == 2
    assert first_stats.waiting_total_tokens == 11
    assert first_stats.free_kv_blocks == 100

    engine_core.output_queue.put_nowait.reset_mock()
    engine_core._maybe_publish_lb_stats()
    second_stats = (
        engine_core.output_queue.put_nowait.call_args.args[0][1].scheduler_stats
    )
    assert second_stats is not None
    assert second_stats.free_kv_blocks == 90

    engine_core.output_queue.put_nowait.reset_mock()
    engine_core._maybe_publish_lb_stats()
    engine_core.output_queue.put_nowait.assert_not_called()


def test_dp_engine_core_publishes_lb_stats_when_waiting_total_tokens_change():
    engine_core = object.__new__(DPEngineCoreProc)
    engine_core.publish_dp_lb_stats = True
    engine_core.scheduler = MagicMock()
    engine_core.scheduler.get_request_counts.return_value = (3, 2)
    engine_core.scheduler.waiting_total_tokens = 11
    engine_core.scheduler.get_num_free_kv_blocks.return_value = 100
    engine_core.last_lb_snapshot = (0, 0, 0, 0)
    engine_core.step_counter = 7
    engine_core.current_wave = 4
    engine_core.output_queue = MagicMock()

    engine_core._maybe_publish_lb_stats()
    first_stats = (
        engine_core.output_queue.put_nowait.call_args.args[0][1].scheduler_stats
    )
    assert first_stats is not None
    assert first_stats.waiting_total_tokens == 11

    engine_core.output_queue.put_nowait.reset_mock()
    engine_core.scheduler.waiting_total_tokens = 17
    engine_core._maybe_publish_lb_stats()
    second_stats = (
        engine_core.output_queue.put_nowait.call_args.args[0][1].scheduler_stats
    )
    assert second_stats is not None
    assert second_stats.waiting_total_tokens == 17

    engine_core.output_queue.put_nowait.reset_mock()
    engine_core._maybe_publish_lb_stats()
    engine_core.output_queue.put_nowait.assert_not_called()


def test_dplb_logs_dispatch_decision(monkeypatch):
    debug_mock = MagicMock()
    monkeypatch.setattr(core_client_mod.logger, "debug", debug_mock)
    monkeypatch.setattr(core_client_mod.logger, "isEnabledFor", lambda level: True)

    client = _make_fake_dplb_client(
        policy="least_cache",
        lb_engines=[[1, 0, 50, 4], [0, 1, 20, 5], [2, 1, 35, 3]],
        eng_start_index=1,
        block_size=10,
    )

    chosen_engine = client.get_core_engine_for_request(_make_request("request-debug"))

    assert chosen_engine == client.core_engines[1]
    debug_mock.assert_called_once_with(
        "DPLB dispatch policy=%s request_id=%s eng_start_index=%d "
        "lb_stats=%s chosen_engine_index=%d chosen_stats=%s "
        "chosen_sort_key=%s",
        "least_cache",
        "request-debug",
        1,
        [[1, 0, 50, 4], [0, 1, 20, 5], [2, 1, 35, 3]],
        1,
        [0, 1, 20, 5],
        (-30,),
    )


def test_dplb_init_logs_dispatch_policy_at_info(monkeypatch):
    info_mock = MagicMock()

    def fake_dp_async_init(
        self,
        vllm_config,
        executor_class,
        log_stats,
        client_addresses=None,
        client_count=1,
        client_index=0,
    ):
        self.vllm_config = vllm_config
        self.client_index = client_index
        self.core_engines = [
            index.to_bytes(2, "little") for index in range(4)
        ]

    monkeypatch.setattr(core_client_mod.DPAsyncMPClient, "__init__",
                        fake_dp_async_init)
    monkeypatch.setattr(core_client_mod.logger, "info", info_mock)

    client = core_client_mod.DPLBAsyncMPClient(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_dispatch_policy="least_batch",
            )),
        executor_class=object,
        log_stats=False,
        client_count=2,
        client_index=1,
    )

    assert client.eng_start_index == 2
    info_mock.assert_called_once_with(
        "Initialized internal DPLB dispatch_policy=%s client_index=%d "
        "client_count=%d managed_engines=%d eng_start_index=%d",
        "least_batch",
        1,
        2,
        4,
        2,
    )
