# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from types import SimpleNamespace

from vllm.v1.engine.core import BatchTimingTicket, EngineCore, logger as core_logger
from vllm.v1.graph_timing import GraphReplayTimingStats, GraphTimingContext
from vllm.v1.outputs import ModelRunnerOutput


def test_graph_timing_context_resolve_includes_seqlen():
    context = GraphTimingContext(
        reply_global_rank=0,
        dp_rank=0,
        tp_rank=0,
        dcp_rank=0,
        node_rank=0,
        runtime_mode="FULL",
        event_pool_size=0,
    )

    context.update_seqlen(1024)
    context.update_seqlen(2048)

    stats = context.resolve()

    assert stats.seqlen == 2048


def test_emit_batch_timing_log_includes_seqlen(monkeypatch, caplog):
    engine_core = EngineCore.__new__(EngineCore)
    engine_core.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(
            enable_logging_step_timing_details=True,
            logging_step_timing_interval=1,
        )
    )

    monkeypatch.setattr(
        "vllm.v1.engine.core.compute_iteration_details",
        lambda _: SimpleNamespace(
            num_ctx_requests=1,
            num_ctx_tokens=8,
            num_generation_requests=1,
            num_generation_tokens=2,
        ),
    )

    ticket = BatchTimingTicket(
        batch_id=0,
        scheduler_output=SimpleNamespace(),
        exec_model_future=Future(),
    )
    model_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        graph_replay_timing_stats=GraphReplayTimingStats(
            reply_global_rank=0,
            dp_rank=0,
            tp_rank=0,
            dcp_rank=0,
            node_rank=0,
            replay_count=1,
            replay_gpu_ms=1.5,
            replay_wall_ms=2.5,
            runtime_mode="FULL",
            graph_impl="cudagraph_wrapper",
            seqlen=4096,
        ),
    )

    with caplog.at_level("INFO", logger=core_logger.name):
        engine_core._emit_batch_timing_log(ticket, model_output)

    assert "seqlen=4096" in caplog.text
