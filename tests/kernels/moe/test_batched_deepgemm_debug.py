# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import torch

import vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe as deep_gemm_moe
from vllm import envs


def test_gate_up_gemm_debug_filter_and_limit(monkeypatch) -> None:
    monkeypatch.setattr(envs, "VLLM_MOE_GEMM_DEBUG", True)
    monkeypatch.setattr(envs, "VLLM_MOE_GEMM_DEBUG_SKIP_DUMMY", False)
    monkeypatch.setattr(envs, "VLLM_MOE_GEMM_DEBUG_DECODE_ONLY", False)
    monkeypatch.setattr(envs, "VLLM_MOE_GEMM_DEBUG_EXPECTED_M", 16)
    monkeypatch.setattr(envs, "VLLM_MOE_GEMM_DEBUG_SKIP_CALLS", 0)
    monkeypatch.setattr(envs, "VLLM_MOE_GEMM_DEBUG_MAX_CALLS", 1)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        deep_gemm_moe.logger,
        "info",
        lambda _format, marker, payload: messages.append((marker, payload)),
    )

    experts = SimpleNamespace(
        _gemm_debug_call_count=0,
        moe_config=SimpleNamespace(layer_name="model.layers.37.mlp.experts"),
    )
    input_tensor = torch.empty((2, 4, 8))
    input_scale = torch.empty_strided((2, 4, 1), (4, 1, 4))
    weight = torch.empty((2, 6, 8))
    weight_scale = torch.empty((2, 1, 1))
    output = torch.empty((2, 4, 6))
    masked_m = torch.tensor([3, 1], dtype=torch.int32)

    log_inputs = deep_gemm_moe.BatchedDeepGemmExperts._maybe_log_gate_up_gemm_inputs
    log_inputs(
        experts,
        (input_tensor, input_scale),
        (weight, weight_scale),
        output,
        masked_m,
        32,
    )
    log_inputs(
        experts,
        (input_tensor, input_scale),
        (weight, weight_scale),
        output,
        masked_m,
        16,
    )
    log_inputs(
        experts,
        (input_tensor, input_scale),
        (weight, weight_scale),
        output,
        masked_m,
        16,
    )

    assert len(messages) == 1
    marker, serialized_payload = messages[0]
    assert marker == deep_gemm_moe.MOE_GEMM_DEBUG_LOG_MARKER

    payload = json.loads(serialized_payload)
    assert payload["event"] == "post_dispatch_pre_gemm"
    assert payload["gemm"] == "gate_up"
    assert payload["layer_index"] == 37
    assert payload["expected_m"] == 16
    assert payload["masked_m"] == [3, 1]
    assert payload["masked_m_sum"] == 4
    assert payload["input"]["shape"] == [2, 4, 8]
    assert payload["input_scale"]["stride"] == [4, 1, 4]
    assert payload["weight"]["shape"] == [2, 6, 8]
    assert payload["output"]["shape"] == [2, 4, 6]
    assert experts._gemm_debug_call_count == 2


def test_gate_up_gemm_debug_only_logs_real_decode(monkeypatch) -> None:
    monkeypatch.setattr(envs, "VLLM_MOE_GEMM_DEBUG", True)
    monkeypatch.setattr(envs, "VLLM_MOE_GEMM_DEBUG_SKIP_DUMMY", True)
    monkeypatch.setattr(envs, "VLLM_MOE_GEMM_DEBUG_DECODE_ONLY", True)
    monkeypatch.setattr(envs, "VLLM_MOE_GEMM_DEBUG_EXPECTED_M", 16)
    monkeypatch.setattr(envs, "VLLM_MOE_GEMM_DEBUG_SKIP_CALLS", 0)
    monkeypatch.setattr(envs, "VLLM_MOE_GEMM_DEBUG_MAX_CALLS", 1)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        deep_gemm_moe.logger,
        "info",
        lambda _format, marker, payload: messages.append((marker, payload)),
    )

    forward_context = SimpleNamespace(
        is_dummy_run=True,
        is_uniform_decode=True,
    )
    monkeypatch.setattr(deep_gemm_moe, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(deep_gemm_moe, "get_forward_context", lambda: forward_context)

    experts = SimpleNamespace(
        _gemm_debug_call_count=0,
        moe_config=SimpleNamespace(layer_name="model.layers.1.mlp.experts"),
    )
    input_tensor = torch.empty((2, 4, 8))
    input_scale = torch.empty((2, 4, 1))
    weight = torch.empty((2, 6, 8))
    weight_scale = torch.empty((2, 1, 1))
    output = torch.empty((2, 4, 6))
    masked_m = torch.tensor([3, 1], dtype=torch.int32)
    log_inputs = deep_gemm_moe.BatchedDeepGemmExperts._maybe_log_gate_up_gemm_inputs

    # Dummy decode-shaped forwards and real prefill forwards are both ignored.
    log_inputs(
        experts,
        (input_tensor, input_scale),
        (weight, weight_scale),
        output,
        masked_m,
        16,
    )
    forward_context.is_dummy_run = False
    forward_context.is_uniform_decode = False
    log_inputs(
        experts,
        (input_tensor, input_scale),
        (weight, weight_scale),
        output,
        masked_m,
        16,
    )

    # The first scheduler-provided uniform decode forward is logged.
    forward_context.is_uniform_decode = True
    log_inputs(
        experts,
        (input_tensor, input_scale),
        (weight, weight_scale),
        output,
        masked_m,
        16,
    )

    assert len(messages) == 1
    assert experts._gemm_debug_call_count == 1
