#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the two-node KIMI issue-1% DeepGEMM shape diagnostic.

This is the eager/debug variant of manual_kimi_2node_issue01_rate5.py. It keeps
that launcher's model, dataset, DP2/TP8/DCP8 topology, max-num-seqs=1024,
memory utilization, request rate, and warmup settings, while setting
max-model-len=490000. Measured requests are submitted with Poisson arrivals for
approximately one minute (rate=5 x duration=60 => 300 requests), then the
harness waits for every in-flight request to finish.

CUDA graphs are disabled with --enforce-eager. The batched DeepGEMM MoE backend
is forced and every real gate/up GEMM call is logged with the marker
``[VLLM_MOE_GEMM_INPUT]``. Unlimited logging intentionally causes device
synchronization, large logs, and lower performance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import manual_kimi_2node_issue01_rate5 as base_runner


DEFAULT_ARGS = (
    "--bench-duration-sec",
    "60",
    "--max-model-len",
    "490000",
    "--run-label",
    "issue01-random-kimi-2node-dp2tp8dcp8-rate5-deepgemm-eager-1min",
)

DEEPGEMM_DEBUG_ENV = {
    "VLLM_USE_DEEP_GEMM": "1",
    "VLLM_MOE_USE_DEEP_GEMM": "1",
    "VLLM_MOE_GEMM_DEBUG": "1",
    "VLLM_MOE_GEMM_DEBUG_SKIP_DUMMY": "1",
    "VLLM_MOE_GEMM_DEBUG_DECODE_ONLY": "0",
    "VLLM_MOE_GEMM_DEBUG_EXPECTED_M": "0",
    "VLLM_MOE_GEMM_DEBUG_SKIP_CALLS": "0",
    "VLLM_MOE_GEMM_DEBUG_MAX_CALLS": "0",
}

_base_build_harness_argv = base_runner.build_harness_argv


def build_eager_harness_argv(
    args: argparse.Namespace,
    *,
    role: str,
    node_rank: int,
    master_addr: str,
    output_dir: Path,
    max_requests: int,
) -> list[str]:
    argv = _base_build_harness_argv(
        args,
        role=role,
        node_rank=node_rank,
        master_addr=master_addr,
        output_dir=output_dir,
        max_requests=max_requests,
    )
    argv.append("--enforce-eager")
    return argv


def main(argv: list[str] | None = None) -> int:
    user_args = sys.argv[1:] if argv is None else argv
    base_runner.DEFAULT_ENV.update(DEEPGEMM_DEBUG_ENV)
    base_runner.build_harness_argv = build_eager_harness_argv
    return base_runner.main([*DEFAULT_ARGS, *user_args])


if __name__ == "__main__":
    raise SystemExit(main())
