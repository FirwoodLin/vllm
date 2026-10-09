#!/usr/bin/env zsh

set -euo pipefail

cd /vllm

readonly input_root="/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/asplos-camera-august/vllm-profile-32x8k-16gpu/inputs"
readonly artifact_root="${KIMI_EAGER_GEMM_ARTIFACT_ROOT:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/asplos-camera-august/vllm-eager-gemm-debug-32x8k-16gpu/results}"

export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
export VLLM_DEEP_GEMM_WARMUP=skip
# Match the original vLLM profiler run. With DP16 this gives a physical
# post-dispatch DeepGEMM workspace M of 256 * 16 = 4096.
export VLLM_MOE_DP_CHUNK_SIZE="${KIMI_EAGER_GEMM_MOE_DP_CHUNK_SIZE:-256}"
export VLLM_MOE_GEMM_DEBUG="${VLLM_MOE_GEMM_DEBUG:-1}"
export VLLM_MOE_GEMM_DEBUG_SKIP_DUMMY="${VLLM_MOE_GEMM_DEBUG_SKIP_DUMMY:-1}"
export VLLM_MOE_GEMM_DEBUG_DECODE_ONLY="${VLLM_MOE_GEMM_DEBUG_DECODE_ONLY:-1}"
export VLLM_MOE_GEMM_DEBUG_EXPECTED_M="${VLLM_MOE_GEMM_DEBUG_EXPECTED_M:-16}"
# The torch profiler starts recording after delay_iterations=32. Skip the same
# number of real uniform-decode forwards, then print the first GEMM in that
# target window. Override MAX_CALLS when all 31 profiled iterations are needed.
export VLLM_MOE_GEMM_DEBUG_SKIP_CALLS="${VLLM_MOE_GEMM_DEBUG_SKIP_CALLS:-32}"
export VLLM_MOE_GEMM_DEBUG_MAX_CALLS="${VLLM_MOE_GEMM_DEBUG_MAX_CALLS:-1}"

eager_args=(
  --ignore-historical-skips
  --no-keep-going

  --frontend-extra-arg=--no-async-scheduling
  --frontend-extra-arg=--routing-mode
  --frontend-extra-arg=explicit_rank_replay
  --frontend-extra-arg=--load-format
  --frontend-extra-arg=dummy
  --frontend-extra-arg=--enforce-eager
  --frontend-extra-arg=--pause-before-measured

  --headless-extra-arg=--no-async-scheduling
  --headless-extra-arg=--load-format
  --headless-extra-arg=dummy
  --headless-extra-arg=--enforce-eager
)

run_case() {
  local case_csv="$1"
  local run_artifact_root="$2"
  local run_label="$3"

  python3 -c '
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "benchmarks"))
import manual_multinode_poisson_runner as runner

runner.DEFAULT_ENV_OVERRIDES.update(
    {
        "VLLM_MOE_GEMM_DEBUG": os.environ["VLLM_MOE_GEMM_DEBUG"],
        "VLLM_MOE_GEMM_DEBUG_SKIP_DUMMY": os.environ[
            "VLLM_MOE_GEMM_DEBUG_SKIP_DUMMY"
        ],
        "VLLM_MOE_GEMM_DEBUG_DECODE_ONLY": os.environ[
            "VLLM_MOE_GEMM_DEBUG_DECODE_ONLY"
        ],
        "VLLM_MOE_GEMM_DEBUG_EXPECTED_M": os.environ[
            "VLLM_MOE_GEMM_DEBUG_EXPECTED_M"
        ],
        "VLLM_MOE_GEMM_DEBUG_SKIP_CALLS": os.environ[
            "VLLM_MOE_GEMM_DEBUG_SKIP_CALLS"
        ],
        "VLLM_MOE_GEMM_DEBUG_MAX_CALLS": os.environ[
            "VLLM_MOE_GEMM_DEBUG_MAX_CALLS"
        ],
        "VLLM_MOE_DP_CHUNK_SIZE": os.environ["VLLM_MOE_DP_CHUNK_SIZE"],
    }
)

runner.CLUSTERS["2node_h200_0_2"] = runner.ClusterSpec(
    # The case CSV still uses this historical key. This reproducer is now
    # launched from h200-rjob1 and uses h200-rjob2 as its remote worker.
    master_addr=os.environ.get("KIMI_EAGER_GEMM_MASTER_ADDR", "10.102.215.76"),
    master_port=29579,
    remote_hosts=("h200-rjob2",),
    local_env_script="/root/.zshrc",
    remote_env_script="/root/.zshrc",
)

runner.STRATEGIES["dp16ep16"] = runner.StrategySpec(
    data_parallel_size=16,
    data_parallel_size_local=8,
    tensor_parallel_size=1,
    decode_context_parallel_size=1,
    data_parallel_backend="mp",
    enable_expert_parallel=True,
    attention_backend="FLASHMLA",
    all2all_backend="deepep_low_latency",
)

runner.main()
' \
    --artifact-root "${run_artifact_root}" \
    --case-csv "${case_csv}" \
    --run-label "${run_label}" \
    "${eager_args[@]}"
}

# This reproducer intentionally runs only the requested DP16 + EP16 topology.
run_case \
  "${input_root}/dp16_ep16/custom_lens.casecsv" \
  "${artifact_root}/dp16_ep16" \
  eager_gemm_debug_32x8k_dp16_ep16
