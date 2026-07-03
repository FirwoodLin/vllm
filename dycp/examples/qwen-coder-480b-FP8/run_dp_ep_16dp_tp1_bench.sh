#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export REPO_ROOT=${REPO_ROOT:-/vllm}
export LAUNCH_DIR=${LAUNCH_DIR:-"${SCRIPT_DIR}"}
export LAUNCH_SCRIPT=${LAUNCH_SCRIPT:-run_qwen_coder_480b_dp_ep_16dp_tp1.sh}
export STRATEGY_NAME=${STRATEGY_NAME:-dp16_tp1}
export PROFILE_MODE=${PROFILE_MODE:-qwen_coder_480b_dp16_tp1}
export CLEAN_BEFORE_START=${CLEAN_BEFORE_START:-1}
export DATASET_PATH=${DATASET_PATH:-"${REPO_ROOT}/dycp/dataset/pure_short_4k_input_output1024_4800.json"}
export NUM_PROMPTS=${NUM_PROMPTS:-4800}
export REQUEST_RATE=${REQUEST_RATE:-16}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-300000}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-144}
export MAX_SEQS_PER_DP=${MAX_SEQS_PER_DP:-${MAX_NUM_BATCHED_TOKENS}}
export RUNNER_NAME=${RUNNER_NAME:-$(basename "$0")}

exec "${SCRIPT_DIR}/run_dycp_16dp_tp1_bench.sh" "$@"
