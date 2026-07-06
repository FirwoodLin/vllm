#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-/vllm}

export REPO_ROOT
export LAUNCH_DIR=${LAUNCH_DIR:-"${SCRIPT_DIR}"}
export LAUNCH_SCRIPT=${LAUNCH_SCRIPT:-run_deepseek_v3_1024k_dp_ep_16dp_tp1.sh}

export MODEL_PATH=${MODEL_PATH:-/mnt/nvme1n1/ml_research/models_4/deepseek-v3-1024k}
export DATASET_PATH=${DATASET_PATH:-/mnt/nvme1n1/ml_research/linbinbin1/vllm-dycp/dycp/dataset/sharegpt4o-random_geminiissue_r0.01_n60000_60k.json}
export NUM_PROMPTS=${NUM_PROMPTS:-3000}
export REQUEST_RATE=${REQUEST_RATE:-16}
export MAX_CONCURRENCY=${MAX_CONCURRENCY:-0}
export LOAD_FORMAT=${LOAD_FORMAT:-dummy}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024000}

export MAX_SEQS_PER_DP=${MAX_SEQS_PER_DP:-104}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-${MAX_SEQS_PER_DP}}
export CUDAGRAPH_MAX_CAPTURE_SIZE=${CUDAGRAPH_MAX_CAPTURE_SIZE:-${MAX_SEQS_PER_DP}}
export CUDAGRAPH_MODE=${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}

SCHEDULER_POLICY=${SCHEDULER_POLICY:-${VLLM_DP_DECODE_LB_POLICY:-iqr_lex_decode}}
case "${SCHEDULER_POLICY}" in
    iqr_lex_decode)
        SCHEDULER_NAME=${SCHEDULER_NAME:-iqr_lex_decode}
        ;;
    queue)
        SCHEDULER_NAME=${SCHEDULER_NAME:-vllm_default_queue}
        ;;
    *)
        echo "Unsupported SCHEDULER_POLICY=${SCHEDULER_POLICY}. Use iqr_lex_decode or queue." >&2
        exit 2
        ;;
esac
export VLLM_DP_DECODE_LB_POLICY="${SCHEDULER_POLICY}"

DATASET_FILE=$(basename -- "${DATASET_PATH}")
DATASET_NAME=${DATASET_NAME:-${DATASET_FILE%.json}}
PARALLEL_STRATEGY=${PARALLEL_STRATEGY:-dp16ep}
export STRATEGY_NAME=${STRATEGY_NAME:-${PARALLEL_STRATEGY}_${SCHEDULER_NAME}}
export PROFILE_MODE=${PROFILE_MODE:-deepseek_v3_1024k_${STRATEGY_NAME}}

RESULTS_ROOT=${RESULTS_ROOT:-"${SCRIPT_DIR}/results"}
REQUEST_RATE_DIR=${REQUEST_RATE_DIR:-${REQUEST_RATE}}
RUN_DIR=${RUN_DIR:-"${RESULTS_ROOT}/${DATASET_NAME}/${STRATEGY_NAME}/${REQUEST_RATE_DIR}"}

export STRATEGY_DIR=${STRATEGY_DIR:-"${RUN_DIR}"}
export RESULT_DIR=${RESULT_DIR:-"${RUN_DIR}"}
export LOG_DIR=${LOG_DIR:-"${RUN_DIR}"}
export CLEAN_BEFORE_START=${CLEAN_BEFORE_START:-1}
export RUNNER_NAME=${RUNNER_NAME:-$(basename "$0")}

mkdir -p "${RUN_DIR}"

exec "${REPO_ROOT}/dycp/examples/qwen-coder-480b-FP8/run_dycp_16dp_tp1_bench.sh" "$@"
