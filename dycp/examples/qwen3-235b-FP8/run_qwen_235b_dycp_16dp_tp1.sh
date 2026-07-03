#!/bin/bash

# Decode-only DYCP launch for Qwen3-235B FP8 on 16 GPUs.
# Topology: 2 nodes x 8 GPUs, global DP=16, TP=1, DP per domain=8.

set -euo pipefail
set -x

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <node_rank> [master_ip]"
    echo "Alternatively set DATA_PARALLEL_ADDRESS=<master_ip>."
    echo "For node 0: bash $0 0 <node0_ip>"
    echo "For node 1: bash $0 1 <node0_ip>"
    exit 1
fi

NODE_RANK=$1
DATA_PARALLEL_ADDRESS=${2:-${DATA_PARALLEL_ADDRESS:-}}
if [ -z "${DATA_PARALLEL_ADDRESS}" ]; then
    echo "DATA_PARALLEL_ADDRESS is required. Pass [master_ip] or set it in env."
    exit 1
fi

if [ "${NODE_RANK}" -ne 0 ]; then
    EXTRA_PARAMS=(--headless)
else
    EXTRA_PARAMS=(--api-server-count 1)
fi

MODEL_PATH=${MODEL_PATH:-/mnt/nvme1n1/ml_research/models_cfs/qwen3-235B-Instruct-2507-FP8/}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-auto}
LOAD_FORMAT=${LOAD_FORMAT:-auto}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-524288}
PORT=${PORT:-8400}
DP_RPC_PORT=${DP_RPC_PORT:-$((PORT + 100))}
KV_PORT=${KV_PORT:-20002}
KV_PARALLEL_SIZE=${KV_PARALLEL_SIZE:-2}
KV_RANK=${KV_RANK:-1}
MAX_SEQS_PER_DP=${MAX_SEQS_PER_DP:-500}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-${MAX_SEQS_PER_DP}}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
LOG_DIR=${LOG_DIR:-.}
PROFILE_MODE=${PROFILE_MODE:-dycp16dp_tp1}
PROFILE_BASE_DIR=${PROFILE_BASE_DIR:-/mnt/nvme1n1/ml_research/linbinbin1/vllm-dycp/dycp/profiles}
if [ "${MAX_NUM_BATCHED_TOKENS}" -lt "${MAX_SEQS_PER_DP}" ]; then
    DEFAULT_CUDAGRAPH_MAX_CAPTURE_SIZE=${MAX_NUM_BATCHED_TOKENS}
else
    DEFAULT_CUDAGRAPH_MAX_CAPTURE_SIZE=${MAX_SEQS_PER_DP}
fi
CUDAGRAPH_MAX_CAPTURE_SIZE=${CUDAGRAPH_MAX_CAPTURE_SIZE:-${DEFAULT_CUDAGRAPH_MAX_CAPTURE_SIZE}}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-}
CUDAGRAPH_MODE=${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}
CUDAGRAPH_CAPTURE_SIZES_FOR_CP=${CUDAGRAPH_CAPTURE_SIZES_FOR_CP:-4}

if [ "${PORT}" -eq "${DP_RPC_PORT}" ]; then
    echo "PORT and DP_RPC_PORT must be different. Got ${PORT}." >&2
    exit 1
fi

export VLLM_USE_V1=1
export VLLM_VERSION=${VLLM_VERSION:-0.13.0}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-380}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}
export VLLM_MOE_DP_CHUNK_SIZE=${VLLM_MOE_DP_CHUNK_SIZE:-${MAX_SEQS_PER_DP}}
export VLLM_DEEPEP_BUFFER_SIZE_MB=${VLLM_DEEPEP_BUFFER_SIZE_MB:-0}
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-1}
export VLLM_ALL2ALL_BACKEND=${VLLM_ALL2ALL_BACKEND:-deepep_low_latency}
export VLLM_IGNORE_TENSOR_PLACEHOLDER=${VLLM_IGNORE_TENSOR_PLACEHOLDER:-1}
export VLLM_USE_FORCE_LOAD_BALANCE=${VLLM_USE_FORCE_LOAD_BALANCE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

export NVSHMEM_HCA_LIST=${NVSHMEM_HCA_LIST:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}
export NVSHMEM_IB_GID_INDEX=${NVSHMEM_IB_GID_INDEX:-3}
export NVSHMEM_IBGDA_NUM_RC_PER_PE=${NVSHMEM_IBGDA_NUM_RC_PER_PE:-8}
export NVSHMEM_IB_TRAFFIC_CLASS=${NVSHMEM_IB_TRAFFIC_CLASS:-186}
export NVSHMEM_DISABLE_NVLS=${NVSHMEM_DISABLE_NVLS:-1}

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_TC=${NCCL_IB_TC:-186}

if [ "${ENABLE_TORCH_PROFILE:-0}" = "1" ]; then
    PROFILE_TAG=${PROFILE_TAG:-$(date +%Y%m%d-%H%M%S)}
    export VLLM_TORCH_PROFILER_DIR=${VLLM_TORCH_PROFILER_DIR:-"${PROFILE_BASE_DIR}/${PROFILE_TAG}/${PROFILE_MODE}_node${NODE_RANK}"}
    mkdir -p "${VLLM_TORCH_PROFILER_DIR}"
    export VLLM_TORCH_PROFILER_WITH_STACK=${VLLM_TORCH_PROFILER_WITH_STACK:-0}
    export VLLM_TORCH_PROFILER_RECORD_SHAPES=${VLLM_TORCH_PROFILER_RECORD_SHAPES:-0}
    export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=${VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY:-0}
    export VLLM_TORCH_PROFILER_WITH_FLOPS=${VLLM_TORCH_PROFILER_WITH_FLOPS:-0}
    export VLLM_TORCH_PROFILER_USE_GZIP=${VLLM_TORCH_PROFILER_USE_GZIP:-1}
    export VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL=${VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL:-1}
    export VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM=${VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM:-0}
    export VLLM_PROFILER_DELAY_ITERS=${VLLM_PROFILER_DELAY_ITERS:-0}
    export VLLM_PROFILER_MAX_ITERS=${VLLM_PROFILER_MAX_ITERS:-0}
    echo "Torch profiler traces will be saved to ${VLLM_TORCH_PROFILER_DIR}"
fi

ulimit -n 1048576

COMMON_ARGS=(
    --trust-remote-code
    --served-model-name "${SERVED_MODEL_NAME}"
    --load-format "${LOAD_FORMAT}"
    --disable-log-requests
)

DEFAULT_CUDAGRAPH_CAPTURE_SIZES=(
    2 4 8 10 12 16 18 24 26 32 34 40 48 56 64 72 80 88 96 104
    112 120 128 136 144 152 160 176 192 208 224 240 256 288 320
    352 384 448 496 500
)

join_cudagraph_capture_sizes() {
    local max_size=$1
    local selected=()
    local size

    for size in "${DEFAULT_CUDAGRAPH_CAPTURE_SIZES[@]}"; do
        if [ -n "${max_size}" ] && [ "${size}" -gt "${max_size}" ]; then
            continue
        fi
        selected+=("${size}")
    done

    if [ "${#selected[@]}" -eq 0 ] && [ -n "${max_size}" ]; then
        selected=("${max_size}")
    fi

    local IFS=,
    echo "${selected[*]}"
}

if [ -z "${CUDAGRAPH_CAPTURE_SIZES}" ]; then
    CUDAGRAPH_CAPTURE_SIZES=$(join_cudagraph_capture_sizes "${CUDAGRAPH_MAX_CAPTURE_SIZE}")
fi

if [[ "${CUDAGRAPH_CAPTURE_SIZES}" == \[* ]]; then
    CUDAGRAPH_CAPTURE_SIZES_JSON=${CUDAGRAPH_CAPTURE_SIZES}
else
    CUDAGRAPH_CAPTURE_SIZES_JSON="[${CUDAGRAPH_CAPTURE_SIZES}]"
fi

COMPILATION_CONFIG=${COMPILATION_CONFIG:-"{\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES_JSON},\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes_for_cp\":${CUDAGRAPH_CAPTURE_SIZES_FOR_CP}}"}

KV_TRANSFER_CONFIG=$(cat <<JSON
{
    "kv_connector": "CrossDPExampleConnector",
    "kv_connector_module_path": "vllm.distributed.kv_transfer.kv_connector.v1.cross_dp_example_connector",
    "kv_role": "kv_consumer",
    "kv_parallel_size": ${KV_PARALLEL_SIZE},
    "kv_port": "${KV_PORT}",
    "engine_id": "decode-${NODE_RANK}",
    "kv_rank": ${KV_RANK},
    "kv_connector_extra_config": {
        "prefill": {
            "dp_size": 1,
            "tp_size": 16
        },
        "decode": {
            "dp_size": 8,
            "tp_size": 1
        }
    }
}
JSON
)

args=(
    --port "${PORT}"
    "${EXTRA_PARAMS[@]}"
    "${COMMON_ARGS[@]}"
    --async-scheduling
    --distributed-executor-backend dmp
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":262144}}'
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --no-enable-prefix-caching
    --data-parallel-size 16
    --tensor-parallel-size 1
    --data-parallel-size-local 8
    --data-parallel-address "${DATA_PARALLEL_ADDRESS}"
    --data-parallel-rpc-port "${DP_RPC_PORT}"
    --data-parallel-start-rank $((NODE_RANK * 8))
    --block-size 64
    --cp-kv-cache-interleave-size 64
    --no-enforce-eager
    --max-num-seqs "${MAX_SEQS_PER_DP}"
    --enable-expert-parallel
    --dp-per-domain 8
    --num-cp-seqs 4
    --compilation-config "${COMPILATION_CONFIG}"
    --kv-transfer-config "${KV_TRANSFER_CONFIG}"
)

mkdir -p "${LOG_DIR}"
vllm serve "${MODEL_PATH}" "${args[@]}" &> "${LOG_DIR}/qwen235b_dycp_16dp_tp1_node${NODE_RANK}.log" &
