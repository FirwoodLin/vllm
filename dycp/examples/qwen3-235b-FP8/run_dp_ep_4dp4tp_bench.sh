#!/usr/bin/env bash

set -euo pipefail

unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

REPO_ROOT=${REPO_ROOT:-/vllm}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=${EXAMPLE_DIR:-"${REPO_ROOT}/dycp/examples/qwen3-235b-FP8"}
LAUNCH_SCRIPT=${LAUNCH_SCRIPT:-run_qwen_235b_dp_ep_4dp4tp.sh}

REMOTE_HOST=${REMOTE_HOST:-h200-rjob2}
MASTER_IP=${MASTER_IP:-10.102.97.183}
LOCAL_NODE_RANK=${LOCAL_NODE_RANK:-0}
REMOTE_NODE_RANK=${REMOTE_NODE_RANK:-1}

PORT=${PORT:-8400}
DP_RPC_PORT=${DP_RPC_PORT:-$((PORT + 100))}
BENCH_HOST=${BENCH_HOST:-localhost}
MODEL_PATH=${MODEL_PATH:-/mnt/nvme1n1/ml_research/models_cfs/qwen3-235B-Instruct-2507-FP8/}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-auto}
export LOAD_FORMAT=${LOAD_FORMAT:-dummy}
DATASET_PATH=${DATASET_PATH:-"${REPO_ROOT}/dycp/dataset/trace_512k_4k_long1pct_10000-output1024.json"}
RESULT_DIR=${RESULT_DIR:-"${REPO_ROOT}/dycp/results/dp4tp4"}
NUM_PROMPTS=${NUM_PROMPTS:-4800}
REQUEST_RATE=${REQUEST_RATE:-16}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-0}
METRIC_PERCENTILES=${METRIC_PERCENTILES:-50,90,99}
ENDPOINT=${ENDPOINT:-/v1/completions}

KV_PORT=${KV_PORT:-20002}
KV_PARALLEL_SIZE=${KV_PARALLEL_SIZE:-2}
KV_RANK=${KV_RANK:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-524288}
MAX_SEQS_PER_DP=${MAX_SEQS_PER_DP:-500}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-${MAX_SEQS_PER_DP}}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
PROFILE_MODE=${PROFILE_MODE:-dp4tp4}
LOG_DIR=${LOG_DIR:-"${EXAMPLE_DIR}/${PROFILE_MODE}/logs"}
CUDAGRAPH_MAX_CAPTURE_SIZE=${CUDAGRAPH_MAX_CAPTURE_SIZE:-}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-}
CUDAGRAPH_MODE=${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}
CUDAGRAPH_CAPTURE_SIZES_FOR_CP=${CUDAGRAPH_CAPTURE_SIZES_FOR_CP:-}
COMPILATION_CONFIG=${COMPILATION_CONFIG:-}

WAIT_FOR_SERVER=${WAIT_FOR_SERVER:-1}
SERVER_READY_RETRIES=${SERVER_READY_RETRIES:-120}
SERVER_READY_INTERVAL=${SERVER_READY_INTERVAL:-10}
CLEAN_BEFORE_START=${CLEAN_BEFORE_START:-0}
RUN_GPU_CLEANUP=${RUN_GPU_CLEANUP:-1}
GPU_CLEANUP_HOSTS=${GPU_CLEANUP_HOSTS:-"dlh200-3 dlh200-2"}
GPU_CLEANUP_SCRIPT=${GPU_CLEANUP_SCRIPT:-/mnt/nvme1n1/ml_research/linbinbin1/scripts/kill_gpu.sh}
GPU_CLEANUP_SELECTION=${GPU_CLEANUP_SELECTION:-all}
RUNNER_NAME=${RUNNER_NAME:-$(basename "$0")}

usage() {
    cat <<USAGE
Usage: ${RUNNER_NAME} [run|start|bench|wait|cleanup]

Default command:
  run       Start local/remote servers, wait for readiness, run benchmark, then cleanup.

Other commands:
  start     Start only the local and remote servers.
  bench     Run only the benchmark.
  wait      Wait until http://${BENCH_HOST}:${PORT}/health is ready.
  cleanup   Kill vLLM server/benchmark processes, then run kill_gpu.sh on ${GPU_CLEANUP_HOSTS}.

Common overrides:
  MASTER_IP=10.102.97.183 REMOTE_HOST=h200-rjob2 PORT=8400 ${RUNNER_NAME}
  REQUEST_RATE=16 NUM_PROMPTS=4800 MAX_CONCURRENCY=0 ${RUNNER_NAME} bench
  LOAD_FORMAT=dummy ${RUNNER_NAME}
  LAUNCH_SCRIPT=${LAUNCH_SCRIPT} ${RUNNER_NAME}
  RUN_GPU_CLEANUP=0 ${RUNNER_NAME} cleanup
USAGE
}

activate_venv() {
    cd "${REPO_ROOT}"
    set +u
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
    set -u
}

export_launch_env() {
    export MODEL_PATH
    export SERVED_MODEL_NAME
    export LOAD_FORMAT
    export PORT
    export DP_RPC_PORT
    export KV_PORT
    export KV_PARALLEL_SIZE
    export KV_RANK
    export MAX_MODEL_LEN
    export MAX_SEQS_PER_DP
    export MAX_NUM_BATCHED_TOKENS
    export GPU_MEMORY_UTILIZATION
    export LOG_DIR
    export PROFILE_MODE
    export CUDAGRAPH_MAX_CAPTURE_SIZE
    export CUDAGRAPH_CAPTURE_SIZES
    export CUDAGRAPH_MODE
    export CUDAGRAPH_CAPTURE_SIZES_FOR_CP
    export COMPILATION_CONFIG
}

start_local_server() {
    echo "Starting local node ${LOCAL_NODE_RANK} with master ${MASTER_IP}..."
    activate_venv
    export_launch_env
    mkdir -p "${LOG_DIR}"
    cd "${EXAMPLE_DIR}"
    bash "${LAUNCH_SCRIPT}" "${LOCAL_NODE_RANK}" "${MASTER_IP}"
}

b64_env_value() {
    printf '%s' "$1" | base64 | tr -d '\n'
}

start_remote_server() {
    echo "Starting remote node ${REMOTE_NODE_RANK} on ${REMOTE_HOST} with master ${MASTER_IP}..."
    ssh "${REMOTE_HOST}" zsh -s -- \
        "${REPO_ROOT}" \
        "${EXAMPLE_DIR}" \
        "${LAUNCH_SCRIPT}" \
        "${REMOTE_NODE_RANK}" \
        "${MASTER_IP}" \
        "${MODEL_PATH}" \
        "${SERVED_MODEL_NAME}" \
        "${LOAD_FORMAT}" \
        "${PORT}" \
        "${DP_RPC_PORT}" \
        "${KV_PORT}" \
        "${KV_PARALLEL_SIZE}" \
        "${KV_RANK}" \
        "${MAX_MODEL_LEN}" \
        "${MAX_SEQS_PER_DP}" \
        "${MAX_NUM_BATCHED_TOKENS}" \
        "${GPU_MEMORY_UTILIZATION}" \
        "${LOG_DIR}" \
        "${PROFILE_MODE}" \
        "CUDAGRAPH_MAX_CAPTURE_SIZE=${CUDAGRAPH_MAX_CAPTURE_SIZE}" \
        "CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES}" \
        "CUDAGRAPH_MODE=${CUDAGRAPH_MODE}" \
        "CUDAGRAPH_CAPTURE_SIZES_FOR_CP=${CUDAGRAPH_CAPTURE_SIZES_FOR_CP}" \
        "COMPILATION_CONFIG_B64=$(b64_env_value "${COMPILATION_CONFIG}")" <<'REMOTE_START'
set -e

unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

decode_b64_env_value() {
    if [ -z "$1" ]; then
        printf ''
    else
        printf '%s' "$1" | base64 -d
    fi
}

repo_root=$1
example_dir=$2
launch_script=$3
node_rank=$4
master_ip=$5
model_path=$6
served_model_name=$7
load_format=$8
port=$9
dp_rpc_port=${10}
kv_port=${11}
kv_parallel_size=${12}
kv_rank=${13}
max_model_len=${14}
max_seqs_per_dp=${15}
max_num_batched_tokens=${16}
gpu_memory_utilization=${17}
log_dir=${18}
profile_mode=${19}
cudagraph_max_capture_size=${20#CUDAGRAPH_MAX_CAPTURE_SIZE=}
cudagraph_capture_sizes=${21#CUDAGRAPH_CAPTURE_SIZES=}
cudagraph_mode=${22#CUDAGRAPH_MODE=}
cudagraph_capture_sizes_for_cp=${23#CUDAGRAPH_CAPTURE_SIZES_FOR_CP=}
compilation_config_b64=${24#COMPILATION_CONFIG_B64=}

export MODEL_PATH="${model_path}"
export SERVED_MODEL_NAME="${served_model_name}"
export LOAD_FORMAT="${load_format}"
export PORT="${port}"
export DP_RPC_PORT="${dp_rpc_port}"
export KV_PORT="${kv_port}"
export KV_PARALLEL_SIZE="${kv_parallel_size}"
export KV_RANK="${kv_rank}"
export MAX_MODEL_LEN="${max_model_len}"
export MAX_SEQS_PER_DP="${max_seqs_per_dp}"
export MAX_NUM_BATCHED_TOKENS="${max_num_batched_tokens}"
export GPU_MEMORY_UTILIZATION="${gpu_memory_utilization}"
export LOG_DIR="${log_dir}"
export PROFILE_MODE="${profile_mode}"
export CUDAGRAPH_MAX_CAPTURE_SIZE="${cudagraph_max_capture_size}"
export CUDAGRAPH_CAPTURE_SIZES="${cudagraph_capture_sizes}"
export CUDAGRAPH_MODE="${cudagraph_mode}"
export CUDAGRAPH_CAPTURE_SIZES_FOR_CP="${cudagraph_capture_sizes_for_cp}"
export COMPILATION_CONFIG="$(decode_b64_env_value "${compilation_config_b64}")"

cd "${repo_root}"
source .venv/bin/activate
mkdir -p "${log_dir}"
cd "${example_dir}"
bash "${launch_script}" "${node_rank}" "${master_ip}"
REMOTE_START
}

check_health() {
    curl -fsS --max-time 2 "http://${BENCH_HOST}:${PORT}/health" >/dev/null
}

wait_for_server() {
    local attempt
    echo "Waiting for http://${BENCH_HOST}:${PORT}/health ..."
    for attempt in $(seq 1 "${SERVER_READY_RETRIES}"); do
        if check_health; then
            echo "Server is ready."
            return 0
        fi

        echo "Server is not ready yet (${attempt}/${SERVER_READY_RETRIES}); sleeping ${SERVER_READY_INTERVAL}s."
        sleep "${SERVER_READY_INTERVAL}"
    done

    echo "Timed out waiting for http://${BENCH_HOST}:${PORT}/health." >&2
    return 1
}

run_benchmark() {
    echo "Running vLLM benchmark against ${BENCH_HOST}:${PORT}..."
    activate_venv
    cd "${REPO_ROOT}"
    mkdir -p "${RESULT_DIR}"

    vllm bench serve \
        --random-use-token-ids \
        --backend openai \
        --dataset-name random \
        --use-local-json "${DATASET_PATH}" \
        --num-prompts "${NUM_PROMPTS}" \
        --request-rate "${REQUEST_RATE}" \
        --max-concurrency "${MAX_CONCURRENCY}" \
        --model "${MODEL_PATH}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --endpoint "${ENDPOINT}" \
        --host "${BENCH_HOST}" \
        --port "${PORT}" \
        --ignore-eos \
        --metric-percentiles "${METRIC_PERCENTILES}" \
        --save-result \
        --result-dir "${RESULT_DIR}"
}

cleanup_local() {
    echo "Cleaning local vLLM processes..."
    pkill -TERM -f '[v]llm bench serve' 2>/dev/null || true
    pkill -TERM -f '[v]llm serve' 2>/dev/null || true
    pkill -TERM -f '[v]llm.entrypoints' 2>/dev/null || true
    pgrep -f 'VLLM::EngineCore_' | xargs -r kill -TERM 2>/dev/null || true
    sleep 2
    pkill -KILL -f '[v]llm bench serve' 2>/dev/null || true
    pkill -KILL -f '[v]llm serve' 2>/dev/null || true
    pkill -KILL -f '[v]llm.entrypoints' 2>/dev/null || true
    pgrep -f 'VLLM::EngineCore_' | xargs -r kill -KILL 2>/dev/null || true
}

cleanup_remote() {
    echo "Cleaning remote vLLM processes on ${REMOTE_HOST}..."
    ssh "${REMOTE_HOST}" zsh -s <<'REMOTE_CLEANUP' || true
set +e
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
pkill -TERM -f '[v]llm bench serve' 2>/dev/null
pkill -TERM -f '[v]llm serve' 2>/dev/null
pkill -TERM -f '[v]llm.entrypoints' 2>/dev/null
pgrep -f 'VLLM::EngineCore_' | xargs -r kill -TERM 2>/dev/null
sleep 2
pkill -KILL -f '[v]llm bench serve' 2>/dev/null
pkill -KILL -f '[v]llm serve' 2>/dev/null
pkill -KILL -f '[v]llm.entrypoints' 2>/dev/null
pgrep -f 'VLLM::EngineCore_' | xargs -r kill -KILL 2>/dev/null
REMOTE_CLEANUP
}

cleanup_gpu_hosts() {
    if [ "${RUN_GPU_CLEANUP}" != "1" ]; then
        echo "Skipping GPU host cleanup because RUN_GPU_CLEANUP=${RUN_GPU_CLEANUP}."
        return 0
    fi

    local hosts=()
    read -r -a hosts <<< "${GPU_CLEANUP_HOSTS}"
    if [ "${#hosts[@]}" -eq 0 ]; then
        echo "Skipping GPU host cleanup because GPU_CLEANUP_HOSTS is empty."
        return 0
    fi

    echo "Running ${GPU_CLEANUP_SCRIPT} on ${GPU_CLEANUP_HOSTS}..."
    GPU_CLEANUP_SCRIPT="${GPU_CLEANUP_SCRIPT}" \
    GPU_CLEANUP_SELECTION="${GPU_CLEANUP_SELECTION}" \
        python3 "${SCRIPT_DIR}/kill_gpu_hosts.py" "${hosts[@]}" || true
}

cleanup_all() {
    set +e
    cleanup_local
    cleanup_remote
    cleanup_gpu_hosts
}

run_all() {
    trap cleanup_all EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    if [ "${CLEAN_BEFORE_START}" = "1" ]; then
        cleanup_all
    fi

    start_local_server
    start_remote_server

    if [ "${WAIT_FOR_SERVER}" = "1" ]; then
        wait_for_server
    fi

    run_benchmark
}

main() {
    local command=${1:-run}
    case "${command}" in
        run)
            run_all
            ;;
        start)
            start_local_server
            start_remote_server
            ;;
        bench)
            run_benchmark
            ;;
        wait)
            wait_for_server
            ;;
        cleanup)
            cleanup_all
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            usage >&2
            return 2
            ;;
    esac
}

main "$@"
