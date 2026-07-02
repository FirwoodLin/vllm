#!/usr/bin/env bash

set -euo pipefail

unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

REPO_ROOT=${REPO_ROOT:-/vllm}
EXAMPLE_DIR=${EXAMPLE_DIR:-"${REPO_ROOT}/dycp/examples/qwen3-235b-FP8"}
LAUNCH_SCRIPT=${LAUNCH_SCRIPT:-run_qwen_235b_dp_ep_4dp4tp.sh}

REMOTE_HOST=${REMOTE_HOST:-h200-rjob2}
MASTER_IP=${MASTER_IP:-10.102.97.183}
LOCAL_NODE_RANK=${LOCAL_NODE_RANK:-0}
REMOTE_NODE_RANK=${REMOTE_NODE_RANK:-1}

PORT=${PORT:-8400}
BENCH_HOST=${BENCH_HOST:-localhost}
MODEL_PATH=${MODEL_PATH:-/mnt/nvme1n1/ml_research/models_cfs/qwen3-235B-Instruct-2507-FP8/}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-auto}
DATASET_PATH=${DATASET_PATH:-"${REPO_ROOT}/dycp/dataset/trace_512k_4k_long1pct_10000-output1024.json"}
RESULT_DIR=${RESULT_DIR:-"${REPO_ROOT}/dycp/results/dp4tp4"}
NUM_PROMPTS=${NUM_PROMPTS:-4800}
REQUEST_RATE=${REQUEST_RATE:-16}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-0}
METRIC_PERCENTILES=${METRIC_PERCENTILES:-50,90,99}
ENDPOINT=${ENDPOINT:-/v1/completions}

WAIT_FOR_SERVER=${WAIT_FOR_SERVER:-1}
SERVER_READY_RETRIES=${SERVER_READY_RETRIES:-120}
SERVER_READY_INTERVAL=${SERVER_READY_INTERVAL:-10}
CLEAN_BEFORE_START=${CLEAN_BEFORE_START:-0}
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
  cleanup   Kill vLLM server/benchmark processes locally and on ${REMOTE_HOST}.

Common overrides:
  MASTER_IP=10.102.97.183 REMOTE_HOST=h200-rjob2 PORT=8400 ${RUNNER_NAME}
  REQUEST_RATE=16 NUM_PROMPTS=4800 MAX_CONCURRENCY=0 ${RUNNER_NAME} bench
  LAUNCH_SCRIPT=${LAUNCH_SCRIPT} ${RUNNER_NAME}
USAGE
}

activate_venv() {
    cd "${REPO_ROOT}"
    set +u
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
    set -u
}

start_local_server() {
    echo "Starting local node ${LOCAL_NODE_RANK} with master ${MASTER_IP}..."
    activate_venv
    cd "${EXAMPLE_DIR}"
    bash "${LAUNCH_SCRIPT}" "${LOCAL_NODE_RANK}" "${MASTER_IP}"
}

start_remote_server() {
    echo "Starting remote node ${REMOTE_NODE_RANK} on ${REMOTE_HOST} with master ${MASTER_IP}..."
    ssh "${REMOTE_HOST}" zsh -s -- \
        "${REPO_ROOT}" \
        "${EXAMPLE_DIR}" \
        "${LAUNCH_SCRIPT}" \
        "${REMOTE_NODE_RANK}" \
        "${MASTER_IP}" <<'REMOTE_START'
set -e

unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

repo_root=$1
example_dir=$2
launch_script=$3
node_rank=$4
master_ip=$5

cd "${repo_root}"
source .venv/bin/activate
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

cleanup_all() {
    set +e
    cleanup_local
    cleanup_remote
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
