#!/bin/bash

set -euo pipefail
set -x

MODE=${MODE:-dp4tp4}
HOST=${HOST:-10.102.97.183}
PORT=${PORT:-8400}
MODEL_PATH=${MODEL_PATH:-/mnt/nvme1n1/ml_research/models_cfs/qwen3-235B-Instruct-2507-FP8/}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-auto}

SHORT_INPUT_LEN=${SHORT_INPUT_LEN:-4000}
LONG_INPUT_LEN=${LONG_INPUT_LEN:-512000}
OUTPUT_LEN=${OUTPUT_LEN:-32}
SHORT_REQS_PER_8GPU=${SHORT_REQS_PER_8GPU:-252}
LONG_REQS_PER_8GPU=${LONG_REQS_PER_8GPU:-4}
NUM_GPUS=${NUM_GPUS:-16}
REQUEST_RATE=${REQUEST_RATE:-inf}
NUM_WARMUPS=${NUM_WARMUPS:-0}
BACKEND=${BACKEND:-openai}
ENDPOINT=${ENDPOINT:-/v1/completions}
RANDOM_USE_TOKEN_IDS=${RANDOM_USE_TOKEN_IDS:-1}
PAUSE_BEFORE_PROFILE=${PAUSE_BEFORE_PROFILE:-1}
PROFILE_REQUEST_SETTLE_SECONDS=${PROFILE_REQUEST_SETTLE_SECONDS:-5}
PROFILE_PAUSE_WAIT_INFLIGHT=${PROFILE_PAUSE_WAIT_INFLIGHT:-0}
PROFILE_PAUSE_CLEAR_CACHE=${PROFILE_PAUSE_CLEAR_CACHE:-0}
CURL=${CURL:-curl}

RESULT_BASE_DIR=${RESULT_BASE_DIR:-/mnt/nvme1n1/ml_research/linbinbin1/vllm-dycp/dycp/results/profile_batch}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}

case "${MODE}" in
    dp4tp4)
        DP_SIZE=${DP_SIZE:-4}
        TP_SIZE=${TP_SIZE:-4}
        DP_PER_DOMAIN=${DP_PER_DOMAIN:-1}
        ROUTE_TARGET_NAME=${ROUTE_TARGET_NAME:-DP}
        ;;
    dycp16dp_tp1|dycp16|dp16dycp)
        DP_SIZE=${DP_SIZE:-16}
        TP_SIZE=${TP_SIZE:-1}
        DP_PER_DOMAIN=${DP_PER_DOMAIN:-8}
        ROUTE_TARGET_NAME=${ROUTE_TARGET_NAME:-DyCP domain}
        MODE=dycp16dp_tp1
        ;;
    *)
        echo "Unsupported MODE=${MODE}. Use dp4tp4 or dycp16dp_tp1." >&2
        exit 1
        ;;
esac

if [ $((NUM_GPUS % 8)) -ne 0 ]; then
    echo "NUM_GPUS=${NUM_GPUS} must be divisible by 8 for this workload." >&2
    exit 1
fi

NUM_8GPU_GROUPS=$((NUM_GPUS / 8))
SHORT_NUM_PROMPTS=${SHORT_NUM_PROMPTS:-$((SHORT_REQS_PER_8GPU * NUM_8GPU_GROUPS))}
LONG_NUM_PROMPTS=${LONG_NUM_PROMPTS:-$((LONG_REQS_PER_8GPU * NUM_8GPU_GROUPS))}
NUM_PROMPTS=${NUM_PROMPTS:-$((SHORT_NUM_PROMPTS + LONG_NUM_PROMPTS))}

if [ $((DP_SIZE % DP_PER_DOMAIN)) -ne 0 ]; then
    echo "DP_SIZE=${DP_SIZE} must be divisible by DP_PER_DOMAIN=${DP_PER_DOMAIN}." >&2
    exit 1
fi
ROUTE_TARGET_COUNT=$((DP_SIZE / DP_PER_DOMAIN))

ROUTE_BY_DP_HEADER=${ROUTE_BY_DP_HEADER:-auto}
if [ "${ROUTE_BY_DP_HEADER}" = "auto" ]; then
    if [ $((SHORT_NUM_PROMPTS % ROUTE_TARGET_COUNT)) -eq 0 ] && [ $((LONG_NUM_PROMPTS % ROUTE_TARGET_COUNT)) -eq 0 ]; then
        ROUTE_BY_DP_HEADER=1
    else
        ROUTE_BY_DP_HEADER=0
    fi
fi

if [ "${ROUTE_BY_DP_HEADER}" = "1" ]; then
    if [ $((SHORT_NUM_PROMPTS % ROUTE_TARGET_COUNT)) -ne 0 ] || [ $((LONG_NUM_PROMPTS % ROUTE_TARGET_COUNT)) -ne 0 ]; then
        echo "Cannot guarantee equal per-${ROUTE_TARGET_NAME} long/short requests: SHORT_NUM_PROMPTS=${SHORT_NUM_PROMPTS}, LONG_NUM_PROMPTS=${LONG_NUM_PROMPTS}, ROUTE_TARGET_COUNT=${ROUTE_TARGET_COUNT}." >&2
        echo "Make both request counts divisible by ROUTE_TARGET_COUNT, or set ROUTE_BY_DP_HEADER=0 to use normal internal load balancing." >&2
        exit 1
    fi
fi

MAX_CONCURRENCY=${MAX_CONCURRENCY:-${NUM_PROMPTS}}
RESULT_DIR=${RESULT_DIR:-"${RESULT_BASE_DIR}/${RUN_TAG}/${MODE}"}
mkdir -p "${RESULT_DIR}"

WORKLOAD_JSON=${WORKLOAD_JSON:-"${RESULT_DIR}/mixed_4k512k_${NUM_PROMPTS}reqs.json"}
LABEL=${LABEL:-"${MODE}-profile-16gpu-${SHORT_INPUT_LEN}x${SHORT_NUM_PROMPTS}-${LONG_INPUT_LEN}x${LONG_NUM_PROMPTS}-${OUTPUT_LEN}out"}
RESULT_FILENAME=${RESULT_FILENAME:-"${LABEL}-${RUN_TAG}.json"}

restore_xtrace=0
case "$-" in
    *x*)
        restore_xtrace=1
        set +x
        ;;
esac
{
    echo "["
    req_idx=0
    short_remaining=${SHORT_NUM_PROMPTS}
    long_remaining=${LONG_NUM_PROMPTS}
    short_assigned=0
    long_assigned=0
    total_remaining=${NUM_PROMPTS}
    group_size=$(((SHORT_REQS_PER_8GPU + LONG_REQS_PER_8GPU) / LONG_REQS_PER_8GPU))
    while [ "${total_remaining}" -gt 0 ]; do
        if [ "${long_remaining}" -gt 0 ] && [ $(((req_idx + 1) % group_size)) -eq 0 ]; then
            input_len=${LONG_INPUT_LEN}
            request_kind=long
            long_remaining=$((long_remaining - 1))
        elif [ "${short_remaining}" -gt 0 ]; then
            input_len=${SHORT_INPUT_LEN}
            request_kind=short
            short_remaining=$((short_remaining - 1))
        else
            input_len=${LONG_INPUT_LEN}
            request_kind=long
            long_remaining=$((long_remaining - 1))
        fi
        if [ "${ROUTE_BY_DP_HEADER}" = "1" ]; then
            if [ "${request_kind}" = "long" ]; then
                dp_rank=$((long_assigned % ROUTE_TARGET_COUNT))
                long_assigned=$((long_assigned + 1))
            else
                dp_rank=$((short_assigned % ROUTE_TARGET_COUNT))
                short_assigned=$((short_assigned + 1))
            fi
        fi
        req_idx=$((req_idx + 1))
        total_remaining=$((total_remaining - 1))
        comma=","
        if [ "${total_remaining}" -eq 0 ]; then
            comma=""
        fi
        if [ "${ROUTE_BY_DP_HEADER}" = "1" ]; then
            echo "  [${input_len}, ${OUTPUT_LEN}, ${dp_rank}]${comma}"
        else
            echo "  [${input_len}, ${OUTPUT_LEN}]${comma}"
        fi
    done
    echo "]"
} > "${WORKLOAD_JSON}"
if [ "${restore_xtrace}" = "1" ]; then
    set -x
fi

cd /vllm
source .venv/bin/activate

cmd=(
    vllm bench serve
    --backend "${BACKEND}"
    --dataset-name random \
    --trust-remote-code \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --model "${MODEL_PATH}" \
    --random-input-len "${LONG_INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --use-local-json "${WORKLOAD_JSON}" \
    --num-prompts "${NUM_PROMPTS}" \
    --num-warmups "${NUM_WARMUPS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --request-rate "${REQUEST_RATE}" \
    --ignore-eos \
    --metric-percentiles "50,90,99" \
    --host "${HOST}" \
    --port "${PORT}" \
    --endpoint "${ENDPOINT}" \
    --temperature 0.6 \
    --save-result \
    --result-dir "${RESULT_DIR}" \
    --result-filename "${RESULT_FILENAME}"
)

if [ "${PAUSE_BEFORE_PROFILE}" != "1" ]; then
    cmd+=(--profile)
fi

if [ "${RANDOM_USE_TOKEN_IDS}" = "1" ]; then
    cmd+=(--random-use-token-ids)
fi

echo "Workload: ${SHORT_NUM_PROMPTS} requests x ${SHORT_INPUT_LEN} input tokens, ${LONG_NUM_PROMPTS} requests x ${LONG_INPUT_LEN} input tokens, output ${OUTPUT_LEN}"
if [ "${ROUTE_BY_DP_HEADER}" = "1" ]; then
    echo "DP routing: X-data-parallel-rank enabled; each ${ROUTE_TARGET_NAME} gets $((SHORT_NUM_PROMPTS / ROUTE_TARGET_COUNT)) short and $((LONG_NUM_PROMPTS / ROUTE_TARGET_COUNT)) long requests."
else
    echo "DP routing: using normal internal load balancing; per-${ROUTE_TARGET_NAME} long/short counts are not guaranteed."
fi
echo "Benchmark command: ${cmd[*]}"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1, not sending requests."
    echo "Workload json: ${WORKLOAD_JSON}"
    exit 0
fi

if [ "${PAUSE_BEFORE_PROFILE}" = "1" ]; then
    BASE_URL="http://${HOST}:${PORT}"
    BENCHMARK_LOG="${RESULT_DIR}/${RESULT_FILENAME%.json}.bench.log"
    profile_started=0

    cleanup_profile_pause() {
        status=$?
        set +e
        if [ "${profile_started}" = "1" ]; then
            "${CURL}" -fsS -X POST "${BASE_URL}/stop_profile" >/dev/null
        fi
        "${CURL}" -fsS -X POST "${BASE_URL}/resume" >/dev/null
        exit "${status}"
    }
    trap cleanup_profile_pause EXIT INT TERM

    echo "Pausing generation before sending benchmark requests..."
    "${CURL}" -fsS -X POST "${BASE_URL}/pause?wait_for_inflight_requests=${PROFILE_PAUSE_WAIT_INFLIGHT}&clear_cache=${PROFILE_PAUSE_CLEAR_CACHE}"

    echo "Starting benchmark while generation is paused. Log: ${BENCHMARK_LOG}"
    "${cmd[@]}" >"${BENCHMARK_LOG}" 2>&1 &
    benchmark_pid=$!

    echo "Waiting ${PROFILE_REQUEST_SETTLE_SECONDS}s for benchmark requests to reach the paused server..."
    sleep "${PROFILE_REQUEST_SETTLE_SECONDS}"

    echo "Starting profiler and resuming generation..."
    "${CURL}" -fsS -X POST "${BASE_URL}/start_profile"
    profile_started=1
    "${CURL}" -fsS -X POST "${BASE_URL}/resume"

    wait "${benchmark_pid}"

    echo "Stopping profiler..."
    "${CURL}" -fsS -X POST "${BASE_URL}/stop_profile"
    profile_started=0
    trap - EXIT INT TERM
else
    "${cmd[@]}"
fi

echo "Benchmark result: ${RESULT_DIR}/${RESULT_FILENAME}"
echo "Workload json: ${WORKLOAD_JSON}"
