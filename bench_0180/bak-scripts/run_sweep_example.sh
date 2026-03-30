#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_ENV="${CLUSTER_ENV:-${SCRIPT_DIR}/cluster.env}"

if [[ ! -f "${CLUSTER_ENV}" ]]; then
  echo "Missing ${CLUSTER_ENV}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${CLUSTER_ENV}"
set +a

shell_join() {
  local parts=()
  local arg
  for arg in "$@"; do
    parts+=("$(printf '%q' "${arg}")")
  done
  local IFS=' '
  printf '%s' "${parts[*]}"
}

: "${MODEL:?MODEL must be set in cluster.env}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set in cluster.env}"

SERVE_BIND_HOST="${SERVE_HOST:-${SWEEP_HOST:-127.0.0.1}}"
SERVE_BIND_PORT="${SERVE_PORT:-${SWEEP_PORT:-8000}}"
BENCH_TARGET_HOST="${BENCH_HOST:-${SWEEP_HOST:-${SERVE_BIND_HOST}}}"
BENCH_TARGET_PORT="${BENCH_PORT:-${SWEEP_PORT:-${SERVE_BIND_PORT}}}"

serve_cmd="$(shell_join \
  "${SCRIPT_DIR}/start_sweep_serve.sh" \
  --model "${MODEL}" \
  --master-addr "${MASTER_ADDR:-auto}" \
  --host "${SERVE_BIND_HOST}" \
  --port "${SERVE_BIND_PORT}" \
  --data-parallel-rpc-port "${DATA_PARALLEL_RPC_PORT:-13345}" \
  --)"
bench_cmd="$(shell_join \
  vllm bench serve \
  --model "${MODEL}" \
  --backend vllm \
  --host "${BENCH_TARGET_HOST}" \
  --port "${BENCH_TARGET_PORT}" \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 200)"
after_bench_cmd="$(shell_join \
  "${SCRIPT_DIR}/reset_sweep_caches.sh" \
  --host "${BENCH_TARGET_HOST}" \
  --port "${BENCH_TARGET_PORT}")"

vllm bench sweep serve \
  --serve-cmd "${serve_cmd}" \
  --bench-cmd "${bench_cmd}" \
  --after-bench-cmd "${after_bench_cmd}" \
  --serve-params "${SCRIPT_DIR}/serve_params.4dp8tp_async_mp.json" \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME:-4dp8tp_async_mp}" \
  --num-runs "${NUM_RUNS:-1}" \
  --server-ready-timeout "${SERVER_READY_TIMEOUT:-1800}"
