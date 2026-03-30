#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/mnt/nvme1n1/ml_research/models/deepseek-v3}"
TP_SIZE="${TP_SIZE:-8}"
EXECUTOR_BACKEND="${EXECUTOR_BACKEND:-mp}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
DECODE_BENCH_FILL_MEAN="${DECODE_BENCH_FILL_MEAN:-0.015}"
DECODE_BENCH_FILL_STD="${DECODE_BENCH_FILL_STD:-0.0}"
DEEP_GEMM_WARMUP="${DEEP_GEMM_WARMUP:-skip}"

INPUT_MIN="${INPUT_MIN:-128}"
INPUT_MAX="${INPUT_MAX:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-2048}"
BATCH_MIN="${BATCH_MIN:-32}"
BATCH_MAX="${BATCH_MAX:-64}"

NUM_ITERS_WARMUP="${NUM_ITERS_WARMUP:-10}"
NUM_ITERS="${NUM_ITERS:-30}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"
VLLM_BIN="${VLLM_BIN:-vllm}"

DEFAULT_MAX_MODEL_LEN=$((INPUT_MAX + OUTPUT_LEN))
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${DEFAULT_MAX_MODEL_LEN}}"
KV_TRANSFER_CONFIG="${KV_TRANSFER_CONFIG:-}"

rand_int() {
  local min="$1"
  local max="$2"
  echo $((min + RANDOM % (max - min + 1)))
}

trap 'echo; echo "Interrupted by Ctrl+C, exiting."; exit 130' INT

if ! command -v "${VLLM_BIN}" >/dev/null 2>&1; then
  echo "Error: '${VLLM_BIN}' not found in PATH." >&2
  exit 127
fi

if ((INPUT_MIN > INPUT_MAX)); then
  echo "Error: INPUT_MIN (${INPUT_MIN}) cannot be greater than INPUT_MAX (${INPUT_MAX})." >&2
  exit 1
fi

if ((BATCH_MIN > BATCH_MAX)); then
  echo "Error: BATCH_MIN (${BATCH_MIN}) cannot be greater than BATCH_MAX (${BATCH_MAX})." >&2
  exit 1
fi

if ((INPUT_MAX + OUTPUT_LEN > MAX_MODEL_LEN)); then
  echo "Error: INPUT_MAX + OUTPUT_LEN = $((INPUT_MAX + OUTPUT_LEN)) exceeds MAX_MODEL_LEN (${MAX_MODEL_LEN})." >&2
  exit 1
fi

if [[ -z "${KV_TRANSFER_CONFIG}" ]]; then
  KV_TRANSFER_CONFIG="$(printf \
    '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":%s,"fill_std":%s}}' \
    "${DECODE_BENCH_FILL_MEAN}" \
    "${DECODE_BENCH_FILL_STD}")"
fi

iteration=0

while true; do
  iteration=$((iteration + 1))
  input_len="$(rand_int "${INPUT_MIN}" "${INPUT_MAX}")"
  batch_size="$(rand_int "${BATCH_MIN}" "${BATCH_MAX}")"
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  cmd=(
    "${VLLM_BIN}"
    bench
    latency
    --model "${MODEL}"
    --tensor-parallel-size "${TP_SIZE}"
    --distributed-executor-backend "${EXECUTOR_BACKEND}"
    --load-format "${LOAD_FORMAT}"
    --kv-transfer-config "${KV_TRANSFER_CONFIG}"
    --enforce-eager
    --max-model-len "${MAX_MODEL_LEN}"
    --input-len "${input_len}"
    --output-len "${OUTPUT_LEN}"
    --batch-size "${batch_size}"
    --num-iters-warmup "${NUM_ITERS_WARMUP}"
    --num-iters "${NUM_ITERS}"
    --disable-log-stats
  )

  if (($# > 0)); then
    cmd+=("$@")
  fi

  printf '\n[%s] iteration=%d input_len=%d output_len=%s batch_size=%d tp=%s max_model_len=%s\n' \
    "${timestamp}" "${iteration}" "${input_len}" "${OUTPUT_LEN}" "${batch_size}" "${TP_SIZE}" "${MAX_MODEL_LEN}"
  printf 'Env: VLLM_DEEP_GEMM_WARMUP=%s\n' "${DEEP_GEMM_WARMUP}"
  printf 'Command: '
  printf '%q ' "${cmd[@]}"
  printf '\n\n'

  VLLM_DEEP_GEMM_WARMUP="${DEEP_GEMM_WARMUP}" "${cmd[@]}"

  if ((SLEEP_SECONDS > 0)); then
    sleep "${SLEEP_SECONDS}"
  fi
done
