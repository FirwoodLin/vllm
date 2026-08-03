#!/usr/bin/env zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
START_SCRIPT="${START_SCRIPT:-${SCRIPT_DIR}/start_multinode_offline_profile.sh}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/rebuttal/ProfileLayer/vLLM-DP4CP8}"
CLUSTER="${CLUSTER:-4node_h200}"
MODEL="${MODEL:-deepseek_v3_1024k}"
DISPATCH_POLICY="${DISPATCH_POLICY:-least_batch}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
PROFILE_DELAY_ITERATIONS="${PROFILE_DELAY_ITERATIONS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1200}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-32}"
DP_SIZE=4
TP_SIZE=8

# Use binary-k prompt lengths (1k = 1024 tokens). Keep the largest prompt at
# 1023k so OUTPUT_LEN fits within MAX_MODEL_LEN=1024k, matching the DP32 sweep.
PROMPT_LENS=(8192 16384 32768 65536 131072 262144 524288 1047552)
# DP4: requests-per-dp is scaled so global requests still equal 32 * N.
REQUESTS_PER_DP=(1024 512 256 128 64 32 16 8)
REQUESTS_PER_GPU=(128 64 32 16 8 4 2 1)
TOTAL_REQUESTS=(4096 2048 1024 512 256 128 64 32)
TAGS=(8k 16k 32k 64k 128k 256k 512k 1024k)

if (( WARMUP_REQUESTS % DP_SIZE != 0 )); then
  echo "WARMUP_REQUESTS=${WARMUP_REQUESTS} is not divisible by DP_SIZE=${DP_SIZE}." >&2
  exit 1
fi

function build_cudagraph_capture_sizes() {
  local per_gpu_requests="$1"
  local max_batch_size="$(( TP_SIZE * per_gpu_requests ))"
  local candidates=(1 2 4 8)
  candidates+=($(( max_batch_size - 16 )))
  candidates+=($(( max_batch_size - 8 )))
  candidates+=("${max_batch_size}")

  local -A seen=()
  local values=()
  local candidate
  for candidate in "${candidates[@]}"; do
    if (( candidate <= 0 )); then
      continue
    fi
    if [[ -n "${seen[${candidate}]:-}" ]]; then
      continue
    fi
    seen[${candidate}]=1
    values+=("${candidate}")
  done
  local sorted_values=("${(@on)values}")
  print -r -- "${(j:,:)sorted_values}"
}

COMMON_EXTRA_ARGS=(
  --max-num-seqs "${MAX_NUM_SEQS}"
)
if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then
  COMMON_EXTRA_ARGS+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi

for (( idx=${#PROMPT_LENS[@]}; idx >= 1; idx-- )); do
  prompt_len="${PROMPT_LENS[$idx]}"
  requests_per_dp="${REQUESTS_PER_DP[$idx]}"
  requests_per_gpu="${REQUESTS_PER_GPU[$idx]}"
  total_requests="${TOTAL_REQUESTS[$idx]}"
  tag="${TAGS[$idx]}"
  case_name="profile_dp4tp8dcp8_ep_total${total_requests}_prompt${tag}_out${OUTPUT_LEN}"
  requests_per_rank="$(( total_requests / DP_SIZE ))"
  cudagraph_capture_sizes="$(build_cudagraph_capture_sizes "${requests_per_gpu}")"

  if (( total_requests % DP_SIZE != 0 )); then
    echo "${case_name}: total_requests=${total_requests} is not divisible by DP_SIZE=${DP_SIZE}." >&2
    exit 1
  fi
  if (( total_requests < WARMUP_REQUESTS )); then
    echo "${case_name}: total_requests=${total_requests} is smaller than WARMUP_REQUESTS=${WARMUP_REQUESTS}." >&2
    exit 1
  fi

  print -r -- "Running ${case_name} (${requests_per_rank} requests per DP rank, cudagraph_capture_sizes=${cudagraph_capture_sizes})"
  zsh "${START_SCRIPT}" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --cluster "${CLUSTER}" \
    --strategy dp4dcp8 \
    --model "${MODEL}" \
    --prompt-len "${prompt_len}" \
    --requests-per-dp "${requests_per_dp}" \
    --output-len "${OUTPUT_LEN}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --dispatch-policy "${DISPATCH_POLICY}" \
    --routing-mode explicit_rank_replay \
    --max-requests csv_rows \
    --max-model-len "${MAX_MODEL_LEN}" \
    --cudagraph-capture-sizes "${cudagraph_capture_sizes}" \
    --profile-delay-iterations "${PROFILE_DELAY_ITERATIONS}" \
    --case-name "${case_name}" \
    --pause-before-profile \
    "${COMMON_EXTRA_ARGS[@]}" \
    "$@"
done
