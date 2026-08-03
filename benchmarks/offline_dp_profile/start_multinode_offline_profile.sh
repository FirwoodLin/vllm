#!/usr/bin/env zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench/profile_multinode}"
CASE_CSV="${CASE_CSV:-${SCRIPT_DIR}/profile_cases/offline_profile_template.casecsv}"
GENERATED_INPUT_ROOT="${GENERATED_INPUT_ROOT:-${REPO_ROOT}/benchmarks/offline_dp_profile/generated_inputs}"
CLUSTER="4node_h200"
STRATEGY="dp32"
MODEL=""
LENS_JSON=""
PROMPT_LEN=""
REQUESTS_PER_DP=""
OUTPUT_LEN=64
WARMUP_REQUESTS=""
MAX_REQUESTS=""
REQUEST_RATE=""
DISPATCH_POLICY=""
ROUTING_MODE="internal_dplb"
CASE_NAME=""
MAX_NUM_SEQS=""
GPU_MEMORY_UTILIZATION=""
DATA_PARALLEL_RPC_PORT=""
MAX_MODEL_LEN=""
CUDAGRAPH_CAPTURE_SIZES=""
PROFILE_DELAY_ITERATIONS=33
PAUSE_BEFORE_PROFILE=0
IGNORE_HISTORICAL_SKIPS=0
FORCE_NO_ASYNC_SCHEDULING=1

function sanitize_tag() {
  local sanitized
  sanitized="$(print -r -- "$1" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_')"
  sanitized="${sanitized#_}"
  sanitized="${sanitized%_}"
  print -r -- "${sanitized:-case}"
}

function usage() {
  cat <<'EOF'
Usage:
  start_multinode_offline_profile.sh --lens-json PATH [options]
  start_multinode_offline_profile.sh --prompt-len N --requests-per-dp N [options]

Options:
  --artifact-root PATH
  --case-csv PATH
  --cluster NAME                    # default: 4node_h200
  --strategy NAME                   # default: dp32
  --model NAME
  --lens-json PATH
  --prompt-len N                    # synthetic fixed prompt_len for every request
  --requests-per-dp N               # synthetic requests per global DP replica
  --output-len N
  --warmup-requests N
  --max-requests N|csv_rows
  --request-rate FLOAT
  --dispatch-policy waiting_x4_plus_running|least_cache|least_batch
  --routing-mode internal_dplb|explicit_rank_replay
  --case-name NAME
  --max-num-seqs N
  --gpu-memory-utilization FLOAT
  --data-parallel-rpc-port N
  --max-model-len N
  --cudagraph-capture-sizes CSV  # e.g. 1,2,4,8,240,248,256,260
  --profile-delay-iterations N      # default: 33
  --async-scheduling               # pass --async-scheduling to harnesses
  --pause-before-profile
  --ignore-historical-skips
  -h, --help

Strategies with built-in offline-profile defaults:
  dp4dcp8, dp8dcp4, dp16cp2, dp32, dp4tp4

Additional runner-supported strategies:
  dp1tp8dcp2, dp4tp8dcp2, dp4tp8dcp2_ar, dp4tp8, dp8tp4, dp16tp2

Notes:
  - Pass either --lens-json, or both --prompt-len and --requests-per-dp.
  - DCP strategies rely on benchmarks/manual_multinode_poisson_runner.py as the
    topology source of truth. For dcp>1 strategies, the runner injects
    --dcp-comm-backend a2a from the strategy table.
  - For strategies without built-in defaults, pass both --max-num-seqs and
    --gpu-memory-utilization explicitly.
  - dp1tp8dcp2 is single-node only; pair it with --cluster 1node_h200.
  - dp4tp4 already enables expert parallel via the runner strategy table.
EOF
}

function strategy_total_dp() {
  case "$1" in
    dp1tp8dcp2)
      print -r -- 1
      ;;
    dp2tp8dcp2)
      print -r -- 2
      ;;
    dp4dcp8|dp4tp8dcp2|dp4tp8dcp2_ar|dp4tp8|dp4tp4)
      print -r -- 4
      ;;
    dp8dcp4|dp8tp4)
      print -r -- 8
      ;;
    dp16cp2|dp16tp2)
      print -r -- 16
      ;;
    dp32)
      print -r -- 32
      ;;
    *)
      echo "Unsupported strategy for synthetic request generation: $1" >&2
      exit 1
      ;;
  esac
}

function strategy_dp_local_size() {
  case "$1" in
    dp1tp8dcp2)
      print -r -- 1
      ;;
    dp2tp8dcp2)
      print -r -- 2
      ;;
    dp4dcp8|dp4tp8dcp2|dp4tp8dcp2_ar|dp4tp8|dp4tp4)
      print -r -- 1
      ;;
    dp8dcp4|dp8tp4)
      print -r -- 2
      ;;
    dp16cp2|dp16tp2)
      print -r -- 4
      ;;
    dp32)
      print -r -- 8
      ;;
    *)
      echo "Unsupported strategy for DP-local-size lookup: $1" >&2
      exit 1
      ;;
  esac
}

while (( $# > 0 )); do
  case "$1" in
    --artifact-root)
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    --case-csv)
      CASE_CSV="$2"
      shift 2
      ;;
    --cluster)
      CLUSTER="$2"
      shift 2
      ;;
    --strategy)
      STRATEGY="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --lens-json)
      LENS_JSON="$2"
      shift 2
      ;;
    --prompt-len)
      PROMPT_LEN="$2"
      shift 2
      ;;
    --requests-per-dp)
      REQUESTS_PER_DP="$2"
      shift 2
      ;;
    --output-len)
      OUTPUT_LEN="$2"
      shift 2
      ;;
    --warmup-requests)
      WARMUP_REQUESTS="$2"
      shift 2
      ;;
    --max-requests)
      MAX_REQUESTS="$2"
      shift 2
      ;;
    --request-rate)
      REQUEST_RATE="$2"
      shift 2
      ;;
    --dispatch-policy)
      DISPATCH_POLICY="$2"
      shift 2
      ;;
    --routing-mode)
      ROUTING_MODE="$2"
      shift 2
      ;;
    --case-name)
      CASE_NAME="$2"
      shift 2
      ;;
    --max-num-seqs)
      MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --data-parallel-rpc-port)
      DATA_PARALLEL_RPC_PORT="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --cudagraph-capture-sizes)
      CUDAGRAPH_CAPTURE_SIZES="$2"
      shift 2
      ;;
    --profile-delay-iterations)
      PROFILE_DELAY_ITERATIONS="$2"
      shift 2
      ;;
    --async-scheduling)
      FORCE_NO_ASYNC_SCHEDULING=0
      shift 1
      ;;
    --pause-before-profile)
      PAUSE_BEFORE_PROFILE=1
      shift 1
      ;;
    --ignore-historical-skips)
      IGNORE_HISTORICAL_SKIPS=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "${LENS_JSON}" && -n "${PROMPT_LEN}" ]]; then
  echo "Pass either --lens-json or --prompt-len/--requests-per-dp, not both." >&2
  usage >&2
  exit 1
fi

if [[ -n "${LENS_JSON}" && -n "${REQUESTS_PER_DP}" ]]; then
  echo "--requests-per-dp cannot be combined with --lens-json." >&2
  usage >&2
  exit 1
fi

if [[ -z "${LENS_JSON}" && -z "${PROMPT_LEN}" ]]; then
  echo "Either --lens-json or --prompt-len must be provided." >&2
  usage >&2
  exit 1
fi

if [[ -n "${PROMPT_LEN}" && -z "${REQUESTS_PER_DP}" ]]; then
  echo "--requests-per-dp is required with --prompt-len." >&2
  usage >&2
  exit 1
fi

if [[ -z "${PROMPT_LEN}" && -n "${REQUESTS_PER_DP}" ]]; then
  echo "--prompt-len is required with --requests-per-dp." >&2
  usage >&2
  exit 1
fi

cd "${REPO_ROOT}"

EFFECTIVE_DISPATCH_POLICY="${DISPATCH_POLICY:-waiting_x4_plus_running}"
DISPATCH_TAG="dispatch_$(sanitize_tag "${EFFECTIVE_DISPATCH_POLICY}")"
ROUTING_TAG=""
if [[ "${ROUTING_MODE}" != "internal_dplb" ]]; then
  ROUTING_TAG="/routing_$(sanitize_tag "${ROUTING_MODE}")"
fi
PREPARE_INPUT_ARGS=()
if [[ -n "${LENS_JSON}" ]]; then
  INPUT_SOURCE_TAG="${${LENS_JSON:t}:r}"
  PREPARE_INPUT_ARGS=(
    --lens-json "${LENS_JSON}"
  )
else
  STRATEGY_DP_SIZE="$(strategy_total_dp "${STRATEGY}")"
  TOTAL_SYNTHETIC_REQUESTS="$(( REQUESTS_PER_DP * STRATEGY_DP_SIZE ))"
  INPUT_SOURCE_TAG="uniform_prompt${PROMPT_LEN}_perdp${REQUESTS_PER_DP}"
  PREPARE_INPUT_ARGS=(
    --uniform-prompt-len "${PROMPT_LEN}"
    --repeat-count "${TOTAL_SYNTHETIC_REQUESTS}"
  )
fi

PREPARED_DIR="${GENERATED_INPUT_ROOT}/${INPUT_SOURCE_TAG}/${STRATEGY}/${DISPATCH_TAG}${ROUTING_TAG}"
PREPARE_ARGS=(
  python3
  benchmarks/offline_dp_profile/prepare_custom_lens_case.py
  --base-case-csv "${CASE_CSV}"
  --output-dir "${PREPARED_DIR}"
  --output-len "${OUTPUT_LEN}"
  --cluster "${CLUSTER}"
  --strategy "${STRATEGY}"
  --routing-mode "${ROUTING_MODE}"
  "${PREPARE_INPUT_ARGS[@]}"
)
if [[ "${ROUTING_MODE}" == "explicit_rank_replay" ]]; then
  STRATEGY_DP_SIZE="$(strategy_total_dp "${STRATEGY}")"
  STRATEGY_DP_LOCAL_SIZE="$(strategy_dp_local_size "${STRATEGY}")"
  PREPARE_ARGS+=(
    --data-parallel-size "${STRATEGY_DP_SIZE}"
    --data-parallel-size-local "${STRATEGY_DP_LOCAL_SIZE}"
  )
  if [[ -n "${WARMUP_REQUESTS}" ]]; then
    PREPARE_ARGS+=(--warmup-short-rows "${WARMUP_REQUESTS}")
  fi
fi
if [[ -n "${MODEL}" ]]; then
  PREPARE_ARGS+=(--model "${MODEL}")
fi
if [[ -n "${WARMUP_REQUESTS}" ]]; then
  PREPARE_ARGS+=(--warmup-requests "${WARMUP_REQUESTS}")
fi
if [[ -n "${MAX_REQUESTS}" ]]; then
  PREPARE_ARGS+=(--max-requests "${MAX_REQUESTS}")
fi
if [[ -n "${REQUEST_RATE}" ]]; then
  PREPARE_ARGS+=(--request-rate "${REQUEST_RATE}")
fi
if [[ -n "${DISPATCH_POLICY}" ]]; then
  PREPARE_ARGS+=(--dispatch-policy "${DISPATCH_POLICY}")
fi
if [[ -n "${CASE_NAME}" ]]; then
  PREPARE_ARGS+=(--case-name "${CASE_NAME}")
fi
if [[ -n "${MAX_NUM_SEQS}" ]]; then
  PREPARE_ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  PREPARE_ARGS+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi
if [[ -n "${DATA_PARALLEL_RPC_PORT}" ]]; then
  PREPARE_ARGS+=(--data-parallel-rpc-port "${DATA_PARALLEL_RPC_PORT}")
fi
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  PREPARE_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi

"${PREPARE_ARGS[@]}"
RUN_CASE_CSV="${PREPARED_DIR}/custom_lens.casecsv"

CUDAGRAPH_CAPTURE_SIZE_VALUES=()
if [[ -n "${CUDAGRAPH_CAPTURE_SIZES}" ]]; then
  CUDAGRAPH_CAPTURE_SIZE_VALUES=("${(@s:,:)CUDAGRAPH_CAPTURE_SIZES}")
  for capture_size in "${CUDAGRAPH_CAPTURE_SIZE_VALUES[@]}"; do
    if ! [[ "${capture_size}" == <-> ]] || (( capture_size <= 0 )); then
      echo "--cudagraph-capture-sizes must contain positive integers, got: ${CUDAGRAPH_CAPTURE_SIZES}" >&2
      exit 1
    fi
  done
fi

FRONTEND_EXTRA_ARGS=(
  --frontend-extra-arg=--profile-after-warmup
  --frontend-extra-arg=--profiler-config.profiler
  --frontend-extra-arg=torch
  --frontend-extra-arg=--profiler-config.torch_profiler_dir
  --frontend-extra-arg='{benchmark_dir}/torch_profiler'
  --frontend-extra-arg=--profiler-config.ignore_frontend
  --frontend-extra-arg=true
  --frontend-extra-arg=--profiler-config.delay_iterations
  --frontend-extra-arg="${PROFILE_DELAY_ITERATIONS}"
  --frontend-extra-arg=--profiler-config.max_iterations
  --frontend-extra-arg=31
  --frontend-extra-arg=--profiler-config.wait_iterations
  --frontend-extra-arg=0
  --frontend-extra-arg=--profiler-config.warmup_iterations
  --frontend-extra-arg=0
)

if [[ "${FORCE_NO_ASYNC_SCHEDULING}" == "1" ]]; then
  FRONTEND_EXTRA_ARGS=(--frontend-extra-arg=--no-async-scheduling "${FRONTEND_EXTRA_ARGS[@]}")
  HEADLESS_ASYNC_ARG=--headless-extra-arg=--no-async-scheduling
else
  FRONTEND_EXTRA_ARGS=(--frontend-extra-arg=--async-scheduling "${FRONTEND_EXTRA_ARGS[@]}")
  HEADLESS_ASYNC_ARG=--headless-extra-arg=--async-scheduling
fi

if [[ "${PAUSE_BEFORE_PROFILE}" == "1" ]]; then
  FRONTEND_EXTRA_ARGS+=(--frontend-extra-arg=--pause-before-profile)
fi
if [[ "${ROUTING_MODE}" != "internal_dplb" ]]; then
  FRONTEND_EXTRA_ARGS+=(
    --frontend-extra-arg=--routing-mode
    "--frontend-extra-arg=${ROUTING_MODE}"
  )
fi
HEADLESS_EXTRA_ARGS=()
if (( ${#CUDAGRAPH_CAPTURE_SIZE_VALUES[@]} > 0 )); then
  FRONTEND_EXTRA_ARGS+=(--frontend-extra-arg=--cudagraph-capture-sizes)
  HEADLESS_EXTRA_ARGS+=(--headless-extra-arg=--cudagraph-capture-sizes)
  for capture_size in "${CUDAGRAPH_CAPTURE_SIZE_VALUES[@]}"; do
    FRONTEND_EXTRA_ARGS+=("--frontend-extra-arg=${capture_size}")
    HEADLESS_EXTRA_ARGS+=("--headless-extra-arg=${capture_size}")
  done
fi

RUNNER_ARGS=()
if [[ "${IGNORE_HISTORICAL_SKIPS}" == "1" ]]; then
  RUNNER_ARGS+=(--ignore-historical-skips)
fi

python3 benchmarks/manual_multinode_poisson_runner.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --case-csv "${RUN_CASE_CSV}" \
  "${RUNNER_ARGS[@]}" \
  "${FRONTEND_EXTRA_ARGS[@]}" \
  "${HEADLESS_ASYNC_ARG}" \
  "${HEADLESS_EXTRA_ARGS[@]}" \
  --headless-extra-arg=--profiler-config.profiler \
  --headless-extra-arg=torch \
  --headless-extra-arg=--profiler-config.torch_profiler_dir \
  --headless-extra-arg='{benchmark_dir}/torch_profiler' \
  --headless-extra-arg=--profiler-config.delay_iterations \
  --headless-extra-arg="${PROFILE_DELAY_ITERATIONS}" \
  --headless-extra-arg=--profiler-config.max_iterations \
  --headless-extra-arg=31 \
  --headless-extra-arg=--profiler-config.wait_iterations \
  --headless-extra-arg=0 \
  --headless-extra-arg=--profiler-config.warmup_iterations \
  --headless-extra-arg=0
