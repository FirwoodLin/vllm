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
OUTPUT_LEN=64
WARMUP_REQUESTS=""
MAX_REQUESTS=""
REQUEST_RATE=""
DISPATCH_POLICY=""
CASE_NAME=""
MAX_NUM_SEQS=""
GPU_MEMORY_UTILIZATION=""
DATA_PARALLEL_RPC_PORT=""
PROFILE_DELAY_ITERATIONS=33
PAUSE_BEFORE_PROFILE=0

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

Options:
  --artifact-root PATH
  --case-csv PATH
  --cluster NAME                    # default: 4node_h200
  --strategy NAME                   # default: dp32
  --model NAME
  --lens-json PATH                  # required
  --output-len N
  --warmup-requests N
  --max-requests N|csv_rows
  --request-rate FLOAT
  --dispatch-policy waiting_x4_plus_running|least_cache|least_batch
  --case-name NAME
  --max-num-seqs N
  --gpu-memory-utilization FLOAT
  --data-parallel-rpc-port N
  --profile-delay-iterations N      # default: 33
  --pause-before-profile
  -h, --help

Supported strategies:
  dp4dcp8, dp8dcp4, dp16cp2, dp32

Notes:
  - This entry only supports the --lens-json flow.
  - DCP strategies rely on benchmarks/manual_multinode_poisson_runner.py as the
    topology source of truth. For dcp>1 strategies, the runner injects
    --dcp-comm-backend a2a from the strategy table.
EOF
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
    --profile-delay-iterations)
      PROFILE_DELAY_ITERATIONS="$2"
      shift 2
      ;;
    --pause-before-profile)
      PAUSE_BEFORE_PROFILE=1
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

if [[ -z "${LENS_JSON}" ]]; then
  echo "--lens-json is required for start_multinode_offline_profile.sh" >&2
  usage >&2
  exit 1
fi

cd "${REPO_ROOT}"

EFFECTIVE_DISPATCH_POLICY="${DISPATCH_POLICY:-waiting_x4_plus_running}"
DISPATCH_TAG="dispatch_$(sanitize_tag "${EFFECTIVE_DISPATCH_POLICY}")"
LENS_STEM="${${LENS_JSON:t}:r}"
PREPARED_DIR="${GENERATED_INPUT_ROOT}/${LENS_STEM}/${STRATEGY}/${DISPATCH_TAG}"
PREPARE_ARGS=(
  python3
  benchmarks/offline_dp_profile/prepare_custom_lens_case.py
  --base-case-csv "${CASE_CSV}"
  --lens-json "${LENS_JSON}"
  --output-dir "${PREPARED_DIR}"
  --output-len "${OUTPUT_LEN}"
  --cluster "${CLUSTER}"
  --strategy "${STRATEGY}"
)
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

"${PREPARE_ARGS[@]}"
RUN_CASE_CSV="${PREPARED_DIR}/custom_lens.casecsv"

FRONTEND_EXTRA_ARGS=(
  --frontend-extra-arg=--profile-after-warmup
  --frontend-extra-arg=--no-async-scheduling
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

if [[ "${PAUSE_BEFORE_PROFILE}" == "1" ]]; then
  FRONTEND_EXTRA_ARGS+=(--frontend-extra-arg=--pause-before-profile)
fi

python3 benchmarks/manual_multinode_poisson_runner.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --case-csv "${RUN_CASE_CSV}" \
  "${FRONTEND_EXTRA_ARGS[@]}" \
  --headless-extra-arg=--no-async-scheduling \
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
