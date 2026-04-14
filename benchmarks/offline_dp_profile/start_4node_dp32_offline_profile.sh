#!/usr/bin/env zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench/profile_dp32}"
CASE_CSV="${CASE_CSV:-${SCRIPT_DIR}/profile_cases/deepseek_issue01_dp32_profile.casecsv}"
LENS_JSON=""
OUTPUT_LEN=64
WARMUP_REQUESTS=""
MAX_REQUESTS=""
REQUEST_RATE=""
DISPATCH_POLICY=""
CASE_NAME=""
PROFILE_DELAY_ITERATIONS=33
PAUSE_BEFORE_PROFILE=0
IGNORE_HISTORICAL_SKIPS=0

function usage() {
  cat <<'EOF'
Usage:
  start_4node_dp32_offline_profile.sh [options]

Options:
  --artifact-root PATH
  --case-csv PATH
  --lens-json PATH
  --output-len N
  --warmup-requests N
  --max-requests N|csv_rows
  --request-rate FLOAT
  --dispatch-policy waiting_x4_plus_running|least_cache|least_batch
  --case-name NAME
  --profile-delay-iterations N   # default: 33
  --pause-before-profile
  --ignore-historical-skips
  -h, --help

When --lens-json is set, the script first converts the nested JSON input lengths
into an offline harness CSV (prompt_len,output_len) and derives a temporary
casecsv from the base case.
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
    --profile-delay-iterations)
      PROFILE_DELAY_ITERATIONS="$2"
      shift 2
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

cd "${REPO_ROOT}"

RUN_CASE_CSV="${CASE_CSV}"

if [[ -n "${LENS_JSON}" ]]; then
  PREPARED_DIR="${REPO_ROOT}/benchmarks/offline_dp_profile/generated_inputs/${${LENS_JSON:t}:r}"
  PREPARE_ARGS=(
    python3
    benchmarks/offline_dp_profile/prepare_custom_lens_case.py
    --base-case-csv "${CASE_CSV}"
    --lens-json "${LENS_JSON}"
    --output-dir "${PREPARED_DIR}"
    --output-len "${OUTPUT_LEN}"
  )
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

  "${PREPARE_ARGS[@]}"
  RUN_CASE_CSV="${PREPARED_DIR}/custom_lens.casecsv"
fi

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

RUNNER_ARGS=()
if [[ "${IGNORE_HISTORICAL_SKIPS}" == "1" ]]; then
  RUNNER_ARGS+=(--ignore-historical-skips)
fi

python3 benchmarks/manual_multinode_poisson_runner.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --case-csv "${RUN_CASE_CSV}" \
  "${RUNNER_ARGS[@]}" \
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
