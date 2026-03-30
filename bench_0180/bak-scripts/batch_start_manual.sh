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

usage() {
  cat <<'EOF'
Usage:
  batch_start_manual.sh \
    --strategies dp4,dp8 \
    --datasets issue01_halfhalf,short_random \
    --rates 30,40 \
    [--duration-sec 600] \
    [--num-prompts N] \
    [--max-num-seqs N] \
    [--gpu-memory-utilization 0.85] \
    [--max-model-len 1000000] \
    [--max-num-batched-tokens 16384] \
    [--kv-transfer-config JSON] \
    [--decodebench-connector] \
    [--cudagraph-mode MODE] \
    [--decodebench-fill-mean 0.015] \
    [--decodebench-fill-std 0.0] \
    [--api-server-count 1] \
    [--timestamp-results|--no-timestamp-results] \
    [--dry-run]

Dataset keys:
  issue01_halfhalf
  issue01_random
  issue05_halfhalf
  issue05_random
  short_halfhalf
  short_random
EOF

  printf '\nStrategy keys for the current cluster layout (%s node(s) x %s GPU(s)/node):\n' \
    "${NODE_COUNT}" "${GPUS_PER_NODE}"
  supported_strategy_lines

  cat <<'EOF'
Examples by topology:
  4 nodes x 8 GPUs -> dp4,dp8,dp16,dp32
  2 nodes x 8 GPUs -> dp2,dp4,dp8,dp16

Examples:
  ./batch_start_manual.sh --strategies dp4 --datasets issue01_halfhalf --rates 30
  ./batch_start_manual.sh --strategies dp4,dp8 --datasets issue01_halfhalf,issue05_random --rates 20,30
  ./batch_start_manual.sh --strategies dp2 --datasets issue01_halfhalf --rates 30 --decodebench-connector
  ./batch_start_manual.sh --strategies dp2 --datasets issue01_halfhalf --rates 30 --decodebench-connector --cudagraph-mode FULL_DECODE_ONLY
EOF
}

shell_join() {
  local parts=()
  local arg
  for arg in "$@"; do
    parts+=("$(printf '%q' "${arg}")")
  done
  local IFS=' '
  printf '%s' "${parts[*]}"
}

split_csv_arg() {
  local input="$1"
  local -n out_ref="$2"
  local token
  local IFS=','
  read -r -a out_ref <<<"${input}"
  for token in "${out_ref[@]}"; do
    if [[ -z "${token}" ]]; then
      echo "Empty entry found in comma-separated list: ${input}" >&2
      exit 2
    fi
  done
}

normalize_rate_tag() {
  local rate="$1"
  rate="${rate//./p}"
  printf '%s' "${rate}"
}

build_decodebench_kv_transfer_config() {
  local fill_mean="$1"
  local fill_std="$2"

  printf \
    '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":%s,"fill_std":%s}}' \
    "${fill_mean}" \
    "${fill_std}"
}

kv_transfer_config_tag() {
  local config="$1"

  if [[ "${config}" == *"DecodeBenchConnector"* ]]; then
    printf '__decodebench'
    return
  fi

  printf '__kvconn'
}

normalize_cudagraph_mode() {
  local mode="$1"

  mode="${mode^^}"
  case "${mode}" in
    NONE|PIECEWISE|FULL|FULL_DECODE_ONLY|FULL_AND_PIECEWISE)
      printf '%s' "${mode}"
      ;;
    *)
      echo "Unsupported cudagraph mode: ${mode}" >&2
      echo "Supported modes: NONE, PIECEWISE, FULL, FULL_DECODE_ONLY, FULL_AND_PIECEWISE" >&2
      exit 2
      ;;
  esac
}

build_compilation_config() {
  local cudagraph_mode="$1"

  if [[ -z "${cudagraph_mode}" ]]; then
    return 0
  fi

  printf '{"cudagraph_mode":"%s"}' "${cudagraph_mode}"
}

compilation_config_tag() {
  local config="$1"
  local mode="$2"

  if [[ -z "${config}" ]]; then
    return 0
  fi

  if [[ -n "${mode}" ]]; then
    printf '__cg%s' "${mode,,}"
    return 0
  fi

  printf '__compcfg'
}

load_remote_nodes() {
  REMOTE_NODES=()

  if [[ -n "${REMOTE_NODE_SSHS:-}" ]]; then
    split_csv_arg "${REMOTE_NODE_SSHS}" REMOTE_NODES
    return
  fi

  local legacy_node
  for legacy_node in "${NODE1_SSH:-}" "${NODE2_SSH:-}" "${NODE3_SSH:-}"; do
    if [[ -n "${legacy_node}" ]]; then
      REMOTE_NODES+=("${legacy_node}")
    fi
  done
}

default_max_num_seqs_for_tp() {
  local tp_size="$1"
  case "${tp_size}" in
    8) echo 512 ;;
    4) echo 512 ;;
    2) echo 256 ;;
    1) echo 128 ;;
    *)
      echo "Unsupported TP size for manual batch benchmark: ${tp_size}" >&2
      exit 2
      ;;
  esac
}

supported_strategy_lines() {
  local total_gpus=$((NODE_COUNT * GPUS_PER_NODE))
  local tp_size
  local dp_size
  local dp_local_size
  local default_max_num_seqs

  for tp_size in 8 4 2 1; do
    if ((GPUS_PER_NODE % tp_size != 0)); then
      continue
    fi
    dp_size=$((total_gpus / tp_size))
    dp_local_size=$((GPUS_PER_NODE / tp_size))
    default_max_num_seqs="$(default_max_num_seqs_for_tp "${tp_size}")"
    printf '  dp%s -> dp=%s tp=%s dcp=%s local=%s default max_num_seqs=%s\n' \
      "${dp_size}" "${dp_size}" "${tp_size}" "${tp_size}" "${dp_local_size}" "${default_max_num_seqs}"
  done
}

supported_strategy_keys_csv() {
  local total_gpus=$((NODE_COUNT * GPUS_PER_NODE))
  local tp_size
  local keys=()

  for tp_size in 8 4 2 1; do
    if ((GPUS_PER_NODE % tp_size == 0)); then
      keys+=("dp$((total_gpus / tp_size))")
    fi
  done

  local IFS=','
  printf '%s' "${keys[*]}"
}

strategy_config() {
  local key="$1"
  local total_gpus=$((NODE_COUNT * GPUS_PER_NODE))
  local dp_size
  local tp_size
  local dp_local_size
  local default_max_num_seqs

  if [[ ! "${key}" =~ ^dp[0-9]+$ ]]; then
    echo "Unknown strategy: ${key}" >&2
    echo "Supported strategies for ${NODE_COUNT} node(s) x ${GPUS_PER_NODE} GPU(s)/node: $(supported_strategy_keys_csv)" >&2
    exit 2
  fi

  dp_size="${key#dp}"
  if ((dp_size < 1 || total_gpus % dp_size != 0)); then
    echo "Unsupported strategy ${key} for ${NODE_COUNT} node(s) x ${GPUS_PER_NODE} GPU(s)/node." >&2
    echo "Supported strategies: $(supported_strategy_keys_csv)" >&2
    exit 2
  fi

  tp_size=$((total_gpus / dp_size))
  if ((GPUS_PER_NODE % tp_size != 0)); then
    echo "Strategy ${key} would require tp=${tp_size}, which does not fit evenly into ${GPUS_PER_NODE} GPU(s)/node." >&2
    echo "Supported strategies: $(supported_strategy_keys_csv)" >&2
    exit 2
  fi

  default_max_num_seqs="$(default_max_num_seqs_for_tp "${tp_size}")"
  dp_local_size=$((GPUS_PER_NODE / tp_size))
  echo "${dp_size} ${tp_size} ${tp_size} ${dp_local_size} ${default_max_num_seqs}"
}

dataset_config() {
  local key="$1"
  case "${key}" in
    issue01_halfhalf)
      echo "issue01_halfhalf /mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-halfhalf_geminiissue_r0.01_n60000_60k.csv"
      ;;
    issue01_random)
      echo "issue01_random /mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv"
      ;;
    issue05_random)
      echo "issue05_random /mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv"
      ;;
    issue05_halfhalf)
      echo "issue05_halfhalf /mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-halfhalf_geminiissue_r0.05_n60000_60k.csv"
      ;;
    short_halfhalf)
      echo "short_halfhalf /mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o/sharegpt4o-mixed-half-half-60k.csv"
      ;;
    short_random)
      echo "short_random /mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o/sharegpt4o-mixed-random-60k.csv"
      ;;
    *)
      echo "Unknown dataset: ${key}" >&2
      exit 2
      ;;
  esac
}

build_num_prompts() {
  local rate="$1"
  local duration="$2"
  awk -v r="${rate}" -v d="${duration}" 'BEGIN { printf "%d", int(r * d) }'
}

is_true() {
  case "${1,,}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
if ! [[ "${GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || ((GPUS_PER_NODE < 1)); then
  echo "GPUS_PER_NODE must be a positive integer" >&2
  exit 2
fi

load_remote_nodes
NODE_COUNT=$((1 + ${#REMOTE_NODES[@]}))
EXPERIMENT_TOPOLOGY_TAG=""
if ((NODE_COUNT != 4 || GPUS_PER_NODE != 8)); then
  EXPERIMENT_TOPOLOGY_TAG="__n${NODE_COUNT}g${GPUS_PER_NODE}"
fi

: "${MODEL:?MODEL must be set in cluster.env}"

RESULT_ROOT="${RESULT_ROOT:-${OUTPUT_DIR:-${SCRIPT_DIR}/results_manual}}"
MANUAL_LOG_ROOT="${MANUAL_LOG_ROOT:-${SCRIPT_DIR}/logs_manual}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
KV_TRANSFER_CONFIG="${KV_TRANSFER_CONFIG:-}"
ENABLE_DECODEBENCH_CONNECTOR="${ENABLE_DECODEBENCH_CONNECTOR:-0}"
DECODE_BENCH_FILL_MEAN="${DECODE_BENCH_FILL_MEAN:-0.015}"
DECODE_BENCH_FILL_STD="${DECODE_BENCH_FILL_STD:-0.0}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-}"
DEFAULT_DURATION_SEC="${DEFAULT_DURATION_SEC:-600}"
MANUAL_NUM_RUNS="${MANUAL_NUM_RUNS:-1}"
MANUAL_API_SERVER_COUNT="${MANUAL_API_SERVER_COUNT:-1}"
TIMESTAMP_RESULTS="${TIMESTAMP_RESULTS:-1}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
SERVE_BIND_HOST="${SERVE_HOST:-${SWEEP_HOST:-127.0.0.1}}"
SERVE_BIND_PORT="${SERVE_PORT:-${SWEEP_PORT:-8000}}"
BENCH_TARGET_HOST="${BENCH_HOST:-${SWEEP_HOST:-${SERVE_BIND_HOST}}}"
BENCH_TARGET_PORT="${BENCH_PORT:-${SWEEP_PORT:-${SERVE_BIND_PORT}}}"

STRATEGIES=()
DATASETS=()
RATES=()
NUM_PROMPTS_OVERRIDE=""
MAX_NUM_SEQS_OVERRIDE=""
DURATION_SEC="${DEFAULT_DURATION_SEC}"
API_SERVER_COUNT="${MANUAL_API_SERVER_COUNT}"
DRY_RUN=0

while (($#)); do
  case "$1" in
    --strategies|--strategy)
      split_csv_arg "$2" STRATEGIES
      shift 2
      ;;
    --datasets|--dataset)
      split_csv_arg "$2" DATASETS
      shift 2
      ;;
    --rates|--rate)
      split_csv_arg "$2" RATES
      shift 2
      ;;
    --duration-sec)
      DURATION_SEC="$2"
      shift 2
      ;;
    --num-prompts)
      NUM_PROMPTS_OVERRIDE="$2"
      shift 2
      ;;
    --max-num-seqs)
      MAX_NUM_SEQS_OVERRIDE="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --max-num-batched-tokens)
      MAX_NUM_BATCHED_TOKENS="$2"
      shift 2
      ;;
    --kv-transfer-config)
      KV_TRANSFER_CONFIG="$2"
      shift 2
      ;;
    --decodebench-connector|--enable-decodebench-connector)
      ENABLE_DECODEBENCH_CONNECTOR=1
      shift
      ;;
    --cudagraph-mode)
      CUDAGRAPH_MODE="$(normalize_cudagraph_mode "$2")"
      shift 2
      ;;
    --decodebench-fill-mean)
      DECODE_BENCH_FILL_MEAN="$2"
      shift 2
      ;;
    --decodebench-fill-std)
      DECODE_BENCH_FILL_STD="$2"
      shift 2
      ;;
    --api-server-count)
      API_SERVER_COUNT="$2"
      shift 2
      ;;
    --timestamp-results)
      TIMESTAMP_RESULTS=1
      shift
      ;;
    --no-timestamp-results)
      TIMESTAMP_RESULTS=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ((${#STRATEGIES[@]} == 0 || ${#DATASETS[@]} == 0 || ${#RATES[@]} == 0)); then
  usage >&2
  exit 2
fi

if [[ -n "${CUDAGRAPH_MODE}" ]]; then
  CUDAGRAPH_MODE="$(normalize_cudagraph_mode "${CUDAGRAPH_MODE}")"
fi

if [[ -n "${COMPILATION_CONFIG}" ]] && [[ -n "${CUDAGRAPH_MODE}" ]]; then
  echo "Use either COMPILATION_CONFIG or --cudagraph-mode/CUDAGRAPH_MODE, not both." >&2
  exit 2
fi

if [[ -z "${KV_TRANSFER_CONFIG}" ]] \
  && [[ "${ENABLE_DECODEBENCH_CONNECTOR}" == "1" ]]; then
  KV_TRANSFER_CONFIG="$(build_decodebench_kv_transfer_config \
    "${DECODE_BENCH_FILL_MEAN}" \
    "${DECODE_BENCH_FILL_STD}")"
fi

if [[ -z "${COMPILATION_CONFIG}" ]] && [[ -n "${CUDAGRAPH_MODE}" ]]; then
  COMPILATION_CONFIG="$(build_compilation_config "${CUDAGRAPH_MODE}")"
fi

KV_TRANSFER_TAG=""
if [[ -n "${KV_TRANSFER_CONFIG}" ]]; then
  KV_TRANSFER_TAG="$(kv_transfer_config_tag "${KV_TRANSFER_CONFIG}")"
fi

COMPILATION_CONFIG_TAG=""
if [[ -n "${COMPILATION_CONFIG}" ]]; then
  COMPILATION_CONFIG_TAG="$(compilation_config_tag "${COMPILATION_CONFIG}" "${CUDAGRAPH_MODE}")"
fi

mkdir -p "${RESULT_ROOT}" "${MANUAL_LOG_ROOT}"

tmp_files=()
cleanup() {
  local file
  for file in "${tmp_files[@]:-}"; do
    rm -f "${file}"
  done
}
trap cleanup EXIT

for strategy_key in "${STRATEGIES[@]}"; do
  read -r dp_size tp_size dcp_size dp_local_size default_max_num_seqs <<<"$(strategy_config "${strategy_key}")"
  max_num_seqs="${MAX_NUM_SEQS_OVERRIDE:-${default_max_num_seqs}}"

  for dataset_key in "${DATASETS[@]}"; do
    read -r dataset_slug dataset_path <<<"$(dataset_config "${dataset_key}")"
    if [[ ! -f "${dataset_path}" ]]; then
      echo "Dataset file not found: ${dataset_path}" >&2
      exit 2
    fi

    for rate in "${RATES[@]}"; do
      num_prompts="${NUM_PROMPTS_OVERRIDE:-$(build_num_prompts "${rate}" "${DURATION_SEC}")}"
      rate_tag="$(normalize_rate_tag "${rate}")"
      experiment_name="${dataset_slug}__${strategy_key}${EXPERIMENT_TOPOLOGY_TAG}${KV_TRANSFER_TAG}${COMPILATION_CONFIG_TAG}__rate${rate_tag}__bs${max_num_seqs}"
      sweep_output_dir="${RESULT_ROOT}"
      sweep_experiment_name="${experiment_name}"
      combo_log_dir="${MANUAL_LOG_ROOT}/${experiment_name}"
      if is_true "${TIMESTAMP_RESULTS}"; then
        run_timestamp="$(date +"%Y%m%d_%H%M%S")"
        sweep_output_dir="${RESULT_ROOT}/${experiment_name}"
        sweep_experiment_name="${run_timestamp}"
        combo_log_dir="${combo_log_dir}/${run_timestamp}"
      fi
      mkdir -p "${combo_log_dir}"

      serve_params_file="$(mktemp /tmp/bench_0180_serve_params.XXXXXX.json)"
      tmp_files+=("${serve_params_file}")
      cat >"${serve_params_file}" <<EOF
[
  {
    "_benchmark_name": "${experiment_name}",
    "gpu_memory_utilization": ${GPU_MEMORY_UTILIZATION},
    "max_num_seqs": ${max_num_seqs},
    "max_num_batched_tokens": ${MAX_NUM_BATCHED_TOKENS},
    "max_model_len": ${MAX_MODEL_LEN}
EOF
      cat >>"${serve_params_file}" <<EOF
  }
]
EOF

      serve_cmd_parts=(
        "${SCRIPT_DIR}/start_sweep_serve.sh"
        --model "${MODEL}"
        --master-addr "${MASTER_ADDR:-auto}"
        --host "${SERVE_BIND_HOST}"
        --port "${SERVE_BIND_PORT}"
        --data-parallel-rpc-port "${DATA_PARALLEL_RPC_PORT:-13345}"
        --log-dir "${combo_log_dir}"
        --dp "${dp_size}"
        --tp "${tp_size}"
        --dcp "${dcp_size}"
        --data-parallel-size-local "${dp_local_size}"
        --api-server-count "${API_SERVER_COUNT}"
        --
      )
      if [[ -n "${KV_TRANSFER_CONFIG}" ]]; then
        serve_cmd_parts+=(--kv-transfer-config "${KV_TRANSFER_CONFIG}")
      fi
      if [[ -n "${COMPILATION_CONFIG}" ]]; then
        serve_cmd_parts+=(--compilation-config "${COMPILATION_CONFIG}")
      fi
      serve_cmd="$(shell_join "${serve_cmd_parts[@]}")"

      bench_cmd_parts=(
        vllm bench serve
        --model "${MODEL}"
        --backend vllm
        --host "${BENCH_TARGET_HOST}"
        --port "${BENCH_TARGET_PORT}"
        --endpoint /v1/completions
        --dataset-name random
        --random-csv-path "${dataset_path}"
        --num-prompts "${num_prompts}"
        --request-rate "${rate}"
        --ignore-eos
        --percentile-metrics ttft,tpot,itl,e2el
        --metric-percentiles 50,90,95,99
        --goodput tpot:100
        --save-detailed
        --no-save-generated-texts
      )
      bench_cmd="$(shell_join "${bench_cmd_parts[@]}")"

      after_bench_cmd_parts=(
        "${SCRIPT_DIR}/reset_sweep_caches.sh"
        --host "${BENCH_TARGET_HOST}"
        --port "${BENCH_TARGET_PORT}"
      )
      after_bench_cmd="$(shell_join "${after_bench_cmd_parts[@]}")"

      echo "================================================================"
      echo "dataset: ${dataset_key}"
      echo "dataset_path: ${dataset_path}"
      echo "strategy: ${strategy_key} (dp=${dp_size}, tp=${tp_size}, dcp=${dcp_size}, local=${dp_local_size})"
      echo "cluster_layout: ${NODE_COUNT} node(s) x ${GPUS_PER_NODE} GPU(s)/node"
      echo "rate: ${rate}"
      echo "num_prompts: ${num_prompts}"
      echo "max_num_seqs: ${max_num_seqs}"
      if [[ -n "${KV_TRANSFER_CONFIG}" ]]; then
        echo "kv_transfer_config: ${KV_TRANSFER_CONFIG}"
      else
        echo "kv_transfer_config: <disabled>"
      fi
      if [[ -n "${COMPILATION_CONFIG}" ]]; then
        echo "compilation_config: ${COMPILATION_CONFIG}"
      else
        echo "compilation_config: <disabled>"
      fi
      echo "logs: ${combo_log_dir}"
      echo "results_root: ${RESULT_ROOT}"
      echo "result_parent: ${sweep_output_dir}"
      echo "result_experiment_name: ${sweep_experiment_name}"
      echo "serve_bind: ${SERVE_BIND_HOST}:${SERVE_BIND_PORT}"
      echo "bench_target: ${BENCH_TARGET_HOST}:${BENCH_TARGET_PORT}"

      if ((DRY_RUN)); then
        echo "[dry-run] serve_params: ${serve_params_file}"
        echo "[dry-run] serve_cmd: ${serve_cmd}"
        echo "[dry-run] bench_cmd: ${bench_cmd}"
        echo "[dry-run] after_bench_cmd: ${after_bench_cmd}"
        continue
      fi

      sweep_cmd=(
        vllm bench sweep serve
        --serve-cmd "${serve_cmd}"
        --bench-cmd "${bench_cmd}"
        --after-bench-cmd "${after_bench_cmd}"
        --serve-params "${serve_params_file}"
        --output-dir "${sweep_output_dir}"
        --num-runs "${MANUAL_NUM_RUNS}"
        --server-ready-timeout "${SERVER_READY_TIMEOUT}"
      )
      if [[ -n "${sweep_experiment_name}" ]]; then
        sweep_cmd+=(--experiment-name "${sweep_experiment_name}")
      fi

      "${sweep_cmd[@]}"
    done
  done
done
