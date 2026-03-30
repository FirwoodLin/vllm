#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_ENV="${CLUSTER_ENV:-${SCRIPT_DIR}/cluster.env}"
LAUNCHER="${LAUNCHER:-${SCRIPT_DIR}/launch_4node_mp_manual.sh}"

if [[ ! -f "${CLUSTER_ENV}" ]]; then
  echo "Missing CLUSTER_ENV=${CLUSTER_ENV}" >&2
  exit 2
fi

if [[ ! -f "${LAUNCHER}" ]]; then
  echo "Missing launcher: ${LAUNCHER}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${CLUSTER_ENV}"
set +a

usage() {
  cat <<'EOF'
Usage:
  start_sweep_serve.sh \
    [--model MODEL] \
    [--master-addr NODE0_IP|auto] \
    [--master-addr-hostname HOSTNAME] \
    [--host HOST] \
    [--port PORT] \
    [--data-parallel-rpc-port PORT] \
    [--log-dir DIR] \
    [--dp N] \
    [--tp N] \
    [--dcp N] \
    [--data-parallel-size-local N] \
    [--api-server-count N] \
    [--load-format FORMAT] \
    [--enable-expert-parallel|--disable-expert-parallel] \
    [--gpu-memory-utilization FLOAT] \
    [--max-num-seqs N] \
    [--max-num-batched-tokens N] \
    [--disable-log-stats] \
    [--enable-async-scheduling|--disable-async-scheduling] \
    -- [extra vllm serve args]

This wrapper is intended to be used as the `vllm bench sweep --serve-cmd`.
It resolves MASTER_ADDR before delegating to launch_4node_mp_manual.sh so the
multi-node startup path matches the working bench_0180_2m flow.

Examples:
  ./start_sweep_serve.sh
  ./start_sweep_serve.sh --model Qwen/Qwen2.5-7B-Instruct -- --max-model-len 1000000
  ./start_sweep_serve.sh --tp 4 --dp 4 --data-parallel-size-local 2 -- --max-num-seqs 512
EOF

  printf '\nDefault CLUSTER_ENV: %s\n' "${CLUSTER_ENV}"
  printf 'Launcher: %s\n' "${LAUNCHER}"
  printf 'MASTER_ADDR=auto resolves via hostname/FQDN lookup in this wrapper.\n'
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

append_kv_arg() {
  local -n args_ref="$1"
  local flag="$2"
  local value="$3"

  if [[ -n "${value}" ]]; then
    args_ref+=("${flag}" "${value}")
  fi
}

pick_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo python
    return 0
  fi
  echo "python3 or python is required for MASTER_ADDR=auto" >&2
  exit 2
}

discover_master_addr_from_hostname() {
  local python_bin
  python_bin="$(pick_python)"

  "${python_bin}" - "${MASTER_ADDR_HOSTNAME:-}" <<'PY'
import socket
import sys

configured = sys.argv[1].strip()
queries = []
seen = set()

for candidate in (
    configured,
    socket.gethostname(),
    socket.getfqdn(),
):
    if not candidate or candidate in seen:
        continue
    seen.add(candidate)
    queries.append(candidate)

for query in queries:
    try:
        infos = socket.getaddrinfo(query, None, family=socket.AF_INET)
    except socket.gaierror:
        continue
    for info in infos:
        ip = info[4][0]
        if ip and not ip.startswith("127."):
            print(ip)
            raise SystemExit(0)

raise SystemExit(
    "Unable to resolve a non-loopback IPv4 from hostname/FQDN. "
    "Set MASTER_ADDR explicitly or set MASTER_ADDR_HOSTNAME."
)
PY
}

MODEL="${MODEL:-}"
MASTER_ADDR="${MASTER_ADDR:-auto}"
MASTER_ADDR_HOSTNAME="${MASTER_ADDR_HOSTNAME:-}"
HOST="${SERVE_HOST:-${SWEEP_HOST:-127.0.0.1}}"
PORT="${SERVE_PORT:-${SWEEP_PORT:-8000}}"
RPC_PORT="${DATA_PARALLEL_RPC_PORT:-13345}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
DP_SIZE="${DP_SIZE:-}"
TP_SIZE="${TP_SIZE:-}"
DCP_SIZE="${DCP_SIZE:-}"
DP_LOCAL_SIZE="${DP_LOCAL_SIZE:-}"
API_SERVER_COUNT="${API_SERVER_COUNT:-}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-}"
LOAD_FORMAT="${LOAD_FORMAT:-}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
DISABLE_LOG_STATS="${DISABLE_LOG_STATS:-}"
EXTRA_VLLM_ARGS=()

while (($#)); do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --master-addr|--data-parallel-address)
      MASTER_ADDR="$2"
      shift 2
      ;;
    --master-addr-hostname)
      MASTER_ADDR_HOSTNAME="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --data-parallel-rpc-port|--rpc-port)
      RPC_PORT="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --dp|--data-parallel-size)
      DP_SIZE="$2"
      shift 2
      ;;
    --tp|--tensor-parallel-size)
      TP_SIZE="$2"
      shift 2
      ;;
    --dcp|--decode-context-parallel-size)
      DCP_SIZE="$2"
      shift 2
      ;;
    --data-parallel-size-local)
      DP_LOCAL_SIZE="$2"
      shift 2
      ;;
    --api-server-count)
      API_SERVER_COUNT="$2"
      shift 2
      ;;
    --load-format)
      LOAD_FORMAT="$2"
      shift 2
      ;;
    --enable-expert-parallel)
      ENABLE_EXPERT_PARALLEL=1
      shift
      ;;
    --disable-expert-parallel)
      ENABLE_EXPERT_PARALLEL=0
      shift
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --max-num-seqs)
      MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --max-num-batched-tokens)
      MAX_NUM_BATCHED_TOKENS="$2"
      shift 2
      ;;
    --disable-log-stats)
      DISABLE_LOG_STATS=1
      shift
      ;;
    --enable-async-scheduling)
      ASYNC_SCHEDULING=1
      shift
      ;;
    --disable-async-scheduling)
      ASYNC_SCHEDULING=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_VLLM_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

: "${MODEL:?MODEL must be set in ${CLUSTER_ENV} or passed via --model}"

if [[ -z "${MASTER_ADDR}" || "${MASTER_ADDR}" == "auto" ]]; then
  MASTER_ADDR="$(discover_master_addr_from_hostname)"
fi

DISPLAY_DP_SIZE="${DP_SIZE:-4}"
DISPLAY_TP_SIZE="${TP_SIZE:-8}"
DISPLAY_DCP_SIZE="${DCP_SIZE:-${DISPLAY_TP_SIZE}}"
DISPLAY_DP_LOCAL_SIZE="${DP_LOCAL_SIZE:-1}"
DISPLAY_API_SERVER_COUNT="${API_SERVER_COUNT:-1}"
DISPLAY_ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
DISPLAY_LOAD_FORMAT="${LOAD_FORMAT:-<vllm default>}"
DISPLAY_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-<vllm default>}"
DISPLAY_DISABLE_LOG_STATS="${DISABLE_LOG_STATS:-0}"

launcher_args=(
  --model "${MODEL}"
  --master-addr "${MASTER_ADDR}"
  --host "${HOST}"
  --port "${PORT}"
  --data-parallel-rpc-port "${RPC_PORT}"
  --log-dir "${LOG_DIR}"
)
append_kv_arg launcher_args --dp "${DP_SIZE}"
append_kv_arg launcher_args --tp "${TP_SIZE}"
append_kv_arg launcher_args --dcp "${DCP_SIZE}"
append_kv_arg launcher_args --data-parallel-size-local "${DP_LOCAL_SIZE}"
append_kv_arg launcher_args --api-server-count "${API_SERVER_COUNT}"

if [[ -n "${ASYNC_SCHEDULING}" ]]; then
  if is_true "${ASYNC_SCHEDULING}"; then
    launcher_args+=(--enable-async-scheduling)
  else
    launcher_args+=(--disable-async-scheduling)
  fi
fi

serve_args=()
append_kv_arg serve_args --load-format "${LOAD_FORMAT}"
append_kv_arg serve_args --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
append_kv_arg serve_args --max-num-seqs "${MAX_NUM_SEQS}"
append_kv_arg serve_args --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"

if [[ -n "${ENABLE_EXPERT_PARALLEL}" ]] && is_true "${ENABLE_EXPERT_PARALLEL}"; then
  serve_args+=(--enable-expert-parallel)
fi

if [[ -n "${DISABLE_LOG_STATS}" ]] && is_true "${DISABLE_LOG_STATS}"; then
  serve_args+=(--disable-log-stats)
fi

if ((${#EXTRA_VLLM_ARGS[@]})); then
  serve_args+=("${EXTRA_VLLM_ARGS[@]}")
fi

echo "start sweep serve"
echo "  cluster_env: ${CLUSTER_ENV}"
echo "  model: ${MODEL}"
echo "  host: ${HOST}"
echo "  port: ${PORT}"
echo "  master_addr: ${MASTER_ADDR}"
echo "  rpc_port: ${RPC_PORT}"
echo "  topology: dp=${DISPLAY_DP_SIZE} tp=${DISPLAY_TP_SIZE} dcp=${DISPLAY_DCP_SIZE} local_dp=${DISPLAY_DP_LOCAL_SIZE}"
echo "  api_server_count: ${DISPLAY_API_SERVER_COUNT}"
echo "  async_scheduling: ${DISPLAY_ASYNC_SCHEDULING}"
echo "  load_format: ${DISPLAY_LOAD_FORMAT}"
echo "  expert_parallel: ${DISPLAY_EXPERT_PARALLEL}"
echo "  disable_log_stats: ${DISPLAY_DISABLE_LOG_STATS}"
echo "  logs: ${LOG_DIR}"
if ((${#EXTRA_VLLM_ARGS[@]})); then
  printf '  extra_vllm_args:'
  for arg in "${EXTRA_VLLM_ARGS[@]}"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
fi

exec env CLUSTER_ENV="${CLUSTER_ENV}" \
  bash "${LAUNCHER}" \
  "${launcher_args[@]}" \
  -- \
  "${serve_args[@]}"
