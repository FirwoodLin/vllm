#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_CLUSTER_ENV="${SCRIPT_DIR}/cluster.env"
FALLBACK_CLUSTER_ENV="${SCRIPT_DIR}/../bench_0180/cluster.2node.env"
CLUSTER_ENV="${CLUSTER_ENV:-${LOCAL_CLUSTER_ENV}}"
LAUNCHER="${SCRIPT_DIR}/../bench_0180/launch_4node_mp_manual.sh"

if [[ ! -f "${CLUSTER_ENV}" && -f "${FALLBACK_CLUSTER_ENV}" ]]; then
  CLUSTER_ENV="${FALLBACK_CLUSTER_ENV}"
fi

if [[ ! -f "${CLUSTER_ENV}" ]]; then
  echo "Missing CLUSTER_ENV=${CLUSTER_ENV}" >&2
  exit 2
fi

if [[ ! -f "${LAUNCHER}" ]]; then
  echo "Missing shared launcher: ${LAUNCHER}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${CLUSTER_ENV}"
set +a

usage() {
  cat <<'EOF'
Usage:
  start_2node_serve.sh \
    [--model MODEL] \
    [--master-addr NODE0_IP|auto] \
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

Examples:
  ./start_2node_serve.sh
  ./start_2node_serve.sh --model Qwen/Qwen2.5-7B-Instruct -- --max-model-len 1000000
  ./start_2node_serve.sh --tp 4 --dp 4 --data-parallel-size-local 2 -- --max-num-seqs 512
EOF

  printf '\nDefault CLUSTER_ENV: %s\n' "${CLUSTER_ENV}"
  printf 'Shared launcher: %s\n' "${LAUNCHER}"
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
DP_SIZE="${DP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-8}"
DCP_SIZE="${DCP_SIZE:-${TP_SIZE}}"
DP_LOCAL_SIZE="${DP_LOCAL_SIZE:-1}"
API_SERVER_COUNT="${API_SERVER_COUNT:-1}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
DISABLE_LOG_STATS="${DISABLE_LOG_STATS:-0}"
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

launcher_args=(
  --model "${MODEL}"
  --master-addr "${MASTER_ADDR}"
  --host "${HOST}"
  --port "${PORT}"
  --data-parallel-rpc-port "${RPC_PORT}"
  --log-dir "${LOG_DIR}"
  --dp "${DP_SIZE}"
  --tp "${TP_SIZE}"
  --dcp "${DCP_SIZE}"
  --data-parallel-size-local "${DP_LOCAL_SIZE}"
  --api-server-count "${API_SERVER_COUNT}"
)

if is_true "${ASYNC_SCHEDULING}"; then
  launcher_args+=(--enable-async-scheduling)
else
  launcher_args+=(--disable-async-scheduling)
fi

serve_args=(
  --load-format "${LOAD_FORMAT}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
)

if is_true "${ENABLE_EXPERT_PARALLEL}"; then
  serve_args+=(--enable-expert-parallel)
fi

if is_true "${DISABLE_LOG_STATS}"; then
  serve_args+=(--disable-log-stats)
fi

if ((${#EXTRA_VLLM_ARGS[@]})); then
  serve_args+=("${EXTRA_VLLM_ARGS[@]}")
fi

echo "start 2-node serve"
echo "  cluster_env: ${CLUSTER_ENV}"
echo "  model: ${MODEL}"
echo "  host: ${HOST}"
echo "  port: ${PORT}"
echo "  master_addr: ${MASTER_ADDR}"
echo "  rpc_port: ${RPC_PORT}"
echo "  topology: dp=${DP_SIZE} tp=${TP_SIZE} dcp=${DCP_SIZE} local_dp=${DP_LOCAL_SIZE}"
echo "  api_server_count: ${API_SERVER_COUNT}"
echo "  async_scheduling: ${ASYNC_SCHEDULING}"
echo "  load_format: ${LOAD_FORMAT}"
echo "  expert_parallel: ${ENABLE_EXPERT_PARALLEL}"
echo "  logs: ${LOG_DIR}"

exec env CLUSTER_ENV="${CLUSTER_ENV}" \
  bash "${LAUNCHER}" \
  "${launcher_args[@]}" \
  -- \
  "${serve_args[@]}"
