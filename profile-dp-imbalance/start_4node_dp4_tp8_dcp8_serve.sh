#!/usr/bin/env zsh

emulate -L zsh
setopt errexit nounset pipefail

# Usage:
#   Node 0:
#     MASTER_ADDR=10.0.0.1 NODE_RANK=0 SERVE_HOST=0.0.0.0 PORT=8000 \
#       zsh /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/profile-dp-imbalance/start_4node_dp4_tp8_dcp8_serve.sh
#
#   Node 1:
#     MASTER_ADDR=10.0.0.1 NODE_RANK=1 \
#       zsh /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/profile-dp-imbalance/start_4node_dp4_tp8_dcp8_serve.sh
#
#   Node 2:
#     MASTER_ADDR=10.0.0.1 NODE_RANK=2 \
#       zsh /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/profile-dp-imbalance/start_4node_dp4_tp8_dcp8_serve.sh
#
#   Node 3:
#     MASTER_ADDR=10.0.0.1 NODE_RANK=3 \
#       zsh /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/profile-dp-imbalance/start_4node_dp4_tp8_dcp8_serve.sh
#
# This configuration targets a 4-node / 32-GPU cluster with EP enabled.
# Supported topologies are:
# - dp=4, tp=8, dcp=8, local_dp=1
# - dp=8, tp=4, dcp=4, local_dp=2
# - dp=16, tp=2, dcp=2, local_dp=4
# - dp=32, tp=1, dcp=1, local_dp=8
#
# The default topology is dp=32 / tp=1 / dcp=1.
#
# Example overrides:
# - DP_SIZE=4 TP_SIZE=8 DCP_SIZE=8
# - DP_SIZE=8 TP_SIZE=4 DCP_SIZE=4
# - DP_SIZE=16 TP_SIZE=2 DCP_SIZE=2
# - DP_SIZE=32 TP_SIZE=1 DCP_SIZE=1 DP_LOCAL_SIZE=8
#
# Other runtime assumptions:
# - 4 nodes
# - EP all2all backend = deepep_low_latency
# - DCP backend = a2a
# - DecodeBenchConnector enabled
# - cudagraph mode = FULL_DECODE_ONLY
# - torch profiler capturing the middle 32 decode iterations for:
#     1 x 256k + 63 x 4k input, output length 64, per DP rank
#
# DecodeBenchConnector leaves exactly one context iteration before decode.
# With 64 decode iterations total, capturing the centered middle 32 means:
# - skip 1 context iteration
# - skip first 16 decode iterations
# - capture decode iterations 17..48
# So the default profiling window is:
# - delay_iterations = 18
# - max_iterations = 32

VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL="${MODEL:-/mnt/nvme1n1/ml_research/models/deepseek-v3-1024k}"
MASTER_ADDR="${MASTER_ADDR:-}"
RPC_PORT="${RPC_PORT:-13345}"
DP_MASTER_PORT="${DP_MASTER_PORT:-}"
NODE_RANK="${NODE_RANK:-}"
SERVE_HOST="${SERVE_HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

EXPECTED_NODE_COUNT=4

DP_SIZE="${DP_SIZE:-32}"
DP_LOCAL_SIZE="${DP_LOCAL_SIZE:-}"
TP_SIZE="${TP_SIZE:-1}"
DCP_SIZE="${DCP_SIZE:-1}"

SHORT_INPUT_LEN="${SHORT_INPUT_LEN:-1024}"
LONG_INPUT_LEN="${LONG_INPUT_LEN:-524288}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHMLA}"

PROFILE_ROOT="${PROFILE_ROOT:-/tmp/profile-dp-imbalance}"
PROFILE_WINDOW_ITERATIONS="${PROFILE_WINDOW_ITERATIONS:-32}"
PROFILE_CONTEXT_ITERATIONS="${PROFILE_CONTEXT_ITERATIONS:-1}"
PROFILE_DIR="${PROFILE_DIR:-${PROFILE_ROOT}/node${NODE_RANK}}"

KV_TRANSFER_CONFIG="${KV_TRANSFER_CONFIG:-{\"kv_connector\":\"DecodeBenchConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"fill_mean\":0.015,\"fill_std\":0.0}}}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-1 2 4 64}"

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "MASTER_ADDR must be set." >&2
  exit 2
fi

if [[ -z "${NODE_RANK}" ]]; then
  echo "NODE_RANK must be set to an integer in [0, $((EXPECTED_NODE_COUNT - 1))]." >&2
  exit 2
fi

if [[ "${NODE_RANK}" != <-> ]]; then
  echo "NODE_RANK must be an integer, got ${NODE_RANK}." >&2
  exit 2
fi

if (( NODE_RANK < 0 || NODE_RANK >= EXPECTED_NODE_COUNT )); then
  echo "NODE_RANK must be in [0, $((EXPECTED_NODE_COUNT - 1))], got ${NODE_RANK}." >&2
  exit 2
fi

case "${DP_SIZE}:${TP_SIZE}:${DCP_SIZE}" in
  "4:8:8")
    expected_dp_local_size=1
    ;;
  "8:4:4")
    expected_dp_local_size=2
    ;;
  "16:2:2")
    expected_dp_local_size=4
    ;;
  "32:1:1")
    expected_dp_local_size=8
    ;;
  *)
    echo "Unsupported topology: dp=${DP_SIZE} tp=${TP_SIZE} dcp=${DCP_SIZE}." >&2
    echo "Supported topologies: dp=4/tp=8/dcp=8, dp=8/tp=4/dcp=4, dp=16/tp=2/dcp=2, dp=32/tp=1/dcp=1." >&2
    exit 2
    ;;
esac

DP_LOCAL_SIZE="${DP_LOCAL_SIZE:-${expected_dp_local_size}}"

if [[ "${DP_LOCAL_SIZE}" != "${expected_dp_local_size}" ]]; then
  echo "For dp=${DP_SIZE} tp=${TP_SIZE} dcp=${DCP_SIZE}, DP_LOCAL_SIZE must be ${expected_dp_local_size}." >&2
  exit 2
fi

if [[ -n "${DP_MASTER_PORT}" && "${DP_MASTER_PORT}" != <-> ]]; then
  echo "DP_MASTER_PORT must be an integer, got ${DP_MASTER_PORT}." >&2
  exit 2
fi

if (( PROFILE_WINDOW_ITERATIONS > OUTPUT_LEN )); then
  echo "PROFILE_WINDOW_ITERATIONS=${PROFILE_WINDOW_ITERATIONS} cannot exceed OUTPUT_LEN=${OUTPUT_LEN}." >&2
  exit 2
fi

remaining_decode_iters=$((OUTPUT_LEN - PROFILE_WINDOW_ITERATIONS))
if (( remaining_decode_iters % 2 != 0 )); then
  echo "OUTPUT_LEN - PROFILE_WINDOW_ITERATIONS must be even to center the profile window." >&2
  echo "Got OUTPUT_LEN=${OUTPUT_LEN}, PROFILE_WINDOW_ITERATIONS=${PROFILE_WINDOW_ITERATIONS}." >&2
  exit 2
fi

skip_decode_before=$((remaining_decode_iters / 2))
PROFILE_DELAY_ITERATIONS="${PROFILE_DELAY_ITERATIONS:-$((PROFILE_CONTEXT_ITERATIONS + skip_decode_before + 1))}"
PROFILE_MAX_ITERATIONS="${PROFILE_MAX_ITERATIONS:-${PROFILE_WINDOW_ITERATIONS}}"
DP_START_RANK=$((NODE_RANK * DP_LOCAL_SIZE))

mkdir -p "${PROFILE_DIR}"

export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY="${VLLM_MOE_ROUTING_SIMULATION_STRATEGY:-uniform_random}"
export VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"

if [[ -n "${DP_MASTER_PORT}" ]]; then
  export VLLM_DP_MASTER_PORT="${DP_MASTER_PORT}"
fi

if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
  export GLOO_SOCKET_IFNAME
fi

if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
  export NCCL_SOCKET_IFNAME
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

typeset -a cudagraph_capture_sizes
cudagraph_capture_sizes=(${=CUDAGRAPH_CAPTURE_SIZES})

resolve_host() {
  python3 - "$1" <<'PY'
import socket
import sys

host = sys.argv[1]
try:
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
except Exception as exc:
    print(f"<unresolved:{exc}>")
    raise SystemExit(0)

addresses = []
for info in infos:
    addr = info[4][0]
    if addr not in addresses:
        addresses.append(addr)

print(",".join(addresses) if addresses else "<unresolved:no-addresses>")
PY
}

if [[ "${SERVE_HOST}" == "0.0.0.0" ]]; then
  serve_host_resolved="0.0.0.0 (all IPv4 interfaces)"
else
  serve_host_resolved="$(resolve_host "${SERVE_HOST}")"
fi
master_addr_resolved="$(resolve_host "${MASTER_ADDR}")"
dp_bind_hint="tcp://${MASTER_ADDR}:${DP_MASTER_PORT:-<auto>}"
serve_bind_hint="${SERVE_HOST}:${PORT}"

  # --no-async-scheduling
typeset -a cmd
cmd=(
  "${VLLM_BIN}" serve "${MODEL}"
  --tensor-parallel-size "${TP_SIZE}"
  --data-parallel-size "${DP_SIZE}"
  --enable-logging-step-timing-details
  --enable-graph-replay-timing
  --data-parallel-backend mp
  --data-parallel-size-local "${DP_LOCAL_SIZE}"
  --data-parallel-address "${MASTER_ADDR}"
  --data-parallel-rpc-port "${RPC_PORT}"
  --data-parallel-start-rank "${DP_START_RANK}"
  --enable-expert-parallel
  --all2all-backend deepep_low_latency
  --load-format dummy
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --attention-backend "${ATTENTION_BACKEND}"
  --no-enable-prefix-caching
  --kv-transfer-config "${KV_TRANSFER_CONFIG}"
  --compilation-config "${COMPILATION_CONFIG}"
  --profiler-config.profiler torch
  --profiler-config.torch_profiler_dir "${PROFILE_DIR}"
  --profiler-config.ignore_frontend true
  --profiler-config.delay_iterations "${PROFILE_DELAY_ITERATIONS}"
  --profiler-config.max_iterations "${PROFILE_MAX_ITERATIONS}"
  --profiler-config.wait_iterations 0
  --profiler-config.warmup_iterations 0
  --cudagraph-capture-sizes
  "${cudagraph_capture_sizes[@]}"
)

if (( DCP_SIZE > 1 )); then
  cmd+=(
    --decode-context-parallel-size "${DCP_SIZE}"
    --dcp-comm-backend a2a
  )
fi

if [[ "${NODE_RANK}" == "0" ]]; then
  cmd+=(
    --host "${SERVE_HOST}"
    --port "${PORT}"
    --api-server-count 1
  )
else
  cmd+=(--headless)
fi

echo "Starting vLLM serve"
echo "  model: ${MODEL}"
echo "  node_rank: ${NODE_RANK}"
echo "  dp_start_rank: ${DP_START_RANK}"
echo "  master_addr: ${MASTER_ADDR}"
echo "  master_addr_resolved: ${master_addr_resolved}"
echo "  dp_master_port: ${DP_MASTER_PORT:-auto}"
echo "  dp_bind_hint: ${dp_bind_hint}"
echo "  rpc_port: ${RPC_PORT}"
echo "  serve_host: ${SERVE_HOST}"
echo "  serve_host_resolved: ${serve_host_resolved}"
echo "  serve_bind: ${serve_bind_hint}"
echo "  port: ${PORT}"
echo "  topology: dp=${DP_SIZE} tp=${TP_SIZE} dcp=${DCP_SIZE} local_dp=${DP_LOCAL_SIZE}"
echo "  attention_backend: ${ATTENTION_BACKEND}"
echo "  log_stats: enabled"
echo "  request shape per DP: 63x${SHORT_INPUT_LEN} + 1x${LONG_INPUT_LEN}, output=${OUTPUT_LEN}"
echo "  profiler_dir: ${PROFILE_DIR}"
echo "  profiler_window: delay=${PROFILE_DELAY_ITERATIONS}, max=${PROFILE_MAX_ITERATIONS}"
echo "  moe_routing_simulation: ${VLLM_MOE_ROUTING_SIMULATION_STRATEGY}"

exec "${cmd[@]}"
