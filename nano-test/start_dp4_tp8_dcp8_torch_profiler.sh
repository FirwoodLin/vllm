#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   Node 0:
#     MASTER_ADDR=10.0.0.1 DP_NODE_RANK=0 bash nano-test/start_dp4_tp8_dcp8_torch_profiler.sh
#   Node 1:
#     MASTER_ADDR=10.0.0.1 DP_NODE_RANK=1 bash nano-test/start_dp4_tp8_dcp8_torch_profiler.sh
#   Node 2:
#     MASTER_ADDR=10.0.0.1 DP_NODE_RANK=2 bash nano-test/start_dp4_tp8_dcp8_torch_profiler.sh
#   Node 3:
#     MASTER_ADDR=10.0.0.1 DP_NODE_RANK=3 bash nano-test/start_dp4_tp8_dcp8_torch_profiler.sh
#
# This script targets:
# - 4 DP ranks total
# - 8 TP per DP rank
# - 8 DCP per DP rank
# - per global DP rank: 127 x 2k-input/16-output + 1 x 256k-input/16-output
# - torch profiler capturing the middle 8 worker iterations of the profiled run
#
# The profiled window is configured with:
# - delay_iterations=6
# - max_iterations=8
#
# For a single generate() with 16 decode tokens, this skips:
# - 1 prefill iteration
# - the first 4 decode iterations
# and captures decode iterations 5-12.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_PARALLEL_SCRIPT="${REPO_ROOT}/examples/offline_inference/data_parallel.py"

export MODEL="${MODEL:-/mnt/nvme1n1/ml_research/models/deepseek-v3-1024k}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export DP_NODE_RANK="${DP_NODE_RANK:-0}"
export DP_NUM_NODES="${DP_NUM_NODES:-4}"
export DP_SIZE="${DP_SIZE:-4}"
export TP_SIZE="${TP_SIZE:-8}"
export DCP_SIZE="${DCP_SIZE:-8}"
export DATA_PARALLEL_SIZE_LOCAL="${DATA_PARALLEL_SIZE_LOCAL:-1}"
export NUM_SHORT_REQUESTS="${NUM_SHORT_REQUESTS:-127}"
export SHORT_INPUT_LEN="${SHORT_INPUT_LEN:-2000}"
export LONG_INPUT_LEN="${LONG_INPUT_LEN:-256000}"
export OUTPUT_LEN="${OUTPUT_LEN:-16}"
export REQUEST_TOKEN_SEED="${REQUEST_TOKEN_SEED:-20260325}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHMLA}"
export DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-a2a}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY="${VLLM_MOE_ROUTING_SIMULATION_STRATEGY:-uniform_random}"
export WARMUP_ITERS="${WARMUP_ITERS:-5}"
export PROFILE_DELAY_ITERATIONS="${PROFILE_DELAY_ITERATIONS:-6}"
export PROFILE_MAX_ITERATIONS="${PROFILE_MAX_ITERATIONS:-8}"
export PROFILE_PREFIX="${PROFILE_PREFIX:-dp4_tp8_dcp8_127x2k_1x256k_torch_mid8}"
export PROFILE_ROOT="${PROFILE_ROOT:-/mnt/nvme1n1/ml_research/linbinbin1/profiler_res_0326}"
export PROFILE_DIR="${PROFILE_DIR:-${PROFILE_ROOT}/node${DP_NODE_RANK}}"
export REQUEST_CONFIG="${REQUEST_CONFIG:-/tmp/dp4_rank_requests_127x2k_1x256k.json}"
export TIMEOUT="${TIMEOUT:-1800}"
export POST_PROFILE_SLEEP="${POST_PROFILE_SLEEP:-5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ALLOW_OVER_MAX_CONTEXT="${ALLOW_OVER_MAX_CONTEXT:-0}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

BATCH_SIZE="$((NUM_SHORT_REQUESTS + 1))"

if [[ "${DP_SIZE}" != "4" ]]; then
  echo "This script is written for DP_SIZE=4. Override only if you also update the request generator." >&2
  exit 1
fi

if [[ "${BATCH_SIZE}" != "128" ]]; then
  echo "This script expects 127 short + 1 long request per DP rank, so BATCH_SIZE must stay 128." >&2
  exit 1
fi

MODEL_CONFIG_MAX_LEN="$(
  python - "${MODEL}" <<'PY'
import json
import os
import sys

config_path = os.path.join(sys.argv[1], "config.json")
try:
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
except FileNotFoundError:
    print(0)
    raise SystemExit(0)

print(int(config.get("max_position_embeddings", 0)))
PY
)"

if [[ "${ALLOW_OVER_MAX_CONTEXT}" != "1" ]] \
  && [[ "${MODEL_CONFIG_MAX_LEN}" != "0" ]] \
  && (( LONG_INPUT_LEN + OUTPUT_LEN > MODEL_CONFIG_MAX_LEN )); then
  echo "Requested long request length ${LONG_INPUT_LEN}+${OUTPUT_LEN} exceeds ${MODEL}/config.json max_position_embeddings=${MODEL_CONFIG_MAX_LEN}." >&2
  echo "The current deepseek-v3 config in this workspace advertises 163840 max positions, so 256k input will not fit as-is." >&2
  echo "Set ALLOW_OVER_MAX_CONTEXT=1 to bypass this guard if you intentionally want to try an override." >&2
  exit 1
fi

mkdir -p "${PROFILE_DIR}" "$(dirname "${REQUEST_CONFIG}")"

python - "${REQUEST_CONFIG}" "${DP_SIZE}" "${NUM_SHORT_REQUESTS}" "${SHORT_INPUT_LEN}" "${LONG_INPUT_LEN}" "${OUTPUT_LEN}" "${REQUEST_TOKEN_SEED}" <<'PY'
import json
import sys

request_config_path = sys.argv[1]
dp_size = int(sys.argv[2])
num_short_requests = int(sys.argv[3])
short_input_len = int(sys.argv[4])
long_input_len = int(sys.argv[5])
output_len = int(sys.argv[6])
request_token_seed = int(sys.argv[7])

sampling_params = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": output_len,
    "min_tokens": output_len,
    "ignore_eos": True,
}

request_table = {}
for global_dp_rank in range(dp_size):
    rank_requests = []
    seed_base = request_token_seed + global_dp_rank * 1000
    for request_idx in range(num_short_requests):
        rank_requests.append({
            "prompt_token_count": short_input_len,
            "token_id_seed": seed_base + request_idx,
            "sampling_params": sampling_params,
        })
    rank_requests.append({
        "prompt_token_count": long_input_len,
        "token_id_seed": seed_base + num_short_requests,
        "sampling_params": sampling_params,
    })
    request_table[str(global_dp_rank)] = rank_requests

with open(request_config_path, "w", encoding="utf-8") as f:
    json.dump(request_table, f, indent=2)
    f.write("\n")
PY

echo "Launching ${DATA_PARALLEL_SCRIPT}"
echo "  MODEL=${MODEL}"
echo "  MASTER_ADDR=${MASTER_ADDR}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "  DP_NODE_RANK=${DP_NODE_RANK}"
echo "  DP_NUM_NODES=${DP_NUM_NODES}"
echo "  PROFILE_DIR=${PROFILE_DIR}"
echo "  REQUEST_CONFIG=${REQUEST_CONFIG}"

cmd=(
  python "${DATA_PARALLEL_SCRIPT}"
  --model "${MODEL}"
  --start-sh-defaults
  --profile
  --warmup-iters "${WARMUP_ITERS}"
  --profile-prefix "${PROFILE_PREFIX}"
  --post-profile-sleep "${POST_PROFILE_SLEEP}"
  --request-config "${REQUEST_CONFIG}"
  --request-token-seed "${REQUEST_TOKEN_SEED}"
  --timeout "${TIMEOUT}"
  -dp "${DP_SIZE}"
  --data-parallel-size-local "${DATA_PARALLEL_SIZE_LOCAL}"
  --dp-num-nodes "${DP_NUM_NODES}"
  --dp-node-rank "${DP_NODE_RANK}"
  --dp-master-addr "${MASTER_ADDR}"
  --dp-master-port "${MASTER_PORT}"
  -tp "${TP_SIZE}"
  -dcp "${DCP_SIZE}"
  --dcp-comm-backend "${DCP_COMM_BACKEND}"
  --attention-backend "${ATTENTION_BACKEND}"
  --all2all-backend deepep_low_latency
  --cudagraph-capture-sizes 1 2 4 "${BATCH_SIZE}"
  --load-format dummy
  --gpu-memory-utilization 0.85
  --profiler-config.profiler torch
  --profiler-config.torch_profiler_dir "${PROFILE_DIR}"
  --profiler-config.ignore_frontend true
  --profiler-config.delay_iterations "${PROFILE_DELAY_ITERATIONS}"
  --profiler-config.max_iterations "${PROFILE_MAX_ITERATIONS}"
  --kv-transfer-config '{ "kv_connector": "DecodeBenchConnector", "kv_role": "kv_both", "kv_connector_extra_config": { "fill_mean": 0.015, "fill_std": 0.0 } }'
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
)

if [[ -n "${MAX_MODEL_LEN}" ]]; then
  cmd+=(--max-model-len "${MAX_MODEL_LEN}")
fi

"${cmd[@]}"
