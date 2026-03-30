#!/usr/bin/env bash
set -euo pipefail

# Benchmark Qwen3.5-397B-A17B-FP8 prefill-heavy bursts on a single 8-GPU node.
#
# Default assumptions:
# - single-node DP=8 + EP
# - DeepEP high-throughput backend for expert all2all
# - random text-only prompts against the OpenAI completion endpoint
# - prefill-focused measurement by setting output_len=1 and request_rate=inf
#
# Notes:
# - This Qwen3.5 model is a Qwen3.5-MoE model, not an MLA model. Do not force
#   FLASHMLA unless you have separately verified it is valid for your build.
# - The script defaults to ATTENTION_BACKEND=auto for that reason.
#
# Usage:
#   ./qwen35prefill.sh
#   ./qwen35prefill.sh all
#   ./qwen35prefill.sh serve
#   ./qwen35prefill.sh bench
#   ./qwen35prefill.sh stop
#
# Common overrides:
#   ATTENTION_BACKEND=FLASH_ATTN ./qwen35prefill.sh
#   SCENARIOS="256x2048 128x4096" ./qwen35prefill.sh
#   PORT=8010 LOG_DIR=/tmp/qwen35prefill ./qwen35prefill.sh

MODE="${1:-all}"

MODEL="${MODEL:-/mnt/nvme1n1/ml_research/models/models--Qwen--Qwen3.5-397B-A17B-FP8}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

DP_SIZE="${DP_SIZE:-8}"
TP_SIZE="${TP_SIZE:-1}"
API_SERVER_COUNT="${API_SERVER_COUNT:-8}"
ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-deepep_high_throughput}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-}"
ENABLE_DBO="${ENABLE_DBO:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"

OUTPUT_LEN="${OUTPUT_LEN:-1}"
NUM_WARMUPS="${NUM_WARMUPS:-1}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"

SCENARIOS="${SCENARIOS:-128x4096 64x8192 32x16384}"

LOG_DIR="${LOG_DIR:-/tmp/qwen35prefill}"
SERVER_LOG="${SERVER_LOG:-$LOG_DIR/server.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/server.pid}"
RESULT_ROOT="${RESULT_ROOT:-$LOG_DIR/results}"

mkdir -p "$LOG_DIR" "$RESULT_ROOT"

ceil_div() {
  local x="$1"
  local y="$2"
  echo $(((x + y - 1) / y))
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

validate_inputs() {
  if [[ ! -f "$MODEL/config.json" ]]; then
    echo "model path does not look valid: $MODEL" >&2
    exit 1
  fi

  if [[ "$DP_SIZE" -le 0 || "$TP_SIZE" -le 0 || "$API_SERVER_COUNT" -le 0 ]]; then
    echo "DP_SIZE/TP_SIZE/API_SERVER_COUNT must all be > 0" >&2
    exit 1
  fi
}

compute_scheduler_limits() {
  MAX_LOCAL_SEQS=1
  MAX_LOCAL_BATCHED_TOKENS=1
  MAX_INPUT_LEN=1

  local scenario
  for scenario in $SCENARIOS; do
    local global_batch="${scenario%x*}"
    local input_len="${scenario#*x}"
    if [[ -z "$global_batch" || -z "$input_len" || "$global_batch" == "$scenario" ]]; then
      echo "invalid scenario: $scenario (expected BATCHxINPUT_LEN)" >&2
      exit 1
    fi

    local local_seqs
    local local_tokens
    local_seqs="$(ceil_div "$global_batch" "$DP_SIZE")"
    local_tokens=$((local_seqs * input_len))

    if (( local_seqs > MAX_LOCAL_SEQS )); then
      MAX_LOCAL_SEQS="$local_seqs"
    fi
    if (( local_tokens > MAX_LOCAL_BATCHED_TOKENS )); then
      MAX_LOCAL_BATCHED_TOKENS="$local_tokens"
    fi
    if (( input_len > MAX_INPUT_LEN )); then
      MAX_INPUT_LEN="$input_len"
    fi
  done

  if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
    EFFECTIVE_MAX_MODEL_LEN="$MAX_MODEL_LEN"
  else
    EFFECTIVE_MAX_MODEL_LEN=$((MAX_INPUT_LEN + OUTPUT_LEN + 64))
  fi
}

print_config() {
  cat <<EOF
Mode:                     $MODE
Model:                    $MODEL
Host:                     $HOST:$PORT
DP / TP:                  $DP_SIZE / $TP_SIZE
EP all2all backend:       $ALL2ALL_BACKEND
Attention backend:        $ATTENTION_BACKEND
Scenarios:                $SCENARIOS
Output len:               $OUTPUT_LEN
Per-rank max_num_seqs:    $MAX_LOCAL_SEQS
Per-rank max_batched_tok: $MAX_LOCAL_BATCHED_TOKENS
Max model len:            $EFFECTIVE_MAX_MODEL_LEN
Log dir:                  $LOG_DIR
Result root:              $RESULT_ROOT
EOF
}

server_is_up() {
  curl -sf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1
}

server_pid_running() {
  local pid="$1"
  kill -0 "$pid" >/dev/null 2>&1
}

wait_for_server() {
  local pid="$1"
  local waited=0

  echo "waiting for server readiness on http://$HOST:$PORT/v1/models ..."
  until server_is_up; do
    if ! server_pid_running "$pid"; then
      echo "server exited before becoming ready; last log lines:" >&2
      tail -n 80 "$SERVER_LOG" >&2 || true
      exit 1
    fi
    if (( waited >= READY_TIMEOUT )); then
      echo "timed out waiting for server readiness after ${READY_TIMEOUT}s" >&2
      tail -n 80 "$SERVER_LOG" >&2 || true
      exit 1
    fi
    sleep "$POLL_INTERVAL"
    waited=$((waited + POLL_INTERVAL))
  done
}

stop_server() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "no pid file found at $PID_FILE"
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  if [[ -z "$pid" ]]; then
    echo "pid file is empty: $PID_FILE" >&2
    return 1
  fi

  if server_pid_running "$pid"; then
    echo "stopping server pid $pid"
    kill "$pid"
    wait "$pid" 2>/dev/null || true
  else
    echo "server pid $pid is not running"
  fi

  rm -f "$PID_FILE"
}

start_server_background() {
  local detach="${1:-0}"
  export VLLM_MOE_ROUTING_SIMULATION_STRATEGY="${VLLM_MOE_ROUTING_SIMULATION_STRATEGY:-uniform_random}"
  export VLLM_RANDOMIZE_DP_DUMMY_INPUTS="${VLLM_RANDOMIZE_DP_DUMMY_INPUTS:-1}"

  local -a cmd
  cmd=(
    vllm serve "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TP_SIZE"
    --data-parallel-size "$DP_SIZE"
    --enable-expert-parallel
    --all2all-backend "$ALL2ALL_BACKEND"
    --max-model-len "$EFFECTIVE_MAX_MODEL_LEN"
    --max-num-seqs "$MAX_LOCAL_SEQS"
    --max-num-batched-tokens "$MAX_LOCAL_BATCHED_TOKENS"
    --api-server-count "$API_SERVER_COUNT"
  )

  if [[ "$ATTENTION_BACKEND" != "auto" ]]; then
    cmd+=(--attention-backend "$ATTENTION_BACKEND")
  fi
  if [[ -n "$FLASH_ATTN_VERSION" ]]; then
    cmd+=(--attention-config.flash_attn_version "$FLASH_ATTN_VERSION")
  fi
  if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    cmd+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
  fi
  if [[ "$ENABLE_DBO" == "1" ]]; then
    cmd+=(--enable-dbo)
  fi
  if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra_server_args=( ${EXTRA_SERVER_ARGS} )
    cmd+=("${extra_server_args[@]}")
  fi

  echo "starting server; log: $SERVER_LOG"
  : > "$SERVER_LOG"
  if [[ "$detach" == "1" ]]; then
    nohup "${cmd[@]}" >>"$SERVER_LOG" 2>&1 &
  else
    "${cmd[@]}" >>"$SERVER_LOG" 2>&1 &
  fi
  local pid=$!
  echo "$pid" > "$PID_FILE"
  echo "server pid: $pid"
  wait_for_server "$pid"
}

run_one_benchmark() {
  local scenario="$1"
  local global_batch="${scenario%x*}"
  local input_len="${scenario#*x}"
  local label="prefill_b${global_batch}_l${input_len}"
  local result_dir="$RESULT_ROOT/$(date +%Y%m%d_%H%M%S)"

  mkdir -p "$result_dir"

  echo
  echo "==> benchmarking scenario: global_batch=${global_batch}, input_len=${input_len}"

  vllm bench serve \
    --backend openai \
    --host "$HOST" \
    --port "$PORT" \
    --endpoint /v1/completions \
    --dataset-name random \
    --model "$MODEL" \
    --random-input-len "$input_len" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$global_batch" \
    --max-concurrency "$global_batch" \
    --request-rate "$REQUEST_RATE" \
    --ignore-eos \
    --num-warmups "$NUM_WARMUPS" \
    --save-result \
    --result-dir "$result_dir" \
    --label "$label" \
    --metadata \
      scenario="$scenario" \
      model_path="$MODEL" \
      dp="$DP_SIZE" \
      tp="$TP_SIZE" \
      ep=1 \
      all2all_backend="$ALL2ALL_BACKEND" \
      attention_backend="$ATTENTION_BACKEND"
}

run_benchmarks() {
  local scenario
  for scenario in $SCENARIOS; do
    run_one_benchmark "$scenario"
  done

  echo
  echo "results saved under: $RESULT_ROOT"
}

cleanup_all_mode() {
  if [[ -f "$PID_FILE" ]]; then
    stop_server || true
  fi
}

main() {
  require_cmd vllm
  require_cmd curl
  validate_inputs
  compute_scheduler_limits
  print_config

  case "$MODE" in
    all)
      trap cleanup_all_mode EXIT
      start_server_background
      run_benchmarks
      ;;
    serve)
      start_server_background 1
      echo "server is ready; stop it with: $0 stop"
      ;;
    bench)
      if ! server_is_up; then
        echo "server is not ready on http://$HOST:$PORT; start it first with: $0 serve" >&2
        exit 1
      fi
      run_benchmarks
      ;;
    stop)
      stop_server
      ;;
    *)
      echo "unknown mode: $MODE" >&2
      echo "usage: $0 [all|serve|bench|stop]" >&2
      exit 1
      ;;
  esac
}

main "$@"
