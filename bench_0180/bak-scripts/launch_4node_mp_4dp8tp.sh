#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_ENV="${CLUSTER_ENV:-${SCRIPT_DIR}/cluster.env}"

if [[ -f "${CLUSTER_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CLUSTER_ENV}"
  set +a
fi

usage() {
  cat <<'EOF'
Usage:
  launch_4node_mp_4dp8tp.sh \
    --model MODEL \
    [--master-addr NODE0_IP|auto] \
    [--host 127.0.0.1] \
    [--port 8000] \
    [--data-parallel-rpc-port 13345] \
    [--log-dir /path/to/logs] \
    -- [extra vllm serve args]

This launcher is intended to be used from vllm bench sweep:
  --serve-cmd '/path/launch_4node_mp_4dp8tp.sh --model ... --master-addr ... --host 127.0.0.1 --port 8000 --'

Everything after the final '--' is forwarded to every vllm serve command.
That is what allows sweep --serve-params to append flags to the launcher.
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

build_wrapper() {
  local workdir="$1"
  local env_script="$2"
  shift 2
  local cmd0="$1"
  shift

  local inner=""
  if [[ -n "${env_script}" ]]; then
    inner+="source $(printf '%q' "${env_script}") && "
  fi
  if [[ "${cmd0}" == */* ]]; then
    inner+="__resolved_cmd=$(printf '%q' "${cmd0}") && "
  else
    inner+="__resolved_cmd=\$(command -v $(printf '%q' "${cmd0}")) && "
  fi
  if [[ -n "${workdir}" ]]; then
    inner+="cd $(printf '%q' "${workdir}") && "
  fi
  inner+="exec \"\${__resolved_cmd}\""
  if (($#)); then
    inner+=" $(shell_join "$@")"
  fi
  printf '%s' "${inner}"
}

split_words_arg() {
  local input="$1"
  local -n out_ref="$2"

  out_ref=()
  if [[ -n "${input}" ]]; then
    read -r -a out_ref <<<"${input}"
  fi
}

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required setting: ${name}" >&2
    exit 2
  fi
}

resolve_ssh_hostname() {
  local ssh_target="$1"
  local resolved

  resolved="$(
    ssh "${ssh_opt_array[@]}" -G "${ssh_target}" 2>/dev/null \
      | awk '$1 == "hostname" { print $2; exit }'
  )"

  if [[ -n "${resolved}" ]]; then
    printf '%s' "${resolved}"
  else
    printf '%s' "${ssh_target}"
  fi
}

discover_master_addr() {
  local route_target="${MASTER_ADDR_DISCOVERY_TARGET:-}"
  local python_bin

  if [[ -z "${route_target}" && -n "${NODE1_SSH:-}" ]]; then
    route_target="$(resolve_ssh_hostname "${NODE1_SSH}")"
  fi
  if [[ -z "${route_target}" ]]; then
    route_target="8.8.8.8"
  fi

  python_bin="$(pick_python)"
  "${python_bin}" - "${route_target}" <<'PY'
import socket
import sys

route_target = sys.argv[1]

def print_and_exit(ip: str) -> None:
    if ip and ":" not in ip and not ip.startswith("127."):
        print(ip)
        raise SystemExit(0)

for port in (22, 13345, 1):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((route_target, port))
        print_and_exit(sock.getsockname()[0])
    except OSError:
        pass
    finally:
        sock.close()

for candidate in (socket.gethostname(), socket.getfqdn()):
    try:
        _, _, addresses = socket.gethostbyname_ex(candidate)
    except socket.gaierror:
        continue
    for address in addresses:
        print_and_exit(address)

raise SystemExit(
    f"Unable to auto-discover a non-loopback IPv4 for MASTER_ADDR. "
    f"Set MASTER_ADDR explicitly or set MASTER_ADDR_DISCOVERY_TARGET."
)
PY
}

MODEL="${MODEL:-}"
MASTER_ADDR="${MASTER_ADDR:-auto}"
HOST="${SWEEP_HOST:-127.0.0.1}"
PORT="${SWEEP_PORT:-8000}"
RPC_PORT="${DATA_PARALLEL_RPC_PORT:-13345}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
VLLM_BIN="${VLLM_BIN:-vllm}"
SSH_OPTS="${SSH_OPTS:- -o BatchMode=yes}"
LOCAL_WORKDIR="${LOCAL_WORKDIR:-$(pwd)}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-${LOCAL_WORKDIR}}"
LOCAL_ENV_SCRIPT="${LOCAL_ENV_SCRIPT:-}"
REMOTE_ENV_SCRIPT="${REMOTE_ENV_SCRIPT:-${LOCAL_ENV_SCRIPT}}"
LOCAL_SHELL="${LOCAL_SHELL:-bash}"
LOCAL_SHELL_FLAGS="${LOCAL_SHELL_FLAGS:--lc}"
REMOTE_SHELL="${REMOTE_SHELL:-${LOCAL_SHELL}}"
REMOTE_SHELL_FLAGS="${REMOTE_SHELL_FLAGS:-${LOCAL_SHELL_FLAGS}}"
HEAD_NODE_SSH="${HEAD_NODE_SSH:-}"
MASTER_ADDR_DISCOVERY_TARGET="${MASTER_ADDR_DISCOVERY_TARGET:-}"
NODE1_SSH="${NODE1_SSH:-}"
NODE2_SSH="${NODE2_SSH:-}"
NODE3_SSH="${NODE3_SSH:-}"
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
      echo "Unknown launcher argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_var MODEL
require_var NODE1_SSH
require_var NODE2_SSH
require_var NODE3_SSH

mkdir -p "${LOG_DIR}"

ssh_opt_array=()
if [[ -n "${SSH_OPTS}" ]]; then
  read -r -a ssh_opt_array <<<"${SSH_OPTS}"
fi

local_shell_flags=()
remote_shell_flags=()
split_words_arg "${LOCAL_SHELL_FLAGS}" local_shell_flags
split_words_arg "${REMOTE_SHELL_FLAGS}" remote_shell_flags

if [[ -z "${MASTER_ADDR}" || "${MASTER_ADDR}" == "auto" ]]; then
  MASTER_ADDR="$(discover_master_addr)"
fi

child_pids=()

start_remote_rank() {
  local dp_rank="$1"
  local ssh_target="$2"
  local log_file="${LOG_DIR}/dp${dp_rank}.log"

  local remote_cmd=(
    "${VLLM_BIN}" serve "${MODEL}"
    --headless
    --tensor-parallel-size 8
    --data-parallel-size 4
    --data-parallel-backend mp
    --data-parallel-size-local 1
    --data-parallel-start-rank "${dp_rank}"
    --data-parallel-address "${MASTER_ADDR}"
    --data-parallel-rpc-port "${RPC_PORT}"
    --async-scheduling
    "${EXTRA_VLLM_ARGS[@]}"
  )

  local wrapper
  wrapper="$(build_wrapper "${REMOTE_WORKDIR}" "${REMOTE_ENV_SCRIPT}" "${remote_cmd[@]}")"

  ssh "${ssh_opt_array[@]}" "${ssh_target}" \
    "$(shell_join "${REMOTE_SHELL}" "${remote_shell_flags[@]}" "${wrapper}")" \
    >"${log_file}" 2>&1 &
  child_pids+=("$!")
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if ((${#child_pids[@]})); then
    kill "${child_pids[@]}" 2>/dev/null || true
    sleep 1
    kill -9 "${child_pids[@]}" 2>/dev/null || true
    wait "${child_pids[@]}" 2>/dev/null || true
  fi
  exit "${rc}"
}
trap cleanup EXIT INT TERM

start_remote_rank 1 "${NODE1_SSH}"
start_remote_rank 2 "${NODE2_SSH}"
start_remote_rank 3 "${NODE3_SSH}"

local_log="${LOG_DIR}/dp0.log"
local_cmd=(
  "${VLLM_BIN}" serve "${MODEL}"
  --tensor-parallel-size 8
  --data-parallel-size 4
  --data-parallel-backend mp
  --data-parallel-size-local 1
  --data-parallel-address "${MASTER_ADDR}"
  --data-parallel-rpc-port "${RPC_PORT}"
  --async-scheduling
  --api-server-count 1
  --host "${HOST}"
  --port "${PORT}"
  "${EXTRA_VLLM_ARGS[@]}"
)

local_wrapper="$(build_wrapper "${LOCAL_WORKDIR}" "${LOCAL_ENV_SCRIPT}" "${local_cmd[@]}")"
"${LOCAL_SHELL}" "${local_shell_flags[@]}" "${local_wrapper}" >"${local_log}" 2>&1 &
child_pids+=("$!")

echo "launcher ready"
echo "  model: ${MODEL}"
if [[ -n "${HEAD_NODE_SSH}" ]]; then
  echo "  head_node: ${HEAD_NODE_SSH}"
fi
echo "  master_addr: ${MASTER_ADDR}"
echo "  host: ${HOST}"
echo "  port: ${PORT}"
echo "  rpc_port: ${RPC_PORT}"
echo "  logs: ${LOG_DIR}"
echo "  local shell: ${LOCAL_SHELL} ${LOCAL_SHELL_FLAGS}"
echo "  remote shell: ${REMOTE_SHELL} ${REMOTE_SHELL_FLAGS}"
echo "  remote ranks: ${NODE1_SSH}, ${NODE2_SSH}, ${NODE3_SSH}"

wait -n "${child_pids[@]}"
