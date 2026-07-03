#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export LAUNCH_SCRIPT=${LAUNCH_SCRIPT:-run_qwen_235b_dycp_16dp_tp1.sh}
export RESULT_DIR=${RESULT_DIR:-/vllm/dycp/results/dycp16dp_tp1}
export LOAD_FORMAT=${LOAD_FORMAT:-dummy}
export RUNNER_NAME=${RUNNER_NAME:-$(basename "$0")}

exec "${SCRIPT_DIR}/run_dp_ep_4dp4tp_bench.sh" "$@"
