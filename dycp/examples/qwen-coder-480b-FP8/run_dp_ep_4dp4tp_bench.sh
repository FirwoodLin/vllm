#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export LAUNCH_SCRIPT=${LAUNCH_SCRIPT:-run_qwen_235b_dp_ep_4dp4tp.sh}
export STRATEGY_NAME=${STRATEGY_NAME:-dp4tp4}
export PROFILE_MODE=${PROFILE_MODE:-qwen_coder_480b_dp4tp4}
export RUNNER_NAME=${RUNNER_NAME:-$(basename "$0")}

exec "${SCRIPT_DIR}/run_dycp_16dp_tp1_bench.sh" "$@"
