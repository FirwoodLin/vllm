#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT

STUB_DIR="${TMP_DIR}/bin"
CALL_LOG="${TMP_DIR}/calls.log"
OUTPUT_LOG="${TMP_DIR}/output.log"
mkdir -p "${STUB_DIR}"
touch "${CALL_LOG}"

write_stub() {
    local name=$1
    local body=$2
    printf '#!/usr/bin/env bash\n%s\n' "${body}" > "${STUB_DIR}/${name}"
    chmod +x "${STUB_DIR}/${name}"
}

write_stub "pkill" 'echo "pkill $*" >> "${CALL_LOG}"; exit 0'
write_stub "pgrep" 'echo "pgrep $*" >> "${CALL_LOG}"; exit 1'
write_stub "ssh" 'echo "ssh $*" >> "${CALL_LOG}"; cat >/dev/null; exit 0'
write_stub "sleep" 'echo "sleep $*" >> "${CALL_LOG}"; exit 0'
write_stub "python3" '{
    echo "python3 $*"
    echo "GPU_CLEANUP_SCRIPT=${GPU_CLEANUP_SCRIPT:-}"
    echo "GPU_CLEANUP_SELECTION=${GPU_CLEANUP_SELECTION:-}"
} >> "${CALL_LOG}"; exit 0'

export PATH="${STUB_DIR}:${PATH}"
export CALL_LOG

RUN_GPU_CLEANUP=1 \
GPU_CLEANUP_HOSTS="host-a host-b" \
GPU_CLEANUP_SCRIPT="/tmp/fake_kill_gpu.sh" \
GPU_CLEANUP_SELECTION="all" \
REMOTE_HOST="remote-clean-test" \
    bash "${SCRIPT_DIR}/run_dp_ep_4dp4tp_bench.sh" cleanup > "${OUTPUT_LOG}"

grep -F "Running /tmp/fake_kill_gpu.sh on host-a host-b..." "${OUTPUT_LOG}" >/dev/null
grep -F "python3 ${SCRIPT_DIR}/kill_gpu_hosts.py host-a host-b" "${CALL_LOG}" >/dev/null
grep -F "GPU_CLEANUP_SCRIPT=/tmp/fake_kill_gpu.sh" "${CALL_LOG}" >/dev/null
grep -F "GPU_CLEANUP_SELECTION=all" "${CALL_LOG}" >/dev/null
