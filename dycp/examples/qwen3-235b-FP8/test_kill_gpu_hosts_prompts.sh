#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT

STUB_DIR="${TMP_DIR}/bin"
PROMPT_LOG="${TMP_DIR}/prompts.log"
mkdir -p "${STUB_DIR}"
touch "${PROMPT_LOG}"

cat > "${STUB_DIR}/ssh" <<'STUB'
#!/usr/bin/env bash
echo "[sudo] password for tester:"
IFS= read -r password
echo "password=${password}" >> "${PROMPT_LOG}"
echo "> "
IFS= read -r selection
echo "selection=${selection}" >> "${PROMPT_LOG}"
STUB
chmod +x "${STUB_DIR}/ssh"

export PATH="${STUB_DIR}:${PATH}"
export PROMPT_LOG

GPU_CLEANUP_SUDO_PASSWORD="secret-password" \
GPU_CLEANUP_SELECTION="all" \
    python3 "${SCRIPT_DIR}/kill_gpu_hosts.py" fake-host \
        --script /tmp/fake_kill_gpu.sh \
        --timeout 5 >/dev/null

grep -F "password=secret-password" "${PROMPT_LOG}" >/dev/null
grep -F "selection=all" "${PROMPT_LOG}" >/dev/null
