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

HOST="${SWEEP_HOST:-127.0.0.1}"
PORT="${SWEEP_PORT:-8000}"

while (($#)); do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: reset_sweep_caches.sh [--host HOST] [--port PORT]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

base_url="http://${HOST}:${PORT}"
for endpoint in /reset_prefix_cache /reset_mm_cache /reset_encoder_cache; do
  curl -fsS -X POST "${base_url}${endpoint}" >/dev/null
done
