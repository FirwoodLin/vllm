#!/usr/bin/env zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench/manual_multinode}"

cd "${REPO_ROOT}"

python3 benchmarks/manual_multinode_poisson_runner.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --case-csv benchmarks/plans-unfinished/dpsk_issue01_random_dp16_dp32_rate40_45.csv

python3 benchmarks/manual_multinode_poisson_runner.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --case-csv benchmarks/plans-unfinished/dpsk_issue03_random_all_full2p5.csv

python3 benchmarks/manual_multinode_poisson_runner.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --case-csv benchmarks/plans-unfinished/dpsk_issue05_random_all_full2p5.csv

python3 benchmarks/manual_multinode_poisson_runner.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --case-csv benchmarks/plans-unfinished/dpsk_long_full_all_coarse1_then_mid0p5.csv

python3 benchmarks/manual_multinode_poisson_runner.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --case-csv benchmarks/plans-unfinished/dpsk_short_random_all_coarse40_then_mid20.csv
