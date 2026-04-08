#!/usr/bin/env zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench/profile_dp32}"
CASE_CSV="${CASE_CSV:-${SCRIPT_DIR}/profile_cases/deepseek_issue01_dp32_profile.casecsv}"

cd "${REPO_ROOT}"

python3 benchmarks/manual_multinode_poisson_runner.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --case-csv "${CASE_CSV}" \
  --frontend-extra-arg=--profile-after-warmup \
  --frontend-extra-arg=--profiler-config.profiler \
  --frontend-extra-arg=torch \
  --frontend-extra-arg=--profiler-config.torch_profiler_dir \
  --frontend-extra-arg='{benchmark_dir}/torch_profiler' \
  --frontend-extra-arg=--profiler-config.ignore_frontend \
  --frontend-extra-arg=true \
  --frontend-extra-arg=--profiler-config.delay_iterations \
  --frontend-extra-arg=17 \
  --frontend-extra-arg=--profiler-config.max_iterations \
  --frontend-extra-arg=31 \
  --frontend-extra-arg=--profiler-config.wait_iterations \
  --frontend-extra-arg=0 \
  --frontend-extra-arg=--profiler-config.warmup_iterations \
  --frontend-extra-arg=0 \
  --headless-extra-arg=--profiler-config.profiler \
  --headless-extra-arg=torch \
  --headless-extra-arg=--profiler-config.torch_profiler_dir \
  --headless-extra-arg='{benchmark_dir}/torch_profiler' \
  --headless-extra-arg=--profiler-config.delay_iterations \
  --headless-extra-arg=17 \
  --headless-extra-arg=--profiler-config.max_iterations \
  --headless-extra-arg=31 \
  --headless-extra-arg=--profiler-config.wait_iterations \
  --headless-extra-arg=0 \
  --headless-extra-arg=--profiler-config.warmup_iterations \
  --headless-extra-arg=0
