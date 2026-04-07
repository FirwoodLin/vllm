#!/usr/bin/env zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench/manual_multinode}"

cd "${REPO_ROOT}"


# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/benchmarks/plans-unfinished/v0/deepseek_issue01_random_dp32_rate40_least_batch_least_cache.csv

# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/benchmarks/plans-unfinished/v0/deepseek_issue03_random_dp32_dp16cp2_5_to_30_step2.5.csv

# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/dpsk_issue03_random_all_full2p5.csv

# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/dpsk_issue05_random_all_full2p5.csv

# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/dpsk_long_full_all_coarse1_then_mid0p5.csv

# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/benchmarks/plans-unfinished/v0/kimi_issue01_random_dp16cp2_dp32_continue_to_tpot100.csv

# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/benchmarks/plans-unfinished/v0/dpsk_long_full_all_step0p25_0p25_to3.csv

# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/benchmarks/plans-unfinished/v0/deepseek_issue01_random_dp32_rate30_min_batch_min_cache.csv

# 1. DPSK: issue03random DP32 rate=21, DP16cp2 rate=26
# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0407/dpsk_issue03random_dp32_dp16cp2_specific.csv

# # 2. DPSK: issue05random DP32 rate=16, DP16cp2 rate=16
# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0407/dpsk_issue05random_dp32_dp16cp2_specific.csv

# # 3. DPSK: Long_full DP32 complete
# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0407/dpsk_long_full_dp32_full2p5.csv

# # 4. DPSK: Long_full DP16cp2 complete
# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0407/dpsk_long_full_dp16cp2_full2p5.csv

# # 5. Kimi: issue01 DP16cp2 rate=57
# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0407/kimi_issue01_dp16cp2_rate57.csv

# # 6. DPSK: issue01random DP32 with least_cache dispatch policy
# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0407/dpsk_issue01random_dp32_least_cache.csv

# # 7. DPSK: issue03random DP32 with least_cache dispatch policy
# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0407/dpsk_issue03random_dp32_least_cache.csv

# # 8. DPSK: issue05random DP32 with least_cache dispatch policy
# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0407/dpsk_issue05random_dp32_least_cache.csv

# # 9. DPSK: issue03random DP16cp2 rate=10,15; DP4dcp8 rate=17.5
# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0408/dpsk_issue03random_dp16cp2_dp4dcp8_specific.csv

# # 10. DPSK: issue05random DP32 rate=17; DP16cp2 rate=17
# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0408/dpsk_issue05random_dp32_dp16cp2_specific.csv

# # 11. DPSK: long_full DP32 and DP16cp2 from rate=0.25 to 3.0, step 0.25
# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0408/dpsk_long_full_dp32_dp16cp2_step0p25_0p25_to3.csv

# 12. DPSK: short_random DP32 with least_cache from rate=20 to 140, step 20
# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0408/dpsk_short_random_dp32_least_cache_step20_20_to140.csv

# 13. Kimi: issue01 DP32 with least_cache from rate=10 to 60, step 10
# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0408/kimi_issue01_dp32_least_cache_step10_10_to60.csv

# 14. DPSK: long_full DP32 with least_cache from rate=0.5 to 3.5, step 0.5
# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0408/dpsk_long_full_dp32_least_cache_step0p5_0p5_to3p5.csv

# 13. Kimi: issue01 DP32 with least_cache from rate=10 to 60, step 10
# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0408/kimi_issue01_dp32_least_cache_step10_10_to60.csv

# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0408/kimi_issue01random_dp16cp2_rate58_59.csv

# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0408/dpsk_long_full_dp32_least_cache_rate0p25.csv

# python3 benchmarks/manual_multinode_poisson_runner.py \
#   --artifact-root "${ARTIFACT_ROOT}" \
#   --case-csv benchmarks/plans-unfinished/0408/dpsk_issue05random_dp32_dp16cp2_rate20_22p5.csv

python3 benchmarks/manual_multinode_poisson_runner.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --case-csv benchmarks/plans-unfinished/0408/dpsk_issue03random_dp32_dp16cp2_rate17p5_20_22p5_25_27p5.csv

python3 benchmarks/manual_multinode_poisson_runner.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --case-csv benchmarks/plans-unfinished/0408/dpsk_short_random_dp8dcp4_dp16cp2_dp32_custom_bs_rate80_100_120_160_180.csv
