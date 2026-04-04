# KIMI 补点
python3 benchmarks/manual_multinode_poisson_runner.py \
    --case-csv benchmarks/kimi_issue01_resume_plan.csv \
    --artifact-root /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench/manual_multinode \
    --run-label issue01-random-kimi-ds3-resume
# Issue 03 random 的 DPSK DP32 场景（rate 从小到大，以 5 为间隔）
  python3 /vllm/benchmarks/manual_multinode_poisson_runner.py \
    --case-csv /vllm/benchmarks/issue03_random_dpsk_dp32_full5.csv \
    --run-label issue03-random-dpsk-dp32-full5
# Issue 05 random 的 DPSK DP32 场景（rate 从小到大，以 5 为间隔）
  python3 /vllm/benchmarks/manual_multinode_poisson_runner.py \
    --case-csv /vllm/benchmarks/issue05_random_dpsk_dp32_full5.csv \
    --run-label issue05-random-dpsk-dp32-full5
# issue 01 random , DPSK sweep
   python3 /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/benchmarks/manual_multinode_poisson_runner.py \
    --all \
    --model deepseek_v3_1024k \
    --dataset issue01_random \
    --rate-plan coarse10_then_mid5 \
    --artifact-root /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench/manual_multinode \
    --run-label  issue01-random-dpsk-resume
    
# issue 03 random  和 issue 05 random ，DPSK 进行Sweep
  python3 /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/benchmarks/manual_multinode_poisson_runner.py \
    --all \
    --model deepseek_v3_1024k \
    --dataset issue03_random \
    --rate-plan coarse10_then_mid5 \
    --artifact-root /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench/manual_multinode \
    --run-label issue03-random-dpsk-resume

  python3 /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/benchmarks/manual_multinode_poisson_runner.py \
    --all \
    --model deepseek_v3_1024k \
    --dataset issue05_random \
    --rate-plan coarse10_then_mid5 \
    --artifact-root /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench/manual_multinode \
    --run-label issue05-random-dpsk-resume
