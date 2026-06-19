#!/bin/bash

# 单台机器 DYCP Decode 启动脚本
# 模型: Qwen3-235B-Instruct-2507-FP8
# 适用: 单台机器 8x H800 GPU

set -x

# ========== 基础环境配置 ==========
export NCCL_DEBUG=WARN
export MODEL_PATH=/mnt/nvme1n1/ml_research/models_cfs/qwen3-235B-Instruct-2507-FP8/
export VLLM_USE_V1=1

export COMMON_ARGS="
    --trust-remote-code
    --served-model-name /mnt/nvme1n1/ml_research/models_cfs/qwen3-235B-Instruct-2507-FP8/
    --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":8}
    --disable-log-requests
"

export VLLM_VERSION=0.13.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# DeepSeek 相关优化（Qwen3-235B 是 MoE 模型，同样适用）
export VLLM_MOE_DP_CHUNK_SIZE=64
export VLLM_DEEPEP_BUFFER_SIZE_MB=0
export VLLM_USE_DEEP_GEMM=1
export VLLM_ALL2ALL_BACKEND=deepep_low_latency
export VLLM_IGNORE_TENSOR_PLACEHOLDER=1

export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FORCE_LOAD_BALANCE=1

# Profiler 设置（可选，取消注释启用）
# export VLLM_TORCH_PROFILER_DIR=${VLLM_TORCH_PROFILER_DIR:-"./profiles"}
# rm -rf $VLLM_TORCH_PROFILER_DIR
# mkdir -p $VLLM_TORCH_PROFILER_DIR
# export VLLM_TORCH_PROFILER_WITH_STACK=0

# ========== DYCP Decode 核心配置 ==========
# 单台机器: 8 GPU, DP=2, TP=4, 无多机通信 (1 个 domain)
MAX_SEQS_PER_DP=64

args=(
    --port 8400
    $COMMON_ARGS
    --async-scheduling
    --distributed-executor-backend dmp          # DYCP 必须使用 dmp
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":262144}}'
    --max-model-len 524288
    --max-num-batched-tokens 128
    --gpu-memory-utilization 0.9
    --no-enable-prefix-caching
    --data-parallel-size 2                        # DP=2 (单机 8 卡)
    --tensor-parallel-size 4                      # TP=4
    --block-size 64
    --cp-kv-cache-interleave-size 64
    --no-enforce-eager
    --max-num-seqs ${MAX_SEQS_PER_DP}
    --enable-expert-parallel
    --dp-per-domain 2                             # 每个 domain 2 DP (1 domain)
    --num-cp-seqs 2                               # DYCP 核心参数
    --compilation-config '{"cudagraph_capture_sizes":[2, 4, 8, 10, 12, 16, 18, 24, 26, 32, 34, 64], "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes_for_cp": 2}'
    --kv-transfer-config
    '{
        "kv_connector": "CrossDPExampleConnector",
        "kv_connector_module_path": "vllm.distributed.kv_transfer.kv_connector.v1.cross_dp_example_connector",
        "kv_role": "kv_consumer",
        "kv_parallel_size": 2,
        "kv_port": "20002",
        "engine_id": "decode-0",
        "kv_rank": 1,
        "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 16
             },
             "decode": {
                    "dp_size": 2,
                    "tp_size": 4
             }
        }
    }'
)

# ========== 启动 vLLM 服务 ==========
vllm serve "${MODEL_PATH}" "${args[@]}" &> "qwen235b_dycp_single_node.log" &
