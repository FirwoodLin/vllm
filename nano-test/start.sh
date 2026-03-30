export MODEL=/mnt/nvme1n1/ml_research/models/deepseek-v3  
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export NODE_RANK=0
export BATCH_SIZE=1024
export INPUT_LEN=4096
export OUTPUT_LEN=16
export ATTENTION_BACKEND=FLASHMLA
# ag_rs or a2a
export DCP_COMM_BACKEND=a2a
export GLOO_SOCKET_IFNAME=eth0
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY="uniform_random"

  # --enable-expert-parallel \

nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  -o nsys_dp1_tp8_dcp8_${DCP_COMM_BACKEND}_bs${BATCH_SIZE}_inputlen${INPUT_LEN}_node${NODE_RANK} \
vllm bench latency \
  --model "$MODEL" \
  --nnodes 1 \
  --node-rank ${NODE_RANK} \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  -dp 1 \
  --data-parallel-size-local 1 \
  -tp 8 \
  -dcp 8 \
  --dcp-comm-backend "$DCP_COMM_BACKEND" \
  --num-iters-warmup 5 \
  --num-iters 1 \
  --batch-size "$BATCH_SIZE" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --attention-backend "$ATTENTION_BACKEND" \
  --all2all-backend deepep_low_latency \
  --cudagraph-capture-sizes 1 2 4 ${BATCH_SIZE}  \
  --load-format dummy \
  --gpu-memory-utilization 0.85 \
  --profiler-config.profiler cuda \
  --profile \
  --kv-transfer-config '{ "kv_connector": "DecodeBenchConnector", "kv_role": "kv_both", "kv_connector_extra_config": { "fill_mean": 0.015, "fill_std": 0.0 } }' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
