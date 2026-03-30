# Qwen3.5 Prefill Benchmark

下面只保留两段最常用命令: 一段启动服务, 一段做 prefill bench。

## Serve

  --all2all-backend deepep_high_throughput \
```bash
VLLM_DEEP_GEMM_WARMUP="skip" VLLM_MOE_ROUTING_SIMULATION_STRATEGY="uniform_random" vllm serve /mnt/nvme1n1/ml_research/models/models--Qwen--Qwen3.5-397B-A17B-FP8 \
  --port 8060 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --data-parallel-size 8 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --enable-expert-parallel \
  --max-model-len 2048 \
  --max-num-seqs 16 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --load-format dummy \
  --no-enable-prefix-caching \
  --api-server-count 1 2>&1 | tee /vllm/qwen35prefill_serve_eplb_8dp.log
```
  --max-num-batched-tokens 512 \

## Bench

`output_len=1`、`request_rate=inf`，偏向测 prefill。
这里把 `cudagraph_mode` 固定成 `PIECEWISE`，因为这个场景只关心 prefill，不需要 decode 相关的 full cudagraph 路径。

```bash
vllm bench serve \
  --backend openai \
  --host 127.0.0.1 \
  --port 8060 \
  --endpoint /v1/completions \
  --dataset-name random \
  --model /mnt/nvme1n1/ml_research/models/models--Qwen--Qwen3.5-397B-A17B-FP8 \
  --random-input-len 1024 \
  --random-output-len 1 \
  --num-prompts 16 \
  --max-concurrency 16 \
  --request-rate inf \
  --ignore-eos \
  --num-warmups 1 2>&1 | tee /vllm/qwen35prefill_bench-1k-8dp_eplb.log
```

```
vllm bench serve \
  --backend openai \
  --port 8060 \
  --host 127.0.0.1 \
  --endpoint /v1/completions \
  --dataset-name random \
  --model /mnt/nvme1n1/ml_research/models/models--Qwen--Qwen3.5-397B-A17B-FP8 \
  --random-input-len 1024 \
  --random-output-len 1 \
  --num-prompts 8 \
  --max-concurrency 8 \
  --request-rate inf \
  --ignore-eos \
  --num-warmups 1 2>&1 | tee /vllm/qwen35prefill_bench-1k.log
```

## 其他可改项

如果要换压测点位，通常只改两处:

- `--random-input-len`
- `--num-prompts`

例如:

- `128 x 2048`
- `64 x 4096`
- `32 x 8192`
