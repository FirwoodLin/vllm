# DP Imbalance Profiling Scripts

This directory contains a minimal setup for the following run shape:

- model: `/mnt/nvme1n1/ml_research/models/deepseek-v3-1024k`
- 2 nodes
- `dp=2`, `tp=8`, `dcp=8`
- expert parallel enabled
- EP all2all backend: `deepep_low_latency`
- DCP comm backend: `a2a`
- `--load-format dummy`
- `DecodeBenchConnector`
- cudagraph mode: `FULL_DECODE_ONLY`
- routing simulation: `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random`
- per DP rank traffic:
  - `63 x 4k` requests
  - `1 x 256k` request
  - output length `64`
- profiling target:
  - capture the middle `32` decode iterations

## Files

- `start_2node_dp2_tp8_dcp8_serve.sh`
  - run on each node with `zsh`
  - node 0 starts the API server
  - node 1 runs `--headless`
- `send_dp_imbalance_profile_requests.py`
  - calls:
    - `POST /pause?mode=keep`
    - sends all requests with `X-data-parallel-rank`
    - `POST /start_profile`
    - `POST /resume`
    - waits for completion
    - `POST /stop_profile`

## Commands

Node 0:

```zsh
MASTER_ADDR=<node0_ip> NODE_RANK=0 HOST=0.0.0.0 PORT=8000 \
  zsh /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/profile-dp-imbalance/start_2node_dp2_tp8_dcp8_serve.sh
```

Node 1:

```zsh
MASTER_ADDR=<node0_ip> NODE_RANK=1 \
  zsh /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/profile-dp-imbalance/start_2node_dp2_tp8_dcp8_serve.sh
```

From node 0, after the service is ready:

```zsh
python /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/profile-dp-imbalance/send_dp_imbalance_profile_requests.py \
  --base-url http://127.0.0.1:8000 \
  --model /mnt/nvme1n1/ml_research/models/deepseek-v3-1024k

# Only send short requests:
python /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/profile-dp-imbalance/send_dp_imbalance_profile_requests.py \
  --base-url http://127.0.0.1:8000 \
  --model /mnt/nvme1n1/ml_research/models/deepseek-v3-1024k \
  --short-only
```

## Profiling Window

These scripts assume `DecodeBenchConnector` is enabled.

For this workload:

- each request has one remaining context iteration after dummy KV fill
- each request then decodes `64` output tokens

So the default profiling window is:

- `delay_iterations = 18`
- `max_iterations = 32`

That captures:

- skip `1` context iteration
- skip first `16` decode iterations
- capture decode iterations `17..48`

## Notes

- `VLLM_SERVER_DEV_MODE=1` is set automatically so `/pause`, `/resume`,
  `/start_profile`, and `/stop_profile` are available.
- `VLLM_RPC_TIMEOUT=1800000` is set automatically because profiler flush can
  take a long time on `/stop_profile`.
- If request bodies are large enough that not all of them are queued before
  resume, increase `--queue-settle-seconds` in the Python script.
- If your cluster auto-picks the wrong NIC, set `GLOO_SOCKET_IFNAME` and/or
  `NCCL_SOCKET_IFNAME` before running the serve script.
