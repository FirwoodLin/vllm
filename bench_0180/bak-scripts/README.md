This directory contains a `vllm bench sweep` bundle for manual multi-node MP serving.

The fixed `4dp8tp` example still targets 4 nodes, while `batch_start_manual.sh`
and `launch_4node_mp_manual.sh` now support a configurable number of nodes.

The default topology in `cluster.env` is still 4 nodes with 8 GPUs per node:

- `data_parallel_backend=mp`
- `tensor_parallel_size=8`
- `data_parallel_size=4`
- `async_scheduling=true`

Files:

- `launch_4node_mp_4dp8tp.sh`: legacy fixed 4-node launcher kept for reference
- `launch_4node_mp_manual.sh`: topology-aware launcher used underneath the sweep wrapper
- `start_sweep_serve.sh`: sweep-facing wrapper that resolves `MASTER_ADDR` before calling the shared launcher
- `batch_start_manual.sh`: batch entrypoint that derives valid `dp*` strategies from the cluster layout
- `reset_sweep_caches.sh`: cache reset hook used by `--after-bench-cmd`
- `cluster.env`: cluster-specific settings
- `serve_params.4dp8tp_async_mp.json`: example serve sweep params
- `run_sweep_example.sh`: example sweep entrypoint

Expected usage:

1. Run `run_sweep_example.sh` on `h200-rjob0`.
2. Make sure `/root/.ssh/config` contains the `h200-rjob0..3` aliases and keys.
3. Keep `MASTER_ADDR=auto` unless you intentionally want to pin rank 0 to a specific IPv4.
4. Make sure `LOCAL_ENV_SCRIPT` / `REMOTE_ENV_SCRIPT` are set if `vllm` is not already on `PATH`.
5. For `batch_start_manual.sh`, set `REMOTE_NODE_SSHS` and `GPUS_PER_NODE` in `cluster.env` to match your actual cluster.
6. If `vllm` only exists in your zsh startup environment, set `LOCAL_SHELL=zsh`, `LOCAL_SHELL_FLAGS=-ic`, `REMOTE_SHELL=zsh`, and `REMOTE_SHELL_FLAGS=-ic`.

Notes:

- The launcher treats everything after the final `--` as `vllm serve` args.
- That is required so `vllm bench sweep serve --serve-params ...` can append flags.
- Use `SWEEP_HOST=127.0.0.1` unless you specifically want the HTTP server bound to another address.
- `MASTER_ADDR=auto` discovers the outbound non-loopback IPv4 of `h200-rjob0` at launch time.
- Remote ranks are started via `h200-rjob1`, `h200-rjob2`, and `h200-rjob3`.
- `launch_4node_mp_manual.sh` prefers `REMOTE_NODE_SSHS=node1,node2,...` and falls back to `NODE1_SSH..NODE3_SSH`.
- The launchers default to `bash -lc`, but you can switch to `zsh -ic` with `LOCAL_SHELL[_FLAGS]` and `REMOTE_SHELL[_FLAGS]`.
- `start_sweep_serve.sh` only emits boolean serve flags when they are enabled, which avoids invalid `--no-disable-log-stats` arguments from sweep JSON.
- Supported `dp*` strategy keys in `batch_start_manual.sh` depend on topology:
  - 4 nodes x 8 GPUs: `dp4,dp8,dp16,dp32`
  - 2 nodes x 8 GPUs: `dp2,dp4,dp8,dp16`
- Non-default topologies add a suffix like `__n2g8` to experiment names so 2-node and 4-node runs do not overwrite each other.
- `batch_start_manual.sh` also supports `--cudagraph-mode MODE` and will pass it through as `--compilation-config '{"cudagraph_mode":"MODE"}'`.
- Example:
  `CLUSTER_ENV=/vllm/bench_0180/cluster.2node.env bash /vllm/bench_0180/batch_start_manual.sh --strategies dp2 --datasets issue01_halfhalf --rates 30 --num-prompts 60 --decodebench-connector --cudagraph-mode FULL_DECODE_ONLY`
