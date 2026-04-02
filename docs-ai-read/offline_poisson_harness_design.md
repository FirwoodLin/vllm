# Offline Poisson Harness Design

## Goal

Design an offline vLLM benchmark harness with the following properties:

- Does not use the OpenAI-compatible API server.
- Does not use HTTP.
- Reads workload definitions from CSV.
- Injects requests according to a Poisson arrival process.
- Allows queueing to naturally occur inside the engine.
- Measures per-request end-to-end latency and TTFT.
- Supports multi-node data-parallel inference such as `32DP+EP` and
  `16DP+2TP+2DCP+EP`.
- Compatible with `DecodeBenchConnector` for decode-instance performance
  testing.

This document is a design only. It does not imply that the code already exists.

## Why This Harness Exists

The current benchmark options split into two groups:

- Online serving benchmarks:
  `vllm bench serve` and `benchmarks/multi_turn/benchmark_serving_multi_turn.py`
  support request-rate driven workloads, but they go through HTTP and the API
  server.
- Offline benchmarks:
  `vllm bench latency` and `vllm bench throughput` avoid HTTP, but they do not
  model a continuous arrival process with queueing semantics that match online
  serving.

For the target use case, we want the workload semantics of serving benchmarks,
but we want the measurement path of offline inference.

## Requirements

The harness must support:

- CSV input.
- Poisson arrival scheduling.
- Per-request metrics:
  - `e2e_ms`
  - `ttft_ms`
  - `queued_time_ms` if available
  - basic request shape fields such as `prompt_len` and `output_len`
- Non-streaming output behavior.
- Multi-node data parallel execution.
- Expert parallel execution.
- A configuration compatible with `32DP+EP`.
- A configuration compatible with `16DP+2TP+2DCP+EP`.
- Optional integration with `DecodeBenchConnector`.

The harness should also support:

- Synthetic prompt-token generation from prompt length only.
- Deterministic replay via seed control.
- One global raw output file plus optional summary and parquet export.

## Non-Goals

- No HTTP benchmarking.
- No chat-template fidelity testing.
- No exact reproduction of external load balancer policies.
- No client-visible token streaming metrics beyond TTFT.
- No requirement to match `vllm bench serve` JSON output schema exactly.

## Key Design Choice

Use the v1 async offline engine interface directly instead of `LLM.generate()`
batch mode or any API server path.

Recommended base:

- `AsyncLLM.add_request(...)`
- `RequestOutputCollector.get()`

This refers to the vLLM v1 `AsyncLLM` defined in
`vllm/v1/engine/async_llm.py`.

Why:

- `LLM.generate()` is batch-oriented and not a natural fit for continuous
  arrivals.
- `AsyncLLM` accepts requests continuously without a harness-managed blocking
  `step()` loop.
- In pure internal DP load-balancing mode, `AsyncLLM` already subscribes to
  per-engine queue stats and dynamically routes each request to a DP instance.
- This keeps all queueing inside the engine and avoids HTTP serialization,
  network stack overhead, and API server CPU overhead.

### Dynamic DP Routing Policy

In the recommended multi-node configuration, requests are **not** assigned to a
specific DP rank ahead of time.

Instead, the frontend runs vLLM pure internal DP load balancing:

- the `DPCoordinator` collects `num_waiting_reqs` and `num_running_reqs` from
  all DP engines
- `DPLBAsyncMPClient` subscribes to those counts
- for each new request with no explicit `data_parallel_rank`, the frontend
  chooses the DP engine with minimum:

```text
score = waiting * 4 + running
```

This is the policy already implemented in vLLM v1 and should be reused by the
harness rather than reimplemented differently.

### Why `LLMEngine.step()` Is Not the Recommended Base

`LLMEngine.step()` is a good fit for single-engine step-driven harnesses, but it
is not the right base for the target pure-internal-LB DP design:

- in vLLM, offline-mode `LLMEngine` setup disables the coordinator path used
  for cluster-wide internal DP load balancing
- the harness would have to invent a second routing/control plane to recover
  dynamic DP dispatch
- `step()` is a blocking IPC call, so the harness cannot inject arrivals while
  waiting for one scheduler iteration to finish

For the target use case, `AsyncLLM` is the correct primary interface.

### Concrete `AsyncLLM` Usage Pattern

For the harness, the intended per-request control flow is:

1. Build `AsyncEngineArgs` for the target topology.
2. Create an `AsyncLLM` frontend.
3. For each request, build token-ID prompt inputs and
   `SamplingParams(output_kind=RequestOutputKind.FINAL_ONLY)`.
4. Call `await engine.add_request(...)` and keep the returned
   `RequestOutputCollector`.
5. In a separate completion task, call `await collector.get()` once and treat
   that result as the finished request output.
6. Read per-request metrics from the final `RequestOutput.metrics`.

Single-node or frontend-managed-local-engine case:

```python
engine_args = AsyncEngineArgs(...)
engine = AsyncLLM.from_engine_args(engine_args)

sampling_params = SamplingParams(
    max_tokens=expected_output_len,
    temperature=0.0,
    output_kind=RequestOutputKind.FINAL_ONLY,
)

collector = await engine.add_request(
    request_id=request_id,
    prompt=prompt_token_ids,  # or {"prompt_token_ids": prompt_token_ids}
    params=sampling_params,
    data_parallel_rank=None,
)

final_output = await collector.get()
assert final_output.finished
metrics = final_output.metrics
```

Important details from the current vLLM v1 implementation:

- `AsyncLLM.add_request(...)` returns a `RequestOutputCollector`, not the final
  output directly.
- With `RequestOutputKind.FINAL_ONLY`, the output processor suppresses
  intermediate updates and only enqueues the final `RequestOutput`.
- The harness should therefore use `add_request(...) + collector.get()` rather
  than `engine.generate()` as its primary benchmark path.
- Length-only CSV mode should pass token IDs directly as the prompt input. Raw
  text prompt generation is unnecessary.

### Multi-Node `AsyncLLM` Construction

For multi-node frontend + externally managed headless engines, the frontend
should not rely on `AsyncLLM.from_engine_args(...)` alone, because that helper
does not accept externally supplied client socket addresses.

Instead, the harness should:

1. Build `vllm_config = AsyncEngineArgs(...).create_engine_config(...)`.
2. Start the local and remote headless engine processes using the harness's own
   `headless-engine` role.
3. Create the frontend with:

```python
engine = AsyncLLM.from_vllm_config(
    vllm_config=vllm_config,
    client_addresses={
        "input_address": ...,
        "output_address": ...,
        "stats_update_address": ...,
    },
)
```

This reuses the existing vLLM v1 engine-core plumbing without introducing HTTP.
Under the hood, `EngineCoreClient.make_async_mp_client(...)` will construct
`DPLBAsyncMPClient` when `data_parallel_size > 1` and
`data_parallel_external_lb = False`, so the internal DP routing policy remains
the standard vLLM pure-internal-LB path.

## Input Format

### Primary CSV Mode

The harness should directly reuse the same minimal CSV schema already accepted
by vLLM random-length serving benchmarks:

```csv
prompt_len,output_len
1024,256
4096,512
8192,128
```

Properties:

- Only two required columns:
  - `prompt_len`
  - `output_len`
- Both must be positive integers.

This is the preferred mode for cluster benchmarking because it removes
tokenization variability from the client path.

### Optional Extended CSV Mode

Future-compatible optional columns:

- `request_id`
- `priority`
- `seed`
- `prompt`
- `prompt_token_ids`

Recommended behavior:

- If `prompt_token_ids` exists, use it directly.
- Else if `prompt` exists, tokenize it.
- Else synthesize tokens from `prompt_len`.

For the initial implementation, supporting only `prompt_len,output_len` is
sufficient.

## Prompt Construction Strategy

For `prompt_len,output_len` CSV input, generate exact-length token sequences
locally, similar to the random CSV handling in vLLM benchmarks.

Recommended rules:

- Use token IDs directly rather than text prompt generation.
- Exclude special tokens.
- Generate deterministic sequences from:
  - global seed
  - global request index or stable request ID
- Keep prompt construction independent of the DP rank eventually chosen by the
  load balancer.

This makes prompt generation:

- reproducible
- low-overhead
- independent of tokenizer decode or chat formatting

## Arrival Model

### Target Semantics

The benchmark should simulate online arrivals even though inference is offline.

For a Poisson process:

- Inter-arrival times follow an exponential distribution.
- Requests may arrive while earlier requests are still waiting or running.
- Queueing naturally occurs when the system is overloaded.

### Recommended Rate Model

Define:

- `lambda_total_rps`: total target request rate for the full cluster
- one global arrival stream driven by the offline harness frontend

The harness samples one global Poisson process:

- `delta_t ~ Exponential(lambda_total_rps)`

and dispatches each arrival at submission time using the live DP load stats
described above.

### Why Global Poisson + Dynamic Dispatch Is Preferred

It preserves the semantics you want:

- the request stream is global
- requests are not pre-bound to a DP rank
- queueing still happens naturally inside the selected engine
- the routing policy adapts to current `running` and `waiting` load

It also matches the current vLLM pure internal DP load balancer instead of
modeling a different system.

## Request Lifecycle

At a high level, the harness should work like this:

1. Load or generate one global request sequence.
2. Precompute one global Poisson arrival trace.
3. Start the `AsyncLLM` frontend on node 0 and headless engine processes on the
   other nodes.
4. Whenever the next arrival time has passed:
   - build the request inputs
   - submit the request with `data_parallel_rank=None`
   - let vLLM choose the target DP engine using internal LB
   - spawn an async completion task that waits for the final output
5. As completion tasks finish, persist per-request records.
6. After all arrivals have been submitted, wait for all in-flight requests to
   finish and write the final summary.

Important:

- The harness must not wait for one request to finish before submitting the next
  one.
- Requests must not be assigned to a DP rank before submission.
- `submit_ts_ns` must record the actual wall-clock time of
  `AsyncLLM.add_request()`, not the theoretical Poisson arrival time.
- The harness should use monotonic time for arrival scheduling, but wall-clock
  time for persisted request timestamps.
- Do not pre-stamp `arrival_time` at trace-generation time. Let the frontend
  stamp arrival at actual submission time so TTFT remains consistent.

## Non-Streaming Output

The harness should operate in non-streaming mode from the benchmark client's
perspective.

Recommended behavior:

- Only write one result record per finished request.
- Do not expose intermediate token deltas to the benchmark output.

This does not mean the engine cannot internally step token-by-token. It only
means benchmark accounting is based on the final completion event.

## DecodeBenchConnector Integration

### What DecodeBenchConnector Does

`DecodeBenchConnector` is a `KVConnectorBase_V1` implementation located at
`vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py`.

It emulates disaggregated prefill-decode by filling the allocated KV cache
blocks with dummy non-zero values on the first scheduler step for each request.
This allows decode-instance performance testing with large input sequence lengths
without running a real prefill.

Enabled via `kv_transfer_config`:

```json
{
  "kv_connector": "DecodeBenchConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "fill_mean": 0.015,
    "fill_std": 0.0
  }
}
```

### How It Interacts with the Harness

`DecodeBenchConnector` hooks into the v1 scheduler and worker internals and is
invoked automatically during internal scheduler iterations. No harness-level
changes are
needed to activate it beyond passing the correct `kv_transfer_config` in
`AsyncEngineArgs`.

The connector is invoked inside the engine at two points:

1. **Scheduler side** (`DecodeBenchConnectorScheduler`): On the first scheduling
   of a new request, `get_num_new_matched_tokens()` returns `(N - 1, False)`,
   telling the scheduler to treat all tokens except the last as externally
   supplied. Blocks are allocated via `update_state_after_alloc()`.
2. **Worker side** (`DecodeBenchConnectorWorker`): `start_load_kv()` fills the
   allocated blocks with dummy values synchronously before the forward pass.

### TTFT Semantics Under DecodeBenchConnector

With `DecodeBenchConnector` active, the effective request lifecycle is:

```
arrive → queue → [step N]: KV fill (sync) + 1 decode token → [step N+1..]: decode → finish
```

There is no real prefill step. As a result, TTFT measures:

```
TTFT = queue_time + kv_fill_time + first_decode_kernel_time
```

This is intentional. The harness is measuring decode-instance performance, not
prefill performance.

Internally, the worker-side fill cost is first collected in
`DecodeBenchConnectorWorkerMetadata.req_batch_load_kv_ns`, then propagated into
`RequestTTFTTrace.first_batch_load_kv_ns` for the request's first token. The
harness should therefore extract `kv_fill_ms` from the final
`RequestOutput.metrics.ttft_trace.first_batch_load_kv_ns`, not by directly
reading scheduler worker metadata.

### KV Fill Batching Behavior

When multiple new requests are scheduled in the same step, the connector fills
their KV cache blocks sequentially inside one `start_load_kv()` call. At high
arrival rates, a large first-step batch will cause the KV fill time to be
distributed across all requests in that batch, inflating the TTFT of later
requests in the batch.

This reflects real disaggregated-decode behavior and should be preserved in
measurements rather than corrected.

### Disabling the Connector

If the harness is run without `kv_transfer_config` (or with a null connector),
real prefill will execute for each request. In this case:

- `kv_fill_ms` will be zero or absent.
- TTFT will measure `queue_time + prefill_time + first_decode_kernel_time`.
- `prefill_time_ms` will be populated from engine metrics.

## Metrics

### Per-Request Metrics

Each finished request record should include:

- `request_id`
- `submit_ts_ns`
- `finish_ts_ns`
- `e2e_ms`
- `ttft_ms`
- `queued_time_ms`
- `kv_fill_ms` (populated when `DecodeBenchConnector` is active and
  `enable_logging_ttft_timing_details` is enabled; zero otherwise)
- `prefill_time_ms` (always populated from engine metrics as
  `first_token_ts - scheduled_ts`; with `DecodeBenchConnector` active, this is
  the first-token interval rather than true model prefill time)
- `decode_time_ms`
- `inference_time_ms`
- `prompt_len`
- `expected_output_len`
- `actual_output_tokens`
- `finish_reason`
- `num_cached_tokens`
- `is_error`
- `error_message`

### Metric Sources

Use two timing sources:

- Harness-side timestamps:
  - `submit_ts_ns`: wall-clock time of `AsyncLLM.add_request()` call
  - `finish_ts_ns`: wall-clock time when the finished output is observed
  - `e2e_ms = (finish_ts_ns - submit_ts_ns) / 1e6`
- Engine-side final request metrics derived from `RequestOutput.metrics`:
  - `ttft_ms = first_token_latency * 1e3`
  - `queued_time_ms = (scheduled_ts - queued_ts) * 1e3`
  - `prefill_time_ms = (first_token_ts - scheduled_ts) * 1e3`
  - `decode_time_ms = (last_token_ts - first_token_ts) * 1e3`
  - `inference_time_ms = (last_token_ts - scheduled_ts) * 1e3`
  - `kv_fill_ms = ttft_trace.first_batch_load_kv_ns / 1e6`

Harness policy:

- The harness must force `log_stats=True`, otherwise
  `RequestOutput.metrics` may be absent.
- The harness must force `enable_logging_ttft_timing_details=True`, otherwise
  `kv_fill_ms` is unavailable.
- If the user passes conflicting settings, the harness should fail fast at
  startup rather than silently degrading metrics.

Recommended interpretation:

- `e2e_ms` is the primary metric for the benchmark.
- `ttft_ms` explains user-visible response start; its composition differs
  depending on whether `DecodeBenchConnector` is active (see above).
- `queued_time_ms` explains overload behavior.
- `kv_fill_ms` isolates the KV fill overhead when running decode-bench mode.

### TTFT Source

TTFT should come from the final `RequestOutput.metrics.first_token_latency`, not
be reconstructed from stream events. `kv_fill_ms` should come from the final
`RequestOutput.metrics.ttft_trace.first_batch_load_kv_ns`.

### Summary Metrics

Global summary should include:

- total requests
- successful requests
- failed requests
- benchmark runtime
- achieved request throughput
- connector mode (`none` | `decode_bench`)
- `e2e_ms`:
  - mean, p50, p90, p95, p99
- `ttft_ms`:
  - mean, p50, p90, p95, p99
- `queued_time_ms`:
  - mean, p50, p90, p95, p99
- `kv_fill_ms` (when connector is active):
  - mean, p50, p90, p95, p99

Definitions:

- `benchmark runtime_s = (last_measured_finish_ts_ns - first_measured_submit_ts_ns)
  / 1e9`
- `achieved request throughput = measured_request_count / benchmark runtime_s`
- The measured window starts at the first measured request submission and ends
  at the last measured request finish.

## Multi-Node Execution Model

### Supported Topologies

#### 32DP+EP

Typical layout:

- 4 nodes, 8 GPUs per node
- `data_parallel_size = 32`
- `data_parallel_size_local = 8`
- `tensor_parallel_size = 1`
- `decode_context_parallel_size = 1`
- `enable_expert_parallel = True`

Total GPU count: 32

#### 16DP+2TP+2DCP+EP

Typical layout:

- 4 nodes, 8 GPUs per node
- `data_parallel_size = 16`
- `data_parallel_size_local = 4`
- `tensor_parallel_size = 2`
- `decode_context_parallel_size = 2`
- `enable_expert_parallel = True`

Total GPU count: 16 × 2 × 2 = 64. Each DP rank occupies 4 GPUs (2 TP × 2 DCP).

Note: `data_parallel_size_local = dp_size / dp_num_nodes`, not GPU count per
node. With 4 nodes and `dp_size = 16`, each node hosts 4 local DP ranks.

### Process Structure

Recommended structure is a **pure internal LB** deployment:

- node 0 runs the offline harness frontend and one `AsyncLLM` client
- node 0 also launches its local DP engine processes
- nodes 1..N-1 run headless engine-only processes
- the node 0 frontend manages local and remote DP engines as one global pool
- one `DPCoordinator` on node 0 publishes per-engine queue stats for routing

This differs from the older offline `data_parallel.py` pattern: there is **not**
one harness worker per local DP rank, because that would pre-assign requests to
specific ranks before dispatch.

Environment variables / topology fields still need to describe the DP layout:

```
VLLM_DP_SIZE=<dp_size>
VLLM_DP_MASTER_IP=<master_ip>
VLLM_DP_MASTER_PORT=<master_port>
```

The frontend process owns:

- one global arrival stream
- one global request table
- all per-request result accounting

Headless engine processes own only the model execution state for their local DP
ranks; they do not own arrival streams or result files.

### Frontend / Engine-Core Boundary

The harness should reuse the vLLM v1 frontend-to-engine-core split, but without
the API server:

- for single-node or locally managed runs, the frontend can use
  `AsyncLLM.from_engine_args(...)`
- for multi-node runs with externally managed headless engines, the frontend
  should use `AsyncLLM.from_vllm_config(..., client_addresses=...)`
- the frontend process owns the `AsyncLLM` instance and its
  `RequestOutputCollector` objects
- `AsyncLLM` owns an `engine_core` client created by
  `EngineCoreClient.make_async_mp_client(...)`
- in pure internal DP load balancing mode, that client is
  `DPLBAsyncMPClient`
- the `headless-engine` role should launch only engine-core processes, e.g.
  via `CoreEngineProcManager(..., local_client=False, handshake_address=...)`,
  and should not start an API server
- each submitted request should use `data_parallel_rank=None` in the default
  benchmark mode
- the routing decision remains inside `DPLBAsyncMPClient.get_core_engine_for_request()`

The harness must not add a second routing policy outside vLLM.

### Request Dispatch Modes

There should be two modes, but only one should be the default.

#### Mode A: Dynamic Global Dispatch

The frontend:

- reads the CSV once
- samples one global Poisson trace
- submits each request with `data_parallel_rank=None`
- lets vLLM internal LB pick the DP engine using:

```text
score = waiting * 4 + running
```

Pros:

- matches the requested semantics
- matches existing vLLM routing behavior
- naturally adapts to uneven `running` / `waiting` counts
- preserves one exact global request trace

Cons:

- requires one centralized frontend process
- request-to-rank assignment becomes a runtime property, not an input file

This is the recommended default mode.

#### Mode B: Explicit Rank Replay

The frontend:

- reads a precomputed request table that already includes
  `data_parallel_rank`
- bypasses internal LB for those requests

Pros:

- exact replay of a prior routed trace
- useful for debugging or A/B comparisons

Cons:

- does not satisfy the desired dynamic-routing semantics
- should not be the default benchmark mode

This mode should exist only as an explicit replay/debug option.

## Result Files

Recommended output layout:

```text
results/
  run_meta.json
  requests.jsonl
  summary.json                 # optional
  merged_requests.parquet      # optional
```

### `run_meta.json`

Contains:

- model
- topology (`dp_size`, `tp_size`, `dcp_size`, `ep_enabled`)
- routing mode (`internal_dplb`)
- dispatch policy (`waiting_x4_plus_running`)
- connector mode
- CSV path
- request rate
- seed
- benchmark start time
- engine args

### `requests.jsonl`

One line per finished request in completion order. The file is append-oriented;
request order in this file is not semantically meaningful for analysis.

Example without DecodeBenchConnector:

```json
{
  "request_id": "r17-000123",
  "submit_ts_ns": 1712345678901234567,
  "finish_ts_ns": 1712345679987654321,
  "e2e_ms": 1086.42,
  "ttft_ms": 233.71,
  "queued_time_ms": 180.14,
  "kv_fill_ms": 0.0,
  "prefill_time_ms": 48.55,
  "decode_time_ms": 837.23,
  "inference_time_ms": 885.78,
  "prompt_len": 4096,
  "expected_output_len": 256,
  "actual_output_tokens": 256,
  "finish_reason": "length",
  "is_error": false
}
```

Example with DecodeBenchConnector active:

```json
{
  "request_id": "r17-000456",
  "submit_ts_ns": 1712345678901234567,
  "finish_ts_ns": 1712345679987654321,
  "e2e_ms": 1064.30,
  "ttft_ms": 218.90,
  "queued_time_ms": 175.20,
  "kv_fill_ms": 38.14,
  "prefill_time_ms": 52.80,
  "decode_time_ms": 845.40,
  "inference_time_ms": 898.20,
  "prompt_len": 4096,
  "expected_output_len": 256,
  "actual_output_tokens": 256,
  "finish_reason": "length",
  "is_error": false
}
```

Note: with `DecodeBenchConnector`, `prefill_time_ms` still follows the standard
engine metric definition `first_token_ts - scheduled_ts`. It is therefore a
first-token interval, not literal prefill compute time. `kv_fill_ms` separately
captures the KV fill component. TTFT ≈ `queued_time_ms + kv_fill_ms +
first_decode_step`.

### `summary.json`

Contains cluster-level aggregates and percentiles. Also records the connector
mode so results from different runs are not inadvertently compared.

## CLI Proposal

### 32DP+EP (4 nodes, node 0 frontend)

```bash
python offline_poisson_harness.py frontend \
  --model /path/to/model \
  --input-csv /path/to/requests.csv \
  --arrival-process poisson \
  --request-rate 40 \
  --seed 1234 \
  --output-dir /path/to/output \
  --data-parallel-size 32 \
  --data-parallel-size-local 8 \
  --tensor-parallel-size 1 \
  --decode-context-parallel-size 1 \
  --enable-expert-parallel \
  --data-parallel-backend mp \
  --nnodes 4 \
  --node-rank 0
```

### 32DP+EP (4 nodes, headless engine node)

```bash
python offline_poisson_harness.py headless-engine \
  --model /path/to/model \
  --data-parallel-size 32 \
  --data-parallel-size-local 8 \
  --tensor-parallel-size 1 \
  --decode-context-parallel-size 1 \
  --enable-expert-parallel \
  --data-parallel-backend mp \
  --nnodes 4 \
  --node-rank 1
```

Run the same headless-engine command on nodes 2 and 3 with `--node-rank 2` and
`--node-rank 3`.

### 16DP+2TP+2DCP+EP (4 nodes, node 0 frontend)

```bash
python offline_poisson_harness.py frontend \
  --model /path/to/model \
  --input-csv /path/to/requests.csv \
  --arrival-process poisson \
  --request-rate 40 \
  --seed 1234 \
  --output-dir /path/to/output \
  --data-parallel-size 16 \
  --data-parallel-size-local 4 \
  --tensor-parallel-size 2 \
  --decode-context-parallel-size 2 \
  --enable-expert-parallel \
  --data-parallel-backend mp \
  --nnodes 4 \
  --node-rank 0
```

Headless nodes use the same topology flags with `headless-engine` and their own
`--node-rank`.

### With DecodeBenchConnector

Add to either topology above:

```bash
  --kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":0.015,"fill_std":0.0}}'
```

### Harness-specific flags

- `frontend|headless-engine` role
- `--input-csv`
- `--csv-format length_csv|prompt_csv`
- `--request-rate`
- `--seed`
- `--max-requests`
- `--warmup-requests`
- `--output-dir`
- `--save-merged-parquet`
- `--arrival-process poisson`
- `--request-id-prefix`
- `--routing-mode internal_dplb|explicit_rank_replay`

### Engine-related flags

- all relevant `AsyncEngineArgs` passed through directly
- DP/TP/DCP/EP topology flags
- `--data-parallel-backend mp`
- `--nnodes`
- `--node-rank`
- `--kv-transfer-config` for connector configuration
- memory and compilation flags matching the cluster setup

## Scheduling Loop Details

Frontend only:

1. Build one global request table from CSV.
2. Generate one global `arrival_deadline_ns` trace using monotonic time.
3. Keep an index of the next request to submit and a set of in-flight tasks.
4. While `arrival_deadline_ns[next] <= now`:
   - build request inputs as prompt token IDs and `SamplingParams` with
     `output_kind=RequestOutputKind.FINAL_ONLY`
   - record `submit_ts_ns = time.time_ns()`
   - call `collector = await engine.add_request(..., data_parallel_rank=None)`
   - spawn a task that waits on `await collector.get()` for the final
     `RequestOutput`
5. In each completion task:
   - wait until the final output is available
   - record `finish_ts_ns = time.time_ns()`
   - derive engine-side timing fields from `RequestOutput.metrics`
   - write one request record
6. Exit only when:
   - all requests have been submitted
   - all in-flight completion tasks have finished

Implementation notes:

- The harness itself should not call `step()`.
- Use `time.perf_counter_ns()` for scheduling against arrival deadlines.
- Use `time.time_ns()` for persisted `submit_ts_ns` / `finish_ts_ns`.
- Do not pre-assign `data_parallel_rank` in the default mode.
- Do not attempt to correct `submit_ts_ns` back to the theoretical Poisson
  arrival time. Record the actual submission time.
- If the event loop wakes up late and multiple deadlines have passed, submit all
  overdue arrivals in one loop iteration until the next deadline is in the
  future.

## Failure Handling

Each request result should indicate whether it failed.

Recommended failure classes:

- engine exception during submission
- engine exception during async output handling
- finished with empty or malformed output
- process termination or timeout

Policy:

- persist per-request failures in raw files
- do not drop failed requests from the summary
- expose failure count and failure ratio at the cluster summary level

## Synchronization

Only minimal cross-node synchronization is required.

Recommended coordination:

- headless engine nodes must be launched before the frontend begins dispatch
- frontend startup should block until all engines have completed handshake
- no per-request coordination should be added in the harness

Because request submission and result writing are centralized in the frontend:

- workers do not write result files, so there is no cross-worker merge step
- `summary.json` can be written directly by the frontend after all completion
  tasks finish

## Why Queueing Is Faithful Enough

This harness does not emulate an external HTTP ingress queue. It emulates queue
build-up inside the vLLM engine pool selected by pure internal DP load
balancing.

For the intended use case, this is acceptable because the main goal is to
remove:

- HTTP parsing cost
- API server scheduling cost
- client-side network effects

while keeping:

- engine request queueing
- vLLM internal DP routing behavior (`waiting * 4 + running`)
- prefill/decode contention
- multi-node DP/EP communication cost
- KV fill cost (when `DecodeBenchConnector` is active)

## Limitations

- It models vLLM's current pure internal DP load balancer, not an arbitrary
  external routing layer.
- It does not model TCP or API-server backpressure.
- The benchmark uses one centralized frontend. At sufficiently high request
  rates, frontend CPU scheduling can become part of the measured system.
- If upstream vLLM changes the internal DP scoring policy, the harness behavior
  will change with it unless pinned to a specific version.
- When `DecodeBenchConnector` is active, `ttft_ms` measures
  `queue_time + kv_fill_time + first_decode_step`, not true prefill TTFT.
  Results must not be compared directly to runs without the connector.
- `kv_fill_ms` is only available when
  `enable_logging_ttft_timing_details = True` in the observability config.
- Engine-side per-request timing fields require `log_stats=True`.

## Recommended Implementation Order

Phase 1:

- Single-node harness
- CSV length input
- global Poisson arrivals
- `AsyncLLM` with `RequestOutputKind.FINAL_ONLY`
- per-request `e2e_ms`, `ttft_ms`, `queued_time_ms`, and `prefill/decode`
  timing
- JSONL outputs
- `DecodeBenchConnector` passthrough via `kv_transfer_config`
- `kv_fill_ms` extraction from `RequestOutput.metrics.ttft_trace`

Phase 2:

- Multi-node pure internal LB deployment
- node 0 frontend + remote headless engine nodes

Phase 3:

- optional explicit-rank replay mode
- optional prompt-text CSV mode
- parquet export
- richer analysis scripts

## Recommendation

For the target case (CSV + Poisson arrivals + multi-node DP/EP +
DecodeBenchConnector + dynamic request dispatch), the first practical milestone
should be:

- `length_csv` input only
- pure internal LB
- `32DP+EP` topology
- node 0 centralized frontend
- `DecodeBenchConnector` enabled via `kv_transfer_config`
- non-streaming final-only accounting
- per-request `e2e_ms`, `ttft_ms`, `queued_time_ms`, and `kv_fill_ms`
- one global `requests.jsonl`
- `connector_mode` recorded in `run_meta.json` to prevent cross-mode
  comparison errors

That gives the workload model you want while removing HTTP and API server
overhead from the measurement path and preserving dynamic DP scheduling based on
live `running` / `waiting` load. The `16DP+2TP+2DCP+EP` topology requires no
harness changes beyond different `AsyncEngineArgs` values and can be tested in
the same phase.
