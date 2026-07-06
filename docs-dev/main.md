# IQR-Aware Lexicographical Decode DP Load Balancing

## Scope

This design targets the P/D disaggregated serving path and only covers decode-side
request placement across data-parallel decode engines.

Out of scope:

- Prefill scheduling and prefill-side load balancing.
- Attention kernels, PagedAttention, sampling, and model runner changes.
- Per-engine token-level scheduling inside `vllm/v1/core/sched/scheduler.py`.
- DYCP / `dp_per_domain > 1` / `CrossDPScheduler` support in the first version.
- Runtime request migration after a decode request has already been admitted.

The intended first supported shape is `dp_per_domain == 1`, where each decode DP
rank maps to one decode EngineCore process and owns an independent KV cache.

## Goal

Implement Algorithm 3, IQR-aware lexicographical decode scheduling, as a decode
DP load-balancing policy.

For each decode admission round, the front-end collects the newly arrived
auto-routed decode requests as a batch `R`. `R` excludes requests with an
explicit `data_parallel_rank`, because those requests are already routed by the
caller.

For each batch `R`:

1. Sort `R` by total sequence length in descending order.
2. For each request in the sorted order, use IQR masking to avoid engines whose
   current virtual KV load is an outlier.
3. Among the remaining engines, choose the lexicographically smallest virtual
   state:

```text
<B_i, K_i>
```

Where:

- `B_i` is decode-side request load for engine `i`.
- `K_i` is decode-side KV cache load for engine `i`.

For this P/D decode-only scope, `B_i` can use the existing running/waiting counts:

```text
B_i = running_requests_i + waiting_requests_i
```

`K_i` should be derived from KV block usage rather than raw logical tokens:

```text
K_i = allocated_kv_blocks_i
```

or, if later needed for heterogeneous cache groups:

```text
K_i = allocated_kv_bytes_i
```

## Current vLLM Fit

The current internal DP load balancer already lives on the front-end/client side.
The key hook is:

```text
vllm/v1/engine/core_client.py
  DPLBAsyncMPClient.get_core_engine_for_request()
```

Today, when `request.data_parallel_rank` is unset, this method scores each engine
using queue counts:

```text
score = waiting * 4 + running
```

After selecting an engine, it performs a local virtual update to reduce
stale-stat misrouting between coordinator updates.

This is still the right architectural layer for IQR-Lex, but the policy should
not be implemented as a pure per-request replacement for the current scoring
function. The front-end should first collect the auto-routed decode requests that
are ready in the current admission round, schedule that batch with Algorithm 3,
and then dispatch the assigned requests to their decode engines. The per-rank
Scheduler remains unchanged after the request reaches the selected decode
engine.

The current DP coordinator path is:

```text
DPEngineCoreProc._maybe_publish_request_counts()
  -> EngineCoreOutputs(scheduler_stats=...)
  -> DPCoordinator
  -> DPAsyncMPClient stats update task
  -> DPLBAsyncMPClient decode admission batch scheduler
```

Currently this path only carries running/waiting counts. This design extends that
stats payload with decode KV load fields.

## Data Model

Introduce a structured decode LB stats payload. This can either extend
`SchedulerStats` or be a nested field inside it. A nested dataclass is preferred
to avoid making the existing positional/count semantics more fragile.

```python
@dataclass
class DPDecodeLBStats:
    num_running_reqs: int
    num_waiting_reqs: int

    kv_cache_usage: float
    num_total_blocks: int
    num_free_blocks: int
    num_allocated_blocks: int

    # Optional in first version, useful for guards and observability.
    preemption_count_delta: int = 0
    timestamp: float = 0.0
```

For first version:

```text
B_i = num_running_reqs + num_waiting_reqs
K_i = num_allocated_blocks
```

`num_allocated_blocks` is computed as:

```text
num_total_blocks - num_free_blocks
```

The decode engine can derive these values from:

- `scheduler.get_request_counts()`
- `scheduler.kv_cache_manager.usage`
- `scheduler.kv_cache_manager.block_pool.get_num_free_blocks()`
- `scheduler.kv_cache_manager.block_pool.num_gpu_blocks`

For hybrid or heterogeneous KV cache managers, the stats API should later return
per-cache-group block or byte usage. That is not required for the first
decode-only implementation.

## Stats Flow

Engine-side publication should change from count-only stats to decode LB stats.

Current shape:

```text
SchedulerStats(num_running_reqs, num_waiting_reqs, step_counter, current_wave)
```

Target shape:

```text
SchedulerStats(
    num_running_reqs=...,
    num_waiting_reqs=...,
    kv_cache_usage=...,
    decode_lb_stats=DPDecodeLBStats(...),
    step_counter=...,
    current_wave=...,
)
```

Coordinator behavior should stay simple:

- Receive latest stats from each decode engine.
- Preserve wave/running-state behavior.
- Publish the latest per-engine decode LB stats to front-end clients.
- Avoid making scheduling decisions in the coordinator.

Client behavior:

- `DPAsyncMPClient` stores the latest per-engine decode LB stats.
- `DPLBAsyncMPClient` uses those stats for placement.
- If richer stats are absent, fall back to the current queue-count policy.

## Decode Admission Batch

The policy runs in `DPLBAsyncMPClient` for requests whose
`request.data_parallel_rank is None`.

The batch `R` is the set of newly pending auto-routed decode requests available
at the front-end in the current admission round. The implementation should reuse
the existing vLLM scheduling/forward cadence rather than introducing a separate
admission cadence. If only one auto-routed request is available in a round, `R`
has one element and the batch algorithm naturally degenerates to a single
placement decision.

Requests with an explicit `request.data_parallel_rank` bypass the policy and are
sent to the specified engine immediately. This preserves external-router
behavior and debugging workflows.

`total_seq_len` should be computed from front-end-visible request metadata:

```text
total_seq_len = current_decode_seq_len(request)
```

For the first version:

- Prefer an exact transferred-KV or externally-computed-token count if the decode
  request carries one.
- Otherwise use the current prompt/token length available in `EngineCoreRequest`
  as the conservative estimate.
- Do not include future output length by default.

If long-output imbalance remains visible, add the optional
`current_plus_expected` cost mode described below.

## Scheduling Policy

The policy starts from the latest per-engine decode LB stats snapshot published
through the coordinator. At the start of each admission round, the front-end
copies those stats into a virtual state table. Every request assignment mutates
only the virtual table; the engine-side truth still comes from the next stats
publication.

Pseudo-code:

```python
def schedule_decode_admission_batch(auto_routed_requests):
    states = copy_latest_decode_lb_states()
    auto_routed_requests.sort(key=estimate_total_seq_len, reverse=True)

    assignments = []
    for request in auto_routed_requests:
        candidates = [
            s for s in states
            if s.healthy
            and not s.draining
            and s.num_free_blocks >= estimate_required_blocks(request)
        ]

        if not candidates:
            target = fallback_current_policy(request, states)
            assignments.append((request, target.engine))
            virtual_update(target, request)
            continue

        safe = iqr_mask(candidates, key=lambda s: s.num_allocated_blocks)
        if not safe:
            safe = candidates

        target = min(
            safe,
            key=lambda s: (
                s.num_running_reqs + s.num_waiting_reqs,
                s.num_allocated_blocks,
                s.engine_index,
            ),
        )

        assignments.append((request, target.engine))
        virtual_update(target, request)

    dispatch_assigned_requests(assignments)
```

IQR masking:

```python
def iqr_mask(candidates):
    if len(candidates) < 4:
        return candidates

    loads = sorted(s.num_allocated_blocks for s in candidates)
    q1 = percentile(loads, 25)
    q3 = percentile(loads, 75)
    threshold = q3 + iqr_k * (q3 - q1)

    safe = [s for s in candidates if s.num_allocated_blocks <= threshold]
    return safe if len(safe) >= min_safe_ranks else candidates
```

Default knobs:

```text
iqr_k = 1.5
min_safe_ranks = 1
fallback_policy = current_queue_count_policy
```

## Virtual Update

Virtual update is required because coordinator stats are periodic and can lag
behind admissions. It is also what makes batch scheduling different from
independent per-request greedy placement: later requests in the same `R` observe
the earlier assignments through the virtual `B_i` and `K_i` values.

At the start of an admission round, copy the latest per-engine decode LB stats.
For each request assigned within sorted `R`, update the copied state:

```python
def virtual_update(state, request):
    estimated_blocks = estimate_required_blocks(request)

    state.num_waiting_reqs += client_count
    state.num_allocated_blocks += estimated_blocks
    state.num_free_blocks = max(0, state.num_free_blocks - estimated_blocks)
    state.kv_cache_usage = (
        state.num_allocated_blocks / max(1, state.num_total_blocks)
    )
```

For P/D decode admission, the request already represents a decode-side request.
The first version can estimate blocks from the request's current decode sequence
length:

```text
estimated_blocks = ceil(current_seq_len / block_size)
```

If the decode request carries enough metadata for transferred KV blocks, prefer
that exact block count.

Future-output cost is optional. The conservative first version can omit it:

```text
cost_mode = current
```

If long-output imbalance remains visible, add:

```text
cost_mode = current_plus_expected
estimated_blocks = current_blocks + alpha * expected_output_blocks
```

with `alpha` in the `0.3..0.5` range.

## Backpressure And Fallback

The first implementation should keep failure behavior conservative.

Fallback cases:

- Missing decode LB stats: use current queue-count policy.
- Empty admission batch: no-op.
- Single-request admission batch: run the same batch algorithm with `|R| = 1`.
- Fewer than 4 candidate engines: skip IQR masking and use lexicographic
  `<B_i, K_i>` selection.
- All engines masked: use all candidates.
- No engine has enough free KV blocks: fall back to current behavior first, then
  evaluate whether existing waiting/backpressure behavior needs to be tightened.
- Explicit `X-data-parallel-rank` / `request.data_parallel_rank`: bypass policy.

Health and draining fields can be added later. They are useful operationally but
not required to validate the algorithm.

## Metrics

Existing metrics already cover request latency, TPOT, decode time, preemption
count, and KV cache usage. This policy needs additional LB observability:

```text
vllm:dp_lb_decode_b_load
vllm:dp_lb_decode_kv_allocated_blocks
vllm:dp_lb_decode_kv_free_blocks
vllm:dp_lb_iqr_threshold_blocks
vllm:dp_lb_iqr_masked_engines
vllm:dp_lb_selected_engine
vllm:dp_lb_virtual_kv_blocks
vllm:dp_lb_fallback_total
```

Validation should focus on distribution and tails:

```text
std(K_i)
max(K_i) / mean(K_i)
std(B_i)
P50/P90/P99 TPOT
request decode time
preemption count
decode tokens/s
MoE all-to-all wait / bubble
```

Throughput alone is not a sufficient success metric for this policy.

## Configuration

Suggested decode LB options:

```text
--dp-lb-policy {queue, iqr_lex_decode}
--dp-lb-iqr-k 1.5
--dp-lb-min-safe-ranks 1
--dp-lb-kv-cost-mode {current,current_plus_expected}
--dp-lb-fallback-policy queue
```

The first patch can keep these internal or experimental if the CLI surface is not
desired immediately.

## Implementation Plan

1. Add a structured decode LB stats type.
2. Add a small KV stats helper on the scheduler or KV cache manager.
3. Extend `DPEngineCoreProc._maybe_publish_request_counts()` to publish running,
   waiting, and KV block stats.
4. Update `DPCoordinator` to store and publish the richer per-engine stats.
5. Update `DPAsyncMPClient` to parse and keep latest decode LB stats.
6. Add a front-end decode admission batch path in `DPLBAsyncMPClient` that
   collects the newly pending auto-routed decode requests for the current
   admission round.
7. Implement `iqr_lex_decode` batch selection:
   sort `R` by total sequence length descending, then assign each request using
   IQR masking and lexicographic `<B_i, K_i>` over the virtual state table.
8. Preserve current queue-count policy as fallback.
9. Add unit tests for batch sorting, IQR masking, lexicographic selection,
   explicit-rank bypass, and virtual update across multiple requests in one
   batch.
10. Add targeted integration coverage for coordinator/client stats propagation
   and front-end batch assignment.
11. Add metrics after the core policy is working.

## Tests

Unit tests:

- IQR mask skips masking when candidate count is below 4.
- IQR mask excludes KV outliers when enough ranks are available.
- Empty safe set falls back to all candidates.
- Admission batch is sorted by total sequence length descending before
  assignment.
- Lexicographic selection chooses smallest virtual `<B_i, K_i>`.
- Tie-breaker is deterministic by engine index.
- Explicit `data_parallel_rank` bypasses IQR-Lex.
- Virtual update changes waiting and KV block state immediately and affects
  later requests in the same admission batch.
- Missing stats falls back to current queue-count policy.

Integration tests:

- Engine publishes decode LB stats through `EngineCoreOutputs`.
- Coordinator stores and republishes per-engine decode LB stats.
- `DPLBAsyncMPClient` receives stats and assigns a multi-request admission batch
  to the expected engines.

## Open Follow-ups

- Whether `K_i` should use allocated blocks or bytes for hybrid cache groups.
- Whether decode admission should reject/wait when all candidate engines report
  insufficient free blocks.
- Whether future-output cost is needed after decode-only P/D validation.
- Whether policy metrics should be exported as per-engine gauges or front-end
  decision counters.

## Final Design

Implement Algorithm 3 as an internal decode DP load-balancing policy:

```text
new auto-routed decode requests in the current admission round
  -> DPLBAsyncMPClient forms R
  -> sort R by total sequence length descending
  -> for each request in R:
       IQR mask engines by virtual KV allocated blocks
       select argmin <virtual running + waiting, virtual allocated_kv_blocks>
       virtual update selected engine state
  -> send assigned requests to selected decode EngineCores
  -> existing per-engine Scheduler continues unchanged
```

This keeps the implementation aligned with vLLM's existing DP LB architecture,
implements Algorithm 3 as batch scheduling rather than per-request admission,
uses running/waiting for decode request load, adds the minimum KV stats needed by
the algorithm, and avoids touching prefill or token-level scheduling.
