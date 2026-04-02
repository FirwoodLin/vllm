# Offline Poisson Harness DP Queue Logging Design

Status: design only

Last updated: 2026-04-02

## 1. Background

Current benchmark launch pattern:

- node-rank=1 machine starts `offline_poisson_harness.py headless-engine`
- node-rank=0 machine starts `offline_poisson_harness.py frontend`
- frontend uses `AsyncLLM` with internal DP load balancing
- per-engine stats are already emitted through the v1 logging path when
  `disable_log_stats=False` and `VLLM_LOG_STATS_INTERVAL` is enabled

Requested new observability for each DP instance:

1. queued request count
2. total queued tokens, or total queued blocks
3. head-of-line queued request tokens, or head-of-line queued request blocks


## 2. Current State

### 2.1 What already exists

The current per-engine stats log already prints:

- `Running: %d reqs`
- `Waiting: %d reqs`

This log is emitted on the frontend side through:

- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/metrics/stats.py`
- `vllm/v1/metrics/loggers.py`
- `vllm/v1/engine/async_llm.py`

The existing waiting count is:

```text
len(waiting) + len(skipped_waiting)
```

So "queued request count" is already available today.

### 2.2 Where the real queue lives

The benchmark harness itself does not maintain a DP request queue.

The real queue for each DP instance lives inside the v1 scheduler:

- `self.waiting`
- `self.skipped_waiting`

`waiting` contains schedulable requests.

`skipped_waiting` contains waiting requests that are temporarily blocked, for
example:

- `WAITING_FOR_FSM`
- `WAITING_FOR_REMOTE_KVS`
- `WAITING_FOR_STREAMING_REQ`

### 2.3 Why the harness is not the right insertion point

`offline_poisson_harness.py` only creates the frontend `AsyncLLM` and submits
requests. It does not own the per-DP queue state. Therefore the requested logs
should not be added in the harness itself.

The correct reuse path is:

```text
Scheduler -> SchedulerStats -> AsyncLLM logger_manager -> LoggingStatLogger
```

This ensures:

- logs are per DP instance
- logs appear in the existing `frontend` log stream
- no benchmark-specific logging path needs to be invented


## 3. Recommendation

### 3.1 Log tokens first, not blocks

Recommended first version:

- keep the existing queued request count
- add queued total tokens
- add queued head-of-line tokens

Do not implement block-based logging in the first patch.

Reason:

- token semantics are easier to understand
- token accounting already exists per request through `num_tokens` and
  `num_computed_tokens`
- block accounting adds ambiguity when block size differs by backend or when
  future cache-group details matter
- token logging is sufficient for backlog visibility and easier to validate

### 3.2 Use remaining queued tokens

Recommended queue token metric:

```text
remaining_waiting_tokens = request.num_tokens - request.num_computed_tokens
```

This is better than logging raw prompt length because it works across:

- fresh waiting requests
- preempted requests
- waiting requests after remote KV receive
- resumed streaming sessions

### 3.3 Keep the queue scope aligned with existing `Waiting`

The new token metric should use the same queue scope as the existing waiting
request count:

```text
waiting + skipped_waiting
```

This keeps the request-count and token-count logs consistent.


## 4. Semantics

### 4.1 Queued request count

Already implemented:

```text
num_waiting_reqs = len(waiting) + len(skipped_waiting)
```

### 4.2 Queued total tokens

New metric:

```text
waiting_total_tokens =
    sum(request.num_tokens - request.num_computed_tokens
        for request in waiting + skipped_waiting)
```

Implementation should maintain this incrementally, not by summing the full
queue on every scheduler step.

### 4.3 Head-of-line queued tokens

New metric:

```text
waiting_head_tokens =
    remaining tokens of the next waiting request that the scheduler would try
    to schedule
```

Important detail:

This should follow the scheduler's real queue-selection behavior, not simply
`waiting[0]`.

For FCFS, the scheduler currently prefers:

```text
skipped_waiting or waiting
```

For PRIORITY, the scheduler compares queue heads using the existing priority
ordering logic.

Therefore the head-of-line metric should be defined as:

- the request returned by `_select_waiting_queue_for_scheduling()`
- then `peek_request()`
- then compute `request.num_tokens - request.num_computed_tokens`

This makes the logged "head" match the actual next scheduling candidate.


## 5. Design

### 5.1 Extend `SchedulerStats`

Add fields to `vllm/v1/metrics/stats.py`:

- `waiting_total_tokens: int = 0`
- `waiting_head_tokens: int = 0`

This keeps the new values in the same transport object already used by:

- per-engine logging
- any future custom stat logger
- potential Prometheus exposure later

### 5.2 Add scheduler-side bookkeeping

Do not compute total queued tokens by scanning the queues inside
`Scheduler.make_stats()`.

Reason:

- `make_stats()` runs in the scheduler hot path
- the benchmark can generate large queues
- an O(queue length) scan on each step adds avoidable overhead

Recommended scheduler state:

- `waiting_total_tokens_ready`
- `waiting_total_tokens_blocked`

Or equivalently:

- one total for `waiting`
- one total for `skipped_waiting`

Recommended helper methods:

- `_request_remaining_waiting_tokens(request) -> int`
- `_add_waiting_request_stats(request)`
- `_remove_waiting_request_stats(request)`
- `_update_waiting_request_stats(request, old_remaining_tokens)`

The exact helper names can differ, but the design goal is:

- all queue token bookkeeping is O(1)
- all queue transitions update the counters at the transition point

### 5.3 Keep head-of-line lookup O(1)

`waiting_head_tokens` does not need a persistent cached value.

At stats creation time:

1. call `_select_waiting_queue_for_scheduling()`
2. if it returns a queue, call `peek_request()`
3. compute remaining tokens for that request

This is O(1) and matches scheduler semantics.

### 5.4 Logger output

Extend the per-engine log line in `vllm/v1/metrics/loggers.py`.

Recommended new fields:

- `Waiting tokens: %d`
- `Waiting head tokens: %d`

Example target output:

```text
Engine 003: Avg prompt throughput: ..., Avg generation throughput: ...,
Running: 7 reqs, Waiting: 19 reqs, Waiting tokens: 286720,
Waiting head tokens: 32768, GPU KV cache usage: ...
```

### 5.5 Aggregated logger behavior

The current harness uses per-engine logging, which is the important case.

For completeness:

- aggregated logger should sum `waiting_total_tokens`
- aggregated logger should not print `waiting_head_tokens` by default

Reason:

- summed backlog across engines is meaningful
- a single aggregated "head-of-line tokens" value across engines is not


## 6. Required Transition Points

If token counters are maintained incrementally, every queue transition that
changes membership or remaining tokens must update them.

The following scheduler paths are important:

### 6.1 New request enqueue

- `add_request()`
- `_enqueue_waiting_request()`

### 6.2 Waiting to running

Inside the waiting scheduling loop:

- request popped from `waiting` or `skipped_waiting`
- request moved to `running`

### 6.3 Waiting to skipped-waiting within a step

Inside the scheduling loop:

- blocked request requeued to `step_skipped_waiting`
- final `step_skipped_waiting -> skipped_waiting`

### 6.4 Running to waiting

- `_preempt_request()`
- `reset_prefix_cache(reset_running_requests=True)`

### 6.5 Waiting removal on finish or abort

- `finish_requests()`

### 6.6 Blocked waiting state promotion

When a request in `skipped_waiting` is promoted back to `WAITING` or
`PREEMPTED`, the queue membership may remain waiting-like, but the remaining
token count may have changed.

Important cases:

- `WAITING_FOR_REMOTE_KVS -> WAITING`
- `WAITING_FOR_REMOTE_KVS -> PREEMPTED`

### 6.7 Invalid block recovery

The invalid-block recovery path can rewrite `request.num_computed_tokens`.

If the request is still in a waiting queue when that happens, token counters
must be updated using the delta.

### 6.8 Streaming session update

`_update_request_as_session()` can change prompt length and therefore remaining
waiting tokens.

If the request is in a waiting queue during this update, token counters must be
updated accordingly.


## 7. Proposed Implementation Plan

### Step 1. Extend the stats payload

Files:

- `vllm/v1/metrics/stats.py`

Work:

- add `waiting_total_tokens`
- add `waiting_head_tokens`

### Step 2. Add scheduler token bookkeeping

Files:

- `vllm/v1/core/sched/scheduler.py`

Work:

- add queue token counter state
- add helper methods for add/remove/update
- define remaining token calculation in one place

### Step 3. Hook all queue transitions

Files:

- `vllm/v1/core/sched/scheduler.py`

Work:

- update counters at every queue transition listed in Section 6
- keep the logic local to scheduler queue operations
- avoid ad hoc counter edits scattered through unrelated code

### Step 4. Populate `SchedulerStats`

Files:

- `vllm/v1/core/sched/scheduler.py`

Work:

- set `waiting_total_tokens`
- compute `waiting_head_tokens` from the actual selected queue head

### Step 5. Extend log formatting

Files:

- `vllm/v1/metrics/loggers.py`

Work:

- append the new fields to per-engine log output
- keep aggregated logger behavior meaningful

### Step 6. Add tests

Files:

- `tests/v1/core/test_scheduler.py`
- optional logger-format test under `tests/v1/engine/`

Work:

- verify queue token totals in normal waiting flow
- verify queue token totals in blocked waiting flow
- verify queue token totals across preemption
- verify logger text contains the new fields


## 8. Test Plan

### 8.1 Basic waiting queue case

Create several waiting requests with different prompt lengths and verify:

- `num_waiting_reqs`
- `waiting_total_tokens`
- `waiting_head_tokens`

### 8.2 Remote KV waiting case

Create requests that enter `WAITING_FOR_REMOTE_KVS` and later return to normal
waiting.

Verify:

- total waiting tokens do not drift
- head-of-line tokens match the promoted request when it becomes selectable

### 8.3 Preemption case

Force preemption and confirm:

- request leaves `running`
- request re-enters waiting
- waiting total tokens increase by the expected remaining amount

### 8.4 Abort or finish removal case

Abort or finish waiting requests and verify:

- total waiting tokens decrease correctly
- head-of-line metric updates correctly

### 8.5 Logging case

Verify the emitted log line contains:

- `Waiting tokens:`
- `Waiting head tokens:`


## 9. Risks and Mitigations

### Risk 1. Counter drift

Problem:

- incremental accounting can drift if a transition path is missed

Mitigation:

- centralize queue add/remove/update helpers
- cover blocked waiting and preemption in tests

### Risk 2. Wrong head-of-line semantics

Problem:

- using `waiting[0]` would not match the actual scheduler decision when
  `skipped_waiting` participates

Mitigation:

- derive the head from `_select_waiting_queue_for_scheduling()`

### Risk 3. Added hot-path overhead

Problem:

- scanning all waiting requests in `make_stats()` would add scheduler overhead

Mitigation:

- maintain total tokens incrementally
- compute only head-of-line at stats time

### Risk 4. Aggregated logger ambiguity

Problem:

- cross-engine head-of-line has no single natural meaning

Mitigation:

- print head-of-line only in per-engine logs
- keep aggregated logger limited to summed total backlog


## 10. Non-Goals for the First Patch

The first patch should not:

- change DP routing policy from `waiting * 4 + running`
- add harness-only logging outside the v1 stats path
- add block-based queue metrics
- add Prometheus metrics for the new fields

These can be follow-up changes if the token-based logs prove useful.


## 11. Recommended First Patch Scope

Recommended patch scope:

1. add `waiting_total_tokens` and `waiting_head_tokens`
2. log them in the existing per-engine stats line
3. add scheduler tests for waiting, blocked waiting, and preemption

This is the smallest change that gives the requested visibility while staying
aligned with the current v1 engine architecture.
