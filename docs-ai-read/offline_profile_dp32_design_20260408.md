# Offline profiling plan for the DP32 serving profile path

Date: 2026-04-08

## Goal

Capture a stable trace that is roughly 32 worker iterations long for the
`dp32 / tp1 / dcp1` topology, while moving away from the current HTTP serving
profile path that can hang.

The target replacement path should be based on the existing offline serving
stack driven by:

- `benchmarks/run_plans_unfinished.sh`
- `benchmarks/manual_multinode_poisson_runner.py`
- `vllm/benchmarks/offline_poisson_harness.py`

This document is a design note only. It does not describe code that has already
been implemented.

## What the current DP32 profile path actually does

The hanging path is:

- `profile-dp-imbalance/start_4node_dp32_serve.sh`
- `benchmarks/send_custom_input_lens_profile_dp_rank.py`

Key facts from the current scripts:

1. Topology:
   - `dp_size=32`
   - `tp_size=1`
   - `dcp_size=1`
   - `dp_local_size=8`
   - `max_num_seqs=256`

2. The serving script enables:
   - `DecodeBenchConnector`
   - `FULL_DECODE_ONLY`
   - torch profiler with:
     - `ignore_frontend=true`
     - `delay_iterations=18`
     - `max_iterations=32`

3. The sender script does not currently use `/pause` or `/resume`.
   Those calls are commented out.

4. The sender script does:
   - flatten `benchmarks/plans-unfinished/jsons/issue01.json`
   - assign each request to a DP rank in advance
   - send `X-data-parallel-rank` headers
   - wait for a fixed `queue_settle_seconds`
   - call `/start_profile`

5. The DP-rank assignment is not generic routing. It is an explicit replay-like
   control path that tries to emulate `least_batch` or `least_cache`.

## Why this path is fragile

The fragility is not mainly about DCP. The user confirmed the DCP path is not
the one hanging.

The risky parts are specific to the DP32 serving path:

1. Profile start is externally triggered on a live HTTP service.

2. The sender uses a fixed time-based heuristic (`queue_settle_seconds=5.0`)
   instead of a deterministic workload phase boundary.

3. The sender marks requests as "started" before the actual HTTP POST completes,
   so the profile start point is only loosely synchronized with real engine
   state.

4. The sender injects `X-data-parallel-rank`, so it is not exercising the same
   routing path as the offline harness's current `internal_dplb` mode.

5. The old path assumes a fixed `output_len=64` and defines the trace window in
   terms of "middle decode iterations of one staged serving workload".
   The offline harness uses CSV workloads with per-request `output_len`, so the
   exact same interpretation does not transfer directly.

## What the offline stack already gives us

The existing offline stack already has most of the mechanics we need:

1. `manual_multinode_poisson_runner.py` already supports the `dp32` strategy.
   Existing defaults are consistent with the old DP32 path:
   - `dp_size=32`
   - `tp_size=1`
   - `dcp_size=1`
   - `dp_local_size=8`
   - `max_num_seqs=256`
   - `gpu_memory_utilization=0.87`

2. `offline_poisson_harness.py` already defaults to:
   - `DecodeBenchConnector`
   - `FULL_DECODE_ONLY`
   - dummy load format
   - step timing and graph replay timing logs

3. The offline harness has a deterministic warmup boundary:
   - `warmup_requests`
   - then measured requests

4. The offline harness currently supports only:
   - `routing_mode=internal_dplb`

5. The offline harness does not currently call:
   - `AsyncLLM.start_profile()`
   - `AsyncLLM.stop_profile()`

   This means passing `--profiler-config.*` alone is not enough to activate the
   worker profiler in the offline path.

## Important profiler semantics to keep in mind

These details matter when designing a "roughly 32 iteration" window.

1. `delay_iterations` is counted in worker steps after `start_profile()`.

2. The profiler starts on the step where
   `active_iteration_count == delay_iterations`.

   Practical consequence:
   - to skip the first `N` worker steps after `start_profile()`
   - use `delay_iterations = N + 1`

3. `max_iterations` is effectively off by one in the current wrapper logic.
   Tests show that:
   - `max_iterations=2` records 3 profiling steps before auto-stop

   Practical consequence:
   - to capture about 32 active worker steps
   - use `max_iterations=31`

4. `ignore_frontend=true` should stay enabled.
   Otherwise the AsyncLLM frontend trace will span the whole run and add CPU
   overhead that is not useful for the target DP32 worker trace.

## Recommended design

### Phase 1: get a stable offline DP32 profile first

This is the recommended first milestone.

Objective:

- reproduce the topology and decode-heavy runtime conditions
- get a stable worker trace that is about 32 iterations long
- do not try to preserve explicit DP-rank replay yet

Recommended design:

1. Keep the control plane based on `manual_multinode_poisson_runner.py`.

2. Do not reuse a sweep CSV for profiling.
   Use a dedicated one-row profile case instead.

3. Add a profile trigger inside `offline_poisson_harness.py` frontend logic:
   - run warmup first
   - call `await async_llm.start_profile(profile_prefix=...)`
   - then start submitting measured requests
   - call `await async_llm.stop_profile()` in a `finally` block

4. Set profiler config on both frontend and headless engine processes:
   - `--profiler-config.profiler torch`
   - `--profiler-config.torch_profiler_dir <case_dir>/torch_profiler`
   - `--profiler-config.ignore_frontend true`
   - `--profiler-config.delay_iterations 17`
   - `--profiler-config.max_iterations 31`
   - `--profiler-config.wait_iterations 0`
   - `--profiler-config.warmup_iterations 0`

5. Use the warmup boundary as the profile synchronization point.
   This is much more deterministic than:
   - "all HTTP requests have been created"
   - plus "sleep 5 seconds"
   - plus "start profile now"

6. Keep the runtime topology aligned with the old DP32 path:
   - strategy: `dp32`
   - `max_num_seqs=256`
   - `gpu_memory_utilization=0.87`
   - same model family as the old run if apples-to-apples comparison matters

### Why `delay_iterations=17` and `max_iterations=31`

This pair is chosen for a practical reason:

1. `delay_iterations=17`
   - skips roughly the first 16 worker steps after the warmup boundary
   - lets the measured phase settle before the trace starts

2. `max_iterations=31`
   - compensates for the current profiler wrapper's off-by-one behavior
   - produces a trace that is roughly 32 active worker iterations long

If the queue is still too sparse at the start of the trace, bump:

- `delay_iterations` from `17` to `25` or `33`

without changing the basic design.

## Recommended workload shape for Phase 1

The offline profile run should be a short, single-case workload, not a sweep.

Recommended case shape:

1. Strategy:
   - `dp32`

2. Warmup:
   - `warmup_requests=32`

3. Measured request cap:
   - `max_requests=256` or `512`

   This is enough to maintain a queue for profiling, but much shorter than a
   full sweep run.

4. Dispatch policy:
   - start with `waiting_x4_plus_running`

   Reason:
   - this matches the current offline harness path
   - it avoids mixing in the separate question of explicit rank replay

5. Dataset:
   - if the immediate goal is a stable DP32 trace, use an existing CSV dataset
     already supported by the offline harness
   - if the goal is comparison with the old `issue01.json` run, that should be
     treated as a second step

## Recommended separation of concerns

The old DP32 path mixes two separate goals:

1. getting a stable profile window
2. replaying a very specific rank allocation pattern

These should be separated.

### Phase 1 goal

Get a stable 32-step worker trace in the offline harness using
`internal_dplb`.

This is the minimum viable replacement for the hanging HTTP serve profile path.

### Phase 2 goal

Only after Phase 1 is stable, decide whether exact replay of the old sender
semantics is still necessary.

That would require additional work because the offline harness currently rejects
`routing_mode=explicit_rank_replay`.

If exact parity is required later, the clean extension is:

1. add an offline input format that includes:
   - `request_id`
   - `prompt_len`
   - `output_len`
   - `target_dp_rank`

2. implement `explicit_rank_replay` in `offline_poisson_harness.py`

3. submit each request through `engine.add_request(..., data_parallel_rank=...)`
   using the recorded `target_dp_rank`

This should be treated as a separate follow-up, not bundled into the first
stabilization step.

## What should not be copied from the old DP32 path

The following parts should not be copied into the new offline design:

1. Fixed `queue_settle_seconds` based profile start.

2. External `/start_profile` against a live HTTP service.

3. Tightly coupling "profile correctness" to `X-data-parallel-rank` replay.

4. Assuming every workload is equivalent to:
   - one staged request group
   - fixed `output_len=64`
   - a centered decode window inside that exact staged group

## Minimal implementation surface for the future code change

The smallest useful implementation surface is:

1. `vllm/benchmarks/offline_poisson_harness.py`
   - add a profile trigger after warmup and before measured submission
   - stop profiling in a `finally` block
   - make the trigger optional via CLI flags

2. `benchmarks/manual_multinode_poisson_runner.py`
   - add a small way to inject profile-related args for one-off runs
   - avoid adding many permanent sweep CSV columns unless profile runs become a
     routine workflow

A dedicated profile wrapper script is acceptable if that is simpler than
teaching the generic sweep runner about many profiler knobs.

## Proposed run shape after implementation

The future run should look like a single-case offline launch, not a serving
session plus a custom HTTP sender.

Conceptually:

```bash
ARTIFACT_ROOT=/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench/profile_dp32 \
python3 benchmarks/manual_multinode_poisson_runner.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --case-csv benchmarks/plans-unfinished/profile_cases/deepseek_issue01_dp32_profile.csv
```

And the underlying frontend/headless harness args should include something
equivalent to:

```text
--profiler-config.profiler torch
--profiler-config.torch_profiler_dir <case_dir>/torch_profiler
--profiler-config.ignore_frontend true
--profiler-config.delay_iterations 17
--profiler-config.max_iterations 31
--profiler-config.wait_iterations 0
--profiler-config.warmup_iterations 0
```

with profile activation tied to:

- `after warmup`
- `before measured submissions`

## Recommendation summary

Recommended path:

1. Replace the hanging DP32 HTTP profile workflow with an offline harness based
   profile workflow.

2. Use the offline harness warmup boundary, not a time-based external trigger,
   as the profile start anchor.

3. First target a stable DP32 worker trace under `internal_dplb`.

4. Treat explicit DP-rank replay as a separate Phase 2 feature only if exact
   parity with the old sender is still needed.

5. For a first implementation, target:
   - `dp32`
   - `warmup_requests=32`
   - `max_requests=256` or `512`
   - `delay_iterations=17`
   - `max_iterations=31`
   - `ignore_frontend=true`

