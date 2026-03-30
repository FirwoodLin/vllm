# vLLM Profiler Iteration Control

Last updated: 2026-03-27
Repo basis: local `/vllm` checkout used in this session

## 1. Goal

This note explains how vLLM's profiling window is actually controlled when using:

- `/start_profile`
- `/stop_profile`
- `--profiler-config.ignore_frontend`
- `--profiler-config.delay_iterations`
- `--profiler-config.max_iterations`

The focus is not generic torch profiler usage. The focus is the exact mechanism
inside vLLM and how to use it to narrow profiling to a desired worker-iteration
window, especially for mixed prefill/decode workloads.

## 2. The key mental model

The most important fact is:

- profiling window control is not implemented at the HTTP layer
- it is implemented inside each worker profiler
- the unit of control is a worker `step()`
- in practice, one such step corresponds to one worker-side
  `execute_model(...)` iteration

Relevant code paths:

- `/vllm/vllm/config/profiler.py`
- `/vllm/vllm/profiler/wrapper.py`
- `/vllm/vllm/v1/worker/gpu_worker.py`
- `/vllm/vllm/v1/engine/async_llm.py`

This means:

- `/start_profile` only arms profiling
- `/stop_profile` only disarms / flushes profiling
- the actual begin/end of the narrow worker window is controlled by the worker
  profiler's internal counters

## 3. What `/start_profile` really does

`AsyncLLM.start_profile()` does two things:

- asks engine cores to start profiling
- optionally starts a frontend CPU torch profiler if frontend profiling is
  enabled

Relevant code:

- `/vllm/vllm/v1/engine/async_llm.py:879`
- `/vllm/vllm/v1/engine/async_llm.py:885`

On the worker side, `profile(is_start=True)` initializes the concrete profiler
wrapper on the first call and then calls `self.profiler.start()`.

Relevant code:

- `/vllm/vllm/v1/worker/gpu_worker.py:855`

Inside `WorkerProfiler.start()`:

- `_active` becomes `True`
- if `delay_iterations == 0`, the underlying profiler starts immediately
- if `delay_iterations > 0`, the profiler does not start yet; it waits for
  worker steps

Relevant code:

- `/vllm/vllm/profiler/wrapper.py:71`

So `/start_profile` does not mean "record the next GPU kernels immediately".
It means "arm the per-worker profiler state machine".

## 4. What one "iteration" means here

The iteration counter is advanced in `gpu_worker.annotate_profile()`:

- first `self.profiler.step()` is called
- then vLLM computes the current iteration's context/generation counts
- then `execute_model(...)` runs inside the profiling annotation scope

Relevant code:

- `/vllm/vllm/v1/worker/gpu_worker.py:729`
- `/vllm/vllm/v1/worker/gpu_worker.py:821`

This gives the practical definition:

- one profiler iteration = one worker `execute_model(...)` pass

This is not:

- one HTTP request
- one generated token globally
- one scheduler API call

For a mixed batch, one worker iteration may contain:

- only context work
- only generation work
- mixed context + generation work

vLLM already annotates each iteration with:

- number of context requests / tokens
- number of generation requests / tokens

Relevant code:

- `/vllm/vllm/v1/utils.py:439`

This is useful for calibration because a long prompt may consume multiple
context iterations before the batch settles into generation-heavy iterations.

## 5. Why `ignore_frontend=true` is strongly recommended

The config comment is explicit:

- frontend profiling does not track iterations
- when delay/limit options are used, frontend profiling can capture the whole
  wall-clock range
- this adds overhead and makes the trace less clean

Relevant code:

- `/vllm/vllm/config/profiler.py:69`
- `/vllm/vllm/config/profiler.py:125`
- `/vllm/vllm/v1/engine/async_llm.py:186`

Practical consequence:

- if you want a narrow GPU-side decode window, set
  `ignore_frontend=true`

Otherwise the worker window may be narrow while the AsyncLLM CPU trace still
covers the full interval from `/start_profile` to `/stop_profile`.

## 6. Exact semantics of `delay_iterations`

`delay_iterations` is checked in `WorkerProfiler.step()`:

- `active_iteration_count` is incremented on every worker step after
  `/start_profile`
- when `active_iteration_count == delay_iterations`, the underlying profiler is
  started

Relevant code:

- `/vllm/vllm/profiler/wrapper.py:83`

The practical 1-based interpretation is:

- `delay_iterations = 1`: start recording from worker iteration 1
- `delay_iterations = 2`: skip iteration 1, start from iteration 2
- `delay_iterations = N`: skip iterations `1..N-1`, start from iteration `N`

This is easy to misread. It is not "skip N full iterations and start at
N+1". It starts on the Nth worker iteration.

This is also how the local example in the repo reasons about it:

- `/vllm/nano-test/start_dp4_tp8_dcp8_torch_profiler.sh:22`

## 7. Exact semantics of `max_iterations`

`max_iterations` is also handled in `WorkerProfiler.step()`:

- once the profiler is running, `_profiling_for_iters` is incremented for each
  profiled worker step
- when the internal count exceeds `max_iterations`, vLLM stops the profiler

Relevant code:

- `/vllm/vllm/profiler/wrapper.py:99`

The subtle but important detail is call order:

- `self.profiler.step()` happens before `execute_model(...)`
- auto-stop also happens before `execute_model(...)`

So when you use the simple mode with:

- `wait_iterations = 0`
- `warmup_iterations = 0`

you can treat:

- `max_iterations = M`

as:

- capture exactly `M` worker `execute_model(...)` iterations

This matches the intended usage in the repo's example scripts and is the most
useful operational interpretation.

One more detail:

- auto-stop does not fully clean state
- you should still call `/stop_profile` at the end so that the profiler flushes
  traces and resets state cleanly

Relevant code:

- `/vllm/vllm/profiler/wrapper.py:128`

## 8. The second layer: torch profiler schedule

vLLM also supports:

- `wait_iterations`
- `warmup_iterations`
- `active_iterations`

If either `wait_iterations > 0` or `warmup_iterations > 0`, vLLM creates a
`torch.profiler.schedule(wait, warmup, active, repeat=1)`.

Relevant code:

- `/vllm/vllm/profiler/wrapper.py:203`

vLLM then treats only the schedule's active part as counting toward
`max_iterations`.

Relevant code:

- `/vllm/vllm/profiler/wrapper.py:232`
- `/vllm/vllm/profiler/wrapper.py:275`

This creates two layers of gating:

1. vLLM-level gating
   - `delay_iterations`
   - `max_iterations`
2. torch schedule gating
   - `wait_iterations`
   - `warmup_iterations`
   - `active_iterations`

For precise decode-window control, this usually makes reasoning harder, not
easier.

Recommendation:

- if your goal is "capture a specific middle decode window", keep
  `wait_iterations = 0` and `warmup_iterations = 0`
- use only `delay_iterations + max_iterations`

If you do enable warmup, treat it only as a small noise filter, not as the main
windowing mechanism.

## 9. Why this matters for mixed prefill/decode workloads

The scheduler does not have a hard-coded "prefill phase" and "decode phase".
It only assigns token budget each iteration.

Relevant code:

- `/vllm/vllm/v1/core/sched/scheduler.py:340`

This matters because:

- a long prompt may span multiple worker iterations
- chunked prefill can produce several context iterations
- your mixed batch may show context and generation work in the same iteration

So the right way to choose `delay_iterations` is not by assuming:

- "iteration 1 is prefill, iteration 2 onwards is decode"

Instead, do one calibration run and inspect iteration annotations.

## 10. Practical calibration method

For a workload such as:

- per DP rank: `63 x 4k input + 1 x 128k input`
- output length: `64`

the recommended process is:

1. Run one short calibration trace.
2. Inspect iteration annotations like
   `execute_context_X(T)_generation_Y(U)`.
3. For each DP rank, find the first iteration where:
   - `context_tokens == 0`
   - `generation_tokens > 0`
4. Take the maximum of those per-rank iteration indices.
5. Set `delay_iterations` to that value.
6. Set `max_iterations` to the number of decode iterations you want to keep.

This gives a practical "all ranks have entered real generation work" start
point.

Example:

- DP0 first pure-generation iteration = 9
- DP1 first pure-generation iteration = 11

Then use:

- `delay_iterations = 11`

If you want to capture 16 worker generation iterations:

- `max_iterations = 16`

## 11. Recommended server-side settings for this use case

For narrow GPU-side profiling windows, the most predictable config is:

```bash
--profiler-config.profiler torch \
--profiler-config.torch_profiler_dir /path/to/trace \
--profiler-config.ignore_frontend true \
--profiler-config.delay_iterations <N_start> \
--profiler-config.max_iterations <window_len> \
--profiler-config.wait_iterations 0 \
--profiler-config.warmup_iterations 0
```

Recommended control flow:

1. `POST /pause?mode=keep`
2. enqueue all requests
3. `POST /start_profile`
4. `POST /resume`
5. wait until the target work completes
6. `POST /stop_profile`

Reason:

- when paused, no worker iteration is advancing
- so the iteration counter only begins moving after `/resume`
- this reduces race between `/start_profile` and the first real worker step

## 12. Important caveat: iteration control is not a proof of "no dummy work"

This mechanism narrows the time window, but it does not strictly prove that all
captured kernels are real decode work.

In DP + EP / MoE style deployments, vLLM may execute a dummy batch when an
engine is still globally running but has no ready local request to execute.

Relevant code:

- `/vllm/vllm/v1/engine/core.py:1699`

So:

- iteration control is useful
- it improves reproducibility
- but it does not by itself guarantee a dummy-free window

That is a separate synchronization problem.

## 13. Bottom-line recommendations

For decode-window profiling, the most practical rules are:

- always set `ignore_frontend=true`
- use `delay_iterations` as the 1-based first worker iteration to capture
- use `max_iterations` as the number of worker `execute_model(...)` iterations
  to keep
- keep `wait_iterations=0` and `warmup_iterations=0` unless you have a very
  specific reason
- always call `/stop_profile` even if auto-stop already happened
- calibrate with iteration annotations instead of assuming one prefill step

If the workload is mixed-length and includes a very long prompt, calibration is
not optional. It is the only reliable way to place the profiling window where
you think it is.
