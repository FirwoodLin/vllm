# Manual Multinode Failure Detection Plan

This document describes a minimal-change plan to make
`benchmarks/manual_multinode_poisson_runner.py` detect sub-engine failures
earlier, stop the active benchmark case, and persist a useful failure reason.

## Problem

The current runner mainly treats a case as failed when one of the following
happens:

- the frontend wrapper exits with a non-zero code
- a remote wrapper exits early with a non-zero code
- the case exceeds `max_bench_duration_sec`
- cleanup or post-cleanup verification fails

This misses an important class of failures: a child engine can die inside the
benchmark while the outer frontend process continues running long enough to
write `summary.json` and exit with code `0`.

In that situation the runner can incorrectly record:

- `case_manifest.json.status = "ok"`
- `case_manifest.json.detail = "completed"`

even when `summary.json` already shows failed requests.

## Goals

- Stop the active case soon after a stable fatal signal appears in
  `frontend.log` or `rank*.log`.
- Reuse the existing cleanup and manifest-writing flow.
- Mark the case as failed when benchmark artifacts show request failures, even
  if the frontend exits with code `0`.
- Keep the implementation local to the runner with simple synchronous polling.

## Non-Goals

- Do not change `offline_poisson_harness.py` behavior in the first iteration.
- Do not introduce threads, async tasks, or a separate watchdog process.
- Do not add a new manifest schema unless the existing `detail` field proves
  insufficient.
- Do not build a general log parsing framework.

## Design Principles

- Prefer small helper functions over new classes or a new state machine.
- Reuse the existing polling loops in the runner.
- Prefer stable string matching over heavy parsing.
- Prefer raising `RuntimeError` with a clear message and letting the existing
  `run_case()` exception path handle cleanup and status updates.
- Preserve the first actionable failure reason instead of collecting every
  downstream symptom.

## Minimal Design

### 1. Add a small set of fatal log patterns

Add a small constant list in the runner for log text that should be treated as
an immediate benchmark failure:

- `AsyncLLM output_handler failed.`
- `EngineCore encountered a fatal error.`
- `Worker proc`
  together with
  `died unexpectedly, shutting down executor.`
- `EngineDeadError: EngineCore encountered an issue.`

The first version should intentionally stay narrow and only match messages that
already appear in the observed failures.

This avoids:

- matching generic `ERROR` lines
- creating false positives during normal startup or teardown
- introducing complicated regex-heavy logic

### 2. Reuse the existing polling loops

Do not create a new monitor thread.

Instead, extend these existing loops:

- `wait_for_headless_startup()`
- `wait_for_frontend()`

On each poll iteration, after the existing process checks, scan the relevant
runtime logs for fatal signals.

Suggested scope:

- `wait_for_headless_startup()`: scan only `rank*.log`
- `wait_for_frontend()`: scan both `frontend.log` and all `rank*.log`

If a fatal signal is detected, raise:

```text
RuntimeError("detected fatal engine failure in <log_path> during <phase>.\n<excerpt>")
```

The current `run_case()` exception handling already converts that into:

- `status = "failed"`
- `detail = str(exc)`
- cleanup via `cleanup_case_runtime()`
- manifest persistence

### 3. Keep log scanning simple

The first implementation should not maintain byte offsets.

Instead, reuse the existing `tail_file()` helper and inspect only the last
`N` lines of each log on each poll. A tail window in the low hundreds of lines
is sufficient for the initial implementation and keeps the code simple.

Why start here:

- the runner already polls once per second
- the logs are recreated per case
- the benchmark stops on the first detection, so repeated matches are not a
  real problem
- complexity stays low

If this later proves too weak for very noisy logs, offset-based scanning can be
added as a follow-up. It should not be part of the first patch.

### 4. Ignore teardown-only noise

The runner should not fail a case merely because teardown logs contain shutdown
issues after the benchmark has already finished writing artifacts.

The practical rule is:

- treat fatal pattern matches as benchmark failures only while the runner is
  still inside startup or benchmark-wait phases
- do not run fatal log scanning inside post-exit shutdown cleanup loops

This keeps the implementation simple and avoids special-case parsing of
teardown markers such as `clean_cluster_shutdown_failed`.

### 5. Validate `summary.json` before declaring success

After:

- frontend exit
- remote shutdown
- local shutdown
- `augment_benchmark_outputs()`

add one final validation step before keeping `status = "ok"`.

This step should:

1. load `benchmark/summary.json`
2. reuse `benchmark_summary_indicates_success()`
3. raise `RuntimeError` if the summary is not successful

The error message should be concise and include the key counters:

- `total_requests`
- `successful_requests`
- `failed_requests`
- `failure_ratio`

Example:

```text
benchmark summary indicates failure:
total_requests=21000 successful_requests=7492 failed_requests=13508 failure_ratio=0.643238
```

This fixes the current mismatch where the runner can write an `ok` manifest
even though the summary already records many failed requests.

### 6. Keep failure persistence in `detail`

For the first patch, do not add new manifest fields.

Persist the failure reason in the existing `detail` string:

- fatal log detected during startup or benchmark wait:
  use the raised exception message directly
- summary validation failed:
  use the summary-based message directly
- cleanup failure after an earlier fatal:
  append the cleanup text using the existing newline-join behavior

This keeps the artifact format stable and avoids touching downstream tooling.

## Proposed Helper Functions

The implementation should stay close to the current runner style. A small set
of helpers is enough:

- `iter_runtime_log_paths(runtime, include_frontend: bool) -> list[Path]`
- `find_fatal_signal_in_log(path: Path, *, lines: int) -> str | None`
- `check_runtime_logs_for_fatal(runtime, *, phase: str, include_frontend: bool) -> None`
- `validate_benchmark_summary(benchmark_dir: Path) -> None`

These helpers should be plain functions or small runner methods. No new
dataclass is needed.

## Control Flow After the Change

The intended runtime flow is:

1. launch headless wrappers
2. during `wait_for_headless_startup()`, fail immediately if a rank log shows a
   fatal engine signal
3. launch frontend
4. during `wait_for_frontend()`, fail immediately if `frontend.log` or any
   `rank*.log` shows a fatal engine signal
5. if frontend exits with non-zero code, fail as today
6. if frontend exits with zero code, continue shutdown as today
7. augment benchmark outputs
8. validate `summary.json`
9. only then keep `status = "ok"`

## Why This Is Intentionally Minimal

This plan avoids several more complicated options on purpose:

- no background watchdog process
- no SSH-side remote health RPC
- no pid-to-engine mapping
- no structured event channel between harness and runner
- no manifest schema migration
- no retry logic

Those approaches may be justified later, but they are not needed to fix the
observed failure mode.

## Validation Plan

After implementation, validate three paths.

### Successful case

- a healthy benchmark still finishes with `status = "ok"`
- no false positive from normal warnings or teardown messages

### Fatal engine case

- inject or replay a log containing one of the stable fatal patterns while the
  benchmark is still running
- verify the runner stops the case before natural completion
- verify `case_manifest.json.status == "failed"`
- verify `detail` contains the source log path and a short excerpt

### Summary failure case

- let the frontend exit normally but produce a summary with
  `failed_requests > 0`
- verify the runner records `failed`
- verify `detail` includes summary counters

## Suggested File Scope

The first patch should ideally touch only:

- `benchmarks/manual_multinode_poisson_runner.py`
- this document

That keeps review scope small and avoids coupling the fix to unrelated parts of
the benchmark stack.
