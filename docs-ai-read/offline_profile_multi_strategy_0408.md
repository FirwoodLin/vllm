# Offline Profile Multi-Strategy Entry Plan

Date: 2026-04-08

## Goal

Turn the current DP32-specific offline profile wrapper into a generic multi-node
offline profiling entry that can launch multiple predefined topology
strategies, including:

- `dp4dcp8` for `DP4 / TP8 / DCP8 + EP`
- `dp8dcp4`
- `dp16cp2`
- `dp32`

The first version should stay aligned with the existing
`benchmarks/manual_multinode_poisson_runner.py` strategy table instead of
adding a free-form topology builder.

This document is an implementation checklist only. It does not imply that the
changes already exist.

## Current State

The current wrapper is:

- `benchmarks/offline_dp_profile/start_4node_dp32_offline_profile.sh`

What is already generic underneath:

- `benchmarks/manual_multinode_poisson_runner.py` already defines strategy
  presets for:
  - `dp4dcp8`
  - `dp8dcp4`
  - `dp16cp2`
  - `dp32`
- the runner already forwards:
  - `--data-parallel-size`
  - `--data-parallel-size-local`
  - `--tensor-parallel-size`
  - `--decode-context-parallel-size`
  - `--enable-expert-parallel`
  - `--all2all-backend`
  - `--dcp-comm-backend`
- `vllm/benchmarks/offline_poisson_harness.py` already accepts the parallel
  args through `AsyncEngineArgs.add_cli_args(...)`

What is still DP32-specific at the wrapper layer:

- script name contains `dp32`
- default artifact root contains `profile_dp32`
- default base casecsv is `deepseek_issue01_dp32_profile.casecsv`
- the only shipped profile base case currently has `strategy=dp32`
- generated output directories do not encode topology, so different strategies
  can overwrite each other if they reuse the same `lens-json`

## Design Decision

Phase 1 should expose `--strategy`, not raw `--dp-size --tp-size --dcp-size`.

Reason:

1. The runner already treats strategy as the source of truth for topology and
   default tuning.
2. The runner already validates cluster/strategy consistency.
3. A raw topology mode would force the wrapper to duplicate strategy-level
   defaults such as:
   - `max_num_seqs`
   - `gpu_memory_utilization`
   - `all2all_backend`
   - `dcp_comm_backend`
4. The EP setting in this stack is not a separate user-facing group size
   parameter. In the current implementation, effective EP size is derived from
   the model-parallel configuration when `enable_expert_parallel=True`.

Recommended rule for v1:

- expose `--strategy`
- keep raw topology composition out of scope
- optionally add raw topology mode later if there is a real need

## Target CLI Shape

Recommended generic command:

```bash
zsh benchmarks/offline_dp_profile/start_multinode_offline_profile.sh \
  --cluster 4node_h200 \
  --strategy dp4dcp8 \
  --model deepseek_v3_1024k \
  --lens-json benchmarks/plans-unfinished/jsons/issue01.json \
  --dispatch-policy least_cache \
  --max-requests csv_rows \
  --profile-delay-iterations 32 \
  --pause-before-profile
```

Recommended compatibility behavior:

- if `--case-csv` is provided:
  - use that casecsv as the template or run source
- if `--case-csv` is not provided:
  - use a neutral profile template casecsv
- if `--strategy` is omitted:
  - default to `dp32` for backward compatibility in phase 1

## Scope

In scope:

- generic wrapper naming and CLI
- strategy-based topology selection
- case derivation that can override strategy-dependent fields
- artifact path isolation across strategies
- test coverage for the new wrapper behavior
- backward-compatible alias path for the old DP32 wrapper

Out of scope:

- arbitrary topology free-form composition
- changing runner strategy semantics
- changing harness routing semantics
- changing profiler semantics

## File-Level Implementation Checklist

### 1. Add a new generic wrapper script

Create:

- `benchmarks/offline_dp_profile/start_multinode_offline_profile.sh`

Checklist:

- copy the current DP32 wrapper as the base
- rename help text so it no longer mentions DP32 only
- add CLI flags:
  - `--cluster`
  - `--strategy`
  - `--model`
  - `--dataset`
  - `--case-csv`
  - keep existing:
    - `--lens-json`
    - `--output-len`
    - `--warmup-requests`
    - `--max-requests`
    - `--request-rate`
    - `--dispatch-policy`
    - `--case-name`
    - `--profile-delay-iterations`
    - `--pause-before-profile`
- default `ARTIFACT_ROOT` to a neutral path such as:
  - `/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench/profile_multinode`
- default `CASE_CSV` to a neutral template instead of a dp32 case
- when `--lens-json` is provided, pass all topology-related overrides into the
  case preparation step
- ensure generated input directory includes enough uniqueness to avoid
  collisions:
  - lens stem
  - strategy
  - dispatch policy
  - optionally model short name

### 2. Keep the old wrapper as a compatibility alias

Keep:

- `benchmarks/offline_dp_profile/start_4node_dp32_offline_profile.sh`

Recommended behavior:

- either keep it unchanged temporarily
- or turn it into a thin forwarding wrapper to the new generic script with:
  - `--cluster 4node_h200`
  - `--strategy dp32`

Checklist:

- do not break existing user commands
- make the old script help text explicitly say it is a compatibility entry

### 3. Introduce a neutral profile template casecsv

Create:

- `benchmarks/offline_dp_profile/profile_cases/offline_profile_template.casecsv`

Why:

- the current base casecsv hardcodes `strategy=dp32`
- the generic wrapper needs a template that can be safely overridden

Template requirements:

- one enabled row
- neutral `name`
- neutral `strategy`
- neutral or common-default values for:
  - `cluster`
  - `model`
  - `dispatch_policy`
  - `request_rate`
  - `warmup_requests`
  - `max_requests`
  - `max_model_len`
  - `data_parallel_rpc_port`
- `dataset` may be a placeholder alias if `--lens-json` or `--dataset`
  overrides it

### 4. Extend `prepare_custom_lens_case.py` to override more fields

Update:

- `benchmarks/offline_dp_profile/prepare_custom_lens_case.py`

Add CLI overrides for:

- `--cluster`
- `--strategy`
- `--model`
- `--dataset`
- `--max-num-seqs`
- `--gpu-memory-utilization`
- `--data-parallel-rpc-port`

Checklist:

- keep existing behavior for:
  - `--warmup-requests`
  - `--max-requests`
  - `--request-rate`
  - `--dispatch-policy`
  - `--case-name`
- when `--dataset` is not provided and `--lens-json` is provided:
  - continue generating the derived length CSV and point `dataset` to it
- when `--dataset` is provided without `--lens-json`:
  - allow direct dataset override without generating a length CSV
- update the derived case name logic so it includes:
  - strategy
  - lens stem or dataset stem
  - dispatch tag when non-default

### 5. Decide how strategy defaults are sourced

There are two implementation choices.

Preferred choice:

- centralize strategy profile defaults in a shared Python helper module

Candidate content:

- `strategy -> max_num_seqs`
- `strategy -> gpu_memory_utilization`
- optional artifact naming helpers

Consumers:

- `benchmarks/manual_multinode_poisson_runner.py`
- `benchmarks/offline_dp_profile/prepare_custom_lens_case.py`
- optionally the new shell wrapper through a tiny Python helper invocation

Avoid:

- duplicating strategy defaults once in shell and again in Python

If phase 1 must stay small:

- keep the runner as the source of truth
- add a small helper script that prints the defaults for one strategy
- have the wrapper query it before generating the casecsv

### 6. Review whether runner changes are actually needed

Expected answer for phase 1:

- minimal or no changes to `benchmarks/manual_multinode_poisson_runner.py`

Reason:

- it already supports the needed strategies
- it already validates `dp == nnodes * dp_local`
- it already forwards the topology flags

Possible small changes only if useful:

- add an exported helper for strategy defaults
- add a stable CLI or importable helper for listing supported strategies

Avoid in phase 1:

- changing strategy semantics
- adding raw topology parsing into the runner

### 7. Keep harness changes out unless there is a concrete need

Expected answer for phase 1:

- no topology-related changes in `vllm/benchmarks/offline_poisson_harness.py`

Reason:

- topology flags already arrive through `AsyncEngineArgs`
- profiling controls are already independent of strategy

Only change the harness if testing reveals a true topology-specific gap.

## Proposed Wrapper Behavior

### Case selection logic

Recommended order:

1. If `--case-csv` is passed:
   - use it as the base case input
2. Else:
   - use the neutral template casecsv
3. If `--lens-json` is passed:
   - derive a length CSV and a temporary casecsv
4. Apply user overrides in this order:
   - cluster
   - model
   - strategy
   - dataset
   - request_rate
   - dispatch_policy
   - warmup_requests
   - max_requests
   - max_num_seqs
   - gpu_memory_utilization
   - data_parallel_rpc_port
   - case_name

### Strategy-derived defaults

Recommended rule:

- if user passes `--strategy` and does not explicitly pass
  `--max-num-seqs` or `--gpu-memory-utilization`, fill them from strategy
  defaults
- if user explicitly passes them, user value wins

This keeps the wrapper generic while still allowing manual tuning.

### Artifact directory naming

Current collision risk:

- `generated_inputs/<lens_stem>` is too coarse

Recommended layout:

```text
generated_inputs/<lens_stem>/<strategy>/<dispatch_tag>/
```

or

```text
generated_inputs/<model_short>/<lens_stem>/<strategy>/<dispatch_tag>/
```

The benchmark artifact root should also encode strategy in the scenario path,
which is already naturally handled by the runner once `case.strategy` changes.

## Suggested Test Plan

### Unit tests for `prepare_custom_lens_case.py`

Add tests covering:

- overriding `strategy`
- overriding `cluster`
- overriding `model`
- overriding `max_num_seqs`
- overriding `gpu_memory_utilization`
- preserving `csv_rows` in `max_requests`
- name generation for multiple strategies
- output path isolation across strategies

### Unit tests for `manual_multinode_poisson_runner.py`

Only if runner helpers are changed.

Add tests covering:

- loading generated casecsv with non-dp32 strategies
- strategy default helper behavior if introduced
- compatibility with `csv_rows`

### Shell-level smoke tests

Add lightweight script tests or command rendering checks for:

- `--strategy dp32`
- `--strategy dp4dcp8`
- `--case-csv` direct mode
- `--lens-json` derived mode
- forwarding of `--pause-before-profile`
- forwarding of `--profile-delay-iterations`

### Manual validation matrix

Run at least:

1. `dp32`
2. `dp4dcp8`

For each:

- verify runner command contains expected topology flags
- verify run artifact `run_meta.json` shows expected:
  - `dp_size`
  - `dp_size_local`
  - `tp_size`
  - `dcp_size`
  - `ep_enabled`
- verify generated casecsv has expected `strategy`
- verify artifact paths do not collide with previous strategy runs

## Recommended Rollout Order

1. Add the neutral template casecsv.
2. Add the new generic wrapper script.
3. Extend `prepare_custom_lens_case.py` with the extra override fields.
4. Decide and implement the shared source for strategy profile defaults.
5. Add tests.
6. Keep the old DP32 wrapper as a compatibility shim.
7. Manually validate `dp32` and `dp4dcp8`.

## Risks

### Risk 1: duplicate defaults drift

If strategy defaults are defined both in shell and Python, they will drift.

Mitigation:

- keep one Python source of truth

### Risk 2: generated inputs overwrite each other

If generated directory naming remains lens-only, multi-strategy runs will
clobber each other.

Mitigation:

- include strategy and dispatch policy in generated input path

### Risk 3: wrapper appears generic but still uses dp32-only template values

If the template keeps dp32-tuned values and the wrapper forgets to override
them, the run may launch a different topology than the user expects.

Mitigation:

- neutral template
- explicit override pass
- command and case manifest logging

### Risk 4: users assume EP is independently configurable here

In this stack, EP is currently a strategy consequence plus
`enable_expert_parallel=True`, not a separate `--ep-size` knob.

Mitigation:

- document this clearly in the wrapper help text
- do not expose a fake `--ep-size` argument in phase 1

## Acceptance Criteria

The implementation is complete when all of the following are true:

1. A single wrapper can launch both `dp32` and `dp4dcp8` profile runs.
2. The wrapper does not require hand-editing casecsv files for strategy
   switching.
3. The generated casecsv correctly reflects the selected strategy.
4. The runner command line shows the expected topology flags.
5. Generated input files and benchmark artifacts do not overwrite each other
   across strategies.
6. The old DP32 wrapper still works as a compatibility path.

## Recommended First Milestone

The smallest useful milestone is:

- new generic wrapper
- neutral template casecsv
- `prepare_custom_lens_case.py` support for `--strategy`
- `dp32` and `dp4dcp8` manual validation

Do not expand to raw topology composition before this milestone is stable.
