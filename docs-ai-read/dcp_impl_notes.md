# vLLM DCP Implementation Notes

Last updated: 2026-03-26
Repo basis: local `/vllm` checkout used in this session

## 1. Goal

This note summarizes how vLLM implements DCP (Decode Context Parallel), with
focus on:

- how DCP groups are formed relative to TP
- how KV cache is sharded and physically placed
- whether KV cache is split at token level or block level
- which code paths build the block table / slot mapping / local seq lens
- how decode and prefill attention consume the sharded KV cache

This is meant to be a fast local reference for future Q&A, not a polished user
doc.

## 2. High-level mental model

For decode, DCP does not add extra GPUs. It reuses the GPUs already inside TP.

- TP shards KV cache on the head dimension `H`
- DCP further shards KV cache on the sequence dimension `T`
- In current implementation, DCP is formed by splitting a TP group
- Therefore `dcp_size <= tp_size`, and `tp_size` must be divisible by
  `dcp_size`

Relevant code:

- `/vllm/vllm/distributed/parallel_state.py:1572`
- `/vllm/vllm/config/parallel.py:297`
- `/vllm/docs/serving/context_parallel_deployment.md:19`

Key source facts:

- `parallel_state.py` says DCP "reuses the GPUs of TP group" and "split one TP
  group into tp_size//dcp_size DCP groups".
- `context_parallel_deployment.md` says DCP reduces KV duplication by sharding
  KV cache along `T`.

## 3. The most important answer: token-level vs block-level

It is configurable. vLLM does not hardcode one single policy.

The controlling knob is:

- `cp_kv_cache_interleave_size`
- legacy alias: `dcp_kv_cache_interleave_size`

Definition from source:

- `interleave_size = 1`: token-level alignment
- `interleave_size = block_size`: block-level alignment

Relevant code:

- `/vllm/vllm/config/parallel.py:317`
- `/vllm/vllm/config/vllm.py:1639`

The exact source comment in `parallel.py` is the clearest spec:

- token-level: token `i` goes to rank `i % total_cp_world_size`
- block-level: tokens first fill rank 0's local portion of a block, then rank 1,
  etc., before moving to the next logical block

So:

- default config (`cp_kv_cache_interleave_size = 1`) means token-level
  interleaved sharding
- setting `cp_kv_cache_interleave_size = block_size` gives block-level sharding

## 4. How storage is modeled

### 4.1 Virtual block vs physical local block

Worker-side block table code uses a crucial idea:

- one physical local KV block on a rank stores only that rank's local shard
- but block table indexing is done with a larger "virtual block" spanning all CP
  ranks

GPU path:

- `/vllm/vllm/v1/worker/gpu/block_table.py:40`

CPU/Numpy path:

- `/vllm/vllm/v1/worker/block_table.py:145`

The key formulas:

- `virtual_block_size = block_size * total_cp_world_size`
- block-table index is computed using `positions // virtual_block_size`

This means:

- for DCP world size `N`, a local physical block of size `block_size`
  corresponds to a global logical chunk of `block_size * N` tokens

### 4.2 Concrete 8TP + 8DCP picture

When `tp=8, dcp=8`, DCP effectively spans the whole 8-GPU TP group.

Assume:

- `block_size = 16`
- `dcp_size = 8`
- `cp_kv_cache_interleave_size = 1`

Then:

- virtual block size = `16 * 8 = 128`
- each request's block table advances one entry per 128 global tokens
- within each 128-token logical span, each rank stores 16 local tokens

With `interleave_size = 1`, the placement is:

- rank0 stores global tokens `0, 8, 16, ..., 120`
- rank1 stores global tokens `1, 9, 17, ..., 121`
- ...
- rank7 stores global tokens `7, 15, 23, ..., 127`

Each rank ends up with 16 local tokens, i.e. one local physical block.

If instead `cp_kv_cache_interleave_size = block_size = 16`, then the same
128-token logical span is placed block-wise:

- rank0 stores `0..15`
- rank1 stores `16..31`
- ...
- rank7 stores `112..127`

So the implementation is:

- block-table-managed at logical block granularity
- token placement inside the logical block is controlled by interleave size

## 5. Where local seq lens come from

Attention backends need per-rank local KV length, because each DCP rank only
holds part of the context.

Core helper:

- `/vllm/vllm/v1/attention/backends/utils.py:787`

Helper name:

- `get_dcp_local_seq_lens(seq_lens, dcp_size, dcp_rank, cp_kv_cache_interleave_size)`

Formula:

- compute full rounds of `dcp_size * interleave`
- distribute the remainder according to rank offset

There is also a CUDA-graph-safe Triton path:

- `/vllm/vllm/v1/worker/gpu/cp_utils.py:8`

This uses the same logic:

- `rounds = seq_lens // (dcp_size * cp_interleave)`
- `remainder = seq_lens % (dcp_size * cp_interleave)`
- local length = `rounds * cp_interleave + clipped remainder`

Where this gets attached to model runner metadata:

- `/vllm/vllm/v1/worker/gpu_model_runner.py:1978`

## 6. How block table and slot mapping are built

### 6.1 Block table shape

GPU worker block-table initialization:

- `/vllm/vllm/v1/worker/gpu/block_table.py:13`

Important line:

- `max_num_blocks = cdiv(max_model_len, block_size * cp_size)`

This is the strongest signal that DCP compresses the block-table axis by the CP
world size. It is not storing one block-table entry per ordinary `block_size`
global tokens anymore.

### 6.2 Slot mapping

Slot mapping decides whether a scheduled token should be written into this rank's
local KV cache, and if so, at which local slot.

GPU Triton path:

- `/vllm/vllm/v1/worker/gpu/block_table.py:212`

CPU/Numpy reference path:

- `/vllm/vllm/v1/worker/block_table.py:133`

GPU logic:

- `block_indices = positions // (block_size * CP_SIZE)`
- `block_offsets = positions % (block_size * CP_SIZE)`
- `is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank`
- `rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)`
- `remainder = block_offsets % CP_INTERLEAVE`
- `local_offsets = rounds * CP_INTERLEAVE + remainder`
- local slot id = `block_number * block_size + local_offsets`
- if not local, slot id is set to `PAD_ID`

Interpretation:

- first choose the global logical block by `virtual_block_size`
- then, inside that logical block, choose the owner rank according to
  interleave
- then compress the offset down to the local block's local coordinate space

### 6.3 Cache write behavior

Cache write kernel ignores negative slot mappings:

- `/vllm/csrc/cache_kernels.cu:142`

Specifically:

- `slot_idx = slot_mapping[token_idx]`
- `if (slot_idx < 0) return`

Therefore, during a decode step:

- every rank sees the scheduled tokens
- only the owning DCP rank actually writes each token's KV to local storage

## 7. Decode attention consumption path

### 7.1 FlashAttention / FlashInfer GQA path

These backends do not rebuild block table for DCP. They mainly:

- keep the DCP-aware block table
- compute local seq lens
- let the backend attend only over the local shard
- all-gather query or combine outputs/LSE across DCP as needed

Relevant places:

- `/vllm/vllm/v1/attention/backends/flash_attn.py:445`
- `/vllm/vllm/v1/attention/backends/flashinfer.py:923`

Observed behavior:

- `get_dcp_local_seq_lens(...)` is used to derive per-rank local KV lengths
- comments explicitly state per-rank maximum KV length becomes
  `ceil(L / (N * I)) * I`, where:
  - `L` = max sequence length
  - `N` = DCP world size
  - `I` = interleave size

This matches the storage model above.

### 7.2 DCP communication backend

Config:

- `/vllm/vllm/config/parallel.py:309`
- `/vllm/vllm/engine/arg_utils.py:828`

Backends:

- `ag_rs`: all-gather + reduce-scatter
- `a2a`: all-to-all based output/LSE exchange

These affect attention-stage communication, not the KV placement rule itself.

## 8. MLA path and why workspace expands

MLA has additional complexity because local KV cache is incomplete under DCP and
some paths explicitly gather and reorganize KV cache.

Relevant initialization:

- `/vllm/vllm/model_executor/layers/attention/mla_attention.py:1488`

Important comments:

- local KV cache is incomplete under DCP
- extra workspace is needed for KV all-gather across the DCP group

### 8.1 Chunked prefill context path

Important code:

- `/vllm/vllm/model_executor/layers/attention/mla_attention.py:2548`

Flow:

1. `ops.cp_gather_cache(...)` gathers this rank's local KV shard for the chunk
2. the chunk-local gathered KV is all-gathered across DCP ranks
3. `reorg_kvcache(...)` restores ordinary TP-style token order expected by the
   attention kernel

Reorg helper:

- `/vllm/vllm/model_executor/layers/attention/mla_attention.py:1981`

This helper is useful for understanding the storage order. The example comment
shows that all-gathered KV arrives grouped by rank-local chunks and then gets
reassembled into normal token order.

## 9. Why block table is not "rewritten for each rank"

One subtlety:

- the block table is DCP-aware at construction time
- it already indexes virtual blocks, not ordinary full-sequence blocks

So the main DCP-specific per-rank customization is not "slice block table rows
for this rank". Instead it is:

- compute local seq lens
- compute per-token local slot mappings
- during attention, use the same DCP-aware block table plus local lengths

For some other features such as chunked local attention, there are explicit local
block-table transforms, but that is not the core DCP KV placement mechanism.

## 10. Practical answer for 8TP + 8DCP long sequence

For one long sequence in `8TP + 8DCP`:

1. TP already shards by KV heads.
2. Inside each TP shard, DCP shards the sequence dimension `T`.
3. The per-request block table indexes virtual blocks of size
   `block_size * 8`.
4. Within each virtual block, token ownership across the 8 DCP ranks is decided
   by `cp_kv_cache_interleave_size`.
5. Only the owning rank writes each token's KV into its local physical block.

So the precise answer is:

- default behavior: token-level interleaved sharding
- not inherently token-only: it can become block-level if
  `cp_kv_cache_interleave_size = block_size`

Short form:

- logical management: block-based
- within-block placement: interleave-based
- default interleave: token-level

## 11. Good files to reopen first next time

If future questions are about DCP KV placement, reopen these first:

- `/vllm/vllm/config/parallel.py`
- `/vllm/vllm/distributed/parallel_state.py`
- `/vllm/vllm/v1/worker/gpu/block_table.py`
- `/vllm/vllm/v1/worker/block_table.py`
- `/vllm/vllm/v1/attention/backends/utils.py`
- `/vllm/vllm/v1/worker/gpu_model_runner.py`
- `/vllm/csrc/cache_kernels.cu`

If the question is specifically about MLA + DCP:

- `/vllm/vllm/model_executor/layers/attention/mla_attention.py`
- `/vllm/vllm/v1/attention/ops/dcp_alltoall.py`
- `/vllm/nano-test/tp_dcp_collectives_analysis.md`
- `/vllm/nano-test/dcp_nsys_analysis_sop.md`

## 12. Current limitations / caveats noticed in source

Source-level limitations observed during this reading:

- some KV cache manager paths assert `dcp_world_size == 1` for unsupported
  features such as sliding window / chunked local attention / mamba / hybrid
  attention:
  - `/vllm/vllm/v1/core/single_type_kv_cache_manager.py:501`
  - `/vllm/vllm/v1/core/single_type_kv_cache_manager.py:673`
  - `/vllm/vllm/v1/core/single_type_kv_cache_manager.py:794`
  - `/vllm/vllm/v1/core/kv_cache_coordinator.py:406`
- MLA path comments say DCP currently does not support some scaled/fp8 KV-cache
  combinations:
  - `/vllm/vllm/model_executor/layers/attention/mla_attention.py:684`
  - `/vllm/vllm/model_executor/layers/attention/mla_attention.py:2556`

These are not central to the storage rule, but matter when diagnosing why a
specific model/backend combination does or does not work with DCP.

## 13. Minimal glossary

- TP: tensor parallel
- DCP: decode context parallel
- PCP: prefill context parallel
- CP: generic context parallel term used in some shared code
- local physical block: one rank's actual KV-cache block of size `block_size`
- virtual block: global logical block spanning `block_size * cp_world_size`
  tokens
- interleave size: number of consecutive global tokens assigned to one rank
  before ownership rotates to the next rank

