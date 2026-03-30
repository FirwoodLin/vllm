# TP / DCP Collective Analysis In DeepSeek-V3 Decode

## Experiment Scope

This analysis is for the scenario in [start.sh](/vllm/nano-test/start.sh#L1):

- `dp=1`
- `tp=8`
- `dcp=8`
- `attention-backend=FLASHMLA`
- `--enable-expert-parallel` is commented out

Relevant lines:

- [`start.sh`](/vllm/nano-test/start.sh#L28)
- [`start.sh`](/vllm/nano-test/start.sh#L30)
- [`start.sh`](/vllm/nano-test/start.sh#L31)
- [`start.sh`](/vllm/nano-test/start.sh#L38)
- [`start.sh`](/vllm/nano-test/start.sh#L39)

That means:

- DCP is enabled for decode attention.
- TP is enabled with world size 8.
- EP is not enabled in this run, even though an all-to-all backend is passed.


## High-Level Conclusion

In this experiment:

- DCP decode attention itself does not introduce `AllReduce`.
- DCP introduces:
  - `AllGather`
  - `AllToAll` for `a2a`
  - `AllGather + ReduceScatter` for `ag_rs`
- Therefore, any `AllReduce` you see in this run should be attributed to TP semantics, not DCP semantics.

However, not every `AllReduce`-looking kernel is a pure communication kernel:

- some are pure TP communication
- some are fused `TP allreduce + compute`


## The Two Kernels You Asked About

### 1. `multimem_all_reduce_kernel<...>`

Kernel:

```text
void <unnamed>::multimem_all_reduce_kernel<c10::BFloat16, (int)16>(...)
```

This is a TP `all_reduce` communication kernel.

Call path:

- [`tensor_model_parallel_all_reduce()`](/vllm/vllm/distributed/communication_op.py#L12)
- TP group `all_reduce`
- [`CudaCommunicator.all_reduce()`](/vllm/vllm/distributed/device_communicators/cuda_communicator.py#L180)
- [`SymmMemCommunicator.all_reduce()`](/vllm/vllm/distributed/device_communicators/symm_mem.py#L127)
- [`torch.ops.symm_mem.multimem_all_reduce_()`](/vllm/vllm/distributed/device_communicators/symm_mem.py#L148)

Why this is TP-specific in vLLM:

- fast allreduce backends are only enabled for TP groups
- see [`cuda_communicator.py`](/vllm/vllm/distributed/device_communicators/cuda_communicator.py#L44)

So this kernel can be treated as:

- `TP all_reduce`
- pure communication time


### 2. `allreduce_fusion_kernel_oneshot_lamport<... Pattern=1 ...>`

Kernel:

```text
void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<... Pattern)1 ...>(...)
```

This is also TP-driven, but it is not a pure communication kernel.

In the local environment, `Pattern=1` maps to:

```text
kARResidualRMSNorm
```

That means the kernel is a fused operation:

- TP `all_reduce`
- plus residual / RMSNorm logic

In vLLM, this comes from the FlashInfer allreduce fusion pass:

- [`AllReduceFusionPass`](/vllm/vllm/compilation/passes/pass_manager.py#L124)
- fusion enable condition in [`vllm.py`](/vllm/vllm/config/vllm.py#L113)
- replacement pattern in [`allreduce_rms_fusion.py`](/vllm/vllm/compilation/passes/fusion/allreduce_rms_fusion.py#L271)
- residual variant in [`allreduce_rms_fusion.py`](/vllm/vllm/compilation/passes/fusion/allreduce_rms_fusion.py#L331)

So the correct interpretation is:

- yes, it is introduced by TP semantics
- but no, its full runtime should not be counted as pure communication
- it is a fused `TP allreduce + norm` kernel


## What DCP Introduces In This Scenario

DCP decode attention communication comes from these paths:

- pre-attention query gather:
  - [`mla_attention.py`](/vllm/vllm/model_executor/layers/attention/mla_attention.py#L688)
- `ag_rs` post-processing:
  - [`common.py`](/vllm/vllm/v1/attention/ops/common.py#L199)
  - [`common.py`](/vllm/vllm/v1/attention/ops/common.py#L227)
- `a2a` post-processing:
  - [`dcp_alltoall.py`](/vllm/vllm/v1/attention/ops/dcp_alltoall.py#L342)
  - [`dcp_alltoall.py`](/vllm/vllm/v1/attention/ops/dcp_alltoall.py#L348)

So DCP contributes:

- `AllGather`
- `AllToAll` in `a2a`
- `AllGather + ReduceScatter` in `ag_rs`

It does not contribute:

- `AllReduce`


## What TP Introduces In This Scenario

### 1. TP `AllReduce` at embedding

`VocabParallelEmbedding` ends with TP allreduce:

- [`vocab_parallel_embedding.py`](/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py#L483)


### 2. TP `AllReduce` at attention `o_proj`

In DeepSeek MLA attention:

- `o_proj` is [`RowParallelLinear`](/vllm/vllm/model_executor/models/deepseek_v2.py#L910)
- `RowParallelLinear` does TP allreduce when `reduce_results=True`
- see [`linear.py`](/vllm/vllm/model_executor/layers/linear.py#L1517)

So every decode attention layer has a TP allreduce tail after `o_proj`.


### 3. TP `AllReduce` in dense MLP layers

Dense MLP uses:

- `gate_up_proj`: column parallel
- `down_proj`: row parallel

See:

- [`deepseek_v2.py`](/vllm/vllm/model_executor/models/deepseek_v2.py#L205)
- [`deepseek_v2.py`](/vllm/vllm/model_executor/models/deepseek_v2.py#L213)

And row parallel reduces at:

- [`linear.py`](/vllm/vllm/model_executor/layers/linear.py#L1517)

DeepSeek-V3 config shows:

- `num_hidden_layers = 61`
- `first_k_dense_replace = 3`
- `moe_layer_freq = 1`

See:

- [`config.json`](/mnt/nvme1n1/ml_research/models/deepseek-v3/config.json#L16)
- [`config.json`](/mnt/nvme1n1/ml_research/models/deepseek-v3/config.json#L25)
- [`config.json`](/mnt/nvme1n1/ml_research/models/deepseek-v3/config.json#L32)

So:

- first 3 layers are dense MLP layers
- those layers contribute TP `AllReduce` at MLP `down_proj`


### 4. TP `AllReduce` in MoE output combine

MoE path in DeepSeekV3:

- after MoE compute, if sequence parallel is not active and `tp_size > 1`,
  it calls `maybe_all_reduce_tensor_model_parallel(...)`
- see [`deepseek_v2.py`](/vllm/vllm/model_executor/models/deepseek_v2.py#L388)

Default MoE runner behavior:

- if output is not already reduced by kernel, it falls back to TP `all_reduce`
- see [`default_moe_runner.py`](/vllm/vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py#L331)

So the MoE layers in this run also contribute TP `AllReduce`.


### 5. TP `AllGather` at logits stage

At output logits:

- `LogitsProcessor` gathers logits across TP ranks
- see [`logits_processor.py`](/vllm/vllm/model_executor/layers/logits_processor.py#L75)
- actual all-gather call is at [`logits_processor.py`](/vllm/vllm/model_executor/layers/logits_processor.py#L83)

This is TP communication, but it is not part of the DCP attention stage.


## What TP Does Not Introduce In The MLA Attention Core

### No TP `AllGather` before/after MLA attention

`q_b_proj`, `q_proj`, and `kv_b_proj` are `ColumnParallelLinear`:

- [`deepseek_v2.py`](/vllm/vllm/model_executor/models/deepseek_v2.py#L887)
- [`deepseek_v2.py`](/vllm/vllm/model_executor/models/deepseek_v2.py#L895)
- [`deepseek_v2.py`](/vllm/vllm/model_executor/models/deepseek_v2.py#L903)

But `ColumnParallelLinear` only all-gathers when `gather_output=True`:

- [`linear.py`](/vllm/vllm/model_executor/layers/linear.py#L584)

That option is not enabled here, so the MLA-side `AllGather` is DCP, not TP.


### No TP `ReduceScatter` in this experiment

TP reduce-scatter is not part of the MLA path here.

Also, MoE sequence parallel is not active in this run:

- `use_sequence_parallel_moe` requires:
  - `enable_expert_parallel=True`
  - `tensor_parallel_size > 1`
  - `data_parallel_size > 1`
- see [`parallel.py`](/vllm/vllm/config/parallel.py#L585)

But this run has:

- `dp=1`
- EP disabled

So that branch is inactive.


### No EP `AllToAll` in this experiment

Even though `--all2all-backend deepep_low_latency` is passed:

- [`start.sh`](/vllm/nano-test/start.sh#L39)

EP is still disabled because:

- `--enable-expert-parallel` is commented out
- [`start.sh`](/vllm/nano-test/start.sh#L14)

So there is no EP all-to-all in this run.


## Why TP And DCP Can Look Similar In Nsight

DCP reuses the same physical GPUs as TP:

- DCP is built by splitting one TP group
- see [`parallel_state.py`](/vllm/vllm/distributed/parallel_state.py#L1575)

So in Nsight:

- TP collectives and DCP collectives may appear on the same 8 GPUs
- kernel names alone are not enough to distinguish them

You need:

- call site
- preceding / following kernels
- whether it is inside the MLA DCP sequence or after TP row-parallel projections


## Practical Attribution Rules

For this experiment, a good rule set is:

- `multimem_all_reduce_kernel`: count as pure TP communication
- `allreduce_fusion_kernel_oneshot_lamport<Pattern=1>`: count as TP-induced fused kernel, not pure communication
- MLA pre/post `AllGather`, `AllToAll`, `ReduceScatter`: count as DCP communication
- logits `AllGather`: count as TP communication, but outside the DCP attention stage


## Short Summary

In this `DeepSeek-V3 + tp=8 + dcp=8 + dp=1` decode benchmark:

- DCP contributes:
  - `AllGather`
  - `AllToAll` or `ReduceScatter`
- TP contributes:
  - embedding `AllReduce`
  - attention `o_proj` `AllReduce`
  - dense MLP `down_proj` `AllReduce`
  - MoE output `AllReduce`
  - logits `AllGather`

For the two kernels you asked about:

- `multimem_all_reduce_kernel`: TP pure communication
- `allreduce_fusion_kernel_oneshot_lamport<Pattern=1>`: TP-induced fused `allreduce + RMSNorm`, not pure communication
