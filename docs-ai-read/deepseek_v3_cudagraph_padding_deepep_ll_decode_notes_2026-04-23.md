# vLLM 中 cudagraph “padding 到最近可复用 graph”与 DeepEP low latency decode 机制笔记

Last updated: 2026-04-23
Repo basis: local `/vllm` checkout used in this session
Environment basis: local installed `deep_ep` package at `/usr/local/lib/python3.12/dist-packages/deep_ep`

## 1. 范围与一句话结论

本文聚焦本地 `vLLM` 源码，不展开讲 SGLang。你问的场景是：

- `DeepSeek V3`
- `DP + EP`
- decode 阶段
- EP 通信后端使用 `deepep_low_latency`

一句话结论：

vLLM 里“padding 到最近可复用的 cudagraph”不是一个单独操作，而是一个分层协议：

1. 先把真实 `num_tokens` 映射到“最近可复用的 captured graph size”
2. 再把 attention 相关 metadata 补到这个静态 shape
3. DP 下再把各 rank 协调到同一个 token 数
4. MoE/EP 侧使用 DeepEP low-latency 自己的固定通信窗口

所以 attention 的 padding 和 EP 通信的 padding 不是同一种 padding。

## 2. 最重要的心智模型

如果只记一件事，我建议记这个：

`padding to reusable graph` 解决的是“让整条前向的输入 shape 落到已经 capture 过的 shape 集合上”。

而 `deepep_low_latency` 解决的是“让 expert all-to-all dispatch/combine 在固定窗口内完成，并把真实有效 token 数作为 metadata 单独携带”。

因此：

- cudagraph padding 的核心对象是 `num_tokens / num_reqs / attention metadata`
- DeepEP low-latency padding 的核心对象是 `expert dispatch buffer window`

两者会同时出现，但职责不同。

## 3. vLLM 到底怎样选“最近可复用 graph”

### 3.1 capture size 列表从哪里来

`vllm/config/vllm.py::_set_cudagraph_sizes()` 定义了默认 capture size 的生成逻辑。

默认情况下：

- 小 batch 会优先捕获 `1, 2, 4`
- 然后是 `8` 到 `255` 的 8 倍数
- 再往上是 `256` 到 `max_cudagraph_capture_size` 的 16 倍数

源码注释直接写了运行时语义：

- 如果 batch size 小于等于某个 capture size，就使用“最近的 padded CUDA graph”
- 如果 batch size 大于最大 capture size，就不用 cudagraph

也就是说，图复用的第一层本质就是一个离散 shape bucket。

### 3.2 真实 batch size 如何映射到 bucket

`vllm/v1/cudagraph_dispatcher.py::_compute_bs_to_padded_graph_size()` 会预先构造一个查表：

- 如果 `bs` 本身就是 capture size，就映射到自己
- 如果 `bs` 落在两个 capture size 中间，就映射到更大的那个

例如 capture sizes 是：

- `[1, 2, 4, 8, 16]`

那么：

- `1 -> 1`
- `2 -> 2`
- `3 -> 4`
- `5, 6, 7 -> 8`
- `9 ... 15 -> 16`

这就是“padding 到最近可复用 graph”的最核心定义。

### 3.3 FULL 和 PIECEWISE 的差异

`vllm/v1/cudagraph_dispatcher.py::_create_padded_batch_descriptor()` 有两个关键分支：

- `FULL` graph + `uniform_decode` 时，既要固定 `num_tokens`，也要固定 `num_reqs`
- 否则更偏向按 token 数匹配，request 维度约束更弱

直观上：

- `FULL` 更像“整段前向都吃静态 shape”
- `PIECEWISE` 更像“只有部分编译/图包装段要求静态 shape”

在你关心的 decode 场景里，真正最关键的是 `FULL + uniform decode` 这条路径。

## 4. decode 一次执行时，padding 是怎样层层发生的

下面按 vLLM 的执行顺序串起来。

### 4.1 先按 sequence parallelism 做第一轮 token 对齐

`vllm/v1/worker/gpu_model_runner.py::_determine_batch_execution_and_padding()` 里，真实 `num_tokens` 会先经过：

- `_pad_for_sequence_parallelism(num_tokens)`

这一步不是 cudagraph 特有逻辑，而是为了满足 SP/TP 相关静态约束。

### 4.2 然后交给 cudagraph dispatcher 选 graph bucket

同一个函数里会调用：

- `self.cudagraph_dispatcher.dispatch(...)`

得到：

- `cudagraph_mode`
- `batch_descriptor`

其中 `batch_descriptor.num_tokens` 就是最终图复用要使用的 padded token 数。

### 4.3 DP 模式下，再把所有 rank 协调到同一个 token 数

如果开启 `data_parallel_size > 1`，同一函数会进一步调用：

- `vllm/v1/worker/dp_utils.py::coordinate_batch_across_dp()`

这一步很重要，因为 cudagraph 不是只要“本 rank 能复用”就够了。在 DP + MoE 场景里，各 DP rank 需要 lockstep。

`coordinate_batch_across_dp()` 会做三件事：

1. 同步所有 DP rank 的 cudagraph mode
2. 必要时把所有 rank pad 到同一个 token 数
3. 决定是否进入 ubatch / chunked 执行

其中 cudagraph mode 的同步语义是取 `min`：

- 只要有 rank 不能用 `FULL`，全体就不能当作 `FULL`
- 只要有 rank 退化到 `NONE`，全体就要退化

所以在 DP 下，最终可复用的 graph shape 不是单 rank 自己决定的，而是所有 rank 共同决定的。

更具体地说，这里确实需要集合通信。只要：

- `data_parallel_size > 1`

`coordinate_batch_across_dp()` 就不会只看本地状态，而是进入：

- `vllm/v1/worker/dp_utils.py::_synchronize_dp_ranks()`

### 4.3.1 用的是什么集合通信

当前实现不是 `all_gather`，也不是 `broadcast`，而是一次：

- `torch.distributed.all_reduce(...)`

对应代码在：

- `vllm/v1/worker/dp_utils.py::_run_ar()`

它先通过：

- `get_dp_group().device_group`

拿到 DP 的 device process group，默认在 DP group 对应设备上做同步。

如果：

- `parallel_config.disable_nccl_for_dp_synchronization=True`

则会退回到：

- `get_dp_group().cpu_group`

也就是用 CPU all-reduce 做同步。`vllm/config/vllm.py` 里还专门有一段逻辑：在 async scheduling 打开时，这个开关默认会被置为 `True`，避免 DP 同步路径引入不合适的 GPU 同步副作用。

### 4.3.2 all-reduce 的张量长什么样

`_run_ar()` 会构造一个很小的 `int32` 张量：

- shape = `[4, dp_size]`

4 行分别表示：

1. `orig_num_tokens_per_ubatch`
2. `padded_num_tokens_per_ubatch`
3. `should_ubatch` 标志位
4. `cudagraph_mode`

然后每个 rank 只写自己那一列：

- `tensor[0][dp_rank] = num_tokens_unpadded`
- `tensor[1][dp_rank] = num_tokens_padded`
- `tensor[2][dp_rank] = 1 if should_ubatch else 0`
- `tensor[3][dp_rank] = cudagraph_mode`

最后做一次：

- `dist.all_reduce(tensor, group=group)`

### 4.3.3 为什么这里用 all-reduce 就能“收集”所有 rank 的状态

这是个很巧的实现。

因为每个 rank 只往自己的那一列写值，其他列保持 0，所以在 sum all-reduce 之后：

- 第 `j` 列的值就等于 rank `j` 原本写进去的那组状态

换句话说，这个 `all_reduce(sum)` 在这里起到了近似 `all_gather` 的效果，只不过 payload 很小，实现上也更直接。

做完这一步后，每个 rank 都会拿到一份完整的 DP 视图：

- 每个 rank 的真实 token 数
- 每个 rank 在本地非 DP padding 后的 token 数
- 每个 rank 想不想 ubatch
- 每个 rank 的本地 cudagraph mode

### 4.3.4 收集完以后如何做决策

`_synchronize_dp_ranks()` 随后会基于这个 `4 x dp_size` 张量做三步后处理。

第一步，同步 cudagraph mode：

- `_post_process_cudagraph_mode()` 直接取第 4 行的 `min`

所以：

- 只要有 rank 是 `NONE(0)`，全体就按 `NONE`
- 否则如果有人只能 `PIECEWISE(1)`，全体就不能按 `FULL(2)`

第二步，决定 ubatch 是否真的启用：

- `_post_process_ubatch()` 要求所有 rank 的 `should_ubatch` 都是 1
- 然后还要检查最后一个 ubatch 会不会为空

这里用到的量是：

- `orig_min_num_tokens = min(all ranks 的真实 token 数)`
- `padded_max_num_tokens = max(all ranks 的 padded token 数)`

如果会出现“有的 rank 第二个 ubatch 实际为空”，就放弃 ubatching。

第三步，决定是否做 DP padding：

- `should_dp_pad = (synced_cudagraph_mode != 0) or should_ubatch`

也就是说，只要：

- 本步最终还在用 cudagraph

或者：

- 本步决定做 ubatching / DBO

那么所有 DP rank 就都要 pad 到同一个 token 数。

### 4.3.5 pad 到哪个数

`_post_process_dp_padding()` 的逻辑很直接：

- 如果 `should_dp_pad=False`，返回每个 rank 自己的 `num_tokens_padded`
- 如果 `should_dp_pad=True`，取所有 rank 的 `max(num_tokens_padded)`，然后给每个 rank 都返回这个最大值

所以 DP 同步后的目标不是“大家都回到各自真实 token 数”，而是：

- 大家都提升到跨 rank 的同一个最大 padded token 数

### 4.3.6 调用方怎样使用这个结果

回到：

- `vllm/v1/worker/gpu_model_runner.py::_determine_batch_execution_and_padding()`

它在拿到 `num_tokens_across_dp` 后，会读取：

- `num_tokens_padded = num_tokens_across_dp[dp_rank]`

然后再调用一次：

- `self.cudagraph_dispatcher.dispatch(num_tokens_padded, valid_modes={synced_cudagraph_mode})`

也就是说，DP 集合通信的结果不会只停留在“决定 pad 不 pad”这个层面，而是会反过来重新生成本步最终使用的：

- `batch_descriptor`
- `cudagraph_mode`

所以从执行语义上说，DP 集合通信是 cudagraph 选图过程的一部分，而不是选图后的附属修补。

### 4.3.7 这一层为什么必要

如果没有这次 DP 集合通信，在 DP + MoE decode 场景里会出现两个问题：

1. 不同 rank 可能选到不同的 cudagraph mode，例如有人能 `FULL`，有人只能 `PIECEWISE` 或 `NONE`
2. 不同 rank 可能在同一步里处理不同 token 数，导致后面的 DBO / MoE chunking / all2all 无法 lockstep

所以 `coordinate_batch_across_dp()` 本质上是在用一次极小的控制面集合通信，去换取后续整条执行链路的静态 shape 一致性。

## 5. attention 侧到底 pad 了什么

这是最容易问细的问题。答案是：attention 真正被 pad 的主要是 metadata，不是“把无效 token 当成真 token 参与注意力”。

### 5.1 `query_start_loc` 怎样 pad

`vllm/v1/worker/gpu_model_runner.py` 在准备输入时会写：

- `query_start_loc[0] = 0`
- `query_start_loc[1:num_reqs+1] = cumsum(real query lens)`

然后把剩余位置全部填成最后一个真实累计值：

- `query_start_loc[num_reqs+1:] = cu_num_tokens[-1]`

源码注释写得很明确：

- 要保证 `query_start_loc` 是 non-decreasing
- 因为像 FlashAttention 这类 kernel 要求这样

这意味着 padded request 的 query length 会变成：

- `0`

因为相邻两个 `query_start_loc` 相等。

### 5.2 `seq_lens` 怎样 pad

同一个准备阶段还会写：

- 前 `num_reqs` 个位置是真实 `num_computed_tokens + num_scheduled_tokens`
- 后面的 padded request 全部填 `0`

所以 padded request 在 attention 看来是：

- `query_len = 0`
- `seq_len = 0`

### 5.3 `block_table` 怎样 pad

`vllm/v1/worker/gpu_model_runner.py::_build_attention_metadata()` 里，`_get_block_table()` 会把超出真实 request 的那些行全部填成：

- `-1`

源码注释说明这是 full cuda graph mode 下 `reshape_and_cache` 等路径需要的无效标记。

### 5.4 `slot_mapping` 怎样 pad

`vllm/v1/worker/gpu_model_runner.py::_get_slot_mappings()` 会把：

- `slot_mapping[num_tokens_unpadded:num_tokens_padded] = -1`

也就是说，多出来的 padded token 在 KV cache 写入时不会映射到任何真实 slot。

### 5.5 为什么这些 padded token 不会污染 KV cache

多个 attention backend 都有同样的注释，例如：

- `vllm/v1/attention/backends/flash_attn.py::do_kv_cache_update()`
- `vllm/v1/attention/backends/flashinfer.py::do_kv_cache_update()`
- `vllm/v1/attention/backends/tree_attn.py::do_kv_cache_update()`

核心意思都是：

- `key/value` tensor 可能已经 pad 了
- 但 `slot_mapping` 不会给这些 padding 分配真实 slot
- cache update op 按 `slot_mapping` 的有效部分决定写入范围

所以 padded K/V 不会真的写坏 cache。

### 5.6 FULL decode graph 下，padded request 为什么还能被当成 decode

关键在：

- `vllm/v1/attention/backends/utils.py::split_decodes_and_prefills()`

当 `require_uniform=True` 时，它专门支持一种情况：

- 所有 query length 都等于真实 decode query length
- 或者等于 `0`

也就是说，在 full-CG uniform decode 里：

- 真正的 decode request 维持统一 query_len
- 补出来的假 request 允许 query_len 为 `0`

这正是“用 padded request 去凑满 captured batch size”的关键机制。

## 6. `CommonAttentionMetadata` 里有一个容易误导的点

`vllm/v1/attention/backend.py::CommonAttentionMetadata` 里有字段：

- `num_actual_tokens`

但源码自己已经注明这个名字有误导性，因为 full-CG 路径里它其实可能已经是 padded token 数。

`gpu_model_runner.py::_build_attention_metadata()` 在 cudagraph full path 中会传入：

- `num_actual_tokens=num_tokens_padded`

`gdn_attn.py` 里也有显式注释：

- `m.num_actual_tokens is already padded by the model runner for CUDAGraph`

所以读这部分代码时，不要被字段名迷惑。语义上它更像：

- “当前 attention backend 需要看到的 token 维度大小”

而不一定是纯真实 token 数。

## 7. DeepSeek V3 在这里的特殊性：它走 MLA

本地 vLLM 里：

- `DeepseekV3ForCausalLM` 复用了 `vllm/model_executor/models/deepseek_v2.py`
- 当 `model_config.use_mla` 时，会走 `DeepseekV2MLAAttention`

也就是说，DeepSeek V3 decode 讨论 attention padding 时，真正要看的是 MLA backend。

### 7.1 MLA backend 对 cudagraph 的要求更偏向 uniform decode

多个 MLA backend 都声明了：

- `_cudagraph_support = AttentionCGSupport.UNIFORM_BATCH`

这说明它们天然更适合：

- decode-only
- uniform query length
- full cudagraph

这和 DeepSeek V3 decode 的典型场景是匹配的。

### 7.2 MLA metadata builder 怎样接受 padded request

`vllm/v1/attention/backends/mla/indexer.py` 构建 metadata 时也会走：

- `split_decodes_and_prefills(..., require_uniform=...)`

因此 padded request 的 `query_len=0` 仍然可以留在 decode 分组里，只要真实 request 的 query_len 统一。

### 7.3 MLA backend 还会对持久 buffer 做额外清理

例如：

- `vllm/v1/attention/backends/mla/flashattn_mla.py`

在 full cudagraph 下会把持久的 `scheduler_metadata` 剩余部分清零，避免旧内容污染输出。

再例如：

- `vllm/v1/attention/backends/mla/flashinfer_mla.py`

单 token decode 时会复用零初始化输出 buffer；
多 token decode 时则在 kernel 后显式处理 padding output。

所以在 MLA 路径里，padding 不只是“metadata 能过 shape check”，而是连持久工作区也要保证尾部是 inert 的。

## 8. DeepEP low latency 的 padding 和 attention padding 完全不是一回事

现在进入你最关心的第二部分：EP 通信。

### 8.1 DeepEP low latency 的 buffer shape 是固定窗口

`vllm/distributed/device_communicators/all2all.py::DeepEPLLAll2AllManager` 会用：

- `deep_ep.Buffer.get_low_latency_rdma_size_hint(...)`

来计算 RDMA buffer 大小。传进去的关键量包括：

- `num_max_dispatch_tokens_per_rank`
- `hidden`
- `num_ranks`
- `num_experts`

然后用这些参数创建 `deep_ep.Buffer(low_latency_mode=True, ...)`。

这说明 low-latency 模式从 API 层就要求：

- 所有 rank 共享一个固定的“每 rank 最多 dispatch 多少 token”的窗口

### 8.2 vLLM 传给 DeepEP low latency 的 `max_tokens_per_rank` 从哪里来

`vllm/model_executor/layers/fused_moe/deepep_ll_prepare_finalize.py` 里，
真正调用 low-latency dispatch 的是：

- `self.buffer.low_latency_dispatch(..., self.max_tokens_per_rank, ...)`

这里的 `self.max_tokens_per_rank` 是 vLLM MoE config 里给定的静态上界。

而 `vllm/envs.py` 对 `VLLM_MOE_DP_CHUNK_SIZE` 的注释非常关键：

- 在 DP + EP + batched all-to-all 场景下
- 所有 DP rank 都按 `VLLM_MOE_DP_CHUNK_SIZE` 这个 token 量子处理

默认值是：

- `256`

所以 decode 阶段常见的情况是：

- 先把跨 DP rank 的 MoE 执行切成按 rank 对齐的 token chunk
- 每个 chunk 内再满足 DeepEP low-latency 的固定 dispatch window

### 8.3 DeepEP low latency dispatch 返回的 tensor 本来就是 padded window

这部分最值得直接记住 `deep_ep.Buffer` 的接口定义。

`/usr/local/lib/python3.12/dist-packages/deep_ep/buffer.py::low_latency_dispatch()` 的文档写得很明确：

- 输入 `x` 形状是 `[num_tokens, hidden]`
- 输入 `topk_idx` 形状是 `[num_tokens, num_topk]`
- 输出 `recv_x` 形状是
  `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]`
- 不是所有位置都有效
- 每个 expert 真实收到多少 token 由 `recv_count` 给出

这就是 low-latency EP padding 的本质：

- 每个本地 expert 都有一个固定长度窗口
- 窗口长度是 `num_ranks * num_max_dispatch_tokens_per_rank`
- 真正有效的 token 个数单独放在 `recv_count`

所以它不是：

- “把 routed token 真的补成假的 routed token”

而是：

- “通信 buffer 的物理 shape 固定，但逻辑有效长度通过 metadata 区分”

### 8.4 combine 阶段也是同一个 contract

`deep_ep.Buffer::low_latency_combine()` 的输入要求同样明确：

- `x` 形状是 `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]`
- `topk_idx/topk_weights` 仍然是 `[num_combined_tokens, num_topk]`
- 输出是 `[num_combined_tokens, hidden]`

这有两个关键含义：

1. combine 的 expert output tensor 继续使用固定窗口
2. 原始 token 维度并没有被“图 padding”成更长的 `topk` 张量

也就是说，low-latency DeepEP 里真正被固定下来的，是专家分发窗口，不是 router 结果的逻辑 token 数。

### 8.5 vLLM finalize 阶段怎样把通信窗口和真实 token 对回去

`deepep_ll_prepare_finalize.py::_finalize()` 里会调用：

- `self.buffer.low_latency_combine(fused_expert_output, combine_topk_ids, combine_topk_weights, handle, out=output, ...)`

其中：

- `fused_expert_output` 是固定窗口上的 expert 输出
- `combine_topk_ids / combine_topk_weights` 还是原始 token 视角的路由表
- 输出 `output` 是真实 token 数对应的 `[num_combined_tokens, hidden]`

所以从语义上看：

- dispatch/combine 内部吃的是“固定专家窗口”
- MoE 层对外暴露的仍然是“真实 token 数”

## 9. DeepSeek V3 + `deepep_low_latency` 下 hidden 维会不会也 pad

vLLM 这条链路里确实有 hidden size roundup 逻辑，但对 DeepSeek V3 通常不是关键问题。

`vllm/model_executor/layers/fused_moe/deepep_ll_prepare_finalize.py` 里定义了 low-latency 支持的 hidden size 列表：

- `[2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]`

`maybe_roundup_layer_hidden_size()` 会把 hidden size 向上 round 到最近的支持值。

但 DeepSeek V3 常见 hidden size 是：

- `7168`

它本来就在支持列表里，所以通常不会发生 hidden dimension padding。

换句话说，在 DeepSeek V3 这个例子里，更重要的是：

- token 维的 graph padding
- request 维的 metadata padding
- expert window 的固定通信 padding

而不是 hidden 维 padding。

## 10. DP + EP 下，为什么 MoE 还能和 cudagraph 共存

关键在于 vLLM 把“跨 DP rank 的步调一致”做成了单独 contract。

### 10.1 DPMetadata 负责把各 rank 的 token chunk 对齐

`vllm/forward_context.py::set_forward_context()` 在 DP + MoE 时会构造：

- `DPMetadata`

其中：

- `DPMetadata.chunked_sizes(...)`
- `DPMetadata.sp_local_sizes(...)`

负责告诉后续 MoE 执行，在当前 chunk 上每个 rank 该处理多少 token。

### 10.2 默认 MoE runner 会按 chunk lockstep 跑

`vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` 会读取：

- `ctx.dp_metadata.max_tokens_across_dp_cpu`
- `self.moe_config.max_num_tokens`

然后按 chunk 迭代，并在每个 chunk 上进入：

- `ctx.dp_metadata.chunked_sizes(...)`

这保证了即使各 rank 真实 token 数不一样，执行也能保持锁步。

### 10.3 为什么 low-latency DeepEP 不会出现 `tensor.numel() == 0` 那个坑

`vllm/model_executor/layers/fused_moe/modular_kernel.py` 有一段关键注释：

- 对 DeepEP high-throughput 这类 cudagraph 不兼容 all2all，可能会出现某个 EP rank 根本没收到 token，于是 `M_full == 0`
- 但 DeepEP low-latency 这类 cudagraph 兼容 all2all 是 always batched 的，不会走 `tensor.numel() == 0` 这个分支

这和前面说的“固定专家窗口”是同一个设计方向：

- 即使某个 expert/rank 实际有效 token 很少
- 物理 buffer 和执行路径依旧保持 batched / static-shape

## 11. 把所有 padding 合在一起看

可以把这条链路总结成下面这张表。

| 层次 | 被 pad 的对象 | pad 后形态 | 无效部分如何标识 |
| --- | --- | --- | --- |
| cudagraph 选图 | `num_tokens` | 映射到最近 capture size | bucket 映射 |
| FULL decode graph | `num_reqs` | 补到 captured batch size | padded request 的 `query_len=0` |
| attention metadata | `query_start_loc` | 尾部重复最后一个前缀和 | 相邻位置相等表示 `query_len=0` |
| attention metadata | `seq_lens` | 尾部填 `0` | `seq_len=0` |
| attention metadata | `block_table` | 多余 request 行仍保留 shape | 行内容填 `-1` |
| KV cache update | `slot_mapping` | token 维补到 padded token 数 | 多余 token 填 `-1` |
| DP 对齐 | 各 rank token 数 | pad 到跨 DP 一致 | all-reduce 后统一 |
| DeepEP LL dispatch | expert recv buffer | `[num_local_experts, num_ranks * max_tokens_per_rank, hidden]` | `recv_count` |
| DeepEP LL combine | expert output buffer | 同上 | 仍按 `recv_count` / handle 对回真实 token |

## 12. 一个具体例子

假设 capture sizes 是：

- `[1, 2, 4, 8, 16, 24, 32, ...]`

某次 decode：

- 真实有 `13` 个 token
- 都是 single-token decode
- `num_reqs = 13`

那么在 full cudagraph uniform decode 路径里，大致会发生：

1. `13 -> 16`，选择 `16` 这个 graph
2. `num_reqs` 也补到 `16`
3. `query_start_loc` 前 14 个值对应真实请求，最后几个值重复最后一个前缀和
4. padded request 的 `query_len=0`
5. `seq_lens` 的尾部填 `0`
6. `block_table` 的尾部 request 行填 `-1`
7. `slot_mapping` 的尾部 token 填 `-1`
8. attention backend 把这些 padded request 当成 inert decode padding

如果这时同时开了 DP，并且要跑 MoE：

1. DP 各 rank 先同步 cudagraph mode 和 token 数
2. MoE runner 再按 `VLLM_MOE_DP_CHUNK_SIZE` 分块锁步
3. 在某个 chunk 内，DeepEP LL dispatch 把 token 发到固定大小的 expert window
4. 每个 expert 实际有效多少 token，由 `recv_count` 单独告诉 kernel

所以这里至少有两层“padding”同时存在：

- 一层是为了复用 cudagraph 的静态 batch/metadata padding
- 一层是为了 low-latency EP 通信的固定 expert window padding

## 13. 回答你最关心的几个具体问题

### 13.1 attention 怎么 padding

对 decode full-CG 来说，核心是：

- `query_start_loc` 尾部重复最后前缀和
- `seq_lens` 尾部填 `0`
- `block_table` 多余 request 行填 `-1`
- `slot_mapping` 多余 token 填 `-1`

因此 padded request / token 是 inert 的，不会真正参与有效 KV 写入。

### 13.2 EP 通信怎么 padding

对 `deepep_low_latency` 来说，核心不是把真实 routed token 造假补齐，而是：

- dispatch/combine buffer 天生就是固定 shape
- 真实有效 token 数通过 `recv_count` 和 handle 携带

也就是：

- 物理张量是 padded window
- 逻辑 token 数仍然是 metadata 驱动

### 13.3 DeepSeek V3 下 hidden 会不会补

通常不会，因为 `7168` 本身就是 low-latency 支持的 hidden size。

### 13.4 这是不是意味着所有 padding 都必须进 attention kernel

不是。

更准确地说：

- attention kernel 看到的是静态 shape 的 metadata / tensor view
- 但 padded 区域通过 `0`、`-1`、`recv_count` 等约定变成 inert 区域

所以系统追求的是：

- 形状静态
- 语义动态

## 14. 代码路径索引

如果你想继续顺着源码往下读，我建议按这个顺序看：

- `vllm/config/vllm.py::_set_cudagraph_sizes`
- `vllm/v1/cudagraph_dispatcher.py::_compute_bs_to_padded_graph_size`
- `vllm/v1/cudagraph_dispatcher.py::_create_padded_batch_descriptor`
- `vllm/v1/worker/gpu_model_runner.py::_determine_batch_execution_and_padding`
- `vllm/v1/worker/dp_utils.py::coordinate_batch_across_dp`
- `vllm/v1/worker/gpu_model_runner.py::_get_slot_mappings`
- `vllm/v1/worker/gpu_model_runner.py::_build_attention_metadata`
- `vllm/v1/attention/backends/utils.py::split_decodes_and_prefills`
- `vllm/v1/attention/backend.py::CommonAttentionMetadata`
- `vllm/v1/attention/backends/mla/indexer.py`
- `vllm/v1/attention/backends/mla/flashattn_mla.py`
- `vllm/v1/attention/backends/mla/flashinfer_mla.py`
- `vllm/distributed/device_communicators/all2all.py::DeepEPLLAll2AllManager`
- `vllm/model_executor/layers/fused_moe/deepep_ll_prepare_finalize.py`
- `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py`
- `/usr/local/lib/python3.12/dist-packages/deep_ep/buffer.py`

## 15. 最终结论

在 `DeepSeek V3 + DP + EP + deepep_low_latency + decode` 这个组合里，vLLM 的设计可以概括成一句话：

它不是把“真实计算”简单粗暴地补零到一个大 batch，而是把系统拆成两套静态 shape contract：

- 一套服务于 cudagraph 复用，负责 token/request/attention metadata 的静态化
- 一套服务于 DeepEP low-latency 通信，负责 expert dispatch/combine window 的静态化

真实有效 token 的边界则始终通过：

- `query_start_loc`
- `seq_lens`
- `block_table=-1`
- `slot_mapping=-1`
- `recv_count`

这些 metadata 被精确表达出来。

这就是为什么它既能复用 graph，又不会把 padding 真的当成业务 token 去算。
