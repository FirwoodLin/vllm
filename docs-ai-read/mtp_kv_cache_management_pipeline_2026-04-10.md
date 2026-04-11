# MTP / EAGLE 场景下 KV Cache 管理与主/草稿模型关系

日期：2026-04-10
基准版本：`/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180`

## 一句话结论

MTP / EAGLE 场景下，草稿分支**不是“没有 KV cache”**。
但它**没有**独立于主模型之外的一套 KV cache 管理器实例；主模型与草稿共享同一套 `kv_cache_config / attn_groups / slot_mapping` 体系，并沿同一份 `block_table` 与内存分配策略运行。
更精确地说：每个注意力层在 `kv_caches` 映射里都有自己的 KV tensor（包括草稿层），但这些 tensor 的生命周期、分组、元数据构造和 `slot mapping` 语义来自同一条 runner 主链路。

## 1. KV cache 在 runner 上的初始化与绑定

- `GPUModelRunner.initialize_kv_cache()` 负责：
  - `initialize_attn_backend`
  - `initialize_metadata_builders`
  - `initialize_kv_cache_tensors`
  （见 [vllm/v1/worker/gpu_model_runner.py](/vllm/v1/worker/gpu_model_runner.py:6536-L6575)）
- `initialize_kv_cache_tensors()` 通过 `bind_kv_cache()` 完成绑定：
  - 把 runner 的 `self.kv_caches` 与每个 attention 层的 `forward_context[layer].kv_cache` 统一赋值
  - 参见 [vllm/v1/worker/utils.py](/vllm/v1/worker/utils.py:457-L515)
- 测试明确覆盖了草稿层的绑定顺序与存在性（`draft_model.layers.*` 会按层号与目标层交织挂载）
  - `tests/v1/worker/test_utils.py::test_bind_kv_cache_draft_model`（[tests/v1/worker/test_utils.py](/vllm/tests/v1/worker/test_utils.py:1-L110)）

## 2. 主模型与草稿模型“是否共享 KV cache”的精确含义

### 2.1 为什么不是“独立 KV cache 管理器”

- `initialize_kv_cache` 对象层只做一次；草稿模型加载前后都在同一 runner 上运行，`kv_cache_config` 不会为草稿单开新管理器。
- MTP / EAGLE 进入 `initialize_metadata_builders` 时会回调 `drafter.initialize_attn_backend(...)`，不是重建一套独立 KV 管理流程（[vllm/v1/worker/gpu_model_runner.py](/vllm/v1/worker/gpu_model_runner.py:6002-L6020)）。

### 2.2 为什么又看起来有“每层独立 KV”

- `bind_kv_cache()` 是按 `layer_name -> kv tensor` 绑定的，目标和草稿的每个 attention 层都会各有一个 `kv_cache` 引用（见 `bind_kv_cache` 与测试用例）。
- 所以在内存对象层面，“主层”和“草稿层”是不同 tensor；但它们都受同一个 runner 的 block 表、组划分和元数据策略调度。
- 这正是和 `cross_layer` / `shared_kv_cache_layers` 等机制一致的一种“按层配置共享语义”的设计：不是完全镜像一份缓存空间，而是共享管理路径与 slot 空间语义。

## 3. 数据流（prepare/execute 的链路）

1. `execute_model()` 进入时先 `_prepare_inputs()`，构建：
   - `positions`, `query_start_loc`, `seq_lens`
   - `block_table` 及 `slot_mapping`（先写入 input batch）
   - `logits_indices`, `SpecDecodeMetadata`（如有草稿 token）
   - 见 [vllm/v1/worker/gpu_model_runner.py](/vllm/v1/worker/gpu_model_runner.py:1678-L1938)
2. 同步进入 ` _get_slot_mappings()`：
   - 先按 `kv_cache_group` 构建 `slot_mappings_by_gid`
   - 再按 `layer_name` 展平为 `slot_mappings_by_layer`
   - 见 [vllm/v1/worker/gpu_model_runner.py](/vllm/v1/worker/gpu_model_runner.py:3473-L3545)
3. ` _build_attention_metadata()` 按组构建注意力元数据；
   - 若开启 spec decode，会额外取 `spec_decode_common_attn_metadata`；
   - Eagle 分支只把 drafter 对应 `kv_cache_gid` 的 metadata 传下去（`kv_cache_gid` 在草稿初始化里确定）
   - 见 [vllm/v1/worker/gpu_model_runner.py](/vllm/v1/worker/gpu_model_runner.py:1930-L2120)
4. `set_forward_context(...)` 进入 target forward：
   - `attn_metadata + slot_mapping` 决定当前批次各层 KV 写入地址策略
   - 见 [vllm/forward_context.py](/vllm/forward_context.py:1-L80) 与 [vllm/v1/worker/gpu_model_runner.py](/vllm/v1/worker/gpu_model_runner.py:3819-L3930)
5. sample 阶段后，`sample_tokens()` 中会调用 `propose_draft_token_ids()`，并把同一组 `spec_decode_common_attn_metadata` 与 `slot_mappings` 继续透传给草稿侧（见下一节）。

## 4. 计算流（草稿侧 forward 如何复用同一套 slot 语义）

- EAGLE/MTP 草稿侧对象是 `EagleProposer`，初始化时会校验所有草稿层落在同一 `kv_cache_group`：`validate_same_kv_cache_group()`（[vllm/v1/spec_decode/eagle.py](/vllm/v1/spec_decode/eagle.py:1542-L1564)）。
- 然后 `initialize_attn_backend()` 根据该 `kv_cache_gid` 选定 `kv_cache_spec` 与 `kernel_block_size`，构建草稿端的 `draft_attn_groups`（同 runner 的块结构对齐）（[vllm/v1/spec_decode/eagle.py](/vllm/v1/spec_decode/eagle.py:1565-L1618)）。
- `propose()` 内部：
  - 用 `set_inputs_first_pass()` 产出 `token_indices_to_sample` 与 `common_attn_metadata`
  - 组装 `per_layer_attn_metadata`
  - `set_forward_context(..., per_layer_attn_metadata, slot_mapping=...)` 再跑草稿模型
  - 若生成多 token 草稿，每一步通过 `eagle_step_update_slot_mapping_and_metadata()` 更新 slot mapping（[vllm/v1/spec_decode/eagle.py](/vllm/v1/spec_decode/eagle.py:360-L433)）
- 这意味着草稿的 KV 写入与主模型处于同一 slot 约定下，`block_table` 编号不会跑偏，而不是“各说各话”。

## 5. MTP 与独立 Draft Model 的差异（与问题关联）

- 配置层面：
  - `method="mtp"` 时，`SpeculativeConfig.__post_init__` 会把 `self.model` 默认回退为 `target_model_config.model`（同一 checkpoint）并继承量化/并行设置（[vllm/config/speculative.py](/vllm/config/speculative.py:360-L374)）。
  - 代码路径仍走 `use_eagle()` 分支，所以草稿 proposer 是 `EagleProposer`（不是 `DraftModelProposer`）（[vllm/v1/spec_decode/draft_model.py](/vllm/v1/spec_decode/draft_model.py:34-L67)）。
- 参数/权重层面：
  - MTP 下 `_maybe_share_embeddings` 与 `_maybe_share_lm_head` 默认是共享目标模型 embedding/head（节约显存）
  - 见 [vllm/v1/spec_decode/eagle.py](/vllm/v1/spec_decode/eagle.py:1311-L1378)

## 6. 回答你的问题

- 草稿模型**有 KV cache 路径参与，但没有独立的一套管理器**；它沿用主模型 runner 的 KV cache 管理语义。
- 换句话说：不是“主模型有，草稿没有”，而是“共享同一套 KV cache 管理机制；每个注意力层仍有自己的 KV tensor 引用”。
- 因此结论应是：**共享管理语义，不是独立 allocator；有层级 KV tensor，不是“无 KV cache”。**
