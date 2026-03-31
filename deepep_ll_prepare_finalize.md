# DeepEP LL `prepare` / `finalize` 梳理

本文只整理 `DeepEPLLPrepareAndFinalize` 这一路径，也就是 EP `all2all_backend=deepep_low_latency` 时，MoE modular kernel 里 `prepare` 和 `finalize` 的职责、输入输出和调用顺序。

对应实现：

- `vllm/model_executor/layers/fused_moe/deepep_ll_prepare_finalize.py`
- `vllm/model_executor/layers/fused_moe/modular_kernel.py`
- `vllm/model_executor/layers/fused_moe/all2all_utils.py`

## 1. 它在整体流程里的位置

`DeepEPLLPrepareAndFinalize` 是 `FusedMoEPrepareAndFinalizeModular` 的一个实现，用来把 DeepEP low-latency 的 dispatch/combine 接到 modular MoE 流程里。

整体链路可以概括为：

```text
hidden_states
  -> prepare / prepare_async
  -> experts kernel
  -> finalize / finalize_async
  -> output
```

在 `modular_kernel.py` 里，大致顺序是：

```text
_prepare()
  -> self.prepare_finalize.prepare(...) 或 prepare_async(...)
  -> 得到 a1q / a1q_scale / expert_tokens_meta

_fused_experts()
  -> experts kernel 在本 rank 上执行专家计算
  -> 得到 fused_out

_finalize()
  -> self.prepare_finalize.finalize(...) 或 finalize_async(...)
  -> 得到最终 output
```

## 2. LL 路径下 `prepare` 是什么

### 2.1 一句话定义

`prepare` 的职责是：

1. 校验 DeepEP LL 的输入约束
2. 必要时先把 router weight 乘到输入上
3. 调用 DeepEP low-latency dispatch 把 token 发到对应 expert
4. 把 dispatch 的结果整理成 experts kernel 需要的 batched 格式

可以把它理解为：

```text
prepare = "发出去之前的整理 + dispatch + 把收到的 expert 输入整理好"
```

### 2.2 `prepare()` 和 `prepare_async()` 的关系

`prepare()` 本身只是同步包装：

```python
hook, receiver = self.prepare_async(...)
hook()
return receiver()
```

所以真正的主体逻辑在 `prepare_async()` 和 `_receiver()`。

### 2.3 `prepare_async()` 实际做了什么

#### 第一步：检查约束

主要检查这些事情：

- `defer_input_quant` 不能为 `True`
- hidden size 必须在 LL 支持列表里
- 如果走 fp8 dispatch，hidden size 必须满足 128 对齐
- 非 `nvfp4` dispatch 不支持 per-token scales

这些限制来自 DeepEP LL 内核本身的能力边界，不是 generic modular kernel 的要求。

#### 第二步：决定量化/dispatch 模式

它会根据 `quant_config` 和环境变量，决定是否：

- 使用 fp8 dispatch
- 使用 nvfp4 dispatch
- 传入全局 scale
- 使用 ue8m0 的打包 scale 传输

这里的重点是：

- LL 路径允许 dispatch 阶段就使用 DeepEP 自己支持的格式
- 但最终 experts kernel 所需要的量化格式，仍然要在 `_receiver()` 里统一整理

#### 第三步：可选地提前应用 router weight

如果 `apply_router_weight_on_input=True`，会先执行：

```python
a1 = a1 * topk_weights.to(a1.dtype)
```

当前只支持 `topk=1`。

这一步的语义是：

- 某些模型希望把 routing weight 乘在输入上
- 那么后面的 combine 阶段就不能再重复乘一次

#### 第四步：把 global expert id 映射成 physical id

LL 路径支持物理 expert 重排，所以 dispatch 前会先做：

```text
global topk_ids -> physical topk_ids
```

对应函数是 `_map_global_to_physical_ids()`。

#### 第五步：调用真正的 DeepEP LL dispatch

核心调用是：

```python
self.buffer.low_latency_dispatch(...)
```

它返回的关键结果有：

- `expert_x`
- `expert_num_tokens`
- `handle`
- `hook`

这几个值的含义分别是：

- `expert_x`: dispatch 后的激活，已经按 expert 组织
- `expert_num_tokens`: 每个 expert 收到了多少 token
- `handle`: 这次 dispatch 对应的上下文，combine 时还要继续用
- `hook`: 用于异步接收/同步的回调

这里 `expert_x` 已经不再是原始 `(M, K)` 的 token 平铺形式，而是 LL batched experts 路径使用的 expert-major 布局。

#### 第六步：保存 handle，返回 hook + receiver

`prepare_async()` 不直接把最终 `PrepareResultType` 算完，而是返回：

- 一个 `hook`
- 一个 `receiver`

`hook` 用来等待或推进底层通信；
`receiver` 用来真正取出并整理结果。

之所以这么设计，是为了给 DBO 和 shared expert overlap 留出空间，可以把通信和计算交错执行。

### 2.4 `_receiver()` 做了什么

`_receiver()` 的职责是把 dispatch 出来的 `expert_x` 进一步整理成 experts kernel 可以直接消费的输入。

它主要做两件事：

#### 1. 调 `_do_quant()` 统一量化表示

`_do_quant()` 会根据 dispatch 阶段到底传了什么格式，做后处理，例如：

- 如果 DeepEP 在 dispatch 阶段已经做了 fp8 量化，直接接收
- 如果格式还不符合后续 experts kernel 预期，先反量化再重数量化
- 把 scales 整理成 batched experts kernel 需要的形状

所以 `_receiver()` 返回的 `expert_x` / `expert_x_scale`，才是后续 experts kernel 真正消费的输入。

#### 2. 构造 `ExpertTokensMetadata`

它会用 `expert_num_tokens` 构造：

```python
ExpertTokensMetadata(
    expert_num_tokens=expert_num_tokens,
    expert_num_tokens_cpu=None,
)
```

这个元数据告诉 batched experts kernel：每个 expert 这一批里实际有多少有效 token。

### 2.5 `prepare` 的输出是什么

LL 路径下，`prepare` 最终返回：

- `expert_x`
- `expert_x_scale`
- `expert_tokens_meta`
- `None`
- `None`

也就是：

```text
(batched expert inputs, scales, 每个 expert 的 token 数, 无额外 topk_ids, 无额外 topk_weights)
```

这里和 HT 路径有一个明显差异：

- HT 往往返回 standard/contiguous 形式的数据，并可能返回重新整理后的 `topk_ids/topk_weights`
- LL 路径返回 batched expert 输入，`topk_ids/topk_weights` 继续沿用原始那份

## 3. LL 路径下 `finalize` 是什么

### 3.1 一句话定义

`finalize` 的职责是：

1. 取回 `prepare` 阶段保存的 handle
2. 调用 DeepEP low-latency combine
3. 把各个 expert 的输出合并回原始 token 顺序
4. 在 combine 阶段完成 topk weight 应用和 reduce

可以把它理解为：

```text
finalize = "专家算完之后的 combine + 收尾"
```

### 3.2 为什么 LL 的权重应用和 reduce 在 finalize 里

LL 路径里有一个很关键的约束：

```python
assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate)
```

这说明在 DeepEP LL 这条路径里：

- topk weight application
- reduction

不是在 Python 层先单独做完再 combine，而是委托给 combine kernel 一起处理。

换句话说，LL 的 `finalize` 不只是简单“收包”，而是包含了结果融合语义。

### 3.3 `_finalize()` 实际做了什么

#### 第一步：取回 prepare 阶段保存的 handle

`prepare_async()` 在 dispatch 后把 handle 存进了 `self.handles[a2a_idx]`。

`finalize()` 这里会把它取出来传给 combine。

这个 handle 的意义可以理解成：

- 它把这次 combine 绑定到前面那次对应的 dispatch 上
- combine 需要知道 token 当时是怎么被打散/发送的

#### 第二步：决定 combine 用的权重

如果前面已经 `apply_router_weight_on_input=True`，那么这里会执行：

```python
combine_topk_weights = torch.ones_like(topk_weights)
```

原因很直接：

- 输入阶段已经乘过权重了
- combine 阶段不能再乘一次

否则就使用原始 `topk_weights`。

#### 第三步：把 global id 再映射成 physical id

和 dispatch 一样，combine 也要基于 physical expert id 工作，所以这里再次做：

```text
global topk_ids -> physical topk_ids
```

#### 第四步：调用真正的 DeepEP LL combine

核心调用是：

```python
self.buffer.low_latency_combine(
    fused_expert_output,
    combine_topk_ids,
    combine_topk_weights,
    handle,
    out=output,
)
```

这一步完成的事情是：

- 按 dispatch 时的路由关系，把各个 expert 的输出发回去
- 按 token 原顺序合并
- 应用 topk 权重
- 做 reduce
- 直接把结果写入 `output`

所以从 modular kernel 的角度看，LL 的 `finalize()` 完成之后，`output` 就已经是最终 routed experts 的输出。

## 4. 从 modular kernel 角度再看一遍

### `prepare` 之前

输入还是普通 hidden states：

```text
hidden_states: (M, K)
topk_ids / topk_weights: router 给出的路由结果
```

### `prepare` 之后

输入已经变成 batched experts 格式：

```text
expert_x: (num_experts, max_tokens, K)
expert_x_scale: 可选
expert_tokens_meta: 每个 expert 有多少有效 token
```

这正是 LL batched experts kernel 更容易直接消费的布局。

### experts kernel 之后

得到的是每个 expert 的输出 `fused_expert_output`，它还不是最终 token 顺序输出，只是“专家侧算完的结果”。

### `finalize` 之后

才重新变回最终 token 输出：

```text
output: (M, K)
```

## 5. 一个简化的时序图

```text
hidden_states, topk_ids, topk_weights
    |
    | prepare_async
    | - 校验 LL 约束
    | - 可选提前乘 topk_weights
    | - global id -> physical id
    | - low_latency_dispatch
    | - 保存 handle
    v
hook + receiver
    |
    | receiver()
    | - _do_quant
    | - 构造 expert_tokens_meta
    v
expert_x, expert_x_scale, expert_tokens_meta
    |
    | experts kernel
    v
fused_expert_output
    |
    | finalize
    | - 取回 handle
    | - 处理 combine_topk_weights
    | - global id -> physical id
    | - low_latency_combine
    v
output
```

## 6. 最短总结

如果只记一句话：

- `prepare` 是“把 token 按 expert 发出去，并整理成 batched experts 输入”
- `finalize` 是“把 expert 输出按 token 合回来，并在 combine 里完成权重应用和归约”

如果再精确一点：

- LL 的 `prepare` 重点在 `low_latency_dispatch`
- LL 的 `finalize` 重点在 `low_latency_combine`
- 两者之间靠 `handle` 串起来
- LL 路径天然是 batched experts 语义，不是 HT 那种 standard contiguous 语义
