# DecodeBenchConnector 对齐 NanoDeploy dummy_prefill 设计

## 1. 文档目的

本文档基于当前本地 checkout 的静态代码分析，回答一个具体问题：

```text
如果要把 vLLM 的 DecodeBenchConnector 行为改得像 NanoDeploy 的 dummy_prefill，
需要修改哪些代码路径，目标状态应该是什么？
```

这里的“像 NanoDeploy”指的是：

1. 首轮只做 KV slot / block 分配，不跑真实 model forward。
2. 首轮结束后，请求状态里已经多出 1 个 synthetic output token。
3. 下一轮第一次真实 GPU 执行时，读到的 query token 是这个 synthetic token。
4. 因而下一轮走的是 pure decode，而不是“最后 1 个 prompt token 的 prefill”。

本文档不包含代码实现，只给出面向当前代码的改造设计。

## 2. 当前两边的真实差异

### 2.1 vLLM 当前 DecodeBenchConnector

当前 `DecodeBenchConnectorScheduler.get_num_new_matched_tokens()` 返回的是：

```text
ext_tokens = prompt_len - 1
load_kv_async = False
```

对应代码：

1. `/vllm/vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py`
2. `/vllm/vllm/v1/core/sched/scheduler.py`

这会导致：

1. scheduler 认为只有 `P - 1` 个 token 是 external KV。
2. 本轮仍会调度 `1` 个 token 去真正跑 model。
3. worker 首次真实 forward 吃到的是最后一个 prompt token。
4. 所以它本质上还是 “1-token prefill/extend”，不是 NanoDeploy 风格的 pure decode。

另外，当前本地分支里 `start_fill_kv()` 已经把真实 `_fill_blocks()` 注释掉了，因此：

1. 逻辑上的 external KV token 数仍然存在。
2. 但物理上没有去写 dummy KV tensor。

### 2.2 NanoDeploy 当前 dummy_prefill

NanoDeploy 的路径分成两步：

1. 调度器先按整段 prompt 长度 `P` 分配 block table。
2. `dummy_prefill=True` 时不跑 prefill model forward，而是直接：

```text
may_append(seq, 1)
append_token(0)
```

于是 dummy prefill 结束后，请求状态变成：

```text
num_prompt_tokens = P
num_output_tokens = 1
num_tokens = P + 1
```

下一次真正 decode 时：

1. `prepare_decode_cpp()` 读取 `last_token` 作为输入。
2. 这个 `last_token` 正是 synthetic token。
3. 因而第一轮真实前向已经是 decode。

### 2.3 需要精确复制的不是“写多少 KV 值”，而是“请求状态机”

如果目标是对齐 NanoDeploy 的 benchmark 语义，关键不在于真的把 `P` 个 prompt token 填进 KV tensor，而在于：

1. 首轮不跑 forward。
2. 请求状态从 `P` 变成 `P + 1`。
3. 下一轮把 synthetic token 当成上一轮已经生成好的 token 来消费。

也就是说，vLLM 要复制的是 **request state transition**，不是 `_fill_blocks()` 本身。

## 3. 目标语义

设：

1. prompt 长度为 `P`
2. synthetic token id 为 `S`
3. 用户请求输出上限为 `M`

则目标状态机应为：

### 第 0 步：dummy-prefill 步

这一步不跑 model forward，只做：

1. 把请求从 `P` 扩成 `P + 1`
2. `output_token_ids = [S]`
3. 为 `P + 1` 个 token 分配 slots / blocks
4. 通过 async-KV 状态机把请求挂到 `WAITING_FOR_REMOTE_KVS`
5. worker 当步立刻汇报 `finished_recving`

这一步结束、scheduler promote 完成后，请求应满足：

```text
num_prompt_tokens = P
num_output_tokens = 1
num_tokens = P + 1
num_computed_tokens = P
```

### 第 1 步：第一轮真实 GPU 执行

这一步 scheduler 只应调度 1 个 token。

worker 看到的输入应等价于：

```text
input_token = output_token_ids[-1] = S
position = P
```

因此这一步已经是 pure decode。

## 4. 推荐实现路线

推荐保留 vLLM 现有的 `WAITING_FOR_REMOTE_KVS -> finished_recving -> promote back` 框架，不单独发明一套 scheduler 特判。

原因很简单：

1. 现有 scheduler 只有在 `load_kv_async=True` 时，才支持“本轮分配 block，但不跑真实 forward”。
2. 这正好匹配 NanoDeploy dummy-prefill 的第一步语义。
3. `_update_waiting_for_remote_kv()` 已经内置了 full-hit 修正：

```text
if request.num_computed_tokens == request.num_tokens:
    request.num_computed_tokens = request.num_tokens - 1
```

对本方案来说，这个修正正好把：

```text
P + 1 -> P
```

变成我们想要的状态。

因此最小侵入方案是：

1. 首轮把请求扩成 `P + 1`
2. connector 声称远端已经准备好了 `P + 1` 个 token 的 KV
3. 走 async recv 路径
4. worker 立刻回报 `finished_recving`

## 5. 必须修改的代码点

## 5.1 `DecodeBenchConnector` 需要进入 async dummy-prefill 模式

文件：

1. `/vllm/vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py`

需要新增一个 mode，例如：

```python
dummy_prefill: bool = False
dummy_output_token_id: int = 0
```

在该 mode 下：

1. `get_num_new_matched_tokens()` 不再返回 `P - 1`
2. 而是返回扩容后的 `request.num_tokens - num_new_local_computed_tokens`
3. 并且 `is_async=True`

换句话说，在 `dummy_prefill=True` 时：

```text
ext_tokens = P + 1
load_kv_async = True
```

而不是：

```text
ext_tokens = P - 1
load_kv_async = False
```

## 5.2 需要一个“只执行一次”的 request prepare hook

当前 `KVConnectorBase_V1` 只有：

1. `get_num_new_matched_tokens()`
2. `update_state_after_alloc()`
3. `build_connector_meta()`

但要精确复刻 NanoDeploy，connector 必须能在第一次 prefix-match 之前先修改请求：

1. append 一个 synthetic output token
2. 把 `num_tokens` 从 `P` 变成 `P + 1`

推荐做法是在：

1. `/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
2. `/vllm/vllm/v1/core/sched/scheduler.py`

增加一个 scheduler-side hook，例如：

```python
def prepare_request_for_external_kv(self, request: Request) -> None:
    return
```

调用时机放在 scheduler 处理 WAITING/PREEMPTED 请求、且调用
`get_num_new_matched_tokens()` 之前。

原因：

1. `get_num_new_matched_tokens()` 注释要求尽量 side-effect free。
2. synthetic token append 是一次性 mutation。
3. 把 mutation 放到显式 hook 里更容易做幂等保护。

如果不想改 base API，也可以退一步：

1. 直接在 `DecodeBenchConnector.get_num_new_matched_tokens()` 内做 mutation
2. 用 `_prepared_requests` 做幂等保护

但这不是推荐路径。

## 5.3 `Request` 必须真的多出 1 个 output token

文件：

1. `/vllm/vllm/v1/request.py`

这里不一定要改 `Request` 类本身的结构，因为它已经有：

1. `_output_token_ids`
2. `_all_token_ids`
3. `append_output_token_ids()`

推荐直接在 prepare hook 内调用：

```python
request.append_output_token_ids(dummy_output_token_id)
```

这样会同时更新：

1. `request.output_token_ids`
2. `request.all_token_ids`
3. `request.num_tokens`
4. block hash

这和 NanoDeploy 的：

```text
append_token(0)
```

在高层语义上是一致的。

## 5.4 `NewRequestData` 必须能携带已有 output token

这是当前代码里最关键的协议缺口。

当前：

1. `Request` 已经支持 output tokens
2. `GpuInputBatch` 也支持 `CachedRequestState.output_token_ids`
3. 但是 scheduler 发给 worker 的 `NewRequestData` 不携带 output tokens
4. `gpu_model_runner.py` 在新请求路径里直接写死 `output_token_ids=[]`

因此即使 scheduler 侧已经 append 了 synthetic token，worker 侧也会把它丢掉。

需要修改的文件：

1. `/vllm/vllm/v1/core/sched/output.py`
2. `/vllm/vllm/v1/worker/gpu_model_runner.py`

建议改法：

### `NewRequestData`

新增字段：

```python
output_token_ids: list[int] | None = None
```

`from_request()` 中填：

```python
output_token_ids=request.output_token_ids.copy()
```

### `gpu_model_runner.py`

构造 `CachedRequestState` 时，不再写死：

```python
output_token_ids=[]
```

而是改成：

```python
output_token_ids=list(new_req_data.output_token_ids or [])
```

这是实现 NanoDeploy parity 的必要条件，不是可选优化。

## 5.5 worker 侧要“立刻 finished_recving”

当前 `DecodeBenchConnector` 只实现了：

1. `start_load_kv()`
2. `build_connector_worker_meta()`

但没有实现：

1. `get_finished()`

因此即使 `start_load_kv()` 什么都不做，scheduler 也拿不到
`finished_recving`，请求无法从 `WAITING_FOR_REMOTE_KVS` 被 promote 回来。

需要在：

1. `/vllm/vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py`

新增 worker-side 状态，例如：

```python
_finished_recving_req_ids: set[str]
```

并实现逻辑：

### `start_load_kv()`

在 dummy-prefill mode 下：

1. 不搬运任何 KV
2. 把当前 metadata 里的请求 id 记入 `_finished_recving_req_ids`

### `get_finished()`

返回：

```python
(None, finished_recving_ids)
```

并在返回后清空 `_finished_recving_req_ids`。

这样 scheduler 就会沿现有路径：

1. 标记 `finished_recving_kv_req_ids`
2. 下一步 `_update_waiting_for_remote_kv()`
3. full-hit 修正
4. 请求重新回到可调度状态

## 5.6 `update_state_after_alloc()` 要覆盖到 `P + 1` tokens

这点不需要额外发明新机制，只要：

1. request 已经先 append 了 synthetic token
2. `ext_tokens` 变成 `P + 1`

则 `allocate_slots()` 会自然按 `P + 1` 分配 slots。

`update_state_after_alloc()` 记录的 block 数也会对应：

```text
ceil((P + 1) / block_size)
```

这和 NanoDeploy 在：

1. 先 allocate prompt blocks
2. 再 `may_append(seq, 1)`

之后的最终 block 覆盖范围是一致的。

## 6. 修改后的一步步状态变化

假设：

1. prompt 长度 `P = 1024`
2. synthetic token `S = 0`

则期望的 vLLM 路径应该是：

### Step A: 新请求进入 waiting

初始：

```text
prompt_token_ids = [t0 ... t1023]
output_token_ids = []
num_tokens = 1024
num_computed_tokens = 0
```

### Step B: connector prepare_request

执行：

```text
append_output_token_ids(0)
```

得到：

```text
prompt_token_ids = [t0 ... t1023]
output_token_ids = [0]
num_tokens = 1025
num_output_tokens = 1
num_computed_tokens = 0
```

### Step C: scheduler 第一次 schedule

connector 返回：

```text
ext_tokens = 1025
load_kv_async = True
```

scheduler：

1. 分配 1025 tokens 对应的 slots
2. 请求进入 `WAITING_FOR_REMOTE_KVS`
3. 本轮不跑 model forward

### Step D: worker dummy load

worker：

1. 不填 KV
2. 立刻回报 `finished_recving={req_id}`

### Step E: scheduler promote

`_update_waiting_for_remote_kv()` 看到：

```text
request.num_computed_tokens == request.num_tokens == 1025
```

于是修正成：

```text
request.num_computed_tokens = 1024
```

此时请求状态变成：

```text
prompt_token_ids = [t0 ... t1023]
output_token_ids = [0]
num_tokens = 1025
num_output_tokens = 1
num_computed_tokens = 1024
```

### Step F: 下一轮第一次真实 forward

scheduler 只会再调度：

```text
num_new_tokens = 1025 - 1024 = 1
```

worker 读到的位置是：

```text
index = num_computed_tokens = 1024
token = output_token_ids[0] = 0
position = 1024
```

于是第一次真实 GPU 执行就是 decode。

## 7. 输出长度与 `max_tokens` 的语义选择

这一步必须提前选清楚，因为 NanoDeploy 的当前语义不是“纯内部虚拟 token”，而是“真的往 sequence 里加了 1 个 completion token”。

这意味着：

1. synthetic token 会占用 1 个 output slot
2. 如果用户请求 `max_tokens=M`，真实模型只会再生成 `M - 1` 个 token

这和 NanoDeploy 当前实现是一致的。

推荐把这件事做成显式配置：

### 方案 A：完全对齐 NanoDeploy

配置：

```text
synthetic_counts_toward_max_tokens = True
```

特点：

1. 实现最简单
2. 语义最像 NanoDeploy
3. benchmark 里真实 decode token 数会比用户请求少 1

### 方案 B：只对齐调度语义，不消耗用户 output budget

配置：

```text
synthetic_counts_toward_max_tokens = False
```

则需要额外处理其一：

1. 内部把 `request.max_tokens` 临时加 1
2. 或者让 synthetic token 不计入 `num_output_tokens`

这会更侵入，不建议作为第一阶段实现。

## 8. TTFT 与 first-token 语义

NanoDeploy 在 dummy-prefill 分支里会立刻记录 first token metric。

vLLM 如果要完全对齐，有两个层级：

### 8.1 阶段 A：只对齐调度/forward 语义

做到：

1. 首轮不跑 prefill 1-token
2. 下一轮第一轮真实 forward 就是 decode

但不额外发 synthetic `EngineCoreOutput`。

这是推荐第一阶段。

### 8.2 阶段 B：再对齐 TTFT 事件语义

做到：

1. dummy-prefill 完成时就把 synthetic token 视作 first token
2. TTFT 直接落在 dummy-prefill 完成时刻

这需要额外考虑：

1. 是否真的生成一条 synthetic `EngineCoreOutput`
2. 是否只更新 metrics，不改用户可见输出

这部分比调度改造更侵入，建议单独做。

## 9. 明确不建议一起支持的组合

在第一版里建议直接 fail closed，不要尝试兜底：

1. prompt logprobs
2. structured output / grammar
3. pooling
4. 多模态
5. prefix caching
6. speculative decoding
7. V2 model runner
8. 需要严格 output_len 与 max_tokens 对齐的线上语义

原因是这些功能都依赖“output token 到底是不是真实 sampled token”的严格语义。

## 10. 推荐测试项

至少补下列测试：

### 10.1 connector unit test

验证：

1. `dummy_prefill=True` 时首次请求会 append 1 个 synthetic token
2. `get_num_new_matched_tokens()` 返回 `P + 1`
3. `load_kv_async=True`
4. `get_finished()` 会在当步返回 `finished_recving`

### 10.2 scheduler state transition test

验证：

1. 第一步后请求进入 `WAITING_FOR_REMOTE_KVS`
2. 收到 `finished_recving` 后 promote
3. promote 后状态为：

```text
num_tokens = P + 1
num_output_tokens = 1
num_computed_tokens = P
```

### 10.3 worker request hydration test

验证：

1. `NewRequestData.output_token_ids` 能正确传到 `CachedRequestState`
2. `GpuInputBatch` 中 token buffer 上 `prompt_len` 位置就是 synthetic token

### 10.4 first real forward input test

验证：

1. promote 后第一次 schedule 只调度 1 个 token
2. worker 读取的 input id 正是 synthetic token

## 11. 建议落地顺序

建议按下面顺序做：

1. 给 `NewRequestData` 加 `output_token_ids`，先打通 worker 侧新请求协议。
2. 给 `KVConnectorBase_V1` 增加 prepare hook，或者在 connector 内先做临时幂等 mutation。
3. 在 `DecodeBenchConnector` 增加 `dummy_prefill=True` 模式：
   - append synthetic token
   - `ext_tokens = P + 1`
   - `load_kv_async = True`
4. worker 侧实现“立即 finished_recving”。
5. 补 scheduler / connector / worker 测试。
6. 最后再决定是否对齐 NanoDeploy 的 first-token / TTFT 语义。

## 12. 一句话总结

要把 vLLM 改得像 NanoDeploy，不是把 `prompt_len - 1` 改成 `prompt_len` 就够了。

真正需要的是三件事一起成立：

1. **请求第一次进入 scheduler 时先 append 1 个 synthetic output token**
2. **connector 走 async full-hit 路径，让首轮只 allocate 不 forward**
3. **worker 新请求协议必须能保住这个 synthetic output token**

只有这样，下一轮第一次真实执行时，vLLM 才会像 NanoDeploy 一样，直接从 synthetic token 开始 pure decode。
