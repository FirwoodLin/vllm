# DecodeBenchConnector V1 Dummy Prefill 实施方案

## 1. 目标

本文档给出一套面向 **vLLM V1 runner** 的实施方案，使 `DecodeBenchConnector` 在 benchmark 场景下做到：

1. 不再对 KV cache 做逐层、逐 block 的显存 fill。
2. 首次进入 scheduler 时，直接为 `prompt + 1 个 synthetic token` 分配完整 KV blocks。
3. 首轮不跑 model forward，只做 block allocation + “远端 KV 已就绪” 的状态推进。
4. 下一轮真正进入 **纯 decode**，而不是现在的“还剩 1 个 prompt token 的 prefill”。
5. 可选地复刻 `NanoDeploy dummy_prefill` 的 TTFT 语义：把 synthetic first token 当作首 token 输出事件。

这里的“纯 decode”指的是：下一次真正执行 GPU model 时，请求已经满足：

```text
num_prompt_tokens = P
num_output_tokens = 1   # synthetic token
num_computed_tokens = P
```

因此下一轮 scheduler 只会再调度 `1` 个 token，worker 也会把它当作 decode token 来处理。

## 2. 背景与现状

### 2.1 当前 `nofill` 方案为什么还不够

你现在已经把 `DecodeBenchConnectorWorker.start_fill_kv()` 里的 `_fill_blocks()` 跳过了，所以：

1. `kv_fill_ms` 已经接近 0。
2. 但是 TTFT 仍然主要耗在首轮真正的 GPU step。

根因是：

1. 当前 `DecodeBenchConnector` 只把 `prompt_len - 1` 个 token 视为 external KV。
2. scheduler / worker 因而仍然把请求视为“还差最后 1 个 prompt token 没算”。
3. 所以第一轮真正执行的仍然是“1-token prefill”，不是 pure decode。

### 2.2 为什么只分配完整 prompt 的 block 仍然不够

假设 prompt 长度是 `P`。

如果只做：

1. 为 `P` 个 prompt token 分配完整 block。
2. 然后把 `num_computed_tokens` 推到 `P`。

那么 scheduler 的现有语义会在 full-hit 时做一次修正：

```text
if request.num_computed_tokens == request.num_tokens:
    request.num_computed_tokens = request.num_tokens - 1
```

这意味着请求最终还是会回到：

```text
num_tokens = P
num_computed_tokens = P - 1
```

下一轮仍然是“最后 1 个 prompt token”的 prefill，不会变成 decode。

### 2.3 结论

要让下一轮真正变成 decode，必须让请求在 KV 就绪后处于：

```text
num_tokens = P + 1
num_computed_tokens = P
```

也就是说，除了完整 prompt KV 之外，还必须额外引入 **1 个 synthetic output token**。

这正是 `NanoDeploy dummy_prefill` 路径的本质。

## 3. 范围与非目标

### 3.1 适用范围

该方案只针对下列场景：

1. `DecodeBenchConnector`
2. `VLLM_USE_V2_MODEL_RUNNER=0`
3. benchmark / profiling 场景
4. 不关心生成语义正确性
5. 允许 dirty KV
6. `load_format=dummy`
7. `ignore_eos=True` 或者至少不依赖真实语义

### 3.2 明确不支持或暂不考虑

第一版建议直接排除这些组合：

1. prompt logprobs
2. structured outputs / grammar
3. pooling models
4. 多模态请求
5. prefix caching
6. V2 model runner
7. 需要严格保持“真实生成 token 数 = max_tokens”的线上语义

## 4. 关键代码路径与约束

### 4.1 当前 V1 runner 选型

`GPUWorker` 在 `use_v2_model_runner=False` 时，会选择：

1. `vllm/v1/worker/gpu_model_runner.py`
2. 而不是 `vllm/v1/worker/gpu/model_runner.py`

所以实现时应以 **V1 runner** 为准，不要按 V2 runner 的数据结构设计。

### 4.2 V1 runner 的请求状态结构

V1 runner 的关键点在于：

1. `CachedRequestState` 把 `prompt_token_ids` 和 `output_token_ids` 分开存。
2. `num_tokens = num_prompt_tokens + len(output_token_ids)`。
3. `InputBatch.add_request()` 已经支持“新请求带已有 output token”。

这说明：

1. V1 runner 本身是可以承载“新请求带 1 个 synthetic output token”的。
2. 现在缺的是 scheduler 到 worker 的传递协议。

### 4.3 现有 connector contract 的一个现实问题

`KVConnectorBase_V1` 的注释里说明：

1. `get_num_new_matched_tokens()` 可能被多次调用。
2. 理论上应该 side-effect free。

但本方案如果要在 scheduler 还没分配 block 之前就把请求扩成 `prompt + 1 synthetic token`，就必须在 connector/scheduler 边界做一次**请求对象的幂等变更**。

因此这里有两个实现选择：

1. **推荐的干净方案**：新增一个 scheduler-side hook，在 `get_num_new_matched_tokens()` 之前、且只调用一次，用于 prepare request。
2. **更快的本地分支方案**：在 `get_num_new_matched_tokens()` 内做幂等 mutation，并用额外状态确保不会重复 append synthetic token。

推荐采用第 1 种。

## 5. 推荐总体设计

## 5.1 核心语义

对每个首次命中的 decode-bench 请求：

1. 先给 request append 一个 synthetic output token。
2. 把请求长度从 `P` 变成 `P + 1`。
3. connector 声称“外部已算好 `P + 1` 个 token 的 KV”。
4. 走 async load 路径，但 worker 侧实际上不做任何数据搬运。
5. worker 在同一步立刻回报 `finished_recving`。
6. scheduler 在 `_update_waiting_for_remote_kv()` 中处理 full-hit 修正后，请求进入：

```text
num_tokens = P + 1
num_computed_tokens = P
```

7. 下一轮 scheduler 只会为该请求调度 `1` 个 token。
8. V1 runner 会把 synthetic token 当作“上一轮已经生成过的 token”，于是这一轮变成真正的 decode。

## 5.2 为什么必须走 async load

现有 scheduler 只有在 `load_kv_async=True` 时，才允许：

1. 分配 block
2. 但本轮不执行 model forward

如果走 sync 路径，最终还是需要在本轮调度正 token 数，无法做到“只 allocate block，不 forward”。

因此本方案必须把 `DecodeBenchConnector` 改成一个**立即完成的 async KV load connector**。

## 5.3 两阶段实施建议

建议分两阶段做，降低排错成本。

### 阶段 A：先打通“真正 pure decode”

目标：

1. 不跑首轮 prefill 1-token forward。
2. 下一轮真正变成 decode。
3. TTFT 先按“首个真实 decode token”来计算。

这一阶段先不强求 synthetic first token 的 request-output 事件。

### 阶段 B：再补齐 NanoDeploy 风格的 synthetic first token

目标：

1. 在 async dummy-prefill 完成时，发出一个 synthetic token 对应的 `EngineCoreOutput`。
2. 让 TTFT 与 NanoDeploy `dummy_prefill` 的语义一致。

这是可选增强，但如果你希望 benchmark 的 TTFT 直接落到“dummy-prefill 结束”的时刻，就必须做这一阶段。

## 6. 详细改动方案

## 6.1 `vllm/distributed/kv_transfer/kv_connector/v1/base.py`

### 目的

给 connector 一个**只调用一次**的 request prepare hook，避免把 mutation 塞进 `get_num_new_matched_tokens()`。

### 建议新增接口

新增一个 scheduler-side 默认 no-op 方法，例如：

```python
def prepare_request_for_external_kv(self, request: Request) -> None:
    return
```

### scheduler 调用时机

在 scheduler 处理 WAITING/PREEMPTED 请求、且调用 `get_num_new_matched_tokens()` 之前，增加：

```python
if self.connector is not None:
    self.connector.prepare_request_for_external_kv(request)
```

### 这样做的好处

1. 保持 `get_num_new_matched_tokens()` 尽量接近 side-effect free。
2. synthetic token append 逻辑的生命周期更清晰。
3. 以后如果要支持别的 dummy connector，也有统一入口。

## 6.2 `vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py`

这是本次改动的核心。

### 6.2.1 新增配置项

建议在 `kv_connector_extra_config` 增加：

1. `dummy_prefill: bool = False`
2. `dummy_output_token_id: int = 0`
3. `emit_synthetic_first_token: bool = False`

说明：

1. `dummy_prefill=False` 时，旧逻辑保持不变。
2. `dummy_output_token_id` 默认可以先用 `0`，因为 benchmark 语义不重要。
3. `emit_synthetic_first_token` 用于控制是否做阶段 B。

### 6.2.2 新增 scheduler-side 状态

建议新增：

1. `_prepared_requests: set[str]`
   - 表示哪些请求已经 append 过 synthetic token
2. `_pending_loads: dict[str, tuple[tuple[list[int], ...], int]]`
   - 记录本轮需要“逻辑上 load”的 block 与 token 数
3. `_synthetic_token_ids: dict[str, int]`
   - 保存每个请求的 synthetic token id
4. `_synthetic_output_emitted: set[str]`
   - 如果做阶段 B，用于防止重复发 synthetic output

### 6.2.3 `prepare_request_for_external_kv()`

在 `DecodeBenchConnector` / `DecodeBenchConnectorScheduler` 中实现：

1. 仅当 `dummy_prefill=True` 且请求从未 prepare 过时生效。
2. 对 request 做一次幂等 mutation：

```python
request.append_output_token_ids(dummy_output_token_id)
```

这样 request 会变成：

```text
num_output_tokens += 1
num_tokens += 1
all_token_ids += [dummy_output_token_id]
block_hashes 也会同步更新
```

### 6.2.4 `get_num_new_matched_tokens()`

两种分支：

1. `dummy_prefill=False`
   - 保持旧逻辑
2. `dummy_prefill=True`
   - 对首个命中请求返回：

```text
num_external_tokens = request.num_tokens - num_computed_tokens
is_async = True
```

注意：

1. 此时 `request.num_tokens` 已经是 `P + 1`。
2. 所以 external KV 会覆盖 `prompt + synthetic token`。
3. 不再是旧逻辑里的 `prompt - 1`。

### 6.2.5 `update_state_after_alloc()`

保留 block 记录逻辑，但语义改成：

1. 这些 block 不再需要 fill。
2. 它们只用于让 worker 知道哪些 request 已经“逻辑上 load 完成”。

### 6.2.6 worker 侧 `start_fill_kv()`

改成真正 no-op，只记录：

1. 本步哪些 req_id 被“load”了
2. 对应的 batch load 时间

可以新增 worker-side 状态：

1. `_latest_finished_recving_req_ids: set[str]`
2. `_latest_req_batch_load_kv_ns: dict[str, int]`

### 6.2.7 worker 侧 `get_finished()`

这是必须加的。

当前 `DecodeBenchConnector` 继承基类默认实现，会返回 `(None, None)`。

但如果 scheduler 走 async load 路径，而 worker 不回报 `finished_recving`，请求会永远卡在 `WAITING_FOR_REMOTE_KVS`。

因此需要实现：

```python
def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
    return None, latest_finished_recving_req_ids or None
```

每一步返回后要清空这批状态，避免重复上报。

### 6.2.8 worker metadata 扩展

如果要做阶段 B，建议把 `DecodeBenchConnectorWorkerMetadata` 扩成：

1. `req_batch_load_kv_ns: dict[str, int]`
2. `req_synthetic_output_token_ids: dict[str, int]`

这样 scheduler 在 `update_from_output()` 时能同时拿到：

1. synthetic token id
2. 对应的 load_kv timing

## 6.3 `vllm/v1/core/sched/scheduler.py`

这个文件有三处关键改动。

### 6.3.1 在 connector match 之前调用 prepare hook

在 WAITING/PREEMPTED 请求的调度路径中，调用：

```python
self.connector.prepare_request_for_external_kv(request)
```

### 6.3.2 async dummy-prefill 这一步也记录第一次 `SCHEDULED`

如果未来要做阶段 B，并让 synthetic token 真正成为 TTFT 的 first token，那么在请求第一次被 dummy-prefill 派发时，必须就记录一次：

```python
request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
```

原因：

1. vLLM 的 metrics 里 `first_token_ts` 和 `scheduled_ts` 是配对使用的。
2. 如果 synthetic first token 先到，但 `scheduled_ts` 还是 0，就会破坏 `prefill_time` / `queued_time` 的统计。
3. 后续真实 decode 再次进入 scheduler 时，即便又记录一次 `SCHEDULED`，`metrics/stats.py` 也只会保留第一次 `scheduled_ts`。

注意：

1. 如果当前只做阶段 A，不做 synthetic output，则这一条先不要加。
2. 否则 TTFT 会提前，但你还没有 first-token output 与之对应。

### 6.3.3 在 `update_from_output()` 中处理 synthetic first token

当前 `update_from_output()` 已经会 special-case `DecodeBenchConnectorWorkerMetadata` 来拿 `req_batch_load_kv_ns`。

建议在同一个位置扩展出：

1. `decode_bench_synthetic_tokens: dict[str, int] | None`

然后新增一条专门的 synthetic output 生成逻辑：

1. 不通过 `_update_request_with_output()`，因为 request 上的 token 已经 append 过了。
2. 直接构造 `EngineCoreOutput`。
3. `new_token_ids=[synthetic_token_id]`
4. `events=request.take_events()`
5. `ttft_trace_update` 里填 `first_batch_load_kv_ns`

### 6.3.4 建议新增一个 helper

例如：

```python
def _make_decode_bench_synthetic_output(...):
    ...
```

这样逻辑更清晰，也能避免把现有“真实 sampled token”路径弄脏。

## 6.4 `vllm/v1/core/sched/output.py`

### 目的

把“新请求自带已有 output token”从 scheduler 传到 V1 worker。

### 建议新增字段

在 `NewRequestData` 中新增：

```python
initial_output_token_ids: list[int] | None = None
```

`from_request()` 里填：

```python
initial_output_token_ids=list(request.output_token_ids)
```

### 需要同步更新

1. `__repr__`
2. `anon_repr`

## 6.5 `vllm/v1/worker/gpu_model_runner.py`

### 6.5.1 `scheduled_new_reqs` 的构造

当前 `GPUModelRunner._update_states()` 在新建 `CachedRequestState` 时写死：

```python
output_token_ids=[]
```

这会把 scheduler 侧准备好的 synthetic token 丢掉。

必须改成：

```python
output_token_ids=new_req_data.initial_output_token_ids or []
```

### 6.5.2 这一步改完后，V1 runner 会自然获得目标状态

当请求在 dummy-prefill 完成后、第一次真正作为 `scheduled_new_req` 加入 V1 runner 时，状态将是：

```text
prompt_token_ids = 原始 prompt
output_token_ids = [synthetic_token]
num_computed_tokens = prompt_len
```

这时：

1. `CachedRequestState.num_tokens = prompt_len + 1`
2. `InputBatch.add_request()` 会把 synthetic token 写进 `token_ids_cpu`
3. 下一次 `_prepare_inputs()` 读取的 query token 会是 synthetic token
4. 该步 forward 就是真 decode

### 6.5.3 为什么 V1 runner 这里不需要额外 special-case

V1 runner 的 token 组装逻辑本来就是按：

```text
当前位置 = num_computed_tokens
query token = token_ids[num_computed_tokens]
```

只要：

1. token buffer 里 `prompt_len` 位置上已经有 synthetic token
2. `num_computed_tokens == prompt_len`

它自然就会进入 decode 语义。

## 6.6 `vllm/benchmarks/offline_poisson_harness.py`

这一项不是必须，但建议在方案里提前说明语义变化。

如果采用阶段 B，并让 synthetic token 计入 request output：

1. benchmark 看到的 `output_len` 里会包含这个 synthetic token。
2. 对 `max_tokens=1024` 的请求，真实模型只会再生成 `1023` 个 token。

建议有两种做法：

1. 第一版先接受这个语义变化，并在文档里写清楚。
2. 后续如果要保留“真实 decode token 数 = max_tokens”，再单独设计不计入 `max_tokens` 的 synthetic token 机制。

对你当前的 benchmark 诉求，建议先选第 1 种。

## 7. 端到端状态机示例

设：

```text
prompt_len = 1024
max_tokens = 1024
dummy_output_token_id = 0
```

### 第 0 步：请求刚进来

```text
request.num_output_tokens = 0
request.num_tokens = 1024
request.num_computed_tokens = 0
```

### 第 1 步：prepare_request_for_external_kv()

append synthetic token：

```text
request.output_token_ids = [0]
request.num_output_tokens = 1
request.num_tokens = 1025
request.all_token_ids = prompt + [0]
```

### 第 2 步：connector claim external KV

```text
num_external_tokens = 1025
load_kv_async = True
```

### 第 3 步：scheduler allocate blocks

为 `1025` 个 token 分配 block，但本步不 forward。

请求进入：

```text
status = WAITING_FOR_REMOTE_KVS
num_computed_tokens = 1025
```

### 第 4 步：worker 立即回报 finished_recving

因为 worker 没有真实搬运 KV，只是同步 no-op。

### 第 5 步：scheduler `_update_waiting_for_remote_kv()`

full-hit 修正后：

```text
request.num_computed_tokens = 1024
request.num_tokens = 1025
```

这正好变成 one-behind-total-length。

### 第 6 步：下一轮真正 schedule

```text
num_new_tokens = request.num_tokens - request.num_computed_tokens = 1
```

worker 新请求状态：

```text
prompt_token_ids = 原始 prompt
initial_output_token_ids = [0]
num_computed_tokens = 1024
```

### 第 7 步：V1 runner prepare input

这一步 query token 读取的是位置 `1024` 的 token，也就是 synthetic token。

因此：

1. 不是最后 1 个 prompt token 的 prefill
2. 而是真正的 decode

## 8. 分阶段实施顺序

## 8.1 第一阶段：只打通 pure decode，不追求 synthetic TTFT

建议先实现这些最小闭环：

1. `base.py` 新增 prepare hook
2. `scheduler.py` 调用 prepare hook
3. `decode_bench_connector.py`
   - append synthetic token
   - external tokens = prompt + synthetic
   - async load
   - worker `get_finished()` 立即回报 finished_recving
4. `output.py` 新增 `initial_output_token_ids`
5. `gpu_model_runner.py` 正确接收该字段

完成后先验证：

1. 请求不再卡在 `WAITING_FOR_REMOTE_KVS`
2. 下一轮确实只 schedule 1 token
3. 这 1 token 是 decode，不是 prefill

这一阶段不要求：

1. synthetic first token 对外可见
2. TTFT 立即变成 NanoDeploy 的 dummy-prefill 语义

## 8.2 第二阶段：补齐 synthetic first token 输出

在第一阶段稳定后，再实现：

1. async dummy-prefill 那一步记录第一次 `SCHEDULED`
2. `DecodeBenchConnectorWorkerMetadata` 带回 synthetic token id
3. `scheduler.update_from_output()` 直接构造 synthetic `EngineCoreOutput`

验证目标：

1. `first_token_ts` 落在 dummy-prefill 完成时
2. `queued_time` / `prefill_time` 统计不乱
3. 后续真实 decode 仍继续工作

## 9. 验证 checklist

## 9.1 功能正确性

建议加临时日志验证以下断言：

1. request 第一次 prepare 后：

```text
num_tokens == prompt_len + 1
num_output_tokens == 1
```

2. async recv 完成后：

```text
status 从 WAITING_FOR_REMOTE_KVS 回到 WAITING
num_computed_tokens == prompt_len
```

3. 下一轮真正 schedule 时：

```text
num_new_tokens == 1
```

4. V1 worker 新请求状态：

```text
len(initial_output_token_ids) == 1
num_computed_tokens == prompt_len
```

## 9.2 性能预期

如果第一阶段实现正确，预期现象应是：

1. `kv_fill_ms` 继续接近 0
2. `prefill_time_ms` 显著下降
3. TTFT 至少会从“约两步 loaded step”下降到“约一步 decode step”

如果第二阶段也做完，则预期：

1. `ttft_ms` 会进一步下降
2. 甚至接近“queue + async bookkeeping”级别

## 9.3 benchmark 输出语义

要明确确认：

1. synthetic token 是否计入 `output_len`
2. `output_len=1024` 时，真实 decode token 是否只有 `1023`

第一版建议接受这个变化，不要在同一轮里试图把它做成“不计入 max_tokens”。

## 10. 风险与坑点

## 10.1 最容易出错的点

1. **重复 append synthetic token**
   - 必须用 `_prepared_requests` 防重
2. **请求卡死在 `WAITING_FOR_REMOTE_KVS`**
   - worker 必须实现 `get_finished()`
3. **worker 丢掉 initial output token**
   - `gpu_model_runner.py` 不能再写死 `output_token_ids=[]`
4. **synthetic token 被重复写进 request**
   - synthetic output path绝不能走 `_update_request_with_output()`

## 10.2 指标层面的坑

如果做第二阶段：

1. synthetic first token 到来前必须已经有第一次 `SCHEDULED`
2. 否则 `first_token_ts < scheduled_ts` 或 `scheduled_ts == 0`，TTFT 指标会错

## 10.3 功能兼容性风险

这个模式本质上是 benchmark hack，建议直接加 hard guard，拒绝以下请求：

1. prompt logprobs
2. structured outputs
3. pooling
4. multimodal
5. prefix caching

## 11. 推荐的代码编辑清单

下一轮实施时，建议按下面顺序改：

1. `vllm/distributed/kv_transfer/kv_connector/v1/base.py`
   - 加 prepare hook
2. `vllm/v1/core/sched/scheduler.py`
   - 调用 prepare hook
3. `vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py`
   - 加 config
   - prepare request
   - async full external KV
   - worker `get_finished()`
4. `vllm/v1/core/sched/output.py`
   - 加 `initial_output_token_ids`
5. `vllm/v1/worker/gpu_model_runner.py`
   - 消费 `initial_output_token_ids`
6. 跑一次 benchmark，只验证 pure decode 是否打通
7. 再补 synthetic first token 输出与 TTFT 逻辑

## 12. 建议的提交策略

如果你下一轮是分步实现，最稳妥的顺序是：

### 提交 1

“让 DecodeBenchConnector 在 V1 runner 下支持 async dummy-prefill，并把新请求真正推进到 pure decode”

只做：

1. async load
2. synthetic token 进入 request state
3. V1 runner 接收 initial output token

### 提交 2

“补齐 synthetic first token 的 TTFT / output 语义”

只做：

1. synthetic `EngineCoreOutput`
2. TTFT trace / metrics 修正

这样更容易定位问题。

## 13. 最终结论

要把 `DecodeBenchConnector` 真正改成“直接分配整段 prompt 的 KV，然后进入 pure decode”，在 **V1 runner** 下最关键的不是 `_fill_blocks()`，而是这三件事：

1. **请求长度必须先扩成 `prompt + 1 synthetic token`**
2. **connector 必须走“立即完成”的 async load 路径**
3. **V1 runner 必须接住这个 synthetic token，作为新请求的已有 output token**

只有三者同时成立，下一轮 GPU 才会真正进入 decode，而不是继续跑最后 1 个 prompt token 的 prefill。

