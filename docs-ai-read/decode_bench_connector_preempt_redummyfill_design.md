# DecodeBenchConnector Preempt 后重新走 Dummy Fill 设计

## 1. 目标

本文档定义一套实现方案，使 `DecodeBenchConnector + dummy_prefill=True` 在请求被 scheduler preempt 之后，恢复时仍然重新走 async dummy fill，而不是退化成真实 prefill。

目标行为：

1. 请求首次调度时，现有 `dummy_prefill` 语义保持不变。
2. 请求在 decode 阶段被 preempt 后，本地 KV 被释放。
3. 请求恢复调度时，connector 必须重新声明该请求的 external KV hit。
4. 恢复路径仍然进入 `WAITING_FOR_REMOTE_KVS -> schedulable`，而不是进入真实 prefill。
5. 不允许重复 append synthetic output token。

非目标：

1. 不改变 `DecodeBenchConnector` 的 benchmark-only 定位。
2. 不让 worker 真的去填充 KV tensor。
3. 不在这一步修改 TTFT 语义。
4. 不在 scheduler 中为 `DecodeBenchConnector` 写硬编码分支。

## 2. 问题背景

`offline_poisson_harness` 默认会启用：

1. `kv_connector = DecodeBenchConnector`
2. `dummy_prefill = True`

对应代码见：

1. `vllm/benchmarks/offline_poisson_harness.py`
2. `vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py`

在这条 benchmark 路径里，首次调度的关键行为是：

1. `prepare_request_for_external_kv()` 只在第一次给 request append 1 个 synthetic output token。
2. `get_num_new_matched_tokens()` 对首次命中的 request 返回完整 external hit，并走 async load。
3. worker 侧不真正填充 KV，只同步回报 `finished_recving`。
4. scheduler 在 `_update_waiting_for_remote_kv()` 中把请求推进到“下一轮可以纯 decode”的状态。

该路径本身没有问题。问题出在 request 后续被 preempt 的场景。

## 3. 当前失效机理

### 3.1 首次 dummy-prefill 调度时的状态

对一个 prompt 长度为 `P` 的新请求：

1. connector 先 append 1 个 synthetic token。
2. request 长度变成 `P + 1`。
3. connector 返回 `P + 1` 个 external tokens，并要求 async load。
4. scheduler 分配 blocks，把 request 置为 `WAITING_FOR_REMOTE_KVS`。
5. worker 同步回报 `finished_recving`。
6. scheduler 在 `_update_waiting_for_remote_kv()` 中缓存 blocks。
7. 如果是 full hit，scheduler 会把 `num_computed_tokens` 从 `P + 1` 修正为 `P`，以便下一轮真正做 1-token decode。

此时 request 已经进入正确的 dummy-prefill 之后的 decode 语义。

### 3.2 preempt 之后为什么会退化成真实 prefill

当前 preempt 路径中，scheduler 会：

1. 释放 request 的本地 KV blocks。
2. 把 `request.status` 置为 `PREEMPTED`。
3. 把 `request.num_computed_tokens` 置回 `0`。

但 `DecodeBenchConnector` scheduler 侧并不知道“本地 KV 已经失效”这件事。其内部状态仍然保留：

1. `_filled_requests` 仍包含该 request id。
2. `_prepared_requests` 仍包含该 request id。

于是 request 恢复调度时：

1. scheduler 因为 `num_computed_tokens == 0`，再次进入 connector 路径。
2. `prepare_request_for_external_kv()` 因 `_prepared_requests` 命中，不会重复 append synthetic token。
3. `get_num_new_matched_tokens()` 因 `_filled_requests` 命中，直接返回 `0` external tokens。

这会导致 scheduler 认为“没有 external KV 可用”，从而回退到本地真实 prefill 路径。

这正是本设计要修复的问题。

## 4. 根因抽象

根因不是 worker 没有收到 preempt 通知。worker 侧已经有 `handle_preemptions()` 机制，主要服务于异步 save 类 connector。

本问题的根因是：

1. scheduler 释放了 request 的本地 KV。
2. 但 connector 的 scheduler-side bookkeeping 没有同步失效。

因此这是一个 scheduler-side state invalidation 问题，而不是 worker-side transfer 问题。

## 5. 设计原则

### 5.1 不在 scheduler 中写 connector-specific 分支

不推荐在 scheduler 里写：

1. `if isinstance(connector, DecodeBenchConnector): ...`
2. 或者只在 `DecodeBenchConnector` 场景下特判 `PREEMPTED`

原因：

1. scheduler 不应该理解单个 connector 的内部状态机。
2. “本地 KV 已失效”是一个通用语义，不是 `DecodeBenchConnector` 私有语义。
3. 将来其它 connector 也可能需要响应同类事件。

### 5.2 清理的是“已填充状态”，不是“已准备状态”

preempt 后必须清掉的是：

1. `_filled_requests`

因为它表示“connector 认为该 request 的 external KV 已经可用并已完成过一次 fill/claim”。

preempt 后不能清掉的是：

1. `_prepared_requests`

因为它保护的是“synthetic token 只 append 一次”。如果把它也清掉，request 恢复时会重复 append synthetic token，导致 request 长度和 output token 语义损坏。

### 5.3 恢复时应重新声明当前完整 request 的 external KV

恢复时不应只重新声明原始 prompt 长度。

正确语义是：

1. request 当前已经包含 prompt token
2. request 还包含首次 dummy-prefill 加进去的 synthetic token
3. request 还可能已经包含若干真实 decode 输出 token

因此 preempt 后若要无重算恢复，就应该重新声明“当前 `request.num_tokens` 对应的完整前缀 KV 都来自 external”。

这恰好与现有 `DecodeBenchConnector` 逻辑兼容：

1. 恢复后 `num_computed_tokens = 0`
2. `get_num_new_matched_tokens()` 会按当前 `request.num_tokens` 返回 full hit
3. `_update_waiting_for_remote_kv()` 在 full hit 时把 `num_computed_tokens` 修正为 `request.num_tokens - 1`
4. 下一轮继续做真正的 decode

## 6. 推荐方案

## 6.1 在 KV connector base 增加 scheduler-side invalidation hook

建议在 `vllm/distributed/kv_transfer/kv_connector/v1/base.py` 增加一个新的 scheduler-side default no-op hook。

建议接口名：

```python
def request_local_kv_invalidated(self, request: Request) -> None:
    return
```

命名可以在实现时微调，但语义必须保持一致：

1. scheduler 已经使某个 live request 的本地 KV 失效
2. request 对象本身还活着，之后可能被重新调度
3. connector 可以在这里丢弃任何“依赖本地 KV 仍然存在”的内部状态

该接口的要求：

1. 默认 no-op，保证对其它 connector 零行为变化。
2. 只允许修改 connector 自身 bookkeeping。
3. 不应直接改写 request 的用户可见输出语义。

## 6.2 在 scheduler preempt 路径调用该 hook

建议在 `vllm/v1/core/sched/scheduler.py::_preempt_request()` 中调用该 hook。

推荐调用顺序：

1. `self.kv_cache_manager.free(request)`
2. `self.encoder_cache_manager.free(request)`
3. `if self.connector is not None: self.connector.request_local_kv_invalidated(request)`
4. `request.status = PREEMPTED`
5. `request.num_computed_tokens = 0`
6. `request.num_external_computed_tokens = 0`

其中：

1. hook 放在本地 KV 真正释放之后，语义最清晰。
2. `num_external_computed_tokens = 0` 不是本修复的唯一关键，但建议一并做掉，以避免 request 上残留过期 external-hit 计数。

## 6.3 `DecodeBenchConnector` 的 hook 实现

在 `DecodeBenchConnectorScheduler` 中实现该 hook，建议行为如下：

```python
def request_local_kv_invalidated(self, request: Request) -> None:
    req_id = request.request_id
    self._filled_requests.discard(req_id)
    self._pending_fills.pop(req_id, None)
```

必须满足：

1. 清掉 `_filled_requests`
2. 清掉 `_pending_fills`
3. 保留 `_prepared_requests`

理由：

1. 清 `_filled_requests` 后，恢复调度时 `get_num_new_matched_tokens()` 才会重新返回 full external hit。
2. 清 `_pending_fills` 是为了避免留下无意义的旧 step 残留状态。
3. 保留 `_prepared_requests` 才能避免重复 append synthetic token。

明确禁止：

```python
self._prepared_requests.discard(req_id)
```

因为这会导致恢复调度时再次 append synthetic token。

## 7. 预期状态机

## 7.1 首次 dummy-prefill 路径

```text
WAITING
  -> connector prepare_request_for_external_kv()
  -> connector get_num_new_matched_tokens() returns full hit, async=True
  -> WAITING_FOR_REMOTE_KVS
  -> finished_recving
  -> WAITING
  -> RUNNING
```

其中进入 `RUNNING` 前，request 应处于：

```text
num_tokens = current_prefix_len
num_computed_tokens = current_prefix_len - 1
```

## 7.2 preempt 后恢复路径

```text
RUNNING
  -> PREEMPTED
  -> connector request_local_kv_invalidated()
  -> reschedule
  -> connector get_num_new_matched_tokens() returns full hit again
  -> WAITING_FOR_REMOTE_KVS
  -> finished_recving
  -> PREEMPTED or WAITING
  -> RUNNING
```

这里恢复后的 full hit 必须覆盖 request 当前完整 prefix，而不是只覆盖原始 prompt。

## 8. 为什么不采用其它方案

### 8.1 方案 A: 只在 `DecodeBenchConnector.get_num_new_matched_tokens()` 里特判 `PREEMPTED`

例如：

```python
if request.status == PREEMPTED and request.num_computed_tokens == 0:
    ...
```

不推荐，原因：

1. 这把“本地 KV 已失效”的通用语义埋进了单个 connector 的私有逻辑。
2. scheduler 明明知道自己刚刚 free 了 blocks，却不通过显式 hook 告诉 connector，抽象层次不对。
3. 未来其它 connector 遇到同类问题还得再复制一遍。

### 8.2 方案 B: preempt 时同时清 `_filled_requests` 和 `_prepared_requests`

不推荐，原因：

1. 恢复调度时会再次 append synthetic token。
2. request 的 `num_tokens`、`num_output_tokens` 和 `all_token_ids` 都会被污染。
3. benchmark 输出语义会被破坏。

### 8.3 方案 C: 依赖 worker-side `handle_preemptions()`

不推荐，原因：

1. 本 bug 是 scheduler-side bookkeeping 失效。
2. worker 侧不知道 request 是否还应该重新声明 external hit。
3. 即使 worker 收到 preempt 通知，也无法替 scheduler 修改 `_filled_requests`。

## 9. 测试计划

建议至少补两层测试。

### 9.1 `DecodeBenchConnector` 单测

在 `tests/v1/kv_connector/unit/test_decode_bench_connector.py` 增加回归测试：

1. 构造 `dummy_prefill=True` 的 request。
2. 先走一轮正常 dummy-prefill，确认：
   - request 进入 decode 语义
   - synthetic token 只追加了一次
3. 人工触发 preempt。
4. 再次 schedule。
5. 断言：
   - `metadata.reqs_to_fill` 再次包含该 request
   - request 再次进入 `WAITING_FOR_REMOTE_KVS`
   - synthetic token 没有重复追加
   - 恢复 promote 后 `num_computed_tokens == request.num_tokens - 1`

### 9.2 scheduler hook 单测

在 `tests/v1/core/test_scheduler.py` 增加通用测试，验证：

1. scheduler preempt request 时会调用 connector 的新 hook
2. 默认 no-op connector 不受影响

这样可以确保本次改动不是只靠 `DecodeBenchConnector` 私有单测兜底。

## 10. 风险与兼容性

### 10.1 对其它 connector 的影响

只要 base hook 默认 no-op，则对其它 connector 的运行时行为应为零变化。

### 10.2 对 `DecodeBenchConnector` 的影响

本设计只改变 preempt 之后的恢复语义。

首次调度路径不应变化：

1. synthetic token 仍然只 append 一次
2. `dummy_prefill` 的 TTFT 语义不变
3. worker 仍然不实际填充 KV

### 10.3 对 async scheduling 和 reset-prefix-cache 的关系

`reset_prefix_cache(reset_running_requests=True)` 也是通过 `_preempt_request()` 路径触发 request 失效。

因此只要 hook 放在 `_preempt_request()` 里，这类“强制 preempt 后再恢复”的路径也能自动获得一致语义，无需单独再写 `DecodeBenchConnector` 特判。

## 11. 实施清单

建议按以下顺序实现：

1. 在 `KVConnectorBase_V1` 增加 scheduler-side invalidation hook。
2. 在 scheduler `_preempt_request()` 中调用该 hook，并清 `request.num_external_computed_tokens`。
3. 在 `DecodeBenchConnectorScheduler` 中实现该 hook：
   - 清 `_filled_requests`
   - 清 `_pending_fills`
   - 保留 `_prepared_requests`
4. 增加 `DecodeBenchConnector` preempt/resume 回归测试。
5. 增加 scheduler hook 调用测试。

## 12. 实现验收标准

满足以下条件即可视为本设计落地成功：

1. `DecodeBenchConnector + dummy_prefill=True` 场景下，request 被 preempt 后恢复时不再进入真实 prefill。
2. 恢复时 `kv_connector_metadata.reqs_to_fill` 会重新出现对应 request。
3. request 不会重复 append synthetic token。
4. 现有 dummy-prefill 单测和 benchmark TTFT 单测不回退。
5. 不启用 `DecodeBenchConnector` 的路径行为不变。
