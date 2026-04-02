# Offline Poisson Harness Dummy-Prefill TTFT 去除 Forward 时间设计

## 1. 目标

这份设计只解决一个很窄但明确的问题：

- 只针对 `offline_poisson_harness.py` 这条 benchmark 链路
- 只针对 `DecodeBenchConnector + dummy_prefill=True` 这套 setting
- 只修正 `requests.jsonl` / `summary.json` 里的 `ttft_ms` 语义

目标语义：

- `ttft_ms` 应当截止到“请求第一次被真实 decode 调度”的时刻
- `ttft_ms` 不包含第一次真实 forward
- `ttft_ms` 也不依赖文档解释，而是直接依据当前代码路径定义

非目标：

- 不改通用 vLLM `first_token_latency` 的定义
- 不改 OpenAI API server 的 TTFT 定义
- 不在这一步追求“dummy-prefill 完成时刻”这一更激进的语义

## 2. 当前代码行为

当前 `offline_poisson_harness` 的默认配置会启用：

- `kv_connector = DecodeBenchConnector`
- `dummy_prefill = True`
- `RequestOutputKind.FINAL_ONLY`

当前 `summary.json` 的 `ttft_ms` 来自：

```text
RequestOutput.metrics.first_token_latency * 1000
```

也就是 harness 直接把“到第一个真实 token 为止”的延迟写成 TTFT。

但在 `dummy_prefill=True` 时，实际请求生命周期不是“正常 prefill -> first token”，而是：

1. 请求进入 scheduler，记录 `QUEUED`
2. scheduler 走 async remote-KV 分支，request 进入 `WAITING_FOR_REMOTE_KVS`
3. 这一步不会产生真实 token，也不会向 frontend 交付 `RequestOutput`
4. 等 `finished_recving` 到达后，请求被重新放回可调度队列
5. 第二次被真正 `SCHEDULED` 时，才会去跑第一次真实 decode
6. 这次真实 decode 完成后，frontend 才在 `OutputProcessor` 里把 `first_token_latency` 记下来

因此，当前 `ttft_ms` 实际包含：

- API preprocess
- frontend 到 decode 侧的 IPC
- engine preprocess
- 进入 scheduler 后的等待
- dummy-prefill / remote KV ready 之前的等待
- KV ready 之后到真正被调度的等待
- 一次真实 decode forward
- sampling / 输出回传 / frontend output processing 的尾巴

换句话说，当前 `ttft_ms` 明确包含一次真实 forward。

## 3. 修正后的目标语义

### 3.1 推荐终点

推荐把这套 setting 下的 TTFT 终点定义为：

```text
第一次真实 decode 被 SCHEDULED 的时刻
```

也就是：

- 请求已经完成 dummy-prefill 相关等待
- 请求已经真正拿到一次 decode 执行资格
- 但还没有进入第一次真实 forward

### 3.2 为什么选这个终点

这是当前代码里最合适的边界点，原因有三条：

1. 它已经存在于现有指标里，不需要改 engine
2. 它天然排除了第一次真实 forward
3. 它仍然保留了 dummy-prefill 期间的等待，因此不会把 TTFT 缩成“只剩 connector 内部耗时”

### 3.3 这一定义不等于什么

它不等于“dummy-prefill 刚完成”的时刻。

也就是说，修正后的 TTFT 仍然会包含：

- KV ready 之后，到请求真正被 scheduler 选中运行 decode 之前的等待

这是刻意保留的，因为本次目标只是“去掉 forward 时间”，不是把 TTFT 改成“KV ready latency”。

## 4. 关键观察：现有字段已经够用

对这套 setting 来说，不需要新增 engine 打点，就已经能在 harness 里算出“去掉 forward”的 TTFT。

原因是当前代码已经提供了两类信息：

1. request-scoped 持续时间
   - `api_preprocess_ns`
   - `ipc_in_decode_ns`
   - `engine_preprocess_ns`

2. engine-core monotonic 时间戳
   - `queued_ts`
   - `scheduled_ts`

在 `dummy_prefill=True` 路径下：

- `queued_ts` 是请求进入 scheduler 队列的时刻
- `scheduled_ts` 是请求第一次真正进入 `RUNNING`、准备执行 decode 的时刻
- async remote-KV 的那一拍不会写入 `SCHEDULED`
- `scheduled_ts` 只记录第一次 `SCHEDULED`，后续 preemption 不会改写它

因此：

```text
(scheduled_ts - queued_ts)
```

在这条路径里恰好就是：

```text
进入 scheduler 后，到第一次真实 decode 调度之前的全部等待
```

它包含 dummy-prefill / remote-KV 等待，但不包含真实 forward。

### 4.1 一个重要限制

`queue_time_ms` 在当前 vLLM 语义下不是“首 token 之前所有等待时间”。

它只表示：

```text
第一次 QUEUED -> 第一次 SCHEDULED
```

这意味着：

1. 它会包含 `WAITING_FOR_REMOTE_KVS` 这类 blocked waiting 时间
2. 但如果请求在第一次真实 `SCHEDULED` 之后、首 token 之前又发生 preemption
3. 那么这段额外等待不会进入 `queue_time_ms`
4. 它会落到 `prefill_time_ms = first_token_ts - scheduled_ts`

所以：

- 如果当前场景里“第一次真实 decode 在首 token 前基本不会被 preempt”，那么 `queue_time_ms` 很适合作为 pre-forward TTFT 的主边界
- 如果这个前提不成立，那么单用 `queue_time_ms` 会低估“首 token 之前但不含 forward 的总等待”

## 5. 推荐公式

### 5.1 新 `ttft_ms`

当且仅当同时满足下面两个条件时：

- `connector_mode == "decode_bench"`
- `dummy_prefill == True`

把 harness 输出里的 `ttft_ms` 改为：

```text
ttft_ms =
    api_preprocess_ms
  + ipc_in_decode_ms
  + engine_preprocess_ms
  + queued_to_first_schedule_ms
```

其中：

```text
api_preprocess_ms = ttft_trace.api_preprocess_ns / 1e6
ipc_in_decode_ms = ttft_trace.ipc_in_decode_ns / 1e6
engine_preprocess_ms = ttft_trace.engine_preprocess_ns / 1e6
queued_to_first_schedule_ms = max(scheduled_ts - queued_ts, 0) * 1000
```

### 5.2 老 TTFT 保留为对照项

当前的：

```text
first_token_latency * 1000
```

不要丢掉，建议改名保留为：

```text
ttft_including_forward_ms
```

这样做有两个好处：

1. 不会把语义变更悄悄藏在同一个字段名后面
2. 可以直接量化“调度边界到首 token 返回”的额外延迟

### 5.3 可选对照项

建议额外输出：

```text
ttft_post_schedule_to_first_token_ms =
    max(ttft_including_forward_ms - ttft_ms, 0)
```

注意这个字段不能命名成 `forward_ms`，因为它实际包含：

- forward
- sampling
- 输出回传
- frontend output processing

它只是“第一次真实 decode 被调度之后，到第一个真实 token 被 frontend 看到之间的 gap”。

## 6. 为什么不推荐用 `kv_fill_ms`

这次修正不应该依赖：

```text
ttft_trace.first_batch_load_kv_ns
```

原因：

1. 这个字段只覆盖 batch load KV，本身不覆盖 load 之前的等待
2. 它不是“去掉 forward TTFT”的完整定义
3. 当前 `dummy_prefill` 路径里存在 synthetic token 先写入 request 的行为，`first_batch_load_kv_ns` 的 merge 时机不适合作为唯一真值

所以推荐完全绕开 `kv_fill_ms`，直接使用：

- trace 里的 preprocess/IPC 持续时间
- `queued_ts -> scheduled_ts` 这段现成边界

## 7. 落地方案

### 7.1 改动范围

只需要改 benchmark harness 侧：

- `vllm/benchmarks/offline_poisson_harness.py`
- `tests/benchmarks/test_offline_poisson_harness.py`

不需要改：

- `vllm/v1/engine/*`
- `vllm/v1/core/*`
- `DecodeBenchConnector` 本体

### 7.2 建议新增 helper

建议在 harness 里新增两个 helper：

```python
def _is_decode_bench_dummy_prefill(kv_transfer_config: Any) -> bool: ...
def _compute_pre_forward_ttft_ms(metrics: Any) -> float | None: ...
```

其中 `_compute_pre_forward_ttft_ms(metrics)` 逻辑：

1. 如果 `metrics is None`，返回 `None`
2. 读取 `metrics.ttft_trace`
3. 读取 `metrics.queued_ts` 和 `metrics.scheduled_ts`
4. 任一缺失则返回 `None`
5. 按推荐公式计算并返回

注意：

- 这个 helper 只计算“到第一次真实 `SCHEDULED` 为止”的 pre-forward TTFT
- 它不覆盖“第一次 `SCHEDULED` 之后、首 token 之前的 preemption 等待”

### 7.3 `build_success_record()` 建议调整

当前 `build_success_record()` 里直接写：

```python
ttft_ms = float(metrics.first_token_latency * 1000.0)
```

建议改成：

1. 总是先算：
   - `ttft_including_forward_ms`
2. 如果当前 run 满足 `decode_bench + dummy_prefill`：
   - 优先把 `ttft_ms` 设为 `_compute_pre_forward_ttft_ms(metrics)`
   - 如果算不出来，再回退到 `ttft_including_forward_ms`
3. 额外写出：
   - `ttft_including_forward_ms`
   - `ttft_post_schedule_to_first_token_ms`

### 7.4 `summary.json` 建议调整

建议 `summary.json` 至少聚合下面几组指标：

- `ttft_ms`
- `ttft_including_forward_ms`
- `ttft_post_schedule_to_first_token_ms`
- `queued_time_ms`

并增加一个显式语义字段，例如：

```json
"ttft_semantics": "decode_bench_dummy_prefill_pre_forward_schedule_boundary"
```

对于非 `dummy_prefill` 场景则保持：

```json
"ttft_semantics": "first_real_token_latency"
```

### 7.5 `run_meta.json` 建议调整

建议把 TTFT 定义也写进 `run_meta.json`，避免后处理脚本只看字段名误读。

示例：

```json
"ttft_definition": {
  "mode": "pre_forward_schedule_boundary",
  "applies_when": "DecodeBenchConnector && dummy_prefill",
  "formula": "api_preprocess_ms + ipc_in_decode_ms + engine_preprocess_ms + (scheduled_ts - queued_ts) * 1000"
}
```

## 8. 示例

假设某请求拿到的原始值是：

```text
api_preprocess_ms = 1.8
ipc_in_decode_ms = 0.9
engine_preprocess_ms = 1.1
queued_time_ms = 122.0
ttft_including_forward_ms = 136.7
```

那么修正后：

```text
ttft_ms = 1.8 + 0.9 + 1.1 + 122.0 = 125.8
ttft_including_forward_ms = 136.7
ttft_post_schedule_to_first_token_ms = 10.9
```

解释：

- `125.8ms` 是“不含第一次真实 forward”的 TTFT
- `10.9ms` 是“进入第一次真实 decode 调度之后，到第一个真实 token 被 frontend 看到”为止的剩余 gap

## 9. 测试方案

建议至少补下面几类测试。

### 9.1 harness 单元测试

新增一个纯 helper 测试：

- 构造带 `queued_ts` / `scheduled_ts` / `ttft_trace` 的 fake metrics
- 断言 `_compute_pre_forward_ttft_ms()` 返回值等于公式结果

### 9.2 `build_success_record()` 回归测试

补两组：

1. `decode_bench + dummy_prefill`
   - `ttft_ms` 使用新公式
   - `ttft_including_forward_ms` 保留旧值
   - `ttft_post_schedule_to_first_token_ms` 正确

2. 非 `dummy_prefill`
   - `ttft_ms` 保持旧语义

### 9.3 `summary.json` 聚合测试

断言：

- `ttft_ms` 聚合的是新值
- `ttft_including_forward_ms` 聚合的是旧值
- `ttft_semantics` 正确写出

## 10. 风险与边界

### 10.1 这不是“GPU kernel start”级别的严格边界

`scheduled_ts` 是 scheduler 侧边界，不是 GPU kernel launch 时间。

因此修正后的 `ttft_ms` 还会排除掉一小段：

- schedule 之后，到真正发起 execute_model 之前的 CPU 侧间隙

这是接受的，因为当前目标是“保证不含 forward”，不是“精确贴住 GPU kernel 起点”。

### 10.2 这也不是“首 token 前全部非-forward 时间”

由于 `queue_time_ms` 只到第一次 `SCHEDULED`：

- 若请求在第一次真实 `SCHEDULED` 之后、首 token 之前被 preempt
- 那么这部分等待会进入 `prefill_time_ms`
- 不会进入本文定义的 `ttft_ms`

因此本文方案隐含一个工程前提：

```text
第一次真实 decode 到首 token 之间通常不会发生显著 preemption
```

如果后续实测发现这个前提不成立，就应该升级方案，新增 request-scoped 的：

- `first_real_decode_scheduled_ts`
- `first_real_decode_started_ts`
- `first_real_decode_completed_ts`

或者至少新增“pre-first-token preemption wait”打点，而不是继续复用现有 `queue_time_ms`。

### 10.3 字段名兼容风险

如果直接重定义 `ttft_ms`，已有分析脚本可能会把新旧 run 混在一起比较。

因此建议：

1. 保留 `ttft_including_forward_ms`
2. 增加 `ttft_semantics`
3. 在 `run_meta.json` 里显式写公式

### 10.4 未来代码路径变化风险

这个方案依赖一个当前成立的代码不变量：

```text
dummy_prefill 的 async remote-KV 那一拍不会写入 SCHEDULED
```

如果未来 scheduler 改了这条语义，这个公式就要重新审视。

所以测试里应当锁住这个假设，而不是只锁数值。

## 11. 备选方案与取舍

### 11.1 备选方案 A：新增 `KV_READY` 事件，把 TTFT 截止到 dummy-prefill 完成

优点：

- 语义更激进，更接近“纯 dummy-prefill 完成时刻”

缺点：

- 需要改 engine/core 事件流
- 会把 `KV ready -> 真正被调度` 这段等待也排除掉
- 超出了当前“只是不想包含 forward 时间”的需求

### 11.2 备选方案 B：继续用 `first_token_latency`，再减一个估算 forward 时间

不推荐。

原因：

- 没有稳定的 request-scoped “forward 纯耗时”可直接减
- 减法容易混入 sampling / IPC / output processor 的尾巴
- 结果会比直接把边界改到 `scheduled_ts` 更脆弱

### 11.3 推荐结论

本次推荐采用：

```text
以第一次真实 decode 的 SCHEDULED 边界作为新 TTFT 终点
```

理由：

- 只改 harness
- 不改 engine
- 不依赖 `kv_fill_ms`
- 能稳定排除 forward
- 语义直接对应当前用户诉求

## 12. 最终建议

最终建议是：

1. 在 `offline_poisson_harness` 中，把 `decode_bench + dummy_prefill` 场景的 `ttft_ms` 改成“pre-forward TTFT”
2. 公式采用：

```text
api_preprocess_ms + ipc_in_decode_ms + engine_preprocess_ms + queued_to_first_schedule_ms
```

3. 保留旧指标为 `ttft_including_forward_ms`
4. 增加 `ttft_post_schedule_to_first_token_ms`
5. 在 `summary.json` 和 `run_meta.json` 写清楚 `ttft_semantics`

这样改完之后，这个脚本 setting 下的 TTFT 就不再包含第一次真实 forward 时间。
