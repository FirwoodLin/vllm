# Model Runner Graph Replay 与 Scheduler Timing 方案

## 背景

当前目标是在如下实验形态下，稳定记录每个 decode batch 的两个核心时间指标：

- `model runner` 一次 `graph replay` 的真实执行时间
- `scheduler` 的 CPU 开销

本次设计以你实际要跑的路径为前提：

- `scheduler_config.async_scheduling = True`
- `cudagraph_mode = FULL_DECODE_ONLY`
- 覆盖以下并行配置：
  - `32DP`，即 `DP=32, TP=1, DCP=1`
  - `4DP8TP8DCP`
  - `8DP4TP4DCP`
  - `16DP2TP2DCP`
- `MAX_NUM_SEQS=128`

这和之前的同步 `step()` 方案不同。async scheduling 下，`schedule()`、`get_grammar_bitmask()`、`update_from_output()` 不再在同一个调用栈里连续发生，因此统计单位必须从“一个 `EngineCore.step()` 调用”改成“一个已入队并最终出队的 batch ticket”。

本方案明确采用“单一 reply rank 哨兵指标”语义：

- 不做 TP/DCP 组内聚合
- `graph replay` 时间只记录 executor 最终返回给 engine 的那个固定 `reply rank`
- 该指标可用于跨实验配置稳定对比，但它不是 TP/DCP 组内最慢 rank 的 critical-path 时间

## 目标

最终希望在日志中得到每个已完成 batch 的结构化记录，至少包含：

- `reply_global_rank`
- `dp_rank`
- `tp_rank`
- `dcp_rank`
- `node_rank`
- `schedule_cpu_ms`
- `grammar_cpu_ms`
- `update_cpu_ms`
- `scheduler_total_cpu_ms`
- `graph_replay_gpu_ms`
- `graph_replay_wall_ms`
- `graph_replay_count`
- `cudagraph_mode`
- `num_generation_tokens`
- `scheduler_us_per_gen_token`
- `replay_us_per_gen_token`

推荐额外记录两个辅助指标，用于解释 async 路径下的时间差：

- `grammar_deferred`
- `async_output_copy_wait_ms`

注意：这里的“一个 token”更准确地说是“一个 decode batch 内每个 generation token 的均摊时间”。`MAX_NUM_SEQS=128` 时，一次 batch 往往是很多 request 各生成 1 个 token，而不是单个 request 的单 token。

由于只采单一 `reply rank`，上述时间字段的解释应为：

- `scheduler_*` 是 engine 侧这个 batch 的调度 CPU 开销
- `replay_*` 是该 batch 在固定 `reply rank` 上观测到的 replay 时间
- 它们适合做配置间趋势比较，不适合直接当作 TP/DCP 组内全局 critical-path

## 核心变化

相对之前的同步版方案，这次设计有三个关键变化：

1. 主统计单位从 `step()` 改成 `batch ticket`
2. 不再依赖每步 `torch.cuda.synchronize()` 拿 replay 时间
3. `UBatchWrapper` 需要纳入第一阶段，而不是后补

原因：

- async scheduling 下 batch queue 深度可能大于 1，调度与结果处理跨多个 engine 调用分离
- 每步强制同步会明显破坏 async scheduling 的重叠收益
- DP 不均衡和 microbatching 存在时，只量 `CUDAGraphWrapper` 会漏掉部分 replay

## 时间边界定义

### 1. Graph Replay 时间

只统计 `cudagraph.replay()` 本体，不混入：

- worker 侧 preprocess
- logits 计算
- sampling
- bookkeeping
- async output copy

这样拿到的是最接近“单次 graph replay 本体”的时间。

### 2. Scheduler 时间

继续拆成三个阶段，但要接受这三段在 async 路径上发生在不同时间点：

- `schedule_cpu_ms`: `self.scheduler.schedule()`
- `grammar_cpu_ms`: `self.scheduler.get_grammar_bitmask()`
- `update_cpu_ms`: `self.scheduler.update_from_output()`

总调度开销定义为：

```text
scheduler_total_cpu_ms = schedule_cpu_ms + grammar_cpu_ms + update_cpu_ms
```

### 3. Async Output Copy 时间

这不是本次主指标，但建议单独记录，不要混进 `graph_replay`：

- `async_output_copy_wait_ms`: `AsyncGPUModelRunnerOutput.get_output()` 中等待 D2H copy 完成的 wall time

原因是 async scheduling 下，engine 侧 `future.result()` 等到的是“输出 ready”，而不是“graph replay 刚结束”。如果不把 copy wait 拆出来，`future.result()` 的等待时间会比 `replay_gpu_ms` 明显大，日志会不好解释。

## 总体设计

建议采用“worker 采 replay，engine 以 batch ticket 聚合并落日志”的方式：

- `graph replay` 时间在 worker/model runner 路径采集
- `scheduler` 时间在 engine core 路径采集
- 只从 executor 已经返回的单一 `reply rank` 拿 `ModelRunnerOutput` 和 replay timing
- `EngineCore.step_with_batch_queue()` 中每个 scheduled batch 创建一个本地 `BatchTimingTicket`
- batch 出队并完成 `update_from_output()` 后，由 engine 统一输出一条结构化日志

这样做的优点：

- 能正确覆盖 async scheduling 的 batch queue 语义
- 不需要把 timing 元数据塞进 `SchedulerOutput` 跨进程传输
- 不需要在热路径额外做全局 `cuda synchronize`
- 可以直接在 `tee *.log` 中按 batch 分析
- 与现有 executor 只返回单一 `output_rank` 的语义一致，改动面更小

## 建议新增配置

建议把之前的 `strict` 设计拿掉，改成更适合 async scheduling 的两个开关：

- `enable_logging_step_timing_details: bool = False`
- `enable_graph_replay_timing: bool = False`

含义：

- `enable_logging_step_timing_details`
  - 控制是否输出单 batch timing 日志（scheduler timing 部分）
  - 这是主开关，关闭时不输出任何 per-batch timing 日志
- `enable_graph_replay_timing`
  - 控制是否记录 replay 的 CUDA event 与 wall time
  - 开启时自动包含 `async_output_copy_wait_ms` 的记录（`get_output()` 中已有 `synchronize()`，多两行 `perf_counter_ns()` 开销可忽略，不需要单独开关）
  - 隐含要求 `enable_logging_step_timing_details = True`

不再设置独立的 `enable_async_output_timing` 开关。原因：async output copy wait 的测量只是在 `get_output()` 现有的 `synchronize()` 前后各加一行 `perf_counter_ns()`，没有额外的 GPU 同步开销，不值得用一个独立配置控制。

不建议把主方案做成”每 batch 强制同步”模式。对于 async scheduling，这种做法会主动破坏 overlap，测出来的值虽然更像同步真值，但不再代表你真正要跑的系统。

## 实现原则

### 1. 不修改 `SchedulerOutput` 数据结构

`SchedulerOutput` 会跨 engine/worker 边界传输。为 timing 直接给它加本地 ID 或浮点字段，会增加序列化负担，也会把本地调试信息混入执行协议。

更合适的做法是：

- 在 engine core 本地维护 `BatchTimingTicket`
- queue 中存 ticket，而不是现在的裸 tuple

### 2. replay 时间在 output ready 边界再 resolve

async scheduling 下，不应该在 `execute_model()` 或 `sample_tokens()` 热路径里同步 GPU。

更合适的做法是：

- replay 时只记录 CUDA event 对和 wall time
- 在 `AsyncGPUModelRunnerOutput.get_output()` 里，等 async copy 完成后再 resolve event elapsed time

这样不会新增额外同步点，因为 `get_output()` 本来就要等待输出 ready。

### 3. 不做 TP/DCP 组内聚合

这是本方案的显式取舍：

- 不额外引入跨 TP/DCP rank 的 timing 汇聚
- 不追求组内最慢 rank 的 replay 时间
- 只要求拿到一个固定、稳定、可重复的哨兵 rank 指标

这样能显著减少实现复杂度，并避免为了 timing 改写 executor 返回协议。

### 4. `UBatchWrapper` 视为第一阶段必做

DP 不均衡场景下，`FULL` graph replay 可能走 `UBatchWrapper`，不是只走 `CUDAGraphWrapper`。如果只改 `CUDAGraphWrapper`，会出现日志中 `graph_replay_count=0` 但系统实际上在 replay 的假阴性。

## 代码改动计划

### 1. 配置层

文件：

- `vllm/config/observability.py`
- `vllm/engine/arg_utils.py`

改动：

- 在 `ObservabilityConfig` 中加入：
  - `enable_logging_step_timing_details`
  - `enable_graph_replay_timing`
- 在 CLI 参数中暴露对应开关
- 在 `create_engine_config` 中把参数写入 `observability_config`

### 2. 定义跨 worker 返回的 replay timing 结果

文件：

- `vllm/v1/outputs.py`

改动：

- 在 `ModelRunnerOutput` 中新增字段：
  - `graph_replay_timing_stats: GraphReplayTimingStats | None = None`

建议新增 dataclass：

- `GraphReplayTimingStats`
  - `reply_global_rank: int`
  - `dp_rank: int`
  - `tp_rank: int`
  - `dcp_rank: int`
  - `node_rank: int`
  - `replay_count: int`
  - `replay_gpu_ms: float`
  - `replay_wall_ms: float`
  - `runtime_mode: str`
  - `graph_impl: str`
  - `async_output_copy_wait_ms: float = 0.0` — 直接作为可选字段内嵌，不需要独立 dataclass

不再单独定义 `AsyncOutputTimingStats`。原因：它只有 `copy_wait_wall_ms` 一个字段，独立为 dataclass 增加了不必要的类型层级。直接在 `GraphReplayTimingStats` 中以可选字段承载即可。

同时去掉了原方案中的 `num_unpadded_tokens` 和 `num_padded_tokens`。原因：这两个值在 engine 侧可以直接从 `SchedulerOutput` 的 `compute_iteration_details()` 获得（方案第 10 节在组装日志时已经引用了它），无需从 worker 侧冗余回传。

注意：

- 不要把 timing 字段并入现有 `CUDAGraphStat`
- `CUDAGraphStat` 会进入聚合统计，加入浮点 timing 会破坏聚合维度

### 3. 在 Forward Context 中传递 replay timing 收集器

文件：

- `vllm/forward_context.py`

改动：

- 在 `ForwardContext` 中新增 typed 字段：
  - `graph_timing_context: GraphTimingContext | None = None`
- `GraphTimingContext` 负责累计：
  - replay 次数
  - replay 来源
  - CUDA event 对
  - replay 的 wall time
  - 当前 worker 的 rank 身份

建议字段：

- `replay_count`
- `graph_impls`
- `event_pairs`
- `replay_wall_time_ns`
- `reply_global_rank`
- `dp_rank`
- `tp_rank`
- `dcp_rank`
- `node_rank`

### 4. 在 CUDAGraphWrapper 中统计 replay 时间

文件：

- `vllm/compilation/cuda_graph.py`

改动位置：

- `CUDAGraphWrapper.__call__()` 的 replay 分支

统计方式：

1. `wall_start_ns = time.perf_counter_ns()`
2. `start_event.record()`
3. `entry.cudagraph.replay()`
4. `end_event.record()`
5. `wall_end_ns = time.perf_counter_ns()`
6. 将 event 对和 wall time 写入 `graph_timing_context`

要求：

- 只包 replay 分支
- capture 分支不纳入 runtime timing（首次 warmup 时 `entry.cudagraph is None`，走 capture 路径，`replay_count` 为 0 是正常行为，日志中无需特殊处理）
- 不做 `torch.cuda.synchronize()`

**CUDA event 复用**：高频 decode 下（128 seq × 数百 step），每次 replay 创建 2 个 `torch.cuda.Event` 会产生大量短生命周期对象。建议在 `GraphTimingContext` 中预分配一个小的 event pool（例如 4 对），循环复用。每次 resolve 后（在 `get_output()` 中）回收 event 对，避免持续累积。

### 5. 在 UBatchWrapper 中补 replay 时间统计

文件：

- `vllm/v1/worker/gpu_ubatch_wrapper.py`

改动位置：

- `num_tokens in self.cudagraphs and cudagraph_runtime_mode is CUDAGraphMode.FULL` 分支

统计方式与 `CUDAGraphWrapper` 一致：

- 记录 replay 的 event 对
- 记录 replay 的 wall time
- 标记 `graph_impl = "ubatch_wrapper"`

### 6. 在 GPUModelRunner 中创建、保存并最终 resolve replay timing

文件：

- `vllm/v1/worker/gpu_model_runner.py`

改动：

- 扩展 `ExecuteModelState`，把 `graph_timing_context` 一起存进去
- 在 `execute_model()` 进入 `set_forward_context(...)` 前创建 `graph_timing_context`
- `execute_model()` 返回 `None` 时，将其放入 `execute_model_state`
- `sample_tokens()` 构造 `ModelRunnerOutput` 时：
  - 非 async scheduling 路径：直接 resolve `graph_timing_context`
  - async scheduling 路径：先把未 resolve 的 context 交给 `AsyncGPUModelRunnerOutput`

建议做法：

- `AsyncGPUModelRunnerOutput` 新增字段：
  - `graph_timing_context: GraphTimingContext | None`
- 在 `get_output()` 中：
  1. 记录 `copy_wait_start_ns = time.perf_counter_ns()`
  2. `async_copy_ready_event.synchronize()`
  3. `copy_wait_wall_ms = (time.perf_counter_ns() - copy_wait_start_ns) / 1e6`
  4. resolve `graph_timing_context` 中的 event elapsed time
  5. 将 `copy_wait_wall_ms` 写入 `GraphReplayTimingStats.async_output_copy_wait_ms`
  6. 将结果回填到 `self._model_runner_output`

这样 replay timing 的 resolve 点落在”输出 ready”边界，不额外新增同步。

**生命周期注意**：`GraphTimingContext` 在 `set_forward_context()` context manager 退出后，`ForwardContext` 被清理，但 `GraphTimingContext` 作为 Python 对象只要有引用就不会被 GC。`execute_model()` 返回 `None` 时需要将其引用存入 `ExecuteModelState`，确保 `sample_tokens()` 仍能访问。建议在代码中加注释说明此引用转移。

### 7. 在 EngineCore 中引入 batch ticket

文件：

- `vllm/v1/engine/core.py`

建议新增 core 本地 dataclass，例如：

- `BatchTimingTicket`
  - `batch_id: int`
  - `scheduler_output: SchedulerOutput`
  - `exec_model_future: Future[Any]`
  - `result_future: Future[ModelRunnerOutput]`
  - `schedule_cpu_ns: int`
  - `grammar_cpu_ns: int`
  - `update_cpu_ns: int`
  - `grammar_deferred: bool`

将当前 batch queue 从：

```text
deque[(future, scheduler_output, exec_future)]
```

改为：

```text
deque[BatchTimingTicket]
```

同时新增：

- `self._batch_timing_id`
- `self._deferred_timing_ticket: BatchTimingTicket | None`

### 8. 只在 `step_with_batch_queue()` 上实现第一阶段

文件：

- `vllm/v1/engine/core.py`

原因：

- 这是你当前真实实验路径
- async scheduling 的关键复杂度都在这里
- 同步 `step()` 可在第二阶段复用同样 helper 补上

建议拆出 helper，避免主流程继续膨胀：

- `_emit_batch_timing_log(ticket, scheduler_output, model_output)` — 唯一的独立 helper，负责组装并输出结构化日志

`schedule` / `grammar` / `update` 三个阶段的计时逻辑相同（`perf_counter_ns` 前后包一行调用），不需要各自一个 helper。建议用 inline `perf_counter_ns` 差值直接写在调用点，例如：

```python
t0 = time.perf_counter_ns()
scheduler_output = self.scheduler.schedule()
ticket.schedule_cpu_ns = time.perf_counter_ns() - t0
```

如果需要复用，可以用一个轻量级 context manager：

```python
@contextmanager
def _timed_ns():
    t0 = time.perf_counter_ns()
    result = [0]
    yield result
    result[0] = time.perf_counter_ns() - t0
```

但不建议为三个单行调用各写一个专用 helper。

### 9. 处理 immediate grammar 与 deferred grammar 两种分支

文件：

- `vllm/v1/engine/core.py`

#### immediate grammar

当前分支：

- `not scheduler_output.pending_structured_output_tokens`

处理方式：

- schedule 后立即测 `grammar_cpu_ms`
- 立刻发起 `sample_tokens(non_block=True)`
- 把 ticket 入队

#### deferred grammar

当前分支：

- `scheduler_output.pending_structured_output_tokens`

处理方式：

- ticket 先只记录 `schedule_cpu_ms`
- 设置 `ticket.grammar_deferred = True`
- 暂存到 `self._deferred_timing_ticket`
- 等上一批结果出队并完成 `update_from_output()` 后，再测 `grammar_cpu_ms`
- 调用 `sample_tokens(non_block=True)` 并把 ticket 入队

注意：

- 这正是 async scheduling 下最容易统计错的路径
- `scheduler_total_cpu_ms` 仍然是三段时间求和，不要求三段时间连续

### 10. 出队时统一落日志

文件：

- `vllm/v1/engine/core.py`

时机：

- `future.result()` 返回
- `scheduler.update_from_output()` 完成之后

日志内容来自三部分：

1. ticket 中累计的 scheduler timing
2. `model_output.graph_replay_timing_stats`
3. `compute_iteration_details(scheduler_output)`

建议日志格式：

```text
ASYNC_STEP_TIMING batch_id=42 node_rank=1 reply_global_rank=24 dp_rank=3 tp_rank=0 dcp_rank=0 ctx_reqs=0 ctx_tokens=0 gen_reqs=128 gen_tokens=128 schedule_cpu_ms=0.31 grammar_cpu_ms=0.00 update_cpu_ms=0.43 scheduler_total_cpu_ms=0.74 grammar_deferred=0 graph_impl=cudagraph_wrapper replay_count=1 replay_gpu_ms=3.61 replay_wall_ms=3.69 async_output_copy_wait_ms=0.28 scheduler_us_per_gen_token=5.78 replay_us_per_gen_token=28.20
```

要求：

- 每条日志必须带出 rank 身份字段
- 后续做多配置对比时，优先比较同一语义的 `reply_global_rank/tp_rank/dcp_rank`

## 为什么这个方案更适合 async scheduling

- 不在热路径调用 `torch.cuda.synchronize()`
- replay timing 通过 CUDA event 采集，对 compute stream 干扰最小
- resolve 放在 `get_output()`，与现有 async output ready 同步点重合
- scheduler timing 绑定到 batch ticket，不会因为 queue 深度大于 1 而错配
- 能正确覆盖 deferred grammar 分支
- 能同时覆盖 `CUDAGraphWrapper` 与 `UBatchWrapper`
- 与 `32DP / 4DP8TP8DCP / 8DP4TP4DCP / 16DP2TP2DCP` 的单-rank 对比需求一致

## 第一阶段实施范围

第一阶段只覆盖你当前实验真正会走到的路径：

- `async_scheduling = True`
- `step_with_batch_queue()`
- decode batch
- 单一 `reply rank` 哨兵指标
- `CUDAGraphWrapper`
- `UBatchWrapper`
- `AsyncGPUModelRunnerOutput`

明确不作为第一阶段目标的内容：

- 同步 `step()` 路径
- Prometheus / metrics logger 聚合导出
- 一般化的 `strict`/`light` 双模式

## 第二阶段可选扩展

如果第一阶段结果稳定，再考虑补：

- 同步 `step()` 的同构实现
- 将 scheduler timing 并入 `SchedulerStats`
- 将 replay timing 导出到 Prometheus
- 额外记录 `future.result()` 的阻塞时间，便于解释 engine 侧 idle/wait

## 验证建议

验证时重点看以下几点：

1. 开启 async scheduling 后，吞吐不应因为 timing 打点明显下降
2. `graph_replay_count` 在 replay 路径上应大于 0（首批 warmup 时为 0 是正常的，走 capture 路径）
3. `UBatchWrapper` 路径下不应出现 replay 漏记
4. `grammar_deferred=1` 的 batch 其 `grammar_cpu_ms` 仍应正确出现
5. `replay_gpu_ms <= replay_wall_ms` 基本成立
6. `async_output_copy_wait_ms` 单独可解释 `future.result()` 与 `replay_gpu_ms` 的差距
7. 多配置实验中，日志必须能明确区分 `reply_global_rank/dp_rank/tp_rank/dcp_rank/node_rank`
8. CUDA event pool 在长时间运行后内存不应持续增长（验证 event 回收正确）

## 结论

对于 async scheduling，正确的设计不是在 `step()` 里强行同步，而是：

- worker 侧只记录 replay event
- output ready 时再 resolve replay timing
- engine 侧用 batch ticket 聚合 `schedule/grammar/update`
- 明确只输出单一 `reply rank` 的哨兵指标
- 最后在 batch 出队时统一落一条结构化日志

这套方案既能拿到你要的 `graph replay` 与 `scheduler` 开销，又不会主动破坏 async scheduling 的本来运行方式。

## 审阅修订记录

以下是基于代码审阅后对原方案的修订，目标是减少非必要路径、降低实现复杂度：

### 1. 配置开关从 3 个合并为 2 个

- 去掉了 `enable_async_output_timing`
- `async_output_copy_wait_ms` 的测量跟随 `enable_graph_replay_timing` 自动开启
- 原因：`get_output()` 中已有 `synchronize()`，额外两行 `perf_counter_ns()` 无可测量开销，不值得单独开关

### 2. 去掉了 `AsyncOutputTimingStats` 独立 dataclass

- `copy_wait_wall_ms` 改为 `GraphReplayTimingStats` 的内嵌可选字段 `async_output_copy_wait_ms`
- 原因：只有一个字段的 dataclass 增加不必要的类型层级

### 3. 去掉了 `GraphReplayTimingStats` 中的 `num_unpadded_tokens` 和 `num_padded_tokens`

- 这两个值在 engine 侧可从 `SchedulerOutput.compute_iteration_details()` 直接获得
- 无需从 worker 侧冗余回传

### 4. 四个 helper 方法简化为一个

- 去掉了 `_measure_schedule_with_timing()`、`_measure_grammar_with_timing()`、`_measure_update_with_timing()`
- 保留 `_emit_batch_timing_log()` 作为唯一 helper
- 三个阶段的计时改为 inline `perf_counter_ns` 差值
- 原因：三个调用的计时逻辑完全相同（包一行调用），单独封装不增加可读性

### 5. 新增 CUDA event pool 建议

- 高频 decode 下每次 replay 创建 2 个 event 对象会大量累积
- 建议在 `GraphTimingContext` 中预分配 event pool 循环复用

### 6. 新增 `GraphTimingContext` 生命周期注意事项

- `ForwardContext` context manager 退出后，`GraphTimingContext` 的引用需要显式转移到 `ExecuteModelState`
- 建议在实现时加注释说明此引用转移语义

### 7. 补充了 capture 分支 warmup 的说明

- 首批 warmup 时走 capture 路径，`replay_count=0` 是正常行为
- 验证建议中增加了对应条目
