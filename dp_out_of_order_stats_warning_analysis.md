# `Received stats for out-of-order step` 告警分析

## 现象

告警示例：

```text
WARNING 04-03 20:30:18 [coordinator.py:347] Received stats for out-of-order step (0, 3357) from engine 10 (expected > (0, 3358))
```

这条日志来自 `vllm/v1/engine/coordinator.py`，由 DP coordinator 在接收各个 data-parallel engine 上报的 `scheduler_stats` 时打印。

从日志字面看，含义是：

- coordinator 当前认为最近一次看到的最新统计步是 `(wave=0, step=3358)`；
- 但此时又收到了 `engine 10` 发来的 `(wave=0, step=3357)`；
- 因为 `3357 < 3358`，于是被当成了 “out-of-order step”。

## 先给结论

这条告警更可能不是“同一个 engine 的消息真的乱序了”，而是 coordinator 当前实现把“所有 engine 的局部 step”错误地当成了“一个全局单调递增的 step”来比较，因此把跨 engine 的正常交错上报误判成了乱序。

换句话说，这通常更像是：

- `engine A` 先上报了 step `3358`
- `engine 10` 随后上报了它自己的 step `3357`

而不是：

- `engine 10` 明明先发了 `3358`
- coordinator 却后收到了它更早的 `3357`

前者在当前架构下是完全可能的，后者反而不太像常态。

## 相关代码路径

### 1. 告警触发点

`vllm/v1/engine/coordinator.py`

coordinator 用两个全局变量记录最近见到的最大统计点：

- `last_stats_wave`
- `last_stats_step`

当收到任意 engine 的 `scheduler_stats` 后，它做如下判断：

- 如果 `(stats_wave, stats_step)` 比当前记录的全局最大值更大，则接受为“新步”；
- 如果不相等但又更小，就打印 warning。

问题在于：这个比较是 **全局一份**，不是 **每个 engine 各自一份**。

### 2. `step_counter` 的来源

`vllm/v1/engine/core.py`

每个 DP engine 内部都维护自己的：

- `self.step_counter`
- `self.current_wave`

其中：

- `step_counter` 在 `_has_global_unfinished_reqs()` 中每轮循环自增一次；
- 每个 wave 结束时，`current_wave += 1`，并且 `step_counter = 0`；
- 只有当本 engine 的请求计数变化时，才会调用 `_maybe_publish_request_counts()` 上报一次 `SchedulerStats(step_counter=..., current_wave=...)`。

这意味着：

- `step_counter` 是 **engine 本地时钟**；
- 它不是由 coordinator 统一分配的全局序号；
- 不同 engine 的消息到达 coordinator 时，天然可能交错。

### 3. 消息通路

`vllm/v1/engine/core.py` 中每个 engine 通过各自的 `PUSH` socket 向 coordinator 的单个 `PULL` socket 发消息。

这带来两个重要性质：

- **同一个 engine 内部** 的消息顺序通常是 FIFO 的；
- **不同 engine 之间** 的消息到达顺序不保证按 step 对齐。

因此 coordinator 如果拿一个全局 `(wave, step)` 去约束所有 engine，就很容易把“跨 engine 正常交错”当成“乱序”。

## 为什么这个 warning 大概率是“误报”

### 原因 1：coordinator 用的是全局最大 step，而不是每个 engine 的最近 step

当前逻辑本质上等价于：

1. 任意 engine 只要先上报了一个更大的 `(wave, step)`，就更新全局 `last_stats_*`
2. 之后另一个 engine 如果再上报一个比这个全局最大值小的 step，就打 warning

但这并不能证明第二个 engine 的消息真的乱序，只能证明：

- 它的本地进度或上报时机落后于“某个别的 engine 的最近一次上报”。

这两者不是一回事。

### 原因 2：`scheduler_stats` 不是每一步都发，只在“请求计数变化”时发

engine 只有在 `running/waiting` 数量发生变化时才上报。

这会导致：

- `engine A` 可能在 step `3358` 因为请求完成而上报；
- `engine B` 可能一直没变化，直到 step `3357` 才因为本地队列变化补发一次；
- coordinator 先看到 `A:3358`，再看到 `B:3357`，就会报警。

这种场景完全不要求任何网络乱序，也不要求任何线程竞态。

### 原因 3：不同 engine 虽然大体同步，但不保证“统计事件”严格同拍

DP engine 在运行时会做 dummy pass，并且每 32 步才通过 `_has_global_unfinished_reqs()` 做一次 unfinished 同步检查。它们大体上会接近同步，但并不意味着：

- 请求进入等待队列
- 请求从 waiting 进入 running
- 请求完成并从 running 移除

这些“计数变化事件”会在所有 engine 上严格发生在同一个本地 step 上。

所以像日志里这种只差 `1` 的情况：

```text
got (0, 3357), expected > (0, 3358)
```

更像是正常抖动，而不是严重异常。

## 补充：`dummy pass`、`32` 步检查，以及 unfinished 信息流向

上面提到：

> DP engine 在运行时会做 dummy pass，并且每 32 步才通过 `_has_global_unfinished_reqs()` 做一次 unfinished 同步检查。

这里需要把三个点拆开说清楚。

### 1. `dummy pass` 是什么

它不是“空转一下”或者简单 sleep。

在 `vllm/v1/engine/core.py` 中，如果这一轮没有真正执行到可运行请求，但 engine 仍处在 running 状态，就会调用：

- `self.execute_dummy_batch()`

往下追代码：

- `vllm/v1/engine/core.py` 的 `execute_dummy_batch()` 会下发到 executor；
- `vllm/v1/executor/multiproc_executor.py` 通过 `collective_rpc("execute_dummy_batch")` 广播给 worker；
- `vllm/v1/worker/gpu_worker.py` 最终执行 `self.model_runner._dummy_run(1, uniform_decode=True)`。

也就是说，DP 运行时的 dummy pass 本质上是一个 **1 token 的 synthetic decode forward**。

### 2. `dummy pass` 实际会跑什么

默认情况下，`VLLM_USE_V2_MODEL_RUNNER=0`，因此 GPU worker 走的是 `vllm/v1/worker/gpu_model_runner.py` 这套实现。

如果显式开启了 V2 runner，那么会走 `vllm/v1/worker/gpu/model_runner.py` 的 `_dummy_run()`；那条路径默认 `skip_attn=True`，因此实现会更轻一些。下面这段分析以默认的 V1 runner 为主。

从这段 `_dummy_run()` 看，dummy pass 至少会做这些事情：

- 构造一个假的 batch 描述，当前调用是 `num_tokens=1, uniform_decode=True`；
- 根据 batch 形态决定执行模式、padding、microbatch/ubatch 切分；
- 准备 slot mappings 等运行时元数据；
- 在某些模式下构造 attention metadata；
- 设置 forward context；
- 真正调用一次 `self.model(...)`；
- 如果开了 speculative decoding，还可能调用 `drafter.dummy_run(...)`；
- 在 DP 场景下，通常还会执行 `eplb_step(is_dummy=True)`，确保 EP load balancing/rearrangement 的同步节奏不被打乱。

所以它不是纯控制流，也不是只做一个布尔判断；它会真的进入模型执行路径，只是喂的是 synthetic 输入。

### 3. `dummy pass` “包含多少算子”

源码里并没有一个固定的“dummy pass 算子数”。

更准确地说，dummy pass 的算子数量是 **动态决定的**，取决于：

- 模型本身有多少层；
- 当前 attention backend 是什么；
- 是否启用了 TP / PP / EP；
- 是否启用了 speculative decoding；
- 是否启用了 LoRA、多模态输入、prompt embeds；
- 当前 cudagraph runtime mode 是 `NONE / PIECEWISE / FULL` 中哪一种。

因此不能把它理解成“固定 N 个算子”的轻量心跳包。

从代码能确定的是：

- 它至少会走一次 model runner 的前向主路径；
- 在某些配置下会附带 attention metadata 构造、drafter dummy run、EPLB 同步等额外工作；
- 但它通常仍然比真实请求处理更轻，因为输入是假的、token 数极小，而且不少路径会走简化分支。

### 4. 为什么是“每 32 步检查一次”

源码里唯一直接说明是这句注释：

```python
# Optimization - only perform finish-sync all-reduce every 32 steps.
```

也就是说，`32` 在当前代码中是一个 **硬编码的优化阈值**。

从代码行为看，最合理的解释是：

- `_has_global_unfinished_reqs()` 的核心动作是对 DP group 做一次 `all_reduce(MAX)`；
- 如果每一步都做跨 rank 同步，分布式通信开销会更高；
- 改成每 32 步检查一次，可以把 finish-sync 的通信成本摊薄。

这部分是根据源码行为做的推断，不是源码中已有的设计文档原话。

### 5. 每 32 步检查一次的代价是什么

代价是：**全局“已经没有 unfinished request”这件事，不会被立刻发现。**

具体说：

- 每轮循环 `step_counter += 1`；
- 只有在 `step_counter % 32 == 0` 时，才真的调用 `ParallelConfig.has_unfinished_dp(...)`；
- 其余 31 步直接返回 `True`，也就是“继续认为系统还在 running”。

这意味着如果所有 DP rank 在第 `k` 步之后其实都已经没有 unfinished requests 了，那么 engine 不会立刻停下来，而是可能继续跑若干轮 dummy/空转执行，直到下一个 32 步边界才确认 wave 结束。

因此这个设计换来的是：

- 好处：减少频繁的 DP collectives；
- 代价：wave completion 的检测存在最多 31 步的延迟。

### 6. unfinished 信息到底发到哪去

这里容易误解。`local_unfinished_reqs` 这个布尔值 **不是每一步都直接发给 coordinator**。

实际路径是：

1. 每个 DP engine 先在本地计算 `local_unfinished_reqs = self.scheduler.has_unfinished_requests()`。
2. 到了检查点时，`ParallelConfig.has_unfinished_dp(dp_group, local_unfinished)` 会把这个布尔值包装成 CPU `int32` tensor。
3. 然后在 `dp_group` 上做一次 `torch.distributed.all_reduce(..., op=ReduceOp.MAX)`。
4. 所有 DP rank 都拿到同一个聚合结果 `aggregated_has_unfinished`。
5. 这个聚合结果只在 engine 本地用于更新 `self.engines_running`。

所以 raw unfinished 布尔值的“第一站”是：

- **DP ranks 之间的 `all_reduce(MAX)`**

而不是 coordinator。

### 7. coordinator 真正收到的是什么

当聚合结果变成 `False` 时，engine 才会对外发更高层的状态变化消息。

在有 coordinator 的场景下：

- 只有 `dp_rank == 0` 会发送 `EngineCoreOutputs(wave_complete=self.current_wave)`；
- 这个消息通过 engine 的输出队列发给 coordinator；
- coordinator 收到后，把全局状态推进到下一 wave，并向 front-end 发布新的 `(current_wave, engines_running)`。

所以 coordinator 看到的不是“每一步 unfinished=True/False 的原始流”，而是：

- 请求计数统计 `scheduler_stats`
- wave 开始/结束的状态事件，例如 `wave_complete`

换句话说，unfinished 信息的传播链路是：

`local_unfinished_reqs`
-> DP 组内 `all_reduce(MAX)`
-> 每个 engine 本地得到 `aggregated_has_unfinished`
-> 在 wave 结束时由 rank 0 生成 `wave_complete`
-> coordinator 再把 `(current_wave, engines_running)` 广播给 front-end

### 8. 这和 out-of-order warning 有什么关系

这段补充和本 warning 的关系在于：

- engine 在“没有真实请求可跑但系统仍被认为 running”时，会继续执行 dummy pass；
- 而全局 finished 状态只有每 32 步才同步一次；
- 因此不同 engine 的统计上报更容易出现轻微错位；
- coordinator 又用全局最大 `(wave, step)` 去比较所有 engine；
- 于是就更容易把这种正常错位误判成 out-of-order。

所以，这里并不是 dummy pass 本身有问题，而是：

- dummy pass + 低频 finish-sync 让局部 step 的轻微错位更容易出现；
- coordinator 当前的全局比较逻辑又把这种错位放大成 warning。

## 还有哪些场景也可能触发

除了正常交错之外，下列情况也可能触发同一条 warning：

### 1. wave 切换边界

每个 wave 结束后，engine 会：

- 递增 `current_wave`
- 把 `step_counter` 重置为 `0`

如果 coordinator 已经先看到了某个 engine 的新 wave 统计，而另一个 engine 的旧 wave 统计稍后才到，也会命中这条 warning。

### 2. 某个 engine 暂时落后

如果某个 engine 因为本地调度、队列处理、CPU 抢占、ZMQ 调度时机等因素慢半拍，它的统计上报比其他 engine 晚，也会触发这个告警。

### 3. 真正的异常情况

虽然概率更低，但也不能完全排除真正异常，例如：

- engine 重启后带着较旧状态重新上报；
- 极端情况下旧消息在通道中积压，直到 coordinator 已前进后才被消费；
- 某处状态机在 wave/step 更新时存在边界 bug。

不过从当前代码结构看，这些并不是第一怀疑对象。尤其是“同一 engine 内消息真正反序”并不符合这条路径的常见行为。

## 会造成什么后果

### 1. 对生成功能正确性的影响通常很小

这批 `scheduler_stats` 的主要用途是 **内部 DP load balancing**，也就是前端 API server 选择把新请求送到哪个 engine。

它们不是模型前向计算的正确性输入，也不是 request 输出内容的真值来源。因此仅凭这条 warning，本身通常不会直接导致：

- token 生成错误；
- 请求结果错乱；
- MoE 计算逻辑错误；
- wave 协调立刻失效。

也就是说，这更像是一个 **负载观测/调度质量问题**，而不是 **推理正确性问题**。

### 2. 会让负载均衡看到“混合代际”的统计快照

这是当前实现更实际的风险。

coordinator 在看到“更大 step 的统计”后，会尝试保留一份 `last_step_counts` 作为上一个 step 的快照；但如果后面又收到较旧 step 的统计，它仍然会把对应 engine 的 `request_counts` 写回当前状态。

结果可能是：

- 某些 engine 的计数来自 step `3358`
- 某些 engine 的计数来自 step `3357`

前端看到的不是一个严格同一步的全局快照，而是一个混合快照。

后果通常是：

- 负载均衡偶尔选择了“并非当前最空闲”的 engine；
- 某些 engine 被短时间高估或低估负载；
- 请求分配不够平滑，局部抖动增加。

### 3. 日志噪声会掩盖真正问题

如果这是误报型 warning，频繁出现会有两个副作用：

- 运维侧容易把它当成严重乱序故障；
- 真正有价值的分布式异常日志更容易被淹没。

因此这条 warning 本身也有“误导排障”的成本。

## 如何判断当前这条日志是否严重

如果你看到的是下面这种模式，通常不算严重：

- 大多只差 `1` 到几步；
- 只发生在同一个 wave 内；
- 服务吞吐和延迟没有明显恶化；
- 没有伴随 engine 重启、请求卡死、wave 卡住等问题。

如果出现下面这些迹象，就要提高警惕：

- step 回退幅度很大；
- 经常跨 wave 回退；
- 总是集中在固定某个 engine；
- 同时伴随某个 engine 不再接单、吞吐明显倾斜、请求堆积；
- 同时出现 reconfiguration / elastic scaling / engine restart 相关日志。

## 更合理的修复思路

### 方案 1：按 engine 维度跟踪最近 step

最合理的改法是把：

- `last_stats_wave`
- `last_stats_step`

改成按 `engine_index` 存储，例如：

- `last_stats_by_engine[eng_index] = (wave, step)`

然后只在 **同一个 engine 的 step 回退** 时才打印 warning。

这能把“跨 engine 正常交错”与“单 engine 真异常”区分开。

### 方案 2：收到旧 step 时不要覆盖当前更近的 engine 视图

即使保留告警，coordinator 也最好避免让“较旧 step 的统计”覆盖当前状态。否则会把当前全局计数回写成更旧的视图，影响 LB 精度。

更稳妥的策略是：

- 如果该 engine 的统计相对它自身是旧的，直接丢弃；
- 只有相对该 engine 自身单调前进时，才更新 `request_counts`。

### 方案 3：降低日志级别或缩窄触发条件

如果短期不改状态跟踪逻辑，至少可以考虑：

- 把它降成 `debug`
- 或只在回退幅度超过阈值时打印
- 或只在同一 engine 连续回退时打印

这样能减少误报噪声。

## 针对本次日志的具体判断

对这条日志：

```text
Received stats for out-of-order step (0, 3357) from engine 10 (expected > (0, 3358))
```

更可能的解释是：

- coordinator 刚刚先收到了别的 engine 的 `(0, 3358)`；
- 随后收到了 `engine 10` 的 `(0, 3357)`；
- 由于当前实现使用“全局最新 step”做比较，所以把它判成 out-of-order。

它最可能带来的后果是：

- coordinator 暂时拿到了一个不是完全同一步的负载视图；
- 前端短时间可能做出稍微不够理想的 engine 选择；
- 但通常不会直接影响生成结果正确性。

## 最终结论

这条 warning 的根因，大概率是 **DP coordinator 的统计顺序判断逻辑过强，错误地把跨 engine 的局部 step 交错当成了全局乱序**。

它的主要后果通常是 **内部负载均衡统计不够精确** 和 **日志噪声增加**，而不是直接破坏推理正确性。

如果后续要修，优先级最高的方向是：

1. 按 engine 记录最近收到的 `(wave, step)`；
2. 只对同一 engine 的回退做告警；
3. 对旧统计直接丢弃，不要覆盖当前更近的状态。
