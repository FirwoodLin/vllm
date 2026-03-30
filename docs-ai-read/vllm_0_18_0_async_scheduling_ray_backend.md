# vLLM 0.18.0 为什么 async scheduling 不能使用 Ray 后端

Last updated: 2026-03-27
Repo basis: `git show v0.18.0:...` on local `/vllm` checkout

## 1. 一句话结论

`vLLM 0.18.0` 里，`async scheduling` 不能和 `ray` distributed executor backend 一起用，不是因为 Ray 在理论上绝对做不到，而是因为 `0.18.0` 的 Ray 执行链路承载不了 async scheduling 依赖的那种“异步输出对象”。

更准确地说：

1. 配置校验阶段就已经显式禁止 `ray + async_scheduling`
2. async scheduling 依赖先返回一个 `AsyncModelRunnerOutput` 占位对象
3. 这个对象内部带有 CUDA event、GPU tensor 引用、尚未完成的异步 D2H copy
4. Ray actor / compiled DAG 边界要求返回值可序列化
5. 所以 Ray worker 里必须提前 `get_output()`，把异步结果同步展开
6. 一旦这里提前同步，async scheduling 最关键的收益就没了，因此在实现层面被禁用

## 2. 直接限制在哪里

`v0.18.0` 在配置校验时只允许以下 distributed executor backend 开启 async scheduling：

- `mp`
- `uni`
- `external_launcher`

对应代码：

- `v0.18.0:vllm/config/vllm.py:714`

也就是说，只要最终的 `distributed_executor_backend` 是 `ray`，并且你显式打开了 `async_scheduling=True`，就会在启动前直接报错，而不是跑到执行阶段再失败。

## 3. 为什么选了 Ray DP 很容易撞上这个限制

如果开启了 data parallel，并且把 `data_parallel_backend` 设为 `ray`，vLLM 会把 distributed executor backend 默认切到 `ray`。

对应代码：

- `v0.18.0:vllm/config/parallel.py:782`
- `v0.18.0:vllm/config/parallel.py:785`

所以很多时候你即使没有手动传 `--distributed-executor-backend ray`，只要走的是 `data_parallel_backend=ray`，最后也会落到 Ray executor 路径上，然后触发上面的 async scheduling 限制。

## 4. async scheduling 真正依赖的是什么

这件事的关键不只是“调度线程更异步”，而是“输出结果延后回收”。

在 `v0.18.0` 中，GPU model runner 在开启 async scheduling 时，返回的是 `AsyncOutput` / `AsyncPoolingOutput`；只有关闭 async scheduling 时，才会立刻返回已经 materialize 完成的普通输出。

对应代码：

- `v0.18.0:vllm/v1/worker/gpu/model_runner.py:1159`
- `v0.18.0:vllm/v1/worker/gpu/model_runner.py:1160`
- `v0.18.0:vllm/v1/worker/gpu/model_runner.py:1201`
- `v0.18.0:vllm/v1/worker/gpu/model_runner.py:1202`

这些异步输出对象的统一抽象是 `AsyncModelRunnerOutput`：

- `v0.18.0:vllm/v1/outputs.py:258`
- `v0.18.0:vllm/v1/outputs.py:263`

源码对 `get_output()` 的说明很明确：这是一个阻塞调用，可能需要等待 device-to-host copy 完成。

## 5. 为什么 `AsyncModelRunnerOutput` 不是普通返回值

`AsyncOutput` 在创建时会保留 GPU tensor 引用，并记录异步 copy 完成事件；真正的等待发生在后续 `get_output()` 里。

对应代码：

- `v0.18.0:vllm/v1/worker/gpu/async_utils.py:22`
- `v0.18.0:vllm/v1/worker/gpu/async_utils.py:50`

所以这个对象内部不是普通的 CPU 侧结构，而是带着以下状态：

- CUDA stream / event 协调信息
- 尚未完成的 non-blocking GPU -> CPU copy
- 仍然存活的 GPU tensor 引用

也正因为这样，vLLM 才能把“上一步输出回收”和“下一步调度推进”重叠起来。

## 6. 为什么 `mp` 可以支持

`mp` 后端的关键优势是：它不需要把这个异步输出对象跨 Ray actor 之类的框架边界传出去。

在 `v0.18.0` 的 `MultiprocExecutor` 里，如果开启 async scheduling，会起一个专门的后台线程来处理异步输出回收。

对应代码：

- `v0.18.0:vllm/v1/executor/multiproc_executor.py:601`
- `v0.18.0:vllm/v1/executor/multiproc_executor.py:603`
- `v0.18.0:vllm/v1/executor/multiproc_executor.py:889`
- `v0.18.0:vllm/v1/executor/multiproc_executor.py:900`
- `v0.18.0:vllm/v1/executor/multiproc_executor.py:908`

这条链路可以概括成：

1. worker 先返回 `AsyncModelRunnerOutput`
2. worker 主执行循环不在当前调用点阻塞等待
3. 后台 copy 线程晚一点再调用 `get_output()`
4. 等异步 D2H copy 真正完成后，再把普通结果放回响应队列

所以 `mp` 可以保住 async scheduling 需要的“调度推进”和“输出回收”重叠。

## 7. 为什么 Ray 在 0.18.0 里不行

Ray 路径里，worker 输出要跨过 Ray actor / compiled DAG 边界。

相关代码可以看到：

- `v0.18.0:vllm/v1/executor/ray_executor.py:469`
- `v0.18.0:vllm/v1/executor/ray_executor.py:479`
- `v0.18.0:vllm/v1/executor/ray_executor.py:595`

问题在于，这个边界要求对象可序列化。因此 `v0.18.0` 的 Ray worker 在返回前，必须先把异步输出展开成普通输出。

对应代码：

- `v0.18.0:vllm/v1/executor/ray_utils.py:146`
- `v0.18.0:vllm/v1/executor/ray_utils.py:155`
- `v0.18.0:vllm/v1/executor/ray_utils.py:158`

源码注释写得非常直接：

- `AsyncModelRunnerOutput holds CUDA events and cannot be pickled`

对应位置：

- `v0.18.0:vllm/v1/executor/ray_utils.py:155`

这就是核心限制。因为一旦 Ray worker 在返回前必须先执行 `get_output()`，就会发生下面几件事：

- worker 线程当场等待 deferred copy 完成
- 返回值不再是异步占位对象，而是已经同步展开的普通结果
- scheduler 无法再像 `mp` 那样，把结果回收延后到本地后台线程去做

换句话说，Ray 虽然还能把“已经展开完成的普通结果”包成 Ray object ref / future，但那只是“远端执行结果的 future”，不是 async scheduling 需要的“异步输出对象 future”。真正关键的异步语义在 worker 侧就已经丢掉了。

## 8. 测试里也明确承认还不支持

`v0.18.0` 的分布式 async DP 测试里，对 `ray + async_scheduling` 是直接跳过的。

对应代码：

- `v0.18.0:tests/v1/distributed/test_async_llm_dp.py:89`

注释内容也很直白：等 async scheduling 支持后再重新启用。

## 9. 最终结论

本质上不是“Ray 和 async scheduling 天生互斥”，而是：

- async scheduling 依赖 `AsyncModelRunnerOutput`
- 这个对象带有 CUDA / 延迟 copy 的状态
- Ray executor 边界要求对象可序列化
- 所以 Ray 路径只能提前 `get_output()`
- 一旦提前 materialize，async scheduling 依赖的异步输出传递语义就没了
- 因此 `vLLM 0.18.0` 选择在实现层面直接禁用这组组合

## 10. 实践建议

如果你在 `vLLM 0.18.0` 里想开 `async scheduling`，应优先使用：

- `mp`
- `uni`
- `external_launcher`

如果你必须使用 `ray` backend，那么在这个版本里就应默认接受：

- `async_scheduling` 需要关闭
