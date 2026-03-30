# 多机 DP 场景下，为什么使用 Ray 作为后端就没法开启 async scheduling？

## 结论

这不是“多机 DP 天生不能配 async scheduling”，而是“当前 vLLM 里的 Ray executor 还没有把 async scheduling 这条链路实现完整，所以被显式禁用了”。

## 直接限制在哪里

vLLM 在配置阶段就把 async scheduling 限定为只支持以下 distributed executor backend：

- `mp`
- `uni`
- `external_launcher`

如果当前 executor 是 `ray`，就会直接报不支持。对应代码：

- `vllm/config/vllm.py:685`

另外，在 DP world size 大于 1 时，如果你显式使用了：

```bash
--data-parallel-backend=ray
```

vLLM 会默认把 distributed executor 也切到 `ray`。对应代码：

- `vllm/config/parallel.py:758`

所以现象上看起来就是：只要多机 DP 走了 Ray，`async scheduling` 就开不起来。

## 根本原因

`async scheduling` 依赖一种“结果先返回占位、真正输出稍后再取”的机制。核心对象是：

- `AsyncModelRunnerOutput`

这类对象内部不是普通的纯 CPU 数据，而是带着：

- CUDA stream / event
- 尚未完成的 GPU -> CPU non-blocking copy
- 仍然挂在设备侧的 tensor 引用

相关实现可以看：

- `vllm/v1/worker/gpu/async_utils.py:12`
- `vllm/v1/worker/gpu_model_runner.py:218`

也就是说，async scheduling 的关键不只是“异步调度”，还包括“异步回收输出”。

## 为什么 `mp` 能支持

在 `mp` 路径下，这个异步输出对象不需要跨进程框架边界做序列化。vLLM 可以：

1. 先把 `AsyncModelRunnerOutput` 保留在本地 worker 进程里
2. 用单独线程异步执行 `get_output()`
3. 等 GPU -> CPU copy 真正完成后，再把结果塞回响应队列

对应代码：

- `vllm/v1/executor/multiproc_executor.py:596`
- `vllm/v1/executor/multiproc_executor.py:898`

所以 `mp` 可以保住 async scheduling 依赖的“调度与结果回收重叠”这一点。

## 为什么 Ray 现在不行

Ray 这条路径里，worker 输出需要跨 actor / compiled DAG 边界传递。这个边界要求对象可序列化。

但 `AsyncModelRunnerOutput` 不满足这个条件。代码里已经直接写明：

> AsyncModelRunnerOutput holds CUDA events and cannot be pickled.

对应代码：

- `vllm/v1/executor/ray_utils.py:145`

因此 Ray 路径只能在 worker 侧立刻把它同步展开成普通输出，也就是直接调用：

```python
output = output.get_output()
```

这一步会把原本应该“延后完成”的异步输出回收，变成当前调用路径里的同步等待。

一旦这里同步化，async scheduling 最关键的收益就没了：

- 调度和输出回收不能再有效重叠
- Ray executor 不能承载当前 async scheduling 的返回语义

所以当前版本选择的是：直接禁用，而不是带着语义不完整的实现硬开。

## 测试也明确说明了这一点

测试里对 `ray + async_scheduling` 是直接跳过的：

- `tests/v1/distributed/test_async_llm_dp.py:88`

注释写得也很直接：

```python
# TODO(NickLucche) Re-enable when async scheduling is supported
```

## 最后一句话总结

本质上不是 “DP + Ray” 在理论上绝对不能做 async scheduling，而是：

**当前 vLLM 的 Ray executor 无法承载 async scheduling 所依赖的异步结果对象与返回语义，因此被实现层面显式禁用。**

## 实践建议

如果你现在想在多机 DP 场景下启用 async scheduling，现阶段应优先考虑：

- `mp`
- `external_launcher`

而不是 Ray。
