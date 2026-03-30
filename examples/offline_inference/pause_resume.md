# vLLM Pause/Resume 功能说明

本文说明 `examples/offline_inference/pause_resume.py` 这个示例在演示什么，以及 vLLM 里的 `pause_generation()` / `resume_generation()` 实际暂停了什么。

## 结论

`Pause/Resume` 在 vLLM 里是“暂停生成调度”，不是暂停整个 Python 进程，也不是把模型实例销毁后再恢复。

它的主要用途是：

- 临时停止生成推进
- 让已有请求按指定策略处理
- 在不重启 engine 的情况下做权重更新或其他需要短暂停机的操作

对应接口注释里写的是：

- `Pause generation to allow model weight updates.`

相关实现位置：

- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/core/sched/interface.py`

## 这个示例演示了什么

`examples/offline_inference/pause_resume.py` 演示的是 `mode="keep"` 的行为：

1. 启动一个生成任务，持续接收输出 token。
2. 记录每个 token 到达的时间。
3. 当已经生成了几个 token 后，调用 `await engine.pause_generation(mode="keep")`。
4. 暂停一段时间。
5. 调用 `await engine.resume_generation()`。
6. 检查暂停前后两个 token 的时间间隔，确认中间确实出现了一段“空窗期”。

如果功能正常，那么恢复前不会继续生成新 token，恢复后会从原来的请求状态继续往下生成。

## 暂停的到底是什么

暂停的是调度器对请求的推进。

从调度器状态定义看，vLLM 内部有两种暂停状态：

- `PAUSED_NEW`：不再调度新请求，但已有运行中的请求可以继续按模式处理
- `PAUSED_ALL`：完全不调度任何请求

`mode="keep"` 使用的是 `PAUSED_ALL`。这意味着：

- 已经在跑的请求不会被丢弃
- 但也不会继续往前生成 token
- `resume_generation()` 之后再继续

所以它更像“冻结生成进度”，而不是“结束后重开”。

## 三种模式的区别

### `abort`

含义：

- 立刻中止当前 in-flight 请求

效果：

- 当前请求会结束
- 结束原因通常是 `abort`
- 新请求在恢复前也不会继续被调度

适合场景：

- 你需要尽快停机
- 不关心当前请求是否被打断

### `wait`

含义：

- 允许当前正在运行的请求自然完成
- 等 engine 排空后再进入暂停

效果：

- 不会粗暴打断当前请求
- 但暂停动作会等到这些请求都结束

适合场景：

- 你希望“先跑完手头这批，再暂停”

### `keep`

含义：

- 冻结当前请求
- 不继续生成，但也不丢弃请求状态
- 恢复后继续生成

效果：

- 能看到明显的 token 时间戳间隔
- 适合做“中途暂停，稍后继续”

适合场景：

- 热更新权重
- 临时同步某些状态
- 希望保留请求上下文并继续生成

## 示例执行时间线

示例脚本里有两个并发任务。

### 1. 生成任务

生成任务调用：

```python
async for output in engine.generate(...):
    ...
```

它会不断接收增量输出，并记录每次 token 到达时间。

### 2. 控制任务

控制任务会：

```python
await engine.pause_generation(mode="keep")
await asyncio.sleep(PAUSE_DURATION)
await engine.resume_generation()
```

逻辑上就是：

- 等前面先生成几个 token
- 调用 pause
- 故意睡眠 3 秒
- 再调用 resume

### 3. 如何验证真的暂停了

脚本最后会计算：

```python
pause_gap = token_times[pause_token_idx][1] - token_times[pause_token_idx - 1][1]
```

如果这个间隔接近 `PAUSE_DURATION`，说明这段时间里生成确实停住了，而不是后台还在偷偷继续跑。

## 一个直观理解

可以把它理解成：

- `abort`：把当前活都取消
- `wait`：把当前活干完再停
- `keep`：把当前活先按下暂停键，之后接着干

其中 `pause_resume.py` 演示的是第三种，也就是“按下暂停键再继续”。

## 代码层面的关键点

### 1. 对外接口

`AsyncLLM.pause_generation()` 的注释说明了用途和三种模式：

- `abort`: 立即中止 in-flight 请求
- `wait`: 等 in-flight 请求完成
- `keep`: 冻结请求，等待 `resume_generation()`

### 2. 调度器状态切换

在 engine core 里：

- `keep` 会把调度器切到 `PAUSED_ALL`
- `abort` 和 `wait` 会切到 `PAUSED_NEW`

这决定了暂停时还能不能继续推进已有请求。

### 3. cache 的行为

`pause_generation()` 默认参数里：

```python
clear_cache=True
```

也就是说，默认暂停时会清理 KV cache / prefix cache。

如果你的目标是更快恢复，并且不希望清 cache，可以显式传：

```python
await engine.pause_generation(mode="keep", clear_cache=False)
```

这对“暂停后很快恢复”的场景更有意义。

## 适用场景

这个功能比较适合：

- 在线服务需要短暂冻结生成
- RLHF 或权重热更新流程
- 多个请求正在跑，但你希望暂时停住而不是直接重启 engine

如果只是想彻底停服务，或者直接重建 engine，那就不是这个接口的主要用途。

## Online Serving 时能不能这么用

可以，但有前提。

在 online serving 场景下，vLLM 确实提供了对应的 HTTP 控制接口：

- `POST /pause`
- `POST /resume`
- `GET /is_paused`

也就是说，服务端运行中时，你可以通过 HTTP 把生成暂停，再恢复。

### 需要满足的前提

这些接口不是默认暴露给普通线上服务的通用 OpenAI API。

要使用它们，通常需要在启动 `vllm serve` 时开启：

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve facebook/opt-125m --enforce-eager
```

如果没有开启 `VLLM_SERVER_DEV_MODE=1`，这些开发接口不会被挂载。

另外，这组接口是在根路径下：

- `/pause`
- `/resume`
- `/is_paused`

不是 OpenAI 兼容接口下面的：

- `/v1/...`

### 最小调用示例

```bash
curl -X POST 'http://localhost:8000/pause?mode=keep'
curl -X POST 'http://localhost:8000/resume'
curl 'http://localhost:8000/is_paused'
```

含义和离线 API 一样：

- `mode=abort`：直接中止当前请求
- `mode=wait`：等当前请求完成后进入暂停
- `mode=keep`：冻结当前请求，恢复后继续生成

### 对在线流式生成的影响

如果客户端正在流式接收 token：

- 调用 `pause?mode=keep` 后，token 流会暂时停住
- 暂停期间不会继续产生新 token
- 调用 `/resume` 后，生成会继续

这和 `examples/offline_inference/pause_resume.py` 的行为是同一套语义，只是控制方式从进程内 API 变成了 HTTP API。

### Data Parallel 场景也支持

仓库里有专门的 online serving 示例：

- `examples/online_serving/data_parallel_pause_resume.py`

这个示例说明在 Data Parallel 场景下，也可以通过 HTTP 触发 pause/resume，并且 pause 会在多个 DP rank 之间同步。

### 生产环境要谨慎

这组接口属于 development endpoints。

换句话说：

- 能用于实验、联调、权重热更新验证
- 但不建议直接裸暴露在生产环境

原因很简单：

- 需要开启 `VLLM_SERVER_DEV_MODE=1`
- 代码里对这类接口有明确的安全警告
- 它们更像内部控制面，而不是面向外部租户的稳定公开 API

如果你要在生产环境做类似能力，通常应该在外层自己包一层受控的管理接口，而不是直接把 dev endpoint 对外暴露。

## 一句话总结

`examples/offline_inference/pause_resume.py` 演示的是：

**vLLM 可以在生成过程中把请求“冻结住”，暂停期间不再产生新 token，等调用 `resume_generation()` 后再从原状态继续生成。**
