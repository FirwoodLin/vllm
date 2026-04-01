# DecodeBenchConnector TTFT 分阶段打点设计

## 1. 目标

在 `DecodeBenchConnector` 场景下，为 TTFT 增加一套类似 `ASYNC_STEP_TIMING` 的细粒度阶段打点，输出一行可直接 grep/聚合的日志，用于定位性能热点，尤其是以下链路中的瓶颈：

1. API Server 预处理
2. 前端到 EngineCore 的 IPC
3. EngineCore 请求预处理
4. 排队等待调度
5. Scheduler 首次调度
6. GPU Model Runner 的首次 batch 执行
7. Sampling
8. EngineCore 输出回传
9. OutputProcessor 和 SSE 首个文本 chunk 发出

该设计的核心目标不是替代现有 TTFT 指标，而是在保持现有指标兼容的前提下，把服务端内部 TTFT 拆成可解释的阶段分布。

## 2. 背景

当前 vLLM 已有两类相关能力：

1. 现有 request 级指标
   - `arrival_time`
   - `QUEUED`
   - `SCHEDULED`
   - `first_token_ts`
   - `first_token_latency`
   - `queued_time`
   - `prefill_time`

2. 现有 batch 级日志
   - `ASYNC_STEP_TIMING`
   - `schedule_cpu_ms`
   - `grammar_cpu_ms`
   - `update_cpu_ms`
   - `scheduler_total_cpu_ms`
   - `replay_gpu_ms`
   - `replay_wall_ms`
   - `async_output_copy_wait_ms`

这些信息足够回答整体 TTFT 高不高，但还不足以回答下面这些问题：

1. `DecodeBenchConnector.start_fill_kv()` 到底占了多少时间。
2. TTFT 里 API 侧 tokenization 占了多少。
3. ZMQ + msgpack 的进出方向各占多少。
4. EngineCore preprocess 和 scheduler queue wait 各占多少。
5. Chat SSE 的首包是不是被“空 role chunk”提前触发，导致客户端 TTFT 偏小。

因此需要新增一套 request-scoped 的 TTFT 分阶段日志。

## 3. 关注的阶段定义

用户给出的 TTFT 拆分如下：

```text
T0 (客户端 time.perf_counter())
 │
 ├─ ① HTTP POST 发送
 ├─ ② API Server 预处理
 ├─ ③ ZMQ IPC 传输 + msgpack 反序列化
 ├─ ④ EngineCore 预处理
 ├─ ⑤ 排队等待调度
 ├─ ⑥ Scheduler 调度
 ├─ ⑦ GPU Model Runner
 ├─ ⑧ Sampling
 ├─ ⑨ 结果回传
 └─ ⑩ HTTP 响应传输回客户端

T1 (客户端收到第一个 SSE chunk with text)
```

这份设计中：

1. 服务端精确负责 ② 到 ⑨。
2. 客户端 benchmark 负责 ① 和 ⑩。
3. 两侧通过 `request_id` 关联。

## 4. 总体设计

建议采用“两层打点”。

### 4.1 第一层：服务端 `TTFT_TIMING`

新增一条 request-scoped 日志，在请求第一次产出文本 token 对应的 SSE chunk 即将发送时打印，例如：

```text
TTFT_TIMING request_id=... connector=DecodeBenchConnector batch_id=1520
api_preprocess_ms=7.31
ipc_in_decode_ms=1.44
engine_preprocess_ms=2.03
queue_wait_ms=18.52
first_batch_schedule_cpu_ms=0.41
first_batch_update_states_ms=0.62
first_batch_prepare_inputs_ms=1.87
first_batch_load_kv_ms=34.91
first_batch_forward_ms=12.54
first_batch_sample_ms=1.73
async_output_copy_wait_ms=0.28
ipc_out_ms=1.16
output_processor_ms=0.72
first_text_sse_emit_ms=0.35
server_ttft_ms=81.69
```

其中：

1. `server_ttft_ms` 表示服务端内部 TTFText，从服务端开始处理请求到服务端发出第一个“带文本”的 SSE chunk。
2. `first_batch_*` 明确表示这些时间来自请求第一次出 token 所在的 batch，不是假装成 request 独占。

### 4.2 第二层：benchmark 客户端补齐边界时间

在 benchmark 客户端记录：

1. `client_send_start_ns`
2. `client_send_done_ns`
3. `client_first_text_chunk_ns`

从而得到：

1. `http_post_send_ms`
2. `http_response_transfer_ms`
3. `client_ttft_ms`

最终可以同时看：

1. `client_ttft_ms`
2. `server_ttft_ms`
3. `client_ttft_ms - server_ttft_ms`

第三项可以作为剩余网络和前端调度开销的 residual。

## 5. 为什么不能只靠现有 request events

当前 `Request.events` 只承载：

1. `QUEUED`
2. `SCHEDULED`
3. `PREEMPTED`

这能覆盖 queue wait，但不能覆盖：

1. API 预处理
2. ZMQ in/out
3. EngineCore preprocess
4. GPU 侧 `load_kv`
5. OutputProcessor 到 SSE 的尾段

所以需要新增 request-scoped timing payload，而不是继续往 `EngineCoreEventType` 塞一堆阶段事件。

原因如下：

1. `EngineCoreEvent` 目前只存一个 `type + timestamp`，不适合存阶段细分。
2. 多数新增阶段是“持续时间”而不是“单点事件”。
3. 一部分数据来自不同进程和不同层次，直接拼 event 序列会让消费方复杂很多。

## 6. 数据模型建议

建议新增两类 timing 数据。

### 6.1 `RequestTTFTTrace`

request-scoped，负责 ②、③、④、⑤、⑨ 的 request 独占阶段。

建议字段：

```text
request_id
api_preprocess_ns
ipc_in_decode_ns
engine_preprocess_ns
queue_wait_ns
ipc_out_ns
output_processor_ns
first_text_sse_emit_ns
first_batch_id
```

挂载路径建议：

1. `EngineCoreRequest`
2. `Request`
3. `RequestStateStats` 或其相邻结构
4. `RequestOutput.metrics` 可见

### 6.2 `FirstBatchTiming`

batch-scoped，但在请求第一次产出 token 时拷贝到对应请求。

建议字段：

```text
batch_id
schedule_cpu_ns
update_states_ns
prepare_inputs_ns
load_kv_ns
forward_ns
sample_ns
async_output_copy_wait_ns
replay_gpu_ms
replay_wall_ms
```

挂载路径建议：

1. scheduler/core 侧用 `BatchTimingTicket`
2. worker/model runner 侧通过 `ModelRunnerOutput` 回传
3. 请求第一次出 token 时写回 request trace

## 7. 各阶段埋点挂点

下面按照阶段给出建议挂点。

### 7.1 ② API Server 预处理

目标覆盖：

1. 请求解析/验证
2. prompt tokenization
3. `SamplingParams` 构造
4. `EngineCoreRequest` 构造

建议挂点：

1. `vllm/v1/engine/async_llm.py`
   - `AsyncLLM.add_request()`
2. `vllm/v1/engine/input_processor.py`
   - `InputProcessor.process_inputs()`

建议做法：

1. 在 `AsyncLLM.add_request()` 进入时开始计时。
2. 在 `process_inputs()` 返回 `EngineCoreRequest` 后结束计时。
3. 结果写入 `EngineCoreRequest.ttft_trace.api_preprocess_ns`。

说明：

1. 这里测到的是服务端前端视角的“真实 API preprocess”。
2. 对于大 prompt，这一段可能非常明显，尤其是 tokenization。

### 7.2 ③ ZMQ IPC 传输 + msgpack 反序列化

目标覆盖：

1. 请求从前端进入 EngineCore 后，到完成 decode 的时间

建议挂点：

1. `vllm/v1/engine/core.py`
   - `EngineCoreProc.process_input_sockets()`

建议做法：

1. 在 `recv_multipart(copy=False)` 返回后记录 `t0`。
2. 在 `add_request_decoder.decode(data_frames)` 结束后记录 `t1`。
3. `ipc_in_decode_ns = t1 - t0`。

说明：

1. 这不是纯网络时间，而是“IPC + decode”的合并值。
2. 先接受这个定义，因为 ZMQ 真实线内传输时间很难单独准确抽出来。

### 7.3 ④ EngineCore 预处理

目标覆盖：

1. 构建内部 `Request`
2. block hash 计算
3. grammar init
4. 入等待队列前的处理

建议挂点：

1. `vllm/v1/engine/core.py`
   - `preprocess_add_request()`

建议做法：

1. `preprocess_add_request()` 入口前后打点。
2. 结果写入 request trace 的 `engine_preprocess_ns`。

说明：

1. 这里正好覆盖 `Request.from_engine_core_request()` 和 request 初始化逻辑。
2. 对 prefix cache 场景，这一段可能包含明显的 block hash 成本。

### 7.4 ⑤ 排队等待调度

目标覆盖：

1. request 入等待队列后，到首次被 `schedule()` 选中之间的时间

建议做法：

直接复用现有 request event：

1. `QUEUED`
2. `SCHEDULED`

即：

```text
queue_wait_ns = scheduled_ts - queued_ts
```

优点：

1. 不需要新增埋点。
2. 语义已经稳定。

### 7.5 ⑥ Scheduler 调度

目标覆盖：

1. request 首次出 token 所在 batch 的 scheduler CPU 时间

建议挂点：

1. `vllm/v1/engine/core.py`
   - `step_with_batch_queue()`
   - 现有 `BatchTimingTicket.schedule_cpu_ns`

建议做法：

1. 不再新增重复计时。
2. 在请求第一次出 token 时，记录其对应 `batch_id`。
3. 把该 batch 的 `schedule_cpu_ns` 作为 `first_batch_schedule_cpu_ns`。

注意：

1. 这是 batch-shared 时间，不是 request-exclusive。
2. 字段必须命名成 `first_batch_schedule_cpu_ms`，避免误导。

### 7.6 ⑦ GPU Model Runner

目标覆盖：

1. `_update_states()`
2. `_prepare_inputs()`
3. `connector.start_load_kv()`
4. `_model_forward()`

建议挂点：

1. `vllm/v1/worker/gpu_model_runner.py`
   - `execute_model()`
   - `_update_states()`
   - `_prepare_inputs()`
   - `_model_forward()`
2. `vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py`
   - `DecodeBenchConnectorWorker.start_fill_kv()`

建议拆分：

1. `first_batch_update_states_ns`
2. `first_batch_prepare_inputs_ns`
3. `first_batch_load_kv_ns`
4. `first_batch_forward_ns`

#### 为什么 `load_kv` 要单独量

`DecodeBenchConnector` 的主要热点很可能就在这里：

1. 按 block 构造 `fill_values`
2. 对多个 layer 的 KV cache 批量写入
3. 时间随 prompt 长度和 block 数增长

当前逻辑在 `start_fill_kv()` 中同步执行，不单独拆出来的话，会被误归到 `forward` 或整体 worker preprocess。

#### `load_kv` 建议定义

定义为：

1. 进入 `start_load_kv() / start_fill_kv()` 前开始
2. `start_fill_kv()` 返回后结束

注意：

1. 这里量到的是 connector 侧 dummy fill 的 wall time。
2. 对 `DecodeBenchConnector` 这是最关键的热点段。

### 7.7 ⑧ Sampling

目标覆盖：

1. logits 后处理
2. 采样
3. 语法 bitmask 应用后到 sampled token 得出

建议挂点：

1. `vllm/v1/worker/gpu_model_runner.py`
   - `sample_tokens()`
   - `_sample()`

建议字段：

1. `first_batch_sample_ns`

补充：

1. 现有 `async_output_copy_wait_ms` 保留，作为 sampling 后异步拷贝等待时间。

### 7.8 ⑨ 结果回传

建议拆三段：

1. EngineCore 输出序列化/发送
2. AsyncLLM output handler + OutputProcessor
3. API server 组装并发出首个文本 SSE chunk

#### 7.8.1 EngineCore -> AsyncLLM IPC out

建议挂点：

1. `vllm/v1/engine/core.py`
   - `process_output_sockets()`
2. `EngineCoreOutputs.timestamp`

建议定义：

1. 发送前记录 `engine_output_send_ns`
2. 前端 `get_output_async()` 收到后记录 `engine_output_recv_ns`
3. 差值作为 `ipc_out_ns`

#### 7.8.2 OutputProcessor

建议挂点：

1. `vllm/v1/engine/async_llm.py`
   - `_run_output_handler()`
2. `vllm/v1/engine/output_processor.py`
   - `process_outputs()`

建议字段：

1. `output_processor_ns`

#### 7.8.3 SSE 发出

建议挂点：

1. `vllm/entrypoints/openai/chat_completion/serving.py`
2. `vllm/entrypoints/openai/completion/serving.py`

建议定义：

1. 在“第一次发送带文本的 chunk”前记录 `t0`
2. `yield` 前或构造 data 完成后记录 `t1`
3. `first_text_sse_emit_ns = t1 - output_processor_done`

## 8. 首个 SSE 的定义必须修正

这是本方案里一个重要细节。

对 OpenAI chat 接口，当前流式输出的第一个 SSE 往往是空的 role chunk，例如：

1. `delta.role = assistant`
2. `delta.content = ""`

这个 chunk 不等价于“收到第一个文本 token”。

因此：

1. 如果把 TTFT 触发点定义成“第一次 yield SSE”
2. 那么 chat 的 TTFT 会系统性偏小

正确做法应为：

1. 服务端日志触发点：第一次发出带文本内容的 SSE chunk
2. benchmark 客户端触发点：第一次收到包含文本内容的 chunk

completion 接口可以按 `text` 非空或 `choices` 中存在文本 delta 的规则处理。

## 9. 日志格式建议

建议新增与 `ASYNC_STEP_TIMING` 同风格的单行日志：

```text
TTFT_TIMING request_id=... external_req_id=... connector=DecodeBenchConnector batch_id=1520
api_preprocess_ms=...
ipc_in_decode_ms=...
engine_preprocess_ms=...
queue_wait_ms=...
first_batch_schedule_cpu_ms=...
first_batch_update_states_ms=...
first_batch_prepare_inputs_ms=...
first_batch_load_kv_ms=...
first_batch_forward_ms=...
first_batch_sample_ms=...
async_output_copy_wait_ms=...
ipc_out_ms=...
output_processor_ms=...
first_text_sse_emit_ms=...
server_ttft_ms=...
```

补充字段建议：

1. `node_rank`
2. `dp_rank`
3. `tp_rank`
4. `dcp_rank`
5. `cudagraph_mode`
6. `replay_gpu_ms`
7. `replay_wall_ms`

这些字段在 batch timing 中已有成熟来源，直接复用即可。

## 10. 配置建议

不建议默认开启。

建议新增 observability 开关，例如：

1. `enable_logging_ttft_timing_details`
2. `logging_ttft_timing_interval`

理由：

1. 这类日志是 request 级的，比 batch 级日志量更大。
2. 对高 QPS 服务，如果每个请求都打一行，日志量会明显增加。
3. 定位热点时通常只需要 sampling 打开。

建议行为：

1. 只有打开 `enable_logging_ttft_timing_details` 时才构建 request trace。
2. 默认 interval 为 1。
3. 后续可支持按请求采样。

## 11. 实现顺序建议

建议按最小闭环分三步做，不要一次把所有阶段都铺满。

### Step 1

先做服务端 `TTFT_TIMING v1`，覆盖：

1. `api_preprocess_ms`
2. `ipc_in_decode_ms`
3. `engine_preprocess_ms`
4. `queue_wait_ms`
5. `first_batch_schedule_cpu_ms`
6. `first_batch_load_kv_ms`
7. `first_batch_forward_ms`
8. `first_batch_sample_ms`
9. `ipc_out_ms`
10. `output_processor_ms`
11. `server_ttft_ms`

这样已经能回答大多数热点问题，尤其是 `DecodeBenchConnector.start_fill_kv()`。

### Step 2

把 GPU 侧拆得更细：

1. `first_batch_update_states_ms`
2. `first_batch_prepare_inputs_ms`

用于进一步区分：

1. persistent batch bookkeeping
2. input tensor 构造
3. connector fill

### Step 3

再补 benchmark 客户端对齐：

1. 修正 chat 的 TTFT 触发条件
2. 新增 `http_post_send_ms`
3. 新增 `http_response_transfer_ms`
4. 打通 `request_id` 关联

## 12. benchmark 客户端配套修改建议

当前 benchmark 对 chat 的 TTFT 计算偏宽松，看到第一个 `choices` chunk 就会开始记 TTFT，但这个 chunk 可能只是 role，不是文本。

建议修改：

1. chat completion:
   - 只在 `choices[0]["delta"].get("content")` 真正出现文本时记 TTFT
2. completion:
   - 只在 `choices[0].get("text")` 对应文本 delta 到达时记 TTFT

此外建议增加：

1. `collect_ttft_breakdown`
2. 从 SSE `usage` 或扩展字段里读 `server_ttft_ms`
3. 最终输出 client/server 对比

## 13. 风险与注意事项

### 13.1 batch-shared 时间不要伪装成 request-exclusive

例如：

1. scheduler CPU
2. update_states
3. prepare_inputs
4. forward
5. sample

这些本质上都属于“该请求第一次出 token 所在 batch”的共享时间。

因此命名必须显式带上 `first_batch_`。

### 13.2 单调时钟和墙钟不能混用

当前链路里同时存在：

1. `time.time()`
2. `time.monotonic()`
3. `time.perf_counter_ns()`

建议：

1. 新增细粒度 duration 全部统一使用 `perf_counter_ns()`
2. 跨已有 request metrics 的队列时间仍按现有 monotonic event 计算
3. 不要直接拿不同 clock domain 的绝对时间做减法

### 13.3 多进程数据传递要轻量

新增 request trace 结构时应避免：

1. 大字典层层复制
2. 每个阶段都跨进程发送完整对象

建议：

1. request trace 只存少量整数 ns 字段
2. worker 只把“第一次 batch 的必要 timing”回传
3. 最终在前端或 output path 汇总并打印

### 13.4 Chat 首个空 chunk 问题

这是最容易让 TTFT 结果偏小的地方。

不修正的话：

1. 客户端 TTFT
2. 服务端 TTFT

都会和“first SSE chunk with text”的定义不一致。

## 14. 建议的最终口径

建议统一三套口径：

1. `client_ttft_ms`
   - 客户端 `T0 -> T1(first text chunk received)`
2. `server_ttft_ms`
   - 服务端开始处理请求到发出第一个文本 SSE chunk
3. `queue_wait_ms`
   - request 首次 `QUEUED -> SCHEDULED`

其中：

1. `client_ttft_ms` 用于用户视角
2. `server_ttft_ms` 用于服务端性能定位
3. `queue_wait_ms` 用于判断是调度拥塞还是执行慢

## 15. 结论

这套设计的关键点有三条：

1. 用 request-scoped `TTFT_TIMING` 补足现有 batch-level `ASYNC_STEP_TIMING` 的缺口。
2. 把 `DecodeBenchConnector.start_fill_kv()` 单独打出来，因为它极可能是该场景下的主热点。
3. 明确区分“第一次 SSE chunk”和“第一次带文本的 SSE chunk”，避免 TTFT 定义漂移。

如果只做第一版，最有价值的最小集合是：

1. `api_preprocess_ms`
2. `ipc_in_decode_ms`
3. `engine_preprocess_ms`
4. `queue_wait_ms`
5. `first_batch_schedule_cpu_ms`
6. `first_batch_load_kv_ms`
7. `first_batch_forward_ms`
8. `first_batch_sample_ms`
9. `ipc_out_ms`
10. `output_processor_ms`
11. `server_ttft_ms`

这已经足够定位 `DecodeBenchConnector` 场景下的大部分 TTFT 热点。
