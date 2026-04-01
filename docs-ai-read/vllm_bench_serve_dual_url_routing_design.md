# vLLM Bench Serve Dual URL Routing Design

Last updated: 2026-03-31
Repo basis: local `/vllm` checkout used in this session

## 1. Goal

在 `vllm bench serve` 的 `random csv` 场景下增加一套分流能力，但本阶段只形成设计文档，不修改业务代码：

- 支持传入两个 server URL / 端口
- 在 benchmark 发请求时按请求长度分流
  - 若 `prompt_len < 100000`，请求发送到 short 端口
  - 若 `prompt_len >= 100000`，请求发送到 long 端口
- 两个端口的请求结果在 benchmark 侧统一统计，输出一份总结果

本设计默认“数据长度”解释为 benchmark 请求的 `prompt_len`，也就是 token 长度。

## 2. Current Code Path

当前相关链路如下：

- `vllm/benchmarks/datasets.py`
  - `RandomDataset._sample_from_csv()`
  - `get_samples()`
- `vllm/benchmarks/serve.py`
  - `add_cli_args()`
  - `main_async()`
  - `benchmark()`
  - `calculate_metrics()`
- `vllm/benchmarks/lib/endpoint_request_func.py`
  - `RequestFuncInput`
  - completion request funcs

当前已确认的关键事实：

- `random csv` 已经在本地代码中实现，不是纯设计态能力
- `RandomDataset._sample_from_csv()` 会把 CSV 行转成：
  - `SampleRequest.prompt = list[int]`
  - `SampleRequest.prompt_len = row["prompt_len"]`
  - `SampleRequest.expected_output_len = row["output_len"]`
- `random csv` 当前只支持 completion 风格 backend
  - `--backend vllm`
  - `--backend openai`
- `RequestFuncInput` 已经把 `api_url` 设计成“每个请求独立携带”的字段
- `benchmark()` 当前只是把同一个 `api_url` 填给所有请求
- `asyncio.gather(*tasks)` 会保持输出顺序与输入任务顺序一致
- `calculate_metrics()` 只依赖 `input_requests[i]` 与 `outputs[i]` 的一一对应关系

因此，这次能力最自然的改动点是：

- 保持 `datasets.py` 不变
- 保持 request function 协议层不变
- 只在 `serve.py` 中把“单一目标 URL”扩展为“按请求选择目标 URL”

## 3. Existing Random CSV Behavior

### 3.1 CSV 采样

`RandomDataset._load_csv_lengths()` 当前会读取：

- `prompt_len`
- `output_len`

并要求两列均为正整数。

### 3.2 请求构造

`RandomDataset._sample_from_csv()` 会为每一行构造一个等长 token-id prompt。

这意味着：

- `prompt_len` 是 benchmark 内部的精确 token 长度
- 不需要额外 decode / re-encode
- 可以直接作为分流条件

### 3.3 当前 backend 边界

`serve.py` 中 `_validate_random_csv_args()` 已限制：

- `--dataset-name` 必须是 `random`
- 不支持 chat completion backend
- endpoint 必须是 `/completions`

这与本次需求天然一致。

## 4. Design Principles

本设计遵循以下原则：

- 分流只发生在 `bench serve` 客户端侧
- 数据集采样逻辑不变
- 请求 payload 构造逻辑不变
- 统计逻辑尽量复用现有总统计实现
- 对用户暴露的语义保持简单明确
- 第一版只支持 random csv completion 场景，不做泛化

## 5. Proposed CLI

建议新增以下参数：

```bash
--routing-prompt-len-threshold 100000
--routing-base-url-short http://127.0.0.1:8000
--routing-base-url-long http://127.0.0.1:8001
```

语义如下：

- `--routing-prompt-len-threshold`
  - 按 `prompt_len` 分流的阈值
  - 默认可设为 `100000`
- `--routing-base-url-short`
  - 小于阈值时使用的 base URL
- `--routing-base-url-long`
  - 大于等于阈值时使用的 base URL

分流规则：

- `prompt_len < threshold` -> short route
- `prompt_len >= threshold` -> long route

### 5.1 与现有参数的关系

建议当 routing 参数启用时：

- `--endpoint` 继续全局共用
- `--header` 继续全局共用
- `--extra-body` 继续全局共用
- `--ignore-eos` 继续全局共用
- `--request-rate` 继续表示“全局 RPS”
- `--max-concurrency` 继续表示“全局最大并发”

### 5.2 推荐约束

建议 routing 模式下要求显式提供：

```bash
--model <model>
```

原因：

- 当前自动探测 model 的逻辑只面向单个 `base_url`
- 若两个 server 的 `/v1/models` 返回不同，自动探测语义会变得不清晰
- 第一版强制显式传 `--model` 最稳

## 6. Scope and Validation Rules

建议新增 `_validate_prompt_len_routing_args(args)`，并在 `main_async()` 早期调用。

第一版建议只允许在以下条件下启用 routing：

- `args.dataset_name == "random"`
- `args.random_csv_path is not None`
- `args.backend in {"vllm", "openai"}`
- `args.endpoint.rstrip("/").endswith("/completions")`
- `args.model is not None`
- `args.skip_tokenizer_init == False`

不建议第一版支持：

- `openai-chat`
- embeddings / pooling / rerank backend
- `random-mm`
- 非 CSV 的 random dataset

如果只设置了部分 routing 参数，应直接报错，而不是静默退回单 URL 模式。

## 7. URL Resolution

当前 `main_async()` 会根据：

- `--base-url`
- 或 `--host --port`

生成单个：

- `base_url`
- `api_url`

第一版 routing 模式建议改成解析出两个目标：

- `route_short_base_url`
- `route_short_api_url = route_short_base_url + args.endpoint`
- `route_long_base_url`
- `route_long_api_url = route_long_base_url + args.endpoint`

非 routing 模式继续保留现有单目标行为。

### 7.1 是否继续支持 `--host/--port`

建议 routing 模式下优先只支持：

- `--routing-base-url-short`
- `--routing-base-url-long`

即不再把 `--host/--port` 作为 short/long 双目标的组合来源。

原因：

- 双目标场景下，`host/port` 接口不够自然
- 两个显式 `base_url` 更容易表达异构部署
- 也更利于未来扩展到 https/self-signed 场景

## 8. Request Routing Design

### 8.1 核心判断点

请求分流应发生在 `benchmark()` 主循环中，也就是当前创建 `RequestFuncInput` 的位置。

当前代码是：

- 从 `SampleRequest` 取出 `prompt_len`
- 构造 `RequestFuncInput(api_url=api_url, ...)`
- 创建任务

新逻辑只需把单一 `api_url` 改为：

- `selected_api_url = route_short_api_url if prompt_len < threshold else route_long_api_url`

然后继续复用原有 `request_func`。

### 8.2 为什么不改 request func

因为 `RequestFuncInput` 已经是：

- 每请求自带 `api_url`

completion request funcs 也都是：

- 从 `request_func_input.api_url` 读取目标地址

因此不需要在 `endpoint_request_func.py` 新增任何 routing 概念。

### 8.3 为什么不改数据集层

因为 random csv 已经产出精确的：

- `prompt_len`
- `expected_output_len`

且分流属于 benchmark 发送策略，而不是 dataset 采样策略。

## 9. Ready Check, Warmup, Profile

这部分是第一版最容易被忽略的行为差异。

### 9.1 Ready Check

当前 ready check 只对单个 `api_url` 做一次测试。

routing 模式下建议：

1. 先遍历 `input_requests`，统计哪些 route 会实际命中
2. 对每个 active route 单独构造 test input 并执行 ready check
3. 若某个 active route ready check 失败，整个 benchmark fail fast
4. 对没有任何请求命中的 route 不做 ready check

这样可以避免：

- long 端口根本不会被打到，但启动前仍然阻塞等待

### 9.2 Warmup

当前 warmup 基于单个测试请求重复发 `num_warmups` 次。

routing 模式下建议：

- 对每个 active route 各做 `num_warmups` 次 warmup

也就是说：

- 若 short/long 都 active，则总 warmup 数变成 `2 * num_warmups`
- 若只有 short active，则只 warmup short

原因：

- 两个端口的缓存、编译、连接池状态相互独立
- 只预热一个端口会导致另一个端口首批真实请求偏慢

### 9.3 Profile

当前 `profile` 逻辑通过：

- `base_url + "/start_profile"`
- `base_url + "/stop_profile"`

控制单个服务。

routing 模式下建议：

- 对每个 active route 分别发 `start_profile`
- benchmark 完成后分别发 `stop_profile`

若其中一个 route profile 失败：

- 建议 warning
- 不阻断整个 benchmark 主流程

## 10. Metrics and Aggregation

### 10.1 总统计

总统计建议完全复用现有 `calculate_metrics()`。

原因：

- 该函数只关心：
  - `input_requests`
  - `outputs`
  - `dur_s`
- 不关心请求发往哪个端口
- 只要 `outputs[i]` 仍与 `input_requests[i]` 对齐，统计就天然正确

因此：

- 吞吐
- TTFT
- TPOT
- ITL
- E2EL
- goodput

都可以继续按“一次 benchmark 的整体结果”计算。

### 10.2 全局 benchmark duration

建议继续保持当前口径：

- 整个 benchmark 的 wall-clock duration

即从第一个任务调度开始，到 `await asyncio.gather(*tasks)` 返回结束。

不要拆成 short/long 各自 duration 后再做加权。

### 10.3 route 级附加统计

虽然总统计可以直接复用，但建议在结果 JSON 中增加 route 级元数据：

- `routing_enabled`
- `routing_threshold_prompt_len`
- `routing_base_url_short`
- `routing_base_url_long`
- `routing_request_counts`
  - `{"short": <count>, "long": <count>}`

可选增加：

- `request_routes`
  - 与 `input_lens` 同长度
  - 每个值为 `"short"` 或 `"long"`

这不会改变现有主统计口径，但会提高可解释性。

## 11. Result JSON Design

建议 routing 模式下在总结果里新增：

```json
{
  "routing_enabled": true,
  "routing_threshold_prompt_len": 100000,
  "routing_base_url_short": "http://127.0.0.1:8000",
  "routing_base_url_long": "http://127.0.0.1:8001",
  "routing_request_counts": {
    "short": 123,
    "long": 456
  }
}
```

若 `--save-detailed`，建议额外保存：

```json
{
  "request_routes": ["short", "short", "long", "short", "long"]
}
```

这样：

- timeline plot 仍可复用现有 `start_times/ttfts/itls`
- 离线分析时也能知道每条请求落到哪个 route

## 12. Concurrency and Connection Pooling

当前实现只使用一个 `aiohttp.ClientSession` 与一个 `TCPConnector`。

routing 模式下有两种选择：

### 方案 1：一个 session，复用现有模式

- 仍只创建一个 `ClientSession`
- 不同请求使用不同 `api_url`
- `TCPConnector` 自然会按 host 维护连接池

优点：

- 改动最小
- 复用度最高

缺点：

- 对 route 级别连接限制的控制较弱

### 方案 2：两个 session，short/long 各自独立

- short route 一个 session
- long route 一个 session
- 任务创建时按 route 选 session

优点：

- 资源隔离更清晰
- route 级 debug 更容易

缺点：

- 改动更大
- 需要把 session 也纳入 per-request 路由逻辑

第一版建议选择方案 1。

原因：

- 当前需求重点是路由正确性与总统计
- 一个 session 已足够支撑双 URL 并发请求
- 不需要改变 request func 协议签名

## 13. Goodput and Spec Decode Metrics

### 13.1 Goodput

goodput 继续复用现有总统计逻辑即可，不需要 route 级单独计算。

### 13.2 Speculative Decoding Metrics

当前实现只对单个 `base_url` 的 `/metrics` 采样前后值。

routing 模式下若两端都开启 spec decode，建议：

- 对每个 active route 分别采集 before / after
- 按 delta 累加得到总 `num_drafts`
- 按 delta 累加得到总 `draft_tokens`
- 按 delta 累加得到总 `accepted_tokens`
- 每 position 的 accepted 数也做 delta 后再合并

然后基于合并后的总量计算：

- `acceptance_rate`
- `acceptance_length`
- `per_position_acceptance_rates`

若任一 active route 无法成功获取 metrics：

- 建议整体跳过 spec decode 统计
- 并打印 warning

不要混合“部分 route 有 spec stats，部分没有”的结果，以免语义不一致。

## 14. Failure Semantics

建议定义如下失败语义：

- short/long 中任一 active route ready check 失败
  - benchmark 直接失败
- 主 benchmark 过程中单请求失败
  - 沿用现有逻辑，只计入 failed
- 某个 inactive route 不可达
  - 不影响 benchmark
- profile 或 spec metrics 获取失败
  - warning，不中断主 benchmark

## 15. Backward Compatibility

非 routing 模式必须保持现有行为完全不变：

- 参数不变
- 单 URL 行为不变
- JSON 输出字段保持兼容
- timeline / dataset stats plot 不受影响

只有当 routing 参数完整启用时，才进入新逻辑。

## 16. Recommended Implementation Plan

建议按如下顺序实施：

1. 在 `vllm/benchmarks/serve.py` 的 `add_cli_args()` 中新增 routing 参数
2. 新增 `_validate_prompt_len_routing_args(args)`
3. 在 `main_async()` 中解析 short/long 两个 base URL 与 api URL
4. 在 `benchmark()` 增加 routing config 入参
5. 在 ready check / warmup / profile / spec metrics 处扩展为按 active route 处理
6. 在主循环创建 `RequestFuncInput` 时按 `prompt_len` 选 `api_url`
7. 在结果 JSON 中增加 routing 元数据
8. 增加测试

## 17. Test Plan

建议至少覆盖以下测试。

### 17.1 CLI 校验

- 仅设置 `--routing-base-url-short`，缺少 long，报错
- routing 模式下未设置 `--model`，报错
- routing 模式下未设置 `--random-csv-path`，报错
- routing 模式下使用 `openai-chat`，报错

### 17.2 路由边界

构造 CSV：

```csv
prompt_len,output_len
99999,8
100000,8
100001,8
```

验证：

- `99999` -> short route
- `100000` -> long route
- `100001` -> long route

### 17.3 双 server 集成测试

启动两个 test server：

- short server
- long server

建议让两个 server 在响应中带上可区分标记，或分别记录收到的请求数。

验证：

- short 收到的请求数正确
- long 收到的请求数正确
- benchmark 总 completed 数正确

### 17.4 结果 JSON

验证保存结果中存在：

- `routing_enabled`
- `routing_threshold_prompt_len`
- `routing_request_counts`

若 `--save-detailed`：

- `request_routes` 长度与请求数一致

### 17.5 统计一致性

验证：

- `completed + failed == len(input_requests)`
- `input_lens` 顺序与 `request_routes` 顺序一致
- route 级请求数之和等于总请求数

## 18. Risks and Non-Goals

### 18.1 Risks

- 两个 server 若配置不同，`--model`、sampling 参数、LoRA 可用性可能不一致
- spec decode metrics 的合并逻辑比主 benchmark 路由更复杂
- 若用户期望“每端口独立 RPS 控制”，第一版不会满足

### 18.2 Non-Goals

第一版不包含：

- 按 output length 分流
- 多于两个 route
- route 级独立 request rate
- route 级独立 max concurrency
- route 级独立 headers / extra body / endpoint
- 泛化到非 random csv 数据集

## 19. Final Recommendation

推荐采用“最小侵入”方案：

- 只在 `serve.py` 增加 routing
- `datasets.py` 不改
- `endpoint_request_func.py` 不改
- 总统计继续复用现有 `calculate_metrics()`

最关键的设计判断是：

- 分流依据使用 `SampleRequest.prompt_len`
- 分流时机放在 `benchmark()` 创建 `RequestFuncInput` 的地方
- 统计口径保持一次 benchmark 的全局合并结果

这条路径改动小、风险低，也最符合当前代码结构。
