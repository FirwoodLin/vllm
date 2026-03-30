# vLLM Bench Random CSV Design

Last updated: 2026-03-26
Repo basis: local `/vllm` checkout used in this session

## 1. Goal

为 `vllm bench serve` 增加两个能力，但本阶段只形成设计方案，不修改代码：

- 新增 `--random-csv-path`
  - 用指定 CSV 文件中的 `prompt_len` 和 `output_len` 作为 benchmark 请求长度分布
- 新增 `--no-save-generated-texts`
  - 在 `--save-detailed` 场景下，不把 `generated_texts` 写入结果 JSON

用户给定的目标 CSV 为：

```text
/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o/sharegpt4o-mixed-random-60k.csv
```

本次调研已确认该文件存在，格式为：

```csv
prompt_len,output_len
213,459
200,664
214,865
...
```

共 60000 行，仅包含两列：

- `prompt_len`
- `output_len`

## 2. Current Code Path

本次需求主要落在以下路径：

- `vllm/benchmarks/datasets.py`
  - `add_dataset_parser()`
  - `add_random_dataset_base_args()`
  - `RandomDataset`
  - `get_samples()`
- `vllm/benchmarks/serve.py`
  - `add_cli_args()`
  - `benchmark()` 返回结果组装
  - `main_async()` 中的结果裁剪和保存逻辑
- `vllm/benchmarks/lib/endpoint_request_func.py`
  - completion / chat backend 的 HTTP payload 构造
- `vllm/entrypoints/openai/completion/protocol.py`
  - `/v1/completions` 的 `prompt` 类型约束
- `vllm/inputs/data.py`
  - prompt 类型对 token-id 输入的支持边界

当前关键事实：

- `bench serve` 的 dataset CLI 入口复用 `datasets.py`
- `random` 数据集当前只支持按配置随机采样长度
- `serve` 详细结果中的字段名是 `generated_texts`
  - 不是 `generated-texts`
- `vllm/inputs/data.py` 已支持 tokenized prompt
  - `DecoderOnlyPrompt` 明确包含 `list[int]`
- `vllm/entrypoints/openai/completion/protocol.py` 中
  - `/v1/completions` 的 `prompt` 明确支持 `list[int]` / `list[list[int]]`
- `bench serve` 的 completion 请求函数当前会把 `request_func_input.prompt`
  - 直接放进 HTTP payload 的 `prompt` 字段
- 但 `bench serve` 的 `openai-chat` 请求函数当前会把 `prompt`
  - 组装成 `messages[].content[].text`
  - 这条路径天然要求文本，而不是 token ids

## 3. Design Decision Summary

### 3.1 `--random-csv-path` 的定位

建议仍把 `--random-csv-path` 加到 `add_random_dataset_base_args()`。

原因：

- `bench serve` 是通过 `add_dataset_parser()` 间接复用这组 random 参数的
- 该参数最自然的挂载点仍然是 random dataset 参数组
- 但本设计范围只覆盖 `bench serve`

范围说明：

- 第一版只保证 `vllm bench serve` 行为正确
- `bench throughput` 不在本次设计和测试范围内

### 3.2 CSV 的语义

CSV 不应只提供长度分布；在 CSV 模式下，应直接生成 token-id prompt，
而不是再退回文本 prompt。

更具体地说：

- 读取 CSV 中的 `prompt_len` / `output_len`
- 直接按 `prompt_len` 构造等长的随机 token id 序列
- `expected_output_len` 直接使用 CSV 中的 `output_len`
- `SampleRequest` 在 CSV 模式下应携带 tokenized prompt，而不是文本 prompt

这是本设计里最重要的决策。

这样做的好处：

- 输入 token 数可以与 CSV 完全一致
- 避免 decode -> re-encode 带来的 token 长度漂移
- 避免无意义的 detokenize / retokenize 开销
- 更贴近 `vllm.inputs` 已有的 token-id 输入能力

### 3.3 `list[int]` completion prompt 应成为主设计

用户提到 `vllm.inputs.TokenInputs.prompt_token_ids`，这个方向是对的。

更准确地说：

- `vllm.inputs` 已允许把 tokenized prompt 表示成 `list[int]`
- `/v1/completions` 协议也直接接受 `list[int]`
- `bench serve` 的 completion 请求函数会把 `prompt` 原样写入 payload

因此，对 `bench serve` 而言，CSV 模式下的规范表示应直接采用：

- `SampleRequest.prompt = list[int]`

这样有两个直接好处：

- 不需要额外的 `serve` 侧 token-id 解包适配层
- `--backend vllm` / `--backend openai` 可以直接复用现有 completion 请求链路

### 3.4 backend 边界应明确，而不是做隐式回退

已确认的现状是：

- `bench serve` 的 completion backend 可以自然接受 `list[int]`
- `bench serve` 的 `openai-chat` backend 会把 prompt 放进
  - `messages[].content[].text`

因此第一版建议：

- 支持 `--backend vllm`
- 支持 `--backend openai`
- 要求请求路径为 completion 风格接口
- 对 `openai-chat` 等文本型 backend 直接 fail fast
- 不做 token ids -> 文本 的隐式回退

## 4. Detailed Design

## 4.1 CLI: `--random-csv-path`

建议新增参数：

```bash
--random-csv-path /path/to/data.csv
```

放置位置：

- `vllm/benchmarks/datasets.py`
- `add_random_dataset_base_args()`

建议 help 文案语义：

- 指定一个包含 `prompt_len` 和 `output_len` 的 CSV 文件
- 当该参数存在时，`random` 数据集的请求长度从 CSV 读取
- 该参数仅对 `--dataset-name random` 生效

建议第一版支持的 CSV 列：

- 必填：`prompt_len`
- 必填：`output_len`
- 其他列忽略

### 参数生效范围

建议只在以下场景生效：

- `--dataset-name random`

不建议第一版让它作用于：

- `random-mm`
- `random-rerank`

原因：

- 这两类数据集含有额外语义
- 本次需求明确只需要把 CSV 长度分布用于 benchmark 数据
- 第一版把语义压缩在 `random` 上最稳

对于 `bench serve` 的 backend，建议第一版只支持：

- `--backend vllm`
- `--backend openai`

且要求请求路径为 completion 风格接口。

第一版不建议支持：

- `--backend openai-chat`
- `--backend openai-audio`
- pooling / embedding / rerank 类 backend

原因：

- 这些 backend 当前请求格式不是 completion `prompt`
- 尤其 `openai-chat` 当前是显式文本消息结构
- 为避免静默退化为文本 prompt，第一版应直接报错

### 参数优先级

当 `--random-csv-path` 被设置时，建议忽略以下参数并给出 warning：

- `--random-input-len`
- `--random-output-len`
- `--random-range-ratio`

`--random-prefix-len` 不必再忽略，可以在 CSV 模式下继续支持，但语义应改成：

- prefix 计入 CSV 的总 `prompt_len`
- 即每条样本的 suffix 长度为 `csv_prompt_len - prefix_len`
- 若任一 CSV 行的 `prompt_len < prefix_len`，直接报错

这样既保留了 prefix 语义，也不会破坏 CSV 总 token 长度。

## 4.2 `RandomDataset` 行为调整

建议在 `RandomDataset` 中增加一条“CSV 长度源 + token-id prompt”分支，
但不改变原有随机文本采样路径。

建议新增状态：

- `self.random_csv_path`
- `self.csv_lengths`

建议新增私有方法：

- `_load_csv_lengths()`

### `_load_csv_lengths()` 设计

读取方式：

- 使用标准库 `csv.DictReader`

校验规则：

- 文件不存在：抛 `FileNotFoundError`
- 文件为空：抛 `ValueError`
- 缺少 `prompt_len` 或 `output_len` 列：抛 `ValueError`
- 列值不是正整数：抛 `ValueError`

输出结构建议为：

```python
list[dict[str, int]]
```

例如：

```python
[
    {"prompt_len": 213, "output_len": 459},
    {"prompt_len": 200, "output_len": 664},
]
```

### `sample()` 设计

当 `csv_lengths` 存在时：

1. 先从 CSV 决定每个请求的目标 `prompt_len` 和 `output_len`
2. 生成精确等长的 `prompt_token_ids`
3. 返回携带 tokenized prompt 的 `SampleRequest`

建议返回形态：

```python
SampleRequest(
    prompt=prompt_token_ids,
    prompt_len=len(prompt_token_ids),
    expected_output_len=output_len,
    request_id=...,
)
```

这里不再调用 `RandomDataset.generate_token_sequence()` 去生成文本 prompt。

相反，建议新增一个仅生成 token ids 的辅助逻辑，例如：

- `_generate_exact_length_token_ids()`
- 或 `_generate_token_id_sequence()`

其职责是：

- 基于 `allowed_tokens` 生成可复现的随机 token id 序列
- 在 CSV 模式下严格保证 `len(prompt_token_ids) == csv_prompt_len`
- 若启用 `prefix_len`，则把 prefix 作为总长度中的一部分

### `prompt_token_ids` 生成性能约束

这部分应作为实现硬约束：

- 必须使用 `numpy` 向量化生成 `prompt_token_ids`
- 不能使用 Python 原生逐 token 追加的方式生成
- `allowed_tokens` 应保持为 `np.ndarray`
- prefix 和 suffix 的拼接也应尽量基于 `numpy` 数组完成，最后再转成 `list[int]`

建议实现方向：

- 继续复用 `RandomDataset` 当前的 `numpy.default_rng`
- 用 `self._rng.integers(...)`、切片、广播、`np.arange(...)`、模运算等方式生成整段 token ids
- 每个请求最多在“按请求组装结果”这一层做 Python 循环
- 不能在单条请求内部按 token 做 Python `for` 循环逐个生成

原因：

- CSV 样本量可能很大，例如当前文件有 60000 行
- `prompt_len` 也可能较长，逐 token Python 循环会引入不必要的解释器开销
- 当前 `RandomDataset.generate_token_sequence()` 已经体现了 `numpy` 驱动的生成思路，CSV token-id 模式应保持同样的性能取向

### CSV 模式下的采样与打乱

建议规则：

- 默认按 `seed` 可复现地 shuffle 后取样
- `--disable-shuffle` 时保持 CSV 原始顺序

原因：

- 与当前 benchmark 数据集设计风格一致
- 保留 deterministic 行为

### CSV 模式下的请求数处理

当 `num_prompts` 与 CSV 行数不一致时，建议规则如下：

- `num_prompts <= csv_rows`
  - 取前 `num_prompts` 个样本
- `num_prompts > csv_rows` 且未设置 `--no-oversample`
  - 循环重复 CSV 样本，直到补足
- `num_prompts > csv_rows` 且设置 `--no-oversample`
  - 只返回 CSV 可提供的行数

这与当前 `maybe_oversample_requests()` 的总体设计方向一致。

## 4.3 `get_samples()` 与 `serve` 请求链路

需要把 `random_csv_path` 透传给 `RandomDataset` 的构造逻辑。

涉及位置：

- `vllm/benchmarks/datasets.py`
  - `get_samples()` 的 `"random"` 分支

除此之外，需要把 `serve` 路径上的 prompt 类型放宽到包含 `list[int]`：

- `SampleRequest.prompt`
- `RequestFuncInput.prompt`

对 `bench serve` completion backend 的要求是：

- 继续复用当前请求函数
- 直接发送 JSON `prompt: list[int]`
- 不新增额外解包层

对 `openai-chat` backend 的要求是：

- 若检测到 `--random-csv-path`
  - 立即报错并提示该 backend 当前不支持 token-id prompt

## 4.4 `--no-save-generated-texts`

建议新增参数：

```bash
--no-save-generated-texts
```

放置位置：

- `vllm/benchmarks/serve.py`
- 与 `--save-result` / `--save-detailed` 同一组

### 语义定义

仅影响 `bench serve` 的结果 JSON 保存行为。

具体规则：

- 默认行为不变
- 当 `--save-detailed` 未开启时，本来就不会保存 `generated_texts`
- 当 `--save-detailed` 开启且指定 `--no-save-generated-texts` 时：
  - 仍保留详细结果中的其他字段
  - 但删除 `generated_texts`

### 实现方式

当前主线已有一段统一裁剪逻辑：

- `if not args.save_detailed: ...`

建议在这段逻辑之后，再单独做一次：

- 若 `args.no_save_generated_texts`
  - 从 `result_json` 删除 `generated_texts`
  - 从 `benchmark_result` 删除 `generated_texts`

这样做的好处：

- 不影响当前默认行为
- 逻辑非常清晰
- 不会误删 `ttfts`、`itls`、`errors` 等其他详细字段

## 5. Why This Design

## 5.1 与主线代码风格一致

本设计尽量复用现有结构：

- 参数仍挂在 dataset parser
- 核心逻辑仍放在 `RandomDataset`
- `serve` completion backend 继续复用既有 payload 构造逻辑
- 只需要补齐类型放宽和 backend fail-fast

这样改动边界最清晰。

## 5.2 风险最低

相比原文档里的“退回文本 prompt”方案，修订后的方案更合理：

- 主线本身已经支持 token-id prompt，不需要人为降级成文本
- 真正需要控制的风险，只是 backend 兼容边界
- 这可以通过 fail-fast 解决
- 仍然不引入额外环境变量控制逻辑，例如 `CSV_MAX_MODEL_LENGTH`

因此，第一版最稳的方案不是“只复用长度分布”，而是：

- 对支持 token ids 的 `bench serve` completion backend，直接走 token ids
- 对不支持的 backend，明确报错

## 5.3 满足用户真实目标

用户当前目标本质上是：

- 用指定 CSV 中的 token 长度分布做 benchmark
- 保存详细结果时不要把生成文本写盘

如果仍然生成文本 prompt，那么 CSV 的 `prompt_len` 只能“近似满足”；
改为直接生成 token ids 后，才能严格满足这个目标。

## 6. Explicit Non-Goals for First Version

第一版不建议引入以下内容：

- 不支持 CSV 里直接放 prompt 文本
- 不支持 `CSV_MAX_MODEL_LENGTH` 之类的环境变量裁剪
- 不支持 `random-mm` / `random-rerank` 复用该 CSV 参数
- 不支持自动根据 server `max_model_len` 修改 CSV 长度
- 不对 `openai-chat` 等文本型 backend 做隐式文本回退
- 不把 `bench throughput` 一并纳入本次实现范围

说明：

- “CSV 模式内部生成 token-id prompt” 已是第一版目标
- 这里的 non-goal 指的是“不支持 CSV 文件直接提供 token ids 列”

这些能力都可以在后续需求明确后再加。

## 7. Proposed File Changes

若后续进入实现阶段，建议修改这些文件：

- `vllm/benchmarks/datasets.py`
- `vllm/benchmarks/serve.py`
- `vllm/benchmarks/lib/endpoint_request_func.py`
- `tests/benchmarks/test_random_dataset.py`
- `tests/benchmarks/test_serve_cli.py`
- 可选：`docs/benchmarking/cli.md`

## 8. Test Plan

## 8.1 Unit tests

建议在 `tests/benchmarks/test_random_dataset.py` 增加以下覆盖：

- 正常读取 CSV
- 缺少 `prompt_len` 列时报错
- 缺少 `output_len` 列时报错
- 空 CSV 报错
- 非整数值报错
- 同 seed 下 CSV 模式采样可复现
- `--disable-shuffle` 时保持原顺序
- `--no-oversample` 时不补齐
- CSV 模式下返回 `prompt=list[int]`
- 每条样本的 `len(prompt)` 与 CSV `prompt_len` 完全一致
- 若指定 `--random-prefix-len`
  - prefix 被计入总长度
  - 且 `csv_prompt_len < prefix_len` 时会报错

## 8.2 Integration tests for `bench serve`

建议在 `tests/benchmarks/test_serve_cli.py` 增加：

- 使用临时 CSV 的 `bench serve` completion 用例
  - 验证 `--random-csv-path` 能跑通
  - 验证请求体最终发送的是 token-id prompt，而不是文本 prompt
- 使用
  - `--save-result`
  - `--save-detailed`
  - `--no-save-generated-texts`
  的用例，验证结果文件中：
  - 不存在 `generated_texts`
  - 仍保留 `ttfts`
  - 仍保留 `itls`
  - 仍保留 `errors`
- `--backend openai-chat` + `--random-csv-path`
  - 应 fail fast
  - 且报错信息明确指出该 backend 当前不支持 token-id prompt

## 9. Open Questions

当前没有阻塞性的设计疑问。

唯一需要在实现时明确写入错误文案的是：

- 哪些 `bench serve` backend 支持 CSV token-id 模式
- 哪些 backend 会被明确拒绝

我的建议是第一版写死为：

- 支持 completion 型 backend
- 拒绝 `openai-chat` 和其他非 completion backend

## 10. Final Recommendation

建议按下面的最小可行方案实现：

1. 在 `datasets.py` 的 random dataset 参数组中新增 `--random-csv-path`
2. 在 `RandomDataset` 中增加 CSV 加载与 token-id 采样逻辑
3. CSV 模式下生成 `prompt=list[int]`，不再生成文本 prompt
4. 在 `get_samples()` 到 `bench serve` 请求链路中透传该参数
5. 放宽 `SampleRequest.prompt` 与 `RequestFuncInput.prompt` 的类型
6. 对 `openai-chat` 等不支持 token-id prompt 的 backend 直接报错
7. 在 `serve.py` 中新增 `--no-save-generated-texts`
8. 在 `--save-detailed` 后处理阶段单独删除 `generated_texts`
9. 补齐 random dataset、serve completion、backend fail-fast 的测试

这是修订后更符合主线能力边界、也更严格满足 CSV token 长度语义的方案。
