# vLLM Log Waiting/KV 时序绘图脚本设计

Status: design only

Last updated: 2026-04-04

## 1. 目标

设计一个离线脚本，解析 `vLLM` 服务日志中的 per-engine stats 行，并绘制以下三个指标随时间变化的曲线：

1. `Waiting tokens`
2. `Waiting head tokens`
3. `GPU KV cache usage`

目标不是修改 `vLLM` 的 logger，而是消费已有日志，输出可直接用于分析或论文插图的时序图。


## 2. 参考脚本与复用点

参考脚本：

- `/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/serving-log-parse/vllm-hol-v5-0228.py`

该脚本已经验证了以下工作流是有效的：

1. 用正则从日志中提取 per-engine 指标
2. 将数据整理为 `pandas.DataFrame`
3. 按 `timestamp` 和 `engine_id` 做透视表
4. 将横轴转换为相对时间
5. 用 `matplotlib` 生成适合分析/汇报的图

本设计复用其整体思路，但做三点调整：

1. 指标从 `Waiting blocks / Waiting blocks head / Free KV blocks` 切换到 `Waiting tokens / Waiting head tokens / GPU KV cache usage`
2. 解析逻辑改为“头部字段 + 单指标独立提取”，避免把整个日志格式写死在一个超长正则里
3. 绘图默认面向工程分析，不绑定外部字体，不强依赖 `seaborn`


## 3. 目标日志格式

### 3.1 当前主目标格式

当前 `vLLM` 主线 logger 的关键输出位于：

- `/vllm/vllm/v1/metrics/loggers.py`

其关键字段包括：

- `Waiting tokens: %d`
- `Waiting head tokens: %d`
- `GPU KV cache usage: %.1f%%`

典型样例如下：

```text
INFO 04-02 21:41:39 [loggers.py:265] Engine 009: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.7 tokens/s, Running: 1 reqs, Waiting: 1 reqs, Waiting tokens: 1025, Waiting head tokens: 1025, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%, External prefix cache hit rate: 100.0%
```


## 4. 脚本位置与依赖建议

建议后续实现为：

- `/vllm/tools/plot_vllm_log_timeseries.py`

最小依赖：

- `argparse`
- `re`
- `pathlib`
- `datetime`
- `pandas`
- `matplotlib`

可选依赖：

- `seaborn`

设计建议是不强依赖 `seaborn`。即使用户环境没有它，脚本也应能正常运行。


## 5. 输入输出设计

### 5.1 CLI 建议

```bash
python tools/plot_vllm_log_timeseries.py \
  --log /path/to/serve.log \
  --out-dir /path/to/output
```

建议参数：

- `--log`: 输入日志文件，必选
- `--out-dir`: 输出目录，必选
- `--year`: 日志年份，默认当前年
- `--engine-ids`: 只分析指定 engine，支持 `0,1,2` 或 `0-7`
- `--time-axis`: `relative` 或 `absolute`，默认 `relative`
- `--plot-mode`: `aggregate`、`per-engine`、`both`，默认 `both`
- `--format`: `png`、`pdf`、`svg`，默认 `png`
- `--dpi`: 默认 `200`
- `--title`: 自定义标题
- `--strict`: 遇到 token 指标缺失时直接报错；默认关闭
- `--dump-csv`: 导出解析后的明细表和聚合表

### 5.2 输出文件建议

对于输入 `foo.log`，建议输出：

- `foo.parsed.csv`
- `foo.aggregate.csv`
- `foo.aggregate.png`
- `foo.per_engine.png`


## 6. 解析设计

### 6.1 为什么不使用单个超长正则

参考脚本使用单个正则一次性匹配整行，这种写法对固定格式很高效，但对下面两类变化不够稳健：

1. logger 中间插入新字段，例如 `Preemptions`
2. 新旧日志共存，某些字段可能完全不存在

因此本设计建议改成“两阶段解析”：

1. 先提取公共头部：`timestamp` 和 `engine_id`
2. 再对每个指标单独 `search`

这样可以显著降低格式耦合。

### 6.2 建议的正则

```python
HEADER_RE = re.compile(
    r"(?P<timestamp>\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?"
    r"Engine\s+(?P<engine_id>\d+):"
)

WAITING_TOKENS_RE = re.compile(r"Waiting tokens:\s+(?P<value>\d+)")
WAITING_HEAD_TOKENS_RE = re.compile(r"Waiting head tokens:\s+(?P<value>\d+)")
GPU_KV_USAGE_RE = re.compile(
    r"GPU KV cache usage:\s+(?P<value>\d+(?:\.\d+)?)%"
)
```

### 6.3 行级过滤

为降低扫描成本，先做快速过滤：

- 不包含 `Engine ` 的行直接跳过
- 不包含 `GPU KV cache usage:` 的行直接跳过

这样可避免对大量无关日志反复执行正则。

### 6.4 时间戳处理

日志时间戳只有 `MM-DD HH:MM:SS`，没有年份，因此需要补年。

默认策略：

- 使用 `--year` 指定年份
- 若未指定，则使用当前年

注意：

- 如果日志跨年，这个简化策略会失效
- 第一版设计不处理跨年日志，后续若需要可增加 `--start-year` 与“跨年回卷检测”

### 6.5 建议的数据结构

建议每条记录解析为如下字段：

- `timestamp`
- `engine_id`
- `waiting_tokens`
- `waiting_head_tokens`
- `gpu_kv_cache_usage_pct`
- `line_number`

其中：

- `timestamp` 为 `datetime`
- `engine_id` 为 `int`
- `waiting_tokens` 为 `Int64` 或浮点兼容列
- `waiting_head_tokens` 为 `Int64` 或浮点兼容列
- `gpu_kv_cache_usage_pct` 为 `float`


## 7. 数据整理设计

### 7.1 明细表

解析完成后先得到明细表：

| timestamp | engine_id | waiting_tokens | waiting_head_tokens | gpu_kv_cache_usage_pct |
| --- | --- | --- | --- | --- |
| 2026-04-02 21:41:39 | 9 | 1025 | 1025 | 0.3 |

### 7.2 相对时间轴

参考脚本的做法，建议默认将横轴转成相对秒数：

```text
relative_seconds = timestamp - first_timestamp
```

这样更适合比较不同实验。

### 7.3 透视表

为了做 per-engine 图，建议分别构造三张透视表：

- `waiting_tokens_pivot`
- `waiting_head_tokens_pivot`
- `gpu_kv_usage_pivot`

统一形式：

- `index = timestamp`
- `columns = engine_id`
- `values = metric`
- `aggfunc = last`

### 7.4 聚合指标

建议额外生成如下聚合序列：

- `waiting_tokens_total = sum(waiting_tokens over engines)`
- `waiting_head_tokens_total = sum(waiting_head_tokens over engines)`
- `gpu_kv_usage_mean = mean(gpu_kv_cache_usage_pct over engines)`
- `gpu_kv_usage_max = max(gpu_kv_cache_usage_pct over engines)`

注意：

- `GPU KV cache usage` 是百分比，不应做求和
- 默认关注 `mean` 和 `max` 即可


## 8. 绘图设计

### 8.1 总体原则

三个指标量纲不同：

- `Waiting tokens` 和 `Waiting head tokens` 往往是大整数
- `GPU KV cache usage` 是 `0-100%`

因此不建议把三者硬塞到单一坐标轴里。更清晰的方案是输出两张图。

### 8.2 图一：聚合视图

文件名建议：

- `foo.aggregate.png`

采用 `3 x 1` 共享横轴子图：

1. `Waiting tokens total` 折线图
2. `Waiting head tokens total` 折线图
3. `GPU KV cache usage` 的 `mean/max` 折线图

细节建议：

- `x` 轴默认用 `relative_seconds`
- token 轴启用科学计数法
- GPU 使用率轴固定到 `0-100`
- 所有子图共享同一 `xlim`
- 仅最底部子图保留 `xlabel`

这样最适合看总体 backlog 与缓存压力的时序关系。

### 8.3 图二：per-engine 视图

文件名建议：

- `foo.per_engine.png`

该图直接复用参考脚本的核心思路，但把指标替换掉，采用 `3 x 1` 共享横轴子图：

1. `Waiting tokens` 的 stackplot，按 `engine_id` 堆叠
2. `Waiting head tokens` 的 stackplot，按 `engine_id` 堆叠
3. `GPU KV cache usage` 的多折线图，每个 engine 一条淡色线，并叠加一条加粗的 `mean` 线

原因：

- `Waiting tokens` 和 `Waiting head tokens` 做堆叠图，可以直观看出 backlog 来自哪些 engine
- `GPU KV cache usage` 是百分比，堆叠没有物理意义，因此应使用多折线

### 8.4 图例与可读性

当 engine 很多时，不建议把所有 engine 都放进图例。

建议策略：

- stackplot 图例仅保留一个总说明，例如 `Engines 0-15`
- 或仅显示 `mean/max`
- 多折线图里 engine 单线用半透明细线，减少图例噪声

### 8.5 缺失指标时的降级策略

若输入是旧日志，只有 `GPU KV cache usage`：

- 仍生成 `aggregate` 图
- 前两个 token 子图标记为 `Metric not available` 或直接跳过
- `per_engine` 图只保留 GPU 使用率子图


## 9. 处理细节与边界情况

### 9.1 旧日志兼容

`Waiting tokens` 和 `Waiting head tokens` 缺失时，默认行为应是：

- 打印 warning
- 保留 `GPU KV cache usage` 结果
- 不让脚本整体失败

只有在 `--strict` 打开时才报错退出。

### 9.2 同一秒多条记录

理论上同一 `timestamp + engine_id` 组合可能出现多条记录。

建议处理：

- 使用 `pivot_table(..., aggfunc="last")`
- 即保留该秒内最后一条

### 9.3 引擎子集分析

大规模 DP 运行时 engine 可能很多，因此建议支持：

- `--engine-ids 0-7`
- `--engine-ids 0,3,7,12`

这有助于只看特定 engine 组。

### 9.4 大日志性能

第一版设计采用单进程、逐行扫描即可。

原因：

- 日志解析主要是 I/O 绑定
- 只提取三类指标，单行处理成本很低
- 先快速字符串过滤，再执行正则，已经足够高效

若后续遇到超大日志，再考虑：

- 分块读取
- 并行解析
- 中间结果缓存


## 10. 推荐实现骨架

```python
def parse_args():
    ...

def parse_log(path: Path, year: int) -> pd.DataFrame:
    records = []
    for line_no, line in enumerate(path.open(...), start=1):
        if "Engine " not in line or "GPU KV cache usage:" not in line:
            continue

        header = HEADER_RE.search(line)
        if not header:
            continue

        waiting_tokens = extract_optional_int(WAITING_TOKENS_RE, line)
        waiting_head_tokens = extract_optional_int(WAITING_HEAD_TOKENS_RE, line)
        gpu_kv_usage = extract_required_float(GPU_KV_USAGE_RE, line)

        records.append(...)

    return pd.DataFrame(records)

def build_views(df: pd.DataFrame):
    ...

def render_aggregate(df_views, output_path: Path):
    ...

def render_per_engine(df_views, output_path: Path):
    ...

def main():
    ...
```


## 11. 验证方案
 
1. 新格式日志，例如 `/vllm/1.log` 

验证点：

1. 新日志能同时绘制三类指标
3. `engine_id` 过滤后曲线数量正确
4. 聚合图中的 token 总量与明细表求和一致
5. `GPU KV cache usage` 聚合使用 `mean/max`，没有出现错误求和


## 12. 结论

这个脚本最合适的落地方式是：

1. 复用参考脚本的“正则解析 + DataFrame + pivot + relative time + matplotlib”主流程
2. 将解析从“整行强绑定正则”改为“头部字段 + 指标独立提取”，增强对 logger 演进的鲁棒性
3. 默认输出两张图：
   - 一张看总体趋势
   - 一张看 per-engine 分布、

如果后续进入实现阶段，建议先完成 `aggregate` 视图，再补 `per_engine` 视图，因为前者更容易验证，也更适合第一轮分析。
