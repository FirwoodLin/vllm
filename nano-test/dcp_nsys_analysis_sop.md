# DCP Nsight Systems 分析 SOP

## 目标

这份 SOP 用于分析 vLLM 的 `nsys` profile，并回答下面这个问题：

> 在 DCP 场景下，引入了多少通信开销？
>
> 当前默认口径：按源码执行顺序拆出
> `DCP 前置 query all_gather / 本地 MLA core / DCP 后置归并`

这份 SOP 主要面向以下常见场景：

- `vllm bench latency`
- `--attention-backend FLASHMLA`
- `-tp <N> -dcp <N>`
- `dcp_comm_backend=ag_rs`，或者未显式指定、走默认行为

相邻场景也可以复用，但 kernel 名称和归因口径可能需要调整。

## 产出要求

按这份 SOP 分析后，最终回答通常至少要包含：

1. 从 `nsys` 文件名提取出的运行配置；如果文件名缺字段，要明确标注哪些字段来自源码默认值推断，哪些字段未知。
2. DCP 前置通信时间：
   - `query all_gather`
3. 本地 MLA core 时间：
   - `flash_fwd_splitkv_mla_kernel`
   - `flash_fwd_mla_combine_kernel`
4. DCP 后置归并时间：
   - `ag_rs`：`lse all_gather + _correct_attn_cp_out_kernel + reduce_scatter`
   - `a2a`：`SendRecv(output) + SendRecv(lse) + _dcp_lse_combine_kernel`
5. 至少一个阶段占比：
   - `pre_share_pct`
   - `mla_core_share_pct`
   - `post_share_pct`
6. 结果口径说明：
   - 这是聚合 GPU kernel 时间，还是
   - 相对 `dcp=1` 基线的端到端时延增量

## 推荐输入

- `.nsys-rep` 或同名 `.sqlite` 文件
  - 推荐文件名直接编码关键信息：`dp / tp / dcp / backend / bs / inputlen / node`
- 与该 profile 对应的本地 vLLM 源码
- 如果用户只想看某类 replay：
  - 目标 NVTX label
  - 例如 `execute_context_0(0)_generation_256(256)`
- 可选补充：
  - 启动脚本
  - 只有在文件名缺字段，或者用户明确要追溯 `model / warmup / num-iters / output-len` 时再看

## 配套脚本

这份 SOP 配套了一个可复用脚本：

- `/vllm/nano-test/analyze_dcp_nsys.py`

这个脚本当前面向：

- MLA decode
- DCP `ag_rs` 和 `a2a`
- 按事件顺序拆分阶段，而不是只按 kernel 名聚合
- 关注：
  - 前置：`query all_gather`
  - 本地：`flash_fwd_splitkv_mla_kernel + flash_fwd_mla_combine_kernel`
  - 后置：
    - `ag_rs`：`lse all_gather + _correct_attn_cp_out_kernel + reduce_scatter`
    - `a2a`：`SendRecv(output) + SendRecv(lse) + _dcp_lse_combine_kernel`

它支持三种常用模式：

1. 全局模式
   - 不传 `--nvtx-label`
   - 统计整个 profile 内完整 DCP MLA cycle 的聚合 kernel 时间
2. replay 模式
   - 传 `--nvtx-label`
   - 只统计目标 NVTX replay 内发起的完整 DCP MLA cycle
3. 批量模式
   - 一次传多个 `.nsys-rep/.sqlite`
   - 用 `--csv` 输出一张汇总表
   - 同 basename 的 `.nsys-rep/.sqlite` 会自动去重，优先使用 `.sqlite`

如果输入是 `.nsys-rep`，脚本会优先复用同名 `.sqlite`；如果 `.sqlite` 不存在，会调用 `nsys stats` 触发导出。

## 文件名约定与解析原则

默认把 `nsys` 文件名当作这份 SOP 的主配置来源，不要先依赖 `start.sh`。

推荐命名：

```text
nsys_dp1_tp8_dcp8_ag_rs_bs256_inputlen4096_node0.nsys-rep
nsys_dp1_tp8_dcp8_a2a_bs256_inputlen4096_node0.sqlite
```

推荐从文件名直接提取这些字段：

- `dp`
- `tp`
- `dcp`
- `dcp_comm_backend`
- `batch size`
- `input length`
- `node rank`

推荐正则口径：

- `dp(?P<dp>\d+)`
- `tp(?P<tp>\d+)`
- `dcp(?P<dcp>\d+)`
- `(?P<backend>ag_rs|a2a)`
- `bs(?P<bs>\d+)`
- `inputlen(?P<input_len>\d+)`
- `node(?P<node_rank>\d+)`

兼容旧文件名时按下面规则处理：

- 如果文件名像 `nsys_dp1_tp8_dcp8_bs256_inputlen4096_node0.nsys-rep`，那 `backend` 视为“文件名缺失”。
- 对缺失的 `backend`，只有在你确认本地代码默认值未改时，才能把它标成“根据源码默认值推断为 `ag_rs`”。
- 对缺失的 `bs / inputlen / tp / dcp`，不要猜；直接标记为未知，除非用户另给证据。
- 不要把脚本中的值无提示地混成“从 profile 本身得出”的结论。

最小示例：

```bash
python3 - <<'PY'
import os
import re

path = "/path/to/nsys_dp1_tp8_dcp8_ag_rs_bs256_inputlen4096_node0.nsys-rep"
name = os.path.basename(path)
patterns = {
    "dp": r"dp(\d+)",
    "tp": r"tp(\d+)",
    "dcp": r"dcp(\d+)",
    "backend": r"(ag_rs|a2a)",
    "bs": r"bs(\d+)",
    "input_len": r"inputlen(\d+)",
    "node_rank": r"node(\d+)",
}
parsed = {}
for key, pattern in patterns.items():
    m = re.search(pattern, name)
    parsed[key] = m.group(1) if m else None
print(parsed)
PY
```

## 脚本快速用法

### 用法 1：先找目标 replay label

先列出可能的 NVTX label：

```bash
python3 /vllm/nano-test/analyze_dcp_nsys.py \
  /path/to/profile.nsys-rep \
  --list-nvtx generation_256
```

示例输出会类似：

```text
Matching NVTX labels
- execute_context_0(0)_generation_256(256): events=88, tids=8, total_us=590955.287
```

如果你已经知道完整 label，可以跳过这一步。

### 用法 2：只分析指定 replay

```bash
python3 /vllm/nano-test/analyze_dcp_nsys.py \
  /path/to/profile.nsys-rep \
  --nvtx-label 'execute_context_0(0)_generation_256(256)'
```

或者先列 label，再立刻分析：

```bash
python3 /vllm/nano-test/analyze_dcp_nsys.py \
  /path/to/profile.nsys-rep \
  --list-nvtx generation_256 \
  --nvtx-label 'execute_context_0(0)_generation_256(256)'
```

### 用法 3：分析整个 profile

```bash
python3 /vllm/nano-test/analyze_dcp_nsys.py \
  /path/to/profile.nsys-rep
```

这时输出口径等价于“整个 profile 的 kernel 聚合”，不会做 replay 过滤。

### 用法 4：批量分析并导出 CSV

```bash
python3 /vllm/nano-test/analyze_dcp_nsys.py \
  /path/to/nsys_dp1_tp8_dcp8_a2a_bs256_inputlen4096_node0.1.nsys-rep \
  /path/to/nsys_dp1_tp8_dcp8_a2a_bs256_inputlen4096_node0.1.sqlite \
  /path/to/nsys_dp1_tp8_dcp8_a2a_bs512_inputlen4096_node0.1.nsys-rep \
  /path/to/nsys_dp1_tp8_dcp8_ag_rs_bs256_inputlen4096_node0.1.nsys-rep \
  --csv /path/to/dcp_mla_stage_breakdown.csv
```

推荐直接把一批 profile 都传进去。脚本会：

- 逐个解析 `backend / bs / inputlen / node`
- 自动去重同 basename 的 `.nsys-rep/.sqlite`
- 每个 profile 输出一行 CSV

## 脚本输出怎么理解

脚本输出主要分 3 段：

1. 基本信息
   - `Scope: replay` 或 `Scope: global`
   - replay 模式下会显示：
     - `NVTX label`
     - `Matched NVTX events`
     - `Matched threads`
     - `Estimated replay count`
2. 阶段统计
   - `pre_query_all_gather_us`
   - `mla_splitkv_us`
   - `mla_combine_us`
   - `mla_core_us`
   - `ag_rs`：
     - `post_lse_all_gather_us`
     - `post_correct_attn_cp_out_us`
     - `post_reduce_scatter_us`
   - `a2a`：
     - `post_sendrecv_output_us`
     - `post_sendrecv_lse_us`
     - `post_dcp_lse_combine_us`
   - `post_total_us`
   - `dcp_stage_total_us`
3. 解析质量与阶段占比
   - `pre_share_pct`
   - `mla_core_share_pct`
   - `post_share_pct`
   - `parsed_cycle_count`
   - `matched_event_count`
   - `discarded_event_count`

这里的核心变化是：

- 脚本不再把所有 `AllGather` 或 `SendRecv` 简单相加后直接下结论
- 而是先按单卡时间顺序匹配完整 cycle，再把同名 NCCL kernel 分配到前置或后置阶段

## 第 1 步：先从 `.nsys` 文件名提取运行配置

优先检查 profile 文件名，而不是启动脚本。先提取以下信息：

- `tp`
- `dcp`
- `dp`
- `dcp_comm_backend`
- `batch size`
- `input length`
- `node rank`

示例：

```bash
basename /path/to/nsys_dp1_tp8_dcp8_ag_rs_bs256_inputlen4096_node0.nsys-rep
```

重点确认：

- 这是不是一个真正的 DCP case，例如 `dcp > 1`。
- 当前分析面对的是 `ag_rs` 还是 `a2a`。
- 当前 profile 文件名是否已经编码了目标 `bs` / `inputlen`。
- profile 里是否只包含一次正式测量，不要靠脚本假设；要靠 NVTX label、replay 次数或运行结果本身确认。

如果文件名没有编码 `backend`，再去代码里看默认值：

```bash
nl -ba /vllm/vllm/config/parallel.py | sed -n '300,314p'
```

当前代码默认值是：

- `dcp_comm_backend = "ag_rs"`

此时建议写成：

- “文件名未包含 `backend` 字段。”
- “基于当前本地源码默认值，推断该 profile 的 DCP backend 为 `ag_rs`。”

不要写成“文件名已经表明 backend 是 `ag_rs`”。

## 第 2 步：先在源码里确认 DCP attention 路径

在给 kernel 命名前，先确认代码路径，不要只凭 kernel 名字主观猜测。

对 MLA decode DCP，优先看：

```bash
nl -ba /vllm/vllm/model_executor/layers/attention/mla_attention.py | sed -n '682,708p'
nl -ba /vllm/vllm/v1/attention/ops/common.py | sed -n '170,235p'
nl -ba /vllm/vllm/v1/attention/ops/dcp_alltoall.py | sed -n '1,40p'
```

对于 `ag_rs` 路径，预期执行顺序是：

1. `query all_gather`
2. 本地 attention kernel
3. `lse all_gather`
4. `_correct_attn_cp_out_kernel`
5. `reduce_scatter`

对于 `a2a` 路径，预期执行顺序是：

1. `query all_gather`
2. 本地 attention kernel
3. `all_to_all` 交换 partial output
4. `all_to_all` 交换 LSE
5. 本地 `dcp_lse_combine_triton`

如果第 1 步从文件名里已经确认 `backend=a2a`，那后面所有 `ag_rs` 的分桶、kernel 名和比例口径都不能直接照搬。

这一步很关键。后面的 kernel 归因必须能回到代码上自洽。

## 第 3 步：导出或复用 SQLite

`nsys stats` 可以直接读 `.nsys-rep`，也可以读 `.sqlite`。
通常第一步实操是让 `nsys` 生成一份 SQLite，便于后续精确查询。

```bash
nsys stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  --output - \
  --timeunit us \
  /path/to/profile.nsys-rep
```

如果对应的 `/path/to/profile.sqlite` 不存在，通常会自动生成。

如果本机没有 `sqlite3` 命令行，直接使用 Python 自带的 `sqlite3` 模块。

如果只是执行这份 SOP 的标准分析，优先直接跑配套脚本即可：

```bash
python3 /vllm/nano-test/analyze_dcp_nsys.py /path/to/profile.nsys-rep
```

需要 replay 过滤时：

```bash
python3 /vllm/nano-test/analyze_dcp_nsys.py \
  /path/to/profile.nsys-rep \
  --nvtx-label 'execute_context_0(0)_generation_256(256)'
```

## 第 4 步：先看高层 kernel 汇总

第一手入口通常是 `cuda_gpu_kern_sum`。这一步能最快看出主要通信和主要 attention kernel。

```bash
nsys stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  --output - \
  --timeunit us \
  /path/to/profile.nsys-rep
```

重点关注：

- NCCL kernel
- attention kernel
- 其他可能被误认为 DCP 通信的 TP / MoE 通信 kernel

如果第 1 步判定 backend 是 `ag_rs`，常见归类如下：

- 通信：
  - `ncclDevKernel_AllGather_*`
  - `ncclDevKernel_ReduceScatter_*`
- attention 核心计算：
  - `flash_fwd_splitkv_mla_kernel`
  - `flash_fwd_mla_combine_kernel`
- DCP 本地后处理：
  - `_correct_attn_cp_out_kernel`

常见但不应默认算进 “DCP 通信” 的 kernel：

- `multimem_all_reduce_kernel`
- TP / MoE 的 all-reduce 或 fused communication kernel

除非用户明确要求“整个模型的总通信开销”，否则不要把这些混进 DCP attention 通信。

## 第 5 步：查看 SQLite schema

先列出有哪些表：

```bash
python3 - <<'PY'
import sqlite3
conn = sqlite3.connect("/path/to/profile.sqlite")
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
for (name,) in cur.fetchall():
    print(name)
PY
```

这类问题通常会用到：

- `CUPTI_ACTIVITY_KIND_KERNEL`
- `StringIds`
- `NVTX_EVENTS`

如果要看列结构：

```bash
python3 - <<'PY'
import sqlite3
conn = sqlite3.connect("/path/to/profile.sqlite")
cur = conn.cursor()
for table in ["CUPTI_ACTIVITY_KIND_KERNEL", "StringIds", "NVTX_EVENTS"]:
    print(f"-- {table} --")
    cur.execute(f"PRAGMA table_info({table})")
    for row in cur.fetchall():
        print(row)
PY
```

## 第 6 步：如果用户指定 replay，先按 NVTX 过滤再抽 kernel

如果用户明确说：

- 只分析正式运行
- 只分析 batch size = 256
- 只分析 `execute_context_0(0)_generation_256(256)` 这样的 replay

那就不要直接用全局 `cuda_gpu_kern_sum` 或全局 kernel 聚合。

正确口径应当是：

1. 先从 `NVTX_EVENTS` 找到目标 replay 对应的区间
2. 用 `globalTid + start/end` 过滤出这个 replay 内发起的 CUDA runtime launch
3. 再用 `correlationId` 关联到 `CUPTI_ACTIVITY_KIND_KERNEL`
4. 最后只对这批 kernel 做通信 / attention 分桶

推荐链路：

```text
NVTX_EVENTS
  -> CUPTI_ACTIVITY_KIND_RUNTIME
  -> CUPTI_ACTIVITY_KIND_KERNEL
```

不要只用 “kernel 时间窗与 NVTX 区间 overlap” 作为唯一条件。

原因：

- 不同 `execute_context_*` 可能并行重叠
- 单纯按时间 overlap 可能把别的 replay 的 kernel 误带进来
- 用 `runtime correlationId` 关联通常更干净

模板如下：

```bash
python3 - <<'PY'
import sqlite3
from collections import defaultdict

path = "/path/to/profile.sqlite"
conn = sqlite3.connect(path)
cur = conn.cursor()

target = "execute_context_0(0)_generation_256(256)"

nvtx_rows = list(cur.execute('''
SELECT start, end, globalTid
FROM NVTX_EVENTS
WHERE COALESCE(text, (SELECT value FROM StringIds WHERE id=NVTX_EVENTS.textId), jsonText)=?
ORDER BY globalTid, start
''', (target,)))

by_tid = defaultdict(list)
for start, end, tid in nvtx_rows:
    by_tid[tid].append((start, end))

all_tids = tuple(by_tid)
min_start = min(s for ivs in by_tid.values() for s, _ in ivs)
max_end = max(e for ivs in by_tid.values() for _, e in ivs)

runtime_q = f'''
SELECT start, end, globalTid, correlationId
FROM CUPTI_ACTIVITY_KIND_RUNTIME
WHERE globalTid IN ({",".join("?" * len(all_tids))})
  AND start >= ?
  AND end <= ?
  AND correlationId IS NOT NULL
'''
runtime_rows = list(cur.execute(runtime_q, list(all_tids) + [min_start, max_end]))

selected_corr = set()
for start, end, tid, corr in runtime_rows:
    for rs, re in by_tid[tid]:
        if start >= rs and end <= re:
            selected_corr.add(corr)
            break

kernel_q = f'''
SELECT s.value AS name,
       COUNT(*) AS instances,
       SUM((k.end-k.start)/1000.0) AS total_us,
       AVG((k.end-k.start)/1000.0) AS avg_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
WHERE k.correlationId IN ({",".join("?" * len(selected_corr))})
GROUP BY s.value
HAVING name LIKE '%ncclDevKernel%'
    OR name LIKE '%flash_fwd_splitkv_mla_kernel%'
    OR name LIKE '%flash_fwd_mla_combine_kernel%'
    OR name LIKE '%_correct_attn_cp_out_kernel%'
ORDER BY total_us DESC
'''

for row in cur.execute(kernel_q, tuple(selected_corr)):
    print('\t'.join(str(x) for x in row))
PY
```

如果用户没有指定 replay，或者明确要整个 profile 的聚合结果，再退回全局 kernel 聚合。

优先建议直接用配套脚本，而不是每次手写 SQL：

```bash
python3 /vllm/nano-test/analyze_dcp_nsys.py \
  /path/to/profile.nsys-rep \
  --nvtx-label 'execute_context_0(0)_generation_256(256)'
```

只有在下面这些场景下，才需要回到手写 SQL：

- 需要改 kernel 分桶
- 需要分析脚本尚未覆盖的新 backend
- 需要做更细的阶段拆分
- 需要额外验证脚本结果

## 第 7 步：精确抽取目标 kernel 及总时长

如果没有 replay 过滤需求，或者你已经在第 6 步拿到了目标 replay 内的 kernel 集合，就进入这一步做聚合。

全局模板如下：

```bash
python3 - <<'PY'
import sqlite3
path = "/path/to/profile.sqlite"
conn = sqlite3.connect(path)
cur = conn.cursor()
q = '''
SELECT s.value AS name,
       COUNT(*) AS instances,
       SUM((k.end-k.start)/1000.0) AS total_us,
       AVG((k.end-k.start)/1000.0) AS avg_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
GROUP BY s.value
HAVING name LIKE '%ncclDevKernel%'
    OR name LIKE '%flash_fwd_splitkv_mla_kernel%'
    OR name LIKE '%flash_fwd_mla_combine_kernel%'
    OR name LIKE '%_correct_attn_cp_out_kernel%'
ORDER BY total_us DESC
'''
for row in cur.execute(q):
    print('\t'.join(str(x) for x in row))
PY
```

## 第 8 步：把时间拆成 3 个阶段

默认按完整 DCP MLA cycle 拆成下面 3 段。

### 阶段 A：DCP 前置通信

两种 backend 都有：

- `query all_gather`

### 阶段 B：本地 MLA core

两种 backend 都有：

- `flash_fwd_splitkv_mla_kernel`
- `flash_fwd_mla_combine_kernel`

### 阶段 C：DCP 后置归并

如果是 `ag_rs`：

- `lse all_gather`
- `_correct_attn_cp_out_kernel`
- `reduce_scatter`

如果是 `a2a`：

- `SendRecv(output)`
- `SendRecv(lse)`
- `_dcp_lse_combine_kernel`

## 第 9 步：按事件顺序拆同名 NCCL kernel

这一步是现在脚本的关键。

因为：

- `query all_gather` 和 `lse all_gather` 在 `ag_rs` 下经常是同一个 kernel 名
- `SendRecv(output)` 和 `SendRecv(lse)` 在 `a2a` 下也是同一个 kernel 名

所以不能只靠 kernel 名做分桶，必须按单卡时间顺序拆。

脚本默认匹配的重复模式是：

- `ag_rs`

```text
AG -> splitkv -> combine -> AG -> correct -> RS
```

- `a2a`

```text
AG -> splitkv -> combine -> SendRecv -> SendRecv -> dcp_lse_combine
```

## 第 10 步：如果需要，先手工验证单卡时间线

如果你怀疑 profile 里存在额外的 NCCL kernel，或者脚本的 `discarded_event_count` 明显偏大，可以先手工看单卡顺序：

```bash
python3 - <<'PY'
import sqlite3
path = "/path/to/profile.sqlite"
conn = sqlite3.connect(path)
cur = conn.cursor()
q = '''
SELECT ROUND(k.start/1000.0,3) AS start_us,
       ROUND((k.end-k.start)/1000.0,3) AS dur_us,
       k.deviceId,
       s.value AS name
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
WHERE k.deviceId = 0 AND (
    s.value LIKE '%ncclDevKernel%'
    OR s.value LIKE '%flash_fwd_splitkv_mla_kernel%'
    OR s.value LIKE '%flash_fwd_mla_combine_kernel%'
    OR s.value LIKE '%_correct_attn_cp_out_kernel%'
    OR s.value LIKE '%_dcp_lse_combine_kernel%'
)
ORDER BY k.start
LIMIT 40
'''
for row in cur.execute(q):
    print('\t'.join(map(str, row)))
PY
```

如果顺序和源码预期一致，再用脚本批量跑即可，不需要手写 SQL 重新累加。

## 第 11 步：NVTX 的使用边界要说清楚

NVTX 可以看，但要分两种用法：

- 用来筛选用户指定 replay
- 用来直接做时间统计

前者通常是推荐做法；后者不能替代 kernel 归因。

原因：

- 有些 profile 里 NVTX string table 不完整
- NVTX 名字可能过粗
- 对“通信 / attention”这种问题，kernel 级别归因通常更可靠
- 如果只用 NVTX 区间与 kernel 做时间 overlap，可能会把并行 replay 混进来

推荐表达：

- “本次先用 NVTX label 过滤出目标 replay，再基于 runtime correlationId 关联到 GPU kernel。”
- “最终统计仍然以 kernel 时长为准，不直接使用 NVTX 区间时长作为通信 / attention 时间。”

可选 NVTX 查询：

```bash
python3 - <<'PY'
import sqlite3
path = "/path/to/profile.sqlite"
conn = sqlite3.connect(path)
cur = conn.cursor()
q = '''
SELECT COALESCE(n.text, s.value) AS name,
       COUNT(*) AS cnt,
       SUM(CASE WHEN n.end IS NOT NULL THEN (n.end-n.start)/1000.0 ELSE 0 END) AS total_us
FROM NVTX_EVENTS n
LEFT JOIN StringIds s ON n.textId = s.id
GROUP BY name
ORDER BY cnt DESC
LIMIT 20
'''
for row in cur.execute(q):
    print(row)
PY
```

## 第 12 步：最终回答里必须写清楚分母口径

一定要明确说明 attention 时间到底怎么定义。

推荐表达：

- “`通信 / attention core` 的分母只包含 `flash_fwd_splitkv_mla_kernel + flash_fwd_mla_combine_kernel`。”
- “`通信 / attention（含 correction）` 的分母额外包含 `_correct_attn_cp_out_kernel`。”

这一步是为了避免歧义。

## 第 13 步：最终回答里必须写清楚 caveat

除非用户已经限定了别的统计口径，否则默认带上下面这些 caveat。

### Caveat 1：这是聚合 GPU kernel 时间，不是 wall time

`cuda_gpu_kern_sum` 给出的是所有 GPU、所有 kernel launch 聚合后的总时长。
它不等于端到端 wall-clock latency 的直接增量。

如果这次做过 replay 过滤，需要改写成：

- “这是目标 replay 内的聚合 GPU kernel 时间，不是该 replay 的 wall-clock latency。”

### Caveat 2：如果要衡量 DCP 真正多出来多少延迟，必须有基线

如果用户问的是：

> DCP 相比不开 DCP 多了多少延迟？

那就必须拿同配置的 `dcp=1` 结果做对照，并保证下面这些条件一致：

- model
- batch size
- input length
- output length
- TP size
- attention backend

### Caveat 3：不要误把非 DCP 通信算进去

大模型里经常同时存在 TP / MoE / expert routing 通信。
如果问题只是在问 DCP attention 开销，就不要默认把这些 kernel 也计入通信。

### Caveat 4：`a2a` backend 不能照搬 `ag_rs` 口径

如果启用了 `--dcp-comm-backend a2a`，通信模式和 kernel 名都可能变化。
不能直接复用 `ag_rs` 的 kernel 归因。

## 下次分析时的输入清单

当用户再次给出 profile 时，优先补齐这些输入：

- `.nsys-rep` 路径
- 文件名中最好直接带：
  - `dp`
  - `tp`
  - `dcp`
  - `backend`
  - `bs`
  - `inputlen`
  - `node`
- 如果只看特定 replay：
  - replay 的 NVTX label
- 用户希望的分析粒度：
  - 只要 `DCP 通信 / attention` 比值
  - 需要通信子阶段拆分
  - 需要和 `dcp=1` 做对比
  - 需要按 layer 还是整次运行聚合
- 可选补充：
  - 启动脚本路径
  - 只在文件名缺字段，或者用户追问 `model / warmup / num-iters / output-len` 时需要

## 推荐执行顺序

下次按下面顺序展开即可：

1. 先解析 `.nsys` 文件名
2. 对照源码确认代码路径
3. 导出或复用 SQLite
4. 如果用户指定 replay，先按 NVTX replay 过滤
5. 看 `cuda_gpu_kern_sum` 或跑 replay 内聚焦 SQL
6. 计算比值
7. 写 caveat

更推荐的简化版顺序：

1. 先解析 `.nsys` 文件名
2. 对照源码确认代码路径
3. 跑 `analyze_dcp_nsys.py`
4. 必要时再回到 SQL 验证边角问题
5. 写 caveat

## 建议的最终回答模板

```text
这个 profile 的文件名可解析为：
- `dp=<...>, tp=<...>, dcp=<...>, backend=<...>, bs=<...>, inputlen=<...>, node=<...>`

如果某些字段不在文件名里，需要额外标注：
- 这是“根据源码默认值推断”
或
- 这是“未知”

代码路径上：
- 如果 `backend=ag_rs`，DCP attention 的通信模式是：query all_gather -> local attention -> lse all_gather -> local correction -> reduce_scatter
- 如果 `backend=a2a`，通信模式是：local attention -> all_to_all(output) -> all_to_all(lse) -> local lse combine

本次统计口径：
- 只统计 NVTX label = `<...>` 对应 replay 内发起的 kernel
或
- 统计整个 profile 的聚合 kernel 时间

按 Nsight 的 kernel 聚合时间统计：
- 通信时间 = ...
- attention core 时间 = ...
- attention（含 correction）时间 = ...

主结果：
- 通信 / attention core = ...%
- 通信 / attention（含 correction） = ...%
- 通信占整个 DCP attention 阶段 = ...%

口径说明：
- 以上结果是聚合 GPU kernel 时间，不是端到端 latency 增量
- 如果需要 DCP 真正引入的额外时延，需要再拿同配置 `dcp=1` 的 profile 做对照
```
