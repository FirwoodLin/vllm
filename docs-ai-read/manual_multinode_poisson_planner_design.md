# Manual Multi-node Poisson Planner Design

Last updated: 2026-04-04
Repo basis: local `/vllm` checkout used in this session

## 1. Goal

为手工多机 Poisson benchmark 增加一个独立的 planner 脚本，用来：

- 读取历史实验结果
- 根据命令行参数筛选目标模型和场景
- 生成“合理的倒序 rate 执行计划”
- 输出为可人工编辑的 CSV

随后由现有 runner 读取该 CSV 执行实验。

本阶段目标是形成明确实施方案，不修改代码。

## 2. Scope

本设计包含两部分：

1. 新增 planner 脚本
2. 为 [`benchmarks/manual_multinode_poisson_runner.py`](/vllm/benchmarks/manual_multinode_poisson_runner.py) 增加 `--case-csv` 输入能力

本设计不包含：

- 改写现有实验 artifact 目录结构
- 移除 runner 现有的内置 case/matrix 能力
- 增加新的 benchmark 指标定义

## 3. Working Model

目标工作流改成两步：

1. planner 根据历史结果生成本次计划 CSV
2. runner 读取 CSV，按给定顺序执行

这样做的目的：

- 历史推导逻辑与执行逻辑解耦
- CSV 便于人工检查和微调
- runner 仍可保留已有 skip/去重保护

## 4. Planner Inputs

建议新增脚本：

- [`benchmarks/manual_multinode_poisson_plan_from_history.py`](/vllm/benchmarks/manual_multinode_poisson_plan_from_history.py)

建议支持以下参数：

- `--artifact-root`
- `--model`
- `--dataset`
- `--strategy`，可重复；默认全部策略
- `--rate-plan`
- `--output-csv`
- `--historical-skip-ignore-bs`
- `--bench-duration-sec`，可选

第一阶段默认面向当前实际需求：

- `model = kimi_k2_instruct_0905`
- `dataset = issue01_random`
- `rate-plan = coarse10_then_mid5`

但参数设计保持泛化，避免后续再拆一次。

## 5. Rate Planning Rules

planner 对每个 `(model, dataset, strategy)` 单独生成倒序 rate 列表。

### 5.1 Candidate rates

先根据 `rate-plan` 生成候选 rate 集合：

- `coarse10` 对应 `10,20,...,90`
- `coarse10_then_mid5` 对应 `10,15,20,...,90`

planner 内部使用去重后的有序 rate 列表。

### 5.2 Historical start-rate selection

对每个 group，按以下优先级决定起跑 rate：

1. 查找历史成功结果中 `tpot_by_e2e.mean >= 100ms` 的最小 rate
2. 若存在，则起跑 rate 取“严格小于该 rate 的最近候选 rate”
3. 若不存在，则查找历史中 `status == timed_out` 的最小 rate
4. 若存在，则起跑 rate 取比该 timed_out rate 小的最近候选 rate
5. 若仍不存在，则起跑 rate 取当前 plan 的最大候选 rate

例子：

- 若最小超阈值成功 rate 是 `40`，则起跑 rate 为 `35`
- 若最小超阈值成功 rate 是 `35`，则起跑 rate 为 `30`
- 若没有超阈值成功记录，但最小 timed_out rate 是 `40`，则起跑 rate 为 `35`
- 若完全没有相关历史，则从 `5` 开始（正序去跑）

### 5.3 Output ordering

一旦确定起跑 rate，planner 输出：

- 从起跑 rate 到最小候选 rate
- 按 rate 从大到小排序

即 planner 只负责“生成本次建议执行列表”，不负责运行时 early-stop。

## 6. Historical Data Sources

planner 需要读取两类历史信息：

1. 成功 benchmark 的 `summary.json`
2. case 级别的 `case_manifest.json`

使用目的：

- 从 `summary.json` 提取 `tpot_by_e2e.mean`
- 从 `case_manifest.json` 提取 `status`，特别是 `timed_out`

历史匹配语义建议与 runner 保持一致：

- 默认按整个 `artifact-root` 扫描
- 默认支持 `historical_skip_ignore_bs`

也就是：

- 可将仅 `bs/max_num_seqs` 不同的历史目录视为同一场景的可参考结果

## 7. CSV Schema

CSV 需要既能被 runner 直接消费，也便于人工调整。

建议至少包含以下字段：

- `enabled`
- `name`
- `cluster`
- `model`
- `dataset`
- `strategy`
- `request_rate`
- `rate_phase`
- `max_num_seqs`
- `gpu_memory_utilization`
- `max_requests`
- `warmup_requests`
- `max_model_len`
- `data_parallel_rpc_port`

建议增加以下审计辅助字段：

- `reason`
- `historical_reference`

字段语义：

- `enabled`: `1` 表示执行，`0` 表示忽略该行
- `reason`: 记录该 rate 被纳入计划的原因
- `historical_reference`: 记录命中的历史 case 或目录

`reason` 示例：

- `start_below_tpot100_at_rate40`
- `start_from_timeout_rate35`
- `start_from_max_rate_no_history`

## 8. Runner Integration

需要为 [`benchmarks/manual_multinode_poisson_runner.py`](/vllm/benchmarks/manual_multinode_poisson_runner.py) 增加：

- `--case-csv <path>`

约束：

- `--case-csv` 与 `--all` / `--case` 互斥

行为：

1. 若指定 `--case-csv`，runner 从 CSV 读取 case
2. 按 CSV 行顺序执行
3. 忽略 `enabled=0` 的行
4. `--list` 在 CSV 模式下打印 CSV 中解析出的 case

CSV 模式下，runner 不再依赖内置 matrix 生成顺序，但仍保留已有保护逻辑。

## 9. Runner Logic To Keep

即使改为 CSV 驱动，runner 仍应保留现有保护：

- 精确 case 已成功则 skip
- 历史成功且 `tpot_by_e2e.mean` 已超过阈值时可 skip
- 历史或运行中已出现 `timed_out` / 失败时可 skip

原因：

- planner 负责生成“合理初始计划”
- runner 负责避免重复实验和无意义继续执行

也就是说：

- planner 是离线计划层
- runner 是在线执行保护层

## 10. Reuse Strategy

优先复用 runner 中已经存在的辅助逻辑，避免 planner 和 runner 各维护一套历史解释规则。

建议优先复用：

- `tpot_by_e2e.mean` 提取逻辑
- benchmark 成功判定逻辑
- case group 匹配逻辑
- `ignore_bs` 匹配逻辑

planner 新增的核心能力主要是：

- 枚举某个 group 的全部历史 rate 及状态
- 计算建议起跑 rate
- 写出 CSV

## 11. Implementation Steps

建议按下面顺序实施：

1. 抽出 runner 内部可复用的历史读取辅助函数
2. 新增 planner 脚本，先支持当前目标场景
3. 定义并固定 CSV schema
4. 给 runner 增加 `--case-csv`
5. 在 runner 中实现 CSV -> `ExperimentCase` 的映射
6. 用 `--list` 和 `--dry-run` 验证 CSV 顺序和参数
7. 再验证 skip 逻辑与 CSV 模式兼容

## 12. Example Workflow

先生成计划：

```bash
python3 /vllm/benchmarks/manual_multinode_poisson_plan_from_history.py \
  --artifact-root /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench/manual_multinode \
  --model kimi_k2_instruct_0905 \
  --dataset issue01_random \
  --rate-plan coarse10_then_mid5 \
  --output-csv /tmp/kimi_issue01_resume_plan.csv
```

再执行计划：

```bash
python3 /vllm/benchmarks/manual_multinode_poisson_runner.py \
  --case-csv /tmp/kimi_issue01_resume_plan.csv \
  --artifact-root /mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/offline_bench/manual_multinode \
  --run-label issue01-random-kimi-ds3-resume
```

## 13. Acceptance Criteria

实施完成后，应满足：

1. 能基于历史结果生成仅包含目标模型/场景的倒序 rate CSV
2. 每个 strategy 的起跑 rate 符合本设计的优先级规则
3. CSV 支持人工编辑后再交给 runner 执行
4. runner 能按 CSV 顺序执行实验
5. runner 的历史 skip/去重保护在 CSV 模式下继续有效

