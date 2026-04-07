# `least_cache` / `least_batch` 实现审计报告（对照 `docs-ai-read/manual_multinode_poisson_dispatch_policy_plan.md`）

日期：2026-04-07  
基线路径：`/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180`（当前仓库 `/vllm`）

## 结论

`least_cache` 与 `least_batch` 已按计划在 vLLM 主链路中落地。当前实现**整体正确**，且默认策略行为仍保持为 `waiting_x4_plus_running`。  
未发现策略定义与核心路由结果不一致的实质性缺陷。

## 关键实现链路（按顺序）

- 策略类型与配置透传：  
  - `DataParallelDispatchPolicy` 与 `data_parallel_dispatch_policy` 默认值在 [vllm/config/parallel.py](/vllm/vllm/config/parallel.py:38-L42) 与 [vllm/config/parallel.py](/vllm/vllm/config/parallel.py:130-L133) 定义。  
  - CLI 参数 `--data-parallel-dispatch-policy` 在 [vllm/engine/arg_utils.py](/vllm/vllm/engine/arg_utils.py:911-L913) 注册，并在 `create_engine_config` 中写入 [vllm/engine/arg_utils.py](/vllm/vllm/engine/arg_utils.py:1778-L1781)。  
  - `DataParallelDispatchPolicy` 读数与参数名通过 `self.vllm_config.parallel_config.data_parallel_dispatch_policy` 下发到 DP LB 客户端。

- LB 快照来源与发布（核心链路）：  
  - Scheduler 新增 `get_num_free_kv_blocks()` 抽象，在 [vllm/v1/core/sched/interface.py](/vllm/v1/core/sched/interface.py:230-L232) 声明，并在 [vllm/v1/core/sched/scheduler.py](/vllm/v1/core/sched/scheduler.py:1855-L1860) 实现。  
  - SchedulerStats 增加 `free_kv_blocks` 字段，并在常规统计里填充到 [vllm/v1/core/sched/scheduler.py](/vllm/v1/core/sched/scheduler.py:2100).  
  - EngineCore 的 DP LB 发布改为 `running/waiting/free_kv_blocks` 快照比对与推送，在 [vllm/v1/engine/core.py](/vllm/v1/engine/core.py:1888-L1903)。

- Coordinator 与客户端共享三元载荷：  
  - Coordinator 将每个引擎 stats 维护为 `[waiting, running, free_kv_blocks]`，见 [vllm/v1/engine/coordinator.py](/vllm/v1/engine/coordinator.py:21-L24,116-119,419-L423)。  
  - 接收 `SchedulerStats` 后同步 `num_waiting_reqs`、`num_running_reqs` 与 `free_kv_blocks` 到三元结构（同文件 [vllm/v1/engine/coordinator.py:330-L367]）。  
  - 客户端同样使用三元索引并在构造列表时默认 `[0, 0, 0]`，见 [vllm/v1/engine/core_client.py](/vllm/v1/engine/core_client.py:67-L70,1215-L1217)。

- 策略路由决策：  
  - 排序函数在 [vllm/v1/engine/core_client.py](/vllm/v1/engine/core_client.py:1406-L1420)：
    - `waiting_x4_plus_running -> (waiting * 4 + running,)`
    - `least_batch -> (running, waiting)`
    - `least_cache -> (-free_kv_blocks, running, waiting)`
  - 与本地 pending 的 `waiting` 增量保留（`client_count` 粒度）避免 100ms 同步窗口内集中打点，见 [vllm/v1/engine/core_client.py](/vllm/v1/engine/core_client.py:1459-L1462)。

- benchmark 侧可配置与追踪：  
  - runner case 字段、CSV 字段、解析、分组与 artifact 命名加入 `dispatch_policy`：  
    - [benchmarks/manual_multinode_poisson_runner.py](/vllm/benchmarks/manual_multinode_poisson_runner.py:152-L157,158-L166,227,700-738,1530-1583,1071-L1102).  
  - plan 重放链路支持指定策略并写入生成 CSV：[benchmarks/manual_multinode_poisson_plan_from_history.py](/vllm/benchmarks/manual_multinode_poisson_plan_from_history.py:47-L55,67-L70,222-L231,251-L274).  
  - harness 记录真实策略到 run_meta：[vllm/benchmarks/offline_poisson_harness.py](/vllm/vllm/benchmarks/offline_poisson_harness.py:1183-L1194).

## 与计划文档的对齐度

| 计划项 | 当前状态 |
| --- | --- |
| 默认策略保持 `waiting_x4_plus_running` | 已实现 |
| 新增 `least_cache` / `least_batch` 入口与透传 | 已实现 |
| Coordinator/Client 三元 wire-format | 已实现（`[waiting, running, free_kv_blocks]`） |
| `least_batch` 使用 `(running, waiting)` | 已实现 |
| `least_cache` 使用剩余 KV blocks 优先 | 已实现 |
| 发布条件覆盖 `free_kv_blocks` 变化 | 已实现（快照完整比较） |
| plan 中不建议本地 free_kv_blocks 预扣减 | 已实现（仅 waiting 增量） |

## 发现的偏差与风险（非阻断）

1. 在 [vllm/v1/engine/coordinator.py](/vllm/v1/engine/coordinator.py:340-L367) 里对迟到 stats 仅告警后仍会按当前 payload 覆盖本地 stats。这会放大瞬时抖动，但属于兼容性较早的行为，不会导致策略定义错误。  
2. `free_kv_blocks` 为“快照值”，与“本地请求执行后真实消耗”存在偏差，属于设计可接受的权衡。  
3. 现网测试仍有已知 `PytestUnknownMarkWarning` 与 CUDA 环境告警，当前与本次策略无直接相关。

## 验证记录

- 直接运行：`pytest tests/v1/engine/test_dplb_dispatch_policy.py -q && pytest tests/benchmarks/test_manual_multinode_poisson_runner.py tests/benchmarks/test_manual_multinode_poisson_plan_from_history.py tests/v1/engine/test_engine_core_client.py -q`  
- 结果：`7 passed, 42 passed, 1 skipped, 0 failed`（见警告略）。  
- 关键回归点覆盖：`least_cache`/`least_batch` 排序、策略透传、case/csv 历史重放、LB stats free_kv 更新。

## 建议

若需继续收敛抖动风险，可在不改策略语义的前提下增加：在 coordinator 对 `step_counter/current_wave` 外层老化控制下做更严格的 out-of-order 屏蔽，但当前实现不影响 least_cache / least_batch 的正确性判定。
