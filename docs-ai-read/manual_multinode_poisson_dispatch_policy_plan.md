# `manual_multinode_poisson_runner.py` 新增 `LeastCache` / `LeastBatch` 调度算法实施计划（已落地）

## 0. 状态说明

这份文档最初按“实施计划”写成；其中 `## 2` 的“现状梳理”描述的是改动前状态，不是当前主线现状。

截至 2026-04-06，这份计划已经按主体方案落地到代码中，核心实现位于：

1. [benchmarks/manual_multinode_poisson_runner.py](/vllm/benchmarks/manual_multinode_poisson_runner.py)
2. [offline_poisson_harness.py](/vllm/vllm/benchmarks/offline_poisson_harness.py)
3. [parallel.py](/vllm/vllm/config/parallel.py)
4. [arg_utils.py](/vllm/vllm/engine/arg_utils.py)
5. [core.py](/vllm/vllm/v1/engine/core.py)
6. [coordinator.py](/vllm/vllm/v1/engine/coordinator.py)
7. [core_client.py](/vllm/vllm/v1/engine/core_client.py)

## 1. 目标

当前 benchmark 路径里的 DP 负载均衡策略实际是固定写死的 `waiting * 4 + running`。

这次改动的目标是：

1. 保留现有默认策略，避免回归。
2. 新增 `LeastCache`。
3. 新增 `LeastBatch`。
4. 让 `benchmarks/manual_multinode_poisson_runner.py` 能显式选择策略，而不是只能跑默认值。

建议最终对外暴露的策略名使用稳定的 snake_case：

1. `waiting_x4_plus_running`
2. `least_cache`
3. `least_batch`

## 2. 改动前现状梳理

从当前代码看，真正的调度决策不在 runner 本身，而在 v1 engine 的内部 DP LB 路径：

1. [benchmarks/manual_multinode_poisson_runner.py](/vllm/benchmarks/manual_multinode_poisson_runner.py)
   目前没有 `dispatch_policy` 字段，也没有对应透传逻辑。
2. [offline_poisson_harness.py](/vllm/vllm/benchmarks/offline_poisson_harness.py:1193)
   `run_meta.json` 里把 `dispatch_policy` 固定写成了 `"waiting_x4_plus_running"`，这只是元数据，不是可配置行为。
3. [offline_poisson_harness.py](/vllm/vllm/benchmarks/offline_poisson_harness.py:1441)
   benchmark parser 自己没有策略字段；真正的 CLI 入口是 `AsyncEngineArgs.add_cli_args(...)`，而当前 `EngineArgs` / `ParallelConfig` 也还没有对应配置。
4. [core_client.py](/vllm/vllm/v1/engine/core_client.py:1416)
   真实选 DP 实例的逻辑是 `score = waiting * 4 + running`。
5. [coordinator.py](/vllm/vllm/v1/engine/coordinator.py:115)
   coordinator 目前只维护 `[waiting, running]` 两个统计。
6. [stats.py](/vllm/vllm/v1/metrics/stats.py:172)
   `SchedulerStats` 里已有 `kv_cache_usage` 这类容量相关指标，但没有 `free_kv_blocks`，coordinator 也没有把这类信息用于 DP 负载均衡。
7. [block_pool.py](/vllm/vllm/v1/core/block_pool.py:479)
   block pool 已有 `get_num_free_blocks()`，说明“剩余 KV Cache Block”在 engine 内部是可取到的。
8. [core.py](/vllm/vllm/v1/engine/core.py:1888)
   发给 coordinator 的 DP LB 统计不是走常规 metrics logger 路径，而是 `EngineCore._maybe_publish_request_counts()` 这条专用瘦路径；当前它只看 waiting/running 是否变化。

结论：

1. 不能只改 runner。
2. 不能只改 `SchedulerStats` 字段定义或 `scheduler.make_stats()`，因为 DP LB 实际用的是 `EngineCore` 里的专用发布路径。
3. 至少要改 runner、engine config 透传、engine -> coordinator 的 LB stats 发布路径、coordinator/client 的 wire format，以及 core client 选路逻辑。

## 3. 策略定义

### 3.1 `waiting_x4_plus_running`

保持现有语义不变：

`score = waiting * 4 + running`

选择 score 最小的 DP 实例。

### 3.2 `least_batch`

语义：

1. 优先选择 `running` 最少的 DP 实例。
2. 如果 `running` 相同，优先 `waiting` 更少的实例。
3. 如果仍然相同，保留现有 `eng_start_index` 的轮转起点，避免固定偏向低 rank。

建议排序键：

`(running, waiting)`

原因：

1. 用户要求的是“running 最少”。
2. 如果只看 `running`，在 coordinator 100ms 更新间隔内，短时间突发请求会集中打到同一个 engine。
3. 把 `waiting` 作为次级键，可以复用现有“本地先加 waiting”的抗抖动逻辑。

如果比较键完全相同，不把 `rotated_rank_order` 显式塞进排序 tuple，而是继续沿用当前“从 `eng_start_index` 开始扫描”的逻辑做 tie-break。

### 3.3 `least_cache`

语义：

1. 优先选择“剩余 KV Cache Block 最多”的 DP 实例。
2. 如果 cache 剩余相同，优先 `running` 更少。
3. 如果仍然相同，再比较 `waiting`。
4. 最后保留轮转起点打散。

建议排序键：

`(-free_kv_blocks, running, waiting)`

不建议直接只使用 `kv_cache_usage` 作为最终实现口径，原因是：

1. `kv_cache_usage` 在“所有 DP 容量完全一致”时与 `free_kv_blocks` 等价。
2. 但用户定义的是“剩余 KV Cache Block 最多”，用 `free_kv_blocks` 更直接，也更不依赖未来容量一定一致。

## 4. 推荐的数据结构改造

当前 coordinator 和 client 之间传的是无类型 `msgspec.msgpack.encode/decode` 的 `list[list[int]]`，语义是 `[waiting, running]`。client 和 coordinator 两端都直接按下标读写。

这里需要区分“代码内表示”和“wire format”：

1. 普通 `dataclass` 不能直接替换当前 wire format。因为这条链路没有 typed decoder，decode 后拿到的仍然会是 `list` / `dict`，不会自动还原成 dataclass 实例。
2. 这次改动的最小可行方案，是先把 wire format 升级成固定字段顺序的三元 list：
   `[waiting, running, free_kv_blocks]`
3. 如果想减少魔法下标，可以在 coordinator / client 内部局部引入常量或 helper accessor；但不建议在这次需求里顺手把整条消息链路重构成普通 dataclass。
4. 如果后续确实要把 wire format 也升级成具名结构，建议使用 `msgspec.Struct` 并在 encoder / decoder 两端同时改 typed schema；这属于更大范围重构，不建议和这次策略改动绑定。

推荐原因：

1. `least_cache` 需要第三个指标。
2. 固定三元 list 的改动面最小，能和现有 coordinator/client slicing、elastic scale-up 逻辑兼容。
3. 后续如果还想加 `waiting_total_tokens`、`kv_cache_usage`、`encoder_cache_usage`，可以再单独做一次 typed-wire-format 重构，而不是在这次策略改动里同时扩大范围。

## 5. 配置透传方案

### 5.1 runner 层

在 [manual_multinode_poisson_runner.py](/vllm/benchmarks/manual_multinode_poisson_runner.py) 的 `ExperimentCase` 增加字段：

```python
dispatch_policy: str = "waiting_x4_plus_running"
```

同时更新：

1. `CASE_CSV_FIELDNAMES`，把 `dispatch_policy` 写入 case 表。
2. `load_cases_from_csv(...)`，让 `--case-csv` 路径也能解析这个字段。
3. case manifest / run metadata 的序列化逻辑。
4. 生成命令行时，统一在 `build_common_harness_argv(...)` 里透传
   `--data-parallel-dispatch-policy <value>`，不要只在 frontend 命令单独拼。
5. 为避免不同策略的历史结果互相污染，非默认策略还应进入
   `case_group_key(...)` / artifact scenario prefix；默认策略保持原有
   路径命名，尽量不破坏既有目录结构。

原因：

1. 这比把策略偷偷塞进 `frontend_extra_args` 更可维护。
2. case 层明确有字段后，批量 sweep 和结果分析时能直接按策略聚合。
3. 如果历史 skip / artifact 分组不区分策略，`least_cache` / `least_batch`
   的历史结果会错误影响默认策略，反之亦然。

### 5.2 harness 层

这里不建议额外加一个 benchmark 私有的 frontend-only `--dispatch-policy`。

原因是：

1. `frontend` 和 `headless-engine` 两个 subparser 都已经调用了 `AsyncEngineArgs.add_cli_args(...)`。
2. `AsyncEngineArgs.from_cli_args(...)` 是按字段同名从 `argparse.Namespace` 里取值。
3. 因此正确的做法是把参数定义放到 `EngineArgs` / `AsyncEngineArgs`，让两个角色共享同一套 engine config CLI。

benchmark harness 自己只需要做两件事：

1. 把 `run_meta.json` 里的 `dispatch_policy` 从硬编码改为记录 `args.data_parallel_dispatch_policy`。
2. 保持对外元数据 key 仍然叫 `dispatch_policy`，方便 benchmark 产物阅读和聚合。

### 5.3 engine config 层

不要把这个配置只停留在 benchmark harness 内部，建议一路进入 engine config：

1. [arg_utils.py](/vllm/vllm/engine/arg_utils.py)
   新增 engine arg 字段 `data_parallel_dispatch_policy`。建议直接用
   `Literal["waiting_x4_plus_running", "least_cache", "least_batch"]`
   约束取值。
2. [parallel.py](/vllm/vllm/config/parallel.py)
   在 `ParallelConfig` 中增加同名字段，默认 `waiting_x4_plus_running`。
3. `AsyncEngineArgs.add_cli_args(...)`
   增加 CLI 参数 `--data-parallel-dispatch-policy`。
4. `AsyncEngineArgs.create_engine_config(...)`
   把字段写入 `ParallelConfig`。
5. 如果将来确实还想保留短别名 `--dispatch-policy`，必须显式把
   `dest` 指到 `data_parallel_dispatch_policy`，并避免重复注册同一参数。
   本次实现不建议引入这个别名。

原因：

1. 实际决策逻辑发生在 `DPLBAsyncMPClient`。
2. 让策略成为 engine config 的一部分，比 benchmark 私有参数更自然。
3. 后续如果在线服务也想复用，会更顺。

## 6. 统计链路改造

### 6.1 先修正 engine -> coordinator 的 DP LB 发布路径

`least_cache` 依赖的不是常规 metrics logger 路径，而是
`EngineCore._maybe_publish_request_counts()` 这条专门给 DP LB 用的瘦路径。

当前这段逻辑有三个限制：

1. 只取 `scheduler.get_request_counts()`。
2. 只在 waiting/running 变化时发布。
3. 手工构造 `SchedulerStats(...)`，并没有自动复用 `scheduler.make_stats()`。

因此必须一并改造，建议：

1. 把 `_maybe_publish_request_counts()` 重命名为 `_maybe_publish_lb_stats()`。
2. 在这里直接构造 LB snapshot，例如 `[waiting, running, free_kv_blocks]`。
3. 只要 snapshot 任一字段变化，就发布新的 `SchedulerStats(...)`。
4. 把 `self.last_counts` 升级为 `self.last_lb_snapshot`。
5. 构造 `SchedulerStats(...)` 时改用关键字参数，不再依赖位置参数，避免字段顺序耦合。

补充说明：

1. 实际落地里，`coordinator/client` 之间的 wire format 仍然是
   `[waiting, running, free_kv_blocks]`。
2. `EngineCore` 内部用于“是否变化”的 `last_lb_snapshot` 采用的是
   `(running, waiting, free_kv_blocks)` tuple；它只用于本地比较，不对外暴露。

### 6.2 SchedulerStats 增加 free block 指标

在 [stats.py](/vllm/vllm/v1/metrics/stats.py:172) 的 `SchedulerStats` 新增：

```python
free_kv_blocks: int = 0
```

填充值建议来自 scheduler 内部的 KV cache manager / block pool：

1. scheduler 已持有 `self.kv_cache_manager`。
2. block pool 已提供 `get_num_free_blocks()`。
3. 实际落地中，还把 `SchedulerInterface` 扩展出了
   `get_num_free_kv_blocks()`，由 `Scheduler` 实现，避免 `EngineCore`
   直接穿透 scheduler 内部结构。

实现上建议在两个地方都填这个字段：

1. `scheduler.make_stats()` 这条常规 metrics 路径。
2. `EngineCore._maybe_publish_lb_stats()` 这条 DP LB 专用发布路径。

只改第 1 条不够，因为 coordinator 实际收到的是第 2 条路径发出的 `SchedulerStats`。

### 6.3 coordinator / client 扩展聚合对象

在 [coordinator.py](/vllm/vllm/v1/engine/coordinator.py)：

1. 把 `EngineState.request_counts = [0, 0]` 升级成三元 list，例如
   `lb_stats = [0, 0, 0]  # [waiting, running, free_kv_blocks]`。
2. 订阅 engine stats 时，除了 `num_waiting_reqs`、`num_running_reqs`，再同步 `free_kv_blocks`。
3. `_get_engine_counts()` 最好重命名为 `_get_engine_lb_stats()`，避免“counts”这个名字继续掩盖第三个维度。
4. `DPLBAsyncMPClient.lb_engines` 的默认值和 elastic scale-up 初始化值，也要同步从 `[0, 0]` 升级为 `[0, 0, 0]`。

建议这里顺手把变量名从 `request_counts` 改成 `lb_stats` 或 `engine_lb_stats`，避免语义继续漂移；但本次 wire format 仍然建议保持三元 list，而不是普通 dataclass。

## 7. DPLB 选路逻辑改造

在 [core_client.py](/vllm/vllm/v1/engine/core_client.py:1366) 的 `DPLBAsyncMPClient` 中做三件事。

### 7.1 读取策略配置

从 `self.vllm_config.parallel_config` 读取 `data_parallel_dispatch_policy`。

### 7.2 用统一函数计算排序键

把当前写死的：

```python
score = waiting * 4 + running
```

替换成统一入口，例如：

```python
def _lb_sort_key(stats, policy):
    ...
```

建议：

1. `waiting_x4_plus_running` 返回 `(waiting * 4 + running,)`
2. `least_batch` 返回 `(running, waiting)`
3. `least_cache` 返回 `(-free_kv_blocks, running, waiting)`

当排序键完全相同时，继续保留当前按 `eng_start_index` 开始扫描的逻辑做 tie-break，不把 rotated rank 再显式编码进排序 tuple。

### 7.3 保留“本地乐观 waiting 增量”

当前逻辑在选中 engine 后会做：

```python
current_counts[eng_index][0] += self.client_count
```

这个机制不应该删掉，只是需要迁移到新 stats 结构上。

原因：

1. coordinator 更新有 100ms 级延迟。
2. 如果不做本地乐观更新，连续突发请求会在 stats 刷新前全部命中同一个 engine。

对 `least_cache` 不建议做“本地 free_kv_blocks 预扣减”，原因是：

1. 单个请求最终吃掉多少 KV blocks 与 prompt 长度、执行进度有关。
2. 预扣减容易引入系统性偏差。
3. 更稳妥的做法是只保留 waiting 的本地影子增量，把 `running` / `free_kv_blocks` 当作最近一次 coordinator 快照。

## 8. 兼容性和默认行为

必须保证以下兼容性：

1. 默认值仍是 `waiting_x4_plus_running`。
2. 不传新参数时，行为与当前主线一致。
3. `run_meta.json` 中准确记录真实策略值，字段名保持 `dispatch_policy`。
4. 多机 benchmark 的 case manifest、`--case-csv` 输入、以及 case 展示字符串里都能看到 / 使用该字段。

## 9. 风险点

### 9.1 只改 `SchedulerStats` 字段定义或 `scheduler.make_stats()`，不足以支撑 `least_cache`

原因是 DP LB 实际依赖的是 `EngineCore._maybe_publish_request_counts()` 这条专用发布路径，而不是常规 metrics 路径。

### 9.2 只在 waiting/running 变化时发布 stats，会让 `least_cache` 长期读到陈旧值

因此 `free_kv_blocks` 加进 `SchedulerStats` 之后，还必须把发布触发条件从“计数变化”升级成“完整 LB snapshot 变化”。

### 9.3 普通 dataclass 不能直接替换 coordinator/client 之间的 wire format

当前链路使用的是无类型 msgpack，decode 后不会自动还原成 dataclass；如果直接切换成普通 dataclass，client / coordinator 两端现有按下标访问的逻辑会失配。

### 9.4 `--dispatch-policy` 与 `data_parallel_dispatch_policy` 命名不一致，容易让配置在 parser -> engine config 之间丢失

`AsyncEngineArgs.from_cli_args(...)` 当前是按字段同名拷贝；CLI 参数名、`argparse` 的 `dest`、以及 engine arg 字段名如果不一致，会导致值没有进入 `ParallelConfig`。

### 9.5 `least_batch` 如果只按 `running` 比较，容易出现瞬时倾斜

因此建议把 `waiting` 作为第二排序键，并保留本地 waiting 影子增量。

### 9.6 `kv_cache_usage` 与 `free_kv_blocks` 的选择

实现上更推荐 `free_kv_blocks`：

1. 更符合用户定义。
2. 对未来不同容量 engine 更稳。
3. 更容易做结果解释。

### 9.7 coordinator 结构升级可能波及 elastic / scale-up 路径

`DPLBAsyncMPClient` 的 stats 更新任务里存在 engine 数量变更逻辑，改动 `lb_engines` 的结构时需要同步处理扩缩容初始化默认值。

## 10. 实施顺序

建议按下面顺序落地，减少返工：

1. 在 `ParallelConfig` / `AsyncEngineArgs` 中引入 `data_parallel_dispatch_policy`，默认保持旧值。
2. 在 runner 的 `ExperimentCase` 中增加 `dispatch_policy` 字段，并同步更新 `CASE_CSV_FIELDNAMES`、`load_cases_from_csv(...)`、manifest / 展示逻辑，以及 `build_common_harness_argv(...)` 的透传。
3. 在 harness 中把 `run_meta.json` 的 `dispatch_policy` 改为记录 `args.data_parallel_dispatch_policy`。
4. 在 `SchedulerStats` 增加 `free_kv_blocks`。
5. 在 `EngineCore` 中把 `_maybe_publish_request_counts()` 升级为基于完整 LB snapshot 的 `_maybe_publish_lb_stats()`。
6. 在 coordinator 和 client 中把 wire format 从二元 `[waiting, running]` 升级为三元 `[waiting, running, free_kv_blocks]`，并补齐 elastic scale-up 默认值。
7. 在 `DPLBAsyncMPClient` 中实现三种策略的统一选路逻辑。
8. 最后补 benchmark 侧 case 模板 / 用例，方便直接新增 `LeastCache` / `LeastBatch` 实验组。

## 11. 验收口径

最小验收标准建议如下：

1. 默认 case 不指定策略时，实际行为与现状一致。
2. `data_parallel_dispatch_policy=least_batch` 时，在构造的 `lb_engines` 快照上能稳定选择最小 `(running, waiting)` 的实例。
3. `data_parallel_dispatch_policy=least_cache` 时，在构造的 `lb_engines` 快照上能稳定选择最大 `free_kv_blocks` 的实例；如果 `free_kv_blocks` 相同，再按 `running`、`waiting` 退化。
4. 当多个实例比较键完全相同，仍然保持当前基于 `eng_start_index` 的轮转打散，不出现固定偏置。
5. 只改变 `free_kv_blocks` 而不改变 waiting/running 时，engine 仍会发布新的 LB stats，client 能收到更新后的快照。
6. benchmark 产物里的 `run_meta.json`、case manifest、以及 `--case-csv` 输入路径都能正确记录 / 解析策略名。
7. elastic scale-up 后新增 engine 的 LB stats 默认值正确，不出现索引错误或结构不一致。

## 12. 我对实现细节的建议

如果只追求这次需求落地，我建议采用下面的折中方案：

1. 策略配置名统一使用 `data_parallel_dispatch_policy`。
2. `least_cache` 直接基于 `free_kv_blocks`，不要绕 `kv_cache_usage`。
3. `least_batch` 的比较键用 `(running, waiting)`，不要只看 `running`。
4. coordinator 和 client 之间的 wire format 先升级为固定三元 list `[waiting, running, free_kv_blocks]`，不要在这次需求里直接切到普通 dataclass。
5. DP LB stats 的发布条件要比较完整 LB snapshot，而不是继续只看 waiting/running。

这样做的好处是：

1. 语义清楚。
2. 配置可以从 runner / harness 一路稳定透传到 `DPLBAsyncMPClient`。
3. `least_cache` 不会因为发布条件错误而 silently 读到陈旧值。
4. 后面再加第四种策略时，不需要再拆一次二元数组结构；如果要做具名 wire format，也可以单独规划，不和这次需求耦合。

## 13. 实际落地补充

下面几条是代码落地后，相比原计划需要额外记住的实现事实：

1. 非默认 `dispatch_policy` 已经进入 runner 的历史分组键和 artifact
   scenario prefix；默认策略 `waiting_x4_plus_running` 继续沿用原有路径命名。
2. `manual_multinode_poisson_plan_from_history.py` 也已经把
   `dispatch_policy` 写入导出的 plan CSV，避免 planner 生成的 case 丢失策略字段。
3. 与这次功能直接相关的轻量回归测试主要放在
   [tests/v1/engine/test_dplb_dispatch_policy.py](/vllm/tests/v1/engine/test_dplb_dispatch_policy.py)，
   benchmark 侧回归放在
   [tests/benchmarks/test_manual_multinode_poisson_runner.py](/vllm/tests/benchmarks/test_manual_multinode_poisson_runner.py)
   和
   [tests/benchmarks/test_manual_multinode_poisson_plan_from_history.py](/vllm/tests/benchmarks/test_manual_multinode_poisson_plan_from_history.py)。
