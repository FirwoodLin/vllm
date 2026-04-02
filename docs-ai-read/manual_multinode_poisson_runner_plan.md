# Manual Multi-node Poisson Runner Plan

## Goal

把当前这套手工启动命令封装成一个单文件 Python 启动脚本，用来驱动
`/vllm/offline_poisson_harness.py frontend|headless-engine` 的多机实验。

这版方案明确采用“脚本内手写实验 case”的方式，不做矩阵式 CLI 生成。
也就是说：

- 你在 Python 文件顶部直接维护集群列表、数据集别名、并行策略别名、实验 case 列表
- 运行时只需要选择 `--case` 或 `--all`
- 不需要再在命令行里拼大量 `--datasets --rates --strategies`

## Why This Direction

你现在的实际工作流不是“批量穷举所有组合”，而是：

- 临时改一组节点
- 临时改一个并行策略
- 临时改一个数据集
- 临时改几个 request rate
- 观察日志，再继续调下一组

这类 workflow 用矩阵 CLI 会引入很多无效抽象，后续修改成本也高。
更合适的是一个很薄的 Python 编排脚本，核心配置都在文件头部，能直接改。

## Scope

第一版脚本只负责四件事：

1. 根据手写配置生成 `frontend` 和各个 `headless-engine` 的启动命令
2. 通过 SSH 在远端节点启动非 rank0 进程
3. 在本机启动 rank0 `frontend`
4. 把日志、命令、manifest、benchmark 输出目录归档好

第一版不做这些事情：

- 不做矩阵枚举
- 不做复杂交互式菜单
- 不做自动 retry
- 不做复杂结果分析器
- 不再套多层 shell 脚本

## Proposed Deliverable

新增一个单文件脚本，建议放在：

- `/vllm/benchmarks/manual_multinode_poisson_runner.py`

这个脚本直接调用：

- `python3 /vllm/offline_poisson_harness.py headless-engine ...`
- `python3 /vllm/offline_poisson_harness.py frontend ...`

不再依赖额外的 `sh -> sh -> py` 调用链。

## Configuration Model

脚本顶部保留一块非常明确的“手动配置区”。

### 1. Cluster presets

用于手动维护常用远程端点列表，至少支持 2 节点和 4 节点。

示意：

```python
CLUSTERS = {
    "2node_h200": {
        "master_addr": "10.102.98.166",
        "master_port": 29579,
        "frontend_host": "h200-rjob0",
        "remote_hosts": ["h200-rjob1"],
        "gpus_per_node": 8,
        "ssh_opts": [
            "-F", "/root/.ssh/config",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UpdateHostKeys=no",
        ],
    },
    "4node_h200": {
        "master_addr": "10.102.98.166",
        "master_port": 29579,
        "frontend_host": "h200-rjob0",
        "remote_hosts": ["h200-rjob1", "h200-rjob2", "h200-rjob3"],
        "gpus_per_node": 8,
        "ssh_opts": [...],
    },
}
```

这里的“手动选择远程端点列表”不做交互菜单，而是通过：

- 直接修改某个 cluster preset
- 或者让某个 case 指向不同的 cluster key

这样更稳定，也更容易复现。

### 2. Dataset aliases

把常用数据集路径集中放在脚本顶部。

```python
DATASETS = {
    "1k1k": "/mnt/.../1024-1024.csv",
    "issue05_random": "/mnt/.../sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv",
}
```

### 3. Parallel strategy aliases

把常见并行策略写成别名，避免每次手填所有参数。

```python
STRATEGIES = {
    "dp16_tp1_dcp1_ep": {
        "nnodes": 2,
        "dp": 16,
        "dp_local": 8,
        "tp": 1,
        "dcp": 1,
        "enable_ep": True,
    },
    "dp8_tp2_dcp2_ep": {
        "nnodes": 2,
        "dp": 8,
        "dp_local": 4,
        "tp": 2,
        "dcp": 2,
        "enable_ep": True,
    },
}
```

### 4. Experiment cases

这是核心。每个实验组合都作为一个 Python case 明确写出来，而不是 CLI 矩阵。

```python
EXPERIMENTS = [
    {
        "name": "async_dp16_1k1k_r100_bs512",
        "cluster": "2node_h200",
        "strategy": "dp16_tp1_dcp1_ep",
        "dataset": "1k1k",
        "request_rate": 100,
        "max_requests": 9000,
        "warmup_requests": 32,
        "max_num_seqs": 512,
        "max_model_len": 32768,
        "data_parallel_rpc_port": 29550,
        "cuda_visible_devices": "0,1,2,3,4,5,6,7",
    },
    {
        "name": "async_dp16_1k1k_r120_bs512",
        "cluster": "2node_h200",
        "strategy": "dp16_tp1_dcp1_ep",
        "dataset": "1k1k",
        "request_rate": 120,
        "max_requests": 9000,
        "warmup_requests": 32,
        "max_num_seqs": 512,
        "max_model_len": 32768,
        "data_parallel_rpc_port": 29550,
        "cuda_visible_devices": "0,1,2,3,4,5,6,7",
    },
]
```

如果你要测多种 request rate、多种数据集、多种并行策略，就直接多写几条 case。
这比矩阵更贴近你现在的实验方式，也更利于临时插入特殊 case。

## Runtime CLI

虽然不做矩阵 CLI，还是保留一个很薄的运行入口：

- `--list`
- `--case <name>`
- `--all`
- `--dry-run`
- `--artifact-root <path>`

设计原则：

- 实验参数主要在 Python 文件里改
- CLI 只负责选择运行哪些 case

## Script Structure

建议结构如下：

1. 顶部常量区
2. `dataclass`
3. 参数解析
4. case 解析与校验
5. 命令构造
6. SSH 启动远端 rank
7. 本地启动 frontend
8. 清理与归档

建议数据结构：

- `ClusterSpec`
- `StrategySpec`
- `ExperimentCase`
- `ArtifactPaths`

## Command Construction

脚本应显式构造两类命令。

### Headless command

对每个远端节点，构造：

```bash
cd /vllm
CUDA_VISIBLE_DEVICES=...
VLLM_LOG_STATS_INTERVAL=1
python3 /vllm/offline_poisson_harness.py headless-engine ...
```

### Frontend command

在 rank0 本机构造：

```bash
cd /vllm
CUDA_VISIBLE_DEVICES=...
VLLM_LOG_STATS_INTERVAL=1
python3 /vllm/offline_poisson_harness.py frontend ...
```

所有公共参数只在一个地方拼装，避免 rank0 和 rank>0 参数漂移。

## Launch Sequence

每个 case 的执行顺序固定如下：

1. 解析 case，合并 cluster/strategy/case 参数
2. 创建归档目录
3. 生成并落盘所有命令文本
4. 通过 SSH 启动 rank1..N-1 的 `headless-engine`
5. 等待一个短暂的启动稳定窗口
6. 在本机启动 rank0 `frontend`
7. 等待 frontend 结束
8. 记录退出码与结果路径
9. 做远端清理

第一版不强依赖复杂 ready check。
更务实的做法是：

- 先保留一个固定 grace period
- 如果需要，再补基于日志或端口的 readiness probe

## Artifact Layout

归档目录建议按“实验批次 / case”组织。

示意：

```text
<artifact_root>/
  20260402-210500_manual_poisson/
    run_manifest.json
    async_dp16_1k1k_r100_bs512/
      case_manifest.json
      frontend.command.sh
      rank1.command.sh
      frontend.log
      rank1.log
      benchmark/
        run_meta.json
        requests.jsonl
        summary.json
```

每个 case 至少保存这些文件：

- `case_manifest.json`
- `frontend.command.sh`
- `frontend.cleanup.command.sh`
- `rank<N>.command.sh`
- `rank<N>.cleanup.command.sh`
- `frontend.pid`
- `rank<N>.pid`
- `frontend.log`
- `rank<N>.log`
- `benchmark/run_meta.json`
- `benchmark/requests.jsonl`
- `benchmark/summary.json`

`case_manifest.json` 里记录：

- case 名称
- 节点列表
- 数据集路径
- 并行参数
- request rate
- max_num_seqs
- master addr/port
- data_parallel_rpc_port
- frontend 命令
- 各 rank 命令
- 开始时间
- 结束时间
- 退出状态

## Logging Strategy

日志分成两层：

1. 编排层日志
2. harness 原生日志

编排层日志负责回答：

- 哪个 case 在什么时候启动
- SSH 到了哪些节点
- 具体执行了什么命令
- frontend 退出码是什么

harness 原生日志继续由 `tee` 或 stdout 重定向保存，便于和现有排障习惯兼容。

## Cleanup Strategy

需要提供一个保守的失败清理逻辑。

基本原则：

- frontend 退出后，尝试结束远端 headless 进程
- 收到 `SIGINT` 或 `SIGTERM` 时，也触发清理
- 不使用破坏性过强的全局清理命令，优先按端口和命令特征清理本次实验相关进程

清理策略需要分两层：

1. `pidfile` 定向清理
2. `pkill` 兜底清理

脚本在启动 frontend 和每个 headless rank 时，都先写：

- `frontend.pid`
- `rank<N>.pid`

清理时优先执行：

```bash
kill -TERM "$(cat <pidfile>)"
sleep 5
kill -KILL "$(cat <pidfile>)"
```

如果 pidfile 丢了，或者父进程已经不在但 worker 残留，还需要保留这些必需的兜底命令：

```bash
pkill -f -- 'python3 /vllm/offline_poisson_harness.py frontend' || true
pkill -f -- 'python3 /vllm/offline_poisson_harness.py headless-engine' || true
pkill -f -- '--master-port <master_port>' || true
pkill -f -- '--data-parallel-rpc-port <rpc_port>' || true
pkill -f '^VLLM::Worker_' || true
```

说明：

- `frontend` 模式的 `pkill` 主要在 rank0 本机使用
- `headless-engine`、`master-port`、`data-parallel-rpc-port` 这几条可以在所有节点作为兜底
- `pkill -f '^VLLM::Worker_'` 会更激进，只适合这批机器专门跑这组实验时使用

第一版可以借鉴 `dual_route_manual_runner.py` 里的思路，但会把这些命令直接落成每个 case 的 `*.cleanup.command.sh`，避免清理逻辑藏在脚本内部。

## Assumptions

第一版默认这些前提成立：

1. 所有节点都能通过 SSH 免密互通
2. 所有节点都能访问相同的 `/vllm` 路径
3. 所有节点都能访问相同的数据集和模型路径
4. artifact 根目录位于共享挂载，所有节点都能直接写同一套 case 目录

因为 artifact 目录是共享挂载，第一版可以直接采用：

- rank0 本地直接把 frontend 日志写到 case 目录
- 远端 headless 也直接把日志写到同一个共享 case 目录
- 不需要额外做 `scp` 或日志回传

如果以后 artifact 目录不再是共享挂载，才需要第二版再补远端日志回传。

## Implementation Plan

### Phase 1

实现最小可用脚本：

- 固定写死 `CLUSTERS`
- 固定写死 `DATASETS`
- 固定写死 `STRATEGIES`
- 固定写死 `EXPERIMENTS`
- 支持 `--list`
- 支持 `--case`
- 支持 `--all`
- 支持 `--dry-run`
- 正常归档 manifest 和日志

### Phase 2

补充稳定性：

- 更明确的远端启动失败检测
- 更明确的 signal cleanup
- 更好的 case 状态落盘

### Phase 3

按需要再补：

- case 分组
- 失败后跳过或继续
- 更强的 ready check
- 结果汇总脚本

## Code References

实现时主要参考这两个文件：

- `/vllm/vllm/benchmarks/offline_poisson_harness.py`
- `/mnt/nvme1n1/ml_research/linbinbin1/rjob-setup/dual_route_manual_runner.py`

取舍原则如下：

- 复用 `offline_poisson_harness.py` 已有参数和输出目录语义
- 只借鉴 `dual_route_manual_runner.py` 的 SSH/manifest/cleanup 组织方式
- 不引入它的双路由和分析状态机复杂度

## Final Recommendation

最终建议是：

- 用一个单文件 Python runner
- 所有实验 case 直接写在 Python 文件里
- CLI 只负责选择 case，不负责生成矩阵
- 保持调用链尽可能薄
- 把归档做清楚，后续才方便做多轮对比实验

这个方向和你现在的实验习惯更一致，后面改参数也最快。
