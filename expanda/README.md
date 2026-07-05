# Operational Uncertainty Screening Experiments

这个文件夹现在只保留博士论文下一章需要运行的 operational uncertainty screening 主线：用同一个 40-batch expanded network，在运营者视角下对候选不确定性做粗筛，再把筛出来的少数因素送入后续正式随机/鲁棒路径规划模型。

## 文件说明

- `baseline3.py`：三目标 NSGA-II 路径规划核心模型。
- `run_operational_screening.py`：新的粗筛实验入口。
- `run_refined_operational_experiments.py`：粗筛之后的细筛实验入口，默认重点分析 M1/M2 两个机制。
- `data/data_expanded.xlsx`：40 个 batch 的 expanded network 数据。
- `baseline3_desktop_original.py`：原始脚本备份。
- `.vscode/tasks.json`：VSCode 运行任务。

旧的 10 因素 OFAT、enhanced OFAT、bottleneck stress 脚本和相关结果已移除，避免和现在的 operational screening 混用。

## 粗筛候选因素

| 编号 | 候选因素 | 运营含义 | 文献母类 |
|---|---|---|---|
| C1 | 干线运输时间不确定性 | 铁路、公路、水运主运段实现运行时间偏离计划 | transit time / travel time reliability |
| C2 | 边境通关/换轨处理时间不确定性 | 口岸通关、查验、换轨、排队等待 | service/transfer time + queueing delay |
| C3 | 场站/枢纽换装时间不确定性 | 场站、港口、枢纽的换装处理时间波动 | transfer time / terminal handling time |
| C4 | 服务能力/节点/舱位容量不确定性 | 链路、节点、枢纽、班次、slot 或舱位执行期不足 | capacity uncertainty |
| C5 | 需求/货量不确定性 | 批次货量或短期 OD 货量波动 | demand uncertainty |
| C6 | 通道/线路中断不确定性 | 走廊、口岸、线路、节点临时降容或不可用 | link-node failure / disruption |

实验规则：

- C1/C2/C3 都属于 realized time uncertainty，但发生位置不同，粗筛时分开测试。
- C4 与 C2/C3 存在“容量不足导致排队延误”的因果关系；粗筛可分别测试，进入最终模型时只选一种表示，避免重复计算。
- 交付时间窗作为可靠性约束和评价口径，不作为粗筛扰动因素。
- 罕见地缘级中断不进入随机粗筛，后续可单列为韧性情景。

## 运行方式

先安装依赖：

```bash
cd expanda
../.venv/bin/python -m pip install -r requirements.txt
```

快速检查程序链路：

```bash
cd expanda
../.venv/bin/python run_operational_screening.py --mode smoke
```

建议用于论文粗筛的运行：

```bash
cd expanda
../.venv/bin/python run_operational_screening.py --mode coarse
```

正式更重的版本：

```bash
cd expanda
../.venv/bin/python run_operational_screening.py --mode full
```

粗筛完成后，运行第二阶段细筛：

```bash
cd expanda
../.venv/bin/python run_refined_operational_experiments.py --mode refined
```

细筛默认使用粗筛结果重组后的两个核心机制：

- `M1` 边境瓶颈拥堵：由边境容量压力内生生成处理延误，合并原来的 C4 和 C2。
- `M2` 通道/线路中断：原来的 C6。

M1 使用 BPR 拥堵函数表达“容量/背景流压力 -> 排队/处理延误”：

```text
utilisation = effective_background_flow / effective_border_capacity
border_delay = base_border_delay * (1 + alpha * utilisation^beta)
```

默认 `alpha=0.15`，`beta=4`。这样 C2 不再作为独立随机延误源，而是 M1 拥堵机制的结果。

细筛和粗筛不同：它固定 baseline 生成的 path-library 拓扑，只在同一套候选路径上改变不确定性参数并重新优化。这样可以减少“每个场景重新随机搜路”导致的 route-change 指标虚高。

细筛保留准时率/交付可靠性 KPI，同时新增三个决策相关指标：

- `recovery_cost_delta`：扰动后的额外恢复成本，作为主要经济判据。
- `rerouted_volume_share`：按边境/通道口径计算的重路由货量比例。
- `border_share_shift`：边境口岸货量份额变化总和。

`M1+M2` 组合测试不是简单检查交互，而是主打“中断诱发拥堵”：通道中断会把货流挤到剩余口岸，提高边境利用率并放大 M1 拥堵。
在组合场景中，被 M2 判定为中断/受损的口岸不再重复进入 M1 的 BPR 拥堵计算；M1 只作用于仍开放的边境节点，避免把“节点不可用”和“开放节点排队拥堵”重复计算。

如果要把 C1/C5 也作为 benchmark 加入细筛：

```bash
cd expanda
../.venv/bin/python run_refined_operational_experiments.py \
  --mode refined \
  --candidates M1 M2 B1 B2
```

也可以只跑某几个候选因素：

```bash
cd expanda
../.venv/bin/python run_operational_screening.py \
  --mode quick \
  --candidates C2 C6 \
  --levels low medium high \
  --seeds 2026 7 99
```

所有模式默认检查 `expected-batches=40`，确保使用的是当前 40-batch 数据。

## 输出

输出目录默认在：

```text
expanda/outputs/operational_screening_<mode>/
```

细筛输出目录默认在：

```text
expanda/outputs/operational_refined_<mode>/
```

主要文件：

- `screening_results.csv`：每个 seed、candidate、level 的原始结果。
- `screening_summary.csv`：按 candidate 和 level 聚合后的 KPI 与决策影响。
- `screening_ranking.csv`：粗筛排序与是否进入细筛的建议。
- `screening_report.md`：可直接复制进实验记录的说明和表格。
- `manifest.json`：运行配置和候选因素定义。

细筛主要文件：

- `refined_results.csv`：每个 seed、scenario、level 的原始结果。
- `refined_summary.csv`：5 档强度下的 KPI 和路径变化汇总。
- `refined_ranking.csv`：细筛后的核心/benchmark/secondary 排序。
- `refined_combination_summary.csv`：M1+M2 组合扰动结果。
- `refined_report.md`：细筛报告。

注意：当前 `baseline3.py` 的 hard-feasible 标记非常严格，只要存在迟到罚时或容量超限就不会被标记为 hard feasible。粗筛脚本在没有 hard-feasible Pareto 解时，会自动选取当前 population 中 penalty 最低的代表解继续计算 KPI 和路径变化；结果表中的 `feasible`、`penalty`、`late_teu_h` 和 `infeasible_teu` 用于判断该代表解的约束表现。

## 粗筛判据

每个候选因素施加执行期扰动后，记录两类影响：

- KPI impact：总成本、总运输时间、准点率、P90/P95 延误、不可行量、容量超限。
- Decision impact：相对于 deterministic baseline 的路径分配变化比例。

候选因素被保留的逻辑是：

- 同时影响 KPI 和路径决策：进入核心细筛池。
- 只影响 KPI：作为 reliability-risk factor 进入细筛或保留为补充。
- 只改变路径但 KPI 不明显：作为 strategy-shift factor 视结果保留。
- 两者都不明显：粗筛淘汰。

这一章的定位不是最终随机优化模型，而是为后续模型提供入口论证：只有经过 operational screening 证明 decision-relevant 的不确定性，才进入下一章正式建模。
