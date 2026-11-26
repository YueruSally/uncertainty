# P80 标准 NSGA-II 独立对照

新增 S0–S8 的 54 次、S9–S11 的 27 次，共 81 次搜索。不重训模型，不改原 P80
流水线的代码、默认参数、模型和结果。不补 Table 7 的网络迁移实验。

## 名称与文献

- 新方法：**NSGA-II (standard framework)**，方法 ID `NSGA2_STANDARD`。
- 原来的 `BASE`：新表中显示为 **Adapted NSGA-II**。旧文件及 ID 不修改。
- 算法框架引用：Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002).
  A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions
  on Evolutionary Computation, 6(2), 182–197. DOI: 10.1109/4235.996017.
- 原文地址：https://dynamics.org/~altenber/UH_ICS/EC_REFS/MULTI_OBJ/DebPratapAgarwalMeyarivan.pdf

这里实现原文的随机初始化、非支配排序、拥挤度、二元锦标赛、父子代合并精英选择，
以及约束支配原则。原文没有规定“运输路径＋拆分份额”的编码或操作算子；本实现
保留项目的问题编码，并明确使用下面的基本算子，不能写成直接运行了作者原始源代码。
Deb 的引用说明 NSGA-II 框架，不为当前 Base 的领域改进提供文献依据。若这些领域
改进来自作者既有 TR-E 论文，论文中还应引用那篇具体文献；此处没有猜测其出处。

## 冻结的实现定义

| 部分 | 标准对照 |
|---|---|
| 种群 / 预算 | 250 / 5,000 次方案评价 |
| 初始化 | 每批随机取 1–3 条路径、随机正份额归一化；全部 250 个个体都随机 |
| 交叉 | 每对父代以 0.90 概率执行逐批分配块均匀交叉；否则复制父代 |
| 变异 | 每个子代以 0.15 概率变异；均匀选一批，在增路径、删路径、随机替换路径、份额扰动之间等概率选择 |
| 路径与份额 | 使用同一条名义路径库；份额为正并归一化；不规定搜索阶段最多三条路径 |
| 不可操作时 | 例如只剩一条路径不能删除，则该次变异不改变染色体，不按收益反复尝试 |
| 份额扰动 | 随机选一条路径，份额乘以 U(0.5,1.5)，合并重复路径并归一化 |
| 选择 | 标准快速非支配排序与拥挤度；锦标赛完全平局时随机选择 |
| 约束 | 与主实验相同的可行性判定、无量纲约束违反量；可行优先，不可行间按违反量比较 |
| 最终输出 | 末代可行非支配解集，含每批路径、运输方式和份额 |

关闭容量感知/贪心/Directed 初始化、混合路径拼接交叉、可靠路径引导变异、
可行性增强和全部 I/C/M 候选规则与学习策略。没有训练或模型选择。
每个新生成个体恰好评价一次，包括复制子代；250 个初始个体加 19 代各 250 个子代，
实际评价数为 5,000。缓存中的继承目标不减少此计数。所有修改仅在独立子进程的内存中
生效，退出后恢复，不改变任何已有 Python 文件。

`kernel.py` 使用同一已冻结的非支配排序、拥挤度和精英筛选函数；仅为标准约束支配
替换内存中的 `dominates`。同违反量的两个不可行解不使用目标值进一步判断支配。

## 与 P80 主实验的配对

从 `ccp_alpha08_single_full_v1/PLAN.json` 读取实例和搜索种子。
保持网络、路径生成种子、100 个优化场景、CCP P80、评价器、CV、过境封顶、罚金、
约束定义不变。逐运行校验优化场景摘要与主实验源记录一致。

所有 81 次搜索先运行。之后等待主 P80 实验的 `oos/COMPLETE.json`，只读其中的
实际独立场景文件。100 / 1,000 / 5,000 在本对照中全部是**复评场景数**，不是
搜索检查点；三个 P80 分位值分别取第 80 / 800 / 4,000 小的值。
先在原 100 场景复现标准对照保存的全部最终目标，再做独立复评。

已有方法的逐方案复评数组直接复用。每个实例用“已有 P80 全部方法＋新标准对照”
全部种子、三组独立场景的目标值重新确定共同边界。所有方法的 HV 在新的目录中
一起重算，参考点 (1.1,1.1,1.1)，除以 1.331；不把新方法的 HV 拼到旧边界的表里。

新表同时给出以 Adapted NSGA-II 和以标准 NSGA-II 为参照的结果；筛选
`reference_method=BASE` 或 `reference_method=NSGA2_STANDARD` 即可获得一张口径统一的表。
均值先种子、再实例、最后配置等权。95% 区间做 50,000 次配对层级 bootstrap：配置固定、
配置内重抽实例、实例内重抽种子。此区间口径明确记录，不搬用旧区间。

如果标准方法最终没有可行解，如实报告空前沿与 HV=0，不另跑种子或取中间前沿替代。
参照 HV 为零时相对变化未定义，保留空值和计数，不加 epsilon，也不丢弃该运行。

## 41 上运行

```bash
cd /home/lunet/cokt2/uncertainty-learning-components/expanda &&
git pull --ff-only origin codex/m-transformer-ppo-threeway &&
bash standard_nsga2_alpha08_v1/launch.sh
```

默认一个搜索进程、单线程、nice=10，可与原 P80 任务并行；两者可能争用 CPU/内存，
本入口不会中止或调整另一个进程。搜索完成后等待原 P80 复评，等待期间不计算目标。

默认新目录 `transformer_ppo_runs/diagnostics/ccp_standard_nsga2_alpha08_v1`。
启动器打印 PID 和日志路径；`STATUS.json` 每 20 秒更新搜索进度与基于已完成运行的估计。
重复同一启动命令复用经哈希核对的已完成任务；未完成尝试另存 `.incomplete-*`。
可用 `run.py --search-only` 只运行搜索，之后 `run.py --report-only` 完成复评及汇总。
重复启动会因文件锁停止第二个进程；不会重复运行第一组。

## 输出

- `PLAN.json`：原主实验版本、新入口 commit、源文件哈希、81 个任务及完整算子定义。
- `searches/.../RUN_COMPLETE.json`：运行时间、实际评价数、零 ICM 调用、核对信息。
- `searches/.../final_feasible_nondominated.json`：完整最终决策与优化 P80。
- `oos/instances/...`：原优化场景复现以及共用独立场景下的新增评价包。
- `RESULT_LOCATION.json`：新汇总目录及文件哈希。
- `RESULTS_100.csv` / `RESULTS_1000.csv` / `RESULTS_5000.csv`：平均 HV、配对提升、
  新区间、胜平负和搜索耗时。主结果为独立 5,000 场景，1,000 场景作为对照。
- `PER_RUN_HV.csv` / `PER_PAIR_HV.csv`、`FINAL_FRONT_COUNTS.csv`、
  `P80_SHIFT_PER_RUN.csv`、`SCENARIO_DIFFERENCES_PER_RUN.csv`、`INSTANCE_BOUNDS.json`。

`BASE` 历史文件里的日志 HV 仍原样保存。标准对照沿用冻结日志归一化器仅供进度记录，
该量不参与搜索选择或新结果表；正式结果一律来自末代方案的独立复评及新共同边界。
