# α=0.9：基础 NSGA-II 补充及完整共同边界报告

此入口不读取 α=0.8 的运行结果。仅复用其**不含置信水平的标准算法内核**
`standard_nsga2_alpha08_v1/kernel.py`，并在专用子进程中显式设置 α=0.9。
旧 P80/P90 代码、默认行为、模型和结果均不修改。

## 范围

- 原 α=0.9 全量最终解复评必须已经完成，来源目录由 `--parent` 指定。
- 立即导出原810次运行在 **100、1000、5000个独立复评场景**下的完整表。
  这是“加入基础NSGA-II之前”的表，使用原报告的HV，不重新搜索或评价。
- 新增基础NSGA-II：S0–S8的18实例×3种子=54次，S9–S11的9实例×3种子=27次。
  实例、种子、网络和优化场景从完成的P90来源记录读取，不从P80结果借用。
- 每次种群250，5,000次方案评价，100个优化场景，三个目标各取第90小的值。
- 仅用搜索结束时的完整可行非支配解集，保存每批路径及份额。
- 原Table 7的27次网络比较复用；此次不增加小网络或大网络的基础NSGA-II搜索。
- 合并后：Table5为756次，Table6为108次，Table7为27次，共891次。

## “基础”的定义

使用Deb等(2002)的非支配排序、拥挤距离、二元锦标赛和父子代精英保留框架。
运输问题编码需要具体算子：随机路径及份额初始化，批次块均匀交叉，随机增删换路径或份额扰动。
交叉概率0.90，变异概率0.15；无规则ICM、无学习模型、无启发式种子、无目标引导修复。
采用可行性优先的约束支配；编码合法性修复保留。初始每批最多3条路径，不限制后续路径数。
这不是声称复刻Deb作者的源代码；新表写作 `NSGA-II (standard framework)`。
原 `BASE` 保留文件标识，展示名称为 `Adapted NSGA-II`。

## 复评与HV

直接读取已保存的P90优化100场景、独立100/1000/5000场景数组。
不另抽随机种子，也不选择性换场景；检查场景摘要与原方法逐项一致。
分别取第90、900、4500小的目标值。原优化100场景额外用于复现检查，不与独立100混称。

每个实例用全部原方法、新基础方法、所有搜索种子、三组独立场景的全部最终方案目标值
确定共同最小/最大边界。参考点为归一化后的(1.1,1.1,1.1)，HV除以1.331。
Table7仍在同一需求实例的大小网络之间合并边界；没有新增方法，其原边界和HV应保持不变。
每个旧运行在旧边界下的HV必须先复现，与原报告差异超过1e-9就停止。
所有方案都保留；不按HV好坏删除点。空可行前沿报告HV=0，不按结果补跑。

均值：实例内平均配对种子，再配置内平均实例，最后配置等权。
95%区间：配置固定，配置内重抽实例、实例内重抽配对种子，50,000次。
这与旧Table5“固定实例、只重抽种子”的区间不同，新表明确记录口径。
相对值分母为0则留空并报告数量，不删掉配对，也不添加epsilon。
搜索时间沿用原运行记录；新基础方法另计实际搜索时间及机器、负载和线程信息。
复评和报告耗时不算入搜索时间。

WTL同时给出三个明确口径，均不改动实际HV或平均提升：

1. `original`：已除体积HV差容差 `1e-12 / 1.331`；
2. `raw_0.01`：原HV差容差0.01，对应报告HV差 `0.01 / 1.331`；
3. `reported_0.01`：已除体积HV差容差0.01。

后二者均不是相对提升±1%，也不是统计等效检验。

## 在41运行

```bash
cd /home/lunet/cokt2/uncertainty-learning-components/expanda
bash standard_nsga2_alpha09_v1/launch.sh
```

可选两个位置参数：来源P90完整复评目录、新输出目录。
默认新输出 `transformer_ppo_runs/diagnostics/ccp_standard_nsga2_alpha09_v1`。
先校验并导出已有三组场景表，再以单进程、单线程、nice=10后台运行。
日志写入文件，不连接交互终端；不会向其他实验进程发信号。
重发相同命令会校验并复用已完成的新搜索/评价块，不覆盖旧实验。
本目录代码在准备时固定哈希，运行中不要修改。

```bash
ROOT=/home/lunet/cokt2/uncertainty-learning-components/expanda/transformer_ppo_runs/diagnostics/ccp_standard_nsga2_alpha09_v1
watch -n 10 "cat \"$ROOT/STATUS.json\""
```

Ctrl+C只退出上述查看窗口。`STATUS.json`记录当前阶段、搜索进度、正在运行的子进程，
搜索阶段根据本轮实际耗时估计剩余时间。复评阶段按实例和场景组更新已完成方案数。

## 输出

- `EXISTING_RESULT_LOCATION.json`：加入新方法前的完整三组场景表（可立即看）。
- `RESULT_LOCATION.json`：全部结束后包含基础NSGA-II的新共同边界正式报告。
- `REPORT.txt`：全部表、三组场景的可读汇总，含HV、提升、95%区间、搜索时间和三种WTL。
- `COMPLETE_TABLES_100_1000_5000.csv`：三组场景横向排列的完整总表。
- `RESULTS_100.csv / RESULTS_1000.csv / RESULTS_5000.csv`：按总体和每配置拆开。
- `ALL_SCENARIO_SUMMARY.csv / OPTIMIZATION_100.csv`：含单独的样本内结果。
- `PER_RUN_HV.csv / PER_PAIR_HV.csv`：逐运行HV和配对结果。
- `DIRECT_COMPARISONS.csv`：SL对随机、SL对全零阈值、随机对规则、PPO对SL。
  同时明确给出以改进版Base为分母的差（百分点）和相对比较方法的提升（%）。
- `WIN_LOSS_MAGNITUDES.csv`：胜幅、负幅和全体配对提升中位数；按容差分别报告。
- `SCENARIO_DIFFERENCES*.csv`：100对1000、100对5000、1000对5000的HV相对差。
- `FINAL_FRONT_COUNTS.csv / P90_SHIFT_*.csv`：最终方案数及优化100与独立5000的P90偏移。
- `ORIGINAL_QUANTILE_COVERAGE_PER_RUN.csv / OUTSIDE_REFERENCE_PER_RUN.csv`：覆盖率与超参考点计数。
- `OLD_NEW_HV_REPRODUCTION.csv / OLD_NEW_SUMMARY.csv`：旧边界复现及新共同边界造成的数值变化。
- `PLAN.json`：全部实例、种子、配置、commit、源文件和计算代码哈希。

Table5/6分别输出相对改进版NSGA-II和基础NSGA-II的结果，不混用分母。
Table7小网络SL相对小网络Base，大网络SL相对小网络SL；不虚构缺失的大网络Base。
