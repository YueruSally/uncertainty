# 6.8：S7 补跑与 S6–S8 汇总

入口：`section68_s7_supplement_v1.py`。不调用 `resume_budget_cap.py`，不修改其余实验目录或默认行为。

## 固定计算版本

- 原实验冻结提交：`1b95b109957f24b343c153f2f627c4c98f7dc0bd`。
- 用户确认的上一轮环境提交：`26ac7559c97bba1e15274eb0195b6aca2d1e3597`。从原实验的预算续跑许可记录核实该提交，读取记录不等于调用续跑程序。
- 新增入口提交：运行时从 Git HEAD 读取，同时验证入口文件与该提交完全相同。拉取入口后实际 checkout 的提交是这一新提交，不能把它误写成 `26ac755`。
- `preflight.py`、`engine.py`、`launch.py` 必须逐字节等于 `1b95b10`；全部底层依赖、模型定义、数据和旧配置须通过原 `PLAN.json` 的文件哈希核对。
- 核对原运行环境的主机、CPU、Python、库版本、CPU affinity 和已明确接受的 MemTotal 记录，差异立即停止。

## 执行顺序

1. 读取、校验 S6/S8 的18次已完成记录，不改写它们。
2. 在新目录中复制原 S6 实例及场景包，以原冻结 worker 从头重跑 `S6 / CCP / 993001`。该 worker 保留原方案选择、全部前沿复评与保存行为。
3. 新前沿的完整 JSON 数据、精确决策、三个目标和实际评价次数必须与旧 S6 运行一致；数值容差为0。不一致时记录 `S6_REPRODUCTION_CHECK.json` 并停止，S7不会启动。
4. 用原 preflight 的同一验证实例选择流程，仅将目标配置选择改为 `S7=(30,1.0)`。取验证实例编号最小者，所有原核对保留。
5. 用原 prepare 构造并保存 S7 的100个优化场景、1个平均输入场景、5000个独立场景。若不存在可复用独立输入包，沿用原种子公式 `368500000 + 7 = 368500007`；存在输入包时沿用原来源审查，不能用方案目标值样本冒充输入包。
6. 从头执行 S7 的9次任务：3个模型 × 993001、993002、993003。最多并行2次。EV为单次平均输入 deterministic；CCP和最坏情况使用同一100场景。
7. 不补齐最后一个评价位置；实际4999须对应 `atomic_pair_budget_remainder`，5000须对应 `evaluation_cap`。其他次数或停止理由停止报告。
8. 使用原 exporter 的汇总、配对公式读取旧18次和新9次的已保存文件。复现运行不重复计入27次汇总。增加的平均排放、平均完工时间和拆分批次比例均读取已保存字段，不重新评价旧方案。

只调整新计划的配置选择、任务数、来源路径和版本元数据；原计算文件不变。S7首次运行必须使用全新目录，入口不支持从已有目录续跑。

## 命令

在41执行：

```bash
cd /home/lunet/cokt2/uncertainty-learning-components/expanda
python3 -B -u section68_s7_supplement_v1.py --workers 2
```

默认新输出：`transformer_ppo_runs/diagnostics/section68_s7_supplement_v1`。
可通过 `--out` 指定另一个不存在的新目录。默认旧来源为 `section68_ev_ccp_worst_runs_v3`，不写入旧目录。

## 输出

- `VERSIONS.json`、`ENVIRONMENT.json`、`SOURCE_HASHES.json`：版本、环境和原文件证据。
- `S6_REPRODUCTION_CHECK.json`、`S6_REPRODUCTION/`：精确复现结论及单独保存的复现运行。
- `S7_PREFLIGHT/`、`SELECTED_S7.json`：选择的真实实例、路径/场景核对和随机种子。
- `S7_RUNS/`：冻结计划、实际场景包、9次完整搜索、逐方案和逐场景数据。
- `STATUS.json`：阶段、运行进度、错误或最终报告路径。
- `COMBINED_S6_S7_S8/RESULTS_*/MAIN_RESULTS_SEED_MEANS.csv`：18行（3实例×3模型×2方案类型）的主要指标。
- 同目录 `PAIRED_SEED_MEANS.csv`：相对EV的配对变化；先逐种子比较，再平均。
- `MAIN_RESULTS_PER_SEED.csv`、`PAIRED_VS_EV.csv`：逐种子数据。
- `SEARCH_TIME_SEED_MEANS.csv`：三个实例各模型的搜索耗时均值和样本标准差。
- `ACTUAL_EVALUATIONS_ALL27.csv`：27次搜索各自的真实评价次数和停止原因。
- `REPRODUCTION_RUNTIME.csv`：额外S6复现运行的次数和时间，单独报告。
- `COMBINED_REPORT_CN.txt`：中文数据汇总，不作因果解释。

原 `rail_border_TEU` 的节点分类未调整；本入口保持原路线指标定义。新增主表的拆分比例直接读取 `split_batch_fraction`，并乘100转换为百分数。
