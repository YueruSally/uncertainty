# 6.8 EV / CCP / 最坏情况：正式搜索前核对

此入口只进行第一阶段的核对，不启动试跑、正式搜索、方案目标评价或训练。
原始实验、代码默认设置和已有输出均不修改。每次调用新建带 UTC 时间的独立目录；
显式指定的输出目录若已经存在，立即停止。

在 lnx-grid-41 的 expanda 目录执行：

```bash
python3 -B -u section68_ev_ccp_worst_v1/preflight.py
```

核对内容：

1. 统一重算 v2 Table 5 完成状态、计划校验、原 Base 的 54 次记录、共同种子、
   冻结引擎/控制器/网络数据文件哈希、250/5000/100/.9 设置、交叉 .90 与变异 .15、
   Base 无附加 I/C/M 控制且未加载模型。保留 Base 本来就有的初始化、交叉、变异、修复。
2. 从 Table 5 当前 PPO 的 training_source 所指原目录读取验证实例，
   选择 S6（30 批、窗口 .5）、S8（30 批、窗口 2）各编号最小的 validation 实例。
   不把 screen 实例自动当成 validation 实例。
3. 使用同一冻结采样器核对 EV 的均值输入和节点铁路均值 40.6 小时，
   重建实例原 ccp100_seed 的 100 个优化场景并记录摘要。
   不把不同实例的相同场景编号当作共同场景。
4. 核对原评价器的第 90/100、100/100 与 4500/5000 次序统计量；
   固定实例路径种子，比较 EV 与 CCP 使用的名义可行路径集合、次序和容量资源。
   此处构造路径及路径诊断，但不调用完整运输方案目标评价器。
5. 查明是否存在 Table 5 的同实例、同种子 Base 前沿；没有则标记条件不适用。
   列出服务器上已有的环境记录、实验计划和可能的独立场景文件，供下一步核对。
   1000 次参考目标样本包不能代替 5000 个原始网络场景。

输出：`PREFLIGHT.json`、`SELECTED_INSTANCES.csv`、`SOURCE_HASHES.json`、
`ENVIRONMENT.json`、`UNCERTAINTY_AUDIT.json`、两个实例的完整副本及名义路径选项、
`EXISTING_SCENARIO_AND_ENVIRONMENT_FILES.json`。

不确定性核对逐项比较原记录中的全部采样输入，以及分布名称、独立性与冻结 SAA 标志。
`sampling_diagnostics` 是每个实例实际抽样后的统计输出，单独保留，不与参数字典比较。
`default_scenario_count=200` 是配置模板默认值；实际场景数由 `scenario_count=100`
及搜索记录的 `training_size=100` 核对。真实参数缺失、不同或出现未知字段仍立即停止，
差异的字段名、原值、预期值写入 `UNCERTAINTY_AUDIT.json`。

`completed=true` 仅表示本入口完成。`ready_for_pilot=false` 表示尚未核对
论文 6.1 的原环境与独立场景来源，不可据此声称正式实验已开跑或已完成。
任何不符均输出 `status=stopped` 并退出，不自动替换实例、放宽检查或修改旧文件。

后续仍须落实：保存/复用每实例 5000 个独立场景，三模型共用；三次完整试跑及
全部输出检查；有同实例 Base 时逐点复现；交叉评价中 WC≥CCP；通过后才扩展到
18 次搜索（最多两个并行），全部前沿复评及选中方案逐批到达时间等输出。

## 已通过预检后的试跑入口

```bash
python3 -B -u section68_ev_ccp_worst_v1/launch.py --stage pilot --workers 2 \
  --preflight transformer_ppo_runs/diagnostics/section68_preflight_v3 \
  --out transformer_ppo_runs/diagnostics/section68_ev_ccp_worst_runs_v3
```

先运行预检生成新报告，再用 `--preflight /绝对路径` 和 `--out /全新目录` 明确指定。
旧的 `section68_preflight_20261124T155523046075Z` 缺少全局过境事件初始化，
其名义路径容量摘要不可复用；新启动器要求 `nominal_options_context_version=2`。
已有目录必须显式 `--resume`；已完成任务必须通过所有输出哈希检查才能复用。
未完成任务使用新的 `attempt_时间戳`，不覆盖上一次搜索或失败记录。
准备过程失败且尚未生成 PLAN 时，应另指定全新目录；不删除旧目录来掩盖失败。

### 已报告的 MemTotal 差异

2026-11-24，历史记录的 `total_node_memory_bytes=33564069888`，本次为
`33565057024`，相差 987136 字节（0.94140625 MiB，约 0.002941%）。
此字段取自 Linux `/proc/meminfo` 的 `MemTotal`，表示内核报告的可用总内存，
不是物理内存条容量，也不是给实验独占分配的内存。当前记录未查明差异原因。

默认仍逐项严格核对。显式选项 `--accept-memtotal-pair ORIGINAL_BYTES CURRENT_BYTES`
仅接受这一对具体字节数；历史值、本次值任一不匹配，或主机、CPU、Python、数值库
等其他检查不通过，仍停止。原环境文件不改写。差异原值、字节数和百分比存入
新目录的 `ENVIRONMENT.json` 并纳入文件哈希；例外写入冻结的 `PLAN.json`。
恢复运行使用冻结的例外，并保存新的 `ENVIRONMENT_RECHECK_*.json`。

保留停止记录，重新预检后在新目录试跑：

```bash
python3 -B -u section68_ev_ccp_worst_v1/preflight.py \
  --out transformer_ppo_runs/diagnostics/section68_preflight_v3

python3 -B -u section68_ev_ccp_worst_v1/launch.py --stage pilot --workers 2 \
  --preflight transformer_ppo_runs/diagnostics/section68_preflight_v3 \
  --out transformer_ppo_runs/diagnostics/section68_ev_ccp_worst_runs_v3 \
  --accept-memtotal-pair 33564069888 33565057024
```

### 预检与搜索的过境事件初始化修复

旧预检只把事件定义传给场景构造函数，没有设置求解器的全局
`BORDER_EVENT_DEFINITIONS`。名义容量模拟另行读取这个全局变量，因此得到的
部分容量占用日期与实际 `run_episode` 不同。S6 验证实例的旧摘要
`c35b5db94f996cc476db66fc1ff0e91073bbf90e8716c08a48d2ef4af6a087da`
已在本地独立复现。路径次序、运输方式和名义到达诊断相同，但 387 个路径选项的
容量映射不同；其中一例的 Moscow→Duisburg 公路弧由第 12 天变为第 13 天。
这属于新预检程序的初始化错误，不是更改主实验的路径或求解规则。

修复后，预检先按冻结 `run_episode` 加载同一份过境事件定义，并在每个实例前
清空活动场景和路径缓存，模拟搜索子进程的初始状态。预检和搜索共用相同的摘要
序列化函数，仍严格比较全部路径顺序、诊断和容量映射；不放宽或删除哈希检查。
若再发生不一致，子进程保存 `NOMINAL_PATH_MISMATCH.json` 的实际选项和两份摘要，
父进程直接显示该子进程的具体错误，不再只显示 `Task stopped`。

启动流程：

1. 检验源文件未变，读取完整服务器场景清单，并将当前 CPU、Python、节点总内存及
   NumPy/PyTorch 与 `ccp_paper_evidence_b5000_v1/ENVIRONMENT.json` 比较。
   节点核数和内存不声称是独占分配；历史分配未记录时保留为空。
2. 匹配已有独立场景的实例身份及配置。出现同实例但无法解释的 OOS 文件或非空独立种子时停止，
   不擅自另抽。可识别的原始网络场景包直接复用。
3. 没有可复用匹配包时，用冻结的原采样器生成独立 5000 场景：S6 为种子 368500006，
   S8 为 368500008。保存实际弧乘子、口岸延迟、事件映射、均值、随机种子和哈希；
   优化 100 场景和 EV 平均输入也分别保存。没有更改任何 CV 或抽样规则。
4. S6、种子 993001，EV/CCP/WORST 各运行一次。使用冻结的 Base 搜索函数，
   仅通过本进程包装器显式切换 `evaluation_mode` 和 `ccp_confidence`。
   没有修改冻结源文件或其默认行为。I/C/M 附加控制器和模型均关闭。
5. 每次检查搜索实际使用的名义路径及场景摘要；在原优化输入上逐方案精确复现
   最终前沿目标值，再复评全部前沿。保存两个选中方案的逐场景目标和逐批到达时间，
   并核对 WC≥CCP、逐批最晚到达值与完工时间完全一致。
6. 若试跑实际次数不是恰好 5000，记录实际次数并停在 `stopped_after_pilot`。
   不补造评价次数，不为补足预算更改 Base 的原子操作规则。

EV 是均值**输入**的一次评价，不是将 100 个情景的输出求平均。
铁路节点均值仍为 40.6 小时；当前两端节点组成的跨境事件均值为 81.2 小时。
CCP/WORST 使用同一个 100 场景包，分别取第 90 小和最大值；独立复评 P90 为第 4500 小。

输出在每个任务的 `attempt_*` 中：

- `search/`：原求解器配置、预算计数、完整最终前沿及日志。
- `FINAL_FRONT.json`、`SELECTION.json`：完整决策、模型目标及选方案的缩放和索引。
- `FRONT_OOS_SUMMARY.csv`：所有前沿方案的三目标均值、P90、最大值和准时率。
- `minimum_cost_SCENARIOS.*`、`compromise_SCENARIOS.*`：各选中方案的三目标逐场景值；
  NPZ 同时包含批次 ID、截止时间、5000 列的逐批到达时间。
- `*_BATCH_ARRIVALS.csv`：逐批、逐场景到达时间和整批准时判定。
- `CROSS_EVALUATION.csv`、`ROUTE_METRICS.csv`、`CHECKS.json`、`COMPLETE.json`。

准时率对批次×场景等权；每批以全部正份额路径的最晚到达时间判定，不按货量加权。
路径数按不同的正份额路径计数；铁路口岸按每条分配路径访问的铁路跨境节点计 TEU，
海运航线按有向海运弧计 TEU，方式份额按 TEU·km 计。
成本最低的平局按时间、排放、精确决策 ID 处理；折中解距离相同时采用相同后续排序。
恒定目标维度的归一化贡献为零。

完成后查看 `STATUS.json` 中的 `pilot.results` 指向的 `PILOT_RESULTS_*`：
`PER_SEED.csv`、`SEED_MEANS.csv`、`PAIRED_VS_EV.csv`、`PAIRED_SEED_MEANS.csv`、
`ROUTE_METRICS.csv`、`ROUTE_SEED_MEANS.csv`、`RUNTIME.csv`、`REPORT_CN.txt`。
运行时间沿用原求解器的 `optimisation_seconds`，复评时间单独报告。

只有 `stage=pilot_passed` 后继续完整 18 次：

```bash
python3 -B -u section68_ev_ccp_worst_v1/launch.py --stage all --workers 2 --resume \
  --out transformer_ppo_runs/diagnostics/section68_ev_ccp_worst_runs_v3
```

该命令重新核验输入、环境及试跑输出后复用已完成三次。正式汇总也使用新的
`RESULTS_时间戳` 目录。任何失败均写入 `FAILURE_*.json` 并停止，不修改历史结果。

## 已确认接受原 Base 的预算上限停止规则后续跑

2026-11-24 的 v3 试跑实际次数为 EV 5000、CCP 4999、WORST 4999。
用户确认采用“5000 次上限，保留原停止规则”继续。原 `launch.py` 和冻结的
`PLAN.json` 均不改写，避免破坏已有文件哈希及三个试跑结果的关联。
改用单独入口：

```bash
python3 -B -u section68_ev_ccp_worst_v1/resume_budget_cap.py \
  --out transformer_ppo_runs/diagnostics/section68_ev_ccp_worst_runs_v3 \
  --workers 2 --accept-original-budget-stop
```

入口重新核验冻结输入、环境和完成记录；必须已有全部三次试跑。只接受
`4999 + atomic_pair_budget_remainder` 或 `5000 + evaluation_cap`，不补评价、不改计数。
已有任务通过哈希检查后复用，最多两次搜索并行；中断后重复同一命令。
新增 `BUDGET_CAP_CONTINUATION_时间戳.json`，记录本次明确接受的规则、旧计划哈希、
入口代码哈希、commit、原状态和已完成任务。正式输出另存 `RESULTS_时间戳`，
其中 `BUDGET_POLICY.json` 记录政策来源和各次实际次数。历史试跑报告保留。
`STATUS.json` 更新当前进度，历史状态继续追加到 `PROGRESS.jsonl`。
