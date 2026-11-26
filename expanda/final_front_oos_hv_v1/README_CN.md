# 最终解集独立场景复评：完整性检查与 S0 试评

`run.py`只实现第1—4步并在试评后停止。用户确认后的第5步使用独立入口`full.py`，见下节。
两个入口都不执行搜索、训练或模型选择。
不改动原求解器、评价器、采样器、模型、参数默认值及已有结果。
所有输出写入新的带时间戳目录；指定的输出目录如果已存在，直接拒绝运行。

在41执行：

```bash
cd /home/lunet/cokt2/uncertainty-learning-components/expanda &&
git pull --ff-only origin codex/m-transformer-ppo-threeway &&
python3 -B -u final_front_oos_hv_v1/run.py
```

如仅做文件清单，不生成场景、不评价任何方案：

```bash
python3 -B -u final_front_oos_hv_v1/run.py --inventory-only
```

## 先检查整个范围，再允许试评

来源目录由 `--table5`、`--table6`、`--table7` 指定，默认分别为：

| 表 | 目录 | 预期运行数 |
|---|---|---:|
| Table 5 | `ccp_shared_sl_ablations_b5000_v2` | 13方法×18实例×3种子=702 |
| Table 6 | `ccp_single_ppo_s9_s11_refresh_v1` | Base、shared SL、新PPO×9实例×3种子=81 |
| Table 7 | `ccp_network_transfer_single_b5000_v1` | 小网络Base、小网络shared SL、大网络shared SL×3批次组×3种子=27 |

Table 6旧Simple和旧PPO仅属于旧报告的归一化池，不是当前显示的方法，不冒充新PPO。
Table 7没有大网络Base，本入口不生成或假定该对照。

检查每个运行的 `RUN_COMPLETE.json`、`configuration.json`、
`final_feasible_nondominated.json`、实例文件、网络及来源哈希。
实例和搜索种子以 `RUN_COMPLETE.json` 中的字段为准，不假定目录一定包含方法层。
原共享监督和当前PPO的 `实例/seed_种子/RUN_COMPLETE.json`，以及其他方法的
`实例/seed_种子/方法/RUN_COMPLETE.json` 均可读取；哈希与身份核对不变。
最终解必须包含全部批次的路径节点、运输方式、非负份额，份额和为1；
决策指纹必须一致，解的目标值必须与完成记录的实际最终前沿相同。
搜索预算是5000；沿用历史的原子子代停止规则，实际4999或5000都逐项记录。
400/1000中间前沿不进入新评价或新归一化。

任一范围缺失或不符，输出完整的已发现问题清单并停止，不生成新场景。
此时应先查看缺失运行是否还有原始备份；若决策确实未保存，不能从HV或目标值反推决策。
冻结的 `episode.py` 已有最终解保存入口；是否补存或重新搜索须另行确认，本程序不执行。

## S0 试评定义

使用Table 5中编号最小的S0实例，而非训练validation实例。
仅取Base和Table 5实际使用的shared SL+PPO，各三个原搜索种子的最终前沿。
保留每个最终解，不按目标值去重，不挑选折中解，不筛掉在新场景下被支配的解。
完全相同的决策可共用一次复评，`SOLUTION_MEMBERSHIP.csv`保留其全部来源。

1. 用原`ccp100_seed`和原参数重建100个优化场景，要求场景摘要相同。
2. 重算全部试评解的三个目标，要求与原目标值一致（rtol=1e-10、atol=1e-8）。
3. 使用**Table 5原边界**计算六次运行的HV/1.331，与原`PER_PAIR.csv`比较（绝对容差1e-10）。
4. 上述全部通过后，才用固定种子381000000、381000001、381000002分别新生成100、1000、5000个场景。
   新种子不得与清单内任何优化种子重叠。三组分别调用同一主实验采样器，不截取大场景组的前缀。
   同一组实际网络抽样数组由六次运行的所有最终解共用，并保存NPZ及键映射、摘要。
5. 对三个目标分别取第`ceil(alpha*n)`小的逐场景值；alpha=.9时为第90、900、4500小。
6. 试评新边界是该S0实例**Base/PPO、三个种子、三组新场景、所有最终解**目标的并集极值；
   参考点为(1.1,1.1,1.1)，精确三维HV除以1.331。原优化100场景不参与新边界。

原值复现与新场景比较使用两套明确分开的边界。试评只评价两种方法，
因此不能声称已得到13方法的全量边界，也不能把试评HV直接拼入未来全量表。
全量阶段须重新合并全部相应方法的复评目标；Table 7跨网络对比还须按同一批次组联合大小网络边界。

程序复用 `fixed_q90_s0_s11_v1/evaluate.py` 的精确决策恢复与评价函数，
调用冻结的 `baseline_uncertainty.build_scenario_set`，保留原分布、CV、封顶及均值校准。
不会调用搜索或学习策略。恢复路径不允许公路替代回退，评价前后决策指纹必须一致。

## 输出及口径

默认新目录：`transformer_ppo_runs/diagnostics/ccp_final_front_oos_hv_pilot_v1_<UTC时间戳>`。

| 文件 | 内容 |
|---|---|
| `CODE_VERSION.json` | 当前commit、入口哈希、工作区已跟踪改动、Python版本 |
| `INVENTORY.csv` / `INVENTORY_SUMMARY.csv` | 全部运行、方法及实例的保存情况 |
| `MISSING_OR_INVALID.csv` | 缺失或不一致范围；无问题时只有表头 |
| `SOURCE_FILES.json` / `PILOT_SOURCE_FILES.json` | 本次读取的原始文件哈希，运行前后检查不变 |
| `PILOT_INPUTS.json` | 实际选中实例、原配置、全部最终决策、来源、固定新种子 |
| `scenarios/*.npz` / `*.json` | 原优化及三组新评估场景的实际抽样数组和键映射 |
| `evaluations/*.npz` / `*.csv` | 全部固定决策的分位数、均值、最大值及可行性诊断 |
| `ORIGINAL_HV_REPRODUCTION.csv` / `REPRODUCTION_PASS.json` | 原场景、原目标、Table 5原边界HV核对 |
| `PILOT_INSTANCE_BOUNDS.json` | 新试评共同边界、参考点及除数 |
| `PER_RUN_HV.csv` | 六次搜索×三组评估场景的HV；保存实际搜索评价次数 |
| `PILOT_SUMMARY.csv` | 各方法三组场景的平均HV及相对Base平均配对变化 |
| `SCENARIO_DIFFERENCES*.csv` | 100对1000/5000的逐种子及汇总相对差 |
| `EVALUATION_LEDGER.jsonl` | 实际固定方案评价记录，不是搜索记录 |
| `REPORT.txt` / `STATUS.json` | 中文数据汇报与完成/停止状态 |

场景组比较的主值为逐种子计算`100*(HV_100-HV_n)/HV_n`后平均；
同时报告平均绝对相对差，以及“先平均HV再算比例”这一不同指标，避免混淆。
相对Base的比例逐种子配对计算。试评只有一个实例、一个配置，种子平均就是该试评的汇总值；
全量阶段才执行种子→实例→配置的等权汇总。

终态为`pilot_complete_awaiting_confirmation`，`full_study_completed=false`。
完成后停止，等待用户确认第5步，不自动追加任何运行。

未来alpha=.8来源可通过`--alpha 0.8`和三项来源目录参数使用同一套流程。
入口要求计划和每次搜索记录的alpha均匹配；不会把现有alpha=.9搜索改名为alpha=.8实验。
此时取第80/800/4000小，输出列仍明确标记实际alpha和场景数。

## 已确认的第5步：810次运行的最终解全量复评

在41执行（`--pilot`为已成功完成、不可改动的S0试评目录）：

```bash
cd /home/lunet/cokt2/uncertainty-learning-components/expanda &&
git pull --ff-only origin codex/m-transformer-ppo-threeway &&
python3 -B -u final_front_oos_hv_v1/full.py \
  --pilot /home/lunet/cokt2/uncertainty-learning-components/expanda/transformer_ppo_runs/diagnostics/ccp_final_front_oos_hv_pilot_v1_20261125T144835267018Z
```

单进程、数值库单线程；不与搜索进程共用随机状态。默认创建新的
`ccp_final_front_oos_hv_full_v1_<时间戳>`目录。首先检查全部810个最终解文件及来源，
再对全部运行重现原优化100场景目标和**旧边界下**的HV；全部通过才开始新场景复评。
任何不一致都停止，原结果不修改。

### 场景与网络

- S0的四组场景直接读取成功试评保存的实际数组及哈希，包括原优化100和新100/1000/5000。
  全量加入所有方法后重新建立边界，**不复用试评HV或试评边界**。
- 其他实例从同一冻结主实验采样器生成，保持分布、CV、口岸3.5倍上限及均值校准不变。
  评估种子写入每个`INPUTS.json`，不同组使用不同固定种子；100/1000/5000不嵌套，
  不从5000里截取前100/1000。每个实例所有方法和搜索种子共用实际抽样数组。
- Table7原优化场景读原缓存；新场景先在大网络生成，再按原协议提取小网络共有弧/口岸。
  检查事件映射、均值及每一个共有抽样值完全相同。两个网络的归一化边界按共同批次组联合确定。
- 不改变任何最终决策或份额；新场景下被支配的解也保留。HV计算本身不计被支配点的贡献。
  新场景可行性另存为诊断，不据此删解或修改原来的名义可行集。
- 100、1000、5000在本输出中只表示**独立复评场景数**；全部用原搜索末尾最终解，
  搜索实际4999/5000次逐行保留，400/1000中间前沿不进入新指标。

### 输出

`STATUS.json`显示阶段、实例、场景组、已完成方案/评价包数和按实测吞吐量更新的剩余计算时间。
完成后`RESULT_LOCATION.json`给出本次汇总目录；其中：

| 文件 | 内容 |
|---|---|
| `MAIN_5000.csv` | 主结果：独立5000场景的HV、相对参照变化、胜平负；总体及逐配置 |
| `CONTRAST_1000.csv` | 相同格式的独立1000场景对照 |
| `ALL_SCENARIO_SUMMARY.csv` | 原优化100及新100/1000/5000的汇总；统一使用新边界 |
| `PER_RUN_HV.csv` | 每个实例×种子×方法×场景组的HV、最终解数、边界/参考点外数量 |
| `FINAL_FRONT_COUNTS.csv` | 每个实例×种子×方法的最终方案数及来源文件 |
| `P90_SHIFT_PER_SOLUTION.csv` | 每个保存方案、每个目标的原100与新5000分位数及有符号差 |
| `P90_SHIFT_PER_RUN.csv` | 每次运行的逐目标相对差均值、中位数 |
| `P90_SHIFT_BY_METHOD.csv` | 按方法汇总，列出层级等权均值、加权中位数、合并方案均值/中位数 |
| `SCENARIO_DIFFERENCES*.csv` | 新100对1000、新100对5000、新1000对5000的HV相对差 |
| `ORIGINAL_QUANTILE_COVERAGE_PER_RUN.csv` | 原100的分位数在新5000中的逐目标覆盖率均值、中位数、低于0.85比例 |
| `INSTANCE_BOUNDS.json` | 全部方法、种子、新三组场景最终目标的联合边界 |
| `REPORT.txt` / `REPORT.json` | 中文数字汇总及统计口径 |

根目录另保存`CODE_VERSION.json`（commit与代码哈希）、`PLAN.json`、
`ORIGINAL_HV_REPRODUCTION.csv`、每个实例的决策/来源索引、场景数组、分块评价值和物理评价日志。

相对Base的提升先按同实例同种子配对，再依次种子→实例→配置等权。
Table7单独给出小网络共享监督对小网络Base、大网络共享监督对小网络共享监督，
不假定存在大网络Base。HV除以1.331一次；W/T/L容差为`1e-12/1.331`。
原100场景在新共同边界下可能超出[0,1]或参考点；不裁剪，不改界，超参考点不贡献HV。
旧边界下的原HV复现另表列出，不能把两种边界下的数字混用。

逐目标相对差为`100*(Q新5000-Q原100)/Q原100`，正值表示目标变大；三个目标均为最小化。
主要方法均值先在每次运行内对方案平均，再种子→实例→配置等权。
方法的加权中位数使每个配置、配置内实例、实例内搜索种子均等权，再将该运行权重均分给其方案。
另列全部方案直接合并的均值、中位数和运行中位数的层级均值，列名区分，不能互换。
同一决策在多个运行中出现时每个运行仍保留其成员资格；评价计算可去重。
分母为零时标为未定义并统计个数，不默默排除该方案或配对。
原搜索时间仅复制原值；新复评时间写评价日志，不加进原搜索时间。
本次未要求重算置信区间，汇总不沿用任何旧区间。

### 中断与续算

程序每64个固定方案保存一个带哈希的完成块。中断后只需：

```bash
python3 -B -u final_front_oos_hv_v1/full.py --resume /本次新输出目录
```

续算要求代码、来源和已完成块均未变化；复用完整块，最多重算一个未完成块。
物理日志保留重试次数；无论续算多少次，都不启动搜索。单目录进程锁防止重复启动。
若准备计划前即失败，应先解决明确问题，再用新的输出目录启动。
全量结束后停止，不自动开展alpha=.8搜索、训练或参数修改。
