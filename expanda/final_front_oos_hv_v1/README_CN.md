# 最终解集独立场景复评：完整性检查与 S0 试评

本入口只实现确认后的第1—4步。没有搜索、训练、模型选择或全量重算入口。
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
