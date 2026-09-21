# Learning-CCP100：位置选择学习

只运行CCP100，不覆盖旧CCP100/300结果。五个mutation算子概率固定相等；
Random、Rule和Learning只改变算子抽中以后的位置选择。

## 在仓库根目录运行

```bash
python -m pip install -r requirements.txt
python expanda/run_learning_ccp100.py --out expanda/learning_runs/random_smoke --pop 20 --gens 5 --evaluation-budget 500
```

确认小规模日志正常后，运行数据收集pilot（不是正式结论）：

```bash
python expanda/run_learning_ccp100.py --out expanda/learning_runs/random_a981101_s981201 --pop 100 --gens 100 --evaluation-budget 15000 --algorithm-seed 981101 --training-seed 981201
```

每次使用新的输出目录，程序拒绝覆盖已有内容。没有断点续跑功能。
`--gens`和`--evaluation-budget`哪个先到就停止；想按预算比较时将`--gens`设得足够大。
预算包含初始种群、mutation前后的真实评价、boost评价；复用同一个体相同决策的结果不计为新评价。
每对offspring保守预留4次评价，可能剩余不足4次未使用；boost保守预留80次，剩余不足时跳过。
因此正式比较还需同时报告实际评价数、boost次数及停止原因，不能只报告N×G。
所有位置方法必须使用相同计数/预算规则。路径库构建、路径级仿真和日志开销不混充完整solution评价。

可选：在**同一次新运行结束后**对所有最终原始Pareto方案做5000 OOS验证：

```bash
python expanda/run_learning_ccp100.py --out expanda/learning_runs/random_with_oos --pop 100 --gens 100 --evaluation-budget 15000 --validate-oos
```

OOS不参与位置特征、训练标签或优化，不进行OOS后candidate过滤。
日志pilot阶段不必运行OOS。正式实验冻结设计前不要反复使用最终OOS调参。

## 改动边界

- 五个算子add/del/mod/mode/replace的**尝试概率**均为0.2；失败不重新抽算子。
- 总mutation触发概率仍为0.15，crossover仍为0.90。
- CCP100固定100场景、三个独立经验q90（第90个顺序统计量）。
- 不改变两类随机源、时刻表、碳成本、等待排放或容量语义。
- 当前容量检查是名义计划时刻下的容量，不是100场景容量均值/q90。
- mod仍只改share，mode仍只改一条arc，replace仍替换整个batch为单路径。
- 保留现有各算子内部merge/normalise；统一结构repair再次验证编码、补缺失分配；非法mutation回滚。
- 统一repair对仅有浮点舍入误差（share总和与1相差不超过1e-12）的合法编码保持幂等，不重复归一化。
- 不新增容量repair；现有road fallback仍保留在原来路径构建/重建位置。
- 原baseline命令不启用日志；请使用新入口。该分支上的五个概率对旧入口也已改为0.2。

## 位置与失败样本

add/replace选择batch；del/mod选择batch+allocation；mode选择batch+allocation+arc。
候选路径、新mode和share幅度仍由原算子选择。
Random保留分层均匀概率，而非对全部arc全局均匀抽样。
支持集中保留不可执行位置并标注eligible=false（如单路径删除/改单路径share、无替代mode），以保留真实失败样本。
Rule/Learning后续不能在未声明的情况下删掉这些候选改变对照。

## 日志

| 文件 | 内容 |
|---|---|
| configuration.json | 场景digest、seeds、代码/输入SHA256、预算、单位、概率 |
| mutation_events.jsonl | 每次尝试、before/after q90、可行性、违反、repair、存活标记 |
| mutation_candidates.jsonl | 当次全部位置、特征、抽样概率、是否被选中 |
| generation_summary.jsonl | 真实评价数、复用次数、累计有效修改率、repair率、训练front、boost |
| final_feasible_nondominated.json | 原始训练非支配可行决策（只按决策去重） |
| best_infeasible.json | 无可行解时的诊断，不伪装成可行Pareto结果 |
| COMPLETE.json | 成功完成标记、实际评价数、事件数、运行时间 |
| oos_summary.json | 仅指定--validate-oos时输出，Median/q90/coverage等 |

before是**交叉后、变异前child**，不是mating parent。delta=before-after，正值为改善。
失败/回滚/决策不变时复用before结果；成功但不变（单路径mod）不能算有效修改。
raw_mutation_changed记录repair前的算子变化，repair_changed_decision记录repair是否改变决策，
decision_changed记录repair后的最终变化。effective_mutation要求算子成功、repair前确实改变且最终未回到原决策。
repair-only变化不归因于所选mutation位置，也不能作为该位置的目标值标签。
不可行方案仍保留，但objective_label_eligible=false，不能把少运货造成的成本下降当改善。
非有限数保存为JSON null，并保留finite_objective_label；不能把null填成零标签。
repair日志不谎称做过repair前完整CCP评价；目前只记录before与修复后违反情况。
各算子内部原有归一化不计入统一repair_action_count。
boost事件另标phase=boost，不与普通offspring成功率混用。
survived_environmental_selection只指当代NSGA-II选择；boost另用retained_after_boost。
训练front保留原始值；本阶段不生成动态缩放的HV伪曲线，也不宣称预测精度或Learning收益。
未来按run/seed分割训练验证集，不能随机拆分高度相关的逐次变异样本。

## 测试

在expanda目录执行：

```bash
python -m unittest tests.test_learning_ccp100_logging tests.test_baseline_uncertainty
```

## 冻结数据与训练模型

训练数据只连接每个事件中实际chosen的candidate和实际outcome，不给未选择位置伪造反事实标签。
训练/验证必须按完整run划分；OOS不进入特征或标签。

```bash
python expanda/build_learning_dataset.py \
  --runs-root expanda/learning_runs \
  --pattern 'random_final_v3_run*' \
  --train-runs 1-7 --validation-runs 8-10 \
  --out expanda/learning_dataset_v3

python expanda/train_location_model.py \
  --dataset expanda/learning_dataset_v3/selected_mutations.csv \
  --manifest expanda/learning_dataset_v3/dataset_manifest.json \
  --out expanda/learning_model_v3
```

模型目标为Pareto-promising：mutation有效且repair后可行，并且child支配before，
或在before本身可行时形成非支配trade-off。训练使用Random日志的截断逆倾向权重。

## Rule与Learning运行

Rule保留Random的分层概率，并在eligible集合上进行条件化；若没有eligible位置则回退Random。
Learning只对eligible位置评分，默认90%选择最高分、10%按原Random概率探索；
日志保存实际混合策略概率，所有候选始终保持非零探索支持。

```bash
python expanda/run_rule_ccp100.py --out RULE_OUT --pop 100 --gens 1000 \
  --evaluation-budget 15000 --algorithm-seed 1 --training-seed 2

python expanda/run_learning_policy_ccp100.py \
  --model expanda/learning_model_v3/location_model.joblib \
  --out LEARNING_OUT --pop 100 --gens 1000 --evaluation-budget 15000 \
  --algorithm-seed 1 --training-seed 2 --epsilon 0.10
```
