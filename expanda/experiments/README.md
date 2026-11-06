# M 完整搜索反馈实验

`run_m_terminal_feedback_v3.sh` 复用已完成的 `single-intervention-ppo-v2`。仍在第一个可用 G10+ 决策点选择一次，其余执行原 Balanced；所有搜索为 CCP100、pop20、B400。

1. 读取原 64 条训练轨迹及配对 Balanced。只补跑缺失的合法动作，各自执行到搜索结束；这些完整反事实搜索全部计入训练成本。
2. 使用全部动作的最终 `HV/HV_Balanced−1` 训练 Transformer，目标为概率加权的期望回报。训练使用 6 个实例，014/016 两个实例仅用于验证；原预训练模型不得用过这两个实例。
3. 全反馈训练最多 40 轮，固定每 5 轮验证。接着做 4 次真正的在线 PPO 更新：每次 6 个完整搜索、各一条学习决策。仅用当前策略采样的数据更新 PPO。
4. 验证直接对保存的各分支结果按策略概率求期望；这不是取最好分支。最终在 021/022 的原 20 个配对种子上实际运行预训练与 PPO 策略，并复用完整匹配的 Balanced。

部署只执行一个采样动作，无额外候选评价。最终表同时给出 Balanced、全反馈预训练和预训练后 PPO。选择 PPO 第 0 次时会明确标记，不能宣称 PPO 产生增益。

新搜索数为 `缺失动作分支数 + 48 次 PPO/配对基线 + 80 次开发搜索`，至多 384 次，每次 400 次评价。旧实验、原预训练和中断成本另行披露。Balanced 耗时来自已完成的源实验，不用于声称严格的同时硬件加速优势。

在 `expanda` 中设置 `INSTANCES`、`NORMALIZATION`，执行：

```bash
export SOURCE_ROOT=transformer_ppo_runs/diagnostics/ccp_m_single_intervention_ppo_v2
export OUT_ROOT=transformer_ppo_runs/diagnostics/ccp_m_complete_terminal_feedback_v3
bash experiments/run_m_terminal_feedback_v3.sh
```

进度是 `STATUS.json`，训练是 `PRETRAINING.json` 和 `TRAINING_PROGRESS.json`，最终结果是 `RESULT_TABLE.csv`、`QUICK_RESULT.json`。一次调用限 45 分钟；同一命令续跑复用分支与已提交更新。通过仍需新实例和独立情景验证。

思路参考 [Bertsekas、Tsitsiklis、Wu 的 rollout 方法](https://www.mit.edu/~dimitrib/rollout.pdf)：尝试一个动作后沿用基准策略，用后续完整结果评价该动作。本文的完整反馈用于训练近似网络；其泛化或性能改进没有该论文条件性理论保证。

## 固定延长训练：40 → 400 步

`extend_terminal_feedback_training.py --source <完整 v3 输出目录>` 只读取已保存的动作最终收益，不启动搜索或 PPO。数据、网络、学习率、损失和实例划分不变；先从原始权重重建 Adam 前 40 步，核对旧训练指标和选中权重完全复现，再训练至 400 步。新增选模点预先固定为 80、120、200、300、400，保留原模型作对照。运行上限 120 秒，输出到源目录下独立的 `training_extension_400_v1`。

结果是两个验证实例上基于已存分支的条件期望收益，不是新的搜索结果。只有两个实例均为正才建议小预算实际搜索；不自动启动搜索或 PPO。重复使用这些验证实例属于开发过程，仍需全新实例确认。完成后直接打印结果表，重复执行复用已完成结果。

`check_terminal_decoding.py --source <同一 v3 输出目录>` 固定延长试验已经选中的权重，只比较原概率抽样与最大概率动作（并列时选 Balanced）。只查已有分支结果，不扫描温度或阈值、不训练、不新增目标评价。输出逐实例条件收益和动作比例；标签只用于评分，不参与选择动作。该比较属于新的部署规则诊断，不能称为修复代码错误或 PPO 收益；即使验证为正，仍须实际搜索和独立确认。

## 扩大训练实例覆盖

`expand_terminal_training_instances.py --source <同一 v3 输出目录>` 增加训练实例 011、015、017、018、019、020，各用固定种子 902001–902008。它们来自已有训练目录，不属于新独立测试集。新增实例不得与原训练、验证、开发实例重名或具有相同批次内容。只新增 48 个训练状态，原 48 个训练状态和 014/016 的 16 个验证状态全部保留。

每个新状态先执行 Balanced，再补齐其余合法动作的完整终局反馈。共至多 240 次 CCP100/pop20/B400 搜索（96,000 次评价），所有分支均计入训练成本。复用同一个实际算子执行函数，核对相同初始种群、情景、动作前输入和预算。2 个单线程工作进程，单次调用限 30 分钟；同一命令可复用已完成分支和已提交训练。

网络、特征归一化、原始预训练初始化、学习率及损失不变，训练 400 步；收益尺度仍只从训练数据计算。固定在 40、80、120、200、300、400 步选模。报告原 6 实例和扩展 12 实例模型在相同验证状态上的 softmax 期望收益与 argmax 收益。只有两个验证实例的 argmax 收益均为正，才建议冻结该策略做实际搜索；不会自动启动搜索或 PPO。选中模型标记为 `single-intervention-argmax`，不得交给旧的随机抽样执行器。最终仍需新实例与独立情景确认。

默认结果目录为 v3 同级的 `ccp_m_terminal_coverage_12inst_v1`。进度见 `STATUS.json`，最终见 `RESULT_TABLE.csv` 和 `QUICK_RESULT.json`，成本包含本轮完整搜索、原实验成本及未提交中断成本的说明。
