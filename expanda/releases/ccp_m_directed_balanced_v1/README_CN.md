# CCP-M Directed Balanced v1

## 版本定位

这是当前已经通过冻结 CCP 优化实验和独立情景复评的 **M（变异）规则版本**。

- 冻结策略名：`REPLACE_G10_JOINT_BALANCED`
- 版本标签：`ccp-m-directed-balanced-v1`
- 核心实现提交：`bce8532`
- 评价口径：成本、排放、完成时间三个目标分别使用 q90；HV 使用固定训练归一化。
- 结论边界：这是一个确定性的 Directed Balanced 变异控制器，也是后续 M Actor 的教师/先验；当前结果不是 PPO 自主学习结果。

## 我们修改了什么

### 1. 把确定性目标评价改成冻结情景 CCP

每个候选方案在同一组冻结随机情景中评价运输时间和边境延误。每个情景完整重算运输、等待班次、成本、排放与完成时间，然后分别取三个目标的 q90。优化期间使用：

- `transport_border_saa_v1`
- 200 个冻结情景
- `alpha = 0.90`
- 不确定性：公路/铁路/水运时间 CV 分别为 0.15/0.10/0.20；边境延误 CV 为 0.50
- 训练专用固定归一化范围和 HV 参考点

### 2. 证明原来的问题不在 Replace 操作本身

早期 I、C、M 和 ICM 的学习实验大多没有超过 Base。对 M 的单次干预、延续稳定性、候选排序和算子筛查表明：

- Replace 确实存在高收益机会；
- 随机选择“哪个批次、替换成哪条路径”会浪费这些机会；
- 单次最终收益标签噪声大，直接训练候选排序器不稳定；
- 仅调整 PPO 学习率、轮数或网络结构不能解决动作生成质量问题。

### 3. 新增 Joint Balanced 定向 Replace

每次需要执行 Replace 时，枚举所有结构可行的“批次 × 替换路径”组合，不增加真实目标评价次数。对每个组合计算五个代理量：

1. `批量 × 单位成本变化`
2. `批量 × 单位排放变化`
3. 名义运输时间变化
4. 最大迟到时间
5. 负的准时概率（最小化它等价于偏好更高的准时概率）

在当前动作集合内逐列 min-max 缩放，然后计算：

```text
balanced_score = scaled_cost + scaled_emission
                 + (scaled_time + scaled_lateness + scaled_negative_reliability) / 3
```

选择分数最小的批次和路径。正式版本从第 10 代开始，以 100% 概率执行 Replace；仍消耗与 Base 相同的 400 次真实目标评价预算。

### 4. 增加严格的配对验证和独立 OOS 复评

- Base 与 Balanced 使用相同实例、算法种子、CCP 情景、预算和归一化。
- 策略筛选只使用种子 777001–777003。
- 777004–777010 是未参与策略选择的 held-out 算法种子。
- OOS 阶段不重新优化，只把冻结的 Base/Balanced 最终 Pareto 解放入 1000 个新情景复评。
- OOS 使用种子 880001，且情景摘要与训练情景不同。
- 所有 100 对 OOS 结果共享一个 pooled Base+Balanced HV 参考点，避免单个实例的 Base HV 为零，同时不改变任何固定决策。

完整数字见 [RESULTS.json](RESULTS.json)，核心文件校验值见 [VERSION_MANIFEST.json](VERSION_MANIFEST.json)。

## 主要结果

### 冻结 CCP 优化内的 held-out 70 对结果

- 平均 HV 提升：**+12.8155%**
- 95% CI：**[+10.1994%, +15.4316%]**
- 中位数：**+9.2067%**
- 68 胜 / 0 平 / 2 负
- 53/70 次提升至少 5%
- 34/70 次提升至少 10%

### 1000 个独立新情景上的 held-out 70 对结果

- 平均 HV 提升：**+9.4172%**
- 95% CI：**[+7.4831%, +11.3512%]**
- 中位数：**+8.0770%**
- 64 胜 / 0 平 / 6 负
- 49/70 次提升至少 5%
- 27/70 次提升至少 10%

因此可以严谨地说：**当前 M 规则在独立不确定性情景上已经超过 5%，held-out 平均提升约 9.42%。** 不能说 PPO 已经学会这一提升。

## 如何取回和运行这个版本

```bash
git fetch --tags origin
git checkout --detach ccp-m-directed-balanced-v1
cd expanda
```

正式冻结 CCP 验证入口：

```bash
bash run_m_directed_balanced_validation_v1.sh
```

独立 OOS 复评入口：

```bash
bash run_m_directed_balanced_oos_v1.sh
```

服务器上已完成的证据目录：

```text
transformer_ppo_runs/diagnostics/ccp_m_directed_balanced_validation_v1_10inst_10seed_b400
transformer_ppo_runs/diagnostics/ccp_m_directed_balanced_oos_v1_10inst_10seed_s1000
```

代码包不包含服务器上的实例目录、训练归一化文件、checkpoint 和大体积逐运行输出；这些属于外部实验输入/证据，路径和摘要均已记录在版本清单中。

## 下一步原则

1. 冻结本版本，不再利用当前 OOS-1000 结果调整权重。
2. 将 Balanced 作为 M 的教师/强先验，让 Actor 学习“何时采用教师、何时回退 Base”，而不是重新学习一个更弱的随机 Replace。
3. I 和 C 先各自建立可超过 Base 的规则教师，再做学习；不能直接回到一起训练 ICM。
4. I、C、M 都要单独报告；最后的组合实验只作为增量验证，不替代单组件结果。
5. 最终论文验证保留一套从未用于开发的共同 OOS-5000 情景。

详细的 I/C 实施顺序见 [NEXT_I_C_PLAN.md](NEXT_I_C_PLAN.md)。
