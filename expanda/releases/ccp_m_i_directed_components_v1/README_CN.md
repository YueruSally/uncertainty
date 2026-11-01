# CCP 定向组件源码包：M v1 + I v1

## 包的用途

这个包把目前已经完成的两个单组件版本分开保存：

1. **M（变异）**：`REPLACE_G10_JOINT_BALANCED`
2. **I（初始化）**：`I_DIRECTED_BALANCED / joint_balanced`

两个版本都在不确定性与 CCP 框架下运行：每个方案在冻结 SAA 情景中完整计算运输、班次等待、边境延误、成本、排放和完成时间，再分别取三个目标的 q90。两者都属于确定性规则教师，不是 PPO 已经学会的策略。

## 最终结果摘要

| 组件 | 冻结 CCP held-out 平均 HV | 独立 OOS 平均 HV | OOS 95% CI | OOS 胜负 | 结论 |
|---|---:|---:|---:|---:|---|
| M | +12.82% | +9.42% | [+7.48%, +11.35%] | 64 胜 / 6 负 | 平均和安全性较稳定 |
| I | +22.46% | +16.35% | [+12.06%, +20.63%] | 53 胜 / 17 负 | 平均泛化强，尾部风险较高 |

详细方法、实验步骤和结果边界分别位于：

- M：`frozen_m_tag/expanda/releases/ccp_m_directed_balanced_v1/`
- I：`current_source/expanda/releases/ccp_i_directed_balanced_v1/`

## 代码结构

下载压缩包解压后包含：

```text
ccp_m_i_directed_components_v1/
├── README_CN.md
├── UPLOAD_TO_GITHUB.md
├── current_source/
│   ├── requirements.txt
│   └── expanda/
│       ├── baseline_uncertainty.py
│       ├── transformer_ppo/
│       ├── M/I 实验入口与评价脚本
│       ├── tests/
│       ├── data/
│       └── releases/ccp_i_directed_balanced_v1/
└── frozen_m_tag/
    └── expanda/
        ├── M v1 精确标签时的核心代码
        └── releases/ccp_m_directed_balanced_v1/
```

`current_source` 是当前可同时支持 M 和 I 的代码树；`frozen_m_tag` 是从 Git tag `ccp-m-directed-balanced-v1` 直接提取的 M 精确历史快照，避免后续 I 修改共享运行器后破坏 M 的可追溯性。

## M 做了哪些调整

- 从第 10 代开始强制定向 Replace。
- 不再随机选择批次和替换路径，而是枚举结构可行的“批次 × 路径”组合。
- 同时使用成本、排放、时间、最大迟到和准时概率五类信号。
- 不增加真实目标评价次数，仍为 400 次预算。
- 在 held-out 70 对中平均提升 12.82%；在 1000 个独立情景中平均提升 9.42%。

## I 做了哪些调整

- 保留 Base 的 12 个固定初始种子。
- 只控制最后 8 个随机初始化个体的路径身份。
- 保留 Base 生成的路径数量、分配比例与随机流。
- 使用与 M 一致的五类信号选择 Balanced 路径。
- 不增加真实目标评价次数，仍为 400 次预算。
- 在 held-out 70 对中平均提升 22.46%；在 1000 个独立情景中平均提升 16.35%。

## 不进入最终版本的实验

I 的 `joint_balanced_hv_gate` 已明确淘汰。它为了比较 Base 和 Directed 初始化额外消耗 8 次目标评价，导致后续搜索预算从 380 次降为 372 次；初始 HV 也不足以稳定预测最终 HV。相关代码仅为审计保留，正式运行必须选择 `joint_balanced`。

## 外部输入

为了控制下载体积，本包不包含：

- 服务器生成的实例 JSON；
- `reward_normalization.json`；
- checkpoint；
- 数百个逐运行结果目录；
- 原始逐情景候选数组。

代码、测试、网络输入表和结果摘要均已包含。复现实验时需要在服务器重新提供上述外部路径。

## 论文/汇报时的准确表述

可以表述：

> 在冻结 CCP 情景、三个 q90 目标、固定训练归一化和相同 400 次真实评价预算下，Directed Balanced 变异与初始化规则均显著提高最终 HV。独立 1000 情景复评中，M held-out 平均提升 9.42%，I held-out 平均提升 16.35%。

必须同时说明：

- 当前提升来自规则教师，不是 PPO 自主学习；
- I 的平均提升强，但逐运行安全性弱于 M；
- C 尚未完成，I+C+M 尚未作为最终结论。
