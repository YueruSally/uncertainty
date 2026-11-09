旧版小预算组件结果扩大验证（第一组实验）

固定范围
10 个新 S0 实例 × 10 个共同算法种子 × 7 种方法 = 700 次搜索。
pop20、B400、CCP100、经验 q90、K8，最多 280,000 次方案评价。
保留原来的 1000 代保护上限和原评价计数实现；每次必须实际完成 400 次评价。
不是 B20 重训实验，也不是最终测试。没有训练、PPO、ICM 组合或独立情景重评。

七种方法
BASE
I_DIRECTED_BALANCED / I_K8_RESIDUAL
C_K8_SIMPLE / C_K8_RESIDUAL
M_BALANCED / M_K8_TRANSFORMER

模型与代码
直接复用 formal_icm_s0_v1 中打包的历史算法引擎及 9 个权重。
I：926001–926003；C：928001–928003；M：size200 的三个冻结模型。
实际候选构造、特征、标准化、门控、Base 搜索逻辑均保持原样。
M 保持原有第 10 代之后启用。I 保持改写全部初始位置的旧规则，
本轮不加入“保留固定种子”修正，以便检验旧实验是否可复现。
原文件不修改；只是重定位权重路径，并将 SHA256 固定到 PLAN.json。
零 HV、没有符合启用条件的决策都作为实际结果保留，不据此换实例或种子。

新实例
从原 catalog 的训练 OD 域生成 20 批货、期限倍率 1.0 的新 S0 实例。
生成种子固定为 126000000 + 100*i，CCP 种子 128000000 + i，i=0..9。
路径种子用原构造可行性检查流程确定（无目标评价），算法种子 961001–961010。
检查 diagnostics 下已有 catalog、原始实例 catalog 和打包资产中的 catalog：
不复用已有生成种子/情景种子或批次签名，遇到冲突停止，不根据结果换样本。
只读取实例清单，不读取最终测试结果。S9–S11 和最终测试继续保留。
实例及生成来源哈希保存在本轮 instances/ 与 GENERATION_PLAN.json。

启动（expanda 目录）
python3 -u component_recheck_b400_v1/run.py --workers 1
默认输出 transformer_ppo_runs/diagnostics/ccp_component_recheck_b400_v1。
默认读取服务器原始 catalog：
/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances/catalog.json
如原目录迁移，用 --source-catalog 指向同一份清单，内容哈希必须一致。
只准备清单及冻结模型可用 --prepare，不执行搜索。

资源与续跑
默认 1 个工作进程，每进程计算线程为 1。允许 --workers 2，但耗时比较要注意资源条件。
每 30 秒输出心跳；单次搜索保护上限 15 分钟，单次调用上限 8 小时。
输出目录有排他锁，防止重复启动。中断后用相同命令续跑，已完成结果不再计算。
未完成尝试保存在 interrupted/；其部分消耗不包含在完成运行的计费合计中。
不覆盖任何历史实验、模型或数据。

结果
RESULT_TABLE.csv：相对 Base 的均值、95% 区间、胜负次数、耗时；学习版相对对应规则的均值与区间。
QUICK_RESULT.json：上述统计及每实例结果、方案评价次数、范围说明。
ALL_RUNS.json：700 次原始运行摘要；每次的轨迹和配置仍在 runs/。
STATUS.json：当前进度；PLAN.json：完整冻结配置、代码/数据/权重哈希和运行环境。
先在实例内汇总配对种子，再对实例等权；区间重采样实例及实例内配对种子。
参考 HV 为零时，不加小常数；总体百分比标为未定义，报告全体绝对差。
本轮不自动准入、改门控、改模型或启动 PPO。

本地正确性检查
python3 component_recheck_b400_v1/checks.py
使用已有样本和测试清单完成七方法小预算验证，不计入新实例的研究结果。
