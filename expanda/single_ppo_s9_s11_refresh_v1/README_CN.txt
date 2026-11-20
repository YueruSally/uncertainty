仅补新版单模型 PPO 在 S9–S11 的 27 次搜索

在 expanda 执行：bash single_ppo_s9_s11_refresh_v1/run.sh --workers 2
默认输出：transformer_ppo_runs/diagnostics/ccp_single_ppo_s9_s11_refresh_v1
重复同一命令断点续跑；支持 --prepare-only，仅检查、冻结输入。

模型：ccp_single_ppo_s0_s8_b5000_v1 的 SELECTION.json 选定更新。
自动核对 u0/u1/u2/u3 的 S0–S8 验证选择记录和单成员权重。
对照：ccp_single_s9_s11_b5000_v1 的已完成 Base、共享监督，分别复用27次。
核对共享监督权重、门控、输入投影维数、缩放和特征定义一致才允许复用。
若验证选回u0则退出提示直接复用共享监督，不启动重复PPO搜索。

沿用原 S9/S10/S11 各3个实例、每实例3种子：27个配对。
不生成新实例，不重新训练、选模型或修改门控；只新增PPO搜索。
pop250/B5000/CCP100/置信水平0.9；400/1000/5000是累计方案评价节点。
原始结果、模型和运行目录只读；新结果另存目录。
已有大场景结果已被查看，本轮是固定新版模型的追加对照，不能声称这些实例从未被查看。

results_union/RESULTS.csv：Base、共享监督、新PPO，提升均相对Base。
results_union/PPO_vs_SHARED.csv：新PPO直接对比共享监督。
results_union/PER_CONFIGURATION.csv：S9/S10/S11分别报告，含两类参照。
results_union/PER_PAIR.csv：逐次HV和配对差。
results_frozen/：相同方案在冻结训练边界下的报告。
所有以上报告的HV及绝对差/区间均已除以1.331，不要重复除。
百分比、百分比区间和胜平负不受常数缩放影响。
保留原四组前沿作为归一化池，加上新PPO，同实例所有种子和节点共用边界。
Simple和旧PPO仅用于保留历史归一化池，不出现在新主表，不重跑。
新增前沿可能改变共同边界，故Base/共享HV会同步重算。

节点累计耗时从控制器创建开始；完整B5000搜索时间还含此前准备与终点处理。
两列分别报告，不将节点时间当作完整搜索时间。区间按配置内实例、种子分层配对bootstrap。
文件保存原始前沿以及最终可行方案，便于之后重算指标。

预计：按旧PPO约369秒/次、两个进程，纯搜索约83分钟；另留来源检查、汇总和负载余量。
不承诺改善，所有27次按固定计划完成并报告。
