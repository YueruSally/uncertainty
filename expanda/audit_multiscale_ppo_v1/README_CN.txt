多配置终局 PPO 只读审计

用途：读取已保存的 32 条训练轨迹、update 0/4/8 权重、更新日志和已完成的开发比较。
不训练，不跑搜索，不改变实例、模型、门控、检查点选择或正在运行的任务。
程序在独立的新输出目录写 CSV 和审计摘要，原目录只读。

运行（从 expanda 目录）：
python3 -u audit_multiscale_ppo_v1/audit.py --out transformer_ppo_runs/diagnostics/ccp_icm_multiscale_joint_v1_policy_audit_v1

默认：每条训练轨迹每个组件最多固定抽取 128 个候选池，不依赖奖励/预测值抽样。
--max-pools 0 可使用全部保存的候选池，但开销更大；默认只使用 1 个 CPU 线程。

POLICY_CHANGE.csv：同候选 KL(u0 || checkpoint)、熵（自然对数）和首选变化率，组件内轨迹等权。
REPLAY_PER_TRAJECTORY.csv：逐轨迹、逐组件数据；候选来源是训练轨迹，不是验证或开发运行。
UPDATE_LOG.csv：已保存的优化步数、KL 提前停止和最后一个 minibatch 的近似 KL。
ADVANTAGES.csv：严格按现有实现计算 R-V_old 的均值、标准差和正值比例。
UPDATE0_REPRODUCTION.csv：已有共享监督与所选 PPO 开发运行的 HV、前沿、评价顺序摘要、计数和选择计数精确比较。
SHARED_UPDATE0_CHOICES.csv：同输入上两种实际部署计算路径是否选同一候选。
VALIDATION_PAIRED.csv：既有单种子验证配对差值；不生成种子不确定性区间，不重新选择模型。
QUICK_RESULT.json：审计结论、范围和限制。

未保存每个优化步的熵、裁剪比例或梯度，不能通过终点重放恢复这些历史量。
小 KL 不等于证明更新被裁剪；大 KL 加负收益不等于证明奖励错误。
单种子配对可以降低比较噪声，但无法估计同实例换种子的波动。
终局 HV 使用训练归一化，不是成本等原始物理单位；当前没有配对 update0 差分奖励，也没有优势标准化。
3 个成员组成同一集成策略，不是 3 次独立 PPO 实验。
只有实际完成且通过检查的结果才计入重现检查；未执行任务不会被伪造为完成，时间不复制。
本脚本不停止或删除重复任务。修改正在执行的冻结调度应另行版本化，并记录复用来源和未执行次数。
