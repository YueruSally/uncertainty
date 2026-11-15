第6.3节：新批次与规模泛化，固定版本 final-generalization-b5000-v1

用户确认：S0-S11每配置3个新实例×3个共同种子×4方法=432次搜索。
Base；Simple（C_RULE，仅交叉采用Simple，其余操作Base）；共享监督；PPO v2 u1。
pop250，B5000，K8，每实例固定100个优化情景，三个目标经验q90。
进度输入仍除以50000，沿用训练定义；不修改模型/门控/损失/运输评价。
保存400、1000、5000节点；真实方案评价计费，包含初始化和可行性增强。
原子双子代可能在4999结束，记录实际数；不插入虚假的第5000次评价。
算法种子1001001-1001003；实例和情景生成种子见protocol.json和实例清单。

1. 新实例：同原训练OD池、货量与释放时间分布，生成全新的整组货运批次。
   S0-S8批次数10/20/30；S9-S11为50；时间窗倍率沿用原SCENARIOS。
   本轮不是“未见OD池”的泛化，也不额外改成奇数货量/释放网格。
   对diagnostics各实验的实例清单核对整组批次签名；碰撞直接报错，不悄悄重抽。
   旧S11曾用于方法开发，不能声称S11配置从未被接触；新实例保持隔离。
2. 事先固定共享监督为B5000主方法，PPO u1保留为联合强化学习比较。
   u1是v2三次更新中较好的更新后模型，含u0的调参最优仍是u0。
   测试不训练、不重选检查点、不根据结果增加版本或挑选实例/种子。
3. 所有主表的百分比均相对同实例/同种子的Base，先实例内平均，再配置等权。
   新批次S0-S8（每方法81次）和规模S9-S11（每方法27次）分开报告。
   95%区间在固定配置内依次抽实例和种子，保留配对；零Base不加epsilon。
4. results_union为预先规定的报告口径：同实例，四方法×三种子×三个节点
   的可行目标点共同确定min/max，参考点(1.1,1.1,1.1)，不除1.331。
   results_frozen同时保留训练期冻结归一化的结果；不能按结果挑更好的一套。
   两套报告都不影响模型或训练奖励。跨新旧实验的HV绝对值不可直接相减。
5. 每次实际运行保存ENVIRONMENT.json和完整原始前沿解；记录搜索、推理、
   特征构造、目标评价耗时及峰值RSS。两进程各1个Torch/BLAS线程，
   不等于服务器只分配2核，也不等于独占这2核。未知的资源预留不填虚构值。
6. 只使用100优化情景；没有另开5000独立情景重评。对应“新实例泛化”，
   不应写成已经验证独立情景下的可靠性。

启动：bash final_generalization_b5000_v1/run.sh --workers 2
只固定实例和计划、不运行搜索：在命令后加 --prepare-only。
默认目录：transformer_ppo_runs/diagnostics/ccp_final_generalization_b5000_v1
中断后同一命令续跑；已完成结果按任务/计划哈希复用，中断目录保留。
绝不覆盖原开发实验或修改原模型文件。无自动后续实验。

输出：
results_union/00_新实例测试_相对Base.csv：3预算×4方法×2范围
results_union/01_终点结果.csv：B5000主表，含绝对HV、配对提升、区间、胜平负、耗时
results_union/PER_RUN.csv、PER_CONFIGURATION.csv：配对明细、分配置结果
results_union/INSTANCE_BOUNDS.json：实际共同归一化界限
results_union/DEPLOYMENT_COSTS.csv：逐次推理、特征、目标评价和总搜索耗时
results_frozen/：同样结果，使用训练期冻结界限
fronts/FINAL_SOLUTIONS.jsonl.gz：432次最终前沿，包含目标和路径/份额方案
runs/.../final_feasible_nondominated.json：每次原始解文件
PLAN.json、EXCLUSION_AUDIT.json、ENVIRONMENT_LAUNCH.json、CONCURRENCY.json、REPORT.json

时间只能估算：按开发集实测均值，S0-S8部分324次约需4.3小时（2进程，
不含启动/汇总/竞争）；S9-S11另108次需按新运行实测追加，不能假定同速。
