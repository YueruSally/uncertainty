与完整联合训练并行的小范围早期检查

固定范围：S0、S4、S8，每配置字典序第一个验证实例，每实例一个共同种子。
四组：Base、ICM独立Transformer、ICM共享监督、完整训练中固定更新2的终局PPO。
pop250，B5000，CCP100，原训练归一化。预算进度分母继续固定50000。
共12次搜索、最多60000次新方案评价，零隐藏候选评价。

完整实验继续按原先的pop250/B50000训练及比较，不受本程序控制。
本程序只读取完整实验已发布的权重，不修改、不暂停、不启动它，不新增PPO训练。
小试只检查早期效果，不选择checkpoint，不淘汰组件，不修改完整实验设置。
每配置只有一个验证实例、一个种子，且验证实例已用于选择监督模型；结果不作独立测试结论。
因此不输出不可靠的种子置信区间。原始训练归一化为主，最终另给四组统一min/max敏感性表。

执行（expanda目录）：
bash joint_icm_fastcheck_v1/run.sh
默认输出：transformer_ppo_runs/diagnostics/ccp_icm_multiscale_fastcheck_b5000_v1
默认额外使用1个worker，线程数1，不调整正在运行的完整实验worker数。

程序先只读检查旧S0和当前S0-S8的checkpoint训练回报、配对验证终局HV。
训练回报在不同更新轮次来自不同实例和种子，其均值不能当作配对学习曲线。
只对完整验证面板计算相对update0的配置等权差，并核对已保存的模型选择。
检查结果保存在 CHECKPOINT_AUDIT.json，也打印在主日志开头。

然后依次运行Base、共享监督、独立训练、PPO。每组完成就更新：
results/00_当前小试对比.csv
results/PER_CONFIGURATION.csv
results/PER_RUN.csv
results/EXECUTION_DIAGNOSTICS.csv
前三组完成后已可看共享与独立的差别。
如果独立模型或update2尚未完成，显示waiting_for_model，每30秒报告完整实验阶段。
等待不会另开训练。若完整实验已暂停或失败，保留已完成的小试并提示；同命令可续跑。
模型等待上限24小时。等待时间不计作搜索耗时。
PPO更新2由方案固定，无论结果好坏均报告，不能替换为表现更好的其他checkpoint。

与完整实验同时运行会争用CPU；运行时间仅作描述，不能直接当作无争用时的速度比较。
同一输出目录有进程锁，完成的搜索只读复用。失败尝试保留在interrupted，不删除。
重复运行不会启动新的完整实验，也不会覆盖任何历史模型。

本地检查：python3 joint_icm_fastcheck_v1/checks.py
