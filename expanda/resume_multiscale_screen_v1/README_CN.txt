多配置联合实验：省去重复 update 0 的接续程序

只用于已经完成训练、验证选回 update 0 的旧实验。
保持原模型、实例、算法种子、优化情景、预算、搜索代码和已完成结果。
不启动新 PPO，不改变检查点选择，不根据 HV 好坏筛选运行。

启动（从 expanda）：
python3 -u resume_multiscale_screen_v1/resume.py --take-over --workers 2

默认源目录：transformer_ppo_runs/diagnostics/ccp_icm_multiscale_joint_v1
默认新结果目录：transformer_ppo_runs/diagnostics/ccp_icm_multiscale_joint_v1_unique_resume_v2

兼容性修复：部分 Anaconda Python 未编译 pidfd 接口；程序会通过 libc.syscall
调用相同的 Linux pidfd 系统调用，仍使用文件描述符固定进程身份，不退化为按PID发信号。
兼容后端仅支持已核对的64位 x86_64/aarch64 ABI。内核不支持或权限不足会在暂停前报错。
先做不影响实验进程的自检：
python3 resume_multiscale_screen_v1/resume.py --check-handoff
首次失败的 v1 目录原样保留。修复后使用 v2 输出目录，避免覆盖冻结的旧接续代码哈希。

过程：
1. 校验原 PLAN、全部冻结代码、模型权重、输入尺度、门控及至少两对已完成的完全相同结果。
2. 若旧调度器仍活动，仅对本用户、这个实验的准确主进程暂停派发。
   Linux pidfd 固定进程身份，避免 PID 重用。已有 worker 不收到停止信号，继续完成。
   此时旧调度器在 ps 中显示 T，旧日志心跳暂时停止；新日志每30秒报告等待情况。
3. 当前 worker 完成后，正常退出旧调度器，取得原目录锁，再启动四个不同策略的剩余任务。
4. 真实新运行继续写原 screen 目录（同一冻结 PLAN）；新报告独立存放。
   不创建未执行 PPO 的 RUN_COMPLETE，不复制共享监督的运行时间到 PPO 行。

四个不同策略：BASE、ICM_HEURISTIC、ICM_INDEPENDENT、ICM_SHARED_SUPERVISED。
共9实例×3种子×4策略=108次真实搜索；之前已完成的 PPO-u0 复现运行另行计数。
剩余 PPO-u0 与共享监督同策略，显式复用指标。不会再跑新的 PPO-u0 搜索。
已有 PPO 运行只作为复现/成本记录，不能当作额外独立样本。

结果：
results/00_联合训练对比.csv：四个实测策略＋一个明确标注的同策略复用行。
  PPO复用行的平均耗时留空；部分已实测 PPO 的耗时另存，不假装全组已测。
results/PER_RUN.csv、STATISTICS.csv：四个不同策略的真实结果。
results/POLICY_REUSE.csv：每个复用指标的来源、哈希及 PPO 是否实际执行。
results/PPO_MEASURED_CHECKS.csv：已真实执行的 PPO-u0 复现运行。
results/UPDATE0_REPRODUCTION.csv：完全相同的已测配对证据。
results/QUICK_RESULT.json：真实运行、真实评价次数、跳过数和别名关系。

恢复：同一命令可续跑，完成的真实结果复用。不要同时重启原 run.sh。
中断接续程序时：等待旧 worker 的阶段会恢复旧调度器；运行新任务的阶段保留完成结果。
仅在接续进程被 SIGKILL 等无法清理的方式杀掉、旧调度器仍显示 T 时使用：
python3 resume_multiscale_screen_v1/resume.py --recover
恢复会检查记录的 PID、启动身份和用户，不向无关进程发信号。

原 STATUS 不会虚报原135次全部执行。接续后的完成状态以新目录 STATUS.json 为准。
校验用合成数据和模拟进程测试；真正 Linux 调度接管只在服务器运行。
