S11 原独立模型与共享监督模型：同机并行

本版新增 S11，不覆盖或恢复旧 S0 目录，不修改旧权重及旧求解器。
S11 = 50 批货，交付时间窗倍率 2.0；S0 = 20 批货、倍率 1.0。
因此这不是只改变批次数的对照。它检验 S0 训练模型在 S11 上的迁移表现。
如果要用 S11 训练，必须另定版本、采集数据并同时重新训练两类模型。

固定设计
10 个新 S11 开发实例 × 30 个共同种子。
原方法：Base、I/C/M 各自规则与独立 Transformer、独立模型 ICM，共 8 组，2400 次。
共享监督：ICM_SHARED_SUPERVISED，一组，300 次。合计 2700 次。
pop250，最多200代，B50000，CCP100，q90，K8。
要求每条完成记录实际评价50000次、无隐藏候选评价；达不到则停止并保留日志。
这延续原完整参数实验，不改写代数上限来掩盖未完成预算。
全部新实例使用已冻结的训练 OD 池，保留原独立测试 OD 池；不用于训练/门控选择。
本入口不启动 PPO，不自动进行5000情景重评。S11独立情景种子预留在实例中。

同机执行
run.py --stage both --workers 2 的 2 是整个入口的总进程上限，不是每组2个。
原8组搜索占一个槽；共享训练或共享搜索占一个槽。每个进程数值库限制1线程。
共享模型默认复用 diagnostics/ccp_icm_formal_joint_v1/shared_model。
如果模型未完成，读取原200训练/20验证S0实例的标签完成共享监督训练，不采集S11标签。
该目录正在被旧训练占用时等待；不重复训练。同一目录有阶段锁，拒绝重复执行。
本入口不停止任何旧进程。若旧总流程仍在执行，不要同时启动它的重复工作。

在服务器 expanda 目录先做9次完整预算检查（检查结果计入最终2700次）：
python3 -u joint_icm_s11_v1/run.py --stage both --workers 2 --smoke

检查结束后，同一命令去掉 --smoke 才会补齐完整2700次：
python3 -u joint_icm_s11_v1/run.py --stage both --workers 2

默认输出：transformer_ppo_runs/diagnostics/ccp_icm_s11_joint_v1
ORIGINAL_STATUS.json / SHARED_STATUS.json：两条线分别的进度。
original.nohup.log / shared.nohup.log：阶段日志；logs/：逐次搜索日志。
ORIGINAL_SMOKE_COMPLETE.json / SHARED_SMOKE_COMPLETE.json：各方法实测耗时、峰值内存及粗略推算。
共享模型搜索无需等待原2400次结束。两条线都完成后自动汇总。
中断后用完全相同命令恢复；完整运行校验后复用，不完整目录转存 interrupted/。

也可分两个终端使用同一输出目录：
--stage original --workers 1
--stage shared --workers 1
不要再额外运行 --stage both。分开运行结束后执行 --stage report。
跨机器时要求同一共享目录、相同 Torch/NumPy/代码/输入哈希。
未共享文件系统的跨机器迁移不在这个入口的支持范围内。
不同机器或并发负载下的耗时不能直接解释为算法速度差异；结果记录实际主机。

结果
汇总时保存训练归一化下的旧HV，并对每个S11实例汇总全部9方法、全部种子的最终
可行非支配解，统一求min/max，以(1.1,1.1,1.1)为参考点重算HV。
输出 hv_union/RESULT_TABLE.csv、PER_RUN.csv、INSTANCE_BOUNDS.json，含配对区间、
胜平负、标准差和运行时间。Base HV为0时相对提升不定义，不添加小常数。
后验界限只用于报告，不进入模型输入、训练或选模。

重要限制
旧模型及本次共享监督使用历史S0、B400标签；新部署为S11、pop250/B50000。
网络允许50批货输入不等于已经验证其泛化能力。旧特征和归一化保持冻结；
共享模型记录每组件输入截断计数、规则回退和选择差异，供解释结果。
这轮不是已完成的实验，不承诺优于Base或启发式。
