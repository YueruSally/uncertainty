按实例合并全部最终前沿后重算 HV

支持两种来源，按 PLAN.json 自动识别，分别计算，禁止混合：
1. 旧 I/C/M/ICM 共同种子实验，3 个实例、10 个种子、8 种方法，共 240 次运行。
2. B400 扩大验证，10 个实例、10 个种子、7 种方法，共 700 次运行（没有 ICM 组合）。
只读取原始结果，不搜索、不评价新方案、不训练。

计算定义
对实例 i，合并本轮全部方法和全部种子的 final_feasible_nondominated.json。
旧实验为每实例 80 个前沿，扩大验证为每实例 70 个前沿，不能跨实验借用组合结果。
读取 optimisation_objectives 中 cost、emission、makespan 三个最小化目标。
每个文件是该次运行自己的最终可行非支配前沿。
用于确定界限的合并集合不再进行全局非支配过滤，以保留各运行前沿的目标极值。
分别计算三个目标 min_ij 和 max_ij。
该实例的每个方法、每个种子共用 z_j=(f_j-min_ij)/(max_ij-min_ij)。
若某一维完全相同，该维统一置零并在 INSTANCE_BOUNDS.json 标注，不加任意 epsilon。
每次运行只用自己的前沿计算 HV；不把多个种子的前沿合并成一次运行。
参考点沿用原程序的 (1.1,1.1,1.1)，精确三维最小化 HV 算法不变。
同一实例按货量等常数除目标再做该实例 min/max，与直接对原始目标做 min/max 等价。

提升率
每个实例、每个种子：100*(HV_method/HV_BASE-1)。
先在实例内平均种子，再对实例等权平均。不用平均 HV 的比值代替配对提升率均值。
bootstrap 随机种子沿用对应来源的报告代码设定。
Base HV 为零时，总体百分比标为未定义，改报绝对差。
实例及实例内配对种子的分层 bootstrap 区间沿用原计算方法；区间以本轮固定界限为条件。

校验
每次运行核对方法、实例、种子、400 次预算、计划哈希、情景摘要、可行性和前沿非支配性。
首先使用每次 configuration.json 的旧归一化重算 HV，必须与 RUN_COMPLETE.json 一致。
源文件 SHA256 存在 INPUT_MANIFEST.json。原文件不修改，输出目录必须独立且不存在。
这只是事后报告指标的重新计算，不能反向用于训练、调门控或声称新增独立测试证据。
使用新的归一化和相同的标准化参考点，也改变了原始单位下的参考点位置，提升率可能改变。
添加新方法或新种子后，如重新拟合这些界限，就必须把该实例全部方法统一重算。

输出
RESULT_TABLE.csv：原提升率、新提升率、区间、相对规则提升、胜负次数。
PER_INSTANCE.csv：每个实例、每种方法的平均 HV 和配对平均提升率。
PER_RUN.csv：每一次运行的原 HV、新 HV、相同种子 Base HV 和配对提升率。
INSTANCE_BOUNDS.json：各实例的三个目标 min/max、参考点和点数。
QUICK_RESULT.json：完整统计及校验结果。

运行
需要 Python 3.9+ 和 NumPy。
仓库 expanda 目录下：
python3 hv_instance_union_v1/recompute.py --source 原八组实验目录 --out 新重算目录

下载包中：
python3 code/expanda/hv_instance_union_v1/recompute.py --source 原八组实验目录 --out 新重算目录
源目录必须含 PLAN.json 和 runs/，其下每次运行必须有 RUN_COMPLETE.json、
configuration.json 和 final_feasible_nondominated.json。
包里只带重算结果和必要代码，不重复包含数百份原始搜索文件。

原表纠正
原始 RESULT_TABLE.csv 中 I_DIRECTED_BALANCED 为 +17.710488%，I_K8_RESIDUAL 为 +16.945898%。
截图将两者放反；本次严格按原始方法标识汇总。
