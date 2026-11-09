扩大验证补充 ICM 启发式与 ICM Transformer

目的
沿用 ccp_component_recheck_b400_v1 已完成的 700 次搜索，补充两组联合方法。
10 个原实例、10 个原算法种子，每种联合方法新增 100 次，共新增 200 次。
保留 pop20、B400、100 个优化情景、q90 和原冻结模型，不训练、不做 PPO。

两组方法
I_C_M_K8_HEURISTIC：Directed 初始化 + Simple 交叉 + Balanced 变异。
I_C_M_K8_TRANSFORMER：原冻结 I residual + C residual + M Transformer。
初始化保持旧实验定义，包括原来全部初始化位置的处理；不混入保留固定种子的新修正。
变异启用时点沿用旧实现。
交叉候选用 Base 变异预览评分，选中后返回变异前的子代对，实际 M 阶段只执行一次变异。
Simple 使用原排序与平局规则，通过已有的延迟变异选择接口执行，不加载神经网络。
只有实际选中的子代真实评价。初始化和搜索共享 B400，联合组不额外真实评价隐藏候选。

校验与续跑
启动时验证旧代码、模型、实例、700 次结果及 Python/NumPy/PyTorch 版本。
每次新运行核对预算与对应旧 Base 的情景摘要。原结果保留在原目录。
原七组代码不修改，新代码与输入写入独立冻结 PLAN.json。
中断后执行同一命令复用完成的运行；未完成尝试保留在 interrupted，部分耗时可能额外发生。

运行（仓库 expanda 目录）
python3 -u component_recheck_b400_icm_v1/extend.py --workers 1
默认读取 transformer_ppo_runs/diagnostics/ccp_component_recheck_b400_v1。
默认输出 transformer_ppo_runs/diagnostics/ccp_component_recheck_b400_icm_v1。
可用 --source、--out 指定同级独立目录；--workers 可选 1 或 2。

HV 重算自动进行
两组完成后，每个实例汇总九种方法、全部十个种子的最终可行非支配前沿。
合并集合不再做全局非支配过滤，按三个原始目标分别取 min/max。
用该实例共同界限重新归一化每次运行的前沿，参考点固定为 (1.1,1.1,1.1)。
重新计算全部 900 次 HV，再按实例、种子配对计算相对 Base 和对应规则的提升及区间。
ICM Transformer 同时直接对比 ICM 启发式。
新增方法可能改变归一化界限，因此原七组的报告数字也会重新计算。
这是事后报告指标，不反馈到模型或搜索；不能把 oracle 或其他实例的结果混入本轮。

输出
补充实验目录的 RESULT_TABLE.csv：全部九组的重算结果。
同级 ccp_component_recheck_b400_icm_v1_hv_union_v1：逐运行 HV、逐实例结果、界限和输入哈希。
QUICK_RESULT.json 分别记录新增搜索成本和 HV 统计；旧 700 次成本不重复计入新增成本。

本地验证
python3 component_recheck_b400_icm_v1/checks.py
python3 hv_instance_union_v1/checks.py
包含实际 B400 联合运行、旧学习组合精确重放、启发式无网络加载、结果复用及九组共同界限检查。
