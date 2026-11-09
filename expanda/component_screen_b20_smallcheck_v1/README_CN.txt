B20 小范围验证：固定初始化种子与 C 门控

目的
先核查两个具体问题，不重训、不调门控、不重跑 567 次比较。
原始 component_screen_b20_v1 源码、模型、归一化、实例和结果保持不变。
本目录是旁路诊断，不能作为新的正式组件筛选结果。

初始化验证
固定选 S1、S0、S8 各自编号最小的一个开发筛选实例，种子 951001、951002。
每对运行：
  BASE_INIT：仅生成并评价 Base 初始种群，250 次评价；
  I_ALL：原 I_RULE，全部初始化位置均经过 Directed 改写；
  I_KEEP_FIXED：保留 Base 前 162 个位置，只对后 88 个随机位置使用 Directed。
162 个位置来自原实现的 75 个容量种子及三类各 29 个贪心种子。
这里“固定”指保留这些位置的 Base 构造结果，容量种子本身仍使用原随机过程。
I_ALL 和 I_KEEP_FIXED 各运行到原 20,000 次预算；其余算子均为 Base。
最多新增 12 次完整搜索及 6 次初始化，完成运行最多 241,500 次评价。
中断/失败尝试可能产生额外成本，日志和部分结果另存，不算成零。

检查
I_ALL 必须与旧 I_RULE 的评价序列摘要及最终 HV 完全相同。
BASE_INIT 必须与旧 Base 的初始化评价摘要相同。
I_KEEP_FIXED 的前 162 个个体指纹必须逐个与 BASE_INIT 相同。
全部优化情景摘要必须一致；以上任意检查失败，不出研究结论。
记录初始化重复数量、可行率、HV，及完整搜索最终 HV 和成本。
主要看六对 KEEP_FIXED − ALL 的最终 HV 绝对差和胜负次数。
这只是方向性诊断；不以六对运行宣称统计确认，不自动接 PPO。
旧 I 学习模型不部署到新的初始化轨迹上，避免混入分布变化。

C 验证
读取原 models/C/VALIDATION.json 全部已有选项，按原规则复算最优门控。
与原 BUNDLE.json 的 gate 逐字段核对；报告最佳启用选项和无阈值选项。
不强制启用 C、不增加评价。此项只能说明保存的验证分数如何导致回退，
不能单凭它判断是数据量、表示还是损失函数导致泛化失败。

运行（在 expanda 目录）
python3 -u component_screen_b20_smallcheck_v1/check.py
默认读取 transformer_ppo_runs/diagnostics/ccp_component_screen_b20_v1，
另存至 transformer_ppo_runs/diagnostics/ccp_component_screen_b20_smallcheck_v1。
两个独立单线程工作进程；每 30 秒心跳。单进程 40 分钟、单次调用 2 小时上限。
若中断，用原命令续跑，完成结果复用，未完成目录保留到 interrupted。

输出
PLAN.json             预先固定样本、比较口径、源文件哈希
RESULT_TABLE.csv      初始化结构及完整搜索结果
C_GATE_AUDIT.json     原 C 门控选择复核
QUICK_RESULT.json     六对差异、检查结果、成本和结论
STATUS.json           运行状态

本地正确性检查
python3 component_screen_b20_smallcheck_v1/checks.py
测试夹具使用小预算，不属于服务器研究结果。
