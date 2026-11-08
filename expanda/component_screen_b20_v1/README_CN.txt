B20：监督学习组件筛选（新实验，不复用旧权重）

目的
在 S0–S8、pop250、CCP100 三个经验 q90 目标、B20000 下比较
BASE、I_RULE/I_LEARN、C_RULE/C_LEARN、M_RULE/M_LEARN。
本阶段不训练 PPO，不做 ICM 组合，不打开独立最终测试或 OOS 重评。

文件入口
protocol.json   固定划分、种子、预算、稀疏采样数量、训练和门控搜索范围
run.py          prepare → collect → train → screen，断点续跑和进度日志
instances.py    用原训练 OD 池生成互不重叠的 54/18/27 个实例
controller.py   预算计数、候选构造、旁路采集、规则/冻结学习执行
learning.py     训练集归一化、负/正残差标签、分组小批次、三模型与门控
report.py       保留配对的配置等权统计、区间、结果表和收敛曲线
worker.py       单次搜索；复用未修改的运输模型和 NSGA-II
checks.py       小规模实现检查，不计入组件选择

旧代码和旧结果保持原样。新运行依赖同仓库 formal_icm_s0_v1/src 的基础
算子和评价函数以及 assets/data/data_expanded.xlsx，不要单独搬走此文件夹。
baseline_uncertainty.py 的 SHA256 与 909315d 完全一致，并在启动时核验。

固定设置
训练每配置6实例，验证每配置2实例，比较每配置3实例，9个配置共99实例。
每实例固定 CCP100 情景种子；筛选每实例3个算法种子、7方法，共567次。
新实例全部来自原训练 OD 池。训练/验证/筛选是开发用途划分，不占最终测试。
原 catalog.json 的散列必须匹配；不自动改用另一个实例目录。
54训练+18验证各采一条 Base B20000 轨迹（C、M共享），另采一条 Directed
初始化轨迹（I，生成250个初始化位置即停止）。

每实例每组件额外标注最多384次完整方案评价：I最多48状态、C最多24状态、
M最多48状态，预设前/中/后稀疏位置。C候选包含两个经 Base 变异的子代。
实际额外计费记录在 DATA.json，不按状态数乘8或16代替。
相同状态的重复候选与已有搜索评价可以复用；审计评价不会写入搜索个体
的有效缓存。旁路构造/评价恢复搜索随机流，不用候选结果推进 Base。
缺失整个采样阶段会停止并留下原因，不自动补采或改变预设位置。

输入、标签、训练
预算进度固定 evaluation/20000，I额外使用 slot/250。不存在 /400、/20
或 /200 的进度分母。按相同货物批次数分组训练，不填充 batch token。
每候选包含各批次 token、候选的评价前特征及当前种群/初始化前缀上下文。
采集日志的诊断HV暂沿用旧训练归一化文件，不用于新标签或筛选结果。
新目标和 HV 界限只由训练轨迹/训练候选可行点生成；输入标准化和标签尺度
也只取训练数据。INPUT_SCALER.json 与 HV_NORMALIZATION.json 分开保存。
成本/排放先除总货量，时间除实例时间跨度，再使用统一训练集 min/max。
不裁剪目标坐标；在参考点(1.1,1.1,1.1)之外的点不贡献 HV。
输入 z 值裁剪在±10，冻结搜索记录裁剪数量和最大绝对 z 值。

P 是已评价初始化前缀(I)或当前种群(C/M)的可行目标点；Y 是候选可行点，
C最多两个点。q=HV(P并Y)-HV(P)；标签=(q-q_ref)/训练标签标准差，可为负。
参考候选重新排列为0号且网络输出固定0；重复候选不能被选中。
I参考Directed，C参考Simple，M参考Balanced Replace。
统一小型Transformer、加权成对排序损失、3种子集成。
仅用18验证实例选择共同轮次和 mean-lambda*std 门控（候选范围见协议）；
先实例内平均、再配置等权。保留全部退回规则的候选，防止验证负收益时
强行切换。冻结后即使筛选表现不好，也不再调门控、网络或补采。
旧训练权重不兼容此输入定义，不加载。

预算和终止
所有七组使用同一评价计数器：真实新评价计数；未改变且保留相同情景+
染色体有效缓存的个体不重算。新构造个体仍评价，不能因为某个历史无关
个体曾有相同染色体就免计费。每个新交叉子代各计一次。
为保留有效继承缓存，worker仅在进程内关闭旧版的无条件缓存清除；准确
指纹检查仍执行。不会修改仓库中的旧评价代码。
初始化、可行性增强包含在预算；预算是上限。双子代原子步骤可能剩1次
不用（19999/20000），记录 stop_reason，不超支，不假称都恰好20000。
不使用200代停止条件。连续20代无新增评价视为实现/搜索停滞异常，保留
运行资料并停止，不把未完成运行当作完整预算成功结果。

收敛点400/1000/5000/10000是当前种群加已评价待选择子代的只读环境选择
结果，不改变实际种群；20000点取最终种群，并记录实际评价次数。
输入/模型文件校验、搜索过程计数、RSS、耗时、调用/回退/切换情况均保存。

运行（在服务器由用户操作）
python3 -u expanda/component_screen_b20_v1/run.py --stage all --workers 2
从仓库根目录执行；也可进入expanda后使用component_screen_b20_v1/run.py。
默认结果目录：expanda/transformer_ppo_runs/diagnostics/ccp_component_screen_b20_v1
原实例目录默认取协议中的 source_catalog_default；不需要复制训练权重。
依赖沿用已可运行旧实验的 Python、numpy、torch、pandas、openpyxl、matplotlib。
本机检查：python3 -W ignore::ResourceWarning expanda/component_screen_b20_v1/checks.py

也可 --stage prepare/collect/train/screen，但严格先采集、再冻结、最后筛选。
启动后每30秒输出心跳和子任务日志路径。有效完成任务自动复用。
一个结果目录只允许一个调度进程。SIGTERM/SIGINT停止子进程，保留记录。
中断目录移入 interrupted 后从该次任务重新运行，不伪造代内恢复。
累计成本必须包含 interrupted/PROGRESS.json 已知成本，不能只报成功任务。
每任务4小时、单次调度7天的防失控上限；超时检查STATUS后同命令续跑。

统计与解释
主表以Base为基准，额外报告各学习方法相对对应规则。
先实例内汇总3种子、再配置内汇总3实例、再对9配置等权。
区间对每个配置内的实例、实例内配对种子两层重采样；不把9配置随机丢弃。
Base HV=0时不加小常数；该百分比标空、报告绝对差，统一改用绝对差排序。
按预先指定主指标选下一阶段研究组件；不能据此声称普遍最优。
没有自动 PPO。结果文件：RESULT_TABLE.csv、PER_CONFIGURATION.csv、
CONVERGENCE.csv/PNG、QUICK_RESULT.json、PAIRED_RUNS.json、COLLECTION_SUMMARY.json。
