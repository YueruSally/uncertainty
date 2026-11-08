S0 正式组件筛选：冻结 Transformer，先测完整预算，再运行全表

这是什么
  在全新生成的 10 个 S0 实例上，每实例 30 个共同算法种子。
  比较 Base、I规则、I Transformer、C规则、C Transformer、M规则、M Transformer、I+C+M Transformer，共 2400 次搜索。
  单组件组只改对应组件，联合组同时改三个组件。比较基准始终为 Base；同时给出学习版本相对对应规则的配对差值。
  这是 S0 域内正式组件筛选，仍不是最后的独立测试，也不代表所有 S0–S11 配置。

固定的预算与方法
  pop250；最多200代；真实目标评价上限50000，初始化和可行性增强都计入。
  两种终止条件先到者停止；程序要求每个有效结果实际达到50000次。若未达标会停止，不混入不等预算结果。
  一次评价指一个完整方案在固定100个情景下的评价，不是一个情景。CCP置信水平0.9；候选数K=8。
  未选候选不得调用目标评价。模型与超参数冻结，无训练、无PPO。I/C使用已冻结残差监督模型，M使用已冻结200实例模型。
  保留原特征编码、原裁剪规则和M从第10代启用的规则，不悄悄按大预算重缩放。本轮检验这些既有模型向目标规模的迁移。
  相同种子不表示决策改变后种群与后续随机轨迹仍相同；保证的是共同实例、路径库种子、优化情景、算法种子、求解器和预算规则。

通过GitHub更新后直接使用
  1. 在服务器现有仓库更新 codex/m-transformer-ppo-threeway 分支，然后进入：
       cd /home/lunet/cokt2/uncertainty-learning-components/expanda/formal_icm_s0_v1
     程序、冻结权重、数据与环境文件均在此目录，无需下载或解压ZIP。
  2. 使用此前能运行实验的Python环境。若建立独立Linux CPU环境：
       conda env create -f environment.yml
       conda activate icm-components
     environment.yml固定服务器使用的主要版本，不包含操作系统或Conda本身。
  3. 首先运行完整预算检查（8次搜索，已成功结果以后复用）：
       python3 -u run.py --smoke
     会先检查九个权重哈希，再从原始OD划分生成10个全新S0实例、冻结计划，最后检查8种方法。
     源catalog默认位置：
       /home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances/catalog.json
     它只用于原始训练OD划分；无需上传或读取最终测试结果。若目录搬迁，用 --source-catalog /实际位置/catalog.json。
     源catalog不在本机附件中，因此本包会在服务器读取并保存副本；不会伪造缺少的OD划分。
  4. SMOKE_REPORT.json完成后，按实测时间安排资源，再执行：
       python3 -u run.py --run
     执行剩余搜索后自动独立情景重评。后台运行可用nohup并重定向日志。
     默认2个进程，每进程1个计算线程。所有正式设置都在 experiment.json。
     需要更改资源数时必须在第一次smoke之前更改；冻结后不覆盖计划。
  5. 中断后原命令 --run 续跑。单次调用最多24小时，单次搜索/重评任务最多4小时；到上限保存已完成部分并停止，可再次续跑。
     中断未完成搜索会被单独保存再重做，额外成本不混入已完成预算；报告明确标记未提交成本可能缺失。
     不要重复启动，不要删除PLAN.json，不要在已冻结目录里改代码、模型或配置。
     每次完整搜索计50000次，2400次总计1.2亿次方案评价；5000情景重评另计，不能按B400耗时直接承诺完成时间。

独立情景检查
  每实例固定另一组5000个情景，8组方法与30种子共用；保持相同不确定性模型、0.9分位数及原训练归一化/参考点。
  对保存的最终解集原样重评，不修复、不重新搜索；不可行解不计HV。真实调用次数和重评成本单独报告。
  Base的HV为零时，相对百分比未定义，不能写成0或删除失败实例；同时汇报覆盖全部配对的绝对归一化HV差值。

看哪些文件
  formal_runs/SMOKE_REPORT.json：8种方法完整预算耗时；全表时间粗估不含OOS与并行资源争用。
  formal_runs/RESULT_TABLE.csv：8组方法的搜索结果，以Base为基准。
  formal_runs/QUICK_RESULT.json：单组件与规则、联合与单组件的完整配对比较及区间。
  formal_runs/OOS_RESULT_TABLE.csv、OOS_RESULT.json：独立情景结果和可行率。
  formal_runs/FINAL_REPORT.json、FORMAL_COMPLETE：搜索和重评均完成后生成。
  formal_runs/PLAN.json、assets/formal_instances/catalog.json：冻结输入哈希、预算、实例及种子。
  总体区间按实例和配对种子分层bootstrap。均值最大的组件不自动等于已证明最优；要结合区间、OOS可行率和耗时判断。
  零Base HV出现时，组件排序改用所有配对的绝对HV增益，避免对未定义百分比进行筛选。

代码重点
  run.py：唯一用户入口、输入检查和阶段调度。
  experiment.json：正式实验配置。
  src/experiment.py：8组公平比较、预算验证、续跑和统计。
  src/initialization.py、crossover.py、mutation.py：I、C、M策略。
  src/combined.py：三组件联合控制。
  src/baseline_uncertainty.py：共用NSGA-II及不确定评价。
  src/support/：上面这些程序实际依赖的运行库，不是另一套实验。
  assets/：冻结权重、网络数据、归一化及生成所需元数据；旧3个实例仅作元数据来源，不用于正式结果。

交付验证
  九个权重哈希及前向计算通过；8种方法在旧20/400配置下与原程序最终HV完全相同。
  pop250的I+C+M通过3250评价短程运行，覆盖初始化、交叉和第10代后的变异。
  新实例生成/续跑/输入变更拒绝、预算参数化、统计重算、零HV处理、固定解独立重评和不可行解排除已检查。
  本机检查使用临时测试实例，不随包交付；没有启动这2400次正式搜索，没有新的性能结论。
