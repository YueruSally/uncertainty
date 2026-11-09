九组 B400 扩大验证的精简完整打包入口

在拥有全部原始结果的服务器仓库 expanda 目录执行：
python3 -u component_recheck_b400_release_v1/package.py --out /home/lunet/cokt2/ICM_B400_900_Compact.zip

程序不搜索、不训练、不修改原结果。读取 700 次原实验、200 次组合补充和九组 HV 重算结果。
校验原前沿、情景配对、冻结代码/模型哈希、报告输入哈希；不接受缺失文件的半成品包。
打包之前，将整理后的运行入口放入临时目录，从压缩原始数据重新计算 HV，核对三张表。
输出 ZIP 包不含测试、日志、Git、旧诊断、旧实验目录；只保留当前九组必需的导入依赖。

包内：README_CN.txt、environment.yml、run.py、src/、assets/、results/。
900 次运行的 RUN_COMPLETE.json、configuration.json、final_feasible_nondominated.json
按记录无损合并在 results/raw_runs.jsonl.gz，一条记录对应一次运行。
assets/provenance.json.gz 保留原计划和源文件哈希，原绝对路径仅作历史记录。
实际复现从包内资产加载，不依赖那些历史路径。

测试仅留在开发仓库：
python3 component_recheck_b400_release_v1/checks.py
包含九组真实 B400 小样本打包前后精确重放、HV 核验、损坏检测和900条合成记录打包/移动/解压核验。
合成记录只用于代码验证，绝不作为用户实验结果交付。
