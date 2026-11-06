# 结果定位与复现（无 tests 版本）

## 1. 校验发布包

在发布目录根部运行：

```bash
sha256sum -c SHA256SUMS
```

压缩包另有同名`.tar.gz.sha256`文件，可在解压前校验归档。

## 2. 正式结果入口

- 简洁结论：`RESULT_SUMMARY.json`、`RESULT_TABLE.csv`。
- 原始正式结果：`results/ccp_m_prescreen_formal_v1_10inst_10seed_b400_oos5000/`。
- 修正统计：上述目录的`statistics_corrected_v1/QUICK_RESULT.json`和`RESULT_TABLE.csv`。
- 监督模型：`results/ccp_m_prescreen_scale_curve_v2/BUNDLE.json`及其引用的checkpoint。
- PPO模型：`results/ccp_m_prescreen_ppo_training_v1/`中的checkpoint，以及`results/ccp_m_prescreen_ppo_selection_v1_20valid/FORMAL_METHODS.json`冻结的方法清单。
- 正式实例：`results/ccp_m_prescreen_locked_final_instances_v1/`。

正式报告必须采用`statistics_corrected_v1`中的两层bootstrap区间。该修正没有新增目标评价，也没有改变模型、HV、均值或逐运行胜负。

## 3. 证据链

1. `ccp_m_prescreen_final_oracle_v1_3inst_10seed_b400`：确认K8候选池在最终HV上存在空间。
2. `ccp_m_prescreen_scale_instances_v1`：冻结200个训练实例和20个验证实例。
3. `ccp_m_prescreen_scale_collection_v2`：收集K8全程反馈。
4. `ccp_m_prescreen_scale_curve_v2`：训练25/50/100/200实例监督模型并冻结size-200模型。
5. `ccp_m_prescreen_scale_search_v1_3inst_10seed_b400`：在新开发实例上完成完整搜索筛查。
6. `ccp_m_prescreen_ppo_training_v1`：只使用200个训练实例进行PPO训练。
7. `ccp_m_prescreen_ppo_selection_v1_20valid`：只使用20个验证实例选定update 2；没有打开最终测试。
8. `ccp_m_prescreen_locked_final_instances_v1`：锁定10个最终实例及独立OOS5000种子。
9. `ccp_m_prescreen_formal_v1_10inst_10seed_b400_oos5000`：四方法、100个配对运行和独立OOS5000重评。

## 4. 无测试目录时直接运行

包内保留了原流水线脚本，但那些开发脚本开头会调用已排除的单元测试。无tests发布版复现时，请直接调用对应Python阶段，或从原始Git仓库恢复tests后再运行shell流水线。核心直接命令为：

```bash
cd repository/expanda

# 重新计算正式统计；不新增目标评价。
python3 -u experiments/recompute_m_prescreen_formal_statistics_v1.py \
  --root ../../results/ccp_m_prescreen_formal_v1_10inst_10seed_b400_oos5000 \
  --out ../../results/ccp_m_prescreen_formal_v1_10inst_10seed_b400_oos5000/statistics_recomputed
```

如果要从头重跑正式搜索，应复制现有结果到新的输出路径，并重新映射冻结清单中的绝对路径；不要覆盖随包保存的正式证据。正式搜索的冻结参数为CCP100、置信度0.90、pop20、B400、每实例10个算法种子和OOS5000。

## 5. 环境

正式运行记录使用Python 3.10、PyTorch 1.12.1和NumPy 1.23.5。完整逐文件来源与SHA256见`RELEASE_MANIFEST.json`。发布包不包含Python环境本身；依赖声明和所有项目源码位于`repository/`。
