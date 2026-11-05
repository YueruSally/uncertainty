# Terminal-HV PPO 独立确认

仅在开发目录 `QUICK_RESULT.json` 的 `development_pass=true` 后运行。冻结同一个选中模型，测试前不再训练或改阈值。支持 `terminal-hv-ppo-v1` 和 `single-intervention-ppo-v2`，从已冻结的开发配置自动识别，不能手动替换执行方式。

单次干预 v2 保留“一次概率采样、后续 Balanced”的原策略；不改为贪心选择，也不增加干预次数。程序核对首次决策输入与 Balanced 一致、推理中没有训练，以及没有切换时最终 HV 与 Balanced 相同。

- 全新实例 10 个，每个 10 个配对种子；Base、Balanced、Transformer-PPO 共 300 次搜索。
- 搜索均为 CCP100、pop20、B400。优化后用每实例独立的 5,000 个情景重评全部保存的解，不再搜索。
- 保留原训练归一化和 HV 参考点。主要指标是独立情景下 PPO 相对 Balanced 的平均配对提升；区间按实例、种子分层重采样。
- 完成标记不代表通过；读取 `independent_confirmation_pass`。失败保留全部结果，不自动换实例、种子或模型重试。

在服务器 `expanda` 目录执行：

```bash
python3 -u validation/run_m_terminal_hv_confirm_v1.py \
  --development transformer_ppo_runs/diagnostics/ccp_m_single_intervention_ppo_v2 \
  --source-instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --out transformer_ppo_runs/diagnostics/ccp_m_single_intervention_confirm_v2
```

`RESULT_TABLE.csv` 保存平均提升和时间，`QUICK_RESULT.json` 保存结论，`COMPLETE.json` 保存完整记录。输入不变时，同一命令复用已完成的搜索和独立重评。一次调用最多运行一小时，未完成部分可续跑；源数据、模型或代码改变会拒绝复用。
