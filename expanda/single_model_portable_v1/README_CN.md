# 导出可运行的单成员部署包

在原 41 实验环境执行：

```bash
python3 -u single_model_portable_v1/export.py --with-wheels
```

读取已完成的单成员面板及共享监督单成员目录，校验冻结模型与九个开发实例，附上历史算法源码、运输网络工作簿、原样保留的用户汇总表以及本机 Python 平台的依赖 wheels。会加载权重检查依赖，但不训练、不搜索。

默认输出 `transformer_ppo_runs/diagnostics/ICM_SINGLE_B5000_RUNNABLE.zip`。已有同名包时拒绝覆盖，可用 `--out` 指定新路径。

`--with-wheels` 需要网络，并可能下载较大的 PyTorch wheel；若省略，只附锁定版本的安装清单，安装时需要网络。

导出包适用于与原实验相同 Python 小版本的 Linux。安装命令 `bash install.sh`，只检查用 `bash run.sh --check`，正式搜索用 `bash run.sh --workers 2`。包里的运行不再依赖原服务器实验目录。

原样提供的 CSV 不作为重新运行的预期答案；新统计另存。只在终点导出胜平负；不附历史逐次运行表或测试文件。原始第三方 wheel 不作删改。
