# Expanda Network Scenario Experiments

这个文件夹用于做导师建议的实验：**同一个 NSGA-II 程序，在不同网络数据上运行，然后比较结果差异**。

## 文件说明

- `baseline3.py`：参数化后的主程序，可以指定输入数据、输出目录、种群规模、迭代代数和运行次数。
- `baseline3_desktop_original.py`：从桌面复制来的原始脚本备份。
- `data/data_original.xlsx`：原始网络数据。
- `data/data_expanded.xlsx`：扩展网络数据。
- `run_network_experiments.py`：一键运行 original 和 expanded 两个网络场景。
- `compare_network_results.py`：汇总两个网络的 `run_summary.xlsx`，生成对比表。
- `.vscode/tasks.json`：VSCode 任务配置。

## 推荐实验流程

如果是在 VS Code Remote SSH 连接学校电脑，建议打开整个 GitHub 仓库后，在 Terminal 里运行：

```bash
cd expanda
python3 -m venv ../.venv
../.venv/bin/python -m pip install -r requirements.txt
```

如果仓库根目录已经有 `.venv`，只需要执行安装依赖这一句即可。

先做 smoke test，确认程序和数据能跑通：

```bash
cd expanda
../.venv/bin/python run_network_experiments.py --mode smoke
../.venv/bin/python compare_network_results.py --mode smoke
```

然后做 quick experiment，用于看趋势：

```bash
cd expanda
../.venv/bin/python run_network_experiments.py --mode quick
../.venv/bin/python compare_network_results.py --mode quick
```

最后做论文正式结果：

```bash
cd expanda
../.venv/bin/python run_network_experiments.py --mode full
../.venv/bin/python compare_network_results.py --mode full
```

## 对比指标

主要看 `outputs/network_comparison_*.xlsx`：

- `pareto_size_mean`：Pareto 解数量，反映可选运输方案丰富度。
- `feas_soft_mean` / `feas_strict_mean`：可行解比例，反映网络是否更容易满足约束。
- `hv_mean`：Hypervolume，越高通常说明 Pareto front 覆盖更好。
- `igd_plus_mean`：IGD+，越低通常越好。
- `spacing_mean`：解分布均匀程度。
- `runtime_s_mean`：计算时间，用于说明网络规模变大后的计算成本。

每个场景的详细输出在：

- `outputs/original_<mode>/`
- `outputs/expanded_<mode>/`

其中 `result.txt` 和 `pareto_points.json` 可以继续用于分析路径选择、运输方式占比、瓶颈节点和成本-排放-时间 trade-off。

## 在 VS Code 里运行

方式一：打开仓库根目录 `uncertainty`，然后使用：

1. `Terminal` -> `Run Task...`
2. 选择 `expanda: smoke run`
3. 再选择 `expanda: smoke compare`
4. smoke 能跑通后，再运行 `expanda: quick run` 和 `expanda: quick compare`

方式二：只打开 `expanda` 文件夹，也可以使用同样的 `Terminal` -> `Run Task...` 运行任务。

正式实验运行 `expanda: full run`。这个会按 `pop=250, gens=200, runs=30` 跑 original 和 expanded 两个网络，耗时会明显比 smoke/quick 长，适合放在学校电脑上跑。
