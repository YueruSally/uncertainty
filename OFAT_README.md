# OFAT sensitivity runner

Run from the project directory on the remote server:

```bash
source .venv/bin/activate
python ofat_sensitivity.py --preset congestion
```

This runs the main congestion-related parameters:

- `P1_border_delay_rail`
- `P2_border_capacity`
- `P3_background_flow`

For the full P1-P10 sweep:

```bash
python ofat_sensitivity.py --preset full --seeds 2026 2027 2028
```

Useful quick-test scale:

```bash
python ofat_sensitivity.py --preset congestion --pop 80 --gens 50 --batches 10
```

Outputs are saved in `ofat_out/`:

- `ofat_results.csv`: one row per scenario and seed
- `ofat_group_summary.csv`: averaged objective and risk metrics
- `ofat_node_summary.csv`: averaged flow and utilisation by bottleneck node
- `manifest.json`: run configuration

The script keeps the model deterministic. It only perturbs input parameters in
temporary workbook copies, then calls `baseline3_v1.py`.

## Formal 40-batch experiment

Use `ofat_experiment.py` for the paper-ready OFAT run:

```bash
python3 ofat_experiment.py run
```

Default full setting:

- 40 batches (`N_BATCHES=None`)
- population size `200`
- generations `150`
- seeds `2026`, `7`, `99`
- representative solution: minimum-cost feasible Pareto solution
- parameters: P1-P10 with `-30%`, `-15%`, `+15%`, `+30%`

The full run is resumable. If the server disconnects, run the same command
again and completed `(parameter, level, seed)` combinations will be skipped.

Formal outputs:

- `ofat_out/ofat_results.csv`
- `ofat_out/summary_by_param_level.csv`
- `ofat_out/sensitivity_ranking.csv`
- `ofat_out/sensitivity_ranking.txt`

For a quick check only:

```bash
python3 ofat_experiment.py run --test
```
