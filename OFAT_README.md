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
