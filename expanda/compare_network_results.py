#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine scenario outputs from run_summary.xlsx into one comparison table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"
SCENARIOS = ("original", "expanded")


def numeric_summary(path: Path) -> dict[str, float | int | str]:
    df = pd.read_excel(path, sheet_name="RunSummary")
    runs = df[pd.to_numeric(df["run_id"], errors="coerce").notna()].copy()
    if runs.empty:
        raise ValueError(f"No numeric run rows found in {path}")

    out: dict[str, float | int | str] = {
        "n_runs": int(len(runs)),
        "runtime_s_mean": float(runs["runtime_s"].mean()),
        "runtime_s_std": float(runs["runtime_s"].std(ddof=1)) if len(runs) > 1 else 0.0,
        "pareto_size_mean": float(runs["final_pareto_size"].mean()),
        "feas_soft_mean": float(runs["final_FeasRatio_soft"].mean()),
        "feas_strict_mean": float(runs["final_FeasRatio_strict"].mean()),
        "hv_mean": float(runs["final_HV_norm"].mean()),
        "hv_std": float(runs["final_HV_norm"].std(ddof=1)) if len(runs) > 1 else 0.0,
        "igd_plus_mean": float(runs["final_IGD_plus"].mean()),
        "spacing_mean": float(runs["final_Spacing"].mean()),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original vs expanded network results.")
    parser.add_argument("--mode", default="quick", help="Mode suffix used by run_network_experiments.py.")
    args = parser.parse_args()

    rows = []
    for scenario in SCENARIOS:
        summary_path = OUTPUT_ROOT / f"{scenario}_{args.mode}" / "run_summary.xlsx"
        if not summary_path.exists():
            print(f"[SKIP] Missing {summary_path}")
            continue
        row = {"scenario": scenario, "summary_file": str(summary_path)}
        row.update(numeric_summary(summary_path))
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No run_summary.xlsx files found for mode={args.mode}")

    df = pd.DataFrame(rows)
    if set(df["scenario"]) >= {"original", "expanded"}:
        base = df.loc[df["scenario"] == "original"].iloc[0]
        for idx, row in df.iterrows():
            if row["scenario"] == "original":
                continue
            df.loc[idx, "delta_hv_vs_original"] = row["hv_mean"] - base["hv_mean"]
            df.loc[idx, "delta_feas_soft_vs_original"] = row["feas_soft_mean"] - base["feas_soft_mean"]
            df.loc[idx, "delta_pareto_size_vs_original"] = (
                row["pareto_size_mean"] - base["pareto_size_mean"]
            )
            df.loc[idx, "runtime_ratio_vs_original"] = (
                row["runtime_s_mean"] / base["runtime_s_mean"]
                if base["runtime_s_mean"] else float("nan")
            )

    out_csv = OUTPUT_ROOT / f"network_comparison_{args.mode}.csv"
    out_xlsx = OUTPUT_ROOT / f"network_comparison_{args.mode}.xlsx"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    df.to_excel(out_xlsx, index=False)

    print(df.to_string(index=False))
    print(f"\n[EXPORT] {out_csv}")
    print(f"[EXPORT] {out_xlsx}")


if __name__ == "__main__":
    main()
