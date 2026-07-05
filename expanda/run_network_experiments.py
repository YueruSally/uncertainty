#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the same NSGA-II program on multiple network scenarios.

Examples:
  python run_network_experiments.py --mode smoke
  python run_network_experiments.py --mode quick
  python run_network_experiments.py --mode full
  python run_network_experiments.py --scenario expanded --mode quick
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "baseline3.py"
OUTPUT_ROOT = ROOT / "outputs"

SCENARIOS = {
    "original": ROOT / "data" / "data_original.xlsx",
    "expanded": ROOT / "data" / "data_expanded.xlsx",
}

SHARED_BATCHES_20 = ROOT / "data" / "batches_20_original.csv"

MODES = {
    "smoke": {"pop": 20, "gens": 5, "runs": 1},
    "quick": {"pop": 80, "gens": 40, "runs": 3},
    "full": {"pop": 250, "gens": 200, "runs": 30},
}


def run_scenario(name: str, data_path: Path, mode: str, seed: int, shared_batches: Path | None) -> None:
    cfg = MODES[mode]
    tag = "b20" if shared_batches else ""
    out_dir = OUTPUT_ROOT / f"{name}_{tag}_{mode}" if tag else OUTPUT_ROOT / f"{name}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mpl_config = OUTPUT_ROOT / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(BASELINE),
        "--data",
        str(data_path),
        "--out",
        str(out_dir),
        "--pop",
        str(cfg["pop"]),
        "--gens",
        str(cfg["gens"]),
        "--runs",
        str(cfg["runs"]),
        "--seed",
        str(seed),
    ]
    if shared_batches:
        cmd.extend([
            "--batches-csv",
            str(shared_batches),
            "--expected-batches",
            "20",
        ])

    print("\n" + "=" * 78, flush=True)
    print(f"[SCENARIO] {name} | mode={mode} | data={data_path.name}", flush=True)
    if shared_batches:
        print(f"[BATCHES]  shared 20-batch demand = {shared_batches.name}", flush=True)
    print(f"[OUTPUT]   {out_dir}", flush=True)
    print("=" * 78, flush=True)

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(mpl_config)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run identical NSGA-II settings on original and expanded networks."
    )
    parser.add_argument(
        "--scenario",
        choices=["all", *SCENARIOS.keys()],
        default="all",
        help="Network scenario to run.",
    )
    parser.add_argument(
        "--mode",
        choices=MODES.keys(),
        default="quick",
        help="Experiment scale. Use smoke for checking, full for final thesis runs.",
    )
    parser.add_argument("--seed", type=int, default=1000, help="Base random seed.")
    parser.add_argument(
        "--workbook-batches",
        action="store_true",
        help="Use each workbook's own Batches sheet. Default uses the shared original 20-batch demand.",
    )
    args = parser.parse_args()

    shared_batches = None if args.workbook_batches else SHARED_BATCHES_20
    if shared_batches and not shared_batches.exists():
        raise FileNotFoundError(shared_batches)

    selected = SCENARIOS if args.scenario == "all" else {args.scenario: SCENARIOS[args.scenario]}
    for name, data_path in selected.items():
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        run_scenario(name, data_path, args.mode, args.seed, shared_batches)

    print("\n[DONE] Finished network scenario experiment runs.")
    print("Run `python compare_network_results.py --mode", args.mode, "` to build the comparison table.")


if __name__ == "__main__":
    main()
