#!/usr/bin/env python3
"""Append OOS scenarios 1001-5000 for the fixed top two validation labels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from multi_instance_plan import ROOT


def run(args):
    plan = json.loads(args.plan.read_text())
    choice = json.loads(args.choice.read_text())
    if choice.get("status") != "needs_5000_extension" or len(choice["top_two"]) != 2:
        raise ValueError("first-stage top-two record required")
    for item in plan["entries"]:
        if item["split"] != "validation":
            continue
        sid = item["instance_id"]
        for repeat in range(1, 6):
            paths = [args.root / "validation_rule" / sid / "rule" / f"run{repeat:02d}"]
            paths.extend(args.root / "validation_labels" / sid / f"{label}_eps0.1" /
                         f"run{repeat:02d}" for label in choice["top_two"])
            for run in paths:
                prefix = run / "oos_1000" / "oos_1000.npz"
                if not prefix.exists():
                    raise ValueError(f"missing 1000 OOS prefix: {run}")
                full = run / "oos_5000"
                if full.exists():
                    continue
                subprocess.run([sys.executable, str(ROOT / "multi_instance_oos.py"),
                    "--instance", item["instance_path"], "--run", str(run),
                    "--out", str(full), "--size", "5000",
                    "--oos-seed", str(item["oos_seed"]),
                    "--extend-from", str(prefix)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--choice", type=Path, required=True)
    run(parser.parse_args())
