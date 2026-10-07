#!/usr/bin/env python3
"""Evaluate one frozen method run using its test manifest's OOS-5000 seed."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from multi_instance_oos import run
from transformer_ppo.instance_catalog import read_instance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    instance, _ = read_instance(args.instance)
    if instance["split"] != "test" or instance.get("oos5000_seed") is None:
        parser.error("OOS evaluation is allowed only for frozen test manifests")
    result = run(SimpleNamespace(
        data=args.data, instance=args.instance, run=args.run, out=args.out,
        size=5000, oos_seed=int(instance["oos5000_seed"]), extend_from=None))
    print({key: value for key, value in result.items() if key != "candidates"})
