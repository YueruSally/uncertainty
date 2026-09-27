#!/usr/bin/env python3
"""Freeze paired CCP100 jobs and execute one supervised study stage at a time."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from multi_instance_catalog import SCENARIOS, VALIDATION, read_instance
from multi_instance_logging import write_json

ROOT = Path(__file__).resolve().parent
METHODS = ("random", "rule", "learning")


def paired_seeds(instance_id, repetition, instance_seed=None):
    n = int(instance_id[1:])
    base = 5_000_000 + n * 1000 + repetition
    return dict(algorithm=base, training=base+100_000,
                path=base+200_000 if instance_seed is None else instance_seed,
                policy=base+300_000)


def make_plan(instances, out):
    if out.exists():
        raise ValueError("plan exists")
    entries = []
    for sid in list(SCENARIOS) + [f"T{i}" for i in range(1, 6)]:
        path = instances / f"{sid}.json"
        record, _ = read_instance(path)
        entries.append(dict(instance_id=sid, instance_path=str(path.resolve()),
                            instance_digest=record["instance_digest"],
                            instance_seed=record["instance_seed"],
                            path_seed=record["path_seed"],
                            split=record["split"], oos_seed=7_000_000+int(sid[1:])))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=ROOT, text=True).strip()
    plan = dict(schema_version=1, entries=entries, evaluation_budget=15000,
                source_commit=commit,
                mutation_probabilities=dict.fromkeys(("add", "del", "mod", "mode", "replace"), .2),
                reference_point=[1.1, 1.1, 1.1],
                audit=[("S0", "random"), ("S0", "rule"),
                       ("S9", "random"), ("S9", "rule")],
                expected_optimisation_runs=254)
    plan["plan_digest"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, plan)
    return plan


def jobs(plan, stage, lock=None):
    by_id = {x["instance_id"]: x for x in plan["entries"]}
    if stage == "audit":
        return [(by_id[sid], method, 0, None, .10) for sid, method in plan["audit"]]
    if stage == "train":
        return [(x, "rule", r, None, .10) for x in plan["entries"]
                if x["split"] == "train" for r in range(1, 6)]
    if stage == "validation_rule":
        return [(x, "rule", r, None, .10) for x in plan["entries"]
                if x["split"] == "validation" for r in range(1, 6)]
    if stage == "validation_labels":
        return [(x, "learning", r, label, .10) for x in plan["entries"]
                if x["split"] == "validation" for label in
                ("selection_survivor", "first_front_member", "parent_relation")
                for r in range(1, 6)]
    if stage == "epsilon_zero":
        if not lock or lock.get("label") not in (
                "selection_survivor", "first_front_member", "parent_relation"):
            raise ValueError("frozen validation label required")
        return [(x, "learning", r, lock["label"], .0) for x in plan["entries"]
                if x["split"] == "validation" for r in range(1, 6)]
    if stage == "test_controls":
        return [(x, method, r, None, .10) for x in plan["entries"]
                if x["split"] == "test" for r in range(1, 11)
                for method in ("random", "rule")]
    if stage == "test_learning":
        if not lock or lock.get("epsilon") not in (0, .10):
            raise ValueError("frozen label and epsilon required")
        return [(x, "learning", r, lock["label"], lock["epsilon"])
                for x in plan["entries"] if x["split"] == "test"
                for r in range(1, 11)]
    raise ValueError(stage)


def run_stage(args):
    plan = json.loads(args.plan.read_text())
    digest = plan.pop("plan_digest")
    if digest != hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest():
        raise ValueError("plan digest mismatch")
    plan["plan_digest"] = digest
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=ROOT, text=True).strip()
    if commit != plan["source_commit"]:
        raise ValueError("current code commit differs from frozen plan")
    lock = json.loads(args.lock.read_text()) if args.lock else None
    for instance, method, repetition, label, epsilon in jobs(plan, args.stage, lock):
        record, _ = read_instance(instance["instance_path"])
        if record["instance_digest"] != instance["instance_digest"]:
            raise ValueError("instance changed after plan freeze")
        suffix = f"{label}_eps{epsilon:g}" if label else method
        run_dir = args.root / args.stage / instance["instance_id"] / suffix / f"run{repetition:02d}"
        complete = run_dir / "COMPLETE.json"
        seeds = paired_seeds(instance["instance_id"], repetition, instance["path_seed"])
        if complete.exists():
            config = json.loads((run_dir / "configuration.json").read_text())
            finished = json.loads(complete.read_text())
            if (not finished.get("completed")
                    or config["instance_digest"] != instance["instance_digest"]
                    or config["git_commit"] != commit
                    or any(config["seeds"][k] != v for k, v in seeds.items())
                    or config["evaluation_budget"] != plan["evaluation_budget"]
                    or finished["stop_reason"] != "evaluation_budget"):
                raise ValueError(f"existing run failed audit: {run_dir}")
        else:
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(ROOT / "run_multi_instance_ccp100.py"),
                "--instance", instance["instance_path"], "--out", str(run_dir),
                "--pop", "100", "--gens", "1000",
                "--evaluation-budget", str(plan["evaluation_budget"]),
                "--policy", method,
                "--algorithm-seed", str(seeds["algorithm"]),
                "--training-seed", str(seeds["training"]),
                "--path-seed", str(seeds["path"]),
                "--policy-seed", str(seeds["policy"])]
            if label:
                model = args.models / f"{label}.joblib"
                command.extend(("--model", str(model), "--epsilon", str(epsilon)))
                if lock and args.stage == "test_learning":
                    sha = hashlib.sha256(model.read_bytes()).hexdigest()
                    if sha != lock["model_sha256"]:
                        raise ValueError("model changed after freeze")
            print("START", run_dir, flush=True)
            with (run_dir.parent / f"run{repetition:02d}.stdout.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            finished = json.loads(complete.read_text())
            if finished["stop_reason"] != "evaluation_budget":
                raise ValueError(f"run stopped before budget: {run_dir}")
        if args.stage in ("audit", "validation_rule", "validation_labels", "epsilon_zero",
                          "test_controls", "test_learning"):
            size = 5000 if args.stage in ("test_controls", "test_learning") else 1000
            oos_dir = run_dir / f"oos_{size}"
            if not oos_dir.exists():
                cmd = [sys.executable, str(ROOT / "multi_instance_oos.py"),
                    "--instance", instance["instance_path"], "--run", str(run_dir),
                    "--out", str(oos_dir), "--size", str(size),
                    "--oos-seed", str(instance["oos_seed"])]
                subprocess.run(cmd, check=True)
            if args.stage == "audit":
                full = run_dir / "oos_5000"
                if not full.exists():
                    subprocess.run([sys.executable, str(ROOT / "multi_instance_oos.py"),
                        "--instance", instance["instance_path"], "--run", str(run_dir),
                        "--out", str(full), "--size", "5000",
                        "--oos-seed", str(instance["oos_seed"]),
                        "--extend-from", str(oos_dir / "oos_1000.npz")], check=True)
        print("DONE", run_dir, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make")
    make.add_argument("--instances", type=Path, required=True)
    make.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--stage", required=True, choices=("audit", "train", "validation_rule",
        "validation_labels", "epsilon_zero", "test_controls", "test_learning"))
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--models", type=Path)
    run.add_argument("--lock", type=Path)
    args = parser.parse_args()
    print(make_plan(args.instances, args.out) if args.command == "make" else run_stage(args))
