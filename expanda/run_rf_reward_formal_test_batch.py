#!/usr/bin/env python3
"""Audit four frozen methods, then run the three paired RF formal-test blocks."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from transformer_ppo.config import TEST_CONFIGS
from transformer_ppo.instance_catalog import canonical_digest


ROOT = Path(__file__).resolve().parent
RF_RUNNER = ROOT / "run_rf_reward_policy_ccp100.py"
OOS_RUNNER = ROOT / "run_transformer_ppo_oos.py"
METHODS = ("random", "rule", "transformer", "mlp")
METHOD_NAMES = {
    "random": "Random-CCP100",
    "rule": "Rule-CCP100",
    "transformer": "transformer-ppo-masked",
    "mlp": "mlp-ppo-masked",
}
RF_METHOD = "rf-reward-regressor"
BLOCKS = (1, 2, 3)
EVALUATION_BUDGET = 15_000
POPULATION = 100
GENERATIONS = 1_000
OOS_SIZE = 5_000
PREDICT_JOBS = 1
EXPECTED_CATALOG_DIGEST = (
    "9d4f557c7f61919c88a21df42dc6a2b56b36bcfdb812f8f45a81f9c8bf752087"
)
EXPECTED_NORMALIZATION_SHA256 = (
    "de73404449c2b612609ad5248f815793f99b676e4260e2068ae403f043addbcd"
)
FIXED_RF_PARAMETERS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "max_features": 0.5,
}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_block_path(value: str) -> tuple[int, Path]:
    block, separator, path = value.partition("=")
    try:
        block_number = int(block)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected BLOCK=PATH with BLOCK 1, 2 or 3") from error
    if not separator or block_number not in BLOCKS or not path:
        raise argparse.ArgumentTypeError("expected BLOCK=PATH with BLOCK 1, 2 or 3")
    return block_number, Path(path)


def _exact_block_mapping(values: list[tuple[int, Path]]) -> dict[int, Path]:
    result = dict(values)
    if len(values) != len(BLOCKS) or set(result) != set(BLOCKS):
        raise ValueError("provide each existing block 1, 2 and 3 exactly once")
    return {block: result[block].resolve() for block in BLOCKS}


def _test_records(catalog: dict) -> list[dict]:
    records = [row for row in catalog["instances"] if row["split"] == "test"]
    counts = Counter(row["configuration"] for row in records)
    expected = Counter({configuration: 10 for configuration in TEST_CONFIGS})
    if len(records) != 120 or counts != expected:
        raise ValueError(f"expected 120 test manifests (10 per S0-S11), found {counts}")
    return sorted(records, key=lambda row: (
        int(row["configuration"][1:]), row["instance_id"]))


def _validate_static_inputs(data: Path, instances: Path, normalization: Path):
    catalog = _read(instances / "catalog.json")
    if catalog.get("catalog_digest") != EXPECTED_CATALOG_DIGEST:
        raise ValueError("formal catalog digest differs from the predeclared digest")
    source_sha = _sha256(data)
    if catalog.get("source_data_sha256") != source_sha:
        raise ValueError("catalog and network workbook digests differ")
    if _sha256(normalization) != EXPECTED_NORMALIZATION_SHA256:
        raise ValueError("reward normalization differs from the predeclared file")
    normalization_values = _read(normalization)
    if normalization_values.get("source") != "training-only":
        raise ValueError("normalization must declare source=training-only")
    normalization_digest = normalization_values.get("normalization_digest")
    if not normalization_digest:
        raise ValueError("normalization has no frozen normalization_digest")
    records = _test_records(catalog)
    for record in records:
        manifest = _read(instances / f"{record['instance_id']}.json")
        digest = manifest.pop("instance_digest", None)
        if (digest != record["instance_digest"]
                or canonical_digest(manifest) != digest
                or manifest.get("split") != "test"
                or manifest.get("configuration") != record["configuration"]):
            raise ValueError(f"test manifest disagrees with catalog: {record['instance_id']}")
    return catalog, records, source_sha, normalization_digest


def _audit_existing_run(root: Path, record: dict, method: str, block: int,
                        index: int, summary: dict) -> dict:
    method_root = root / "instances" / record["instance_id"] / method
    run_dir = method_root / "run"
    oos_dir = method_root / "oos"
    complete_path = run_dir / "COMPLETE.json"
    config_path = run_dir / "configuration.json"
    front_path = run_dir / "final_feasible_nondominated.json"
    oos_json_path = oos_dir / "oos_5000.json"
    oos_npz_path = oos_dir / "oos_5000.npz"
    required = (complete_path, config_path, front_path, oos_json_path, oos_npz_path)
    if not all(path.exists() for path in required):
        raise ValueError(f"existing formal output is incomplete: {method_root}")
    complete = _read(complete_path)
    config = _read(config_path)
    front = _read(front_path)
    oos = _read(oos_json_path)
    algorithm_seed = (
        int(summary["algorithm_seed_base"]) + (block - 1) * 1_000_000 + index)
    policy_seed = (
        int(summary["policy_seed_base"]) + (block - 1) * 1_000_000 + index)
    seeds = config.get("seeds", {})
    if (complete.get("completed") is not True
            or complete.get("method") != METHOD_NAMES[method]
            or complete.get("stop_reason") != "evaluation_budget"
            or complete.get("evaluation_budget") != EVALUATION_BUDGET
            or not EVALUATION_BUDGET - 3 <= int(
                complete.get("ccp_evaluation_count", -1)) <= EVALUATION_BUDGET
            or config.get("method") != METHOD_NAMES[method]
            or config.get("instance_id") != record["instance_id"]
            or config.get("instance_digest") != record["instance_digest"]
            or config.get("population") != POPULATION
            or config.get("generations") != GENERATIONS
            or config.get("evaluation_budget") != EVALUATION_BUDGET
            or seeds.get("algorithm") != algorithm_seed
            or seeds.get("training") != record["ccp100_seed"]
            or seeds.get("path") != record["path_seed"]
            or not isinstance(front, list)):
        raise ValueError(f"existing formal run metadata mismatch: {run_dir}")
    if method in ("random", "rule"):
        if seeds.get("policy") != policy_seed:
            raise ValueError(f"existing control policy seed mismatch: {run_dir}")
    else:
        checkpoint = summary["checkpoints"][method]
        provenance = config.get("policy_checkpoint") or {}
        if (config.get("policy_training") is not False
                or provenance.get("checkpoint_sha256") != checkpoint["sha256"]
                or provenance.get("policy_seed") != block
                or provenance.get("catalog_digest") != EXPECTED_CATALOG_DIGEST
                or provenance.get("normalization_digest")
                != summary["normalization_digest"]):
            raise ValueError(f"existing PPO provenance mismatch: {run_dir}")
    if (oos.get("instance_id") != record["instance_id"]
            or oos.get("instance_digest") != record["instance_digest"]
            or oos.get("size") != OOS_SIZE
            or oos.get("oos_seed") != record["oos5000_seed"]
            or oos.get("source_sha256") != _sha256(front_path)
            or oos.get("evaluated_decisions") != len(front)
            or len(oos.get("candidates", [])) != len(front)
            or [row.get("decision_fingerprint") for row in oos.get("candidates", [])]
            != [row.get("decision_fingerprint") for row in front]):
        raise ValueError(f"existing OOS metadata mismatch: {oos_dir}")
    return {
        "method": method,
        "run_configuration_sha256": _sha256(config_path),
        "front_sha256": _sha256(front_path),
        "oos_json_sha256": _sha256(oos_json_path),
        "oos_npz_sha256": _sha256(oos_npz_path),
        "scenario_digest": config["scenario_digest"],
        "oos_master_digest": oos["oos_master_digest"],
        "algorithm_seed": algorithm_seed,
        "policy_seed": policy_seed if method in ("random", "rule") else None,
    }


def _audit_existing_block(block: int, root: Path, records: list[dict],
                          source_sha: str, normalization_digest: str) -> dict:
    config_path = root / "batch_configuration.json"
    complete_path = root / "BATCH_COMPLETE.json"
    if not config_path.exists() or not complete_path.exists():
        raise ValueError(f"existing block {block} lacks completion metadata: {root}")
    config, summary = _read(config_path), _read(complete_path)
    expected = {
        "formal": True,
        "replication": block,
        "catalog_digest": EXPECTED_CATALOG_DIGEST,
        "normalization_digest": normalization_digest,
        "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
        "source_data_sha256": source_sha,
        "test_instance_count": 120,
        "methods": list(METHODS),
        "evaluation_budget": EVALUATION_BUDGET,
        "population": POPULATION,
        "generations": GENERATIONS,
        "oos_size": OOS_SIZE,
    }
    for key, value in expected.items():
        if config.get(key) != value or summary.get(key) != value:
            raise ValueError(f"existing block {block} has invalid {key}")
    if (summary.get("completed") is not True
            or summary.get("completed_instance_count") != 120
            or summary.get("method_instance_count") != 480):
        raise ValueError(f"existing block {block} is not a complete formal block")
    for method in ("transformer", "mlp"):
        checkpoint = summary.get("checkpoints", {}).get(method, {})
        path = Path(checkpoint.get("path", ""))
        if not path.exists() or _sha256(path) != checkpoint.get("sha256"):
            raise ValueError(f"existing {method} checkpoint changed for block {block}")
    rows_path = root / "oos_hv_rows.json"
    statistics_path = root / "formal_statistics.json"
    if (_sha256(rows_path) != summary.get("oos_hv_rows_sha256")
            or _sha256(statistics_path) != summary.get("formal_statistics_sha256")):
        raise ValueError(f"existing block {block} completion artifacts changed")
    instances = []
    for index, record in enumerate(records):
        methods = [
            _audit_existing_run(root, record, method, block, index, summary)
            for method in METHODS
        ]
        if len({row["scenario_digest"] for row in methods}) != 1:
            raise ValueError(f"block {block} methods do not share CCP100 scenarios")
        if len({row["oos_master_digest"] for row in methods}) != 1:
            raise ValueError(f"block {block} methods do not share OOS scenarios")
        instances.append({
            "instance_id": record["instance_id"],
            "instance_digest": record["instance_digest"],
            "configuration": record["configuration"],
            "algorithm_seed": methods[0]["algorithm_seed"],
            "scenario_digest": methods[0]["scenario_digest"],
            "oos_master_digest": methods[0]["oos_master_digest"],
            "methods": methods,
        })
    return {
        "block": block,
        "root": str(root),
        "batch_configuration_sha256": _sha256(config_path),
        "batch_complete_sha256": _sha256(complete_path),
        "checkpoints": {
            method: {
                "sha256": summary["checkpoints"][method]["sha256"],
                "policy_seed": summary["checkpoints"][method]["policy_seed"],
            }
            for method in ("transformer", "mlp")
        },
        "ignored_legacy_hv": True,
        "instances": instances,
    }


def _audit_rf_training(root: Path, source_sha: str) -> dict[int, dict]:
    config_path = root / "formal_training_configuration.json"
    complete_path = root / "FORMAL_TRAINING_COMPLETE.json"
    config, complete = _read(config_path), _read(complete_path)
    expected = {
        "catalog_digest": EXPECTED_CATALOG_DIGEST,
        "normalization_digest": EXPECTED_NORMALIZATION_SHA256,
        "source_data_sha256": source_sha,
        "instance_count_per_trajectory": 450,
        "trajectories": 3,
        "collection_count": 1350,
        "evaluation_budget_per_instance": 5000,
        "maximum_ccp100_evaluations": 6_750_000,
        "fixed_rf_parameters": FIXED_RF_PARAMETERS,
        "formal_test_or_oos_used": False,
    }
    for key, value in expected.items():
        if complete.get(key) != value:
            raise ValueError(f"RF formal training has invalid {key}")
    if complete.get("completed") is not True or config.get("fixed_rf_parameters") != FIXED_RF_PARAMETERS:
        raise ValueError("RF formal training is incomplete or used different parameters")
    if config.get("catalog_digest") != EXPECTED_CATALOG_DIGEST:
        raise ValueError("RF formal training used another catalog")
    results = {}
    for row in complete.get("trajectory_results", []):
        block = int(row["trajectory"])
        model_path = root / f"trajectory_{block}" / "model" / "rf_reward.joblib"
        summary_path = root / f"trajectory_{block}" / "model" / "training_summary.json"
        summary = _read(summary_path)
        if (block not in BLOCKS
                or _sha256(model_path) != row.get("model_sha256")
                or _sha256(model_path) != summary.get("model_sha256")
                or summary.get("training_data_digest") != row.get("training_data_digest")
                or summary.get("instance_count") != 450
                or summary.get("fixed_parameters") != FIXED_RF_PARAMETERS
                or summary.get("sample_weighting") != "none"
                or summary.get("maximum_archive_delta_error") != 0.0):
            raise ValueError(f"RF model provenance mismatch for block {block}")
        results[block] = {
            "block": block,
            "path": str(model_path.resolve()),
            "sha256": row["model_sha256"],
            "training_data_digest": row["training_data_digest"],
            "rf_seed": row["rf_seed"],
            "sample_count": row["sample_count"],
            "training_summary_sha256": _sha256(summary_path),
        }
    if set(results) != set(BLOCKS):
        raise ValueError("RF formal training must provide exactly three models")
    return results


def audit_inputs(data: Path, instances: Path, normalization: Path,
                 rf_training_root: Path, existing_blocks: dict[int, Path]):
    _, records, source_sha, normalization_digest = _validate_static_inputs(
        data, instances, normalization)
    models = _audit_rf_training(rf_training_root, source_sha)
    block_audits = {
        str(block): _audit_existing_block(
            block, existing_blocks[block], records, source_sha, normalization_digest)
        for block in BLOCKS
    }
    audit = {
        "schema_version": 1,
        "formal_hv_values_read": False,
        "legacy_hv_ignored": True,
        "catalog_digest": EXPECTED_CATALOG_DIGEST,
        "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
        "normalization_digest": normalization_digest,
        "source_data_sha256": source_sha,
        "existing_run_count": len(records) * len(METHODS) * len(BLOCKS),
        "existing_oos_count": len(records) * len(METHODS) * len(BLOCKS),
        "rf_models": {str(block): models[block] for block in BLOCKS},
        "blocks": block_audits,
    }
    return records, models, audit


def _rf_paths(out: Path, block: int, instance_id: str):
    root = out / f"block_{block}" / "instances" / instance_id / "rf"
    return root / "run", root / "oos"


def _completed_rf(run_dir: Path, oos_dir: Path, record: dict, model: dict,
                  pairing: dict) -> bool:
    complete_path = run_dir / "COMPLETE.json"
    config_path = run_dir / "configuration.json"
    front_path = run_dir / "final_feasible_nondominated.json"
    oos_json_path = oos_dir / "oos_5000.json"
    oos_npz_path = oos_dir / "oos_5000.npz"
    if not all(path.exists() for path in (
            complete_path, config_path, front_path, oos_json_path, oos_npz_path)):
        return False
    complete, config = _read(complete_path), _read(config_path)
    front, oos = _read(front_path), _read(oos_json_path)
    seeds = config.get("seeds", {})
    provenance = config.get("policy_checkpoint") or {}
    if (complete.get("completed") is not True
            or complete.get("method") != RF_METHOD
            or complete.get("stop_reason") != "evaluation_budget"
            or complete.get("evaluation_budget") != EVALUATION_BUDGET
            or not EVALUATION_BUDGET - 3 <= int(
                complete.get("ccp_evaluation_count", -1)) <= EVALUATION_BUDGET
            or config.get("method") != RF_METHOD
            or config.get("instance_id") != record["instance_id"]
            or config.get("instance_digest") != record["instance_digest"]
            or config.get("population") != POPULATION
            or config.get("generations") != GENERATIONS
            or config.get("evaluation_budget") != EVALUATION_BUDGET
            or config.get("scenario_digest") != pairing["scenario_digest"]
            or seeds.get("algorithm") != pairing["algorithm_seed"]
            or seeds.get("training") != record["ccp100_seed"]
            or seeds.get("path") != record["path_seed"]
            or config.get("policy_training") is not False
            or provenance.get("model_sha256") != model["sha256"]
            or provenance.get("training_data_digest") != model["training_data_digest"]
            or provenance.get("rf_seed") != model["rf_seed"]
            or provenance.get("predict_jobs") != PREDICT_JOBS
            or not isinstance(front, list)):
        raise ValueError(f"completed RF formal run metadata mismatch: {run_dir}")
    if (oos.get("instance_id") != record["instance_id"]
            or oos.get("instance_digest") != record["instance_digest"]
            or oos.get("size") != OOS_SIZE
            or oos.get("oos_seed") != record["oos5000_seed"]
            or oos.get("oos_master_digest") != pairing["oos_master_digest"]
            or oos.get("source_sha256") != _sha256(front_path)
            or oos.get("evaluated_decisions") != len(front)
            or len(oos.get("candidates", [])) != len(front)
            or [row.get("decision_fingerprint") for row in oos.get("candidates", [])]
            != [row.get("decision_fingerprint") for row in front]):
        raise ValueError(f"completed RF OOS metadata mismatch: {oos_dir}")
    return True


def audit_rf_test_outputs(root: Path, records: list[dict], models: dict[int, dict],
                          audit: dict) -> dict:
    config_path = root / "formal_test_configuration.json"
    complete_path = root / "RF_FORMAL_COMPLETE.json"
    manifest_path = root / "rf_formal_source_manifest.json"
    if not all(path.exists() for path in (config_path, complete_path, manifest_path)):
        raise ValueError(f"RF formal-test root is incomplete: {root}")
    config, complete = _read(config_path), _read(complete_path)
    if (complete.get("completed") is not True
            or complete.get("formal") is not True
            or complete.get("completed_task_count") != 360
            or complete.get("rf_optimization_count") != 360
            or complete.get("rf_oos_count") != 360
            or complete.get("evaluation_budget") != EVALUATION_BUDGET
            or complete.get("population") != POPULATION
            or complete.get("generations") != GENERATIONS
            or complete.get("formal_hv_values_read") is not False
            or complete.get("hv_computed") is not False
            or complete.get("legacy_hv_used") is not False
            or complete.get("existing_source_audit_digest") != _digest(audit)
            or complete.get("rf_formal_source_manifest_sha256")
            != _sha256(manifest_path)):
        raise ValueError("RF formal-test completion metadata mismatch")
    for key in (
            "catalog_digest", "normalization_sha256", "source_data_sha256",
            "blocks", "test_instances_per_block", "rf_optimization_count",
            "rf_oos_count", "evaluation_budget", "population", "generations",
            "oos_size", "rf_predict_jobs", "pairing"):
        if complete.get(key) != config.get(key):
            raise ValueError(f"RF formal-test config/completion mismatch: {key}")
    for block in BLOCKS:
        for index, record in enumerate(records):
            pairing = audit["blocks"][str(block)]["instances"][index]
            run_dir, oos_dir = _rf_paths(root, block, record["instance_id"])
            if not _completed_rf(run_dir, oos_dir, record, models[block], pairing):
                raise ValueError(f"RF formal-test task is incomplete: {run_dir}")
    return complete


def _run_command(command: list[str], log_path: Path, environment: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as handle:
        handle.write("[COMMAND] " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(
            command, cwd=ROOT.parent, env=environment,
            stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {log_path}")


def _run_one(task, args, records, models, audit):
    block, index = task
    record = records[index]
    pairing = audit["blocks"][str(block)]["instances"][index]
    model = models[block]
    run_dir, oos_dir = _rf_paths(args.out, block, record["instance_id"])
    if _completed_rf(run_dir, oos_dir, record, model, pairing):
        return block, record["instance_id"], "skipped", 0.0
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "",
    })
    instance_path = args.instances / f"{record['instance_id']}.json"
    started = time.perf_counter()
    run_log = args.out / f"block_{block}" / "logs" / f"{record['instance_id']}_run.log"
    _run_command([
        sys.executable, "-u", str(RF_RUNNER),
        "--data", str(args.data), "--instance", str(instance_path),
        "--model", model["path"], "--normalization", str(args.normalization),
        "--out", str(run_dir), "--algorithm-seed", str(pairing["algorithm_seed"]),
        "--evaluation-budget", str(EVALUATION_BUDGET),
        "--pop", str(POPULATION), "--gens", str(GENERATIONS),
        "--predict-jobs", str(PREDICT_JOBS),
    ], run_log, environment)
    oos_log = args.out / f"block_{block}" / "logs" / f"{record['instance_id']}_oos.log"
    _run_command([
        sys.executable, "-u", str(OOS_RUNNER),
        "--data", str(args.data), "--instance", str(instance_path),
        "--run", str(run_dir), "--out", str(oos_dir),
    ], oos_log, environment)
    if not _completed_rf(run_dir, oos_dir, record, model, pairing):
        raise RuntimeError(f"RF formal task exited without valid completion: {run_dir}")
    return block, record["instance_id"], "completed", time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--rf-training-root", type=Path, required=True)
    parser.add_argument("--existing-block", type=_parse_block_path,
                        action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    try:
        existing_blocks = _exact_block_mapping(args.existing_block)
    except ValueError as error:
        parser.error(str(error))
    for name in ("data", "instances", "normalization", "rf_training_root", "out"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        records, models, audit = audit_inputs(
            args.data, args.instances, args.normalization,
            args.rf_training_root, existing_blocks)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    audit_digest = _digest(audit)
    config = {
        "schema_version": 1,
        "stage": "RF-Reward formal test E; RF optimization and OOS only",
        "formal": True,
        "catalog_digest": EXPECTED_CATALOG_DIGEST,
        "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
        "source_data_sha256": audit["source_data_sha256"],
        "blocks": 3,
        "test_instances_per_block": 120,
        "rf_optimization_count": 360,
        "rf_oos_count": 360,
        "evaluation_budget": EVALUATION_BUDGET,
        "maximum_ccp100_evaluations": 360 * EVALUATION_BUDGET,
        "population": POPULATION,
        "generations": GENERATIONS,
        "oos_size": OOS_SIZE,
        "rf_predict_jobs": PREDICT_JOBS,
        "workers": args.workers,
        "pairing": "existing block algorithm/path/CCP100/OOS seeds",
        "rf_model_block_mapping": {str(key): value for key, value in models.items()},
        "existing_source_audit_digest": audit_digest,
        "legacy_hv_used": False,
        "partial_hv_computed": False,
        "git_commit": _git_commit(),
        "implementation_sha256": {
            path.name: _sha256(path) for path in (
                Path(__file__), RF_RUNNER, OOS_RUNNER,
                ROOT / "transformer_ppo/rf_reward.py",
                ROOT / "transformer_ppo/episode.py",
            )
        },
    }
    if args.preflight_only:
        print(json.dumps({
            "existing_source_audit_digest": audit_digest,
            "existing_run_count": audit["existing_run_count"],
            "existing_oos_count": audit["existing_oos_count"],
            "rf_models": {str(key): value["sha256"] for key, value in models.items()},
            "rf_optimization_count": 360,
            "rf_oos_count": 360,
            "maximum_ccp100_evaluations": 360 * EVALUATION_BUDGET,
            "formal_hv_values_read": False,
        }, sort_keys=True))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "formal_test_configuration.json"
    audit_path = args.out / "existing_source_audit.json"
    if config_path.exists() or audit_path.exists():
        if (not config_path.exists() or not audit_path.exists()
                or _read(config_path) != config or _read(audit_path) != audit):
            parser.error("existing stage-E configuration differs; use a new output directory")
    else:
        _atomic_json(audit_path, audit)
        _atomic_json(config_path, config)
    tasks = [(block, index) for block in BLOCKS for index in range(len(records))]
    incomplete = []
    complete_count = 0
    for block, index in tasks:
        record = records[index]
        pairing = audit["blocks"][str(block)]["instances"][index]
        run_dir, oos_dir = _rf_paths(args.out, block, record["instance_id"])
        completed = _completed_rf(run_dir, oos_dir, record, models[block], pairing)
        if completed:
            complete_count += 1
            continue
        for path in (run_dir, oos_dir):
            if path.exists() and any(path.iterdir()):
                incomplete.append(str(path))
        for suffix in ("run", "oos"):
            log = args.out / f"block_{block}" / "logs" / f"{record['instance_id']}_{suffix}.log"
            if log.exists():
                incomplete.append(str(log))
    if incomplete:
        raise RuntimeError(
            "incomplete outputs require inspection before resume:\n"
            + "\n".join(sorted(set(incomplete))))
    print(
        f"[RF-FORMAL-E] total=360 already_complete={complete_count} "
        f"workers={args.workers} budget={EVALUATION_BUDGET} "
        f"audit_digest={audit_digest}", flush=True)
    started = time.perf_counter()
    completed = complete_count
    durations = []
    pending = []
    for task in tasks:
        block, index = task
        run_dir, oos_dir = _rf_paths(args.out, block, records[index]["instance_id"])
        pairing = audit["blocks"][str(block)]["instances"][index]
        if not _completed_rf(run_dir, oos_dir, records[index], models[block], pairing):
            pending.append(task)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_one, task, args, records, models, audit): task
            for task in pending
        }
        try:
            for future in as_completed(futures):
                block, instance_id, status, duration = future.result()
                completed += 1
                if status == "completed":
                    durations.append(duration)
                elapsed = time.perf_counter() - started
                new_count = max(1, completed - complete_count)
                eta = elapsed / new_count * (len(tasks) - completed) / 3600.0
                print(
                    f"[RF-FORMAL-E] {completed:03d}/360 block={block} "
                    f"{instance_id} {status} seconds={duration:.1f} eta_hours={eta:.2f}",
                    flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    rows = []
    actual_evaluations = 0
    for block, index in tasks:
        record = records[index]
        pairing = audit["blocks"][str(block)]["instances"][index]
        run_dir, oos_dir = _rf_paths(args.out, block, record["instance_id"])
        if not _completed_rf(run_dir, oos_dir, record, models[block], pairing):
            raise RuntimeError(f"incomplete RF formal output: {run_dir}")
        complete = _read(run_dir / "COMPLETE.json")
        actual_evaluations += int(complete["ccp_evaluation_count"])
        rows.append({
            "block": block,
            "instance_id": record["instance_id"],
            "configuration": record["configuration"],
            "model_sha256": models[block]["sha256"],
            "run_complete_sha256": _sha256(run_dir / "COMPLETE.json"),
            "front_sha256": _sha256(run_dir / "final_feasible_nondominated.json"),
            "oos_json_sha256": _sha256(oos_dir / "oos_5000.json"),
            "oos_npz_sha256": _sha256(oos_dir / "oos_5000.npz"),
        })
    provenance_path = args.out / "rf_formal_source_manifest.json"
    _atomic_json(provenance_path, rows)
    summary = {
        **config,
        "completed": True,
        "completed_task_count": 360,
        "resumed_completed_count": complete_count,
        "new_completed_count": len(durations),
        "actual_ccp100_evaluations": actual_evaluations,
        "wall_seconds": time.perf_counter() - started,
        "mean_new_task_seconds": sum(durations) / len(durations) if durations else 0.0,
        "rf_formal_source_manifest_sha256": _sha256(provenance_path),
        "formal_hv_values_read": False,
        "hv_computed": False,
    }
    _atomic_json(args.out / "RF_FORMAL_COMPLETE.json", summary)
    print(json.dumps({key: summary[key] for key in (
        "completed", "completed_task_count", "actual_ccp100_evaluations",
        "formal_hv_values_read", "hv_computed")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
