"""One bounded paired fit; same samples, model size, seeds, updates and gate."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from m_candidate_information import (METHODS, TRAIN_INSTANCES, VALID_INSTANCE,
    TRAIN_SEEDS, THRESHOLD_SCALED, SPECIFIC_NAMES)
from m_operator_offline import (TinyOperatorTransformer, fit_scaler, transform,
    train_transformer, predict_transformer, choose, read_json, write_json, digest)


def initial_hash(dim, seed):
    torch.manual_seed(seed)
    model = TinyOperatorTransformer(dim)
    return hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()


def fit(dataset, out):
    torch.set_num_threads(1)
    data = read_json(dataset)
    if data["protocol"] != "candidate-information-v1" or not data["quality_pass"]:
        raise ValueError("requires verified concrete-candidate data")
    rows = data["rows"]
    names = np.array([r["instance"] for r in rows])
    train = np.flatnonzero(np.isin(names, TRAIN_INSTANCES))
    valid = np.flatnonzero(names == VALID_INSTANCE)
    if set(names) != set(TRAIN_INSTANCES + (VALID_INSTANCE,)) or not len(train) or not len(valid):
        raise ValueError("frozen training/validation split is incomplete")
    summary = np.asarray([r["summary"] for r in rows], dtype=float)
    specific = np.asarray([r["specific"] for r in rows], dtype=float)
    mask = np.asarray([r["eligible"] for r in rows], dtype=bool)
    y = np.asarray([r["relative_gain"] for r in rows], dtype=float)
    x = np.concatenate([summary, specific], axis=2)
    if (not np.isfinite(x).all() or not np.isfinite(y[mask]).all() or not mask[:, 4].all()
            or not (y[:, 4] == 0).all() or specific.shape[-1] != len(SPECIFIC_NAMES)):
        raise ValueError("invalid features/labels/masks")
    scaler = fit_scaler(x[train], mask[train], y[train])
    xx, yy = transform(x, scaler), (y / scaler["target_scale"]).astype(np.float32)
    summary_dim = summary.shape[-1]
    checkpoints, histories, initializations, offline = {}, {}, {}, {}
    for method in METHODS[1:]:
        features = xx.copy()
        if method == "STATE_ONLY":
            features[:, :, summary_dim:] = 0.
        predictions = []
        checkpoints[method], histories[method], initializations[method] = [], [], []
        for seed in TRAIN_SEEDS:
            print(f"[M-CAND-TRAIN] {method} seed={seed} start; 20 epochs", flush=True)
            initializations[method].append(initial_hash(x.shape[-1], seed))
            model, history = train_transformer((features[train], mask[train], yy[train]),
                (features[valid], mask[valid], yy[valid]), seed, epochs=20, patience=21,
                progress=lambda msg: print(f"[M-CAND-TRAIN] {method} seed={seed} {msg}", flush=True))
            if history["history"][-1]["epoch"] != 20:
                raise RuntimeError("paired training budget changed")
            path = (out / method / f"seed_{seed}.pt").resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "method": method}, path)
            checkpoints[method].append(str(path))
            histories[method].append(history)
            predictions.append(predict_transformer(model, features, mask)*scaler["target_scale"])
        prediction = np.mean(predictions, axis=0)
        actions = choose(prediction[valid], mask[valid], THRESHOLD_SCALED*scaler["target_scale"])
        gains = y[valid, actions]
        offline[method] = {"validation_states": len(valid), "mean_net_gain": float(gains.mean()),
            "switches": int((actions != 4).sum()), "role": "descriptive only; does not select threshold or block the bounded search"}
    if initializations[METHODS[1]] != initializations[METHODS[2]]:
        raise RuntimeError("paired initial weights differ")
    bundle = {"protocol": "candidate-information-v1", "dataset_sha256": digest(dataset),
        "scaler": scaler, "summary_dim": summary_dim, "specific_names": SPECIFIC_NAMES,
        "threshold_scaled": THRESHOLD_SCALED, "threshold_raw": THRESHOLD_SCALED*scaler["target_scale"],
        "training_instances": TRAIN_INSTANCES, "validation_instance": VALID_INSTANCE,
        "training_states": len(train), "validation_states": len(valid), "seeds": TRAIN_SEEDS,
        "epochs": 20, "paired_initialization_exact": True, "initialization_hashes": initializations,
        "checkpoints": checkpoints, "checkpoint_sha256": {p: digest(p) for paths in checkpoints.values() for p in paths},
        "checkpoint_selection": "minimum inner-validation MSE among epochs 0..20; no search outcomes used",
        "gate_selection": "fixed 0.02 times training RMS; no threshold tuning",
        "offline": offline, "histories": histories, "ppo_updates": 0}
    write_json(out / "BUNDLE.json", bundle)
    print("[M-CAND-TRAIN] COMPLETE; both information arms proceed to the predeclared B400 diagnostic", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fit(args.dataset, args.out)
