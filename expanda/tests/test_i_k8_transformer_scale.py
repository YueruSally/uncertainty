import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from experiments.run_i_k8_transformer_search_v1 import (
    BALANCED, BASE, METHODS, ORACLE, SIMPLE, TRANSFORMER,
    compatible_frozen_plan, hydrate_scenario_digest, summarize)
from i_k8_surrogate import (CandidateGainTransformer, fit_scaler,
    gain_targets, pack_rows, pairwise_ranking_loss, score)
from i_k8_scale_policy import FrozenIInitializationRanker
from m_operator_offline import digest
from train_i_k8_transformer_scale_v1 import SIZES, instance_order


def synthetic_rows():
    rows = []
    for state in range(4):
        labels = []
        for action in range(8):
            labels.append({
                "index": action,
                "feasible": True,
                "objectives": [10.0 + action, 20.0, 30.0],
                "hv_contribution": .01 * action,
            })
        rows.append({
            "instance": f"instance-{state % 2}",
            "trajectory_seed": 10 + state,
            "event_index": state,
            "slot_fraction": state / 19.0,
            "population_points": [] if state == 0 else [[.4, .5, .6]],
            "candidate_features": [
                [float(action), float(state), 1.0] for action in range(8)
            ],
            "candidate_batch_tokens": [[
                [float(action), float(batch), float(state), 1.0]
                for batch in range(3)] for action in range(8)],
            "labels": labels,
            "simple_selected_index": 1,
        })
    return rows


class InitializationTransformerScaleTests(unittest.TestCase):
    def test_legacy_run_restores_scenario_digest_from_configuration(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "configuration.json").write_text(json.dumps({
                "scenario_digest": "frozen-ccp100",
            }))
            row = hydrate_scenario_digest({"strategy": BASE}, root)
            self.assertEqual(row["scenario_digest"], "frozen-ccp100")

    def test_resume_allows_only_summarizer_hash_change(self):
        script = str(Path(
            __import__(
                "experiments.run_i_k8_transformer_search_v1",
                fromlist=["__file__"]).__file__).resolve())
        existing = {"workers": 3, "input_sha256": {
            script: "old", "/fixed/input": "same"}}
        generated = {"workers": 3, "input_sha256": {
            script: "new", "/fixed/input": "same"}}
        self.assertTrue(compatible_frozen_plan(existing, generated))
        generated["input_sha256"]["/fixed/input"] = "changed"
        self.assertFalse(compatible_frozen_plan(existing, generated))

    def test_tensor_contract_and_pairwise_loss(self):
        rows = synthetic_rows()
        scaler = fit_scaler(rows)
        packed = pack_rows(rows, scaler)
        model = CandidateGainTransformer(
            packed[0].shape[-1], packed[1].shape[-1], packed[2].shape[-1],
            width=16)
        prediction = model(*packed)
        self.assertEqual(tuple(prediction.shape), (4, 8))
        self.assertTrue(torch.equal(prediction[:, 0], torch.zeros(4)))
        loss = pairwise_ranking_loss(prediction, gain_targets(rows, .01))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_score_uses_base_index_zero_and_compares_simple(self):
        rows = synthetic_rows()
        predictions = np.tile(np.arange(8, dtype=float), (len(rows), 1))
        predictions[:, 0] = 0.0
        result = score(rows, predictions)
        self.assertAlmostEqual(result["transformer_mean_net_gain_vs_base"], .07)
        self.assertAlmostEqual(result["simple_mean_net_gain_vs_base"], .01)
        self.assertAlmostEqual(result["transformer_minus_simple_mean"], .06)

    def test_nested_instance_order_is_deterministic(self):
        names = {f"instance-{index:03d}" for index in range(200)}
        first = instance_order(names)
        self.assertEqual(first, instance_order(reversed(first)))
        self.assertEqual(SIZES, (25, 50, 100, 200))
        for smaller, larger in zip(SIZES, SIZES[1:]):
            self.assertTrue(set(first[:smaller]) < set(first[:larger]))

    def test_frozen_ranker_loads_three_checkpoints_and_selects(self):
        rows = synthetic_rows()
        scaler = fit_scaler(rows)
        packed = pack_rows(rows[:1], scaler)
        with TemporaryDirectory() as folder:
            checkpoints = []
            hashes = {}
            for seed in (1, 2, 3):
                torch.manual_seed(seed)
                model = CandidateGainTransformer(
                    packed[0].shape[-1], packed[1].shape[-1],
                    packed[2].shape[-1], width=16)
                path = Path(folder) / f"seed_{seed}.pt"
                torch.save({
                    "protocol": "i-k8-scale-transformer-v1",
                    "training_instances": 200,
                    "width": 16,
                    "batch_feature_count": packed[0].shape[-1],
                    "candidate_feature_count": packed[1].shape[-1],
                    "context_feature_count": packed[2].shape[-1],
                    "model": model.state_dict(),
                }, path)
                checkpoints.append(str(path))
                hashes[str(path)] = digest(path)
            bundle = {
                "protocol": "i-k8-scale-transformer-curve-v1",
                "selected_size": 200,
                "models": {"200": {
                    "scaler": scaler,
                    "target_scale": .01,
                    "checkpoints": checkpoints,
                    "checkpoint_sha256": hashes,
                }},
            }
            ranker = FrozenIInitializationRanker(bundle, digest)
            selected = ranker.select(rows[0])
            self.assertIn(selected, range(8))

    def test_search_summary_uses_base_as_reference(self):
        factors = {
            BASE: 1.0,
            BALANCED: 1.2,
            SIMPLE: 1.1,
            TRANSFORMER: 1.3,
            ORACLE: 1.4,
        }
        runs = []
        for instance in (
                "prescreen-search-S0-001", "prescreen-search-S0-002",
                "prescreen-search-S0-003"):
            for seed in range(918001, 918011):
                for method in METHODS:
                    runs.append({
                        "instance": instance,
                        "algorithm_seed": seed,
                        "strategy": method,
                        "scenario_digest": "same",
                        "pairing": None,
                        "final_hypervolume": factors[method],
                        "optimisation_seconds": 1.0,
                        "real_objective_evaluation_count": 400,
                        "initialization_k8_hidden_candidate_objective_evaluations": (
                            140 if method == ORACLE else 0),
                    })
        result = summarize(runs)
        self.assertEqual(result["reference_method"], BASE)
        self.assertAlmostEqual(result["methods"][BASE]["vs_base"]["mean"], 0.0)
        self.assertGreater(result["methods"][TRANSFORMER]["vs_base"]["mean"], 0.0)
        self.assertGreater(result["methods"][TRANSFORMER]["vs_simple"]["mean"], 0.0)
        self.assertTrue(result["development_search_pass"])


if __name__ == "__main__":
    unittest.main()
