import tempfile
import unittest
from pathlib import Path

import torch

from c_k8_scale_policy import FrozenCCrossoverRanker
from i_k8_surrogate import CandidateGainTransformer, fit_scaler, pack_rows
from m_operator_offline import digest
from experiments.run_c_k8_transformer_search_v1 import (
    BASE, DIRECTED, INSTANCES, METHODS, ORACLE, SEEDS, SIMPLE, TRANSFORMER,
    summarize)


def row():
    return {
        "population_points": [[.1, .2, .3], [.2, .1, .4]],
        "slot_fraction": .5,
        "candidate_features": [[float(i + j) for j in range(15)]
                               for i in range(8)],
        "candidate_batch_tokens": [
            [[float(i + j) for j in range(30)]] for i in range(8)],
    }


class CK8TransformerScaleTests(unittest.TestCase):
    def test_pair_state_packs_for_small_transformer(self):
        state = row(); scaler = fit_scaler([state])
        packed = pack_rows([state], scaler)
        model = CandidateGainTransformer(
            packed[0].shape[-1], packed[1].shape[-1],
            packed[2].shape[-1], width=16)
        self.assertEqual(tuple(model(*packed).shape), (1, 8))

    def test_frozen_policy_loads_three_checkpoints(self):
        state = row(); scaler = fit_scaler([state]); packed = pack_rows([state], scaler)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoints = []; hashes = {}
            for seed in (1, 2, 3):
                model = CandidateGainTransformer(
                    packed[0].shape[-1], packed[1].shape[-1],
                    packed[2].shape[-1], width=16)
                path = root / f"{seed}.pt"
                torch.save({"protocol": "c-k8-scale-transformer-v1",
                    "training_instances": 200, "width": 16,
                    "batch_feature_count": packed[0].shape[-1],
                    "candidate_feature_count": packed[1].shape[-1],
                    "context_feature_count": packed[2].shape[-1],
                    "model": model.state_dict()}, path)
                checkpoints.append(str(path)); hashes[str(path)] = digest(path)
            bundle = {"protocol": "c-k8-scale-transformer-curve-v1",
                "selected_size": 200, "models": {"200": {
                    "scaler": scaler, "target_scale": 1.0,
                    "checkpoints": checkpoints,
                    "checkpoint_sha256": hashes}}}
            policy = FrozenCCrossoverRanker(bundle, digest)
            self.assertIn(policy.select(state), range(8))

    def test_development_summary_requires_gain_over_base_and_simple(self):
        values = {BASE: 1.0, DIRECTED: 1.01, SIMPLE: 1.03,
                  TRANSFORMER: 1.06, ORACLE: 1.10}
        runs = []
        for instance in INSTANCES:
            for seed in SEEDS:
                for method in METHODS:
                    runs.append({"instance": instance,
                        "algorithm_seed": seed, "strategy": method,
                        "scenario_digest": "same", "pairing": {"seed": seed},
                        "final_hypervolume": values[method],
                        "optimisation_seconds": 1.0,
                        "real_objective_evaluation_count": 400,
                        "crossover_k8_hidden_candidate_objective_evaluations":
                            14 if method == ORACLE else 0})
        result = summarize(runs)
        self.assertTrue(result["development_search_pass"])
        self.assertEqual(result["reference_method"], BASE)
        self.assertGreater(result["methods"][TRANSFORMER]["vs_simple"]["mean"], 0)


if __name__ == "__main__": unittest.main()
