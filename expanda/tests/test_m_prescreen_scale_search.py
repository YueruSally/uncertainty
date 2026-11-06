from pathlib import Path
from copy import deepcopy
import tempfile
import unittest

import numpy as np
import torch

import baseline_uncertainty as base
from experiments.run_m_prescreen_scale_search_v1 import (
    INSTANCES, METHODS, SEEDS, summarize)
from generate_m_prescreen_search_instances import (CCP_SEED_BASE, COUNT,
    LOCKED_FINAL_TEST_MASTER_SEED, MANIFEST_ALGORITHM_SEED_BASE, MASTER_SEED,
    PATH_SEED_OFFSET)
from m_operator_offline import digest
from m_prescreen_scale_policy import (FrozenScaleRanker,
    ScaleSearchController, ScaleSearchPolicy, choose_first_maximum)
from m_prescreen_surrogate import CandidateGainTransformer
from transformer_ppo.archive import RewardArchive, RewardConfig
import tests.test_m_operator_execution_consistency as fixtures


class PrescreenScaleSearchTests(unittest.TestCase):
    def test_search_seed_namespaces_precede_locked_final_test(self):
        values = {MASTER_SEED+index*100 for index in range(COUNT)}
        values |= {MASTER_SEED+index*100+PATH_SEED_OFFSET
                   for index in range(COUNT)}
        values |= {CCP_SEED_BASE+index for index in range(COUNT)}
        values |= {MANIFEST_ALGORITHM_SEED_BASE+index
                   for index in range(COUNT)}
        self.assertEqual(COUNT, 3)
        self.assertTrue(all(value < LOCKED_FINAL_TEST_MASTER_SEED
                            for value in values))
        self.assertEqual(len(INSTANCES), 3)
        self.assertEqual(len(SEEDS), 10)

    def test_first_maximum_keeps_balanced_on_tie(self):
        self.assertEqual(choose_first_maximum([0, 0, -1]), 0)
        self.assertEqual(choose_first_maximum([0, .2, .2]), 1)

    def test_frozen_ranker_loads_three_fixed_models_and_predicts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); checkpoints = []; hashes = {}
            for seed in (1, 2, 3):
                torch.manual_seed(seed)
                model = CandidateGainTransformer(4, 3, 14, width=16)
                path = root/f"seed_{seed}.pt"
                torch.save({"protocol": "m-prescreen-scale-ranker-v1",
                    "training_instances": 200, "seed": seed, "epoch": 30,
                    "width": 16, "model": model.state_dict(),
                    "batch_feature_count": 4,
                    "candidate_feature_count": 3,
                    "context_feature_count": 14}, path)
                checkpoints.append(str(path)); hashes[str(path)] = digest(path)
            scaler = {"clip": 10.0,
                "batch": {"mean": [0]*4, "std": [1]*4},
                "candidate": {"mean": [0]*3, "std": [1]*3},
                "context": {"mean": [0]*14, "std": [1]*14},
                "target": {"mean": [0]*3, "std": [1]*3}}
            bundle = {"models": {"200": {"scaler": scaler,
                "target_scale": .01, "checkpoints": checkpoints,
                "checkpoint_sha256": hashes}}}
            predictor = FrozenScaleRanker(bundle, digest)
            row = {"generation": 10, "evaluation": 220,
                "population_points": [[.2, .3, .4], [.3, .2, .5]],
                "candidate_features": np.zeros((8, 3)).tolist(),
                "candidate_batch_tokens": np.zeros((8, 2, 4)).tolist()}
            values = predictor.predict(row)
            self.assertEqual(values.shape, (8,))
            self.assertEqual(values[0], 0.0)
            self.assertTrue(np.isfinite(values).all())

    def test_search_controller_never_evaluates_hidden_candidates(self):
        fixture = fixtures.OperatorExecutionConsistencyTests()
        fixture.setUp()

        class Predictor:
            def predict(self, row):
                self.row = row
                return np.arange(8, dtype=float)

        try:
            for method in METHODS:
                predictor = (Predictor()
                    if method == "TRANSFORMER_K8_SIZE200" else None)
                policy = ScaleSearchPolicy("fixture", 31, method, predictor)
                with tempfile.TemporaryDirectory() as folder:
                    controller = ScaleSearchController(Path(folder),
                        "fixture", "frozen100", 400, trainer=policy,
                        archive=RewardArchive(fixture.normalizer),
                        reward_config=RewardConfig(),
                        batches=fixture.batches, generations=1000,
                        pop_size=20, method=policy.name,
                        instance={"instance_id": "fixture"},
                        seeds={"algorithm": 31})
                    controller.generation = 10
                    controller.phase = "offspring"
                    controller._active_mixture_event_index = 1
                    fixture.parent.od_allocations[fixture.key] = [
                        base.PathAllocation(
                            fixture.paths[("A", "B")][0], 1.0)]
                    population_member = deepcopy(fixture.parent)
                    base._evaluate_individual_impl(population_member,
                        fixture.batches, fixture.arcs, fixture.tt, 0.0, 0.0)
                    controller.current_population = [population_member]
                    base.ACTIVE_MUTATION_LOGGER = controller
                    try:
                        controller._mutate_operator_only(
                            deepcopy(fixture.parent), fixture.batches,
                            fixture.paths, fixture.tt, fixture.lookup,
                            fixture.arcs, 0.0, 0.0, fixture.options, 1, {})
                    finally:
                        base.ACTIVE_MUTATION_LOGGER = None
                    self.assertEqual(controller.evaluations, 1)
                    self.assertEqual(controller.feedback_evaluations, 0)
                    audit = controller.summary()["k8_scale_search"]
                    self.assertEqual(
                        audit["hidden_candidate_objective_evaluations"], 0)
                    controller.close()
        finally:
            fixture.tearDown()

    @staticmethod
    def make_run(instance, seed, method, transformer_gain):
        baseline = .5+1e-5*(seed-SEEDS[0])
        hv = baseline*(1+transformer_gain/100.0) \
            if method == "TRANSFORMER_K8_SIZE200" else baseline
        return {"instance": instance, "algorithm_seed": seed,
            "strategy": method, "final_hypervolume": hv,
            "real_objective_evaluation_count": 400,
            "evaluation_budget": 400, "ppo_updates": 0,
            "training_feedback_evaluations": 0,
            "counterfactual_evaluation_count": 0,
            "optimisation_seconds": 1.0,
            "pairing": {"seed": seed, "instance": instance},
            "k8_scale_search": {"method": method, "pool_events": 10,
                "pool_candidates": 80,
                "hidden_candidate_objective_evaluations": 0,
                "switches": 2 if method == "TRANSFORMER_K8_SIZE200" else 0}}

    def test_summary_requires_positive_mean_on_every_instance(self):
        runs = [self.make_run(instance, seed, method, 1.0)
                for instance in INSTANCES for seed in SEEDS
                for method in METHODS]
        result = summarize(runs)
        self.assertTrue(result["development_search_pass"])
        self.assertEqual(result["pooled_equal_instance"]["positive_instances"], 3)
        for row in runs:
            if (row["instance"] == INSTANCES[-1]
                    and row["strategy"] == "TRANSFORMER_K8_SIZE200"):
                row["final_hypervolume"] /= 1.03
        result = summarize(runs)
        self.assertFalse(result["development_search_pass"])


if __name__ == "__main__":
    unittest.main()
