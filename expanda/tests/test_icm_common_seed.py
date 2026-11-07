import copy
import contextlib
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np

import baseline_uncertainty as base
from combined_k8_controller import (
    CombinedFrozenK8Controller, create_component_directories)
from m_prescreen_scale_policy import ScaleSearchPolicy
from m_operator_offline import write_json
from tests.test_c_k8_crossover import fixture
from transformer_ppo.archive import (
    ObjectiveNormalizer, RewardArchive, RewardConfig)
from experiments.run_icm_common_seed_v1 import (
    BASE, C_HEURISTIC, C_TRANSFORMER, ICM_TRANSFORMER, INSTANCES,
    I_HEURISTIC, I_TRANSFORMER, METHODS, M_HEURISTIC, M_TRANSFORMER,
    SEEDS, bundles_compatible, freeze_or_verify_plan, summarize)


class ICMCommonSeedTests(unittest.TestCase):
    def test_resume_preserves_plan_and_rejects_changed_source_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_path = root / "PLAN.json"
            plan = {"input_sha256": {"combined_k8_controller.py": "old"}}
            freeze_or_verify_plan(plan_path, plan)
            write_json(root / "RUN_COMPLETE.json", {"saved": True})
            original = plan_path.read_bytes()
            result = (root / "RUN_COMPLETE.json").read_bytes()
            freeze_or_verify_plan(plan_path, plan)
            with self.assertRaisesRegex(ValueError, "new --out"):
                freeze_or_verify_plan(plan_path, {
                    "input_sha256": {"combined_k8_controller.py": "fixed"}})
            self.assertEqual(plan_path.read_bytes(), original)
            self.assertEqual((root / "RUN_COMPLETE.json").read_bytes(), result)
            fresh_plan = root / "fresh" / "PLAN.json"
            freeze_or_verify_plan(fresh_plan, {
                "input_sha256": {"combined_k8_controller.py": "fixed"}})
            self.assertTrue(fresh_plan.is_file())

    def test_combined_controller_completes_real_nsga2_lifecycle(self):
        # Only replace the unavailable server model loaders.  Exercise real
        # controller construction, CCP100 evaluation, I/C/M hooks and teardown.
        batch, p1, p2, _, arcs = fixture()
        paths = [parent.od_allocations[("A", "D", 7)][0].path
                 for parent in (p1, p2)]
        for arc in arcs:
            arc.mode = "road"
        for path in paths:
            path.modes = ["road"] * len(path.arcs)
        options = {("A", "D", 7): [
            base.ReliablePathOption(path, 1.0, 0.0, {}) for path in paths]}
        path_lib = {("A", "D"): paths}

        class InitializationPolicy:
            def select(self, row):
                return row["event_index"] % 8

        class CrossoverPolicy:
            def select(self, row):
                return 1

        predictor = mock.Mock()
        predictor.predict.return_value = np.arange(8, dtype=float)
        policy = ScaleSearchPolicy("fixture", 930001,
                                   "TRANSFORMER_K8_SIZE200", predictor)
        normalizer = ObjectiveNormalizer((0., 0., 0.), (10000., 10000., 100.))
        saved = {name: getattr(base, name) for name in (
            "ACTIVE_SCENARIO_SET", "ACTIVE_MUTATION_LOGGER", "RISK_METRIC",
            "_PATH_SCENARIO_CACHE", "CONFIDENCE_COST", "CONFIDENCE_EMISSION",
            "CONFIDENCE_TIME")}
        py_state, np_state = random.getstate(), np.random.get_state()
        controller = None
        try:
            random.seed(930001)
            np.random.seed(930001)
            base.RISK_METRIC = "ccp"
            base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = .9
            base.configure_scenario_set(arcs, {}, size=100, seed=123, stochastic=True)
            with tempfile.TemporaryDirectory() as folder, \
                    mock.patch("combined_k8_controller.read_json", return_value={}), \
                    mock.patch("combined_k8_controller.FrozenIResidualGate",
                               return_value=InitializationPolicy()), \
                    mock.patch("combined_k8_controller.FrozenCResidualGate",
                               return_value=CrossoverPolicy()):
                controller = CombinedFrozenK8Controller(
                    Path(folder), "fixture", "ccp100", 400,
                    trainer=policy, archive=RewardArchive(normalizer),
                    reward_config=RewardConfig(), batches=[batch],
                    generations=1000, pop_size=20, graph_observation={},
                    i_bundle="synthetic-i", c_bundle="synthetic-c",
                    reliable_options=options, arcs=arcs,
                    method=ICM_TRANSFORMER, instance={"instance_id": "fixture"},
                    seeds={"algorithm": 930001})
                base.ACTIVE_MUTATION_LOGGER = controller
                try:
                    with contextlib.redirect_stdout(io.StringIO()), \
                            mock.patch.object(base, "_evaluate_individual_impl",
                                              wraps=base._evaluate_individual_impl) as evaluate:
                        base.run_nsga2(
                            ["A", "X", "K", "J", "D"], {}, {}, {}, {},
                            arcs, [], [batch], 0., 0., {}, {}, {}, {}, {}, {},
                            path_lib, options, pop_size=20, generations=1000)
                        controller.finish_episode()
                    report = controller.summary()
                    self.assertEqual(controller.evaluations, 400)
                    self.assertEqual(evaluate.call_count, 400)
                    self.assertEqual(controller.i_controller.pool_events, 20)
                    self.assertGreater(controller.c_controller.eligible_pool_events, 0)
                    self.assertGreater(controller.pool_events, 0)
                    self.assertEqual(report["combined_hidden_candidate_objective_evaluations"], 0)
                finally:
                    controller.close()
                    self.assertTrue(all(handle.closed for child in (
                        controller, controller.i_controller, controller.c_controller)
                        for handle in child.files.values()))
        finally:
            for name, value in saved.items():
                setattr(base, name, value)
            random.setstate(py_state)
            np.random.set_state(np_state)

    def test_combined_controller_creates_child_log_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = create_component_directories(Path(folder))
            self.assertTrue((root / "I").is_dir())
            self.assertTrue((root / "C").is_dir())

    def test_size_200_ranker_is_independent_of_failed_curve_admission_flag(self):
        i_bundle = {"protocol": "i-k8-directed-residual-v1"}
        c_bundle = {"protocol": "c-k8-simple-residual-v1"}
        m_bundle = {
            "protocol": "m-prescreen-scale-learning-curve-v1",
            "selected_size": None,
            "models": {"200": {"checkpoints": ["one", "two", "three"]}},
        }
        self.assertTrue(bundles_compatible(i_bundle, c_bundle, m_bundle))
        m_bundle["models"]["200"]["checkpoints"].pop()
        self.assertFalse(bundles_compatible(i_bundle, c_bundle, m_bundle))

    def test_summary_compares_all_components_on_one_grid(self):
        factors = {
            BASE: 1.0, I_HEURISTIC: 1.10, I_TRANSFORMER: 1.12,
            C_HEURISTIC: 1.03, C_TRANSFORMER: 1.05,
            M_HEURISTIC: 1.08, M_TRANSFORMER: 1.11,
            ICM_TRANSFORMER: 1.15,
        }
        rows = []
        for instance in INSTANCES:
            for seed in SEEDS:
                for method in METHODS:
                    rows.append({
                        "instance": instance, "algorithm_seed": seed,
                        "strategy": method,
                        "scenario_digest": "same", "pairing": {"same": True},
                        "final_hypervolume": factors[method],
                        "optimisation_seconds": 1.0,
                        "real_objective_evaluation_count": 400,
                    })
        result = summarize(copy.deepcopy(rows))
        self.assertEqual(result["best_single_component"], I_TRANSFORMER)
        self.assertGreater(
            result["component_comparisons"][I_TRANSFORMER]["summary"]["mean"], 0)
        self.assertGreater(result["combined_vs_best_single"]["mean"], 0)


if __name__ == "__main__":
    unittest.main()
