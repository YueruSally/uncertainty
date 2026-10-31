import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformer_ppo.archive import (
    ObjectiveNormalizer, RewardArchive, exact_hv_3d, nondominated,
)
from transformer_ppo.config import PPOConfig, SCENARIOS, TEST_CONFIGS, TRAIN_CONFIGS
from transformer_ppo.initialization_observations import (
    INITIALIZATION_CANDIDATE_FEATURES_V3, _candidate_tokens,
)
from transformer_ppo.instance_catalog import _draw_quantity, _draw_release_time, _split_od_pool
from transformer_ppo.observations import (
    CANDIDATE_FEATURES, CANDIDATE_PATH_V3_FEATURES,
    CANDIDATE_PATH_V4_FEATURES,
    GLOBAL_FEATURES, PARENT_FEATURES,
    action_prior, build_observation, candidate_tokens,
)
from transformer_ppo.statistics import formal_report, holm
from transformer_ppo.supervised_prior import SupervisedMutationPrior
from run_transformer_ppo_formal_batch import _seeds
from train_transformer_ppo_ccp100 import _ordered_manifests


class ArchiveTests(unittest.TestCase):
    def test_exact_hv_single_point(self):
        self.assertAlmostEqual(exact_hv_3d([(0.2, 0.3, 0.4)], (1., 1., 1.)),
                               .8 * .7 * .6)

    def test_archive_is_nondominated_and_deduplicated(self):
        archive = RewardArchive(ObjectiveNormalizer((0., 0., 0.), (10., 10., 10.),
                                                     reference=(1., 1., 1.)))
        first = archive.observe((2., 3., 4.), True, "a")
        duplicate = archive.observe((1., 1., 1.), True, "a")
        dominated = archive.observe((5., 6., 7.), True, "b")
        infeasible = archive.observe((1., 1., 1.), False, "c")
        self.assertGreater(first, 0.)
        self.assertEqual(duplicate, 0.)
        self.assertEqual(dominated, 0.)
        self.assertEqual(infeasible, 0.)
        self.assertEqual(len(archive.points), 1)

    def test_normalization_uses_instance_divisor(self):
        normalizer = ObjectiveNormalizer((0., 0., 0.), (10., 10., 10.)).with_divisor((2., 4., 5.))
        self.assertEqual(normalizer.transform((10., 20., 25.)), (.5, .5, .5))

    def test_zero_reward_diagnostics_are_multilabel_and_non_mutating(self):
        archive = RewardArchive(ObjectiveNormalizer(
            (0., 0., 0.), (10., 10., 10.), reference=(1., 1., 1.)))
        archive.observe((2., 2., 2.), True, "front")
        before = (archive.points, archive.hypervolume)
        dominated = archive.diagnose_candidate((5., 5., 12.), True, "new")
        duplicate = archive.diagnose_candidate((2., 2., 2.), True, "front")
        infeasible = archive.diagnose_candidate((12., 3., 3.), False, "bad")
        self.assertTrue(dominated["dominated_by_archive"])
        self.assertTrue(dominated["outside_hv_reference"])
        self.assertTrue(duplicate["duplicate_fingerprint"])
        self.assertTrue(infeasible["infeasible"])
        self.assertTrue(infeasible["outside_hv_reference"])
        self.assertEqual((archive.points, archive.hypervolume), before)

    def test_archive_preview_matches_commit_without_mutating(self):
        archive = RewardArchive(ObjectiveNormalizer(
            (0., 0., 0.), (10., 10., 10.), reference=(1., 1., 1.)))
        archive.observe((5., 5., 5.), True, "existing")
        before = (archive.points, archive.hypervolume)
        candidates = [((2., 6., 4.), True, "candidate")]
        preview = archive.preview_many(candidates)
        self.assertEqual((archive.points, archive.hypervolume), before)
        committed = archive.observe_many(candidates)
        self.assertEqual(preview, committed)


class ObservationTests(unittest.TestCase):
    def test_supervised_mutation_prior_reweights_only_eligible_replace_actions(self):
        base_dimension = 67 + 35 + 2 * 19 + 4 + 5
        weights = [0.0] * (3 * base_dimension)
        weights[0] = 1.0
        prior = SupervisedMutationPrior({
            "schema_version": 1,
            "feature_mode": "state_relative",
            "selected_operator": "replace",
            "effective_weights": weights,
            "alpha": 0.5,
            "temperature": 0.25,
        })
        candidates = [[0.0] * 67 for _ in range(3)]
        candidates[0][0] = -1.0
        candidates[1][0] = 1.0
        candidates[2][0] = 100.0
        observation = {
            "parent": [[0.0] * 19],
            "candidates": candidates,
            "eligible": [True, True, False],
            "action_prior": [0.5, 0.5, 0.0],
            "operator_id": 4,
            "global": [0.0] * 35,
            "candidate_graph_tokens": [[], [], []],
        }
        updated, diagnostics = prior.apply(observation)
        self.assertEqual(updated["base_action_prior"], [0.5, 0.5, 0.0])
        self.assertGreater(updated["action_prior"][1], .5)
        self.assertEqual(updated["action_prior"][2], 0.0)
        self.assertGreater(diagnostics["total_variation_from_base"], 0.0)

    def test_mutation_v4_exposes_replacement_path_relative_features(self):
        values = {
            "replacement_delta_cost_per_teu": -3.,
            "replacement_delta_emission_per_teu": 4.,
            "replacement_delta_nominal_time_h": -2.,
            "replacement_delta_distance": 10.,
            "replacement_delta_bottleneck_capacity": 20.,
            "replacement_on_time_probability": .8,
            "replacement_max_lateness_h": 1.5,
            "replacement_is_current_path": 0.,
        }
        rows, eligible = candidate_tokens([{
            "target": {"batch_id": 0, "replacement_path_id": "path-2"},
            "eligible": True,
            "candidate_operator": "replace",
            **values,
        }], "replace", representation="path_v4")
        feature = {name: rows[0][index]
                   for index, name in enumerate(CANDIDATE_PATH_V4_FEATURES)}
        self.assertEqual(eligible, [True])
        self.assertEqual(len(rows[0]), len(CANDIDATE_PATH_V4_FEATURES))
        for name, value in values.items():
            self.assertAlmostEqual(feature[name], value)

    def test_mutation_v3_identifies_operator_candidates(self):
        rows, eligible = candidate_tokens([{
            "target": {"batch_id": 0}, "eligible": True,
            "candidate_operator": "replace",
            "operator_stage": True,
            "operator_legal_target_count": 3,
            "operator_legal_target_fraction": .75,
        }], "add", representation="path_v3")
        feature = {name: rows[0][index]
                   for index, name in enumerate(CANDIDATE_PATH_V3_FEATURES)}
        self.assertEqual(eligible, [True])
        self.assertEqual(len(rows[0]), len(CANDIDATE_PATH_V3_FEATURES))
        self.assertEqual(feature["candidate_is_operator_stage"], 1.)
        self.assertEqual(feature["candidate_is_add"], 0.)
        self.assertEqual(feature["candidate_is_replace"], 1.)
        self.assertEqual(feature["operator_legal_target_count"], 3.)
        self.assertEqual(feature["operator_legal_target_fraction"], .75)

    def test_initialization_v3_exposes_planned_share_ranks_and_population_use(self):
        def path(path_id, cost, emission, travel_time, capacity, mode="road"):
            arc = SimpleNamespace(
                from_node="O", to_node="D", mode=mode, distance=100.,
                capacity=capacity, cost_per_teu_km=1.,
                emission_per_teu_km=1., speed_kmh=50., is_border_arc=False)
            return SimpleNamespace(
                path_id=path_id, nodes=("O", "D"), modes=(mode,), arcs=(arc,),
                base_cost_per_teu=cost, base_emission_per_teu=emission,
                base_travel_time_h=travel_time)

        cheap = path(1, 10., 30., 5., 100.)
        clean = path(2, 20., 10., 8., 50., "rail")
        batch = SimpleNamespace(
            batch_id=7, origin="O", destination="D", quantity=40., ET=0., LT=24.)
        key = ("O", "D", 7)
        partial = SimpleNamespace(od_allocations={
            key: [SimpleNamespace(path=clean, share=.25)],
        })
        population = [SimpleNamespace(od_allocations={
            key: [SimpleNamespace(path=cheap, share=1.)],
        })]
        candidates = [
            {"eligible": True, "path": cheap},
            {"eligible": True, "path": clean},
        ]
        rows, eligible = _candidate_tokens(
            candidates, {
                "batch": batch, "selection_step": 1,
                "required_path_count": 2, "planned_share": .75,
                "remaining_planned_share": .75,
            }, partial, population, "path_v3")
        feature = {name: rows[0][index]
                   for index, name in enumerate(INITIALIZATION_CANDIDATE_FEATURES_V3)}
        self.assertEqual(eligible, [True, True])
        self.assertEqual(len(rows[0]), len(INITIALIZATION_CANDIDATE_FEATURES_V3))
        self.assertAlmostEqual(feature["planned_share"], .75)
        self.assertAlmostEqual(feature["allocated_quantity"], 30.)
        self.assertAlmostEqual(feature["candidate_cost_rank"], 0.)
        self.assertAlmostEqual(feature["candidate_emission_rank"], 1.)
        self.assertAlmostEqual(feature["candidate_capacity_rank"], 0.)
        self.assertAlmostEqual(feature["population_path_use_fraction"], 1.)
        self.assertGreater(feature["selected_edge_novelty"], 0.)

        count_rows, count_eligible = _candidate_tokens([
            {"eligible": True, "path_count": 1},
            {"eligible": True, "path_count": 2},
            {"eligible": True, "path_count": 3},
        ], {
            "batch": batch, "decision_stage": "path_count",
            "maximum_path_count": 3,
        }, SimpleNamespace(od_allocations={}), [], "path_v3")
        count_feature = {
            name: count_rows[1][index]
            for index, name in enumerate(INITIALIZATION_CANDIDATE_FEATURES_V3)}
        self.assertEqual(count_eligible, [True, True, True])
        self.assertEqual(
            len(count_rows[0]), len(INITIALIZATION_CANDIDATE_FEATURES_V3))
        self.assertEqual(count_feature["candidate_is_path_count"], 1.)
        self.assertAlmostEqual(
            count_feature["proposed_path_count_fraction"], 2. / 3.)

    def test_action_prior_renormalizes_base_mass_over_legal_candidates(self):
        candidates = [
            {"selection_probability": 0.2},
            {"selection_probability": 0.5},
            {"selection_probability": 0.3},
        ]
        self.assertEqual(
            action_prior(candidates, [True, False, True]),
            [0.4, 0.0, 0.6])

    def test_mask_and_operator_are_separate_from_list_position(self):
        rows = [
            {"target": {"batch_id": 2}, "eligible": False, "p_rule": 0.},
            {"target": {"batch_id": 7}, "eligible": True, "p_rule": 1.},
        ]
        values, eligible = candidate_tokens(rows, "add")
        self.assertEqual(eligible, [False, True])
        self.assertEqual(len(values[0]), len(values[1]))
        self.assertEqual(len(values[0]), len(CANDIDATE_FEATURES))
        self.assertNotIn(2., values[0])  # random/list batch IDs are not numeric features

    def test_complete_observation_matches_declared_dimensions(self):
        path = SimpleNamespace(arcs=[], base_cost_per_teu=2.,
                               base_emission_per_teu=3., base_travel_time_h=4.)
        allocation = SimpleNamespace(path=path, share=1.)
        individual = SimpleNamespace(
            od_allocations={("O", "D", 0): [allocation]}, objectives=(10., 20., 30.),
            feasible=True, rank=0)
        batch = SimpleNamespace(batch_id=0, origin="O", destination="D", quantity=5.)
        candidates = [{"target": {"batch_id": 0}, "eligible": True, "p_rule": 1.}]
        context = {"operator": "add", "K": 1, "deadline_window_alpha": 1.,
                   "generation": 0, "phase": "offspring", "feasible_before": True,
                   "violation_before": 0., "q90_cost_before": 10.,
                   "q90_emission_before": 20., "q90_makespan_before": 30.}
        observation = build_observation(
            individual, [batch], candidates, context, [individual], [],
            {"archive_size": 1, "archive_hv": .2},
            ObjectiveNormalizer((0., 0., 0.), (100., 100., 100.)), 100, 10, 5)
        self.assertEqual(len(observation["parent"][0]), len(PARENT_FEATURES))
        self.assertEqual(len(observation["candidates"][0]), len(CANDIDATE_FEATURES))
        self.assertEqual(len(observation["global"]), len(GLOBAL_FEATURES))


class CatalogTests(unittest.TestCase):
    def test_configuration_sets(self):
        self.assertEqual(len(TRAIN_CONFIGS), 9)
        self.assertEqual(len(TEST_CONFIGS), 12)
        self.assertEqual(SCENARIOS["S0"], (20, 1.0))
        self.assertEqual(SCENARIOS["S11"], (50, 2.0))

    def test_train_test_numeric_grids_are_disjoint(self):
        train_rng, test_rng = np.random.default_rng(1), np.random.default_rng(1)
        train_q = {_draw_quantity(train_rng, 81., 82., "train") for _ in range(100)}
        test_q = {_draw_quantity(test_rng, 81., 82., "test") for _ in range(100)}
        self.assertFalse(train_q & test_q)
        train_t = {_draw_release_time(train_rng, 48., 6., "train") for _ in range(100)}
        test_t = {_draw_release_time(test_rng, 48., 6., "test") for _ in range(100)}
        self.assertFalse(train_t & test_t)

    def test_od_split_is_disjoint_and_reproducible(self):
        pool = [("O", f"D{i}") for i in range(20)]
        a, b = _split_od_pool(pool, 123)
        a2, b2 = _split_od_pool(list(reversed(pool)), 123)
        self.assertEqual((a, b), (a2, b2))
        self.assertFalse(set(a) & set(b))


class ConfigTests(unittest.TestCase):
    def test_default_config_validates_and_digests(self):
        config = PPOConfig()
        config.validate()
        self.assertEqual(config.digest(), config.digest())

    def test_invalid_attention_width_fails(self):
        with self.assertRaises(ValueError):
            PPOConfig(hidden_dim=127, attention_heads=4).validate()

    def test_invalid_dropout_fails(self):
        for dropout in (-0.01, 1.0):
            with self.subTest(dropout=dropout), self.assertRaises(ValueError):
                PPOConfig(dropout=dropout).validate()

    def test_invalid_actor_policy_fails(self):
        with self.assertRaisesRegex(ValueError, "actor_policy"):
            PPOConfig(actor_policy="replace-base").validate()

    def test_invalid_residual_logit_limit_fails(self):
        with self.assertRaisesRegex(ValueError, "residual_logit_limit"):
            PPOConfig(residual_logit_limit=0.0).validate()

    def test_invalid_candidate_representation_fails(self):
        with self.assertRaisesRegex(ValueError, "candidate_representation"):
            PPOConfig(candidate_representation="path-hash").validate()

    def test_invalid_credit_assignment_fails(self):
        with self.assertRaisesRegex(ValueError, "credit_assignment"):
            PPOConfig(credit_assignment="cross-operation").validate()

    def test_invalid_mutation_control_fails(self):
        with self.assertRaisesRegex(ValueError, "mutation_control"):
            PPOConfig(mutation_control="operator-only").validate()

    def test_replace_path_control_requires_v4_terminal_credit(self):
        PPOConfig(
            mutation_control="replace_path",
            candidate_representation="path_v4",
            credit_assignment="operation_terminal").validate()
        with self.assertRaisesRegex(ValueError, "requires path_v4"):
            PPOConfig(mutation_control="replace_path").validate()

    def test_initialization_control_is_validated_separately_from_features(self):
        PPOConfig(
            candidate_representation="path_v3",
            initialization_control="path").validate()
        PPOConfig(
            candidate_representation="path_v3",
            initialization_control="path_count").validate()
        with self.assertRaisesRegex(ValueError, "initialization_control"):
            PPOConfig(initialization_control="count-only").validate()
        with self.assertRaisesRegex(ValueError, "requires path_v3"):
            PPOConfig(initialization_control="path_count").validate()


class MutationCreditTests(unittest.TestCase):
    def test_policy_state_log_preserves_full_observation_and_event_identity(self):
        import json

        from learning_mutation import MutationTarget
        from transformer_ppo.archive import RewardConfig
        from transformer_ppo.controller import PPOController

        trainer = SimpleNamespace(
            name="fake-m", updates=0,
            config=SimpleNamespace(
                mutation_control="replace_path",
                credit_assignment="operation_terminal",
                candidate_representation="path_v4"),
            observation_provider=None)
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        observation = {
            "parent": [[1.0, 2.0]],
            "candidates": [[3.0], [4.0]],
            "global": [5.0],
            "eligible": [True, True],
            "action_prior": [0.5, 0.5],
            "operator_id": 4,
        }
        candidates = [{
            "target": MutationTarget(7, 0, None, str(path_id)),
            "path_id": str(path_id),
            "candidate_operator": "replace",
            "eligible": True,
        } for path_id in (101, 202)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = PPOController(
                root, "run", "scenario", 20, trainer=trainer,
                archive=RewardArchive(normalizer), reward_config=RewardConfig(),
                batches=[], generations=2, pop_size=2, method=trainer.name,
                instance={"instance_id": "toy", "K": 1,
                          "deadline_window_alpha": 1.0},
                seeds={"algorithm": 7})
            try:
                controller.current_individual = SimpleNamespace()
                controller._active_mixture_event_index = 13
                with patch(
                        "transformer_ppo.controller.build_observation",
                        return_value=observation):
                    actual = controller._observation(
                        candidates, {"operator": "replace"})
                self.assertIs(actual, observation)
                row = json.loads(
                    (root / "mutation_policy_states.jsonl")
                    .read_text(encoding="utf-8").strip())
                self.assertEqual(row["mixture_event_index"], 13)
                self.assertEqual(row["operator"], "replace")
                self.assertEqual(row["observation"], observation)
                self.assertEqual(
                    [item["target"]["replacement_path_id"]
                     for item in row["candidates"]],
                    ["101", "202"])
            finally:
                controller.close()

    def test_model_choice_event_enables_exactly_one_global_mutation_call(self):
        from transformer_ppo.archive import RewardConfig
        from transformer_ppo.controller import PPOController

        trainer = SimpleNamespace(
            name="fake-m", updates=0,
            config=SimpleNamespace(
                mutation_control="location", credit_assignment="continuous"),
            observation_provider=None)
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        with tempfile.TemporaryDirectory() as directory:
            controller = PPOController(
                Path(directory), "run", "scenario", 20, trainer=trainer,
                archive=RewardArchive(normalizer), reward_config=RewardConfig(),
                batches=[], generations=2, pop_size=2, method=trainer.name,
                instance={"instance_id": "toy", "K": 1,
                          "deadline_window_alpha": 1.0}, seeds={},
                model_choice_event=2)
            try:
                self.assertEqual(controller._next_control_branch(), (False, 1))
                self.assertEqual(controller._next_control_branch(), (True, 2))
                self.assertEqual(controller._next_control_branch(), (False, 3))
                summary = controller.summary()
                self.assertEqual(summary["model_choice_event"], 2)
                self.assertEqual(summary["mixture_event_count"], 3)
            finally:
                controller.close()

    def test_model_choice_event_rejects_probability_gating(self):
        from transformer_ppo.archive import RewardConfig
        from transformer_ppo.controller import PPOController

        trainer = SimpleNamespace(
            name="fake-m", updates=0,
            config=SimpleNamespace(
                mutation_control="location", credit_assignment="continuous"),
            observation_provider=None)
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "probability gating"):
                PPOController(
                    Path(directory), "run", "scenario", 20, trainer=trainer,
                    archive=RewardArchive(normalizer),
                    reward_config=RewardConfig(), batches=[], generations=2,
                    pop_size=2, method=trainer.name,
                    instance={"instance_id": "toy", "K": 1,
                              "deadline_window_alpha": 1.0}, seeds={},
                    model_choice_probability=0.1, model_choice_event=2)

    def test_location_control_can_end_each_mutation_operation(self):
        from multi_instance_logging import MutationLogger
        from transformer_ppo.archive import RewardConfig
        from transformer_ppo.controller import PPOController

        class Trainer:
            name = "fake-m"
            updates = 0

            def __init__(self):
                self.config = SimpleNamespace(
                    mutation_control="location",
                    credit_assignment="operation_terminal")
                self.pending = None
                self.sequence_policy_actions = 0
                self.observation_provider = None
                self.feature_construction_seconds = 0.0
                self.ppo_update_seconds = 0.0
                self.update_records = []
                self.done = []

            def observe_outcome(self, _reward, done=False):
                self.done.append(bool(done))
                self.pending = None

        trainer = Trainer()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        individual = SimpleNamespace(
            od_allocations={}, objectives=(2.0, 3.0, 4.0), feasible=True)

        def fake_location_mutation(controller, *_args, **_kwargs):
            trainer.pending = {"synthetic": True}
            controller.pending.append((None, {"decision_kind": "model_choice"}))
            return "mod", True

        with tempfile.TemporaryDirectory() as directory:
            controller = PPOController(
                Path(directory), "run", "scenario", 20, trainer=trainer,
                archive=RewardArchive(normalizer), reward_config=RewardConfig(),
                batches=[], generations=2, pop_size=2, method=trainer.name,
                instance={"instance_id": "toy", "K": 1,
                          "deadline_window_alpha": 1.0}, seeds={})
            try:
                with patch.object(
                        MutationLogger, "mutate", autospec=True,
                        side_effect=fake_location_mutation):
                    controller.mutate(
                        individual, [], {}, {}, {}, [], 0.0, 0.0)
                self.assertEqual(trainer.done, [True])
                self.assertEqual(
                    controller.summary()["credit_assignment"],
                    "operation_terminal")
            finally:
                controller.close()


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
class CheckpointTests(unittest.TestCase):
    def test_known_advantages_move_actor_probability_in_expected_direction(self):
        import torch
        from transformer_ppo.ppo import PPOTrainer

        config = PPOConfig(
            rollout_steps=2, gae_horizon=1, hidden_dim=16,
            attention_heads=4, transformer_layers=1, update_epochs=3,
            actor_learning_rate=1e-2, critic_learning_rate=1e-5,
            entropy_coefficient=0.0, dropout=0.0,
            actor_policy="base_residual")
        observation = {
            "parent": [[0.0] * len(PARENT_FEATURES)],
            "candidates": [
                [0.0] * len(CANDIDATE_FEATURES),
                [1.0] + [0.0] * (len(CANDIDATE_FEATURES) - 1),
            ],
            "eligible": [True, True], "action_prior": [0.5, 0.5],
            "operator_id": 0,
            "global": [0.0] * len(GLOBAL_FEATURES),
        }
        trainer = PPOTrainer(
            "mlp", config, lambda *_: observation,
            device="cpu", policy_seed=123, training=True)
        candidates = [{"eligible": True}, {"eligible": True}]
        with patch("torch.multinomial", side_effect=[
                torch.tensor([[0]]), torch.tensor([[1]])]):
            trainer.choose(candidates)
            trainer.observe_outcome(1.0, done=True)
            trainer.choose(candidates)
            trainer.observe_outcome(-1.0, done=True)
        trainer.finish()
        trainer.training = False
        decision = trainer.choose(candidates)
        self.assertGreater(decision.probabilities[0], 0.5)
        record = trainer.update_records[-1]
        self.assertGreater(
            record["post_update_total_variation_mean"], 0.0)
        self.assertIn("actor_head_gradient_norm", record)
        self.assertIn("critic_explained_variance", record)

    def test_terminal_reward_reaches_last_learnable_action_before_forced_action(self):
        from transformer_ppo.ppo import PPOTrainer

        config = PPOConfig(
            hidden_dim=16, attention_heads=4, transformer_layers=1,
            dropout=0.0)
        observation = {
            "parent": [[0.0] * len(PARENT_FEATURES)],
            "candidates": [[0.0] * len(CANDIDATE_FEATURES)] * 2,
            "eligible": [True, True], "operator_id": 0,
            "global": [0.0] * len(GLOBAL_FEATURES),
        }
        trainer = PPOTrainer(
            "mlp", config, lambda *_: observation,
            device="cpu", policy_seed=123, training=True)
        trainer.begin_sequence()
        trainer.choose([{"eligible": True}, {"eligible": True}])
        trainer.observe_outcome(0.0, done=False)
        # The next sub-action is forced, so it creates no pending transition.
        trainer.end_sequence(2.5, done=True)
        self.assertEqual(len(trainer.buffer), 1)
        self.assertAlmostEqual(trainer.buffer[0].reward, 2.5)
        self.assertTrue(trainer.buffer[0].done)

    def test_base_residual_actor_starts_at_declared_prior(self):
        from transformer_ppo.ppo import PPOTrainer

        config = PPOConfig(
            hidden_dim=16, attention_heads=4, transformer_layers=1,
            dropout=0.0, actor_policy="base_residual")
        observation = {
            "parent": [[0.0] * len(PARENT_FEATURES)],
            "candidates": [[0.0] * len(CANDIDATE_FEATURES)] * 3,
            "eligible": [True, True, False],
            "action_prior": [0.25, 0.75, 0.0],
            "operator_id": 0,
            "global": [0.0] * len(GLOBAL_FEATURES),
        }
        for architecture in ("mlp", "transformer"):
            with self.subTest(architecture=architecture):
                trainer = PPOTrainer(
                    architecture, config, lambda *_: observation,
                    device="cpu", policy_seed=123, training=False)
                decision = trainer.choose([
                    {"eligible": True}, {"eligible": True},
                    {"eligible": False},
                ])
                self.assertEqual(decision.chosen_index, 1)
                self.assertAlmostEqual(
                    decision.probabilities[0], 0.25, places=6)
                self.assertAlmostEqual(
                    decision.probabilities[1], 0.75, places=6)
                self.assertEqual(decision.probabilities[2], 0.0)
                self.assertEqual(
                    decision.metadata["actor_policy"], "base_residual")

    def test_forced_single_candidate_creates_no_ppo_transition(self):
        from transformer_ppo.ppo import PPOTrainer

        config = PPOConfig(hidden_dim=16, attention_heads=4,
                           transformer_layers=1)
        trainer = PPOTrainer(
            "mlp", config,
            lambda *_: self.fail("forced action must not build an observation"),
            device="cpu", policy_seed=123, training=True)
        decision = trainer.choose([
            {"eligible": False},
            {"eligible": True},
        ])
        self.assertEqual(decision.chosen_index, 1)
        self.assertEqual(decision.metadata["decision_kind"], "single_candidate")
        self.assertIsNone(trainer.pending)
        self.assertEqual(trainer.buffer, [])

    def test_forced_candidate_index_executes_requested_eligible_action(self):
        from transformer_ppo.ppo import PPOTrainer

        config = PPOConfig(
            hidden_dim=16, attention_heads=4, transformer_layers=1,
            dropout=0.0)
        observation = {
            "parent": [[0.0] * len(PARENT_FEATURES)],
            "candidates": [[0.0] * len(CANDIDATE_FEATURES)] * 3,
            "eligible": [True, False, True], "operator_id": 4,
            "global": [0.0] * len(GLOBAL_FEATURES),
        }
        trainer = PPOTrainer(
            "mlp", config, lambda *_: observation,
            device="cpu", policy_seed=123, training=False,
            inference_action_mode="sample")
        decision = trainer.choose(
            [{"eligible": True}, {"eligible": False}, {"eligible": True}],
            forced_action_index=2)
        self.assertEqual(decision.chosen_index, 2)
        self.assertEqual(decision.metadata["forced_action_index"], 2)
        self.assertEqual(trainer.pending["action"], 2)
        trainer.observe_outcome(0.0, done=True)
        with self.assertRaisesRegex(ValueError, "not eligible"):
            trainer.choose(
                [{"eligible": True}, {"eligible": False}],
                forced_action_index=1)

    def test_model_choice_reports_measured_inference_time(self):
        from transformer_ppo.ppo import PPOTrainer

        config = PPOConfig(hidden_dim=16, attention_heads=4,
                           transformer_layers=1)
        observation = {
            "parent": [[0.0] * len(PARENT_FEATURES)],
            "candidates": [[0.0] * len(CANDIDATE_FEATURES)] * 2,
            "eligible": [True, True], "operator_id": 0,
            "global": [0.0] * len(GLOBAL_FEATURES),
        }
        trainer = PPOTrainer(
            "mlp", config, lambda *_: observation,
            device="cpu", policy_seed=123, training=False)
        decision = trainer.choose([{"eligible": True}, {"eligible": True}])
        self.assertEqual(decision.metadata["decision_kind"], "model_choice")
        self.assertGreater(decision.metadata["inference_seconds"], 0.0)
        self.assertGreater(trainer.feature_construction_seconds, 0.0)

    def test_frozen_sample_mode_does_not_enable_training(self):
        import torch
        from transformer_ppo.ppo import PPOTrainer

        config = PPOConfig(
            hidden_dim=16, attention_heads=4, transformer_layers=1,
            dropout=0.0)
        observation = {
            "parent": [[0.0] * len(PARENT_FEATURES)],
            "candidates": [[0.0] * len(CANDIDATE_FEATURES)] * 2,
            "eligible": [True, True], "operator_id": 0,
            "global": [0.0] * len(GLOBAL_FEATURES),
        }
        trainer = PPOTrainer(
            "mlp", config, lambda *_: observation,
            device="cpu", policy_seed=123, training=False,
            inference_action_mode="sample")
        with torch.no_grad():
            for parameter in trainer.model.parameters():
                parameter.zero_()
        with patch("torch.multinomial", return_value=torch.tensor([[1]])):
            decision = trainer.choose([
                {"eligible": True}, {"eligible": True},
            ])
        trainer.observe_outcome(1.0)
        self.assertEqual(decision.chosen_index, 1)
        self.assertFalse(trainer.training)
        self.assertEqual(trainer.buffer, [])
        self.assertEqual(trainer.updates, 0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            trainer.save(checkpoint)
            restored = PPOTrainer.load(
                checkpoint, PPOConfig, lambda *_: observation,
                device="cpu", training=False,
                inference_action_mode="sample", inference_policy_seed=456)
            self.assertFalse(restored.training)
            self.assertEqual(restored.inference_action_mode, "sample")
            self.assertEqual(restored.policy_seed, 456)

    def test_invalid_inference_action_mode_fails(self):
        from transformer_ppo.ppo import PPOTrainer

        config = PPOConfig(hidden_dim=16, attention_heads=4,
                           transformer_layers=1)
        with self.assertRaisesRegex(ValueError, "inference_action_mode"):
            PPOTrainer(
                "mlp", config, lambda *_: None,
                device="cpu", training=False,
                inference_action_mode="invalid")

    def test_transformer_actor_receives_global_state_before_candidate_encoder(self):
        import torch
        from transformer_ppo.models import TransformerActorCritic
        from transformer_ppo.observations import pad_observations

        torch.manual_seed(7)
        model = TransformerActorCritic(
            hidden_dim=16, heads=4, layers=1, dropout=0.0)
        model.eval()
        observations = []
        for progress in (0.0, 0.75):
            global_features = [0.0] * len(GLOBAL_FEATURES)
            global_features[GLOBAL_FEATURES.index("generation_fraction")] = progress
            observations.append({
                "parent": [[0.0] * len(PARENT_FEATURES)],
                "candidates": [
                    [float(index == feature) for feature in range(
                        len(CANDIDATE_FEATURES))]
                    for index in range(3)
                ],
                "eligible": [True, True, True],
                "operator_id": 0,
                "global": global_features,
            })
        captured = []
        handle = model.candidate_encoder.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach().clone()))
        try:
            logits, _ = model(pad_observations(observations, torch.device("cpu")))
        finally:
            handle.remove()
        self.assertEqual(len(captured), 1)
        self.assertFalse(torch.allclose(captured[0][0], captured[0][1]))
        centred = logits - logits.mean(dim=1, keepdim=True)
        self.assertFalse(torch.allclose(centred[0], centred[1]))

    def test_update_records_transition_count_and_diagnostics(self):
        from transformer_ppo.ppo import PPOTrainer

        config = PPOConfig(
            rollout_steps=2, gae_horizon=1, hidden_dim=16,
            attention_heads=4, transformer_layers=1, update_epochs=1,
            dropout=0.0)
        observation = {
            "parent": [[0.0] * len(PARENT_FEATURES)],
            "candidates": [[0.0] * len(CANDIDATE_FEATURES)] * 2,
            "eligible": [True, True], "operator_id": 0,
            "global": [0.0] * len(GLOBAL_FEATURES),
        }
        trainer = PPOTrainer(
            "mlp", config, lambda *_: observation,
            device="cpu", policy_seed=123, training=True)
        for reward in (0.0, 1.0):
            trainer.choose([{"eligible": True}, {"eligible": True}])
            trainer.observe_outcome(reward)
        trainer.finish()
        self.assertEqual(trainer.updates, 1)
        self.assertEqual(len(trainer.update_records), 1)
        record = trainer.update_records[0]
        self.assertEqual(record["transition_count"], 2)
        self.assertEqual(record["positive_reward_transition_count"], 1)
        self.assertEqual(record["nonzero_reward_transition_count"], 1)
        self.assertEqual(record["zero_reward_transition_count"], 1)
        self.assertAlmostEqual(record["reward_total"], 1.0)
        self.assertAlmostEqual(record["reward_mean"], 0.5)
        self.assertEqual(record["minibatch_count"], 1)
        self.assertEqual(record["optimizer_step_count"], 1)
        self.assertEqual(record["update_epochs_started"], 1)
        self.assertFalse(record["early_stopped_on_target_kl"])
        self.assertLess(
            record["first_minibatch_pre_step_logp_abs_error_max"], 1e-6)
        self.assertLess(
            record["first_minibatch_pre_step_ratio_abs_error_max"], 1e-6)
        self.assertGreater(record["update_seconds"], 0.0)
        self.assertIn("policy_loss", record)
        self.assertIn("value_loss", record)

    def test_training_checkpoint_restores_all_owned_random_streams(self):
        import torch
        from transformer_ppo.ppo import PPOTrainer

        config = PPOConfig(hidden_dim=16, attention_heads=4,
                           transformer_layers=1)
        trainer = PPOTrainer("mlp", config, lambda *_: None,
                             device="cpu", policy_seed=123, training=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            trainer.save(path)
            expected_torch = torch.rand(4)
            expected_action = torch.rand(4, generator=trainer.action_generator)
            expected_minibatch = trainer.minibatch_rng.random(4)

            restored = PPOTrainer.load(
                path, PPOConfig, lambda *_: None, device="cpu", training=True)
            self.assertEqual(restored.checkpoint_provenance["checkpoint_sha256"],
                             __import__("hashlib").sha256(path.read_bytes()).hexdigest())
            self.assertEqual(restored.checkpoint_provenance["architecture"], "mlp")
            self.assertTrue(torch.equal(torch.rand(4), expected_torch))
            self.assertTrue(torch.equal(
                torch.rand(4, generator=restored.action_generator), expected_action))
            np.testing.assert_array_equal(restored.minibatch_rng.random(4),
                                          expected_minibatch)

    def test_legacy_transformer_checkpoint_is_rejected_after_actor_change(self):
        import torch
        from transformer_ppo.ppo import PPOTrainer

        config = PPOConfig(
            hidden_dim=16, attention_heads=4, transformer_layers=1,
            dropout=0.0)
        trainer = PPOTrainer(
            "transformer", config, lambda *_: None,
            device="cpu", policy_seed=123, training=True)
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.pt"
            legacy = Path(directory) / "legacy.pt"
            trainer.save(current)
            restored = PPOTrainer.load(
                current, PPOConfig, lambda *_: None,
                device="cpu", training=False)
            self.assertEqual(
                restored.checkpoint_provenance["actor_conditioning"],
                "candidate-plus-operator-plus-global-before-encoder")
            try:
                artifact = torch.load(
                    current, map_location="cpu", weights_only=False)
            except TypeError:
                artifact = torch.load(current, map_location="cpu")
            artifact["schema_version"] = 3
            artifact.pop("actor_conditioning")
            torch.save(artifact, legacy)
            with self.assertRaisesRegex(ValueError, "retrain"):
                PPOTrainer.load(
                    legacy, PPOConfig, lambda *_: None,
                    device="cpu", training=False)


class FormalBatchTests(unittest.TestCase):
    def test_empty_development_manifest_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no training manifests"):
                _ordered_manifests(Path(directory), 1, 1, "S0")

    def test_development_manifest_limit_cannot_exceed_available_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train-S0-001.json").write_text(
                '{"split":"train","configuration":"S0"}',
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only 1 matching"):
                _ordered_manifests(root, 1, 2, "S0")

    def test_explicit_training_subset_preserves_declared_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for instance_id in ("train-S0-001", "train-S0-002"):
                (root / f"{instance_id}.json").write_text(
                    '{"split":"train","configuration":"S0"}',
                    encoding="utf-8")
            manifests, _ = _ordered_manifests(
                root, 1, None, "S0",
                ["train-S0-002", "train-S0-001"])
            self.assertEqual(
                [path.stem for path in manifests],
                ["train-S0-002", "train-S0-001"])

    def test_explicit_training_subset_rejects_missing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train-S0-001.json").write_text(
                '{"split":"train","configuration":"S0"}',
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unavailable"):
                _ordered_manifests(
                    root, 1, None, "S0", ["train-S0-999"])

    def test_replication_seed_namespaces_are_paired_and_disjoint(self):
        args = SimpleNamespace(
            replication=1, algorithm_seed_base=80_000_000,
            policy_seed_base=90_000_000)
        self.assertEqual(_seeds(args, 0), (80_000_000, 90_000_000))
        self.assertEqual(_seeds(args, 119), (80_000_119, 90_000_119))
        args.replication = 2
        self.assertEqual(_seeds(args, 0), (81_000_000, 91_000_000))


class StatisticsTests(unittest.TestCase):
    def test_holm_is_monotone_in_sorted_p_values(self):
        adjusted = holm([.03, .001, .02])
        self.assertEqual(adjusted[1], .003)
        self.assertGreaterEqual(adjusted[0], adjusted[2])

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy not installed")
    def test_formal_report_keeps_configurations_separate(self):
        rows = []
        for configuration in TEST_CONFIGS:
            for index in range(10):
                instance = f"test-{configuration}-{index:03d}"
                for method, value in (
                        ("transformer", .8), ("mlp", .7),
                        ("rule", .6), ("random", .5)):
                    rows.append({"configuration": configuration,
                                 "instance_id": instance,
                                 "method": method, "oos_hv": value})
        report = formal_report(rows)
        self.assertEqual(len(report["families"]["primary_transformer_vs_rule"]), 12)
        self.assertEqual(
            report["families"]["primary_transformer_vs_rule"][0]["wins"], 10)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy not installed")
    def test_duplicate_method_instance_is_rejected(self):
        rows = []
        for configuration in TEST_CONFIGS:
            for index in range(10):
                instance = f"test-{configuration}-{index:03d}"
                for method in ("transformer", "mlp", "rule", "random"):
                    rows.append({"configuration": configuration,
                                 "instance_id": instance,
                                 "method": method, "oos_hv": .5})
        rows.append(dict(rows[0]))
        with self.assertRaises(ValueError):
            formal_report(rows)


if __name__ == "__main__":
    unittest.main()
