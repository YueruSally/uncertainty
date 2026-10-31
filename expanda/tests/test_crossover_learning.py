from pathlib import Path
import importlib.util
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline_uncertainty as base
from transformer_ppo.archive import ObjectiveNormalizer
from transformer_ppo.crossover import (
    apply_cut, finalize_children, prepare_common_node_crossover,
)
from transformer_ppo.crossover_observations import (
    CROSSOVER_MODEL_SPEC, build_crossover_observation,
    CROSSOVER_GRAPH_MODEL_SPEC_V3, build_crossover_shared_observation,
)
from multi_instance_policy import Decision


class _FixedRng:
    def randrange(self, _size):
        return 0


class _FakeTrainer:
    name = "fake-ppo-c-masked"
    component = "C"
    updates = 0

    def __init__(self):
        self.pending = None
        self.sequence_depth = 0
        self.sequence_policy_actions = 0
        self.rewards = []
        self.done = []
        self.observation_provider = None
        self.chosen_index = 0

    def begin_sequence(self):
        self.sequence_depth = 1
        self.sequence_policy_actions = 0

    def choose(self, candidates, context=None):
        self.observation_provider(candidates, context or {})
        self.pending = True
        self.sequence_policy_actions += 1
        probabilities = [0.0] * len(candidates)
        probabilities[self.chosen_index] = 1.0
        return Decision(self.chosen_index, probabilities, [None] * len(candidates), {
            "decision_kind": "model_choice", "inference_seconds": 0.001,
        })

    def observe_outcome(self, reward, done=False):
        if self.pending:
            self.rewards.append(float(reward))
            self.done.append(bool(done))
        self.pending = None

    def end_sequence(self, reward, done=False):
        self.observe_outcome(reward, done=done)
        self.sequence_depth = 0
        self.sequence_policy_actions = 0

    def finish(self):
        if self.pending or self.sequence_depth or self.sequence_policy_actions:
            raise RuntimeError("unfinished fake policy sequence")


def _path(path_id, nodes):
    arcs = [base.Arc(a, b, "road", 1.0, 100.0, 1.0, 1.0, 1.0)
            for a, b in zip(nodes[:-1], nodes[1:])]
    return base.Path(
        path_id, nodes[0], nodes[-1], list(nodes), ["road"] * len(arcs), arcs,
        float(len(arcs)), float(len(arcs)), float(len(arcs)))


class CrossoverPlanningTests(unittest.TestCase):
    def setUp(self):
        self.batch = base.Batch(7, "A", "D", 10.0, 0.0, 24.0)
        self.path1 = _path(1, ["A", "X", "J", "D"])
        self.path2 = _path(2, ["A", "X", "K", "J", "D"])
        key = ("A", "D", 7)
        self.parent1 = base.Individual({key: [base.PathAllocation(self.path1, 1.0)]})
        self.parent2 = base.Individual({key: [base.PathAllocation(self.path2, 1.0)]})

    def test_legal_cuts_are_stable_and_follow_parent_chromosome_order(self):
        _, _, decisions = prepare_common_node_crossover(
            self.parent1, self.parent2, [self.batch], _FixedRng())
        self.assertEqual(len(decisions), 1)
        self.assertEqual(
            [row["join_node"] for row in decisions[0].candidates], ["X", "J"])

    def test_selected_cut_preserves_base_structure_and_shares(self):
        child1, child2, decisions = prepare_common_node_crossover(
            self.parent1, self.parent2, [self.batch], _FixedRng())
        all_arcs = {**{(a.from_node, a.to_node, a.mode): a for a in self.path1.arcs},
                    **{(a.from_node, a.to_node, a.mode): a for a in self.path2.arcs}}
        changed = apply_cut(
            child1, child2, decisions[0], decisions[0].candidates[0], {}, all_arcs)
        finalize_children(child1, child2)
        key = ("A", "D", 7)
        self.assertTrue(changed)
        self.assertEqual(child1.od_allocations[key][0].path.nodes, self.path2.nodes)
        self.assertEqual(child2.od_allocations[key][0].path.nodes, self.path1.nodes)
        self.assertAlmostEqual(child1.od_allocations[key][0].share, 1.0)
        self.assertAlmostEqual(child2.od_allocations[key][0].share, 1.0)

    def test_crossover_observation_uses_declared_raw_dimensions(self):
        _, _, decisions = prepare_common_node_crossover(
            self.parent1, self.parent2, [self.batch], _FixedRng())
        for parent in (self.parent1, self.parent2):
            parent.objectives = (1.0, 2.0, 3.0)
            parent.feasible = True
            parent.normalized_violation = 0.0
        observation = build_crossover_observation(
            self.parent1, self.parent2, [self.batch], decisions[0].candidates,
            {"K": 1, "deadline_window_alpha": 1.0, "generation": 0},
            [self.parent1, self.parent2], [], {"archive_size": 0, "archive_hv": 0.0},
            ObjectiveNormalizer((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
            200, 20, 10)
        self.assertEqual(len(observation["parent"][0]), CROSSOVER_MODEL_SPEC["parent_dim"])
        self.assertEqual(
            len(observation["candidates"][0]), CROSSOVER_MODEL_SPEC["candidate_dim"])
        self.assertEqual(len(observation["global"]), CROSSOVER_MODEL_SPEC["global_dim"])
        self.assertEqual(observation["operator_id"], 0)

    def test_cached_shared_observation_is_exactly_equivalent(self):
        _, _, decisions = prepare_common_node_crossover(
            self.parent1, self.parent2, [self.batch], _FixedRng())
        for parent in (self.parent1, self.parent2):
            parent.objectives = (1.0, 2.0, 3.0)
            parent.feasible = True
            parent.normalized_violation = 0.0
        context = {
            "K": 1, "deadline_window_alpha": 1.0, "generation": 0,
        }
        population = [self.parent1, self.parent2]
        archive = {"archive_size": 0, "archive_hv": 0.0}
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
        arguments = (
            self.parent1, self.parent2, [self.batch],
            decisions[0].candidates, context, population, [], archive,
            normalizer, 200, 20, 10,
        )
        rebuilt = build_crossover_observation(*arguments)
        shared = build_crossover_shared_observation(
            self.parent1, self.parent2, [self.batch], context, population, [],
            archive, normalizer, 200, 20, 10)
        cached = build_crossover_observation(
            *arguments, shared_observation=shared)
        self.assertEqual(cached, rebuilt)

    def test_controller_settles_final_children_jointly_without_extra_evaluation(self):
        from transformer_ppo.archive import RewardArchive, RewardConfig
        from transformer_ppo.crossover_controller import CrossoverPPOController

        trainer = _FakeTrainer()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0), reference=(1.1, 1.1, 1.1))
        with tempfile.TemporaryDirectory() as directory:
            controller = CrossoverPPOController(
                Path(directory), "run", "scenario", 20, trainer=trainer,
                archive=RewardArchive(normalizer), reward_config=RewardConfig(),
                batches=[self.batch], generations=2, pop_size=2,
                method=trainer.name,
                instance={"instance_id": "toy", "K": 1,
                          "deadline_window_alpha": 1.0}, seeds={})
            try:
                lookup = {
                    (arc.from_node, arc.to_node, arc.mode): arc
                    for path in (self.path1, self.path2) for arc in path.arcs
                }
                child1, child2 = controller.crossover_common_node(
                    self.parent1, self.parent2, [self.batch], {}, lookup)
                child1.objectives, child1.feasible = (2.0, 3.0, 4.0), True
                child2.objectives, child2.feasible = (3.0, 2.0, 4.0), True
                controller.settle_crossover(child1, child2)
                self.assertEqual(len(trainer.rewards), 1)
                self.assertGreater(trainer.rewards[0], 0.0)
                self.assertEqual(len(controller.archive.points), 2)
                self.assertEqual(controller.evaluations, 0)
                self.assertFalse(controller._pending_child_ids)
            finally:
                controller.close()

    def test_operation_terminal_credit_closes_each_crossover_trajectory(self):
        from transformer_ppo.archive import RewardArchive, RewardConfig
        from transformer_ppo.crossover_controller import CrossoverPPOController

        trainer = _FakeTrainer()
        trainer.config = SimpleNamespace(
            credit_assignment="operation_terminal",
            candidate_representation="legacy")
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        with tempfile.TemporaryDirectory() as directory:
            controller = CrossoverPPOController(
                Path(directory), "run", "scenario", 20, trainer=trainer,
                archive=RewardArchive(normalizer), reward_config=RewardConfig(),
                batches=[self.batch], generations=2, pop_size=2,
                method=trainer.name,
                instance={"instance_id": "toy", "K": 1,
                          "deadline_window_alpha": 1.0}, seeds={})
            try:
                lookup = {
                    (arc.from_node, arc.to_node, arc.mode): arc
                    for path in (self.path1, self.path2) for arc in path.arcs
                }
                child1, child2 = controller.crossover_common_node(
                    self.parent1, self.parent2, [self.batch], {}, lookup)
                child1.objectives, child1.feasible = (2.0, 3.0, 4.0), True
                child2.objectives, child2.feasible = (3.0, 2.0, 4.0), True
                controller.settle_crossover(child1, child2)
                self.assertTrue(trainer.done[-1])
                self.assertEqual(
                    controller.summary()["credit_assignment"],
                    "operation_terminal")
            finally:
                controller.close()

    def test_learned_cut_consumes_the_paired_base_random_draw(self):
        from transformer_ppo.archive import RewardArchive, RewardConfig
        from transformer_ppo.crossover_controller import CrossoverPPOController

        random.seed(1234)
        base.crossover_common_node(
            self.parent1, self.parent2, [self.batch], {}, {
                (arc.from_node, arc.to_node, arc.mode): arc
                for path in (self.path1, self.path2) for arc in path.arcs
            })
        expected_state = random.getstate()

        trainer = _FakeTrainer()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0), reference=(1.1, 1.1, 1.1))
        with tempfile.TemporaryDirectory() as directory:
            controller = CrossoverPPOController(
                Path(directory), "run", "scenario", 20, trainer=trainer,
                archive=RewardArchive(normalizer), reward_config=RewardConfig(),
                batches=[self.batch], generations=2, pop_size=2,
                method=trainer.name,
                instance={"instance_id": "toy", "K": 1,
                          "deadline_window_alpha": 1.0}, seeds={})
            try:
                random.seed(1234)
                controller.crossover_common_node(
                    self.parent1, self.parent2, [self.batch], {}, {
                        (arc.from_node, arc.to_node, arc.mode): arc
                        for path in (self.path1, self.path2) for arc in path.arcs
                    })
                self.assertEqual(random.getstate(), expected_state)
            finally:
                controller.close()

    def test_v2_zero_model_probability_executes_paired_base_cut(self):
        from transformer_ppo.archive import RewardArchive, RewardConfig
        from transformer_ppo.crossover_controller import CrossoverPPOController

        trainer = _FakeTrainer()
        trainer.choose = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("zero model probability must not call PPO"))
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        lookup = {
            (arc.from_node, arc.to_node, arc.mode): arc
            for path in (self.path1, self.path2) for arc in path.arcs
        }
        random.seed(1234)
        base.crossover_common_node(
            self.parent1, self.parent2, [self.batch], {}, lookup)
        expected_state = random.getstate()

        with tempfile.TemporaryDirectory() as directory:
            controller = CrossoverPPOController(
                Path(directory), "run", "scenario", 20, trainer=trainer,
                archive=RewardArchive(normalizer), reward_config=RewardConfig(),
                batches=[self.batch], generations=2, pop_size=2,
                method=trainer.name,
                instance={"instance_id": "toy", "K": 1,
                          "deadline_window_alpha": 1.0},
                seeds={"algorithm": 1234}, model_choice_probability=0.0)
            try:
                random.seed(1234)
                controller.crossover_common_node(
                    self.parent1, self.parent2, [self.batch], {}, lookup)
                self.assertEqual(random.getstate(), expected_state)
                selected = controller._active_event["selected_cuts"][0]
                self.assertEqual(
                    selected["chosen_index"], selected["shadow_base_index"])
                self.assertEqual(
                    selected["decision_kind"], "base_mixture_choice")
                self.assertEqual(controller._active_event["model_decisions"], 0)
            finally:
                controller.close()

    def test_multibatch_zero_model_probability_replays_base_children_and_rng(self):
        from multi_instance_logging import fingerprint
        from transformer_ppo.archive import RewardArchive, RewardConfig
        from transformer_ppo.crossover_controller import CrossoverPPOController

        second_batch = base.Batch(8, "A", "D", 8.0, 0.0, 24.0)
        key1, key2 = ("A", "D", 7), ("A", "D", 8)
        parent1 = base.Individual({
            key1: [base.PathAllocation(self.path1, 1.0)],
            key2: [base.PathAllocation(self.path1, 1.0)],
        })
        parent2 = base.Individual({
            key1: [base.PathAllocation(self.path2, 1.0)],
            key2: [base.PathAllocation(self.path2, 1.0)],
        })
        batches = [self.batch, second_batch]
        lookup = {
            (arc.from_node, arc.to_node, arc.mode): arc
            for path in (self.path1, self.path2) for arc in path.arcs
        }
        random.seed(8128)
        expected_children = base.crossover_common_node(
            parent1, parent2, batches, {}, lookup)
        expected_fingerprints = tuple(map(fingerprint, expected_children))
        expected_state = random.getstate()

        trainer = _FakeTrainer()
        trainer.choose = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("zero model probability must not call PPO"))
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        with tempfile.TemporaryDirectory() as directory:
            controller = CrossoverPPOController(
                Path(directory), "run", "scenario", 20, trainer=trainer,
                archive=RewardArchive(normalizer), reward_config=RewardConfig(),
                batches=batches, generations=2, pop_size=2,
                method=trainer.name,
                instance={"instance_id": "toy", "K": 1,
                          "deadline_window_alpha": 1.0},
                seeds={"algorithm": 8128}, model_choice_probability=0.0)
            try:
                random.seed(8128)
                actual_children = controller.crossover_common_node(
                    parent1, parent2, batches, {}, lookup)
                self.assertEqual(
                    tuple(map(fingerprint, actual_children)),
                    expected_fingerprints)
                self.assertEqual(random.getstate(), expected_state)
            finally:
                controller.close()

    def test_path_v3_joint_override_is_one_policy_action_with_direct_credit(self):
        from transformer_ppo.archive import RewardArchive, RewardConfig
        from transformer_ppo.crossover_controller import CrossoverPPOController

        second_batch = base.Batch(8, "A", "D", 8.0, 0.0, 24.0)
        key1, key2 = ("A", "D", 7), ("A", "D", 8)
        parent1 = base.Individual({
            key1: [base.PathAllocation(self.path1, 1.0)],
            key2: [base.PathAllocation(self.path1, 1.0)],
        })
        parent2 = base.Individual({
            key1: [base.PathAllocation(self.path2, 1.0)],
            key2: [base.PathAllocation(self.path2, 1.0)],
        })
        trainer = _FakeTrainer()
        trainer.chosen_index = 1
        trainer.config = SimpleNamespace(
            credit_assignment="operation_terminal",
            candidate_representation="path_v3")
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        lookup = {
            (arc.from_node, arc.to_node, arc.mode): arc
            for path in (self.path1, self.path2) for arc in path.arcs
        }
        with tempfile.TemporaryDirectory() as directory:
            controller = CrossoverPPOController(
                Path(directory), "run", "scenario", 20, trainer=trainer,
                archive=RewardArchive(normalizer), reward_config=RewardConfig(),
                batches=[self.batch, second_batch], generations=2, pop_size=2,
                method=trainer.name,
                instance={"instance_id": "toy", "K": 1,
                          "deadline_window_alpha": 1.0}, seeds={})
            try:
                child1, child2 = controller.crossover_common_node(
                    parent1, parent2, [self.batch, second_batch], {}, lookup)
                event = controller._active_event
                self.assertEqual(event["model_decisions"], 1)
                self.assertEqual(
                    event["selected_joint_action"], "single_cut_override")
                self.assertEqual(sum(
                    row["different_from_shadow_base"]
                    for row in event["selected_cuts"]), 1)
                self.assertEqual(trainer.sequence_policy_actions, 1)
                child1.objectives, child1.feasible = (2.0, 3.0, 4.0), True
                child2.objectives, child2.feasible = (3.0, 2.0, 4.0), True
                controller.settle_crossover(child1, child2)
                self.assertEqual(len(trainer.rewards), 1)
                self.assertTrue(trainer.done[-1])
            finally:
                controller.close()

    def test_path_v3_base_passthrough_replays_exact_base_children_and_rng(self):
        from multi_instance_logging import fingerprint
        from transformer_ppo.archive import RewardArchive, RewardConfig
        from transformer_ppo.crossover_controller import CrossoverPPOController

        lookup = {
            (arc.from_node, arc.to_node, arc.mode): arc
            for path in (self.path1, self.path2) for arc in path.arcs
        }
        random.seed(6317)
        expected = base.crossover_common_node(
            self.parent1, self.parent2, [self.batch], {}, lookup)
        expected_fingerprints = tuple(map(fingerprint, expected))
        expected_state = random.getstate()

        trainer = _FakeTrainer()
        trainer.config = SimpleNamespace(
            credit_assignment="operation_terminal",
            candidate_representation="path_v3")
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        with tempfile.TemporaryDirectory() as directory:
            controller = CrossoverPPOController(
                Path(directory), "run", "scenario", 20, trainer=trainer,
                archive=RewardArchive(normalizer), reward_config=RewardConfig(),
                batches=[self.batch], generations=2, pop_size=2,
                method=trainer.name,
                instance={"instance_id": "toy", "K": 1,
                          "deadline_window_alpha": 1.0}, seeds={})
            try:
                random.seed(6317)
                actual = controller.crossover_common_node(
                    self.parent1, self.parent2, [self.batch], {}, lookup)
                self.assertEqual(
                    controller._active_event["selected_joint_action"],
                    "base_passthrough")
                self.assertEqual(
                    tuple(map(fingerprint, actual)), expected_fingerprints)
                self.assertEqual(random.getstate(), expected_state)
            finally:
                controller.close()

    def test_path_v3_joint_candidate_schema_and_prior_are_declared(self):
        from transformer_ppo.crossover_controller import CrossoverPPOController

        parent1, parent2, decisions = prepare_common_node_crossover(
            self.parent1, self.parent2, [self.batch], _FixedRng())
        candidates = CrossoverPPOController._joint_override_candidates(decisions)
        observation = build_crossover_observation(
            parent1, parent2, [self.batch], candidates,
            {"K": 1, "deadline_window_alpha": 1.0, "generation": 0},
            [], [], {"archive_size": 0, "archive_hv": 0.0},
            ObjectiveNormalizer((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
            200, 20, 10, candidate_representation="path_v3")
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["joint_action"], "base_passthrough")
        self.assertEqual(candidates[0]["action_group"], "base")
        self.assertTrue(candidates[0]["safe_action"])
        self.assertEqual(candidates[1]["action_group"], "override")
        self.assertFalse(candidates[1]["safe_action"])
        self.assertAlmostEqual(observation["action_prior"][0], 0.5)
        self.assertAlmostEqual(sum(observation["action_prior"]), 1.0)
        self.assertTrue(all(
            len(row) == CROSSOVER_GRAPH_MODEL_SPEC_V3["candidate_dim"]
            for row in observation["candidates"]))


class CrossoverActionGroupTests(unittest.TestCase):
    def test_grouped_argmax_compares_total_group_mass(self):
        from transformer_ppo.action_groups import grouped_argmax

        candidates = [
            {"action_group": "base", "safe_action": True},
            {"action_group": "override", "safe_action": False},
            {"action_group": "override", "safe_action": False},
            {"action_group": "override", "safe_action": False},
        ]
        action, diagnostics = grouped_argmax(
            candidates, [0.40, 0.25, 0.20, 0.15])
        self.assertEqual(action, 1)
        self.assertEqual(diagnostics["selected_group"], "override")
        self.assertAlmostEqual(diagnostics["masses"]["base"], 0.40)
        self.assertAlmostEqual(diagnostics["masses"]["override"], 0.60)

    def test_grouped_argmax_margin_falls_back_to_safe_group(self):
        from transformer_ppo.action_groups import grouped_argmax

        candidates = [
            {"action_group": "base", "safe_action": True},
            {"action_group": "override", "safe_action": False},
            {"action_group": "override", "safe_action": False},
        ]
        action, diagnostics = grouped_argmax(
            candidates, [0.40, 0.35, 0.25], margin=0.25)
        self.assertEqual(action, 0)
        self.assertEqual(diagnostics["selected_group"], "base")

    def test_grouped_argmax_tie_falls_back_to_safe_group(self):
        from transformer_ppo.action_groups import grouped_argmax

        candidates = [
            {"action_group": "base", "safe_action": True},
            {"action_group": "override", "safe_action": False},
            {"action_group": "override", "safe_action": False},
        ]
        action, diagnostics = grouped_argmax(
            candidates, [0.50, 0.30, 0.20])
        self.assertEqual(action, 0)
        self.assertEqual(diagnostics["selected_group"], "base")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
class CrossoverPPOSequenceTests(unittest.TestCase):

    def test_one_terminal_reward_is_not_duplicated_across_subdecisions(self):
        from transformer_ppo.config import PPOConfig
        from transformer_ppo.ppo import PPOTrainer

        observation = {
            "parent": [[0.0] * CROSSOVER_MODEL_SPEC["parent_dim"]],
            "candidates": [[0.0] * CROSSOVER_MODEL_SPEC["candidate_dim"]] * 2,
            "eligible": [True, True], "operator_id": 0,
            "global": [0.0] * CROSSOVER_MODEL_SPEC["global_dim"],
        }
        trainer = PPOTrainer(
            "mlp", PPOConfig(hidden_dim=16, attention_heads=4, transformer_layers=1),
            lambda *_: observation, device="cpu", policy_seed=5, training=True,
            component="C", model_spec=CROSSOVER_MODEL_SPEC)
        candidates = [{"eligible": True}, {"eligible": True}]
        trainer.begin_sequence()
        trainer.choose(candidates)
        trainer.observe_outcome(0.0)
        trainer.choose(candidates)
        trainer.end_sequence(3.5)
        self.assertEqual([item.reward for item in trainer.buffer], [0.0, 3.5])
        self.assertEqual(trainer.sequence_depth, 0)
        self.assertIsNone(trainer.pending)

    def test_rollout_update_occurs_at_next_sequence_boundary(self):
        from transformer_ppo.config import PPOConfig
        from transformer_ppo.ppo import PPOTrainer

        observation = {
            "parent": [[0.0] * CROSSOVER_MODEL_SPEC["parent_dim"]],
            "candidates": [[0.0] * CROSSOVER_MODEL_SPEC["candidate_dim"]] * 2,
            "eligible": [True, True], "operator_id": 0,
            "global": [0.0] * CROSSOVER_MODEL_SPEC["global_dim"],
        }
        trainer = PPOTrainer(
            "mlp", PPOConfig(rollout_steps=2, gae_horizon=2, hidden_dim=16,
                             attention_heads=4, transformer_layers=1),
            lambda *_: observation, device="cpu", policy_seed=6, training=True,
            component="C", model_spec=CROSSOVER_MODEL_SPEC)
        candidates = [{"eligible": True}, {"eligible": True}]
        trainer.begin_sequence()
        trainer.choose(candidates)
        trainer.observe_outcome(0.0)
        trainer.choose(candidates)
        trainer.end_sequence(1.0)
        self.assertEqual(trainer.updates, 0)
        trainer.begin_sequence()
        trainer.choose(candidates)
        self.assertEqual(trainer.updates, 1)
        self.assertEqual(len(trainer.buffer), 0)
        trainer.end_sequence(0.0)

    def test_controller_settles_two_final_children_as_one_archive_update(self):
        from transformer_ppo.archive import RewardArchive, RewardConfig
        from transformer_ppo.config import PPOConfig
        from transformer_ppo.crossover_controller import CrossoverPPOController
        from transformer_ppo.ppo import PPOTrainer

        planning = CrossoverPlanningTests()
        planning.setUp()
        trainer = PPOTrainer(
            "mlp", PPOConfig(hidden_dim=16, attention_heads=4, transformer_layers=1),
            lambda *_: None, device="cpu", policy_seed=7, training=True,
            component="C", model_spec=CROSSOVER_MODEL_SPEC)
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0), reference=(1.1, 1.1, 1.1))
        with tempfile.TemporaryDirectory() as directory:
            controller = CrossoverPPOController(
                Path(directory), "run", "scenario", 20, trainer=trainer,
                archive=RewardArchive(normalizer), reward_config=RewardConfig(),
                batches=[planning.batch], generations=2, pop_size=2,
                method=trainer.name,
                instance={"instance_id": "toy", "K": 1,
                          "deadline_window_alpha": 1.0}, seeds={})
            try:
                lookup = {
                    (arc.from_node, arc.to_node, arc.mode): arc
                    for path in (planning.path1, planning.path2) for arc in path.arcs
                }
                child1, child2 = controller.crossover_common_node(
                    planning.parent1, planning.parent2, [planning.batch], {}, lookup)
                child1.objectives, child1.feasible = (2.0, 3.0, 4.0), True
                child2.objectives, child2.feasible = (3.0, 2.0, 4.0), True
                controller.settle_crossover(child1, child2)
                self.assertEqual(len(trainer.buffer), 1)
                self.assertGreater(trainer.buffer[0].reward, 0.0)
                self.assertEqual(len(controller.archive.points), 2)
                self.assertEqual(controller.positive_reward_actions, 1)
                self.assertFalse(controller._pending_child_ids)
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
