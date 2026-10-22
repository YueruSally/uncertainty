from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline_uncertainty as base
from transformer_ppo.archive import ObjectiveNormalizer
from transformer_ppo.crossover import (
    apply_cut, finalize_children, prepare_common_node_crossover,
)
from transformer_ppo.crossover_observations import (
    CROSSOVER_MODEL_SPEC, build_crossover_observation,
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
        self.observation_provider = None

    def begin_sequence(self):
        self.sequence_depth = 1
        self.sequence_policy_actions = 0

    def choose(self, candidates, context=None):
        self.observation_provider(candidates, context or {})
        self.pending = True
        self.sequence_policy_actions += 1
        probabilities = [1.0] + [0.0] * (len(candidates) - 1)
        return Decision(0, probabilities, [None] * len(candidates), {
            "decision_kind": "model_choice", "inference_seconds": 0.001,
        })

    def observe_outcome(self, reward, done=False):
        if self.pending:
            self.rewards.append(float(reward))
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
