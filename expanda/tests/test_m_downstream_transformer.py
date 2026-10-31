import unittest

from train_m_downstream_transformer import (
    _rank_loss,
    _safe_anchor_loss,
    _safe_quantile_loss,
    subset_observation,
)


class MutationDownstreamTransformerTests(unittest.TestCase):
    def test_subset_keeps_state_graph_and_aligned_candidate_rows(self):
        observation = {
            "parent": [[1.0]],
            "global": [2.0],
            "graph": {"nodes": [[3.0]], "edges": [], "edge_index": []},
            "candidates": [[10.0], [20.0], [30.0]],
            "eligible": [True, False, True],
            "action_prior": [0.4, 0.0, 0.6],
            "candidate_graph_nodes": [[0], [], [0]],
            "candidate_graph_tokens": [
                [[0, 0, 0, 0]], [], [[0, 0, 0, 1]],
            ],
        }
        result = subset_observation(observation, [2, 0])
        self.assertEqual(result["candidates"], [[30.0], [10.0]])
        self.assertEqual(result["eligible"], [True, True])
        self.assertEqual(result["action_prior"], [0.6, 0.4])
        self.assertEqual(result["candidate_graph_nodes"], [[0], [0]])
        self.assertEqual(result["graph"], observation["graph"])
        self.assertEqual(result["parent"], observation["parent"])

    def test_subset_rejects_ineligible_audited_candidate(self):
        observation = {
            "candidates": [[10.0], [20.0]],
            "eligible": [True, False],
            "action_prior": [1.0, 0.0],
        }
        with self.assertRaisesRegex(ValueError, "not eligible"):
            subset_observation(observation, [1])

    def test_rank_loss_is_finite_with_padding_and_no_informative_pairs(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        logits = torch.tensor([
            [0.1, 0.2, torch.finfo(torch.float32).min],
            [0.3, torch.finfo(torch.float32).min,
             torch.finfo(torch.float32).min],
        ])
        targets = torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
        mask = torch.tensor([
            [True, True, False], [True, False, False],
        ])
        loss = _rank_loss(torch, logits, targets, mask, 2.0)
        self.assertTrue(torch.isfinite(loss))

    def test_safe_base_losses_are_finite_and_penalise_false_positive(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        mask = torch.tensor([[True, True]])
        targets = torch.tensor([[5.0, -5.0]])
        correct = torch.tensor([[1.0, -1.0]])
        wrong = torch.tensor([[-1.0, 1.0]])
        correct_anchor = _safe_anchor_loss(
            torch, correct, targets, mask, 0.1, 3.0)
        wrong_anchor = _safe_anchor_loss(
            torch, wrong, targets, mask, 0.1, 3.0)
        correct_quantile = _safe_quantile_loss(
            torch, correct, targets, mask, 0.25, 3.0, 10.0)
        wrong_quantile = _safe_quantile_loss(
            torch, wrong, targets, mask, 0.25, 3.0, 10.0)
        self.assertTrue(torch.isfinite(correct_anchor))
        self.assertTrue(torch.isfinite(correct_quantile))
        self.assertLess(correct_anchor, wrong_anchor)
        self.assertLess(correct_quantile, wrong_quantile)


if __name__ == "__main__":
    unittest.main()
