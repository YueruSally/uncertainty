import unittest

from train_m_downstream_transformer import subset_observation


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


if __name__ == "__main__":
    unittest.main()
