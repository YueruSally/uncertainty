from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformer_ppo.graph_features import (
    GRAPH_EDGE_FEATURES, GRAPH_EDGE_FEATURES_V2, GRAPH_MODEL_SPEC,
    GRAPH_MODEL_SPEC_V2, GRAPH_NODE_FEATURES,
    build_graph_observation,
)
from transformer_ppo.observations import (
    CANDIDATE_FEATURES, CANDIDATE_PATH_V2_FEATURES, GLOBAL_FEATURES,
    PARENT_FEATURES, candidate_graph_tokens, candidate_tokens,
    pad_observations,
)


class SharedGraphObservationTests(unittest.TestCase):
    def setUp(self):
        self.nodes = ["D", "A", "B"]
        self.regions = {"A": "CN", "B": "KZ", "D": "WE"}
        self.hold = {"A": 2.0, "B": 1.0, "D": 4.0}
        self.processing = {"A": 1.0, "B": 3.0, "D": 2.0}
        self.transfer = {"A": 1.5, "B": 2.5, "D": 3.5}
        self.arcs = [
            SimpleNamespace(
                from_node="B", to_node="D", mode="rail", distance=20.0,
                capacity=200.0, cost_per_teu_km=2.0,
                emission_per_teu_km=3.0, speed_kmh=40.0,
                is_border_arc=True),
            SimpleNamespace(
                from_node="A", to_node="B", mode="road", distance=10.0,
                capacity=100.0, cost_per_teu_km=1.0,
                emission_per_teu_km=2.0, speed_kmh=80.0,
                is_border_arc=False),
        ]
        self.batches = [SimpleNamespace(
            origin="A", destination="D", quantity=12.0, ET=2.0, LT=30.0)]

    def _build(self, nodes=None, arcs=None):
        return build_graph_observation(
            self.nodes if nodes is None else nodes, self.regions,
            self.hold, self.processing, self.transfer,
            self.arcs if arcs is None else arcs, self.batches)

    def test_dimensions_match_shared_model_spec(self):
        graph = self._build()
        self.assertEqual(graph["schema"], "shared-network-tokens-v1")
        self.assertTrue(all(
            len(row) == GRAPH_MODEL_SPEC["graph_node_dim"] == len(GRAPH_NODE_FEATURES)
            for row in graph["nodes"]))
        self.assertTrue(all(
            len(row) == GRAPH_MODEL_SPEC["graph_edge_dim"] == len(GRAPH_EDGE_FEATURES)
            for row in graph["edges"]))
        self.assertEqual(len(graph["edges"]), len(graph["edge_index"]))

    def test_order_is_canonical_and_not_input_list_order(self):
        first = self._build()
        second = self._build(
            nodes=list(reversed(self.nodes)), arcs=list(reversed(self.arcs)))
        self.assertEqual(first, second)

    def test_continuous_graph_columns_use_bounded_positive_scaling(self):
        graph = self._build()
        for row in graph["nodes"]:
            self.assertTrue(all(-1.0 <= value <= 1.0 for value in row))
        for row in graph["edges"]:
            self.assertTrue(all(0.0 <= value <= 1.0 for value in row))

    def test_v2_graph_adds_timetable_features_and_edge_lookup(self):
        timetables = [SimpleNamespace(
            from_node="B", to_node="D", mode="rail", frequency_per_week=7.0,
            first_departure_hour=4.0, headway_hours=24.0,
            travel_time_h=8.0)]
        graph = build_graph_observation(
            self.nodes, self.regions, self.hold, self.processing, self.transfer,
            self.arcs, self.batches, timetables, "path_v2")
        self.assertEqual(graph["schema"], "ordered-node-edge-paths-v2")
        self.assertTrue(all(
            len(row) == len(GRAPH_EDGE_FEATURES_V2)
            == GRAPH_MODEL_SPEC_V2["graph_edge_dim"] for row in graph["edges"]))
        self.assertIn(("B", "D", "rail"), graph["edge_lookup"])

    def test_ordered_candidate_tokens_alternate_nodes_and_edges(self):
        graph = self._build()
        candidate = {
            "path_nodes": ["A", "B", "D"],
            "path_modes": ["road", "rail"],
        }
        tokens = candidate_graph_tokens([candidate], graph)[0]
        self.assertEqual([row[0] for row in tokens], [0, 1, 0, 1, 0])
        self.assertEqual([row[3] for row in tokens], [0, 0, 1, 1, 2])


class RawMutationCandidateTests(unittest.TestCase):
    def test_decoded_path_objectives_do_not_enter_candidate_token(self):
        common = {
            "target": {"batch_id": 1, "allocation_index": 0, "arc_index": 0},
            "eligible": True, "batch_quantity": 10.0, "share": 1.0,
            "path_count": 1, "path_length": 2,
            "alternative_mode_count": 1, "available_reliable_options": 2,
            "unallocated_path_count": 3, "from_node": "A", "to_node": "B",
            "current_mode": "road",
        }
        first = dict(common, path_cost_per_teu=1.0,
                     path_emission_per_teu=2.0, path_nominal_time_h=3.0,
                     p_rule=0.1)
        second = dict(common, path_cost_per_teu=10_000.0,
                      path_emission_per_teu=20_000.0,
                      path_nominal_time_h=30_000.0, p_rule=0.9)
        first_token, _ = candidate_tokens([first], "mode")
        second_token, _ = candidate_tokens([second], "mode")
        self.assertEqual(first_token, second_token)

    def test_path_v2_candidate_token_contains_business_features(self):
        candidate = {
            "target": {"batch_id": 1, "allocation_index": 0, "arc_index": 0},
            "eligible": True, "batch_quantity": 10.0, "share": 0.5,
            "path_count": 1, "path_length": 2,
            "path_modes": ["road", "rail"],
            "path_cost_per_teu": 100.0, "path_emission_per_teu": 30.0,
            "path_nominal_time_h": 12.0, "path_total_distance": 500.0,
            "path_bottleneck_capacity": 25.0,
            "alternative_mode_count": 1, "available_reliable_options": 2,
            "unallocated_path_count": 3, "from_node": "A", "to_node": "B",
            "current_mode": "road", "arc_distance": 100.0,
            "arc_capacity": 25.0, "arc_cost_rate": 1.0,
            "arc_emission_rate": 0.1, "arc_speed": 50.0,
            "arc_is_border": True,
        }
        batch = SimpleNamespace(
            batch_id=1, origin="A", destination="D", quantity=10.0,
            ET=2.0, LT=30.0)
        token, _ = candidate_tokens(
            [candidate], "mode", [batch], representation="path_v2")
        self.assertEqual(len(token[0]), len(CANDIDATE_PATH_V2_FEATURES))
        self.assertIn(100.0, token[0])
        self.assertIn(500.0, token[0])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
class NetworkEncodingModelTests(unittest.TestCase):
    def test_network_tokens_run_with_variable_candidate_paths(self):
        import torch
        from transformer_ppo.config import PPOConfig
        from transformer_ppo.models import build_model

        graph = SharedGraphObservationTests()
        graph.setUp()
        graph_observation = graph._build()
        observation = {
            "parent": [[0.0] * len(PARENT_FEATURES)],
            "candidates": [[0.0] * len(CANDIDATE_FEATURES) for _ in range(2)],
            "eligible": [True, True],
            "operator_id": 0,
            "global": [0.0] * len(GLOBAL_FEATURES),
            "graph": graph_observation,
            "candidate_graph_nodes": [[0, 1, 2], [0, 2]],
        }
        padded = pad_observations([observation])
        spec = {
            "parent_dim": len(PARENT_FEATURES),
            "candidate_dim": len(CANDIDATE_FEATURES),
            "global_dim": len(GLOBAL_FEATURES),
            "action_count": 5,
            **GRAPH_MODEL_SPEC,
        }
        config = PPOConfig(hidden_dim=16, attention_heads=4, transformer_layers=1)
        for architecture in ("mlp", "transformer"):
            model = build_model(architecture, config, spec)
            logits, value = model(padded)
            self.assertEqual(tuple(logits.shape), (1, 2))
            self.assertEqual(tuple(value.shape), (1,))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(torch.isfinite(value).all())

    def test_v2_ordered_node_edge_candidates_run_for_both_models(self):
        import torch
        from transformer_ppo.config import PPOConfig
        from transformer_ppo.models import build_model

        fixture = SharedGraphObservationTests()
        fixture.setUp()
        graph = build_graph_observation(
            fixture.nodes, fixture.regions, fixture.hold, fixture.processing,
            fixture.transfer, fixture.arcs, fixture.batches,
            representation="path_v2")
        candidates = [
            {"path_nodes": ["A", "B", "D"],
             "path_modes": ["road", "rail"]},
            {"path_nodes": ["A", "B"], "path_modes": ["road"]},
        ]
        observation = {
            "parent": [[0.0] * len(PARENT_FEATURES)],
            "candidates": [
                [0.0] * len(CANDIDATE_PATH_V2_FEATURES) for _ in candidates],
            "eligible": [True, True], "action_prior": [0.5, 0.5],
            "operator_id": 0, "global": [0.0] * len(GLOBAL_FEATURES),
            "graph": graph, "candidate_graph_nodes": [[0, 1, 2], [0, 1]],
            "candidate_graph_tokens": candidate_graph_tokens(candidates, graph),
        }
        padded = pad_observations([observation])
        spec = {
            "parent_dim": len(PARENT_FEATURES),
            "candidate_dim": len(CANDIDATE_PATH_V2_FEATURES),
            "global_dim": len(GLOBAL_FEATURES), "action_count": 5,
            **GRAPH_MODEL_SPEC_V2,
        }
        config = PPOConfig(
            hidden_dim=16, attention_heads=4, transformer_layers=1,
            dropout=0.0, actor_policy="base_residual",
            candidate_representation="path_v2")
        for architecture in ("mlp", "transformer"):
            with self.subTest(architecture=architecture):
                logits, value = build_model(architecture, config, spec)(padded)
                self.assertEqual(tuple(logits.shape), (1, 2))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(value).all())


if __name__ == "__main__":
    unittest.main()
