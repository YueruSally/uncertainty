import json
from pathlib import Path
import tempfile
import unittest

from freeze_s0_v2_i_ablation_plan import (
    FIT_RANKS, VALIDATION_RANKS, build_plan,
)


class FreezeS0V2IAblationPlanTest(unittest.TestCase):
    def test_freezes_disjoint_static_fit_and_validation_sets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            instances = root / "instances"
            instances.mkdir()
            ranking = []
            for rank in range(1, 51):
                instance_id = f"train-S0-{rank:03d}"
                digest = f"digest-{rank}"
                (instances / f"{instance_id}.json").write_text(json.dumps({
                    "split": "train", "configuration": "S0",
                    "instance_id": instance_id,
                    "instance_digest": digest,
                }))
                ranking.append({
                    "rank": rank, "instance_id": instance_id,
                    "instance_digest": digest,
                    "candidate_path_count": 100 + rank,
                    "candidate_path_token_count": 1000 + rank,
                })
            selection = root / "selection.json"
            selection.write_text(json.dumps({
                "configuration": "S0", "instance_count": 50,
                "test_instances_used": False,
                "optimisation_outcomes_used": False,
                "selection_digest": "source-digest", "ranking": ranking,
            }))
            normalization = root / "normalization.json"
            normalization.write_text(json.dumps({
                "source": "training-only", "configuration": "S0",
                "test_instances_used": False,
                "validation_instances_used": False,
                "normalization_digest": "normal-digest",
            }))
            rule = root / "rule.json"
            rule.write_text(json.dumps({
                "normalization_digest": "rule-digest",
            }))

            plan = build_plan(instances, selection, normalization, rule)

            self.assertEqual(
                [row["rank"] for row in plan["fitting_instances"]],
                list(FIT_RANKS))
            self.assertEqual(
                [row["rank"] for row in plan["validation_instances"]],
                list(VALIDATION_RANKS))
            self.assertTrue(
                {row["instance_id"] for row in plan["fitting_instances"]}
                .isdisjoint({
                    row["instance_id"]
                    for row in plan["validation_instances"]
                }))
            self.assertEqual(
                [row["top_k"] for row in plan["settings"]],
                [4, 8, 16, None])
            self.assertFalse(plan["test_instances_used"])


if __name__ == "__main__":
    unittest.main()
