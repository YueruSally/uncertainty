"""Scientific-contract tests for the shared I/C/M reward archive."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformer_ppo.archive import (  # noqa: E402
    ObjectiveNormalizer,
    RewardArchive,
    RewardConfig,
    hv_reward,
)


class LearningRewardArchiveTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = ObjectiveNormalizer(
            (0., 0., 0.), (10., 10., 10.), reference=(1., 1., 1.))

    def test_action_children_are_settled_jointly_and_order_independently(self):
        children = [((2., 8., 5.), True, "left"),
                    ((8., 2., 5.), True, "right")]
        forward = RewardArchive(self.normalizer)
        reverse = RewardArchive(self.normalizer)
        first = forward.observe_many(children)
        second = reverse.observe_many(reversed(children))
        self.assertAlmostEqual(first.delta_hv, second.delta_hv)
        self.assertAlmostEqual(first.hv_after, second.hv_after)
        self.assertEqual(first.archive_size, 2)
        self.assertEqual(first.accepted_fingerprints, 2)

    def test_batch_is_compared_with_archive_before_any_child_is_inserted(self):
        archive = RewardArchive(self.normalizer)
        archive.observe((6., 6., 6.), True, "existing")
        before = archive.hypervolume
        update = archive.observe_many([
            ((4., 7., 4.), True, "a"),
            ((7., 4., 4.), True, "b"),
        ])
        self.assertAlmostEqual(update.hv_before, before)
        self.assertAlmostEqual(update.delta_hv, update.hv_after - before)

    def test_duplicate_and_infeasible_candidates_do_not_change_reward(self):
        archive = RewardArchive(self.normalizer)
        archive.observe((2., 2., 2.), True, "existing")
        before = archive.hypervolume
        update = archive.observe_many([
            ((1., 1., 1.), True, "existing"),
            ((1., 1., 1.), False, "infeasible"),
        ])
        self.assertEqual(update.delta_hv, 0.)
        self.assertEqual(update.hv_after, before)
        self.assertEqual(update.accepted_fingerprints, 0)

    def test_archive_reset_starts_an_independent_run(self):
        archive = RewardArchive(self.normalizer)
        archive.observe((2., 2., 2.), True, "a")
        archive.reset()
        self.assertEqual(archive.points, ())
        self.assertEqual(archive.hypervolume, 0.)
        self.assertGreater(archive.observe((2., 2., 2.), True, "a"), 0.)

    def test_cached_hypervolume_matches_exact_recomputation(self):
        archive = RewardArchive(self.normalizer)
        archive.observe_many([
            ((2., 8., 5.), True, "left"),
            ((8., 2., 5.), True, "right"),
        ])
        from transformer_ppo.archive import exact_hv_3d
        self.assertAlmostEqual(
            archive.hypervolume,
            exact_hv_3d(archive.points, self.normalizer.reference),
        )

    def test_normalization_does_not_clip_values_outside_frozen_bounds(self):
        self.assertEqual(self.normalizer.transform((12., -2., 5.)), (1.2, -.2, .5))

    def test_reward_is_only_scaled_hv_and_is_not_clipped(self):
        config = RewardConfig(positive_hv_scale=.01)
        self.assertEqual(hv_reward(0., config), 0.)
        self.assertEqual(hv_reward(.005, config), .5)
        self.assertEqual(hv_reward(.05, config), 5.)

    def test_reward_only_zeroes_floating_point_negative_noise(self):
        config = RewardConfig(floating_tolerance=1e-9)
        self.assertEqual(hv_reward(-1e-10, config), 0.)
        with self.assertRaises(ValueError):
            hv_reward(-1e-5, config)


if __name__ == "__main__":
    unittest.main()
