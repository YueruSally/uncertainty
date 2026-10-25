"""Scientific-contract tests for the shared I/C/M reward archive."""
import random
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformer_ppo.archive import (  # noqa: E402
    ObjectiveNormalizer,
    RewardArchive,
    RewardConfig,
    exact_hv_3d,
    hv_reward,
)


def slow_exact_hv_3d(points, reference):
    """Original slab implementation retained as an independent oracle."""
    from transformer_ppo.archive import nondominated

    clipped = nondominated(
        point for point in points
        if len(point) == 3
        and all(point[index] < reference[index] for index in range(3))
    )
    if not clipped:
        return 0.0
    xs = sorted({point[0] for point in clipped} | {reference[0]})
    volume = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        active = [(point[1], point[2]) for point in clipped if point[0] <= left]
        ys = sorted({point[0] for point in active} | {reference[1]})
        area = 0.0
        for low_y, high_y in zip(ys[:-1], ys[1:]):
            zs = [z for y, z in active if y <= low_y]
            if zs:
                area += (high_y - low_y) * max(0.0, reference[2] - min(zs))
        volume += (right - left) * area
    return volume


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
        self.assertAlmostEqual(
            archive.hypervolume,
            exact_hv_3d(archive.points, self.normalizer.reference),
        )

    def test_fast_exact_hv_matches_original_algorithm_on_random_inputs(self):
        generator = random.Random(20261025)
        reference = (1.1, 1.1, 1.1)
        for size in (0, 1, 2, 10, 50, 200):
            points = [
                tuple(generator.uniform(-0.2, 1.3) for _ in range(3))
                for _ in range(size)
            ]
            self.assertAlmostEqual(
                exact_hv_3d(points, reference),
                slow_exact_hv_3d(points, reference),
                places=12,
            )

    def test_incremental_archive_matches_batch_recomputation(self):
        generator = random.Random(101)
        archive = RewardArchive(self.normalizer)
        observed = []
        for index in range(200):
            objectives = tuple(generator.uniform(0., 12.) for _ in range(3))
            observed.append(objectives)
            archive.observe(objectives, True, str(index))
            self.assertAlmostEqual(
                archive.hypervolume,
                exact_hv_3d(
                    (self.normalizer.transform(point) for point in observed),
                    self.normalizer.reference,
                ),
                places=12,
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
