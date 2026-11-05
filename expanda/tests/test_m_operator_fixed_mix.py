import copy
import random
import unittest

from m_operator_fixed_mix import FixedMixPolicy, METHODS, INSTANCES, SEEDS, summarize, validate_pair


def synthetic_pair(instance, seed, percent):
    """Synthetic calculation fixtures; no transport-performance claim."""
    rows = []
    for method, probability, hv in ((METHODS[0], 0., 1.), (METHODS[1], .2, 1. + percent / 100)):
        rows.append({"instance": instance, "algorithm_seed": seed, "strategy": method,
            "pairing": {"scenario": "ccp100", "initial_population": ["same"], "seed": seed},
            "minimum_control_generation": 10, "real_objective_evaluation_count": 400,
            "evaluation_budget": 400, "ppo_updates": 0, "counterfactual_evaluation_count": 0,
            "final_hypervolume": hv, "optimisation_seconds": 1.,
            "fixed_mix": {"mode_probability": probability, "selection_seed": seed,
                "mode_selected": 0 if not probability else 10, "supported_decisions": 50,
                "mode_ineligible_fallbacks": 0}})
    return rows


class FixedMixTests(unittest.TestCase):
    def test_probability_and_rng_are_separate_and_reproducible(self):
        candidates = [{"eligible": True}] * 5
        state = random.getstate()
        a, b = FixedMixPolicy("fixture", 896001, .2), FixedMixPolicy("fixture", 896001, .2)
        first = [a.choose(candidates).chosen_index for _ in range(10000)]
        self.assertEqual(first, [b.choose(candidates).chosen_index for _ in range(10000)])
        self.assertEqual(state, random.getstate())
        self.assertTrue(.18 < first.count(3) / len(first) < .22)
        self.assertEqual(set(first), {3, 4})
        self.assertEqual(a.updates, 0)
        self.assertEqual(a.model_evaluations, 0)

    def test_illegal_mode_falls_back_and_balanced_unavailable_abstains(self):
        policy = FixedMixPolicy("fixture", 1, 1.)
        self.assertEqual(policy.choose([{"eligible": True}] * 5).chosen_index, 3)
        mask = [True, False, False, False, True]
        self.assertEqual(policy.choose([{"eligible": v} for v in mask]).chosen_index, 4)
        self.assertEqual(policy.mode_ineligible_fallbacks, 1)
        self.assertIsNone(policy.choose([{"eligible": True}] * 4 + [{"eligible": False}]).chosen_index)
        self.assertEqual(policy.unsupported, 1)

    def test_zero_probability_matches_balanced_action_for_every_legal_mask(self):
        policy = FixedMixPolicy("fixture", 7, 0.)
        for number in range(32):
            mask = [bool(number & (1 << i)) for i in range(5)]
            decision = policy.choose([{"eligible": v} for v in mask])
            self.assertEqual(decision.chosen_index, 4 if mask[4] else None)
        self.assertEqual(policy.switches, 0)

    def test_pairing_and_budget_mismatches_are_rejected(self):
        pair = synthetic_pair(INSTANCES[0], SEEDS[0], 2.)
        self.assertAlmostEqual(validate_pair(pair)["paired_percent_vs_heuristic"], 2.)
        broken = copy.deepcopy(pair)
        broken[1]["pairing"]["scenario"] = "different"
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            validate_pair(broken)
        broken = copy.deepcopy(pair)
        broken[1]["real_objective_evaluation_count"] = 399
        with self.assertRaisesRegex(ValueError, "exactly 400"):
            validate_pair(broken)

    def test_all_predeclared_pairs_required(self):
        with self.assertRaisesRegex(ValueError, "all 40"):
            summarize(synthetic_pair(INSTANCES[0], SEEDS[0], 2.))

    def test_instance_specific_effect_and_paired_contrast(self):
        rows = [r for instance, effect in zip(INSTANCES, (-2., 3.)) for seed in SEEDS
                for r in synthetic_pair(instance, seed, effect)]
        report = summarize(rows)
        self.assertEqual(report["conclusion"], "INSTANCE_SENSITIVE_FIXED_MIX")
        self.assertAlmostEqual(report["instance_contrast_022_minus_021_percentage_points"]["mean"], 5.)
        self.assertTrue(report["fixed_mix_benefit_detected"])
        self.assertFalse(report["automatic_learning_admission"])

    def test_noisy_average_does_not_automatically_admit_learning(self):
        rows = [r for instance in INSTANCES for i, seed in enumerate(SEEDS)
                for r in synthetic_pair(instance, seed, -3. if i % 2 else 3.)]
        report = summarize(rows)
        self.assertEqual(report["conclusion"], "FIXED_MIX_GAIN_NOT_ESTABLISHED")
        self.assertFalse(report["fixed_mix_benefit_detected"])


if __name__ == "__main__":
    unittest.main()
