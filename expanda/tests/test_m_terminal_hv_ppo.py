from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.distributions import Categorical

from m_candidate_information import CandidatePolicy, SPECIFIC_NAMES
from m_operator_offline import TinyOperatorTransformer, write_json, digest
from m_terminal_hv_ppo import (initialize, TerminalActorCritic, TerminalPolicy, load_checkpoint,
    optimizer_for, update_ppo, save_checkpoint, checkpoint_model)
from run_m_terminal_hv_ppo_v1 import verify_pair, report_search


class TerminalHVTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dim, self.summary_dim = 8+len(SPECIFIC_NAMES), 8
        paths = []
        for seed in (1, 2, 3):
            torch.manual_seed(seed)
            path = self.root/f"seed_{seed}.pt"
            torch.save({"model": TinyOperatorTransformer(self.dim).state_dict(), "method": "STATE_PLUS_CANDIDATE"}, path)
            paths.append(str(path))
        self.bundle = self.root/"BUNDLE.json"
        write_json(self.bundle, {"protocol": "candidate-information-v1", "threshold_scaled": .02,
            "summary_dim": self.summary_dim, "scaler": {"mean": [0.]*self.dim, "std": [1.]*self.dim,
                "feature_clip": 10., "target_scale": .013}, "checkpoints": {"STATE_PLUS_CANDIDATE": paths},
            "checkpoint_sha256": {p: digest(p) for p in paths}})
        self.model = initialize(self.bundle)
        self.checkpoint = self.root/"update_000.pt"
        self.save(self.model, optimizer_for(self.model))

    def tearDown(self):
        self.temp.cleanup()

    def save(self, model, optimizer):
        save_checkpoint(self.checkpoint, {"protocol": "terminal-hv-ppo-v1", "bundle_sha256": digest(self.bundle),
            "feature_count": self.dim, "ensemble_count": 3, "update": 0,
            "model": model.state_dict(), "optimizer": optimizer.state_dict()})

    def episodes(self, model):
        rng = np.random.default_rng(8)
        episodes = []
        for i in range(4):
            tokens = torch.tensor(rng.normal(size=(12,5,self.dim)), dtype=torch.float32)
            mask = torch.ones((12,5), dtype=torch.bool)
            mask[:, 1] = False
            with torch.no_grad():
                logits, values = model(tokens, mask)
                dist = Categorical(logits=logits)
                actions = torch.tensor([0,2,3,4]*3)
                logp = dist.log_prob(actions)
            transitions = [{"tokens": tokens[j].tolist(), "eligible": mask[j].tolist(), "action": int(actions[j]),
                "old_log_probability": float(logp[j]), "old_value": float(values[j])} for j in range(12)]
            episodes.append({"final_hv": .8+.15*i, "balanced_hv": 1., "transitions": transitions})
        return episodes

    def test_initial_actor_matches_pretrained_actions_with_masks(self):
        old = CandidatePolicy("fixture", 7, "STATE_PLUS_CANDIDATE", self.bundle)
        new = TerminalPolicy("fixture", 7, self.bundle, self.checkpoint, digest(self.checkpoint))
        rng = np.random.default_rng(2)
        for _ in range(30):
            summary, specific = rng.normal(size=(5,8)), rng.normal(size=(5,len(SPECIFIC_NAMES)))
            mask = np.array([True, False, True, bool(rng.integers(2)), True])
            self.assertEqual(old.decide(summary, specific, mask).chosen_index,
                             new.decide(summary, specific, mask).chosen_index)

    def test_sampling_is_private_and_illegal_actions_are_never_sampled(self):
        policy = TerminalPolicy("fixture", 7, self.bundle, self.checkpoint, digest(self.checkpoint), sample=True)
        py, np_state = random.getstate(), np.random.get_state()
        for _ in range(20):
            action = policy.decide(np.zeros((5,8)), np.zeros((5,len(SPECIFIC_NAMES))), np.array([True,False,False,False,True]))
            self.assertIn(action.chosen_index, (0,4))
        self.assertEqual(random.getstate(), py)
        np.testing.assert_equal(np.random.get_state(), np_state)
        self.assertEqual(len(policy.transitions), 20)

    def test_real_on_policy_update_changes_actor_and_reproduces_after_resume(self):
        episodes = self.episodes(self.model)
        optimizer = optimizer_for(self.model)
        before = deepcopy(self.model.actors.state_dict())
        report = update_ppo(self.model, optimizer, episodes, 1)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in self.model.actors.state_dict().items()))
        np.testing.assert_allclose(report["terminal_rewards"], [-.2,-.05,.1,.25])
        self.assertLess(report["on_policy_log_probability_max_error"], 2e-5)
        self.save(self.model, optimizer)
        restored, saved = checkpoint_model(self.checkpoint, digest(self.checkpoint))
        restored_optimizer = optimizer_for(restored)
        restored_optimizer.load_state_dict(saved["optimizer"])
        episodes = self.episodes(self.model)
        left = update_ppo(self.model, optimizer, episodes, 2)
        right = update_ppo(restored, restored_optimizer, episodes, 2)
        self.assertEqual(left, right)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)

    def test_off_policy_data_and_invalid_baselines_are_rejected(self):
        episodes = self.episodes(self.model)
        episodes[0]["transitions"][0]["old_log_probability"] += .1
        with self.assertRaisesRegex(ValueError, "not on-policy"):
            update_ppo(self.model, optimizer_for(self.model), episodes, 1)
        episodes = self.episodes(self.model)
        episodes[0]["balanced_hv"] = 0
        with self.assertRaisesRegex(ValueError, "invalid terminal HV"):
            update_ppo(self.model, optimizer_for(self.model), episodes, 1)

    def test_old_torch_compatibility_does_not_hide_other_failures(self):
        with patch("m_terminal_hv_ppo.torch.load", side_effect=[TypeError("weights_only is invalid"), {"ok": True}]):
            self.assertTrue(load_checkpoint("trusted-local")["ok"])
        with patch("m_terminal_hv_ppo.torch.load", side_effect=TypeError("something else")):
            with self.assertRaisesRegex(TypeError, "something else"):
                load_checkpoint("trusted-local")

    def test_reject_changed_pairing_and_missing_methods(self):
        row = {"pairing": {"scenario": "same"}, "real_objective_evaluation_count": 400,
            "ppo_updates": 0, "counterfactual_evaluation_count": 0, "training_feedback_evaluations": 0,
            "source_bundle_sha256": "same", "candidate_contract": "concrete-proposal-v1",
            "minimum_control_generation": 10, "final_hypervolume": 1.}
        verify_pair([row, deepcopy(row)])
        other = deepcopy(row)
        other["pairing"]["scenario"] = "different"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_pair([row, other])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            report_search([], {"test_instances": ["fixture"], "test_seeds": [7]})


if __name__ == "__main__":
    unittest.main()
