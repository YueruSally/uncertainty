"""On-policy PPO with whole-search terminal returns and concrete candidates.

The legacy search controller is an inference executor only. This module owns
the actual PPO updates, after complete episodes from an unchanged actor.
"""
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from m_candidate_information import CandidatePolicy, THRESHOLD_SCALED, private_seed
from m_operator_fixed_mix import FixedMixPolicy
from m_operator_offline import TinyOperatorTransformer, transform, read_json, digest
from multi_instance_policy import Decision

TEMPERATURE = .25
METHODS = ("HEURISTIC_BALANCED", "TRANSFORMER_PRETRAINED", "TRANSFORMER_PPO")


def load_checkpoint(path):
    # Only locally generated checkpoints whose digest is checked by the caller.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")


def save_checkpoint(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    torch.save(data, temporary)
    temporary.replace(path)


class TerminalActorCritic(nn.Module):
    def __init__(self, feature_count, ensemble_count):
        super().__init__()
        self.actors = nn.ModuleList([TinyOperatorTransformer(feature_count) for _ in range(ensemble_count)])
        self.critic = nn.Sequential(nn.Linear(feature_count, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(self, tokens, eligible):
        if not bool(eligible[:, 4].all()):
            raise ValueError("PPO supports states with a legal Balanced fallback only")
        raw = torch.stack([m(tokens, eligible) for m in self.actors]).mean(0)
        # At update zero argmax is precisely the frozen pretrained gate.
        offset = raw.new_tensor([THRESHOLD_SCALED]*4 + [0.])
        logits = ((raw-offset)/TEMPERATURE).masked_fill(~eligible, -torch.inf)
        pooled = (tokens*eligible.unsqueeze(-1)).sum(1)/eligible.sum(1, keepdim=True)
        return logits, self.critic(pooled).squeeze(-1)


def initialize(bundle):
    info = read_json(bundle)
    frozen = CandidatePolicy("initialization", 0, "STATE_PLUS_CANDIDATE", bundle)
    torch.manual_seed(898100)
    model = TerminalActorCritic(len(info["scaler"]["mean"]), len(frozen.models))
    for target, source in zip(model.actors, frozen.models):
        target.load_state_dict(source.state_dict())
    return model


def checkpoint_model(checkpoint, expected_sha):
    if digest(checkpoint) != expected_sha:
        raise ValueError("PPO checkpoint digest changed")
    saved = load_checkpoint(checkpoint)
    if saved["protocol"] != "terminal-hv-ppo-v1":
        raise ValueError("wrong PPO checkpoint protocol")
    model = TerminalActorCritic(saved["feature_count"], saved["ensemble_count"])
    model.load_state_dict(saved["model"], strict=True)
    return model, saved


class TerminalPolicy(FixedMixPolicy):
    def __init__(self, instance, seed, bundle, checkpoint, checkpoint_sha, sample=False):
        super().__init__(instance, seed, 0.)
        info = read_json(bundle)
        self.model, saved = checkpoint_model(checkpoint, checkpoint_sha)
        if saved["bundle_sha256"] != digest(bundle):
            raise ValueError("actor/scaler provenance mismatch")
        self.scaler, self.summary_dim = info["scaler"], info["summary_dim"]
        self.model.eval()
        self.sample, self.collect, self.transitions = sample, False, []
        self.action_rng = np.random.default_rng(private_seed("terminal-hv-ppo", instance, seed))
        self.name = "terminal-hv-ppo-candidate-policy"
        self.checkpoint_provenance = {"sha256": checkpoint_sha, "bundle_sha256": digest(bundle),
            "outer_ppo_updates": saved["update"],
            "role": "on-policy rollout for outer PPO" if sample else "frozen deterministic evaluation"}

    def decide(self, summary, specific, mask):
        started = time.perf_counter()
        if not mask[4]:
            self.unsupported += 1
            return Decision(None, [0.]*5, [None]*5, {"decision_kind": "terminal_ppo_unsupported",
                "inference_seconds": 0., "switched": False})
        if summary.shape[-1] != self.summary_dim:
            raise ValueError("candidate input schema changed")
        x = transform(np.concatenate([summary, specific], axis=1)[None], self.scaler)
        with torch.no_grad():
            logits, value = self.model(torch.as_tensor(x), torch.as_tensor(mask[None]))
            distribution = Categorical(logits=logits)
            probs = distribution.probs[0].numpy().astype(float)
            if self.sample:
                action = int(self.action_rng.choice(5, p=probs/probs.sum()))
            else:
                # Balanced wins exact ties, matching CandidatePolicy.choose.
                action = int(logits[0].argmax()) if float(logits[0].max()) > 1e-12 else 4
            logp = float(distribution.log_prob(torch.tensor([action]))[0])
        if self.sample:
            self.transitions.append({"tokens": x[0].tolist(), "eligible": mask.tolist(),
                "action": action, "old_log_probability": logp, "old_value": float(value[0])})
        self.model_evaluations += 1
        self.action_counts[str(action)] += 1
        self.switches += int(action != 4)
        scores = [float(logits[0, i]) if mask[i] else None for i in range(5)]
        return Decision(action, probs.tolist(), scores, {"decision_kind": "terminal_ppo_rollout" if self.sample else "terminal_ppo_frozen",
            "inference_seconds": time.perf_counter()-started, "switched": action != 4})


def optimizer_for(model):
    return torch.optim.Adam(model.parameters(), lr=1e-4)


def update_ppo(model, optimizer, episodes, update_index):
    """gamma=1 Monte Carlo terminal return; no counterfactual/offline PPO data."""
    transitions, returns, rewards = [], [], []
    for episode in episodes:
        hv, baseline = episode["final_hv"], episode["balanced_hv"]
        if not np.isfinite([hv, baseline]).all() or baseline <= 0:
            raise ValueError("invalid terminal HV pair")
        reward = hv/baseline-1.
        rewards.append(reward)
        if not episode["transitions"]:
            raise ValueError("episode contains no model actions")
        transitions.extend(episode["transitions"])
        returns.extend([reward]*len(episode["transitions"]))
    x = torch.tensor([t["tokens"] for t in transitions], dtype=torch.float32)
    mask = torch.tensor([t["eligible"] for t in transitions], dtype=torch.bool)
    actions = torch.tensor([t["action"] for t in transitions])
    old_logp = torch.tensor([t["old_log_probability"] for t in transitions])
    old_values = torch.tensor([t["old_value"] for t in transitions])
    returns = torch.tensor(returns, dtype=torch.float32)
    model.eval()  # No dropout, identical probability computation during sampling/update.
    with torch.no_grad():
        logits, _ = model(x, mask)
        reproduced = Categorical(logits=logits).log_prob(actions)
    maximum_error = float((reproduced-old_logp).abs().max())
    if maximum_error > 2e-5:
        raise ValueError(f"not on-policy: saved log probabilities mismatch {maximum_error}")
    advantages = returns-old_values
    advantages = (advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-6)
    rng = np.random.default_rng(898100+update_index)
    batches, epochs, last_kl = 0, 0, 0.
    for epoch in range(4):
        order = rng.permutation(len(x))
        for offset in range(0, len(x), 64):
            ix = order[offset:offset+64]
            logits, values = model(x[ix], mask[ix])
            dist = Categorical(logits=logits)
            ratio = (dist.log_prob(actions[ix])-old_logp[ix]).exp()
            objective = torch.minimum(ratio*advantages[ix], ratio.clamp(.8, 1.2)*advantages[ix])
            loss = -objective.mean() + .5*((values-returns[ix])**2).mean() - .005*dist.entropy().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite terminal PPO loss")
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("non-finite terminal PPO gradient")
            optimizer.step()
            batches += 1
        epochs += 1
        with torch.no_grad():
            logits, _ = model(x, mask)
            log_ratio = Categorical(logits=logits).log_prob(actions)-old_logp
            last_kl = float((log_ratio.exp()-1-log_ratio).mean())
        if last_kl > .03:
            break
    return {"ppo_update": update_index, "complete_episodes": len(episodes), "actions": len(x),
        "terminal_rewards": rewards, "mean_terminal_reward": float(np.mean(rewards)),
        "on_policy_log_probability_max_error": maximum_error, "optimizer_steps": batches,
        "optimization_epochs": epochs, "approximate_kl": last_kl,
        "return_definition": "gamma=1; only terminal (final_HV / paired_Balanced_HV - 1); no per-action HV shaping"}
