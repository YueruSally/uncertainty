"""On-policy clipped PPO for variable-length mutation candidate sets."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from multi_instance_policy import Decision

from .models import build_model
from .observations import pad_observations


@dataclass
class Transition:
    observation: dict
    action: int
    old_log_probability: float
    value: float
    reward: float
    done: bool


class PPOTrainer:
    def __init__(self, architecture, config, observation_provider, device=None,
                 policy_seed=0, training=True):
        config.validate()
        self.architecture, self.config = architecture, config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # The policy owns its random streams.  In particular, PPO minibatch and
        # action sampling must not consume NumPy/Python randomness used by NSGA-II.
        torch.manual_seed(int(policy_seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(policy_seed))
        self.action_generator = torch.Generator(device=self.device.type)
        self.action_generator.manual_seed(int(policy_seed) + 1)
        self.minibatch_rng = np.random.default_rng(int(policy_seed) + 2)
        self.model = build_model(architecture, config).to(self.device)
        critic_ids = {id(p) for p in self.model.heads.critic_head.parameters()}
        actor = [p for p in self.model.parameters() if id(p) not in critic_ids]
        critic = list(self.model.heads.critic_head.parameters())
        self.optimizer = torch.optim.Adam([
            {"params": actor, "lr": config.actor_learning_rate},
            {"params": critic, "lr": config.critic_learning_rate},
        ])
        self.observation_provider = observation_provider
        self.buffer: list[Transition] = []
        self.pending = None
        self.training = bool(training)
        self.updates = 0

    @property
    def name(self):
        return f"{self.architecture}-ppo-masked"

    def _distribution(self, observation):
        batch = pad_observations([observation], self.device)
        logits, value = self.model(batch)
        return Categorical(logits=logits), value

    def choose(self, candidates, context=None):
        eligible = [i for i, row in enumerate(candidates) if row.get("eligible")]
        if not eligible:
            return Decision(None, [0.0] * len(candidates), [None] * len(candidates),
                            {"no_eligible_candidate": True, "fallback": None})
        observation = self.observation_provider(candidates, context or {})
        if self.training and len(self.buffer) >= self.config.rollout_steps:
            self.update(self._value(observation))
        self.model.train(self.training)
        with torch.no_grad():
            distribution, value = self._distribution(observation)
            action = (torch.multinomial(
                distribution.probs, 1, generator=self.action_generator).squeeze(-1)
                if self.training else torch.argmax(distribution.logits, dim=-1))
            log_probability = distribution.log_prob(action)
            probabilities = distribution.probs.squeeze(0).cpu().tolist()
            entropy = distribution.entropy().item()
        self.pending = {
            "observation": observation, "action": int(action.item()),
            "old_log_probability": float(log_probability.item()), "value": float(value.item())}
        return Decision(int(action.item()), probabilities, [None] * len(probabilities), {
            "no_eligible_candidate": False, "fallback": None,
            "old_log_probability": float(log_probability.item()),
            "state_value": float(value.item()), "entropy": float(entropy),
            "policy_update": self.updates, "architecture": self.architecture,
        })

    def observe_outcome(self, reward, done=False):
        if self.pending is None:
            return
        if self.training:
            self.buffer.append(Transition(reward=float(reward), done=bool(done), **self.pending))
        self.pending = None

    def _value(self, observation):
        with torch.no_grad():
            _, value = self._distribution(observation)
        return float(value.item())

    def finish(self):
        if self.pending is not None:
            raise RuntimeError("episode ended before pending action received a reward")
        if self.training and self.buffer:
            self.buffer[-1].done = True
            self.update(0.0)

    def _advantages(self, bootstrap_value):
        rewards = np.asarray([x.reward for x in self.buffer], dtype=np.float32)
        values = np.asarray([x.value for x in self.buffer] + [bootstrap_value], dtype=np.float32)
        dones = np.asarray([x.done for x in self.buffer], dtype=bool)
        deltas = rewards + self.config.gamma * values[1:] * (~dones) - values[:-1]
        advantages = np.zeros_like(rewards)
        for start in range(len(rewards)):
            factor = 1.0
            for offset in range(self.config.gae_horizon):
                index = start + offset
                if index >= len(rewards):
                    break
                advantages[start] += factor * deltas[index]
                if dones[index]:
                    break
                factor *= self.config.gamma * self.config.gae_lambda
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, bootstrap_value=0.0):
        if not self.training or not self.buffer:
            return {}
        advantages, returns = self._advantages(float(bootstrap_value))
        old_logp = torch.as_tensor([x.old_log_probability for x in self.buffer],
                                   dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([x.action for x in self.buffer],
                                  dtype=torch.long, device=self.device)
        advantages_t = torch.as_tensor(advantages, device=self.device)
        returns_t = torch.as_tensor(returns, device=self.device)
        indices = np.arange(len(self.buffer))
        diagnostics = {"policy_loss": [], "value_loss": [], "entropy": [], "kl": []}
        stop = False
        for _ in range(self.config.update_epochs):
            indices = self.minibatch_rng.permutation(indices)
            for start in range(0, len(indices), 64):
                selected = indices[start:start + 64]
                observations = [self.buffer[i].observation for i in selected]
                batch = pad_observations(observations, self.device)
                logits, value = self.model(batch)
                distribution = Categorical(logits=logits)
                index = torch.as_tensor(selected, dtype=torch.long, device=self.device)
                new_logp = distribution.log_prob(actions[index])
                ratio = torch.exp(new_logp - old_logp[index])
                unclipped = ratio * advantages_t[index]
                clipped = torch.clamp(ratio, 1 - self.config.clip_ratio,
                                      1 + self.config.clip_ratio) * advantages_t[index]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (value - returns_t[index]).pow(2).mean()
                entropy = distribution.entropy().mean()
                loss = (policy_loss + self.config.value_coefficient * value_loss
                        - self.config.entropy_coefficient * entropy)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                approximate_kl = (old_logp[index] - new_logp).mean().item()
                diagnostics["policy_loss"].append(float(policy_loss.item()))
                diagnostics["value_loss"].append(float(value_loss.item()))
                diagnostics["entropy"].append(float(entropy.item()))
                diagnostics["kl"].append(float(approximate_kl))
                if approximate_kl > self.config.target_kl:
                    stop = True
                    break
            if stop:
                break
        self.buffer.clear()
        self.updates += 1
        return {name: float(np.mean(values)) for name, values in diagnostics.items() if values}

    def save(self, path, metadata=None):
        path = Path(path)
        if path.exists():
            raise ValueError(f"checkpoint already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": 1, "architecture": self.architecture,
                    "ppo_config": asdict(self.config), "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "updates": self.updates,
                    "metadata": metadata or {}}, path)

    @classmethod
    def load(cls, path, config_type, observation_provider, device=None, training=False):
        artifact = torch.load(path, map_location=device or "cpu")
        if artifact.get("schema_version") != 1:
            raise ValueError("unsupported checkpoint schema")
        config = config_type(**artifact["ppo_config"])
        result = cls(artifact["architecture"], config, observation_provider,
                     device=device, training=training)
        result.model.load_state_dict(artifact["model"])
        if training:
            result.optimizer.load_state_dict(artifact["optimizer"])
        result.updates = int(artifact.get("updates", 0))
        return result
