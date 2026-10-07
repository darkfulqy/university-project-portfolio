"""Small, explicitly added cue-to-neural-input learning interface.

This contextual bandit is an engineering controller. The connectome recurrence
is frozen; learning here is not a claim of endogenous fly synaptic plasticity.
Rewards are computed from separately simulated body trajectories by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CuePolicy:
    logits: np.ndarray

    @classmethod
    def initialize(cls, n_cues: int, n_actions: int, seed: int) -> "CuePolicy":
        if n_cues < 1 or n_actions < 1:
            raise ValueError("At least one cue and action are required")
        return cls(np.random.default_rng(seed).normal(0.0, 0.01, (n_cues, n_actions)))

    def probabilities(self, cue: int) -> np.ndarray:
        z = self.logits[cue] - np.max(self.logits[cue])
        p = np.exp(z)
        return p / p.sum()

    def choose(self, cue: int, rng: np.random.Generator, *, greedy: bool = False) -> int:
        p = self.probabilities(cue)
        return int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))

    def update(self, cue: int, action: int, advantage: float, learning_rate: float) -> None:
        """REINFORCE score-function update; no motor target enters the actor."""
        gradient = -self.probabilities(cue)
        gradient[action] += 1.0
        self.logits[cue] += learning_rate * float(advantage) * gradient
        # Shift invariance keeps numbers bounded without changing probabilities.
        self.logits[cue] -= self.logits[cue].mean()


def train_policy(reward_bank: np.ndarray, *, seed: int, episodes: int = 1200,
                 learning_rate: float = 0.12,
                 initial_policy: CuePolicy | None = None) -> tuple[CuePolicy, list[dict]]:
    """Train on a measured bank [training ensemble, cue, neural action].

    Reusing responses is exact for independent reset trials with frozen neural
    and body parameters. The response bank and all variants must stay together
    in the training split. Test ensembles must not be supplied to this function.
    """
    bank = np.asarray(reward_bank, dtype=np.float64)
    if bank.ndim != 3 or not np.isfinite(bank).all() or min(bank.shape) < 1:
        raise ValueError("reward_bank must be finite [ensemble,cue,action]")
    if episodes < 1 or learning_rate <= 0:
        raise ValueError("Positive episodes and learning_rate required")
    n_ensembles, n_cues, n_actions = bank.shape
    policy = (CuePolicy.initialize(n_cues, n_actions, seed) if initial_policy is None
              else CuePolicy(np.array(initial_policy.logits, dtype=float, copy=True)))
    if policy.logits.shape != (n_cues, n_actions) or not np.isfinite(policy.logits).all():
        raise ValueError("Initial policy has incompatible or non-finite logits")
    baseline = np.zeros(n_cues, dtype=float)
    rng = np.random.default_rng(seed + 104729)
    history: list[dict] = []
    for episode in range(episodes):
        cue = int(rng.integers(n_cues))
        ensemble = int(rng.integers(n_ensembles))
        action = policy.choose(cue, rng)
        selection_probability = float(policy.probabilities(cue)[action])
        reward = float(bank[ensemble, cue, action])
        advantage = reward - baseline[cue]
        policy.update(cue, action, advantage, learning_rate)
        baseline[cue] += 0.05 * (reward - baseline[cue])
        history.append(dict(episode=episode, cue=cue, ensemble_index=ensemble,
                            action=action, reward=reward,
                            selection_probability=selection_probability,
                            post_update_probability=float(policy.probabilities(cue)[action])))
    return policy, history


def make_yoked_reward_bank(bank: np.ndarray) -> np.ndarray:
    """Same expected reinforcement per cue/ensemble, independent of action.

    This is a deterministic noncontingent expected-reward control, not a replay
    of a living animal's temporally yoked reward sequence.
    """
    bank = np.asarray(bank, dtype=float)
    return np.broadcast_to(bank.mean(axis=-1, keepdims=True), bank.shape).copy()
