"""Behavioral tests for the added learning interface, independent of biology."""
import numpy as np
from flycue.learning import train_policy, make_yoked_reward_bank


def test_learns_arbitrary_mapping_and_reversal():
    rewards = np.full((3, 3, 4), -1.0)
    mapping = [2, 0, 3]
    for cue, action in enumerate(mapping):
        rewards[:, cue, action] = 1.0
    policy, _ = train_policy(rewards, seed=71, episodes=2400)
    assert [int(np.argmax(policy.probabilities(c))) for c in range(3)] == mapping
    assert all(policy.probabilities(c)[a] > 0.90 for c, a in enumerate(mapping))
    reversed_policy, _ = train_policy(rewards[:, [1, 0, 2], :], seed=71, episodes=2400)
    assert [int(np.argmax(reversed_policy.probabilities(c))) for c in range(3)] == [0, 2, 3]


def test_noncontingent_control_removes_action_information():
    bank = np.arange(24, dtype=float).reshape(2, 3, 4)
    yoked = make_yoked_reward_bank(bank)
    np.testing.assert_allclose(yoked[..., 0], bank.mean(axis=-1))
    np.testing.assert_allclose(yoked, np.broadcast_to(yoked[..., :1], bank.shape))
