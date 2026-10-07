"""Small independent checks of physical readout, reset replay and cue learning.

These use synthetic trajectories and a mock body; they establish software
boundaries, not validity of the real connectome or muscle/servo interface.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from flycue.learning import CuePolicy, make_yoked_reward_bank, train_policy
from flycue.readout import MotorReadout


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cue_runner_for_tests", ROOT / "scripts/run_cue_leg.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
CONFIG = json.loads((ROOT / "configs/cue_leg.json").read_text())


def trajectory(left=0., right=0.):
    times = np.arange(181) * .005
    forelegs = np.zeros((len(times), 2))
    pulse = (times >= .15) & (times <= .35)
    forelegs[pulse] = [left, right]
    tips = np.zeros((len(times), 6))
    tips[:, [0, 3]] = forelegs
    positions = np.zeros((len(times), 6, 3))
    positions[:, :, 2] = tips
    return times, {"foreleg_lift_mm": forelegs, "tip_lift_mm": tips,
                   "tip_body_mm": positions}


def metrics(t, measured):
    return runner.movement_metrics(CONFIG, t, measured, .1)


def test_behavior_label_comes_from_measured_trajectory_not_actuator_target():
    times, measured = trajectory()
    measured["joint_targets_rad"] = np.full((len(times), 42), 1000.)
    blank = metrics(times, measured)
    assert blank["success_by_cue"] == [False, False, True]
    assert blank["observed_active_legs"] == []
    times, measured = trajectory(left=.5)
    left = metrics(times, measured)
    assert left["success_by_cue"] == [True, False, False]
    assert left["observed_active_legs"] == ["LF"]
    np.testing.assert_allclose(left["latency_s"][0], .05)


def test_wrong_leg_bilateral_and_failed_return_are_not_correct_actions():
    times, measured = trajectory(right=.5)
    assert metrics(times, measured)["success_by_cue"] == [False, True, False]
    times, measured = trajectory(left=.5, right=.5)
    assert metrics(times, measured)["success_by_cue"] == [False, False, False]
    times, measured = trajectory(left=.5)
    measured["foreleg_lift_mm"][-20:, 0] = .2
    measured["tip_lift_mm"][-20:, 0] = .2
    measured["tip_body_mm"][-20:, 0, 2] = .2
    assert metrics(times, measured)["success_by_cue"] == [False, False, False]
    times, measured = trajectory(left=.5)
    measured["foreleg_lift_mm"][5, 0] = .2
    measured["tip_lift_mm"][5, 0] = .2
    measured["tip_body_mm"][5, 0, 2] = .2
    assert metrics(times, measured)["success_by_cue"] == [False, False, False]


@pytest.mark.parametrize("leg,axis,displacement", [(1, 0, .2), (3, 2, -.2), (5, 1, -.2)])
def test_non_target_lateral_or_downward_motion_rejects_unilateral_success(leg, axis, displacement):
    times, measured = trajectory(left=.5)
    measured["tip_body_mm"][40, leg, axis] = displacement
    if axis == 2:
        measured["tip_lift_mm"][40, leg] = displacement
        if leg == 3:
            measured["foreleg_lift_mm"][40, 1] = displacement
    result = metrics(times, measured)
    assert result["success_by_cue"] == [False, False, False]
    assert result["non_target_max_displacement_mm"][0] == abs(displacement)


def test_motor_readout_is_cell_identified_and_symmetric_under_side_swap():
    readout = MotorReadout((np.array([4, 7]), np.array([2, 9])), rate_scale=20.)
    # Monitor order differs from group order; map by cell index, never column side.
    monitors = np.array([9, 4, 2, 7])
    actual_state = np.array([[0., 5., 0., 15.], [3., 0., 17., 0.]])
    drive = readout.transform(actual_state, monitors)
    expected = -np.expm1(-1.)
    np.testing.assert_allclose(drive, [[expected, 0.], [0., expected]])
    np.testing.assert_allclose(readout.transform(np.zeros_like(actual_state), monitors), 0.)


class ResetBody:
    """Stateful deterministic mock tests the runner's action-bank reset contract."""

    def __init__(self):
        self.reset_calls = 0
        self.state = np.zeros(2)

    def observation(self):
        return {key: self.state.copy() for key in (
            "foreleg_lift_mm", "tip_lift_mm", "tip_body_mm", "tip_world_mm",
            "tip_head_mm", "joint_angles_rad", "joint_velocities_rad_s",
            "joint_targets_rad", "contact_forces_flygym", "thorax_world_mm")}

    def reset(self, seed):
        assert seed == 0
        self.reset_calls += 1
        self.state = np.zeros(2)
        return self.observation()

    def step(self, drive, dt):
        self.state += np.asarray(drive) * dt
        return self.observation()


def test_response_bank_body_is_reset_and_uses_only_neural_drive():
    body = ResetBody()
    times = np.arange(4) * .005
    left_drive = np.tile([1., 0.], (4, 1))
    first, _ = runner.body_trial(body, times, left_drive)
    runner.body_trial(body, times, np.tile([0., 1.], (4, 1)))
    repeated, _ = runner.body_trial(body, times, left_drive)
    assert body.reset_calls == 3
    np.testing.assert_array_equal(first["foreleg_lift_mm"], repeated["foreleg_lift_mm"])
    np.testing.assert_allclose(first["foreleg_lift_mm"][:, 1], 0.)


def test_training_changes_cue_mapping_and_noncontingent_control_has_no_action_signal():
    # Three arbitrary actions already measured to serve LF, RF and blank.
    bank = np.tile(np.eye(3)[None, :, :], (2, 1, 1))
    control_bank = make_yoked_reward_bank(bank)
    assert np.all(control_bank == 1. / 3.)
    initial = CuePolicy.initialize(3, 3, 31)
    learned, history = train_policy(bank, seed=31, episodes=1200)
    control, control_history = train_policy(control_bank, seed=31, episodes=1200)
    assert len(history) == len(control_history) == 1200
    assert np.mean([learned.probabilities(i)[i] for i in range(3)]) > .94
    assert abs(np.mean([control.probabilities(i)[i] for i in range(3)]) - 1. / 3.) < .03
    # Initial policy is still the same pre-training comparison for this seed.
    np.testing.assert_array_equal(initial.logits, CuePolicy.initialize(3, 3, 31).logits)
    assert all(0. < row["selection_probability"] <= 1. for row in history)
