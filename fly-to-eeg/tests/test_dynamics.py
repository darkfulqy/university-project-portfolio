"""Numerical and biological-interface checks for continuous activity dynamics."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

from flycue.dynamics import RateNetwork, RateParameters, sample_parameters


def fixed_parameters(n, seed=1):
    return RateParameters(np.ones(n), np.full(n, 7.5), np.full(n, 200.), np.full(n, .020), seed)


def test_parameter_ensemble_reproducible_and_size_scaling_explicit():
    sizes = np.array([0.5, 1.0, 3.0])
    scaled = sample_parameters(sizes, 47)
    base = sample_parameters(np.ones(3), 47)
    np.testing.assert_allclose(scaled.gain * sizes, base.gain)
    np.testing.assert_allclose(scaled.threshold / sizes, base.threshold)
    np.testing.assert_array_equal(scaled.rmax, base.rmax)
    np.testing.assert_array_equal(scaled.tau_s, base.tau_s)
    assert scaled.as_dict() == sample_parameters(sizes, 47).as_dict()
    assert not np.array_equal(scaled.gain, sample_parameters(sizes, 48).gain)


def test_isolated_cell_matches_analytic_pulse_and_decay():
    p = fixed_parameters(1)
    net = RateNetwork(csr_matrix((1, 1)), [1.], seed=1, parameters=p)
    dt = .001
    stim = np.zeros((100, 1))
    stim[:50] = 30.
    result = net.simulate(duration_s=.1, dt_s=dt, stimulation_indices=[0], stimulation=stim)
    plateau = 200 * np.tanh((30. - 7.5) / 200)
    expected = np.where(result.time_s <= .05,
                        plateau * -np.expm1(-result.time_s / .02),
                        plateau * -np.expm1(-.05 / .02) * np.exp(-(result.time_s - .05) / .02))
    np.testing.assert_allclose(result.activity[:, 0], expected, rtol=1e-13, atol=1e-13)
    assert result.metadata["activity_kind"] == "abstract_continuous_rate"
    assert not result.metadata["state_clipping"]


def test_signed_recurrence_and_post_pre_orientation():
    W = csr_matrix(([80., -80.], ([1, 2], [0, 0])), shape=(3, 3))
    net = RateNetwork(W, np.ones(3), seed=1, parameters=fixed_parameters(3))
    result = net.simulate(duration_s=.2, dt_s=.0005, stimulation_indices=[0, 2], stimulation=[30., 15.])
    assert result.activity[-1, 1] > 20.  # Excitatory 0 -> 1 pathway propagates.
    assert result.activity[-1, 2] < .01  # Inhibitory 0 -> 2 suppresses tonic drive.
    zero = net.simulate(duration_s=.1, dt_s=.001)
    assert np.count_nonzero(zero.activity) == 0


@pytest.mark.parametrize("solver,error_ratio,max_error", [
    ("exponential_leak_euler", .6, .3),
    # Rectification has derivative kinks, so global fourth-order scaling is
    # not assumed for this trajectory; still require a clear error decrease.
    ("rk4", .5, .005),
])
def test_recurrent_time_step_convergence_against_adaptive_reference(solver, error_ratio, max_error):
    # Delayed inhibitory feedback makes the integration error observable.
    W = csr_matrix([[0., -75., 0.], [65., 0., 20.], [30., 0., 0.]])
    net = RateNetwork(W, np.ones(3), seed=1, parameters=fixed_parameters(3))
    drive = np.array([40., 0., 0.])
    p = net.parameters

    def rhs(t, r):
        target = p.rmax * np.maximum(np.tanh(p.gain / p.rmax * (drive + .03 * (W @ r) - p.threshold)), 0.)
        return (target - r) / p.tau_s

    reference = solve_ivp(rhs, (0., .2), np.zeros(3), method="DOP853", rtol=1e-10, atol=1e-12, dense_output=True)
    coarse = net.simulate(duration_s=.2, dt_s=.0005, stimulation_indices=[0], stimulation=[40.], solver=solver)
    fine = net.simulate(duration_s=.2, dt_s=.00025, stimulation_indices=[0], stimulation=[40.], output_every=2, solver=solver)
    truth = reference.sol(coarse.time_s).T
    coarse_error = np.sqrt(np.mean((coarse.activity - truth) ** 2))
    fine_error = np.sqrt(np.mean((fine.activity - truth) ** 2))
    assert reference.success
    assert fine_error < error_ratio * coarse_error
    assert np.max(np.abs(fine.activity - truth)) < max_error


def test_rk4_zero_order_hold_pulse_matches_analytic_solution_and_rejects_large_dt():
    net = RateNetwork([[0.]], [1.], seed=1, parameters=fixed_parameters(1))
    dt = .001
    called_at = []

    def drive(t):
        called_at.append(t)
        return [30. if t < .05 else 0.]

    result = net.simulate(duration_s=.1, dt_s=dt, stimulation_indices=[0], stimulation=drive, solver="rk4")
    plateau = 200 * np.tanh((30. - 7.5) / 200)
    expected = np.where(result.time_s <= .05,
                        plateau * -np.expm1(-result.time_s / .02),
                        plateau * -np.expm1(-.05 / .02) * np.exp(-(result.time_s - .05) / .02))
    np.testing.assert_allclose(result.activity[:, 0], expected, rtol=5e-7, atol=1e-6)
    assert len(called_at) == 100  # No callback at intermediate stages or pulse edge.
    assert result.metadata["solver"] == "rk4"
    assert result.diagnostics["target_saturation_observations_per_step"] == 4
    with pytest.raises(ValueError, match="requires dt_s"):
        net.simulate(duration_s=.1, dt_s=.05, solver="rk4")


def test_monitors_callback_and_last_sample_do_not_change_network():
    net = RateNetwork(csr_matrix([[0., 0.], [30., 0.]]), np.ones(2), seed=7)
    full = net.simulate(duration_s=.01, dt_s=.001, stimulation_indices=[0], stimulation=[20.])
    subset = net.simulate(duration_s=.01, dt_s=.001, stimulation_indices=[0], stimulation=lambda t: [20.],
                          monitor_indices=[1], output_every=3, record_all=True)
    np.testing.assert_allclose(subset.time_s, [0., .003, .006, .009, .010])
    np.testing.assert_array_equal(subset.activity[:, 0], full.activity[[0, 3, 6, 9, 10], 1])
    np.testing.assert_array_equal(subset.all_activity, full.activity[[0, 3, 6, 9, 10]])
    np.testing.assert_array_equal(subset.final_state, full.final_state)


def test_saturation_is_reported_and_bad_numerics_raise():
    net = RateNetwork([[0.]], [1.], seed=1, parameters=fixed_parameters(1))
    result = net.simulate(duration_s=.2, dt_s=.001, stimulation_indices=[0], stimulation=[1e6])
    assert result.diagnostics["all_finite"]
    assert result.diagnostics["ever_saturated_cell_fraction"] == 1.
    assert result.diagnostics["saturated_cell_time_fraction"] > .5
    assert np.max(result.activity) <= 200.
    with pytest.raises(ValueError, match="non-finite"):
        net.simulate(duration_s=.1, dt_s=.001, stimulation_indices=[0], stimulation=[np.inf])
    with pytest.raises(ValueError, match="invalid input"):
        net.simulate(duration_s=.1, dt_s=.001, stimulation_indices=[0], stimulation=lambda t: [np.nan])
    explosive = RateNetwork([[1e308]], [1.], seed=1, parameters=fixed_parameters(1))
    with pytest.raises(FloatingPointError):
        explosive.simulate(duration_s=.1, dt_s=.001, initial_state=[100.])


@pytest.mark.parametrize("sizes", [[0.], [-1.], [np.nan], []])
def test_invalid_sizes_rejected(sizes):
    with pytest.raises(ValueError):
        sample_parameters(sizes, 1)
