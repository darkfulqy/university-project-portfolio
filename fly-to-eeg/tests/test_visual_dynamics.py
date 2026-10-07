import numpy as np
import pytest
from scipy import sparse

from flyvision.dynamics import ContinuousCNS, Parameters, flash_train, parameters


def chain():
    # Photoreceptor inhibits relay, relay excites a downstream cell.
    w = sparse.csr_matrix([[0, 0, 0], [-1, 0, 0], [0, 1, 0]], dtype=float)
    p = Parameters(np.full(3, 0.02), np.array([0.25, 0.6, 0.25]), 0.8, 1)
    return ContinuousCNS(w, p)


def test_inhibitory_visual_response_and_output_block():
    m = chain()
    s = flash_train(0.4, .001, .1, .3)
    r = m.simulate(s, .001, [0], sparse.eye(3))
    change = r['population_activity_au'][150] - r['population_activity_au'][0]
    assert change[0] > 0 and change[1] < 0 and change[2] < 0
    w = m.W.copy().tolil(); w[:, 0] = 0
    b = ContinuousCNS(w.tocsr(), m.p).simulate(s, .001, [0], sparse.eye(3))
    np.testing.assert_allclose(b['population_activity_au'][:, 1:], b['population_activity_au'][0, 1:][None, :] + np.zeros((401, 2)), atol=1e-10)


def test_blank_equilibrium_and_repeatability():
    m = chain()
    a = m.simulate(np.zeros(200), .001, [0], sparse.eye(3))
    b = m.simulate(np.zeros(200), .001, [0], sparse.eye(3))
    np.testing.assert_array_equal(a['population_activity_au'], b['population_activity_au'])
    assert np.max(a['max_cell_change_au']) < 1e-10


def test_flash_boundaries_and_convergence():
    a = flash_train(.4, .001, .1, .3)
    assert a.sum() == 100 and a[100] == 1 and a[150] == 0 and a[300] == 0
    with pytest.raises(ValueError):
        flash_train(.4, .003, .1, .3)
    m = chain()
    coarse = m.simulate(a, .001, [0], sparse.eye(3))['population_activity_au']
    fine = m.simulate(flash_train(.4, .0005, .1, .3), .0005, [0], sparse.eye(3), output_every=2)['population_activity_au']
    assert np.max(np.abs(coarse - fine)) < .003


def test_rejects_unbounded_weight_normalization():
    with pytest.raises(ValueError):
        ContinuousCNS(np.eye(3) * 2, chain().p)


def test_parameter_ranges_are_executable_and_defaults_preserved():
    p = parameters(8, 211, tau_range_s=(.03, .03), tonic_range_au=(.4, .4))
    np.testing.assert_array_equal(p.tau_s, np.full(8, .03))
    np.testing.assert_array_equal(p.tonic_input, np.full(8, .4))
    p = parameters(8, 211)
    old_rng = np.random.default_rng(211)
    np.testing.assert_array_equal(p.tau_s, old_rng.uniform(.018,.022,8))
    np.testing.assert_array_equal(p.tonic_input, old_rng.uniform(.225,.275,8))
    with pytest.raises(ValueError):
        parameters(8, 211, tau_range_s=(-.01, .02))
