"""Focused checks for population observer helpers (NEURON not required)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from flyobserver import population as pop

try:
    import lfpykit
except ImportError:  # geometry tests need the observer environment
    lfpykit = None


class TimeAxisAndColumns(unittest.TestCase):
    def test_accepts_uniform_axis(self):
        record = pop.validate_time_axis(np.arange(2401) / 1000, 1000, 2.4)
        self.assertEqual(record['n_samples'], 2401)

    def test_rejects_nonuniform_nonmonotonic_nonfinite_or_inconsistent_axes(self):
        t = np.arange(2401) / 1000
        jitter = t.copy(); jitter[100] += 2e-4
        for bad, rate, duration in [(jitter, 1000, 2.4), (t[::-1], 1000, 2.4), (np.r_[t[:-1], np.nan], 1000, 2.4),
                                    (t, 500, 2.4), (t, 1000, 2.0), (t + .001, 1000, 2.4)]:
            with self.assertRaises(pop.GateFailure):
                pop.validate_time_axis(bad, rate, duration)

    def test_column_lookup_rejects_duplicates_and_missing_traces(self):
        self.assertEqual(pop.column_lookup(np.array([5, 7, 9]), [9, 5]), {5: 0, 7: 1, 9: 2})
        with self.assertRaises(pop.GateFailure):
            pop.column_lookup(np.array([5, 7, 5]), [5])
        with self.assertRaises(pop.GateFailure):
            pop.column_lookup(np.array([5, 7]), [8])


class SignedAggregation(unittest.TestCase):
    def test_signed_sum_keeps_raw_amplitudes_and_groups(self):
        stack = np.zeros((3, 4, 2)); stack[0] = 3.0; stack[1] = -2.0; stack[2] = .5
        result = pop.aggregate_contributions(stack, ['Mi1_1', 'L1_2', 'Mi1_3'], ['Mi1', 'L1', 'Mi1'])
        np.testing.assert_array_equal(result['total'], np.full((4, 2), 1.5))
        np.testing.assert_array_equal(result['groups']['Mi1'], np.full((4, 2), 3.5))
        np.testing.assert_array_equal(result['groups']['L1'], np.full((4, 2), -2.0))
        self.assertEqual(result['group_order'], ['Mi1', 'L1'])
        with self.assertRaises(pop.GateFailure):
            pop.aggregate_contributions(stack, ['a', 'a', 'b'], ['x', 'x', 'y'])

    def test_aggregator_hook_receives_label_mapping_and_cannot_rescale(self):
        stack = np.random.default_rng(0).normal(size=(3, 5, 4))
        seen = []

        def reverse_order(mapping):
            seen.append(list(mapping))
            return sum(list(mapping.values())[::-1])

        custom = pop.aggregate_contributions(stack, ['Mi1_1', 'L1_2', 'L2_3'], ['Mi1', 'L1', 'L2'], aggregator=reverse_order)
        self.assertLess(custom['aggregator_relative_error'], 1e-12)
        self.assertIn(['Mi1_1', 'L1_2', 'L2_3'], seen)
        mean = lambda m: np.mean(np.stack(list(m.values())), axis=0)
        normalized = lambda m: sum(v / np.abs(v).max() for v in m.values())
        for bad in [mean, normalized]:
            with self.assertRaises(pop.GateFailure):
                pop.aggregate_contributions(stack, ['a', 'b', 'c'], ['x', 'y', 'z'], aggregator=bad)

    def test_reference_and_baseline_commute_with_summation(self):
        rng = np.random.default_rng(1)
        stack = rng.normal(size=(3, 50, 16)); mask = np.arange(50) < 10
        _, cells = pop.reference_and_baseline(stack, 15, mask)
        referenced, total = pop.reference_and_baseline(stack.sum(axis=0), 15, mask)
        np.testing.assert_allclose(total, cells.sum(axis=0), rtol=0, atol=1e-12)
        self.assertEqual(float(np.max(np.abs(referenced[:, 15]))), 0.0)
        np.testing.assert_allclose(total[mask].mean(axis=0), 0, atol=1e-12)
        order = pop.summation_order_check(stack)
        self.assertLess(max(order['forward_vs_reverse'], order['forward_vs_vectorized']), 1e-12)


class ErrorsAndDescriptors(unittest.TestCase):
    def test_relative_error_floor_is_reported_and_inactive_for_real_signals(self):
        base = np.array([[5e-6, -1e-6]]); test = base + 5e-9
        record = pop.relative_max_error(test, base, 1e-8)
        self.assertFalse(record['floor_active']); self.assertAlmostEqual(record['relative_max_error'], 1e-3)
        record = pop.relative_max_error(np.full((1, 2), 1e-10), np.zeros((1, 2)), 1e-8)
        self.assertTrue(record['floor_active']); self.assertAlmostEqual(record['relative_max_error'], 1e-2)

    def test_cancellation_ratio_limits_and_undefined_counts(self):
        x = np.zeros((2, 3, 2))
        x[:, 0, 0] = [1.0, 2.0]    # same sign -> 1
        x[:, 1, 0] = [1.0, -1.0]   # full cancellation -> 0
        x[:, 2, 0] = [0.0, 0.0]    # undefined
        x[:, :, 1] = [[3.0], [-1.0]]  # 2/4
        ratio, summary = pop.cancellation_ratio(x, 1e-12)
        self.assertEqual(ratio[0, 0], 1.0); self.assertEqual(ratio[1, 0], 0.0)
        self.assertTrue(np.isnan(ratio[2, 0])); np.testing.assert_allclose(ratio[:, 1], .5)
        self.assertEqual(summary['undefined_entries'], 1); self.assertEqual(summary['defined_entries'], 5)
        _, masked = pop.cancellation_ratio(x, 1e-12, channel_mask=np.array([False, True]))
        self.assertEqual(masked['undefined_entries'], 0)

    def test_power_of_sum_equals_auto_plus_cross_terms(self):
        rng = np.random.default_rng(2); fs = 1000; t = np.arange(2000) / fs
        x = rng.normal(size=(3, 2000, 4)) + np.sin(2 * np.pi * 10 * t)[None, :, None]
        spectra = pop.spectral_decomposition(x, fs)
        self.assertLess(spectra['identity_relative_max_error'], 1e-12)
        self.assertEqual(spectra['cross_terms'].shape[0], 3)
        # Anti-phase equal sinusoids cancel: cross term is the negative of the auto sum.
        y = np.stack([np.sin(2 * np.pi * 10 * t), -np.sin(2 * np.pi * 10 * t)])[:, :, None]
        s = pop.spectral_decomposition(y, fs)
        k = int(np.argmin(np.abs(s['frequency_hz'] - 10)))
        self.assertLess(s['total_power'][k, 0], 1e-20)
        np.testing.assert_allclose(s['cross_terms'][0, k, 0], -s['auto_power'][:, k, 0].sum(), rtol=1e-12)


@unittest.skipIf(lfpykit is None, 'lfpykit not installed in this environment')
class KernelGeometry(unittest.TestCase):
    starts = np.array([[0., 0., 0.]]); ends = np.array([[10., 0., 0.]])
    point = np.array([[30., .5, 0.]])  # 20 um beyond the finite segment, .5 um off its infinite line

    def test_infinite_line_cutoff_is_counted_despite_finite_clearance(self):
        record = pop.geometry_gate(self.starts, self.ends, np.array([2.]), self.point, 5.0)
        self.assertGreater(record['min_finite_centerline_distance_um'], 5.0)
        self.assertEqual(record['pairs_inside_segment_radius'], 0)
        self.assertEqual(record['line_kernel_radius_cutoff_pairs'], 1)
        self.assertFalse(record['passed'])
        far = pop.geometry_gate(self.starts, self.ends, np.array([2.]), np.array([[5., 30., 0.]]), 5.0)
        self.assertTrue(far['passed'])

    def test_invalid_geometry_inputs_are_rejected_not_counted_as_clear(self):
        good = dict(starts=self.starts, ends=self.ends, diameters=np.array([2.]), electrodes=self.point, min_clearance_um=5.0)
        nan_point = np.array([[np.nan, 0., 0.]])
        for change in [dict(electrodes=nan_point), dict(starts=np.array([[np.nan, 0., 0.]])),
                       dict(ends=self.starts.copy()), dict(diameters=np.array([0.])), dict(diameters=np.array([np.nan])),
                       dict(diameters=np.array([2., 2.])), dict(min_clearance_um=-1.0), dict(min_clearance_um=np.nan),
                       dict(electrodes=np.zeros((0, 3))), dict(electrodes=np.zeros((2, 2)))]:
            with self.assertRaises(pop.GateFailure, msg=str(change)):
                pop.geometry_gate(**{**good, **change})
        with self.assertRaises(pop.GateFailure):
            pop.line_kernel_cutoff_mask(self.starts, self.ends, [2.], nan_point)
        with self.assertRaises(pop.GateFailure):
            pop.finite_centerline_distance(self.starts, self.starts, self.point)

    def test_mask_tracks_installed_library_regularization(self):
        from lfpykit import CellGeometry, RecExtElectrode

        def matrix(diameter):
            geometry = CellGeometry(x=np.array([[0., 10.]]), y=np.array([[0., 0.]]), z=np.array([[0., 0.]]),
                                    d=np.array([diameter]))
            return RecExtElectrode(geometry, sigma=.3, x=np.array([30.]), y=np.array([.5]), z=np.array([0.]),
                                   method='linesource').get_transformation_matrix()[0, 0]

        masks = {d: bool(pop.line_kernel_cutoff_mask(self.starts, self.ends, [d], self.point)[0, 0]) for d in [2., .5, .4]}
        self.assertEqual(masks, {2.: True, .5: False, .4: False})
        # The library value depends on the diameter only when the cutoff is active.
        self.assertNotEqual(matrix(2.), matrix(.5))
        self.assertEqual(matrix(.5), matrix(.4))


if __name__ == '__main__':
    unittest.main()
