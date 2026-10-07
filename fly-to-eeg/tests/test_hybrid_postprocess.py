"""Tests against the real unchanged upstream hybridLFPy PostProcess (no mocks).

Requires .venv-hybrid-postproc; skipped elsewhere. Run without pytest:
  .venv-hybrid-postproc/bin/python -m unittest tests.test_hybrid_postprocess -v
"""
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for dep in ('h5py', 'mpi4py'):
    if importlib.util.find_spec(dep) is None:
        raise unittest.SkipTest(f'{dep} unavailable; use .venv-hybrid-postproc')

import numpy as np
from flyobserver import hybrid_postprocess as hp

spec = importlib.util.spec_from_file_location('aggregate_script', ROOT / 'scripts/aggregate_hybrid_population.py')
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def synthetic():
    s = script.synthetic_contributions()
    labels = [str(k) for k in s['labels']]
    contributions = {k: s['per_cell_potential_uV'][i] for i, k in enumerate(labels)}
    return contributions, s['time_s'], s['electrodes_um'], s['expected_unreferenced_uV']


class HybridPostprocessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_adapter(self, name='out', ref=1, **overrides):
        contributions, t, xyz, _ = synthetic()
        kwargs = dict(contributions_uV=contributions, time_s=t, electrodes_um=xyz,
                      reference_index=ref, output_dir=self.tmp / name)
        kwargs.update(overrides)
        return hp.aggregate(**kwargs)

    def test_real_postprocess_total_cancels_and_reinforces(self):
        _, _, _, expected = synthetic()
        result = self.run_adapter()
        np.testing.assert_allclose(result.unreferenced_uV, expected, rtol=0, atol=1e-12)
        self.assertEqual(np.abs(result.unreferenced_uV[:, 0]).max(), 0.0)
        self.assertGreater(np.abs(result.unreferenced_uV[:, 1]).max(), 0.0)
        rec = result.record
        self.assertEqual(rec['upstream']['module_sha256'], hp.UPSTREAM_POSTPROC_SHA256)
        self.assertEqual(rec['upstream']['commit'], hp.UPSTREAM_COMMIT)
        self.assertEqual(rec['upstream']['methods_invoked'],
                         ['PostProcess.__init__', 'PostProcess.calc_measure'])
        self.assertEqual(rec['runtime']['mpi_comm_world_size'], 1)
        self.assertEqual(rec['runtime']['mpi_comm_world_rank'], 0)
        self.assertIn('MPICH', rec['runtime']['mpi_library_version'])
        self.assertNotIn('neuron', sys.modules)
        self.assertFalse(rec['claims']['full_hybridLFPy_network_run'])
        module = hp.load_upstream_postprocess()
        self.assertTrue(hasattr(module, 'PostProcess'))
        self.assertEqual(Path(module.__file__).resolve(), hp.UPSTREAM_POSTPROC.resolve())
        for sub in ('cells', 'figures', 'populations'):
            self.assertTrue((result.output_dir / sub).is_dir())

    def test_random_float64_cancellation_order_and_reference(self):
        import math
        rng = np.random.default_rng(20260915)
        n_time, n_channel, ref = 64, 5, 3
        shape = (n_time, n_channel)
        a = rng.normal(scale=5.67e-6, size=shape)
        b = rng.normal(scale=3e-6, size=shape)
        c = -(a + b) + rng.normal(scale=1e-12, size=shape)
        d = rng.normal(scale=2e-6, size=shape)
        e = -d * (1 + rng.uniform(-1e-3, 1e-3, size=shape))
        contributions = {'randA': a, 'randB': b, 'randC': c, 'randD': d, 'randE': e}
        result = hp.aggregate(contributions, np.arange(n_time) / 2000.0,
                              rng.normal(scale=100.0, size=(n_channel, 3)), ref,
                              self.tmp / 'random', sample_rate_hz=2000.0)
        stacked = np.stack(list(contributions.values()))
        magnitude = np.abs(stacked).sum(axis=0)
        exact = np.array([[math.fsum(stacked[:, i, j]) for j in range(n_channel)]
                          for i in range(n_time)])
        self.assertLess(np.abs(exact).max(), 1e-3 * magnitude.max())
        factor = (2 * len(contributions) + 4) * np.finfo(np.float64).eps
        self.assertTrue((np.abs(result.unreferenced_uV - exact) <= factor * magnitude).all())
        ref_magnitude = magnitude + magnitude[:, [ref]]
        self.assertTrue((np.abs(result.referenced_uV - (exact - exact[:, [ref]]))
                         <= factor * ref_magnitude).all())
        # Summing per-cell referenced contributions adds n subtractions: allow 2x.
        per_cell = sum(x - x[:, [ref]] for x in contributions.values())
        self.assertTrue((np.abs(result.referenced_uV - per_cell) <= 2 * factor * ref_magnitude).all())
        np.testing.assert_array_equal(result.referenced_uV[:, ref], 0.0)
        self.assertLessEqual(result.record['checks']['independent_sum_max_error_over_tolerance'], 1.0)

    def test_import_writes_no_bytecode_and_restores_setting(self):
        snapshot_cache = hp.UPSTREAM_POSTPROC.parent / '__pycache__'
        existed = snapshot_cache.exists()
        copy = self.tmp / 'postproc.py'
        copy.write_bytes(hp.UPSTREAM_POSTPROC.read_bytes())
        cached = hp._loaded.pop(hp.UPSTREAM_POSTPROC_SHA256, None)
        previous = sys.dont_write_bytecode
        try:
            hp.load_upstream_postprocess(copy)
        finally:
            hp._loaded.pop(hp.UPSTREAM_POSTPROC_SHA256, None)
            if cached is not None:
                hp._loaded[hp.UPSTREAM_POSTPROC_SHA256] = cached
        self.assertEqual(sys.dont_write_bytecode, previous)
        self.assertFalse((self.tmp / '__pycache__').exists())
        hp.load_upstream_postprocess()
        self.assertEqual(snapshot_cache.exists(), existed)
        if not existed:
            self.assertFalse(snapshot_cache.exists())

    def test_input_hdf5_unchanged_and_transposed(self):
        import h5py
        result = self.run_adapter()
        for k, info in result.record['inputs'].items():
            self.assertEqual(info['hdf5_sha256_before_calc_measure'],
                             info['hdf5_sha256_after_calc_measure'])
            path = ROOT / info['hdf5_path'] if not Path(info['hdf5_path']).is_absolute() \
                else Path(info['hdf5_path'])
            self.assertEqual(hp.sha256_file(path), info['hdf5_sha256_before_calc_measure'])
            with h5py.File(path, 'r') as f:
                np.testing.assert_array_equal(f['data'][()], result.contributions_uV[k].T)
                self.assertEqual(f.attrs['units'], 'uV')

    def test_reference_linearity(self):
        contributions, t, xyz, _ = synthetic()
        result = self.run_adapter(ref=2)
        per_cell_referenced = sum(a - a[:, [2]] for a in contributions.values())
        np.testing.assert_allclose(result.referenced_uV, per_cell_referenced, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(result.referenced_uV[:, 2], 0.0)
        doubled = dict(contributions, synthA=2 * contributions['synthA'])
        result2 = self.run_adapter('doubled', ref=2, contributions_uV=doubled)
        a = contributions['synthA']
        np.testing.assert_allclose(result2.unreferenced_uV - result.unreferenced_uV, a,
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(result2.referenced_uV - result.referenced_uV,
                                   a - a[:, [2]], rtol=0, atol=1e-12)

    def assert_rejected(self, exc=ValueError, name='bad', **overrides):
        with self.assertRaises(exc):
            self.run_adapter(name, **overrides)
        self.assertFalse((self.tmp / name).exists())

    def test_rejects_invalid_inputs_before_creating_files(self):
        contributions, t, xyz, _ = synthetic()
        bad_shape = dict(contributions, synthC=contributions['synthC'][:, :2])
        self.assert_rejected(contributions_uV=bad_shape)
        self.assert_rejected(time_s=t[:4])
        self.assert_rejected(time_s=np.array([0, 1e-3, 3e-3, 4e-3, 5e-3]))
        self.assert_rejected(time_s=t[::-1])
        self.assert_rejected(time_s=np.array([0, 1e-3, 1e-3, 2e-3, 3e-3]))
        self.assert_rejected(sample_rate_hz=500.0)
        self.assert_rejected(units='mV')
        self.assert_rejected(electrodes_um=xyz[:, :2])
        self.assert_rejected(electrodes_um=xyz[:2])
        self.assert_rejected(electrodes_um=np.where(xyz == 50., np.nan, xyz))
        self.assert_rejected(ref=3)
        self.assert_rejected(ref=-1)
        self.assert_rejected(ref=True)
        self.assert_rejected(ref=1.0)
        nonfinite = {k: v.copy() for k, v in contributions.items()}
        nonfinite['synthB'][2, 1] = np.inf
        self.assert_rejected(contributions_uV=nonfinite)
        for label in ('', '../x', 'a/b', '.hidden', 'a b'):
            self.assert_rejected(contributions_uV={label: contributions['synthA']})
        self.assert_rejected(contributions_uV={'Cell': contributions['synthA'],
                                               'cell': contributions['synthB']})
        self.assert_rejected(contributions_uV={})
        self.assert_rejected(FileNotFoundError, output_dir=self.tmp / 'missing_parent' / 'out',
                             name='missing_parent')

    def test_overwrite_protection(self):
        existing = self.tmp / 'existing'
        existing.mkdir()
        (existing / 'keep.txt').write_text('keep')
        with self.assertRaises(FileExistsError):
            self.run_adapter('existing')
        self.assertEqual(sorted(p.name for p in existing.iterdir()), ['keep.txt'])
        self.run_adapter('once')
        with self.assertRaises(FileExistsError):
            self.run_adapter('once')

    def test_mpi_tcp_interface_option(self):
        import os
        hp.load_upstream_postprocess()
        for bad in ('', 'en0;x', '../en0', 'en 0'):
            with self.assertRaises(ValueError):
                hp.configure_mpi_tcp_interface(bad)
        current = {k: os.environ.get(k) for k in hp.MPI_ENV_KEYS}
        other = 'en9' if current.get('FI_TCP_IFACE') != 'en9' else 'en8'
        with self.assertRaises(RuntimeError):
            hp.configure_mpi_tcp_interface(other)
        self.assertEqual({k: os.environ.get(k) for k in hp.MPI_ENV_KEYS}, current)
        runtime = self.run_adapter().record['runtime']
        self.assertEqual(runtime['mpi_runtime_env'], current)
        if current == {'FI_PROVIDER': 'tcp', 'FI_TCP_IFACE': 'en0'}:
            s = script.synthetic_contributions()
            s.pop('expected_unreferenced_uV')
            np.savez(self.tmp / 'b.npz', **s)
            (self.tmp / 'm.json').write_text('{}')
            script.main(['--mpi-tcp-interface', 'en0', 'aggregate', '--bundle', str(self.tmp / 'b.npz'),
                         '--metadata', str(self.tmp / 'm.json'), '--output-dir', str(self.tmp / 'en0')])
            record = json.loads((self.tmp / 'en0/postprocess_record.json').read_text())
            self.assertIn('FI_TCP_IFACE=en0', record['runtime']['mpi_interface_selection'])
            hp._mpi_interface_request = None

    def test_rejects_modified_upstream_copy(self):
        tampered = self.tmp / 'postproc.py'
        tampered.write_bytes(hp.UPSTREAM_POSTPROC.read_bytes() + b'\n')
        with self.assertRaises(ValueError):
            hp.load_upstream_postprocess(tampered)
        self.assert_rejected(upstream_path=tampered)

    def test_cli_bundle_round_trip_and_bundle_validation(self):
        s = script.synthetic_contributions()
        expected = s.pop('expected_unreferenced_uV')
        bundle, meta = self.tmp / 'bundle.npz', self.tmp / 'meta.json'
        np.savez(bundle, **s)
        meta.write_text(json.dumps({'split_group': 'synthetic'}))
        script.main(['aggregate', '--bundle', str(bundle), '--metadata', str(meta),
                     '--output-dir', str(self.tmp / 'cli')])
        record = json.loads((self.tmp / 'cli/postprocess_record.json').read_text())
        self.assertEqual(record['provenance']['bundle_sha256'], script.sha(bundle))
        self.assertEqual(record['provenance']['metadata']['split_group'], 'synthetic')
        import h5py
        with h5py.File(self.tmp / 'cli/summed_potential_uV.h5', 'r') as f:
            np.testing.assert_allclose(f['unreferenced_uV'][()], expected, rtol=0, atol=1e-12)

        ids = dict(s, cell_ids=np.array([11, 12, 13]))
        del ids['labels']
        np.savez(self.tmp / 'ids.npz', **ids)
        result = script.aggregate_bundle(self.tmp / 'ids.npz', meta, self.tmp / 'ids_out')
        self.assertEqual(result.labels, ('cell_11', 'cell_12', 'cell_13'))

        duplicate = dict(s, cell_ids=np.array([11, 11, 13]))
        np.savez(self.tmp / 'dup.npz', **duplicate)
        with self.assertRaisesRegex(ValueError, 'Duplicate cell_ids'):
            script.aggregate_bundle(self.tmp / 'dup.npz', meta, self.tmp / 'dup_out')
        self.assertFalse((self.tmp / 'dup_out').exists())

        for name, change in {'extra': {'per_cell_potential_mV': s['per_cell_potential_uV']},
                             'units': {'units': np.array('mV')},
                             'object': {'labels': np.array(['a', 'b', 'c'], dtype=object)}}.items():
            np.savez(self.tmp / f'{name}.npz', **dict(s, **change))
            with self.assertRaises(ValueError):
                script.aggregate_bundle(self.tmp / f'{name}.npz', meta, self.tmp / f'{name}_out')
            self.assertFalse((self.tmp / f'{name}_out').exists())


if __name__ == '__main__':
    unittest.main()
