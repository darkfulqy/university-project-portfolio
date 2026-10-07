#!/usr/bin/env python3
"""Hand a PASSED population observer run to the unchanged hybridLFPy PostProcess.

Run in .venv-hybrid-postproc (numpy, h5py, mpi4py; NEURON is never imported).
For each variant the saved raw signed per-cell potentials are exported to an
interchange bundle, the reviewed adapter scripts/aggregate_hybrid_population.py
runs as a separate process with explicit argv, exit status 0 is required, and
the upstream HDF5 totals are compared with the saved population totals. The
completed membrane run directory is only read; everything is written to a new
output directory, including failures.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO
sys.path.insert(0, str(REPO / 'src'))
import numpy as np
from flyobserver import population as pop
from flyobserver import hybrid_postprocess as hp
from flyobserver.population import GateFailure

ADAPTER = REPO / 'scripts/aggregate_hybrid_population.py'
VARIANTS = ['base', 'blank', 'half_dt', 'half_segment', 'radius_half', 'radius_double']
CODE = ['scripts/postprocess_population_hybrid.py', 'scripts/aggregate_hybrid_population.py',
        'src/flyobserver/hybrid_postprocess.py', 'src/flyobserver/population.py']
SPECTRAL_IDENTITY_TOLERANCE = 1e-9


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n')


def require(condition, message):
    if not condition:
        raise GateFailure(message)


def verify_run(run_dir):
    """Check a completed run's status, gates, hashes, inputs and common grid."""
    run_dir = Path(run_dir)
    status = json.loads((run_dir / 'status.json').read_text())
    require(status.get('state') == 'passed', f'Run state is {status.get("state")}, not passed')
    summary = json.loads((run_dir / 'summary.json').read_text())
    checks = summary.get('checks_passed', {})
    require(bool(checks) and all(checks.values()), 'Run summary gates are not all passed')
    snapshot = run_dir / 'code_snapshot'
    config = json.loads((snapshot / 'config.json').read_text())
    contacts_path = {int(c['body_id']): c['contacts'] for c in config['cells']}
    common = None; variants = {}
    for name in VARIANTS:
        vdir = run_dir / 'variants' / name
        meta = json.loads((vdir / 'metadata.json').read_text())
        require(sha(vdir / 'population.npz') == meta['population_npz_sha256'], f'{name}: population.npz hash mismatch')
        require(sha(snapshot / 'config.json') == meta['config_sha256'], f'{name}: config snapshot hash mismatch')
        for rel, digest in meta['source_files_sha256'].items():
            require(sha(snapshot / rel.replace('/', '__')) == digest, f'{name}: code snapshot mismatch for {rel}')
        for body, d in meta['diagnostics'].items():
            if body != 'population':
                require(sha(vdir / 'cells' / f'{body}.npz') == d['file_sha256'], f'{name}: cell {body} file hash mismatch')
        require(meta['empirical_validation'] is False and meta['whole_CNS_LFP'] is False, f'{name}: validation labels changed')
        require(meta['trajectory'] is None and meta['movement_label'] is None, f'{name}: behavior labels present')
        require(meta['split_group'] == summary['split_group'], f'{name}: split group mismatch')
        require(str(meta['units']['extracellular_voltage']).startswith('uV'), f'{name}: potential units are not uV')
        with np.load(vdir / 'population.npz', allow_pickle=False) as z:
            t = z['time_s']; xyz = z['electrodes_um']
            ids = z['cell_body_ids'].tolist(); labels = [str(x) for x in z['cell_labels']]
            ref = int(z['reference_channel_zero_based'])
            stack_shape = z['per_cell_unreferenced_potential_uV'].shape
        axis = pop.validate_time_axis(t, meta['sampling_rate_hz'], meta['stimulus']['duration_s'])
        require(hashlib.sha256(xyz.tobytes()).hexdigest() == meta['observer_geometry']['electrode_array_sha256'],
                f'{name}: electrode array differs from recorded common geometry')
        require(ref == meta['reference_channel_zero_based'], f'{name}: reference index mismatch')
        require(stack_shape == (len(ids), len(t), len(xyz)), f'{name}: per-cell potential shape mismatch')
        require(sha(ROOT / meta['source_trial']) == meta['source_trial_sha256'], f'{name}: source trial changed')
        for cell in meta['cells']:
            require(sha(ROOT / cell['swc']) == cell['swc_sha256'], f'{name}: SWC changed for {cell["body_id"]}')
            require(sha(ROOT / contacts_path[int(cell['body_id'])]) == cell['contacts_sha256'],
                    f'{name}: contacts changed for {cell["body_id"]}')
        key = {'time_s_sha256': hashlib.sha256(t.tobytes()).hexdigest(),
               'electrode_array_sha256': meta['observer_geometry']['electrode_array_sha256'],
               'cell_body_ids': ids, 'cell_labels': labels, 'reference_index': ref,
               'split_group': meta['split_group'], 'source_trial_sha256': meta['source_trial_sha256'],
               'cells': meta['cells'], 'sampling_rate_hz': meta['sampling_rate_hz'], 'stimulus': meta['stimulus']}
        if common is None:
            common = key
        require(key == common, f'{name}: cells, geometry, time axis, reference or source differ between variants')
        variants[name] = {'population_npz_sha256': meta['population_npz_sha256'],
                          'metadata_sha256': sha(vdir / 'metadata.json'), 'time_axis': axis}
    return {'run_dir': str(run_dir), 'run_id': summary['run_id'], 'status_sha256': sha(run_dir / 'status.json'),
            'summary_sha256': sha(run_dir / 'summary.json'), 'config_snapshot_sha256': sha(snapshot / 'config.json'),
            'common': common, 'variants': variants, 'inputs_rehashed_unchanged': True}


def export_bundle(data, sampling_rate_hz, path):
    """Interchange bundle with the original raw signed per-cell potentials."""
    np.savez(path, time_s=data['time_s'], electrodes_um=data['electrodes_um'],
             per_cell_potential_uV=data['per_cell_unreferenced_potential_uV'],
             reference_index=np.int64(data['reference_channel_zero_based']),
             cell_ids=np.asarray(data['cell_body_ids'], dtype=np.int64),
             labels=np.asarray([str(x) for x in data['cell_labels']], dtype=str),
             units=np.array('uV'), sample_rate_hz=np.float64(sampling_rate_hz))
    return sha(path)


def compare_upstream(upstream_dir, data, meta, bundle_sha, metadata_sha, interface):
    """Compare actual upstream HDF5 totals with saved population arrays."""
    import h5py
    record = json.loads((upstream_dir / 'postprocess_record.json').read_text())
    with h5py.File(upstream_dir / 'summed_potential_uV.h5', 'r') as f:
        unref = f['unreferenced_uV'][()]; referenced = f['referenced_uV'][()]
        labels = [x.decode() if isinstance(x, bytes) else str(x) for x in f['labels'][()]]
        t = f['time_s'][()]; xyz = f['electrodes_um'][()]; ref = int(f['reference_index'][()])
    stack = data['per_cell_unreferenced_potential_uV']
    baseline = np.asarray(data['baseline_mask'], bool)
    n = stack.shape[0]; eps = np.finfo(float).eps
    factor = (2 * n + 4) * eps
    magnitude = np.abs(stack).sum(axis=0)
    mag_ref = magnitude + magnitude[:, [ref]]
    # Unreferenced/referenced: the adapter's own documented float bound.
    e_unref = np.abs(unref - data['total_unreferenced_potential_uV'])
    e_ref = np.abs(referenced - data['total_referenced_potential_uV'])
    # Baseline: the independently applied mean over k samples adds at most about
    # k*eps of the largest baseline magnitude on each side, on top of the referenced bound.
    k = int(baseline.sum())
    _, corrected = pop.reference_and_baseline(unref, ref, baseline)
    e_base = np.abs(corrected - data['total_baseline_corrected_referenced_potential_uV'])
    base_bound = (factor + 2 * k * eps) * (mag_ref + mag_ref[baseline].max(axis=0, keepdims=True))
    onset = meta['stimulus']['onset_s']; offset = meta['stimulus']['offset_s']
    stimulus = (t >= onset) & (t < offset)
    spectra = pop.spectral_decomposition(data['per_cell_baseline_corrected_referenced_potential_uV'][:, stimulus],
                                         meta['sampling_rate_hz'], total=corrected[stimulus])
    prov = record['provenance']
    checks = {
        'labels_match': labels == [str(x) for x in data['cell_labels']],
        'time_axis_identical': bool(np.array_equal(t, data['time_s'])),
        'electrodes_identical': bool(np.array_equal(xyz, data['electrodes_um'])),
        'reference_index_matches': ref == int(data['reference_channel_zero_based']),
        'unreferenced_total_within_bound': bool(np.all(e_unref <= factor * magnitude)),
        'referenced_total_within_bound': bool(np.all(e_ref <= factor * mag_ref)),
        'upstream_reference_channel_zero': float(np.max(np.abs(referenced[:, ref]))) == 0.0,
        'independent_baseline_within_bound': bool(np.all(e_base <= base_bound)),
        'spectral_identity': spectra['identity_relative_max_error'] <= SPECTRAL_IDENTITY_TOLERANCE,
        'upstream_input_hdf5_unchanged': record['checks']['input_hdf5_unchanged'] is True,
        'upstream_contributions_round_trip_exact': record['checks']['contributions_round_trip_exact'] is True,
        'upstream_no_per_cell_normalization': record['checks']['per_cell_normalization'] is False,
        'upstream_module_hash': record['upstream']['module_sha256'] == hp.UPSTREAM_POSTPROC_SHA256,
        'upstream_commit': record['upstream']['commit'] == hp.UPSTREAM_COMMIT,
        'upstream_methods': record['upstream']['methods_invoked'] == ['PostProcess.__init__', 'PostProcess.calc_measure'],
        'upstream_serial_mpi': record['runtime']['mpi_comm_world_size'] == 1,
        'upstream_no_neuron_import': record['runtime']['neuron_imported'] is False,
        'mpi_interface_as_requested': (interface is None or
                                       record['runtime']['mpi_runtime_env'].get('FI_TCP_IFACE') == interface),
        'record_bundle_hash_matches': prov['bundle_sha256'] == bundle_sha,
        'record_metadata_hash_matches': prov['metadata_sha256'] == metadata_sha,
    }
    scale = max(float(magnitude.max()), np.finfo(float).tiny)
    files = {str(p.relative_to(upstream_dir)): sha(p) for p in sorted(upstream_dir.rglob('*')) if p.is_file()}
    detail = {'checks': {k: bool(v) for k, v in checks.items()},
              'max_abs_error_uV': {'unreferenced': float(e_unref.max()), 'referenced': float(e_ref.max()),
                                   'baseline_corrected': float(e_base.max())},
              'max_error_relative_to_max_sum_abs_contribution': {'unreferenced': float(e_unref.max()) / scale,
                                                                 'referenced': float(e_ref.max()) / scale,
                                                                 'baseline_corrected': float(e_base.max()) / scale},
              'bitwise_equal': {'unreferenced': bool(np.array_equal(unref, data['total_unreferenced_potential_uV'])),
                                'referenced': bool(np.array_equal(referenced, data['total_referenced_potential_uV']))},
              'tolerance_rules': {'unreferenced': '(2n+4)*eps*sum_i|phi_i|',
                                  'referenced': '(2n+4)*eps*(sum_i|phi_i|[ch] + sum_i|phi_i|[ref])',
                                  'baseline_corrected': '((2n+4)+2k)*eps*(S_ref[t,ch] + max_baseline S_ref[ch]), k baseline samples'},
              'spectral_identity_relative_max_error': spectra['identity_relative_max_error'],
              'upstream_record': {'upstream': record['upstream'], 'runtime': record['runtime'], 'checks': record['checks'],
                                  'outputs': record['outputs'], 'inputs': record['inputs']},
              'upstream_files_sha256': files}
    return detail, {'unreferenced_uV': unref, 'referenced_uV': referenced, 'baseline_corrected_referenced_uV': corrected}


def run(run_dir, output_dir, interface='en0', python=None):
    run_dir = Path(run_dir); out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    frozen = {p: (REPO / p).read_bytes() for p in CODE}
    snapshot = out / 'code_snapshot'; snapshot.mkdir()
    for p, data in frozen.items():
        (snapshot / p.replace('/', '__')).write_bytes(data)
    status = {'started_at': datetime.now(timezone.utc).isoformat(), 'state': 'running', 'run_dir': str(run_dir)}
    dump(out / 'status.json', status)
    results = {}
    try:
        verification = verify_run(run_dir)
        for sub in ['bundles', 'upstream', 'logs']:
            (out / sub).mkdir()
        totals = {}
        python = python or sys.executable
        for name in VARIANTS:
            vdir = run_dir / 'variants' / name
            meta = json.loads((vdir / 'metadata.json').read_text())
            with np.load(vdir / 'population.npz', allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            bdir = out / 'bundles' / name; bdir.mkdir()
            bundle_sha = export_bundle(data, meta['sampling_rate_hz'], bdir / 'bundle.npz')
            attached = {k: v for k, v in meta.items() if k != 'diagnostics'}
            attached.update(population_run_dir=str(run_dir), population_npz_sha256=verification['variants'][name]['population_npz_sha256'],
                            population_metadata_sha256=verification['variants'][name]['metadata_sha256'],
                            population_status_sha256=verification['status_sha256'],
                            population_summary_sha256=verification['summary_sha256'],
                            bundle_potentials='per_cell_unreferenced_potential_uV: raw signed, unscaled')
            dump(bdir / 'metadata.json', attached)
            metadata_sha = sha(bdir / 'metadata.json')
            argv = [python, str(ADAPTER)] + (['--mpi-tcp-interface', interface] if interface else []) + \
                   ['aggregate', '--bundle', str(bdir / 'bundle.npz'), '--metadata', str(bdir / 'metadata.json'),
                    '--output-dir', str(out / 'upstream' / name)]
            tick = time.perf_counter()
            proc = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)
            (out / 'logs' / f'{name}.stdout.txt').write_text(proc.stdout)
            (out / 'logs' / f'{name}.stderr.txt').write_text(proc.stderr)
            results[name] = {'argv': argv, 'returncode': proc.returncode, 'wall_time_s': time.perf_counter() - tick,
                             'bundle_sha256': bundle_sha, 'metadata_sha256': metadata_sha,
                             'stdout_sha256': sha(out / 'logs' / f'{name}.stdout.txt'),
                             'stderr_sha256': sha(out / 'logs' / f'{name}.stderr.txt')}
            print(json.dumps({'variant': name, 'returncode': proc.returncode}), flush=True)
            if proc.returncode != 0:
                raise GateFailure(f'Adapter exited with status {proc.returncode} for {name}; see logs/{name}.stderr.txt')
            detail, arrays = compare_upstream(out / 'upstream' / name, data, meta, bundle_sha, metadata_sha, interface)
            results[name].update(detail)
            for key, value in arrays.items():
                totals[f'{name}__upstream_total_{key}'] = value
        with np.load(run_dir / 'variants/base/population.npz', allow_pickle=False) as z:
            np.savez_compressed(out / 'hybrid_totals.npz', **totals, time_s=z['time_s'], electrodes_um=z['electrodes_um'],
                                reference_channel_zero_based=z['reference_channel_zero_based'],
                                baseline_mask=z['baseline_mask'], cell_labels=z['cell_labels'], cell_body_ids=z['cell_body_ids'])
        checks = {f'{name}/{k}': v for name in VARIANTS for k, v in results[name]['checks'].items()}
        checks.update({f'{name}/adapter_exit_0': results[name]['returncode'] == 0 for name in VARIANTS})
        checks['code_unchanged'] = all((REPO / p).read_bytes() == d for p, d in frozen.items())
        checks['population_run_unchanged'] = verify_run(run_dir) == verification
        summary = {
            'label': 'actual upstream hybridLFPy PostProcess totals of saved raw signed per-cell potentials; model-derived uV; not whole-CNS LFP; not empirically validated',
            'upstream': {'repository': hp.UPSTREAM_REPOSITORY, 'commit': hp.UPSTREAM_COMMIT,
                         'module_sha256': hp.UPSTREAM_POSTPROC_SHA256,
                         'methods_invoked': ['PostProcess.__init__', 'PostProcess.calc_measure']},
            'adapter_python': python, 'mpi_tcp_interface': interface,
            'population_run': verification, 'variants': results, 'checks_passed': checks,
            'hybrid_totals_npz_sha256': sha(out / 'hybrid_totals.npz'),
            'code_sha256': {p: hashlib.sha256(d).hexdigest() for p, d in frozen.items()},
            'empirical_validation': False, 'whole_CNS_LFP': False}
        dump(out / 'hybrid_summary.json', summary)
        state = 'passed' if all(checks.values()) else 'failed_gates'
        status.update(state=state, finished_at=datetime.now(timezone.utc).isoformat(),
                      failed_checks=[k for k, v in checks.items() if not v])
        dump(out / 'status.json', status)
        if state != 'passed':
            raise GateFailure(f'Upstream comparison gates failed: {status["failed_checks"]}')
        return summary
    except BaseException as error:
        if status['state'] == 'running':
            dump(out / 'partial_results.json', results)
            status.update(state='failed_gates' if isinstance(error, GateFailure) else 'error',
                          finished_at=datetime.now(timezone.utc).isoformat(), message=str(error),
                          traceback=traceback.format_exc())
            dump(out / 'status.json', status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', required=True, help='passed population run directory')
    parser.add_argument('--output-dir', required=True, help='new directory for bundles, upstream outputs and summary')
    parser.add_argument('--mpi-tcp-interface', default='en0')
    args = parser.parse_args()
    try:
        summary = run(args.run, args.output_dir, args.mpi_tcp_interface)
    except GateFailure as error:
        print(f'GATE FAILURE: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
    print(json.dumps({'state': 'passed', 'checks': len(summary['checks_passed'])}, indent=2))


if __name__ == '__main__':
    main()
