#!/usr/bin/env python3
"""Run the complete retained Male CNS graph with retinal-terminal stimulation.

Outputs are population activity in a.u. They are not predicted LFP or EEG.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy import sparse

from flyvision.dynamics import ContinuousCNS, flash_train, parameters


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def make_observer(groups, n):
    rows, cols, values, names = [], [], [], []
    for group in groups:
        indices = np.asarray(group['indices'], dtype=int)
        if not len(indices):
            continue
        row = len(names)
        names.append(group['name'])
        rows.extend([row] * len(indices)); cols.extend(indices)
        values.extend([1 / len(indices)] * len(indices))
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(names), n)), names


def run(args):
    source_paths = ['src/flyvision/dynamics.py', 'src/flyvision/connectome.py',
                    'scripts/run_visual_full_cns.py', 'configs/visual_full_cns.json',
                    'requirements-sim.lock.txt']
    # Freeze actual bytes before loading data or starting integration.
    source_bytes = {rel: (ROOT/rel).read_bytes() for rel in source_paths}
    cfg = json.loads(source_bytes['configs/visual_full_cns.json'])
    graph_dir = ROOT / cfg['graph']
    graph_hashes = {name: sha(graph_dir/name) for name in
                    ['W_normalized.npz','arrays.npz','nodes.feather','groups.json','manifest.json']}
    manifest = json.loads((graph_dir / 'manifest.json').read_text())
    arrays = np.load(graph_dir / 'arrays.npz', allow_pickle=False)
    ids = arrays['ids']
    indices = arrays['input_indices']
    groups = json.loads((graph_dir / 'groups.json').read_text())
    if isinstance(groups, dict):
        groups = groups['groups']
    observer, names = make_observer(groups, len(ids))
    w = sparse.load_npz(graph_dir / 'W_normalized.npz').tocsr()
    if w.shape != (len(ids), len(ids)):
        raise ValueError('graph/ID dimension mismatch')
    cfg['parameter_seed'] = args.seed
    cfg['split_group'] = f'male_cns_v1_0_visual_source_{args.seed}'
    if args.short:
        cfg['stimulus'].update(onset_s=.2, offset_s=2.2, duration_s=2.4)
    dt = cfg['dynamics']['dt_s'] if args.dt is None else args.dt
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    cfg['dynamics']['dt_s'] = dt
    cfg['dynamics']['output_every'] = int(round(.001 / dt))
    if cfg['dynamics']['output_every'] < 1 or not np.isclose(cfg['dynamics']['output_every'] * dt, .001):
        raise ValueError('dt must divide the 1 ms output interval')
    stim_cfg = cfg['stimulus']
    stimulus = flash_train(stim_cfg['duration_s'], dt, stim_cfg['onset_s'],
        stim_cfg['offset_s'], stim_cfg['frequency_hz'], stim_cfg['duty_cycle'])
    if args.condition == 'blank':
        stimulus[:] = 0
    elif args.condition == 'visual_output_block':
        # Close outgoing transmission from driven cells, retaining their drive
        # and all neurons. Do not renormalize remaining connections.
        scale = np.ones(len(ids)); scale[indices] = 0
        w.data *= scale[w.indices]
        w.eliminate_zeros()
    run_name = f'{args.condition}_seed_{args.seed}_{"short" if args.short else "packet"}_dt_{dt:g}'
    output = ROOT / cfg['output'] / run_name
    if (output / 'trial.npz').exists() and not args.overwrite:
        raise FileExistsError(f'Existing trial; use a new run or explicit --overwrite: {output}')
    p = parameters(len(ids), args.seed, cfg['dynamics']['coupling'],
                   tau_range_s=cfg['dynamics']['tau_uniform_s'],
                   tonic_range_au=cfg['dynamics']['tonic_input_uniform_au'])
    model = ContinuousCNS(w, p)
    print(json.dumps({'condition':args.condition, 'n_cells':len(ids), 'n_edges':w.nnz,
                      'n_input_cells':len(indices), 'duration_s':stim_cfg['duration_s'], 'dt_s':dt}), flush=True)
    result = model.simulate(stimulus, dt, indices, observer,
        amplitude=stim_cfg['amplitude_au'], output_every=cfg['dynamics']['output_every'],
        progress=lambda message: print(message, flush=True))
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = result.pop('diagnostics')
    result['stimulus_interval_gate'] = stimulus
    result['stimulus_interval_drive_au'] = stimulus * stim_cfg['amplitude_au']
    result['group_names'] = np.asarray(names)
    result['input_cell_ids'] = ids[indices]
    result['all_cell_ids'] = ids
    np.savez_compressed(output / 'trial.npz', **result)
    np.savez_compressed(output / 'parameters.npz', tau_s=p.tau_s, tonic_input_au=p.tonic_input)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    sources = {}
    for rel, frozen_bytes in source_bytes.items():
        dest = snapshot / rel.replace('/', '__')
        dest.write_bytes(frozen_bytes)
        sources[rel] = {'sha256':hashlib.sha256(frozen_bytes).hexdigest(),
                        'snapshot':str(dest.relative_to(ROOT)),
                        'unchanged_during_run':(ROOT/rel).read_bytes() == frozen_bytes}
    graph_checks = {name: {'sha256_at_start':digest,
                           'unchanged_during_run':sha(graph_dir/name) == digest}
                    for name,digest in graph_hashes.items()}
    if not all(x['unchanged_during_run'] for x in graph_checks.values()):
        raise RuntimeError('Graph artifacts changed during the run; result is not validated')
    metadata = {
        'created_at':datetime.now(timezone.utc).isoformat(), 'trial_id':run_name,
        'condition':args.condition, 'signal_kind':'population_activity_proxy',
        'units':'arbitrary_units', 'sampling_rate_hz':1000,
        'stimulus_input_sampling_rate_hz':1/dt,
        'state_sampling_note':'state includes endpoints; input describes left-held intervals',
        'stimulus_array_semantics':'gate is 0/1; drive_au includes the configured terminal amplitude',
        'dataset':cfg['dataset'], 'source_model_seed':args.seed,
        'parameter_ensemble':{'file':'parameters.npz','sha256':sha(output/'parameters.npz'),
                              'coupling':p.coupling},
        'configuration':cfg, 'source_files':sources,
        'graph_artifact_hashes':graph_checks,
        'connectome_manifest':{'path':str(graph_dir.relative_to(ROOT)/'manifest.json'),
                               'sha256':sha(graph_dir/'manifest.json'),
                               'counts':manifest.get('counts',{})},
        'cell_identifiers':'trial.npz:all_cell_ids; input_cell_ids',
        'observer_geometry':None,
        'observer_definition':'arithmetic means of explicitly listed anatomical cell groups',
        'observer_groups':{'path':str((graph_dir/'groups.json').relative_to(ROOT)),
                           'sha256':sha(graph_dir/'groups.json')},
        'reference':'each population minus its own prestimulus mean for plots only',
        'trajectory':None, 'movement_label':None,
        'behavior_missing_reason':'no body model run; stimulus is not a movement label',
        'nuisance_parameters':{'measurement_noise':None,'spontaneous_noise':None,
                               'electrode_reference':None,'adaptation':None},
        'split_group':cfg['split_group'],
        'parent_source_group':cfg['split_group'],
        'analysis_role':'engineering_development_check_no_empirical_fit_or_test_set',
        'lfp_forward_model':False,'empirically_validated':False,
        'diagnostics':diagnostics,
        'outputs':{'trial.npz':sha(output/'trial.npz')},
    }
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'output':str(output), 'diagnostics':diagnostics}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=['flash','blank','visual_output_block'], default='flash')
    parser.add_argument('--seed', type=int, default=211)
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--dt', type=float)
    parser.add_argument('--overwrite', action='store_true')
    run(parser.parse_args())
