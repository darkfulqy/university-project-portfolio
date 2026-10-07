#!/usr/bin/env python3
"""Replay frozen full-CNS activity into several real fly cable cells and sum
their signed extracellular contributions on one shared electrode array.

Cells are constructed, run and closed serially because NEURON state is global.
Each cell keeps its own SWC, contacts and full transmembrane current. Potentials
are summed without scaling, then referenced and baseline-corrected. Output is a
model-derived contribution of the selected cells only.
"""
import argparse
import hashlib
import json
import sys
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pyarrow.feather as feather
from flyobserver.cable import PassiveFlyCable, CableParameters, live_section_names
from flyobserver import population as pop
from flyobserver.population import GateFailure

CODE = ['src/flyobserver/cable.py', 'src/flyobserver/population.py',
        'scripts/run_population_observer.py', 'requirements-observer.lock.txt']
UNITS = {'current': 'nA', 'intracellular_voltage': 'mV', 'extracellular_voltage': 'uV (model-derived)',
         'conductance': 'uS', 'geometry': 'um', 'lead_field': 'mV/nA', 'activity': 'arbitrary_units',
         'power_spectral_density': 'uV^2/Hz (model-derived)'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n')


def variant_specs(cfg):
    p = CableParameters(**cfg['cable_parameters']); dt = cfg['integration']['dt_ms']
    specs = {'base': (p, dt, False), 'blank': (p, dt, True), 'half_dt': (p, dt / 2, False),
             'half_segment': (replace(p, max_segment_um=p.max_segment_um / 2), dt, False),
             'radius_half': (replace(p, radius_scale=p.radius_scale * .5), dt, False),
             'radius_double': (replace(p, radius_scale=p.radius_scale * 2), dt, False)}
    if list(specs) != cfg['variants'] or not set(cfg['geometry_preflight_variants']) <= set(specs):
        raise GateFailure('Configured variants differ from the implemented variant set')
    return specs


def software_record(cfg):
    import neuron, lfpykit, scipy, pyarrow
    from lfpykit import lfpcalc, models
    record = {'python': sys.version, 'neuron': neuron.__version__, 'lfpykit': lfpykit.__version__,
              'numpy': np.__version__, 'scipy': scipy.__version__, 'pyarrow': pyarrow.__version__,
              'lfpykit_lfpcalc_sha256': sha(lfpcalc.__file__), 'lfpykit_models_sha256': sha(models.__file__),
              'manifests': {}, 'license_hashes_match': True, 'hybridlfpy': cfg['hybridlfpy']}
    for rel in cfg['software_manifests']:
        content = json.loads((ROOT / rel).read_text())
        record['manifests'][rel] = {'sha256': sha(ROOT / rel), 'content': content}
        entries = content.values() if isinstance(content, dict) else content
        for entry in entries:
            for lic in (entry.get('licenses', []) if isinstance(entry, dict) else []):
                path = ROOT / lic['path']
                record['license_hashes_match'] &= path.exists() and sha(path) == lic['sha256']
    return record


def load_inputs(cfg):
    source_trial = ROOT / cfg['source_output'] / 'trial.npz'
    source_meta_path = ROOT / cfg['source_output'] / 'metadata.json'
    meta = json.loads(source_meta_path.read_text())
    hashes = {'source_trial': sha(source_trial), 'source_metadata': sha(source_meta_path)}
    if hashes['source_trial'] != meta['trial_sha256']:
        raise GateFailure('Population source trial hash differs from its metadata')
    if meta.get('status') != 'passed' or not all(meta['checks_passed'].values()):
        raise GateFailure('Population source did not pass its preservation gates')
    if (meta['source_model_seed'] != cfg['source_model_seed'] or meta['split_group'] != cfg['split_group']
            or meta['units'] != 'arbitrary_units' or meta['signal_kind'] != 'individual_continuous_activity'):
        raise GateFailure('Source seed, split group, units or signal kind mismatch')
    src = np.load(source_trial)
    time_s = src['time_s']
    time_check = pop.validate_time_axis(time_s, meta['source_sampling_rate_hz'], meta['stimulus']['duration_s'])
    activity = src['individual_activity_au']; ids = src['recorded_cell_ids']
    gate = src['stimulus_interval_gate']
    if (activity.shape != (len(time_s), len(ids)) or not np.isfinite(activity).all() or np.any(activity < 0)
            or gate.shape != (len(time_s) - 1,)):
        raise GateFailure('Source activity/gate shape, finiteness or nonnegativity check failed')
    contacts_manifest = json.loads((ROOT / cfg['contacts_manifest']).read_text())
    if not contacts_manifest['all_checks_pass']:
        raise GateFailure('Contact extraction did not pass its gates')
    hashes['contacts_manifest'] = sha(ROOT / cfg['contacts_manifest'])
    cells = []
    required = []
    for spec in cfg['cells']:
        body = int(spec['body_id'])
        contacts_path = ROOT / spec['contacts']
        record = contacts_manifest['targets'][str(body)]
        digest = sha(contacts_path)
        if digest != record['output']['sha256']:
            raise GateFailure(f'Contacts for {body} differ from extraction manifest')
        table = feather.read_table(contacts_path)
        retained = table.filter(table['included_in_unsigned_graph'])
        body_post = table['body_post'].to_numpy()
        signs = retained['pre_model_output_sign'].to_numpy().astype(int)
        if np.any(body_post != body) or not set(np.unique(signs)) <= {-1, 0, 1}:
            raise GateFailure(f'Contact table for {body} has foreign rows or invalid signs')
        pre = retained['body_pre'].to_numpy()
        required.extend(pre.tolist() + [body])
        xyz = np.column_stack([retained[k].to_numpy() for k in ['x_post_um', 'y_post_um', 'z_post_um']])
        cells.append({'body_id': body, 'cell_type': spec['cell_type'], 'instance': spec['instance'],
                      'swc': ROOT / spec['swc'], 'body_pre': pre, 'xyz': xyz, 'signs': signs,
                      'raw_contacts': table.num_rows, 'retained_contacts': retained.num_rows,
                      'enabled_contacts': int(np.count_nonzero(signs))})
        hashes[f'swc_{body}'] = sha(ROOT / spec['swc'])
        hashes[f'contacts_{body}'] = digest
    lookup = pop.column_lookup(ids, required)
    for cell in cells:
        cell['columns'] = np.array([lookup[int(b)] for b in cell['body_pre']], dtype=np.int64)
    electrode_path = ROOT / cfg['observer']['electrodes_source']
    hashes['electrodes_source'] = sha(electrode_path)
    if hashes['electrodes_source'] != cfg['observer']['electrodes_source_sha256']:
        raise GateFailure('Electrode source file differs from configured hash')
    electrodes = np.array(np.load(electrode_path)[cfg['observer']['electrodes_key']], dtype=float)
    if electrodes.ndim != 2 or electrodes.shape[1] != 3 or not np.isfinite(electrodes).all():
        raise GateFailure('Electrode array must be finite (n, 3)')
    if not 0 <= cfg['observer']['reference_channel_zero_based'] < len(electrodes):
        raise GateFailure('Reference index outside electrode array')
    reg = cfg['regression']
    reg_meta = json.loads((ROOT / reg['v1_base_metadata']).read_text())
    hashes['regression_trial'] = sha(ROOT / reg['v1_base_trial'])
    if hashes['regression_trial'] != reg_meta['trial_sha256']:
        raise GateFailure('Regression trial hash differs from its metadata')
    return {'meta': meta, 'time_s': time_s, 'time_check': time_check, 'activity': activity, 'ids': ids,
            'gate': gate, 'cells': cells, 'electrodes': electrodes,
            'electrode_sha256': hashlib.sha256(electrodes.tobytes()).hexdigest(), 'hashes': hashes,
            'source_dt_ms': 1000 / meta['source_sampling_rate_hz']}


def closed_after(cell):
    cell.close()
    leftover = live_section_names()
    if leftover:
        raise GateFailure(f'NEURON sections remained after closing a cell: {leftover[:5]}')


def geometry_preflight(cfg, inputs, specs):
    records = {}
    for cell in inputs['cells']:
        records[str(cell['body_id'])] = {}
        for name in cfg['geometry_preflight_variants']:
            c = PassiveFlyCable(cell['swc'], specs[name][0])
            try:
                records[str(cell['body_id'])][name] = pop.geometry_gate(
                    c.starts, c.ends, c.diameters_um, inputs['electrodes'],
                    cfg['observer']['minimum_centerline_clearance_um'])
            finally:
                closed_after(c)
    passed = all(v['passed'] for r in records.values() for v in r.values())
    return {'passed': passed, 'electrode_array_sha256': inputs['electrode_sha256'],
            'electrodes_source': cfg['observer']['electrodes_source'],
            'rule': 'finite-centerline clearance >= minimum, no electrode inside segment radius, and zero installed LFPykit infinite-line radius cutoffs',
            'cells': records}


def replay_cell(cfg, inputs, cell, params, dt, blank, check_sigma):
    c = PassiveFlyCable(cell['swc'], params)
    try:
        c.map_contacts(cell['xyz'], cell['columns'], cell['signs'], len(inputs['ids']))
        geometry = pop.geometry_gate(c.starts, c.ends, c.diameters_um, inputs['electrodes'],
                                     cfg['observer']['minimum_centerline_clearance_um'])
        if not geometry['passed']:
            raise GateFailure(f'Geometry gate failed during replay for {cell["body_id"]}')
        activity = inputs['activity']
        drive = np.broadcast_to(activity[0], activity.shape) if blank else activity
        tick = time.perf_counter()
        result = c.run(drive, source_dt_ms=inputs['source_dt_ms'], dt_ms=dt, warmup_ms=cfg['integration']['warmup_ms'])
        matrix = c.lead_field(inputs['electrodes'])
        current = result['transmembrane_current_nA']
        potential = current @ matrix.T * 1000  # nA x mV/nA x 1000 uV/mV
        sigma = None
        if check_sigma:
            sigma = {}
            for s in cfg['validation']['conductivity_sensitivity_S_m']:
                scaled = c.lead_field(inputs['electrodes'], sigma=s)
                expected = matrix * params.sigma_S_m / s
                sigma[str(s)] = float(np.max(np.abs(scaled - expected)) / max(np.max(np.abs(expected)), np.finfo(float).tiny))
        arrays = dict(result, extracellular_potential_uV=potential, lead_field_mV_per_nA=matrix,
                      electrodes_um=inputs['electrodes'], segment_starts_um=c.starts, segment_ends_um=c.ends,
                      segment_diameter_um=c.diameters_um, segment_area_um2=c.area_um2,
                      segment_swc_node_id=c.segment_swc_node_id,
                      segment_swc_parent_node_id=c.segment_swc_parent_node_id,
                      segment_position_in_edge=c.segment_position_in_edge,
                      contact_body_pre=cell['body_pre'], contact_xyz_um=cell['xyz'], contact_sign=cell['signs'],
                      contact_segment_indices=c.contact_segment_indices,
                      contact_projection_distance_um=c.contact_projection_distance_um,
                      source_cell_ids=inputs['ids'], observer_cell_id=np.array([cell['body_id']]),
                      time_s=inputs['time_s'])
        diagnostics = {**pop.current_balance(current), 'n_segments': len(c.segments),
                       'area_um2': float(c.area_um2.sum()), 'wall_time_s': time.perf_counter() - tick,
                       'voltage_min_mV': float(result['membrane_voltage_mV'].min()),
                       'voltage_max_mV': float(result['membrane_voltage_mV'].max()),
                       'swc_root_node_id': c.swc_root_node_id, 'geometry': geometry,
                       'projection_distance_um_quantiles': np.quantile(c.contact_projection_distance_um, [0, .5, .95, 1]).tolist(),
                       'conductivity_scaling_relative_error': sigma}
        return arrays, diagnostics
    finally:
        closed_after(c)


def rel_error(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.shape != b.shape:
        return {'shape_match': False, 'relative_max_error': float('inf'), 'bit_exact': False}
    return {'shape_match': True, 'bit_exact': bool(np.array_equal(a, b)),
            'relative_max_error': float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), np.finfo(float).tiny))}


def execute(cfg, config_bytes, run_dir, frozen):
    specs = variant_specs(cfg)
    inputs = load_inputs(cfg)
    software = software_record(cfg)
    dump(run_dir / 'inputs.json', {'hashes': inputs['hashes'], 'time_axis': inputs['time_check'],
                                    'electrode_array_sha256': inputs['electrode_sha256'], 'software': software,
                                    'cells': [{k: v for k, v in c.items() if k in ('body_id', 'cell_type', 'instance', 'raw_contacts', 'retained_contacts', 'enabled_contacts')}
                                              for c in inputs['cells']]})
    preflight = geometry_preflight(cfg, inputs, specs)
    dump(run_dir / 'geometry_preflight.json', preflight)
    if not preflight['passed']:
        raise GateFailure('Common electrode geometry failed preflight; see geometry_preflight.json. '
                          'No per-cell recentering was attempted; a single shared replacement geometry needs review.')
    time_s = inputs['time_s']; meta = inputs['meta']
    reference = cfg['observer']['reference_channel_zero_based']
    baseline = time_s < meta['stimulus']['onset_s']
    stimulus = (time_s >= meta['stimulus']['onset_s']) & (time_s < meta['stimulus']['offset_s'])
    types = [c['cell_type'] for c in inputs['cells']]
    bodies = [c['body_id'] for c in inputs['cells']]
    labels = [f"{c['cell_type']}_{c['body_id']}" for c in inputs['cells']]
    val = cfg['validation']
    processed = {}; diag = {}; totals = {}
    for name, (params, dt, blank) in specs.items():
        vdir = run_dir / 'variants' / name / 'cells'; vdir.mkdir(parents=True)
        stack = []; diag[name] = {}
        for cell in inputs['cells']:
            arrays, d = replay_cell(cfg, inputs, cell, params, dt, blank, check_sigma=(name == 'base'))
            np.savez_compressed(vdir / f"{cell['body_id']}.npz", **arrays)
            d['file_sha256'] = sha(vdir / f"{cell['body_id']}.npz")
            diag[name][str(cell['body_id'])] = d
            stack.append(arrays['extracellular_potential_uV'])
            print(json.dumps({'variant': name, 'body_id': cell['body_id'], 'max_abs_current_balance_nA': d['max_abs_current_balance_nA'],
                              'wall_time_s': d['wall_time_s']}), flush=True)
            del arrays
        stack = np.stack(stack)
        agg = pop.aggregate_contributions(stack, labels, types)
        order = pop.summation_order_check(stack)
        cell_ref, cell_proc = pop.reference_and_baseline(stack, reference, baseline)
        total_ref, total_proc = pop.reference_and_baseline(agg['total'], reference, baseline)
        group_stack = np.stack([agg['groups'][g] for g in agg['group_order']])
        _, group_proc = pop.reference_and_baseline(group_stack, reference, baseline)
        scale = order['scale_sum_of_cell_max_abs_uV']
        # Recompute the total from saved per-cell currents and lead fields.
        recomputed = sum(np.load(vdir / f'{b}.npz')['transmembrane_current_nA'] @ np.load(vdir / f'{b}.npz')['lead_field_mV_per_nA'].T * 1000
                         for b in bodies)
        diag[name]['population'] = {
            'aggregator_relative_error': agg['aggregator_relative_error'],
            'group_sum_relative_error': agg['group_sum_relative_error'],
            'summation_order': order,
            'sum_identity_from_saved_files_relative_error': float(np.max(np.abs(recomputed - agg['total']))) / scale,
            'processed_linearity_relative_error': float(np.max(np.abs(total_proc - cell_proc.sum(axis=0)))) / scale,
            'reference_channel_max_abs_uV': float(np.max(np.abs(total_ref[:, reference]))),
        }
        pdir = run_dir / 'variants' / name
        np.savez_compressed(pdir / 'population.npz', time_s=time_s, stimulus_interval_gate=inputs['gate'],
                            cell_body_ids=np.array(bodies), cell_types=np.array(types), cell_labels=np.array(labels),
                            group_labels=np.array(agg['group_order']),
                            electrodes_um=inputs['electrodes'], reference_channel_zero_based=reference, baseline_mask=baseline,
                            per_cell_unreferenced_potential_uV=stack, per_cell_referenced_potential_uV=cell_ref,
                            per_cell_baseline_corrected_referenced_potential_uV=cell_proc,
                            group_unreferenced_potential_uV=group_stack,
                            group_baseline_corrected_referenced_potential_uV=group_proc,
                            total_unreferenced_potential_uV=agg['total'], total_referenced_potential_uV=total_ref,
                            total_baseline_corrected_referenced_potential_uV=total_proc)
        params_record = asdict(params)
        dump(pdir / 'metadata.json', {
            'created_at': datetime.now(timezone.utc).isoformat(), 'run_id': run_dir.name, 'trial_id': name,
            'population_npz': 'population.npz', 'population_npz_sha256': sha(pdir / 'population.npz'),
            'dataset': cfg['dataset'], 'signal_kind': cfg['signal_kind'],
            'label': 'model-derived extracellular contribution of three selected fly cells; not whole-CNS LFP; not empirically validated',
            'source_model_seed': cfg['source_model_seed'], 'split_group': cfg['split_group'],
            'parent_source_group': meta['split_group'],
            'source_trial': cfg['source_output'] + '/trial.npz', 'source_trial_sha256': inputs['hashes']['source_trial'],
            'source_metadata_sha256': inputs['hashes']['source_metadata'],
            'source_parameter_ensemble': meta['parameter_ensemble'], 'stimulus': meta['stimulus'],
            'sampling_rate_hz': meta['source_sampling_rate_hz'], 'sample_times': 'population.npz:time_s',
            'cable_parameter_ensemble': params_record, 'variant': {'name': name, 'dt_ms': dt, 'hold_source_at_baseline': blank},
            'cells': [{'body_id': c['body_id'], 'cell_type': c['cell_type'], 'instance': c['instance'],
                       'swc': str(c['swc'].relative_to(ROOT)), 'swc_sha256': inputs['hashes'][f"swc_{c['body_id']}"],
                       'contacts_sha256': inputs['hashes'][f"contacts_{c['body_id']}"],
                       'raw_contacts': c['raw_contacts'], 'retained_contacts': c['retained_contacts'],
                       'enabled_contacts': c['enabled_contacts']} for c in inputs['cells']],
            'observer_geometry': {'electrodes_source': cfg['observer']['electrodes_source'],
                                  'electrodes_source_sha256': inputs['hashes']['electrodes_source'],
                                  'electrode_array_sha256': inputs['electrode_sha256'],
                                  'coordinates': cfg['observer']['coordinates'], 'forward_model': cfg['observer']['forward_model']},
            'reference_channel_zero_based': reference, 'reference': cfg['observer']['reference'],
            'baseline': cfg['observer']['baseline'], 'units': UNITS,
            'conductance_time_semantics': 'interval arrays apply on [time[k],time[k+1]); applied arrays align with the previous integration interval of each current sample; initial sample follows baseline warmup',
            'nuisance_parameters': {'sigma_S_m': params.sigma_S_m, 'noise': None, 'electrode_boundary': None},
            'trajectory': None, 'movement_label': None, 'spike_conversion': False, 'feedback_to_CNS': False,
            'source_files_sha256': {p: hashlib.sha256(d).hexdigest() for p, d in frozen.items()},
            'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
            'input_files_sha256': inputs['hashes'], 'software': {k: v for k, v in software.items() if k != 'manifests'},
            'empirical_validation': False, 'whole_CNS_LFP': False,
            'diagnostics': diag[name]})
        processed[name] = {'cells': cell_proc, 'total': total_proc}
        totals[name] = {'unreferenced': agg['total'], 'stack': stack}
    return inputs, software, preflight, specs, processed, diag, totals, baseline, stimulus


def gates_and_descriptors(cfg, run_dir, inputs, processed, diag, totals, stimulus):
    val = cfg['validation']; floor = val['relative_error_denominator_floor_uV']
    bodies = [c['body_id'] for c in inputs['cells']]
    reference = cfg['observer']['reference_channel_zero_based']
    checks = {}; details = {}
    balances = {f'{v}/{b}': diag[v][b]['max_abs_current_balance_nA'] for v in diag for b in diag[v] if b != 'population'}
    checks['current_balance'] = all(x <= val['current_balance_abs_tolerance_nA'] for x in balances.values())
    blank = {str(b): float(np.max(np.abs(processed['blank']['cells'][i]))) for i, b in enumerate(bodies)}
    blank['total'] = float(np.max(np.abs(processed['blank']['total'])))
    checks['blank'] = all(x <= val['blank_max_delta_uV_tolerance'] for x in blank.values())
    details['blank_max_abs_uV'] = blank
    for variant, key, tol in [('half_dt', 'time_step', val['time_step_relative_max_tolerance']),
                              ('half_segment', 'space_step', val['space_step_relative_max_tolerance'])]:
        errors = {str(b): pop.relative_max_error(processed[variant]['cells'][i], processed['base']['cells'][i], floor)
                  for i, b in enumerate(bodies)}
        errors['total'] = pop.relative_max_error(processed[variant]['total'], processed['base']['total'], floor)
        details[f'{key}_errors'] = errors
        checks[key] = all(e['relative_max_error'] <= tol for e in errors.values())
    details['radius_sensitivity_descriptor'] = {
        v: {**{str(b): pop.relative_max_error(processed[v]['cells'][i], processed['base']['cells'][i], floor)['relative_max_error']
               for i, b in enumerate(bodies)},
            'total': pop.relative_max_error(processed[v]['total'], processed['base']['total'], floor)['relative_max_error']}
        for v in ['radius_half', 'radius_double']}
    sigma = {b: diag['base'][b]['conductivity_scaling_relative_error'] for b in diag['base'] if b != 'population'}
    checks['conductivity_scaling'] = all(e <= 1e-12 for s in sigma.values() for e in s.values())
    details['conductivity_scaling_relative_error'] = sigma
    checks['common_geometry'] = all(diag[v][b]['geometry']['passed'] for v in diag for b in diag[v] if b != 'population')
    tol = val['summation_relative_tolerance']
    summation = {v: diag[v]['population'] for v in diag}
    checks['summation_order'] = all(max(s['summation_order']['forward_vs_reverse'], s['summation_order']['forward_vs_vectorized']) <= tol
                                    for s in summation.values())
    checks['sum_identity_from_saved_files'] = all(s['sum_identity_from_saved_files_relative_error'] <= tol for s in summation.values())
    checks['processed_linearity'] = all(s['processed_linearity_relative_error'] <= tol for s in summation.values())
    checks['reference_channel_zero'] = all(s['reference_channel_max_abs_uV'] == 0 for s in summation.values())
    # Regression: the Mi1 cell must reproduce the saved single-cell v1 base.
    reg = cfg['regression']; body = reg['body_id']
    v1 = np.load(ROOT / reg['v1_base_trial'])
    mine = np.load(run_dir / 'variants/base/cells' / f'{body}.npz')
    regression = {'membrane_current': rel_error(mine['transmembrane_current_nA'], v1['transmembrane_current_nA']),
                  'same_electrode_unreferenced_potential': rel_error(mine['extracellular_potential_uV'], v1['extracellular_potential_uV']),
                  'lead_field': rel_error(mine['lead_field_mV_per_nA'], v1['lead_field_mV_per_nA']),
                  'electrodes_identical': bool(np.array_equal(mine['electrodes_um'], v1['electrodes_um'])),
                  'contact_body_pre_identical': bool(np.array_equal(mine['contact_body_pre'], v1['contact_body_pre'])),
                  'contact_xyz_identical': bool(np.array_equal(mine['contact_xyz_um'], v1['contact_xyz_um'])),
                  'contact_segments_identical': bool(np.array_equal(mine['contact_segment_indices'], v1['contact_segment_indices']))}
    details['regression_to_v1_base'] = regression
    checks['mi1_current_regression'] = regression['membrane_current']['relative_max_error'] <= reg['relative_tolerance']
    checks['mi1_voltage_regression'] = (regression['same_electrode_unreferenced_potential']['relative_max_error'] <= reg['relative_tolerance']
                                        and regression['lead_field']['relative_max_error'] <= reg['relative_tolerance']
                                        and regression['electrodes_identical'])
    checks['mi1_contacts_regression'] = all(regression[k] for k in ['contact_body_pre_identical', 'contact_xyz_identical', 'contact_segments_identical'])
    # Descriptors on the base variant after identical reference and baseline.
    desc_cfg = cfg['descriptors']
    cells = processed['base']['cells']; total = processed['base']['total']
    channels = np.ones(cells.shape[2], bool); channels[reference] = False
    ratio, cancel_stim = pop.cancellation_ratio(cells, desc_cfg['cancellation_denominator_floor_uV'], channels, stimulus)
    _, cancel_full = pop.cancellation_ratio(cells, desc_cfg['cancellation_denominator_floor_uV'], channels, None)
    fs = inputs['meta']['source_sampling_rate_hz']
    spectra = pop.spectral_decomposition(cells[:, stimulus, :], fs, total=total[stimulus])
    checks['spectral_identity'] = spectra['identity_relative_max_error'] <= val['spectral_identity_relative_tolerance']
    peak_cells = np.max(np.abs(cells[:, stimulus, :]), axis=1)
    peak_total = np.max(np.abs(total[stimulus]), axis=0)
    display = int(np.flatnonzero(channels)[np.argmax(peak_total[channels])])
    ddir = run_dir / 'descriptors'; ddir.mkdir()
    np.savez_compressed(ddir / 'base_cancellation.npz', cancellation_ratio=ratio, stimulus_mask=stimulus,
                        channel_mask=channels, floor_uV=desc_cfg['cancellation_denominator_floor_uV'])
    np.savez_compressed(ddir / 'base_spectra.npz', frequency_hz=spectra['frequency_hz'], total_power_uV2_per_Hz=spectra['total_power'],
                        auto_power_uV2_per_Hz=spectra['auto_power'], cross_terms_uV2_per_Hz=spectra['cross_terms'],
                        pairs=np.array(spectra['pairs']), cell_body_ids=np.array(bodies))
    at = {}
    for f in desc_cfg['report_frequencies_hz']:
        bin_hz, tot = pop.value_at_frequency(spectra['frequency_hz'], spectra['total_power'], f)
        _, auto = pop.value_at_frequency(spectra['frequency_hz'], spectra['auto_power'].sum(axis=0), f)
        _, cross = pop.value_at_frequency(spectra['frequency_hz'], spectra['cross_terms'].sum(axis=0), f)
        at[str(f)] = {'bin_hz': bin_hz, 'display_channel_total': float(tot[display]),
                      'display_channel_auto_sum': float(auto[display]), 'display_channel_cross_sum': float(cross[display])}
    descriptors = {
        'interpretation': desc_cfg['interpretation'], 'variant': 'base', 'window': desc_cfg['window'],
        'display_channel_zero_based': display,
        'display_channel_rule': 'non-reference channel with the largest max |total| in the stimulus window',
        'peak_abs_in_stimulus_window_uV': {'cells': {str(b): peak_cells[i].tolist() for i, b in enumerate(bodies)},
                                           'total': peak_total.tolist(),
                                           'sum_of_cell_peaks': peak_cells.sum(axis=0).tolist()},
        'cancellation_stimulus_window': cancel_stim, 'cancellation_full_trial': cancel_full,
        'spectral_identity_relative_max_error': spectra['identity_relative_max_error'],
        'spectral_preprocessing': spectra['preprocessing'], 'spectral_values_at_frequencies_uV2_per_Hz': at,
        'frequency_resolution_hz': float(spectra['frequency_hz'][1]),
        'files_sha256': {f'descriptors/{name}': sha(ddir / name) for name in ['base_cancellation.npz', 'base_spectra.npz']}}
    return checks, details, descriptors


def run(config_path, run_id=None):
    config_path = Path(config_path)
    config_bytes = config_path.read_bytes(); cfg = json.loads(config_bytes)
    frozen = {p: (ROOT / p).read_bytes() for p in CODE}
    run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = ROOT / cfg['run_output_root'] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = run_dir / 'code_snapshot'; snapshot.mkdir()
    for p, data in frozen.items():
        (snapshot / p.replace('/', '__')).write_bytes(data)
    (snapshot / 'config.json').write_bytes(config_bytes)
    status = {'run_id': run_id, 'started_at': datetime.now(timezone.utc).isoformat(), 'state': 'running'}
    dump(run_dir / 'status.json', status)
    start = time.perf_counter()
    try:
        if live_section_names():
            raise GateFailure('NEURON already has sections before the run')
        inputs, software, preflight, specs, processed, diag, totals, baseline, stimulus = execute(cfg, config_bytes, run_dir, frozen)
        checks, details, descriptors = gates_and_descriptors(cfg, run_dir, inputs, processed, diag, totals, stimulus)
        rehash = {k: v for k, v in inputs['hashes'].items()}
        current = dict(rehash)
        current['source_trial'] = sha(ROOT / cfg['source_output'] / 'trial.npz')
        current['source_metadata'] = sha(ROOT / cfg['source_output'] / 'metadata.json')
        current['contacts_manifest'] = sha(ROOT / cfg['contacts_manifest'])
        current['electrodes_source'] = sha(ROOT / cfg['observer']['electrodes_source'])
        current['regression_trial'] = sha(ROOT / cfg['regression']['v1_base_trial'])
        for c in cfg['cells']:
            current[f"swc_{c['body_id']}"] = sha(ROOT / c['swc'])
            current[f"contacts_{c['body_id']}"] = sha(ROOT / c['contacts'])
        checks['source_unchanged'] = current['source_trial'] == rehash['source_trial']
        checks['inputs_unchanged'] = current == rehash
        checks['code_unchanged'] = all((ROOT / p).read_bytes() == d for p, d in frozen.items()) and config_path.read_bytes() == config_bytes
        checks['no_live_sections'] = not live_section_names()
        checks['software_license_hashes_match'] = bool(software['license_hashes_match'])
        checks = {k: bool(v) for k, v in checks.items()}
        summary = {
            'run_id': run_id, 'signal_kind': cfg['signal_kind'],
            'label': 'model-derived extracellular contribution of three selected fly cells (Mi1_R, L1_R, L2_R); not whole-CNS LFP',
            'source_integrated_cells': int(inputs['meta']['diagnostics']['n_integrated_cells']),
            'source_recorded_columns': int(len(inputs['ids'])),
            'observed_membrane_cells': len(inputs['cells']),
            'cells': [{'body_id': c['body_id'], 'cell_type': c['cell_type'], 'instance': c['instance'],
                       'raw_contacts': c['raw_contacts'], 'retained_contacts': c['retained_contacts'],
                       'enabled_contacts': c['enabled_contacts']} for c in inputs['cells']],
            'same_column_established': cfg['cell_selection']['same_column_established'],
            'independent_source_parameter_ensembles': 1, 'source_model_seed': cfg['source_model_seed'],
            'split_group': cfg['split_group'], 'units': UNITS,
            'reference_channel_zero_based': cfg['observer']['reference_channel_zero_based'],
            'geometry_preflight_passed': preflight['passed'],
            'geometry_min_finite_distance_um': {b: min(v['min_finite_centerline_distance_um'] for v in r.values())
                                                for b, r in preflight['cells'].items()},
            'geometry_line_kernel_cutoff_pairs': {b: {n: v['line_kernel_radius_cutoff_pairs'] for n, v in r.items()}
                                                  for b, r in preflight['cells'].items()},
            'time_axis': inputs['time_check'], 'checks_passed': checks, 'details': details, 'descriptors': descriptors,
            'variant_diagnostics': diag, 'empirical_validation': False, 'whole_CNS_LFP': False,
            'trajectory': None, 'movement_label': None, 'wall_time_s': time.perf_counter() - start}
        dump(run_dir / 'summary.json', summary)
        state = 'passed' if all(checks.values()) else 'failed_gates'
        status.update(state=state, finished_at=datetime.now(timezone.utc).isoformat(),
                      failed_checks=[k for k, v in checks.items() if not v])
        dump(run_dir / 'status.json', status)
        if state != 'passed':
            raise GateFailure(f'Validation gates failed: {status["failed_checks"]}; evidence retained in {run_dir}')
        return summary
    except BaseException as error:
        if status['state'] == 'running':
            status.update(state='failed_gates' if isinstance(error, GateFailure) else 'error',
                          finished_at=datetime.now(timezone.utc).isoformat(), message=str(error),
                          traceback=traceback.format_exc())
            dump(run_dir / 'status.json', status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/fly_observer_population.json'))
    parser.add_argument('--run-id', default=None)
    args = parser.parse_args()
    try:
        summary = run(args.config, args.run_id)
    except GateFailure as error:
        print(f'GATE FAILURE: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
    print(json.dumps({'run_id': summary['run_id'], 'checks_passed': summary['checks_passed']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
