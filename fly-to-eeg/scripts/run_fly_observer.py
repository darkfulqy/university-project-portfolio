#!/usr/bin/env python3
"""Validate a physical single-fly-cell contribution using full-CNS replay."""
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pyarrow.feather as feather
from lfpykit import CellGeometry, RecExtElectrode
from flyobserver.cable import PassiveFlyCable, CableParameters, synthetic_probe


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_unit_check():
    # +1/-1 nA closed source/sink pair, far from contact radius cutoff.
    geom=CellGeometry(x=np.array([[0.,0.],[20.,20.]]),
                      y=np.array([[0.,0.],[0.,0.]]),
                      z=np.array([[-.5,.5],[-.5,.5]]),d=np.ones(2))
    matrix=RecExtElectrode(geom,sigma=.3,x=np.array([100.]),y=np.array([0.]),
                           z=np.array([0.]),method='pointsource').get_transformation_matrix()
    predicted=float((matrix@np.array([1.,-1.]))[0]*1000)
    # SI: A / (S/m * m) = V. Convert V to µV only here.
    analytic=(1e-9/(4*np.pi*.3*100e-6)-1e-9/(4*np.pi*.3*80e-6))*1e6
    np.testing.assert_allclose(predicted,analytic,rtol=1e-12,atol=1e-12)
    return {'analytic_uV':analytic,'lfpykit_uV':predicted,
            'absolute_error_uV':abs(predicted-analytic)}


def main():
    cfg_path=ROOT/'configs/fly_observer_mi1.json'; cfg=json.loads(cfg_path.read_text())
    out=ROOT/cfg['output']
    if (out/'summary.json').exists():
        raise FileExistsError('Preserve completed results; use a new version for reruns')
    source_paths=['src/flyobserver/cable.py','scripts/run_fly_observer.py',
                  'configs/fly_observer_mi1.json','requirements-observer.lock.txt']
    frozen={p:(ROOT/p).read_bytes() for p in source_paths}
    source_sha=sha(ROOT/cfg['source_trial'])
    parent_metadata=json.loads((ROOT/cfg['source_metadata']).read_text())
    if (parent_metadata['trial_sha256']!=source_sha or
        parent_metadata['source_model_seed']!=cfg['source_model_seed'] or
        parent_metadata['split_group']!=cfg['split_group']):
        raise ValueError('Source hash, seed or split group mismatch')
    input_hashes={p:sha(ROOT/cfg[p]) for p in ['contacts','swc','source_metadata']}
    src=np.load(ROOT/cfg['source_trial']); time_s=src['time_s']
    activity=src['individual_activity_au']; ids=src['recorded_cell_ids']
    column={int(n):i for i,n in enumerate(ids)}
    contacts=feather.read_table(ROOT/cfg['contacts'])
    retained=contacts.filter(contacts['included_in_unsigned_graph'])
    xyz=np.column_stack([retained[k].to_numpy() for k in ['x_post_um','y_post_um','z_post_um']])
    precol=np.array([column[int(n)] for n in retained['body_pre'].to_numpy()])
    signs=retained['pre_model_output_sign'].to_numpy()
    p=CableParameters(**cfg['cable_parameters'])
    reference=cfg['observer']['reference_channel_zero_based']
    base_dt=cfg['integration']['dt_ms']
    source_dt=1000/parent_metadata['source_sampling_rate_hz']
    variants=[('base',p,base_dt,False),('blank',p,base_dt,True),
              ('half_dt',p,base_dt/2,False),('half_segment',replace(p,max_segment_um=p.max_segment_um/2),base_dt,False),
              ('radius_half',replace(p,radius_scale=p.radius_scale*.5),base_dt,False),
              ('radius_double',replace(p,radius_scale=p.radius_scale*2),base_dt,False)]
    summary={'physical_unit_check':physical_unit_check(),'variants':{},
             'source_trial_sha256':source_sha,'source_integrated_cells':len(src['all_cell_ids']),
             'observed_membrane_cells':1,'retained_input_contacts':len(retained),
             'enabled_input_contacts':int(np.count_nonzero(signs)),
             'independent_source_parameter_ensembles':1}
    signals={}; start=time.perf_counter(); electrodes=None
    for name,params,dt,blank in variants:
        cell=PassiveFlyCable(ROOT/cfg['swc'],params)
        cell.map_contacts(xyz,precol,signs,len(ids))
        if electrodes is None:
            electrodes=synthetic_probe(cell.xyz_um,cfg['observer']['probe_offset_um'])
        direction=cell.ends-cell.starts
        fraction=np.clip(np.einsum('csi,si->cs',electrodes[:,None,:]-cell.starts,direction)/
                         (direction*direction).sum(axis=1),0,1)
        distance=np.linalg.norm(electrodes[:,None,:]-(cell.starts+fraction[:,:,None]*direction),axis=2)
        minimum_distance=float(distance.min())
        radius=np.asarray(cell.geometry.d)/2
        if (minimum_distance<cfg['observer']['minimum_centerline_clearance_um'] or
            np.any(distance<=radius[None,:])):
            raise ValueError('Synthetic probe violates cable clearance; do not rely on radius regularization')
        drive=np.broadcast_to(activity[0],activity.shape) if blank else activity
        tick=time.perf_counter()
        result=cell.run(drive,source_dt_ms=source_dt,dt_ms=dt,warmup_ms=cfg['integration']['warmup_ms'])
        matrix=cell.lead_field(electrodes)
        current=result['transmembrane_current_nA']
        potential=current@matrix.T*1000 # nA × mV/nA × 1000 µV/mV
        referenced=potential-potential[:,reference,None]
        baseline=time_s<parent_metadata['stimulus']['onset_s']
        delta=referenced-referenced[baseline].mean(axis=0)
        balance=float(np.max(np.abs(current.sum(axis=1))))
        currentscale=float(np.max(np.sum(np.abs(current),axis=1)))
        m={'parameters':asdict(params),'dt_ms':dt,'n_segments':len(cell.segments),
           'area_um2':float(cell.area_um2.sum()),'wall_time_s':time.perf_counter()-tick,
           'max_abs_delta_uV':float(np.max(abs(delta))),
           'max_abs_current_balance_nA':balance,
           'max_sum_abs_membrane_current_nA':currentscale,
           'relative_current_balance':balance/max(currentscale,1e-30),
           'minimum_electrode_centerline_distance_um':minimum_distance,
           'minimum_electrode_distance_over_segment_radius':float(np.min(distance/radius[None,:])),
           'radius_regularization_triggered':False,
           'voltage_min_mV':float(result['membrane_voltage_mV'].min()),
           'voltage_max_mV':float(result['membrane_voltage_mV'].max()),
           'projection_distance_um_quantiles':np.quantile(cell.contact_projection_distance_um,[0,.5,.95,1]).tolist()}
        # No external current is injected: transmembrane sources and sinks balance.
        if balance>cfg['validation']['current_balance_abs_tolerance_nA']:
            raise AssertionError(f'Current conservation failed: {m}')
        np.testing.assert_array_equal(referenced[:,reference],np.zeros(len(time_s)))
        variant_dir=out/name; variant_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(variant_dir/'trial.npz',time_s=time_s,**result,
            extracellular_potential_uV=potential,referenced_potential_uV=referenced,
            baseline_corrected_referenced_potential_uV=delta,
            lead_field_mV_per_nA=matrix,electrodes_um=electrodes,
            segment_starts_um=cell.starts,segment_ends_um=cell.ends,segment_area_um2=cell.area_um2,
            contact_body_pre=retained['body_pre'].to_numpy(),contact_xyz_um=xyz,
            contact_segment_indices=cell.contact_segment_indices,
            contact_projection_distance_um=cell.contact_projection_distance_um,
            source_cell_ids=ids,observer_cell_ids=np.array([cfg['body_id']]),
            stimulus_interval_gate=src['stimulus_interval_gate'])
        metadata={'created_at':datetime.now(timezone.utc).isoformat(),'trial_id':name,
            'dataset':cfg['dataset'],'source_model_seed':cfg['source_model_seed'],'split_group':cfg['split_group'],
            'parent_source_group':cfg['split_group'],
            'source_trial':cfg['source_trial'],'source_trial_sha256':source_sha,
            'source_metadata':cfg['source_metadata'],'source_metadata_sha256':input_hashes['source_metadata'],
            'stimulus':parent_metadata['stimulus'], 'parameter_ensemble':asdict(params),
            'configuration':cfg,'variant':{'name':name,'dt_ms':dt,'hold_source_at_baseline':blank},
            'sampling_rate_hz':1000/source_dt,'signal_kind':'single_cell_model_extracellular_contribution',
            'conductance_time_semantics':'interval arrays apply on [time[k],time[k+1]); applied arrays align with the previous integration interval of each current sample; initial sample follows baseline warmup',
            'units':{'current':'nA','intracellular_voltage':'mV','extracellular_voltage':'uV',
                     'conductance':'uS','geometry':'um','lead_field':'mV/nA'},
            'observer_geometry':'trial.npz:electrodes_um, segment_starts_um, segment_ends_um',
            'reference_channel_zero_based':reference,'trajectory':None,'movement_label':None,
            'nuisance_parameters':{'sigma_S_m':params.sigma_S_m,'noise':None,'electrode_boundary':None},
            'source_files_sha256':{p:hashlib.sha256(data).hexdigest() for p,data in frozen.items()},
            'input_files_sha256':input_hashes,
            'morphology_manifest':'references/male_cns_morphology_manifest.json',
            'empirical_validation':False,'whole_CNS_LFP':False,'diagnostics':m,
            'trial_sha256':sha(variant_dir/'trial.npz')}
        (variant_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        summary['variants'][name]=m; signals[name]=delta
        if name=='base':
            for sigma in cfg['validation']['conductivity_sensitivity_S_m']:
                scaled=cell.lead_field(electrodes,sigma=sigma)
                np.testing.assert_allclose(scaled,matrix*params.sigma_S_m/sigma,rtol=1e-12,atol=1e-12)
            summary['conductivity_scaling_verified_S_m']=cfg['validation']['conductivity_sensitivity_S_m']
        cell.close()
        print(json.dumps({'variant':name,**m}),flush=True)
    scale=max(float(np.max(abs(signals['base']))),1e-30)
    summary['half_dt_relative_max_error']=float(np.max(abs(signals['half_dt']-signals['base']))/scale)
    summary['half_segment_relative_max_error']=float(np.max(abs(signals['half_segment']-signals['base']))/scale)
    summary['checks_passed']={
        'blank':np.max(abs(signals['blank']))<cfg['validation']['blank_max_delta_uV_tolerance'],
        'time_step':summary['half_dt_relative_max_error']<cfg['validation']['time_step_relative_max_tolerance'],
        'space_step':summary['half_segment_relative_max_error']<cfg['validation']['space_step_relative_max_tolerance'],
        'source_unchanged':sha(ROOT/cfg['source_trial'])==source_sha,
        'input_files_unchanged':all(sha(ROOT/cfg[p])==digest for p,digest in input_hashes.items())}
    summary['checks_passed']={k:bool(v) for k,v in summary['checks_passed'].items()}
    summary['wall_time_s']=time.perf_counter()-start
    snapshot=out/'observer_source_snapshot'; snapshot.mkdir(exist_ok=True)
    for path,data in frozen.items():
        (snapshot/path.replace('/','__')).write_bytes(data)
        if data!=(ROOT/path).read_bytes():
            raise RuntimeError(f'Source changed during run: {path}')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    if not all(summary['checks_passed'].values()):
        raise AssertionError('A validation gate failed; retain and report this result')


if __name__=='__main__':
    main()
