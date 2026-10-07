#!/usr/bin/env python3
"""Plot model activity, keeping population proxies separate from LFP voltage."""
from pathlib import Path
import json
import hashlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/visual_full_cns/v0'


def load(name):
    path = OUT / name
    return np.load(path/'trial.npz', allow_pickle=False), json.loads((path/'metadata.json').read_text())


def summarize():
    prefix = 'flash_seed_211_'
    main_name = prefix + ('packet_dt_0.001' if (OUT/(prefix+'packet_dt_0.001')/'trial.npz').exists()
                          else 'short_dt_0.001')
    a, meta = load(main_name)
    t, y = a['time_s'], a['population_activity_au']
    names = a['group_names'].tolist()
    stim = meta['configuration']['stimulus']
    on, off = stim['onset_s'], stim['offset_s']
    baseline = y[t < on].mean(axis=0)
    dy = y - baseline
    selected = ['R1_R6_L', 'L1_L', 'L2_L', 'Mi1_L', 'visual_projection_L', 'cb_intrinsic_L']
    selected = [x for x in selected if x in names]
    colors = ['#6b4eb5','#007a8b','#25844d','#ca8427','#b34467','#3e597d']
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'figure.facecolor':'white','axes.titleweight':'bold'})
    fig = plt.figure(figsize=(13, 12), constrained_layout=True)
    grid = fig.add_gridspec(5, 2, height_ratios=[.6,1.2,1.2,1.2,1.2])
    ax = fig.add_subplot(grid[0,:])
    dt = meta['configuration']['dynamics']['dt_s']
    # The first frozen v0 runs stored a 0/1 gate under the older input key.
    drive = (a['stimulus_interval_drive_au'] if 'stimulus_interval_drive_au' in a
             else a['stimulus_interval_input_au'] * stim['amplitude_au'])
    input_t = np.arange(len(drive)) * dt
    ax.step(input_t, drive, where='post', color='#536c40', lw=.7)
    ax.set(xlim=(0,t[-1]),ylim=(-.08,.95),ylabel='Drive (a.u.)',xlabel='Time (s)',
           title='R1–R6 terminal input · 10 Hz, 50% duty · no colour or phototransduction model')
    summary = []
    for k, name in enumerate(selected):
        j = names.index(name)
        ax = fig.add_subplot(grid[1+k//2,k%2])
        show = (t >= on-.1) & (t <= on+.7)
        ax.plot(t[show]-on,dy[show,j],color=colors[k],lw=1.4)
        ax.axvline(0,color='#999',ls=':',lw=1)
        ax.axhline(0,color='#ddd',lw=.7)
        ax.set(title=name, xlabel='Time from stimulus onset (s)', ylabel='Δ activity (a.u.)')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2),useMathText=True)
        summary.append({'group':name,'baseline_activity_au':float(baseline[j]),
                        'max_abs_response_au':float(np.max(np.abs(dy[:,j]))),
                        'minimum_response_au':float(dy[:,j].min()),
                        'maximum_response_au':float(dy[:,j].max())})
    ax = fig.add_subplot(grid[4,0])
    cycles = np.arange(on+.5, off-.1+1e-9, .1)
    phase = np.arange(100)/1000
    for k,name in enumerate(selected):
        j=names.index(name)
        epochs = np.array([dy[int(round(start*1000)):int(round(start*1000))+100,j] for start in cycles])
        if not len(epochs):
            continue
        curve = epochs.mean(axis=0)
        curve -= curve.mean()
        peak = np.max(np.abs(curve))
        if peak > 1e-15:
            ax.plot(phase*1000,curve/peak,color=colors[k],label=name,lw=1.3)
    ax.set(title='Cycle shape (each group scaled separately)',xlabel='Phase within a 100 ms cycle (ms)',
           ylabel='Centered response / own peak')
    ax.legend(fontsize=7,ncol=2,frameon=False)
    ax = fig.add_subplot(grid[4,1])
    active = (t>=on+.5) & (t<off)
    for k,name in enumerate(selected):
        j=names.index(name)
        freq,power=welch(dy[active,j],fs=1000,nperseg=min(1000,int(active.sum())),detrend='constant')
        valid=(freq>=1)&(freq<=60)
        ax.semilogy(freq[valid],np.maximum(power[valid],1e-30),color=colors[k],lw=1.2)
    ax.axvline(10,color='#999',ls=':',lw=1)
    ax.set(title='Stimulus-period spectrum · actual a.u.²/Hz',xlabel='Frequency (Hz)',ylabel='PSD (a.u.²/Hz)')
    fig.suptitle('Whole-CNS visual response: population activity prototype\n'
                 '165,122 cells · 24,539,704 active signed edges · arbitrary units',fontsize=15)
    fig.savefig(OUT/'visual_response.png',dpi=180)
    plt.close(fig)
    checks = {}
    coarse_dir=OUT/'flash_seed_211_short_dt_0.001'
    fine_dir=OUT/'flash_seed_211_short_dt_0.0005'
    if (coarse_dir/'trial.npz').exists() and (fine_dir/'trial.npz').exists():
        coarse,cm=load(coarse_dir.name);fine,fm=load(fine_dir.name)
        np.testing.assert_array_equal(coarse['time_s'],fine['time_s'])
        error=np.max(np.abs(coarse['population_activity_au']-fine['population_activity_au']),axis=0)
        scale=np.max(np.abs(fine['population_activity_au']-fine['population_activity_au'][0]),axis=0)
        eligible=scale>1e-6
        rel=error[eligible]/scale[eligible]
        full_state_error=float(np.max(np.abs(coarse['final_state_au']-fine['final_state_au'])))
        checks['half_dt']={'coarse_dt_s':.001,'fine_dt_s':.0005,'duration_s':2.4,
             'maximum_group_absolute_difference_au':float(error.max()),
             'maximum_group_relative_error_for_amplitude_gt_1e_6':float(rel.max()),
             'n_eligible_groups':int(eligible.sum()),'full_state_endpoint_max_absolute_error_au':full_state_error,
             'scope':'short-trial check, not full 22s trajectory convergence',
             'groups':[{'name':n,'max_error_au':float(e),'reference_response_au':float(s)}
                       for n,e,s in zip(coarse['group_names'],error,scale)]}
    for condition in ['blank','visual_output_block']:
        name=condition+'_seed_211_short_dt_0.001'
        if (OUT/name/'trial.npz').exists():
            c,m=load(name)
            input_ids=set(c['input_cell_ids'].tolist())
            outside=np.array([int(x) not in input_ids for x in c['all_cell_ids']])
            checks[condition]={'maximum_all_cell_change_au':float(c['max_cell_change_au'].max()),
                 'maximum_nondriven_cell_change_au':float(c['max_cell_change_au'][outside].max()),
                 'duration_s':2.4,'initialization':'separate fixed point for each condition',
                 'diagnostics':m['diagnostics']}
    report={'signal_kind':'population_activity_proxy','units':'arbitrary_units',
            'empirical_LFP_replication_completed':False,'main_trial':main_name,
            'parameter_seed':211,'independent_parameter_ensembles':1,
            'n_integrated_cells':meta['diagnostics']['n_integrated_cells'],
            'graph_counts':meta['connectome_manifest']['counts'],
            'main_diagnostics':meta['diagnostics'],'selected_groups':summary,'checks':checks,
            'interpretation':'forced stimulus-locked response in an uncalibrated contractive model; no endogenous oscillation claim',
            'central_brain_note':'very small population-mean response is retained without voltage rescaling',
            'plot_note':'left cell groups shown; side means do not denote electrode locations; independent y axes',
            'analysis_source':{'path':'scripts/summarize_visual_full_cns.py',
                               'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'artifact_audit':'artifact_hash_audit_completed_runs.json',
            'figure':'visual_response.png'}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    summarize()
