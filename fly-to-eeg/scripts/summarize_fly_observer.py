#!/usr/bin/env python3
"""Render saved physical single-cell results without physiological rescaling."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fly_observer/v1'


def main():
    d=np.load(OUT/'base/trial.npz'); summary=json.loads((OUT/'summary.json').read_text())
    t=d['time_s']; potential=d['baseline_corrected_referenced_potential_uV']*1000
    channels=np.argsort(np.max(abs(potential),axis=0))[-3:][::-1]
    a=d['segment_starts_um']; b=d['segment_ends_um']; xyz=np.vstack([a,b]); center=xyz.mean(axis=0)
    _,_,axes=np.linalg.svd(xyz-center,full_matrices=False)
    transform=lambda p:(p-center)@axes[:2].T
    segments=np.stack([transform(a),transform(b)],axis=1)
    contacts=transform(d['contact_xyz_um']); probe=transform(d['electrodes_um'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False})
    fig=plt.figure(figsize=(13,8),layout='constrained')
    gs=fig.add_gridspec(3,2,width_ratios=[1,1.6])
    ax=fig.add_subplot(gs[:2,0])
    ax.add_collection(LineCollection(segments,colors='#344054',linewidths=1.1))
    ax.scatter(*contacts.T,s=7,alpha=.45,color='#df9c25',label='405 retained input contacts')
    near=abs(probe[:,0])<90
    ax.scatter(*probe[near].T,marker='s',s=45,color='#3563a9',label='Synthetic point electrodes')
    for k in np.flatnonzero(near):
        ax.annotate(f'C{k}',probe[k],xytext=(4,5),textcoords='offset points',fontsize=8)
    ax.autoscale(); ax.set_aspect('equal')
    ax.set(xlabel='Projection along cell axis (µm)',ylabel='Orthogonal projection (µm)',
           title='A  Real Male CNS Mi1_R · body 17871\nGold: input contacts · Blue: synthetic electrodes')
    ax.text(.02,.02,'Only nearby contacts shown.\nFull 16-contact geometry saved.',
            transform=ax.transAxes,fontsize=8,color='#666666')
    ax=fig.add_subplot(gs[0,1])
    for key,label,color in [('interval_exc_conductance_uS','Excitatory','#df9c25'),
                            ('interval_inh_conductance_uS','Inhibitory','#7756a2')]:
        g=d[key].sum(axis=1)*1000
        ax.plot(t[:-1],g-g[t[:-1]<.2].mean(),label=label,color=color,lw=1.2)
    ax.set(title='B  Full-CNS continuous activity → membrane conductance',ylabel='Δ conductance (nS)')
    ax.legend(loc='upper right',fontsize=8); ax.axvspan(.2,2.2,color='#bbbbbb',alpha=.12)
    ax=fig.add_subplot(gs[1,1])
    im=d['transmembrane_current_nA']; delta=im-im[t<.2].mean(axis=0)
    source=np.maximum(delta,0).sum(axis=1)*1000
    sink=np.minimum(delta,0).sum(axis=1)*1000
    ax.plot(t,source,color='#b34c37',label='Outward current change')
    ax.plot(t,sink,color='#256fa1',label='Inward current change')
    ax.plot(t,delta.sum(axis=1)*1000,color='#444444',ls='--',label='Net change')
    ax.set(title='C  Spatial source and return currents balance',ylabel='Δ membrane current (pA)')
    ax.legend(loc='upper right',fontsize=8)
    ax=fig.add_subplot(gs[2,1])
    for channel in channels:
        ax.plot(t,potential[:,channel],label=f'C{channel} − C15',lw=1.2)
    blank=np.load(OUT/'blank/trial.npz')['baseline_corrected_referenced_potential_uV'][:,channels[0]]*1000
    ax.plot(t,blank,color='#333333',ls=':',label='Baseline held constant')
    ax.set(title='D  Predicted contribution of this one cell',xlabel='Time (s)',
           ylabel='Δ extracellular potential (nV)')
    ax.legend(loc='upper right',fontsize=8)
    ax=fig.add_subplot(gs[2,0]); ax.axis('off')
    ax.text(0,.95,'MODEL REUSE CHECK',fontsize=12,weight='bold',va='top')
    ax.text(0,.76,'165,122 CNS cells integrated → 1 cable cell observed\n'
        '403 enabled contacts → 344 membrane compartments\n'
        '10 Hz input from 0.2 to 2.2 s; no spikes sampled\n\n'
        f'Time-step error: {100*summary["half_dt_relative_max_error"]:.3f}%\n'
        f'Space-step error: {100*summary["half_segment_relative_max_error"]:.3f}%\n'
        'Voltage units derived from currents and geometry.\n'
        'Amplitude and probe placement are not empirically calibrated.',
        fontsize=10,va='top',linespacing=1.45)
    fig.suptitle('Fly membrane current → extracellular voltage: reuse pilot',fontsize=17,weight='bold')
    fig.savefig(OUT/'observer_response.png',dpi=170)
    print(OUT/'observer_response.png')


if __name__=='__main__':
    main()
