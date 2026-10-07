#!/usr/bin/env python3
"""Render a saved three-cell population observer run on linear physical scales.

With --hybrid, the total traces are the actual upstream hybridLFPy PostProcess
totals exported by postprocess_population_hybrid.py (only if its gates passed).
The figure is written to an explicit new path; completed runs are not modified.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from flyobserver import population as pop

COLORS = ['#c4572e', '#2f7d4f', '#3f67b1', '#8a5a9e', '#b08a1e']


def render(run_dir, output, hybrid_dir=None, audit=None):
    run_dir = Path(run_dir); out = Path(output)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    audit_line = None
    if audit is not None:
        record = json.loads(Path(audit).read_text())
        counts = record['status_counts']
        audit_line = (f'Independent audit (separate reviewer, not producer gates): '
                      f'{counts.get("pass", 0)}/{sum(counts.values())} pass, overall {record["overall"]}')
    summary = json.loads((run_dir / 'summary.json').read_text())
    base = np.load(run_dir / 'variants/base/population.npz')
    blank = np.load(run_dir / 'variants/blank/population.npz')
    cancel = np.load(run_dir / 'descriptors/base_cancellation.npz')
    d = summary['descriptors']
    ref = int(base['reference_channel_zero_based'])
    t = base['time_s']; gate = base['stimulus_interval_gate']; stim = cancel['stimulus_mask']
    labels = [f'{ty} · {b}' for ty, b in zip(base['cell_types'], base['cell_body_ids'])]
    cells_uV = base['per_cell_baseline_corrected_referenced_potential_uV']
    checks = dict(summary['checks_passed'])
    if hybrid_dir is not None:
        hybrid = json.loads((Path(hybrid_dir) / 'hybrid_summary.json').read_text())
        if not all(hybrid['checks_passed'].values()):
            raise ValueError('Hybrid postprocessing gates did not pass; refusing to plot its totals')
        totals = np.load(Path(hybrid_dir) / 'hybrid_totals.npz')
        if [str(x) for x in totals['cell_labels']] != [str(x) for x in base['cell_labels']]:
            raise ValueError('Hybrid totals belong to different cells')
        total_uV = totals['base__upstream_total_baseline_corrected_referenced_uV']
        blank_total_uV = totals['blank__upstream_total_baseline_corrected_referenced_uV']
        total_label = f'Total: actual upstream hybridLFPy PostProcess ({hybrid["upstream"]["commit"][:7]})'
        checks.update({f'hybrid:{k}': v for k, v in hybrid['checks_passed'].items()})
    else:
        total_uV = base['total_baseline_corrected_referenced_potential_uV']
        blank_total_uV = blank['total_baseline_corrected_referenced_potential_uV']
        total_label = 'Signed sum of 3 cells'
    channels = np.ones(total_uV.shape[1], bool); channels[ref] = False
    peak_total = np.max(np.abs(total_uV[stim]), axis=0)
    peak_cells = np.max(np.abs(cells_uV[:, stim]), axis=1)
    ch = int(np.flatnonzero(channels)[np.argmax(peak_total[channels])])
    fs = 1 / (t[1] - t[0])
    spectra = pop.spectral_decomposition(cells_uV[:, stim], fs, total=total_uV[stim])
    nV = 1e3
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(14, 11), layout='constrained')
    gs = fig.add_gridspec(4, 2, height_ratios=[.5, 1.25, 1, 1])

    ax = fig.add_subplot(gs[0, :])
    ax.step(t[:-1], gate, where='post', color='#333333', lw=1)
    ax.set(ylabel='R1–R6 drive\n(on/off)', yticks=[0, 1], xlim=(t[0], t[-1]),
           title='A  Stimulus: 10 Hz square-wave drive to R1–R6 terminals of the frozen full-CNS source (continuous a.u. activity, no spikes)')

    ax = fig.add_subplot(gs[1, :])
    for i, label in enumerate(labels):
        ax.plot(t, cells_uV[i, :, ch] * nV, color=COLORS[i], lw=1.1, label=f'{label} (single cell)')
    ax.plot(t, total_uV[:, ch] * nV, color='black', lw=2.0, label=total_label)
    ax.plot(t, blank_total_uV[:, ch] * nV, color='#777777', lw=1, ls=':', label='Total, source held at baseline')
    ax.axhline(0, color='#bbbbbb', lw=.7)
    ax.set(xlim=(t[0], t[-1]), xlabel='Time (s)', ylabel=f'Δ potential, C{ch} − C{ref} (nV)',
           title=f'B  Per-cell and total contributions at C{ch} (non-reference contact with largest peak |total|; zero-based index)')
    # Legend in its own row below the axis so no stimulus cycle is hidden.
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.2), ncol=5, fontsize=8, frameon=False,
              handlelength=2.2, columnspacing=1.6)

    ax = fig.add_subplot(gs[2, 0])
    idx = np.arange(total_uV.shape[1])
    for i in range(len(labels)):
        ax.plot(idx, peak_cells[i] * nV, 'o-', color=COLORS[i], ms=3.5, lw=1, label=labels[i])
    ax.plot(idx, peak_total * nV, 's-', color='black', ms=4, lw=1.6, label='|total|')
    ax.plot(idx, peak_cells.sum(axis=0) * nV, '--', color='#888888', lw=1, label='Σ single-cell peaks')
    ax.axvline(ref, color='#bbbbbb', lw=6, alpha=.5, zorder=0)
    ax.annotate(f'reference C{ref}', (ref, 0), xytext=(-4, 4), textcoords='offset points', ha='right', fontsize=7.5, color='#666666')
    ax.set(xlabel='Electrode contact (zero-based)', ylabel='Peak |Δ potential| in stimulus window (nV)',
           title='C  Amplitude across the shared 16-contact synthetic array')
    ax.legend(fontsize=7.5, loc='upper left')

    ax = fig.add_subplot(gs[2, 1])
    ratio = cancel['cancellation_ratio'][:, ch]
    ax.plot(t[stim], ratio[stim], color='#444444', lw=1)
    undefined = int(np.count_nonzero(~np.isfinite(ratio[stim])))
    ax.set(ylim=(-.02, 1.05), xlim=(t[stim][0], t[stim][-1]), xlabel='Time (s)', ylabel='|Σ φᵢ| / Σ |φᵢ|',
           title=f'D  Cancellation at C{ch}: 1 = same sign, 0 = complete cancellation')
    q = d['cancellation_stimulus_window']['quantiles_0_5_50_95_100']
    note = (f'All non-reference contacts, stimulus window: median {q[2]:.2f} (5–95%: {q[1]:.2f}–{q[3]:.2f})\n'
            f'Undefined samples at C{ch}: {undefined} (denominator ≤ {float(cancel["floor_uV"]):g} µV)') if q else 'No defined samples'
    ax.text(.01, .04, note, transform=ax.transAxes, fontsize=7.5, color='#555555')

    ax = fig.add_subplot(gs[3, 0])
    f = spectra['frequency_hz']; keep = f <= 60; psd = 1e6  # µV²/Hz → nV²/Hz
    ax.plot(f[keep], spectra['total_power'][keep, ch] * psd, color='black', lw=1.8, label='P(total)')
    ax.plot(f[keep], spectra['auto_power'].sum(axis=0)[keep, ch] * psd, color='#c4572e', lw=1.1, label='Σ auto terms')
    ax.plot(f[keep], spectra['cross_terms'].sum(axis=0)[keep, ch] * psd, color='#3f67b1', lw=1.1, label='2 Σ Re cross terms')
    ax.axhline(0, color='#bbbbbb', lw=.7)
    ax.set(xlabel='Frequency (Hz)', ylabel='PSD (nV²/Hz)',
           title=f'E  Spectral terms at C{ch}, stimulus window (identity error {spectra["identity_relative_max_error"]:.1e})')
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[3, 1]); ax.axis('off')
    failed = [k for k, v in checks.items() if not v]
    ax.text(0, 1, 'MODEL PILOT · 3 SELECTED MEMBRANE CELLS', fontsize=12, weight='bold', va='top')
    gate_kind = 'Producer + upstream-adapter gates' if hybrid_dir is not None else 'Producer gates'
    aggregation = ('Total = actual upstream hybridLFPy PostProcess aggregation\n'
                   'of the 3 raw signed per-cell potentials (no per-cell scaling).\n'
                   if hybrid_dir is not None else 'Total = signed sum without per-cell scaling.\n')
    ax.text(0, .86,
            f'Source: full-CNS continuous activity model (a.u.), {summary["source_integrated_cells"]:,} cells,\n'
            f'1 parameter ensemble (seed {summary["source_model_seed"]}); no spikes, no feedback.\n'
            f'Membrane models: {", ".join(labels)}; same column not established.\n'
            'Real Male CNS morphologies and synapse locations; full membrane\n'
            'current incl. returns; one shared synthetic electrode array.\n'
            + aggregation +
            f'Reference C{ref} subtracted after aggregation. Model-derived nV;\n'
            'σ, T4-transferred membrane parameters, conductance gain and probe\n'
            'placement uncalibrated; no empirical fit. Not whole-CNS LFP. The\n'
            '10 Hz component is imposed by the stimulus; not empirical validation.\n\n'
            f'{gate_kind}: {len(checks) - len(failed)}/{len(checks)} passed' + (f' — FAILED: {", ".join(failed[:4])}' if failed else '')
            + (f'\n{audit_line}' if audit_line else ''),
            fontsize=8.6, va='top', linespacing=1.4, color='#b00020' if failed else 'black')
    fig.suptitle('Fly visual cells (model): full-CNS source → 3 membrane cells → shared-electrode extracellular contribution',
                 fontsize=14, weight='bold')
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', required=True, help='population run directory')
    parser.add_argument('--output', required=True, help='new PNG path')
    parser.add_argument('--hybrid', default=None, help='passed postprocess_population_hybrid.py output directory')
    parser.add_argument('--audit', default=None, help='optional independent_audit.json, reported separately')
    args = parser.parse_args()
    print(render(args.run, args.output, args.hybrid, args.audit))
