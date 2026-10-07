"""Joint trial-bootstrap comparison against a fitted source-additive reference.

See paper/paper_B_utility/JOINT_NULL_PROTOCOL_2026-09-11.md. Conditional,
exploratory diagnostic; NOT an unbiased latent-headroom or learnability test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from selection_policy import cell_weights
from joint_cache import load_joint_cache, iter_joint_bootstrap

ARMS = {
    'wang_eegnet': ('Wang2026', 'eegnet'),
    'wang_specialist': ('Wang2026', 'specialist'),
    'zhou_eegnet': ('Zhou2020', 'eegnet'),
}
BLOCKS = (10, 5, 20)
TEST_SEED = 2026091101
PILOT_SEED = 2026091102
PROTOCOL = HERE.parents[1] / 'paper/paper_B_utility/JOINT_NULL_PROTOCOL_2026-09-11.md'
TOL = 1e-9


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def structure(keys):
    subjects = keys.subject_id.to_numpy()
    weight = cell_weights(subjects)
    source = list(keys.groupby(['subject_id', 'seed'], sort=True).indices.values())
    person = list(keys.groupby('subject_id', sort=True).indices.values())
    if len(person) < 2:
        raise ValueError('LOSO requires at least two subjects')
    return dict(weights=weight, sources=source, persons=person,
                source_weights=np.array([weight[ix].sum() for ix in source]), n_cells=len(keys))


def _array(Y, st):
    Y = np.asarray(Y, dtype=np.float64)
    single = Y.ndim == 2
    if single:
        Y = Y[None, ...]
    if Y.ndim != 3 or Y.shape[1] != st['n_cells'] or not Y.shape[2] or not np.isfinite(Y).all():
        raise ValueError('Expected finite [cell,method] or [draw,cell,method] scores')
    return Y, single


def fit_additive(Y, st):
    """Each source: method mean + cell mean - grand mean (equal days)."""
    Y, single = _array(Y, st)
    fitted = np.empty_like(Y)
    for ix in st['sources']:
        current = Y[:, ix, :]
        fitted[:, ix, :] = (current.mean(axis=1, keepdims=True)
                            + current.mean(axis=2, keepdims=True)
                            - current.mean(axis=(1, 2), keepdims=True))
    return fitted[0] if single else fitted


def statistics(Y, st):
    """Re-select all deliberate label oracles and train-only LOSO in each draw."""
    Y, single = _array(Y, st)
    person = np.stack([Y[:, ix, :].mean(axis=1) for ix in st['persons']], axis=1)
    source = np.stack([Y[:, ix, :].mean(axis=1) for ix in st['sources']], axis=1)
    train = (person.sum(axis=1, keepdims=True) - person) / (person.shape[1] - 1)
    chosen = train.argmax(axis=-1)
    fixed_gains = np.take_along_axis(person, chosen[..., None], axis=-1)[..., 0]
    q = np.zeros(len(Y))
    for ix in st['sources']:
        current = Y[:, ix, :]
        residual = (current - current.mean(axis=1, keepdims=True)
                    - current.mean(axis=2, keepdims=True)
                    + current.mean(axis=(1, 2), keepdims=True))
        q += np.einsum('bn,n->b', np.mean(residual**2, axis=2), st['weights'][ix])
    out = {'interaction_residual_mean_square': q}
    for mode in ('forced', 'with_r0'):
        use_r0 = mode == 'with_r0'
        best = lambda values: np.maximum(0., values.max(axis=-1)) if use_r0 else values.max(axis=-1)
        oracle_cell = best(Y) @ st['weights']
        oracle_source = best(source) @ st['source_weights']
        oracle_subject = best(person).mean(axis=1)
        fixed = (np.where(train.max(axis=-1) > 0., fixed_gains, 0.)
                 if use_r0 else fixed_gains).mean(axis=1)
        gap = oracle_cell - oracle_source
        if np.any(gap < -TOL):
            raise AssertionError('Session oracle lower than its source-fixed restriction')
        for name, value in dict(loso_fixed=fixed, global_label_oracle=best(person.mean(axis=1)),
                                subject_label_oracle=oracle_subject, source_label_oracle=oracle_source,
                                session_label_oracle=oracle_cell,
                                session_minus_source=np.maximum(0., gap),
                                session_minus_loso=oracle_cell-fixed).items():
            out[f'{mode}__{name}'] = value
    return {key: float(value[0]) if single else value for key, value in out.items()}


def mc_tail(null, observed):
    null = np.asarray(null, dtype=float)
    if null.ndim != 1 or not len(null) or not np.isfinite(null).all() or not np.isfinite(observed):
        raise ValueError('Finite null vector and observed statistic required')
    B = len(null)
    k = int(np.count_nonzero(null >= observed - TOL))
    lo = float(beta.ppf(.025, k, B-k+1)) if k else 0.
    hi = float(beta.ppf(.975, k+1, B-k)) if k < B else 1.
    return dict(exceedances=k, replicates=B, p_mc=(k+1)/(B+1),
                mc_tail_ci_lo=lo, mc_tail_ci_hi=hi)


def holm(p):
    p = np.asarray(p, float)
    result = np.empty(len(p))
    rank = np.argsort(p)
    result[rank] = np.minimum(1., np.maximum.accumulate(p[rank] * (len(p)-np.arange(len(p)))))
    return result


def run_one(cache, arm, L, args, signature):
    out = args.out / f'{arm}_L{L}'
    if (out / 'result.json').exists():
        prior = json.loads((out / 'result.json').read_text())
        if not args.resume or prior['signature'] != signature:
            raise FileExistsError(f'Existing output or incompatible resume: {out}')
        if (prior['input_digest'] != cache['audit']['input_files_digest']
                or prior['input_table_sha256'] != cache['audit']['table_sha256']):
            raise RuntimeError('Input data changed since the completed configuration')
        for name, expected in prior['outputs_sha256'].items():
            if sha(out / name) != expected:
                raise RuntimeError(f'Changed completed output: {name}')
        print(f'REUSE {arm} L={L}', flush=True)
        return prior
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Incomplete directory exists; inspect and use a new --out: {out}')
    out.mkdir(parents=True, exist_ok=True)
    st = structure(cache['keys'])
    point = cache['delta']
    theta = fit_additive(point, st)
    observed, reference = statistics(point, st), statistics(theta, st)
    if abs(reference['forced__session_minus_source']) > TOL:
        raise AssertionError('Additive reference must have zero forced switching gap')
    simulations = np.empty((args.draws,) + point.shape, dtype=np.float32)
    pilots = iter_joint_bootstrap(cache, args.pilot, L, PILOT_SEED, dtype=np.float32)
    trials = iter_joint_bootstrap(cache, args.draws, L, TEST_SEED, dtype=np.float32)
    audits, covariances, pilot_centres = [], {}, np.zeros_like(point)
    start = time.monotonic()
    for i, (pilot, trial) in enumerate(zip(pilots, trials)):
        ix = trial['cell_indices']
        np.testing.assert_array_equal(ix, pilot['cell_indices'])
        centre = pilot['mean_delta']
        pilot_centres[ix] = centre
        # Centre estimated independently of the Monte Carlo exceedance draws.
        simulations[:, ix, :] = (theta[ix] + trial['draws'].astype(np.float64) - centre).astype(np.float32)
        tag = f"S{trial['subject_id']:02d}_D{trial['target_session']}"
        covariances[tag+'__covariance'] = trial['covariance']
        covariances[tag+'__seed_mean_covariance'] = trial['seed_mean_covariance']
        covariances[tag+'__cell_indices'] = ix
        audits.append(dict(tag=tag, pilot=pilot['audit'], test=trial['audit']))
        if i % 10 == 0 or i+1 == len(cache['groups']):
            print(f'{arm} L={L} joint groups {i+1}/{len(cache["groups"])} ({time.monotonic()-start:.1f}s)', flush=True)
    if len(audits) != len(cache['groups']):
        raise AssertionError('Missing subject-day draws')
    null = {name: np.empty(args.draws) for name in observed}
    for start_ix in range(0, args.draws, 256):
        part = statistics(simulations[start_ix:start_ix+256], st)
        for name, value in part.items():
            null[name][start_ix:start_ix+len(value)] = value
    del simulations
    main_stat = 'forced__session_minus_source'
    tail = mc_tail(null[main_stat], observed[main_stat])
    summary = {name: dict(observed=observed[name], fitted_reference=reference[name],
                         bootstrap_mean=float(values.mean()),
                         bootstrap_q025=float(np.quantile(values, .025)),
                         bootstrap_q975=float(np.quantile(values, .975)),
                         observed_minus_bootstrap_mean=float(observed[name]-values.mean()))
               for name, values in null.items()}
    np.savez_compressed(out / 'null_statistics.npz', **null)
    np.savez_compressed(out / 'joint_covariances.npz', **covariances)
    np.savez_compressed(out / 'reference.npz', observed=point, theta0=theta, pilot_centres=pilot_centres)
    cache['keys'].to_csv(out / 'keys.csv', index=False)
    (out / 'resampling_audit.json').write_text(json.dumps(audits, indent=2) + '\n')
    result = dict(status='complete', arm=arm, block_length=L, signature=signature,
                  n_cells=len(point), n_subjects=len(st['persons']), n_methods=point.shape[1],
                  n_seeds=len(cache['keys'].seed.unique()), n_subject_days=len(cache['groups']),
                  pilot_replicates=args.pilot, main_statistic=main_stat, **tail,
                  statistics=summary, seconds=time.monotonic()-start,
                  input_digest=cache['audit']['input_files_digest'],
                  input_table_sha256=cache['audit']['table_sha256'],
                  rejection_rate=sum(a['test']['rejected'] for a in audits)/sum(a['test']['attempts'] for a in audits),
                  n_runs=sum(len(a['test']['run_sizes']) for a in audits),
                  n_unchanged_short_runs=sum(n <= L for a in audits for n in a['test']['run_sizes']),
                  fraction_trials_in_unchanged_short_runs=(sum(n for a in audits for n in a['test']['run_sizes'] if n <= L)
                                                           / sum(n for a in audits for n in a['test']['run_sizes'])),
                  interpretation='Conditional fitted-additive bootstrap comparison, not latent debiasing or learnability.',
                  mc_interval_scope='Binomial tail probability conditional on data, reference and independent pilot centre.',
                  outputs_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'COMPLETE {arm} L={L}: H={observed[main_stat]:.6f}, reference bootstrap mean={null[main_stat].mean():.6f}, k={tail["exceedances"]}/{args.draws}, p={tail["p_mc"]:.6f}', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=HERE / 'out_joint_null_20260911')
    ap.add_argument('--draws', type=int, default=9999)
    ap.add_argument('--pilot', type=int, default=4096)
    ap.add_argument('--arms', nargs='+', choices=list(ARMS), default=list(ARMS))
    ap.add_argument('--block-lengths', type=int, nargs='+', default=list(BLOCKS))
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if args.draws < 2 or args.pilot < 2 or len(set(args.arms)) != len(args.arms) or len(set(args.block_lengths)) != len(args.block_lengths) or any(L <= 0 for L in args.block_lengths):
        raise ValueError('Unique positive scopes and at least two pilot/test draws required')
    args.out.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), HERE / 'joint_cache.py', HERE.parent / 'selection_policy.py', PROTOCOL]
    spec = dict(arms=args.arms, block_lengths=args.block_lengths, pilot=args.pilot, draws=args.draws,
                pilot_seed=PILOT_SEED, test_seed=TEST_SEED, comparison_tolerance=TOL,
                code_protocol_sha256={str(p): sha(p) for p in files})
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    before = args.out / 'protocol_before_run.json'
    if before.exists():
        if not args.resume or json.loads(before.read_text()) != spec:
            raise FileExistsError('Existing or incompatible protocol; use a new output directory')
    else:
        before.write_text(json.dumps(spec, indent=2) + '\n')
    results = []
    for arm in args.arms:
        cache = load_joint_cache(*ARMS[arm])
        (args.out / f'input_audit_{arm}.json').write_text(json.dumps(cache['audit'], indent=2) + '\n')
        for L in args.block_lengths:
            results.append(run_one(cache, arm, L, args, signature))
    rows = []
    for result in results:
        effect = result['statistics'][result['main_statistic']]
        rows.append({**{k: result[k] for k in ('arm', 'block_length', 'n_cells', 'n_subjects', 'n_seeds', 'n_subject_days', 'n_methods', 'pilot_replicates', 'exceedances', 'replicates', 'p_mc', 'mc_tail_ci_lo', 'mc_tail_ci_hi', 'rejection_rate', 'n_runs', 'n_unchanged_short_runs', 'fraction_trials_in_unchanged_short_runs')},
                     'observed_forced_switch_gap': effect['observed'],
                     'reference_bootstrap_mean': effect['bootstrap_mean'],
                     'reference_bootstrap_q025': effect['bootstrap_q025'],
                     'reference_bootstrap_q975': effect['bootstrap_q975'],
                     'observed_minus_reference_mean': effect['observed_minus_bootstrap_mean'],
                     'reference_with_r0_switch_gap': result['statistics']['with_r0__session_minus_source']['fitted_reference'],
                     'observed_with_r0_switch_gap': result['statistics']['with_r0__session_minus_source']['observed']})
    summary = pd.DataFrame(rows)
    summary['p_holm'] = holm(summary.p_mc)
    summary['comparison_family_size'] = len(summary)
    summary.to_csv(args.out / 'summary.csv', index=False)
    for path, expected in spec['code_protocol_sha256'].items():
        if sha(path) != expected:
            raise RuntimeError(f'Implementation changed during run: {path}')
    canonical = (set(args.arms) == set(ARMS) and set(args.block_lengths) == set(BLOCKS)
                 and args.draws == 9999 and args.pilot == 4096)
    manifest = dict(status='complete', matches_frozen_protocol=canonical, signature=signature, specification=spec,
                    interpretation='Exploratory conditional comparison; Holm covers this family only.',
                    outputs_sha256={str(p.relative_to(args.out)): sha(p) for p in args.out.rglob('*')
                                    if p.is_file() and p.name != 'manifest.json'})
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
