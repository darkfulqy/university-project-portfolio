"""Population observer helpers for replayed fly cable cells.

Each cell's full transmembrane current is projected through its own lead field
onto one shared electrode array in native Male CNS coordinates. Signed
potentials are summed without per-cell scaling; the reference and baseline are
applied identically afterwards. Results are model-derived extracellular
contributions of explicitly selected cells, not a whole-CNS LFP and not
empirically validated. Nothing here converts activity to spikes or feeds
potentials back into the CNS source.
"""
from itertools import combinations

import numpy as np


class GateFailure(RuntimeError):
    """A validation gate failed; evidence has been or must be saved."""


def sha256_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- inputs

def validate_time_axis(time_s, sampling_rate_hz, duration_s, atol_s=1e-9):
    """Require a finite, strictly increasing, uniform axis from 0 to duration."""
    t = np.asarray(time_s, dtype=float)
    rate = float(sampling_rate_hz)
    if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all():
        raise GateFailure('time_s must be a finite 1D array with at least two samples')
    if not np.isfinite(rate) or rate <= 0:
        raise GateFailure('sampling rate must be positive and finite')
    step = np.diff(t)
    if np.any(step <= 0):
        raise GateFailure('time_s must be strictly increasing')
    dt = 1.0 / rate
    deviation = float(np.max(np.abs(step - dt)))
    if deviation > atol_s:
        raise GateFailure(f'time_s is not uniform at {rate} Hz (max step deviation {deviation} s)')
    if abs(t[0]) > atol_s or abs(t[-1] - float(duration_s)) > atol_s:
        raise GateFailure(f'time_s must span 0..{duration_s} s, got {t[0]}..{t[-1]}')
    expected = int(round(float(duration_s) * rate)) + 1
    if t.size != expected:
        raise GateFailure(f'expected {expected} samples, got {t.size}')
    return {'dt_s': dt, 'n_samples': int(t.size), 'max_abs_step_deviation_s': deviation,
            'first_s': float(t[0]), 'last_s': float(t[-1])}


def column_lookup(recorded_ids, required_ids):
    """Map body IDs to source columns; reject duplicate or missing columns."""
    ids = np.asarray(recorded_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise GateFailure('recorded cell IDs must be a 1D integer array')
    unique, counts = np.unique(ids, return_counts=True)
    if np.any(counts > 1):
        raise GateFailure(f'duplicate source columns: {unique[counts > 1][:10].tolist()}')
    lookup = {int(n): i for i, n in enumerate(ids)}
    missing = sorted({int(n) for n in np.asarray(required_ids).ravel()} - set(lookup))
    if missing:
        raise GateFailure(f'{len(missing)} required cells lack source traces: {missing[:10]}')
    return lookup


# -------------------------------------------------------------- geometry

def _xyz(name, values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise GateFailure(f'{name} must be a nonempty finite (N, 3) array')
    return a


def validate_geometry(starts, ends, points, diameters=None, min_clearance_um=None):
    """Reject inputs for which NaN comparisons would silently report no cutoff."""
    starts = _xyz('segment starts', starts); ends = _xyz('segment ends', ends)
    points = _xyz('electrode points', points)
    if starts.shape != ends.shape:
        raise GateFailure('segment starts and ends must have the same shape')
    if np.any(np.linalg.norm(ends - starts, axis=1) <= 0):
        raise GateFailure('zero-length segments are not allowed')
    d = None
    if diameters is not None:
        d = np.asarray(diameters, dtype=float)
        if d.shape != (starts.shape[0],) or not np.isfinite(d).all() or np.any(d <= 0):
            raise GateFailure('one positive finite diameter per segment is required')
    if min_clearance_um is not None:
        c = float(min_clearance_um)
        if not np.isfinite(c) or c < 0:
            raise GateFailure('minimum clearance must be finite and nonnegative')
    return starts, ends, points, d


def finite_centerline_distance(starts, ends, points):
    """Distance from each point to each finite segment centerline, (points, segments)."""
    starts, ends, points, _ = validate_geometry(starts, ends, points)
    direction = ends - starts
    fraction = np.clip(np.einsum('psi,si->ps', points[:, None, :] - starts, direction)
                       / (direction * direction).sum(axis=1), 0, 1)
    nearest = starts + fraction[:, :, None] * direction
    return np.linalg.norm(points[:, None, :] - nearest, axis=2)


def line_kernel_cutoff_mask(starts, ends, diameters, points):
    """Pairs where the installed LFPykit line-source kernel clamps distance.

    Uses the installed lfpykit.lfpcalc helpers exactly as
    ``_calc_lfp_linesource`` does: squared perpendicular distance to the
    *infinite* line through the segment compared with ``r_limit**2``, where
    ``RecExtElectrode`` sets ``r_limit = d / 2`` for 1D diameters.
    """
    from lfpykit import lfpcalc
    starts, ends, points, d = validate_geometry(starts, ends, points, diameters)
    r_limit = d / 2
    xs, ys, zs = starts.T; xe, ye, ze = ends.T
    delta = lfpcalc._deltaS_calc(xs, xe, ys, ye, zs, ze)
    mask = np.zeros((len(points), len(starts)), dtype=bool)
    for k, (x, y, z) in enumerate(points):
        h_ = lfpcalc._h_calc(xs, xe, ys, ye, zs, ze, delta, x, y, z)
        r2 = lfpcalc._r2_calc(xe, ye, ze, x, y, z, h_)
        mask[k] = r2 < r_limit * r_limit
    return mask


def geometry_gate(starts, ends, diameters, electrodes, min_clearance_um):
    """Finite clearance and actual kernel-cutoff checks for one cell geometry."""
    validate_geometry(starts, ends, electrodes, diameters, min_clearance_um)
    distance = finite_centerline_distance(starts, ends, electrodes)
    radius = np.asarray(diameters, float) / 2
    cutoff = line_kernel_cutoff_mask(starts, ends, diameters, electrodes)
    below = distance < float(min_clearance_um)
    inside = distance <= radius[None, :]
    record = {
        'n_electrodes': int(len(electrodes)), 'n_segments': int(len(starts)),
        'minimum_clearance_um_required': float(min_clearance_um),
        'min_finite_centerline_distance_um': float(distance.min()),
        'min_distance_over_segment_radius': float(np.min(distance / radius[None, :])),
        'max_segment_radius_um': float(radius.max()),
        'pairs_below_minimum_clearance': int(below.sum()),
        'pairs_inside_segment_radius': int(inside.sum()),
        'line_kernel_radius_cutoff_pairs': int(cutoff.sum()),
        'line_kernel_cutoff_pair_indices_electrode_segment': np.argwhere(cutoff).tolist(),
    }
    record['passed'] = (record['pairs_below_minimum_clearance'] == 0 and
                        record['pairs_inside_segment_radius'] == 0 and
                        record['line_kernel_radius_cutoff_pairs'] == 0)
    return record


# ------------------------------------------------------------ aggregation

def signed_sum(contributions):
    """Default aggregator: plain signed sum of a label -> (n_time, n_channels) mapping."""
    return np.sum(np.stack(list(contributions.values())), axis=0)


def aggregate_contributions(potentials, labels, groups, aggregator=signed_sum, rtol=1e-12):
    """Sum signed single-cell potentials into group and total contributions.

    ``potentials``: (n_cells, n_time, n_channels) unreferenced model potentials
    computed in one conductor on one electrode array and time axis; never
    normalized. ``labels``: unique per-cell labels. ``groups``: one group label
    per cell (e.g. cell type). ``aggregator`` receives an ordered mapping
    label -> (n_time, n_channels) and returns their (n_time, n_channels) sum; it
    is called once for the total and once per group. This matches the input
    contract of ``flyobserver.hybrid_postprocess.aggregate``, so an adapter such
    as ``lambda m: aggregate(m, time_s, electrodes_um, ref, fresh_dir()).unreferenced_uV``
    can be supplied. Every result is checked against the plain signed sum, so
    no aggregator can rescale or drop contributions.
    """
    stack = np.asarray(potentials, dtype=float)
    labels = [str(x) for x in labels]; groups = list(groups)
    if (stack.ndim != 3 or stack.shape[0] != len(labels) or len(groups) != len(labels)
            or not np.isfinite(stack).all()):
        raise GateFailure('potentials must be finite (n_cells, n_time, n_channels) with one label and group per cell')
    if len(set(labels)) != len(labels):
        raise GateFailure('cell labels must be unique')
    order = list(dict.fromkeys(groups))
    group = {}
    for name in order:
        members = {l: stack[i] for i, (l, g) in enumerate(zip(labels, groups)) if g == name}
        group[name] = np.asarray(aggregator(members), dtype=float)
    total = np.asarray(aggregator(dict(zip(labels, stack))), dtype=float)
    reference = np.sum(stack, axis=0)
    scale = max(float(np.sum(np.max(np.abs(stack), axis=(1, 2)))), np.finfo(float).tiny)
    for label, value in [('total', total)] + list(group.items()):
        if value.shape != stack.shape[1:]:
            raise GateFailure(f'aggregator returned shape {value.shape} for {label}')
    error = float(np.max(np.abs(total - reference))) / scale
    group_error = float(np.max(np.abs(sum(group.values()) - reference))) / scale
    if error > rtol or group_error > rtol:
        raise GateFailure(f'aggregator is not a signed sum (relative error {error}, groups {group_error})')
    return {'total': total, 'groups': group, 'group_order': order,
            'aggregator_relative_error': error, 'group_sum_relative_error': group_error}


def summation_order_check(stack):
    """Relative spread between forward, reverse and vectorized signed sums."""
    stack = np.asarray(stack, float)
    forward = np.zeros(stack.shape[1:]); reverse = np.zeros(stack.shape[1:])
    for x in stack:
        forward = forward + x
    for x in stack[::-1]:
        reverse = reverse + x
    vector = np.sum(stack, axis=0)
    scale = max(float(np.sum(np.max(np.abs(stack), axis=(1, 2)))), np.finfo(float).tiny)
    return {'forward_vs_reverse': float(np.max(np.abs(forward - reverse))) / scale,
            'forward_vs_vectorized': float(np.max(np.abs(forward - vector))) / scale,
            'scale_sum_of_cell_max_abs_uV': scale}


def reference_and_baseline(potential, reference_index, baseline_mask):
    """Subtract the reference channel, then the per-channel baseline mean.

    Works on (..., n_time, n_channels). Both steps are linear, so applying them
    to a sum equals summing processed contributions.
    """
    x = np.asarray(potential, float)
    mask = np.asarray(baseline_mask, bool)
    if mask.ndim != 1 or mask.shape[0] != x.shape[-2] or not mask.any():
        raise GateFailure('baseline mask must select at least one sample on the time axis')
    referenced = x - x[..., reference_index:reference_index + 1]
    corrected = referenced - referenced[..., mask, :].mean(axis=-2, keepdims=True)
    return referenced, corrected


# ------------------------------------------------------------- diagnostics

def current_balance(imem_nA):
    imem = np.asarray(imem_nA, float)
    balance = float(np.max(np.abs(imem.sum(axis=1))))
    scale = float(np.max(np.sum(np.abs(imem), axis=1)))
    return {'max_abs_current_balance_nA': balance, 'max_sum_abs_membrane_current_nA': scale,
            'relative_current_balance': balance / max(scale, np.finfo(float).tiny)}


def relative_max_error(test, base, floor):
    """max|test-base| / max(max|base|, floor).

    The floor only prevents division by a numerically zero reference; it is a
    configured absolute level (the blank tolerance) and is reported when used.
    """
    test = np.asarray(test, float); base = np.asarray(base, float)
    if test.shape != base.shape:
        raise GateFailure(f'shape mismatch {test.shape} vs {base.shape}')
    absolute = float(np.max(np.abs(test - base)))
    peak = float(np.max(np.abs(base)))
    denominator = max(peak, float(floor))
    return {'absolute_max_error': absolute, 'reference_max_abs': peak,
            'denominator': denominator, 'floor': float(floor), 'floor_active': peak < float(floor),
            'relative_max_error': absolute / denominator}


def cancellation_ratio(contributions, floor, channel_mask=None, time_mask=None):
    """|sum_i phi_i| / sum_i |phi_i| per time and channel.

    Values lie in [0, 1]: 1 means all nonzero contributions share a sign, 0 means
    complete cancellation. Entries whose denominator is <= floor are undefined
    (NaN) and counted rather than silently dropped.
    """
    x = np.asarray(contributions, float)
    signed = np.abs(x.sum(axis=0))
    unsigned = np.abs(x).sum(axis=0)
    defined = unsigned > float(floor)
    ratio = np.full(signed.shape, np.nan)
    ratio[defined] = signed[defined] / unsigned[defined]
    tmask = np.ones(x.shape[1], bool) if time_mask is None else np.asarray(time_mask, bool)
    cmask = np.ones(x.shape[2], bool) if channel_mask is None else np.asarray(channel_mask, bool)
    window = ratio[np.ix_(tmask, cmask)]
    valid = window[np.isfinite(window)]
    summary = {'floor_uV': float(floor), 'entries': int(window.size),
               'defined_entries': int(valid.size), 'undefined_entries': int(window.size - valid.size),
               'quantiles_0_5_50_95_100': (np.quantile(valid, [0, .05, .5, .95, 1]).tolist()
                                           if valid.size else None),
               'per_channel_median': [float(np.nanmedian(c)) if np.isfinite(c).any() else None
                                      for c in window.T]}
    return ratio, summary


def spectral_decomposition(contributions, sampling_rate_hz, total=None):
    """One-sided periodograms of the total, auto terms and pairwise cross terms.

    Every signal gets the same preprocessing: subtract its own time mean, apply
    one Hann window, rfft, and one scaling vector. The total is computed from the
    separately preprocessed summed signal (``total`` or the signed sum), so the
    identity P_total = sum_i P_ii + 2 sum_{i<j} Re S_ij checks that pipeline.
    Shapes: contributions (n_cells, n_time, n_channels); outputs (n_freq, n_channels).
    """
    x = np.asarray(contributions, float)
    fs = float(sampling_rate_hz)
    summed = x.sum(axis=0) if total is None else np.asarray(total, float)
    n = x.shape[1]
    window = np.hanning(n)
    scale = np.full(n // 2 + 1, 2.0 / (fs * np.sum(window ** 2)))
    scale[0] /= 2
    if n % 2 == 0:
        scale[-1] /= 2

    def transform(y):
        return np.fft.rfft((y - y.mean(axis=-2, keepdims=True)) * window[:, None], axis=-2)

    X = transform(x)
    T = transform(summed)
    total_power = (np.abs(T) ** 2) * scale[:, None]
    auto = (np.abs(X) ** 2) * scale[None, :, None]
    pairs = list(combinations(range(x.shape[0]), 2))
    cross = np.array([2 * np.real(X[i] * np.conj(X[j])) * scale[:, None] for i, j in pairs]) \
        if pairs else np.zeros((0,) + total_power.shape)
    reconstructed = auto.sum(axis=0) + cross.sum(axis=0)
    denominator = max(float(np.max(total_power)), float(np.max(auto.sum(axis=0))), np.finfo(float).tiny)
    identity = float(np.max(np.abs(total_power - reconstructed))) / denominator
    return {'frequency_hz': np.fft.rfftfreq(n, 1 / fs), 'total_power': total_power,
            'auto_power': auto, 'cross_terms': cross, 'pairs': pairs,
            'identity_relative_max_error': identity,
            'preprocessing': 'per-signal mean removal; Hann window; rfft; one-sided periodogram scaling 2/(fs*sum(w^2)), DC and Nyquist not doubled'}


def value_at_frequency(frequency_hz, spectrum, target_hz):
    """Nearest-bin value; a descriptor only, not a validation statistic."""
    k = int(np.argmin(np.abs(np.asarray(frequency_hz) - target_hz)))
    return float(np.asarray(frequency_hz)[k]), np.asarray(spectrum)[k]
