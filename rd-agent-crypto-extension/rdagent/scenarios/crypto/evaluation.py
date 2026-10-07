"""Cross-sectional RankIC evaluation for 5-minute crypto factors.

Pure pandas/numpy module (no ``rdagent`` imports) implementing section 4 of ``docs/crypto_5m/DESIGN.md``.

All series/frames are indexed by a ``MultiIndex`` named ``("datetime", "instrument")`` where ``datetime`` is tz-naive
UTC bar-open time on a ``bar_minutes`` grid.  Spearman correlations are computed as average ranks followed by Pearson,
vectorised over timestamps with ``groupby``/``bincount`` so the full panel (~2.2M rows) evaluates in seconds.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

_METRIC_KEYS = (
    "n_ts",
    "ic_mean",
    "ic_std",
    "ic_ir",
    "ic_tstat",
    "ic_tstat_naive",
    "ic_tstat_dynamic",
    "ic_pos_rate",
    "coverage",
    "rank_autocorr_1",
    "q_spread_bp_gross",
    "q_spread_bp_net",
)
_SEGMENTS = ("train", "valid", "test")
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation configuration (see DESIGN.md section 4.1).

    Segment bounds are ISO date/datetime strings interpreted as inclusive tz-naive UTC.  A date-only end bound
    (``"2026-06-30"``) covers the whole day; an explicit datetime end bound is used as-is.
    """

    bar_minutes: int = 5
    horizons_minutes: tuple[int, ...] = (5, 15)
    primary_horizon_minutes: int = 15
    train: tuple[str, str] = ("2026-02-25", "2026-06-30")
    valid: tuple[str, str] = ("2026-07-01", "2026-07-31")
    test: tuple[str, str] = ("2026-08-01", "2026-08-23 23:59:00")
    purge_minutes: int = 15
    min_assets: int = 30
    n_quantiles: int = 5
    fee_bp_per_side: float = 10.0
    slippage_bp_per_side: float = 5.0
    gate_min_tstat: float = 3.0
    gate_min_abs_ic: float = 0.01
    gate_min_coverage: float = 0.5
    gate_max_corr: float = 0.7
    gate_min_ir_gain: float = 0.0
    gate_max_abs_ic: float = 0.25
    """Implausibility cap: a valid |ic_mean| above this at the primary horizon is treated as look-ahead (gate R0)."""

    @property
    def primary_key(self) -> str:
        """Horizon key of the primary horizon, e.g. ``"15m"``."""
        return f"{self.primary_horizon_minutes}m"

    @property
    def nw_lags(self) -> int:
        """Newey-West lag count for ``ic_tstat``: two times the longest horizon in bars (overlapping targets)."""
        return max(0, 2 * int(max(self.horizons_minutes)) // int(self.bar_minutes))

    def to_dict(self) -> dict[str, Any]:
        """Plain-python dict (tuples become lists) suitable for JSON/pickle."""
        out: dict[str, Any] = {}
        for k, v in dataclasses.asdict(self).items():
            out[k] = list(v) if isinstance(v, tuple) else v
        return out


# --------------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------------


def _py(x: Any) -> Any:
    """Convert numpy scalars to plain python scalars (NaN stays ``float('nan')``)."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    return x


def _parse_bound(value: str | pd.Timestamp, *, end: bool) -> pd.Timestamp:
    """Parse an inclusive segment bound into a tz-naive UTC Timestamp."""
    if isinstance(value, str) and end and _DATE_ONLY_RE.match(value.strip()):
        ts = pd.Timestamp(value.strip()) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    else:
        ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _datetime_values(index: pd.Index) -> np.ndarray:
    """Return the datetime level of a ``(datetime, instrument)`` MultiIndex as ``datetime64[ns]`` ndarray."""
    if isinstance(index, pd.MultiIndex):
        level = "datetime" if "datetime" in (index.names or []) else 0
        values = index.get_level_values(level)
    else:
        values = index
    dt = pd.DatetimeIndex(values)
    if dt.tz is not None:
        dt = dt.tz_convert("UTC").tz_localize(None)
    return dt.to_numpy(dtype="datetime64[ns]")


def _codes(index: pd.Index) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Integer group codes for the datetime level plus the sorted unique datetimes."""
    dt = _datetime_values(index)
    codes, uniques = pd.factorize(dt, sort=True)
    return codes.astype(np.int64), pd.DatetimeIndex(uniques, name="datetime")


def _group_rank(values: np.ndarray, codes: np.ndarray, pct: bool = False) -> np.ndarray:
    """Average rank of ``values`` within each group defined by ``codes``."""
    if len(values) == 0:
        return np.empty(0, dtype=float)
    s = pd.Series(values)
    return s.groupby(codes, sort=False).rank(method="average", pct=pct).to_numpy(dtype=float)


def _corr_from_ranks(rx: np.ndarray, ry: np.ndarray, c: np.ndarray, n_groups: int, min_assets: int) -> np.ndarray:
    """Pearson correlation per group of two already-ranked arrays (``c`` = group code per row)."""
    out = np.full(n_groups, np.nan)
    if c.size == 0:
        return out
    n = np.bincount(c, minlength=n_groups).astype(float)
    sx = np.bincount(c, weights=rx, minlength=n_groups)
    sy = np.bincount(c, weights=ry, minlength=n_groups)
    sxy = np.bincount(c, weights=rx * ry, minlength=n_groups)
    sxx = np.bincount(c, weights=rx * rx, minlength=n_groups)
    syy = np.bincount(c, weights=ry * ry, minlength=n_groups)
    num = n * sxy - sx * sy
    var = (n * sxx - sx * sx) * (n * syy - sy * sy)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.sqrt(np.clip(var, 0.0, None))
        corr = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    corr[n < max(int(min_assets), 2)] = np.nan
    return np.clip(corr, -1.0, 1.0)


def _spearman_by_group(x: np.ndarray, y: np.ndarray, codes: np.ndarray, n_groups: int, min_assets: int) -> np.ndarray:
    """Per-group Spearman correlation (average ranks then Pearson).

    Returns an array of length ``n_groups`` with NaN for groups with fewer than ``min_assets`` non-NaN pairs or a
    constant ranking (undefined correlation).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    c = codes[mask]
    if c.size == 0:
        return np.full(n_groups, np.nan)
    return _corr_from_ranks(_group_rank(x[mask], c), _group_rank(y[mask], c), c, n_groups, min_assets)


def _shift_on_grid(series: pd.Series, bar_minutes: int, periods: Sequence[int]) -> dict[int, np.ndarray]:
    """Shift a ``(datetime, instrument)`` series per instrument on the complete bar grid.

    For each instrument the series is re-indexed onto the full ``bar_minutes`` grid between its first and last
    datetime, shifted by ``p`` bars (``p > 0`` = lag, ``p < 0`` = lead; ``pandas.shift`` semantics, no fill) and
    re-indexed back to the original rows.  Returns ``{p: values aligned with series.index}``.
    """
    n = len(series)
    out = {p: np.full(n, np.nan) for p in periods}
    if n == 0:
        return out
    dt_all = _datetime_values(series.index)
    vals = series.to_numpy(dtype=float)
    freq = pd.Timedelta(minutes=int(bar_minutes))
    inst_level = "instrument" if "instrument" in (series.index.names or []) else 1
    groups = series.groupby(level=inst_level, sort=False).indices
    for pos in groups.values():
        pos = np.asarray(pos)
        dts = pd.DatetimeIndex(dt_all[pos])
        s = pd.Series(vals[pos], index=dts)
        if not s.index.is_monotonic_increasing:
            s = s.sort_index()
        if s.index.has_duplicates:
            s = s[~s.index.duplicated(keep="last")]
        grid = pd.date_range(s.index[0], s.index[-1], freq=freq)
        g = s.reindex(grid)
        for p in periods:
            out[p][pos] = g.shift(p).reindex(dts).to_numpy(dtype=float)
    return out


def _empty_metrics(n_ts: int = 0) -> dict[str, Any]:
    d: dict[str, Any] = {k: float("nan") for k in _METRIC_KEYS}
    d["n_ts"] = int(n_ts)
    return d


def _newey_west_tstat(ic: np.ndarray, nw_lags: int) -> float:
    """t-statistic of the mean of ``ic`` with a Newey-West (Bartlett kernel) long-run variance.

    Per-bar ICs of overlapping targets are autocorrelated, so the naive ``mean / (std / sqrt(n))`` is inflated; with
    ``nw_lags`` >= the target overlap in bars the Bartlett estimate ``gamma_0 + 2 * sum_l (1 - l / (L + 1)) gamma_l``
    corrects most of it.  Returns NaN when the long-run variance is not positive (constant series).
    """
    n = int(ic.size)
    if n < 2:
        return float("nan")
    mean = float(ic.mean())
    d = ic - mean
    v = float((d * d).sum()) / n
    lags = min(int(nw_lags), n - 1)
    for lag in range(1, lags + 1):
        v += 2.0 * (1.0 - lag / (lags + 1.0)) * float((d[lag:] * d[:-lag]).sum()) / n
    if not (v > 0.0):
        return float("nan")
    return mean / math.sqrt(v / n)


def _ic_stats(ic: np.ndarray, nw_lags: int = 0) -> dict[str, Any]:
    """Summary statistics of a per-timestamp IC series (NaN entries are dropped first).

    ``ic_tstat`` is the Newey-West t-statistic with ``nw_lags`` Bartlett lags (``nw_lags=0`` reduces to the naive
    ``ic_ir * sqrt(n_ts)``, which is always reported as ``ic_tstat_naive``).
    """
    ic = ic[~np.isnan(ic)]
    n = int(ic.size)
    d = _empty_metrics(n)
    if n == 0:
        return d
    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if n > 1 else float("nan")
    if std == std and std < 1e-12:  # constant IC series (e.g. the factor equals the target): exact zero, not 1e-17
        std = 0.0
    ir = mean / std if (std == std and std > 0) else float("nan")
    d["ic_mean"] = mean
    d["ic_std"] = std
    d["ic_ir"] = ir
    d["ic_tstat_naive"] = ir * math.sqrt(n) if ir == ir else float("nan")
    d["ic_tstat"] = _newey_west_tstat(ic, nw_lags) if (nw_lags > 0 and ir == ir) else d["ic_tstat_naive"]
    d["ic_pos_rate"] = float((ic > 0).mean())
    return d


def _train_demeaned(factor_vals: np.ndarray, index: pd.Index, train_rows: np.ndarray) -> np.ndarray:
    """``factor - per-instrument mean over the train rows`` (the dynamic, within-instrument part of a factor).

    Instruments without a finite train value get NaN (their static level is unknown).
    """
    inst_level = "instrument" if "instrument" in (index.names or []) else -1
    inst_codes, uniques = pd.factorize(index.get_level_values(inst_level))
    inst_codes = inst_codes.astype(np.int64)
    n_inst = len(uniques)
    ok = np.isfinite(factor_vals) & np.asarray(train_rows, dtype=bool)
    cnt = np.bincount(inst_codes[ok], minlength=n_inst).astype(float)
    tot = np.bincount(inst_codes[ok], weights=factor_vals[ok], minlength=n_inst)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1.0), np.nan)
    level = means[inst_codes]
    resid = factor_vals - level
    # a per-instrument constant must demean to an exact zero, not to rounding noise that re-encodes the ranking
    with np.errstate(invalid="ignore"):
        tiny = np.abs(resid) <= 1e-9 * np.maximum(np.abs(level), np.abs(factor_vals))
    resid[tiny] = 0.0
    return resid


def _nanmean(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else float("nan")


# --------------------------------------------------------------------------------------------------------------------
# public API (DESIGN.md 4.1)
# --------------------------------------------------------------------------------------------------------------------


def forward_returns(close: pd.Series, cfg: EvalConfig) -> pd.DataFrame:
    """Forward simple returns per horizon on the per-instrument bar grid.

    ``ret_h[t, i] = close[t + h, i] / close[t, i] - 1`` where ``t + h`` must exist as a bar (no forward fill: a
    missing future bar yields NaN, never the next available bar).  Returns a DataFrame with the same index as
    ``close`` and columns ``ret_5m``, ``ret_15m``, ... in ``cfg.horizons_minutes`` order.
    """
    periods: list[int] = []
    for h in cfg.horizons_minutes:
        if h % cfg.bar_minutes != 0 or h <= 0:
            raise ValueError(f"horizon {h}m is not a positive multiple of bar_minutes={cfg.bar_minutes}")
        periods.append(-(h // cfg.bar_minutes))
    shifted = _shift_on_grid(close, cfg.bar_minutes, periods)
    c = close.to_numpy(dtype=float)
    data = {}
    for h, p in zip(cfg.horizons_minutes, periods):
        with np.errstate(invalid="ignore", divide="ignore"):
            data[f"ret_{h}m"] = shifted[p] / c - 1.0
    return pd.DataFrame(data, index=close.index)


def segment_masks(datetimes: pd.Index, cfg: EvalConfig) -> dict[str, np.ndarray]:
    """Boolean masks (aligned with ``datetimes``) for ``train``/``valid``/``test`` origins.

    An origin ``t`` belongs to a segment when ``start <= t <= end`` and ``t + max(horizons) + purge <= end`` so
    that no target crosses into the next segment.  All comparisons are tz-naive UTC.
    """
    dt = _datetime_values(datetimes)
    tail = pd.Timedelta(minutes=int(max(cfg.horizons_minutes)) + int(cfg.purge_minutes))
    out: dict[str, np.ndarray] = {}
    for seg in _SEGMENTS:
        start_s, end_s = getattr(cfg, seg)
        start = _parse_bound(start_s, end=False).to_datetime64()
        end = _parse_bound(end_s, end=True)
        latest = (end - tail).to_datetime64()
        out[seg] = (dt >= start) & (dt <= end.to_datetime64()) & (dt <= latest)
    return out


def rankic_by_timestamp(factor: pd.Series, ret: pd.Series, min_assets: int) -> pd.Series:
    """Cross-sectional Spearman IC per datetime (index = datetime).

    Datetimes with fewer than ``min_assets`` instruments having both values (or an undefined correlation) are
    skipped.  ``ret`` is aligned to ``factor``'s index.
    """
    if len(factor) == 0:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="datetime"), name="ic")
    if not ret.index.equals(factor.index):
        ret = ret.reindex(factor.index)
    codes, uniques = _codes(factor.index)
    corr = _spearman_by_group(factor.to_numpy(dtype=float), ret.to_numpy(dtype=float), codes, len(uniques), min_assets)
    s = pd.Series(corr, index=uniques, name="ic")
    return s.dropna()


def cross_sectional_rank(factor: pd.Series) -> pd.Series:
    """Per-datetime percentile rank in ``(0, 1]`` minus 0.5 (centred); NaN stays NaN."""
    if len(factor) == 0:
        return factor.astype(float).copy()
    codes, _ = _codes(factor.index)
    vals = factor.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    mask = np.isfinite(vals)
    if mask.any():
        out[mask] = _group_rank(vals[mask], codes[mask], pct=True) - 0.5
    return pd.Series(out, index=factor.index, name=factor.name)


def _composite_from_ranks(ranks: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Signed mean over columns of a (rows x factors) matrix of centred ranks, skipping NaN."""
    if ranks.ndim != 2 or ranks.shape[1] == 0:
        return np.full(ranks.shape[0] if ranks.ndim == 2 else 0, np.nan)
    signed = ranks * signs.reshape(1, -1)
    valid = ~np.isnan(signed)
    cnt = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, np.where(valid, signed, 0.0).sum(axis=1) / np.where(cnt > 0, cnt, 1), np.nan)


def composite(library: pd.DataFrame, signs: dict[str, int]) -> pd.Series:
    """Equal-weight composite: mean over columns of ``sign * cross_sectional_rank(col)`` (NaN skipped)."""
    if library is None or library.shape[1] == 0:
        return pd.Series(dtype=float, name="composite")
    ranks = np.column_stack([cross_sectional_rank(library[col]).to_numpy(dtype=float) for col in library.columns])
    sgn = np.array([float(int(signs.get(col, 1)) or 1) for col in library.columns])
    return pd.Series(_composite_from_ranks(ranks, sgn), index=library.index, name="composite")


def _per_timestamp_series(factor: pd.Series, rets: pd.DataFrame, cfg: EvalConfig) -> tuple[pd.DatetimeIndex, dict]:
    """Compute every per-datetime statistic once over the whole panel (segments are selected afterwards)."""
    if not factor.index.equals(rets.index):
        factor = factor.reindex(rets.index)
    codes, uniques = _codes(rets.index)
    g = len(uniques)
    f = factor.to_numpy(dtype=float)
    f_ok = np.isfinite(f)

    n_all = np.bincount(codes, minlength=g).astype(float)
    n_f = np.bincount(codes[f_ok], minlength=g).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        coverage = np.where(n_all > 0, n_f / np.where(n_all > 0, n_all, 1.0), np.nan)

    lag = _shift_on_grid(factor, cfg.bar_minutes, [1])[1]
    autocorr = _spearman_by_group(f, lag, codes, g, cfg.min_assets)

    ic: dict[str, np.ndarray] = {}
    spread: dict[str, np.ndarray] = {}
    nq = int(cfg.n_quantiles)
    for h in cfg.horizons_minutes:
        key = f"{h}m"
        r = rets[f"ret_{h}m"].to_numpy(dtype=float)
        mask = f_ok & np.isfinite(r)
        c = codes[mask]
        sp = np.full(g, np.nan)
        if c.size == 0:
            ic[key] = np.full(g, np.nan)
            spread[key] = sp
            continue
        rr = r[mask]
        rf = _group_rank(f[mask], c)  # shared by the IC and the quantile spread
        ic[key] = _corr_from_ranks(rf, _group_rank(rr, c), c, g, cfg.min_assets)
        n_pairs = np.bincount(c, minlength=g).astype(float)
        pct = rf / n_pairs[c]
        q = np.clip(np.ceil(pct * nq - 1e-12), 1, nq)
        top, bot = q == nq, q == 1
        n_top = np.bincount(c[top], minlength=g).astype(float)
        n_bot = np.bincount(c[bot], minlength=g).astype(float)
        s_top = np.bincount(c[top], weights=rr[top], minlength=g)
        s_bot = np.bincount(c[bot], weights=rr[bot], minlength=g)
        ok = (n_top > 0) & (n_bot > 0) & (n_pairs >= max(int(cfg.min_assets), 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_top = s_top / np.where(n_top > 0, n_top, 1.0)
            mean_bot = s_bot / np.where(n_bot > 0, n_bot, 1.0)
            sp = np.where(ok, mean_top - mean_bot, np.nan)
        spread[key] = sp * 1e4

    # Static-component control (gate R1): IC of the within-instrument demeaned factor at the primary horizon.  A
    # static per-instrument ranking has a persistent IC series whose naive t-stat is inflated; demeaning over the
    # train rows removes that static part so ``ic_tstat_dynamic`` measures the time-varying information only.
    train_rows = segment_masks(rets.index, cfg)["train"]
    f_dyn = _train_demeaned(f, rets.index, train_rows)
    r_primary = rets[f"ret_{cfg.primary_horizon_minutes}m"].to_numpy(dtype=float)
    ic_dynamic = _spearman_by_group(f_dyn, r_primary, codes, g, cfg.min_assets)
    return uniques, {"coverage": coverage, "autocorr": autocorr, "ic": ic, "spread": spread, "ic_dynamic": ic_dynamic}


def factor_metrics(factor: pd.Series, rets: pd.DataFrame, cfg: EvalConfig) -> dict:
    """Per segment x horizon metrics dict ``{segment: {"5m": {...}, "15m": {...}}}``.

    Metrics: ``n_ts`` (datetimes with a valid IC), ``ic_mean``, ``ic_std`` (ddof=1), ``ic_ir = ic_mean / ic_std``,
    ``ic_tstat`` (Newey-West t-statistic of ``ic_mean`` with ``cfg.nw_lags`` Bartlett lags, robust to the
    autocorrelation of overlapping targets), ``ic_tstat_naive = ic_ir * sqrt(n_ts)``, ``ic_tstat_dynamic``
    (Newey-West t-statistic of the IC of the factor demeaned per instrument over the train rows; defined at the
    primary horizon only, NaN elsewhere), ``ic_pos_rate``, ``coverage`` (mean over datetimes of non-NaN factor rows /
    panel rows at that datetime), ``rank_autocorr_1`` (mean cross-sectional Spearman between ``factor[t]`` and
    ``factor[t - 1 bar]``; 1.0 = static), ``q_spread_bp_gross`` (mean of top-quantile minus bottom-quantile mean
    forward return, in bp) and ``q_spread_bp_net``.  Only finite factor values count as observations (``inf`` is
    treated like NaN).

    Cost approximation: ``q_spread_bp_net = gross - turnover_estimate * 2 * (fee_bp + slippage_bp)`` where
    ``turnover_estimate = clip(1 - rank_autocorr_1, 0, 1)``.  This treats one minus the rank autocorrelation as the
    fraction of the long-short book that is rebalanced every bar and charges a round trip (two sides) on it; it is a
    crude proxy, not a backtest.  Undefined values are ``float('nan')``; ``n_ts`` is 0 when nothing is defined.
    ``factor`` is aligned to ``rets.index`` (the panel).
    """
    if len(rets) == 0:
        return {seg: {f"{h}m": _empty_metrics() for h in cfg.horizons_minutes} for seg in _SEGMENTS}
    uniques, per_ts = _per_timestamp_series(factor, rets, cfg)
    masks = segment_masks(uniques, cfg)
    cost = 2.0 * (float(cfg.fee_bp_per_side) + float(cfg.slippage_bp_per_side))
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for seg in _SEGMENTS:
        m = masks[seg]
        coverage = _nanmean(per_ts["coverage"][m])
        autocorr = _nanmean(per_ts["autocorr"][m])
        turnover = min(max(1.0 - autocorr, 0.0), 1.0) if autocorr == autocorr else float("nan")
        out[seg] = {}
        dyn = _ic_stats(per_ts["ic_dynamic"][m], cfg.nw_lags)["ic_tstat"]
        for h in cfg.horizons_minutes:
            key = f"{h}m"
            d = _ic_stats(per_ts["ic"][key][m], cfg.nw_lags)
            d["ic_tstat_dynamic"] = dyn if h == cfg.primary_horizon_minutes else float("nan")
            d["coverage"] = coverage
            d["rank_autocorr_1"] = autocorr
            gross = _nanmean(per_ts["spread"][key][m])
            d["q_spread_bp_gross"] = gross
            defined = gross == gross and turnover == turnover
            d["q_spread_bp_net"] = gross - turnover * cost if defined else float("nan")
            out[seg][key] = {k: _py(v) for k, v in d.items()}
    return out


def factor_sign(metrics: dict, cfg: EvalConfig) -> int:
    """Sign (+1/-1) of the train ``ic_mean`` at the primary horizon; zero or NaN -> +1."""
    try:
        v = metrics["train"][cfg.primary_key]["ic_mean"]
    except (KeyError, TypeError):
        return 1
    if v is None or v != v or v == 0:
        return 1
    return -1 if v < 0 else 1


def max_abs_corr_with_library(
    new: pd.Series, library: pd.DataFrame, datetimes_mask: np.ndarray, min_assets: int = 2
) -> tuple[float, str | None]:
    """Max |mean cross-sectional Spearman| between ``new`` and each library column over masked rows.

    ``datetimes_mask`` is a boolean array aligned with ``new.index`` rows (e.g. train|valid from
    :func:`segment_masks` applied to ``new.index.get_level_values("datetime")``).  Returns ``(0.0, None)`` when the
    library is empty or no correlation is defined.
    """
    if library is None or library.shape[1] == 0 or len(new) == 0:
        return 0.0, None
    mask = np.asarray(datetimes_mask, dtype=bool)
    if mask.shape[0] != len(new):
        raise ValueError(f"datetimes_mask length {mask.shape[0]} != len(new) {len(new)}")
    if not library.index.equals(new.index):
        library = library.reindex(new.index)
    sub_index = new.index[mask]
    codes, uniques = _codes(sub_index)
    g = len(uniques)
    x = new.to_numpy(dtype=float)[mask]
    x_ok = np.isfinite(x)
    cx, xs = codes[x_ok], x[x_ok]
    rx_full = _group_rank(xs, cx) if cx.size else np.empty(0)
    best, best_name = 0.0, None
    for col in library.columns:
        y = library[col].to_numpy(dtype=float)[mask]
        if cx.size == 0:
            break
        y_ok = np.isfinite(y[x_ok])
        if y_ok.all():  # common case: same rows -> the rank of ``new`` can be reused
            corr = _corr_from_ranks(rx_full, _group_rank(y[x_ok], cx), cx, g, min_assets)
        else:
            corr = _spearman_by_group(x, y, codes, g, min_assets)
        mean_corr = _nanmean(corr)
        if mean_corr == mean_corr and abs(mean_corr) > best:
            best, best_name = abs(mean_corr), str(col)
    return float(best), best_name


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "nan"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if fv != fv:
        return "nan"
    return f"{fv:.{nd}f}"


def gate(
    new_metrics: dict, max_corr: float, composite_before: dict | None, composite_after: dict, cfg: EvalConfig
) -> tuple[bool, list[str]]:
    """Deterministic acceptance gate on the ``valid`` segment at the primary horizon.

    R0: degenerate / look-ahead suspected: ``ic_std == 0`` (constant IC series, e.g. the factor equals the target) or
    ``|ic_mean| > gate_max_abs_ic`` (implausibly strong for a 5-minute cross-section).
    R1: ``|ic_mean| >= gate_min_abs_ic``, ``sign(valid ic_mean) == sign(train ic_mean)``,
    ``|ic_tstat| >= gate_min_tstat`` (Newey-West) and ``|ic_tstat_dynamic| >= gate_min_tstat`` (the within-instrument
    demeaned factor must be significant too, so a static per-instrument ranking cannot pass on persistence alone).
    R2: ``coverage >= gate_min_coverage``.  R3: ``max_corr < gate_max_corr``.
    R4: composite ``valid`` ``ic_ir`` after >= before + ``gate_min_ir_gain`` (passes when there is no library before).
    Returns ``(passed, reasons)`` with one reason string per failed check, including the numbers.
    """
    hk = cfg.primary_key
    reasons: list[str] = []
    valid = (new_metrics.get("valid") or {}).get(hk) or {}
    train = (new_metrics.get("train") or {}).get(hk) or {}

    def _f(d: dict, k: str) -> float:
        v = d.get(k)
        try:
            return float("nan") if v is None else float(v)
        except (TypeError, ValueError):
            return float("nan")

    ic_v, ic_t, tstat, cov = _f(valid, "ic_mean"), _f(train, "ic_mean"), _f(valid, "ic_tstat"), _f(valid, "coverage")
    ic_std, tstat_dyn = _f(valid, "ic_std"), _f(valid, "ic_tstat_dynamic")
    if ic_std == ic_std and ic_std == 0.0:
        reasons.append(f"R0 valid.{hk} ic_std=0 (degenerate IC series, look-ahead suspected)")
    if abs(ic_v) > cfg.gate_max_abs_ic:
        reasons.append(
            f"R0 valid.{hk} |ic_mean|={_fmt(abs(ic_v))} > {cfg.gate_max_abs_ic} (implausible, look-ahead suspected)"
        )
    if not (abs(ic_v) >= cfg.gate_min_abs_ic):
        reasons.append(f"R1 valid.{hk} |ic_mean|={_fmt(abs(ic_v))} < {cfg.gate_min_abs_ic}")
    if not (ic_v == ic_v and ic_t == ic_t and np.sign(ic_v) == np.sign(ic_t) and ic_v != 0):
        reasons.append(f"R1 valid.{hk} ic_mean={_fmt(ic_v)} sign differs from train.{hk} ic_mean={_fmt(ic_t)}")
    if not (abs(tstat) >= cfg.gate_min_tstat):
        reasons.append(f"R1 valid.{hk} |ic_tstat|={_fmt(abs(tstat), 2)} < {cfg.gate_min_tstat}")
    if not (abs(tstat_dyn) >= cfg.gate_min_tstat):
        reasons.append(
            f"R1 valid.{hk} |ic_tstat_dynamic|={_fmt(abs(tstat_dyn), 2)} < {cfg.gate_min_tstat} "
            "(within-instrument demeaned factor not significant: static ranking suspected)"
        )
    if not (cov >= cfg.gate_min_coverage):
        reasons.append(f"R2 valid.{hk} coverage={_fmt(cov, 3)} < {cfg.gate_min_coverage}")
    mc = float(max_corr) if max_corr is not None else float("nan")
    if not (mc < cfg.gate_max_corr):
        reasons.append(f"R3 max_corr={_fmt(mc, 3)} >= {cfg.gate_max_corr}")
    if composite_before:
        before = _f(((composite_before.get("valid") or {}).get(hk) or {}), "ic_ir")
        after = _f(((composite_after or {}).get("valid") or {}).get(hk) or {}, "ic_ir")
        if before == before and not (after >= before + cfg.gate_min_ir_gain):
            reasons.append(
                f"R4 composite valid.{hk} ic_ir after={_fmt(after)} < before={_fmt(before)} + {cfg.gate_min_ir_gain}"
            )
    return len(reasons) == 0, reasons


def _train_sign(factor: pd.Series, rets: pd.DataFrame, train_mask: np.ndarray, cfg: EvalConfig) -> int:
    """Cheap sign estimate (train primary-horizon IC mean) used for library factors whose metrics are not needed."""
    ret = rets[f"ret_{cfg.primary_horizon_minutes}m"]
    ic = rankic_by_timestamp(factor[train_mask], ret[train_mask], cfg.min_assets)
    if ic.empty:
        return 1
    m = float(ic.mean())
    return -1 if m < 0 else 1


def summary_text(ev: dict, cfg: EvalConfig) -> str:
    """Render the 5-10 line train/valid summary (``ev["summary_text"]``) from an ExperimentEval dict."""
    hk = cfg.primary_key
    lines = [
        f"Panel: {ev['n_instruments']} instruments, {ev['panel_start']} .. {ev['panel_end']}; "
        f"library {len(ev['library_before'])} -> {len(ev['library_after'])} factors; "
        f"accepted this round: {ev['accepted'] or 'none'}",
    ]

    def _seg_line(label: str, metrics: dict | None) -> str:
        if not metrics:
            return f"{label}: (empty library)"
        parts = []
        for seg in ("train", "valid"):
            m = (metrics.get(seg) or {}).get(hk) or {}
            parts.append(
                f"{seg} {hk} ic_mean={_fmt(m.get('ic_mean'))} ic_ir={_fmt(m.get('ic_ir'), 3)} "
                f"tstat={_fmt(m.get('ic_tstat'), 2)} n_ts={m.get('n_ts', 0)}"
            )
        return f"{label}: " + " | ".join(parts)

    lines.append(_seg_line("composite before", ev["composite_before"]))
    lines.append(_seg_line("composite after", ev["composite_after"]))
    for name, info in ev["factors"].items():
        m = info["metrics"]
        tr = (m.get("train") or {}).get(hk) or {}
        va = (m.get("valid") or {}).get(hk) or {}
        status = "ACCEPTED" if info["accepted"] else "REJECTED: " + "; ".join(info["gate_reasons"])
        lines.append(
            f"{name}: sign={info['sign']:+d} train {hk} ic_mean={_fmt(tr.get('ic_mean'))} "
            f"tstat={_fmt(tr.get('ic_tstat'), 2)}"
            f" | valid {hk} ic_mean={_fmt(va.get('ic_mean'))} ic_ir={_fmt(va.get('ic_ir'), 3)} "
            f"tstat={_fmt(va.get('ic_tstat'), 2)} coverage={_fmt(va.get('coverage'), 2)} "
            f"ac1={_fmt(va.get('rank_autocorr_1'), 2)} spread_net={_fmt(va.get('q_spread_bp_net'), 1)}bp"
            f" | max_corr={_fmt(info['max_corr'], 2)}"
            + (f" (with {info['max_corr_with']})" if info["max_corr_with"] else "")
            + f" -> {status}"
        )
    return "\n".join(lines)


def evaluate_experiment(new_factors: pd.DataFrame, library: pd.DataFrame, close: pd.Series, cfg: EvalConfig) -> dict:
    """Evaluate candidate factors against the current library with greedy acceptance (DESIGN.md 4.3).

    ``new_factors`` and ``library`` are aligned to ``close.index`` (the panel).  Candidates are processed in column
    order; an accepted candidate joins the library (with its sign) before the next one is evaluated.  Every value in
    the returned dict is a plain python scalar/str/list/dict so it is JSON- and pickle-safe.
    """
    if not isinstance(close.index, pd.MultiIndex):
        raise ValueError("close must be indexed by a (datetime, instrument) MultiIndex")
    close = close.astype(float)
    if library is None:
        library = pd.DataFrame(index=close.index)
    if new_factors is None:
        new_factors = pd.DataFrame(index=close.index)
    if not library.index.equals(close.index):
        library = library.reindex(close.index)
    if not new_factors.index.equals(close.index):
        new_factors = new_factors.reindex(close.index)
    library = library.astype(float)
    new_factors = new_factors.astype(float)

    rets = forward_returns(close, cfg)
    dt_rows = pd.DatetimeIndex(_datetime_values(close.index))
    masks = segment_masks(dt_rows, cfg)
    tv_mask = masks["train"] | masks["valid"]

    lib_cols: list[str] = [str(c) for c in library.columns]
    lib = library.copy()
    lib.columns = lib_cols
    signs: dict[str, int] = {c: _train_sign(lib[c], rets, masks["train"], cfg) for c in lib_cols}
    # centred cross-sectional ranks of every library column are cached so composites are cheap
    rank_cache: dict[str, np.ndarray] = {c: cross_sectional_rank(lib[c]).to_numpy(dtype=float) for c in lib_cols}

    def _comp_metrics(cols: list[str]) -> dict | None:
        if not cols:
            return None
        ranks = np.column_stack([rank_cache[c] for c in cols])
        sgn = np.array([float(signs[c]) for c in cols])
        comp = pd.Series(_composite_from_ranks(ranks, sgn), index=close.index, name="composite")
        return factor_metrics(comp, rets, cfg)

    composite_before = _comp_metrics(lib_cols)
    current_before = composite_before
    factors_out: dict[str, dict[str, Any]] = {}
    accepted: list[str] = []
    for col in new_factors.columns:
        name = str(col)
        f = new_factors[col]
        metrics = factor_metrics(f, rets, cfg)
        sign = factor_sign(metrics, cfg)
        max_corr, max_corr_with = max_abs_corr_with_library(f, lib, tv_mask, cfg.min_assets)
        prev_sign, prev_rank = signs.get(name), rank_cache.get(name)
        signs[name] = sign
        rank_cache[name] = cross_sectional_rank(f).to_numpy(dtype=float)
        trial_cols = list(lib.columns) + ([] if name in lib.columns else [name])
        comp_after = _comp_metrics(trial_cols)
        cand_before = current_before
        passed, reasons = gate(metrics, max_corr, cand_before, comp_after, cfg)
        if passed:
            if name in lib.columns:
                lib[name] = f.to_numpy(dtype=float)
            else:
                lib = pd.concat([lib, f.rename(name)], axis=1)
            accepted.append(name)
            current_before = comp_after
        elif prev_sign is None:
            signs.pop(name, None)
            rank_cache.pop(name, None)
        else:
            signs[name], rank_cache[name] = prev_sign, prev_rank
        factors_out[name] = {
            "metrics": metrics,
            "sign": int(sign),
            "max_corr": float(max_corr),
            "max_corr_with": max_corr_with,
            "gate_passed": bool(passed),
            "gate_reasons": [str(r) for r in reasons],
            "accepted": bool(passed),
            "composite_before": cand_before,
            "composite_after": comp_after,
        }

    if lib.shape[1] == 0:
        composite_after = factor_metrics(pd.Series(np.nan, index=close.index, dtype=float), rets, cfg)
    else:
        composite_after = current_before if current_before is not None else _comp_metrics(list(lib.columns))

    n_inst = int(close.index.get_level_values(-1).nunique()) if len(close) else 0
    ev: dict[str, Any] = {
        "cfg": cfg.to_dict(),
        "n_instruments": n_inst,
        "panel_start": str(dt_rows.min()) if len(dt_rows) else "",
        "panel_end": str(dt_rows.max()) if len(dt_rows) else "",
        "library_before": list(lib_cols),
        "library_after": [str(c) for c in lib.columns],
        "composite_before": composite_before,
        "composite_after": composite_after,
        "factors": factors_out,
        "accepted": accepted,
        "summary_text": "",
    }
    ev["summary_text"] = summary_text(ev, cfg)
    return ev
