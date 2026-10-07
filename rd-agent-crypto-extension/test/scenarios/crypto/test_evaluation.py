"""Tests for rdagent.scenarios.crypto.evaluation (DESIGN.md sections 4.2 and 11)."""

from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.crypto.evaluation import (
    EvalConfig,
    _ic_stats,
    composite,
    cross_sectional_rank,
    evaluate_experiment,
    factor_metrics,
    factor_sign,
    forward_returns,
    gate,
    max_abs_corr_with_library,
    rankic_by_timestamp,
    segment_masks,
)

N_INST = 40
N_DAYS = 30
BARS_PER_DAY = 288

# The synthetic "planted" factor below is ret_15m plus noise (|IC| ~ 0.8), far above the real-data implausibility cap
# of gate R0 (0.25); the cap is lifted here and exercised explicitly by the R0 tests with ``CFG_CAP``.
CFG = EvalConfig(
    train=("2026-03-01", "2026-03-18"),
    valid=("2026-03-19", "2026-03-24"),
    test=("2026-03-25", "2026-03-30 23:59:00"),
    gate_max_abs_ic=0.95,
)
CFG_CAP = EvalConfig(**{**CFG.__dict__, "gate_max_abs_ic": 0.25})


def _panel(seed: int = 0, n_inst: int = N_INST, n_days: int = N_DAYS) -> pd.Series:
    """Random-walk close for ``n_inst`` instruments over ``n_days`` of 5-minute bars, index (datetime, instrument)."""
    rng = np.random.default_rng(seed)
    dts = pd.date_range("2026-03-01", periods=n_days * BARS_PER_DAY, freq="5min")
    inst = [f"C{i:02d}USDT" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dts, inst], names=["datetime", "instrument"])
    lr = rng.normal(0.0, 0.002, size=(len(dts), n_inst))
    close = 100.0 * np.exp(np.cumsum(lr, axis=0))
    return pd.Series(close.ravel(), index=idx, name="$close")


@pytest.fixture(scope="module")
def close() -> pd.Series:
    return _panel()


@pytest.fixture(scope="module")
def rets(close: pd.Series) -> pd.DataFrame:
    return forward_returns(close, CFG)


def _planted(rets: pd.DataFrame, seed: int = 1, noise_scale: float = 0.8) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = rets["ret_15m"]
    noise = rng.normal(0.0, noise_scale * float(r.std()), size=len(r))
    return pd.Series(r.to_numpy() + noise, index=rets.index, name="planted")


def _noise(index: pd.MultiIndex, seed: int = 2) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=len(index)), index=index, name="noise")


# ---------------------------------------------------------------------------------------------------------------------
# 4.2 planted signal / sign / noise
# ---------------------------------------------------------------------------------------------------------------------


def test_planted_signal_positive_ic(rets: pd.DataFrame) -> None:
    m = factor_metrics(_planted(rets), rets, CFG)
    assert m["valid"]["15m"]["ic_mean"] > 0.3
    assert m["train"]["15m"]["ic_mean"] > 0.3
    assert m["valid"]["15m"]["ic_tstat"] > 3.0
    assert m["valid"]["15m"]["ic_pos_rate"] > 0.9
    assert m["valid"]["15m"]["coverage"] == pytest.approx(1.0)
    assert m["valid"]["15m"]["q_spread_bp_gross"] > 0
    assert m["valid"]["15m"]["q_spread_bp_net"] < m["valid"]["15m"]["q_spread_bp_gross"]
    assert factor_sign(m, CFG) == 1


def test_reversed_sign_negative_ic(rets: pd.DataFrame) -> None:
    m = factor_metrics(-_planted(rets), rets, CFG)
    assert m["valid"]["15m"]["ic_mean"] < -0.3
    assert factor_sign(m, CFG) == -1


def test_pure_noise_small_ic(rets: pd.DataFrame) -> None:
    m = factor_metrics(_noise(rets.index), rets, CFG)
    for seg in ("train", "valid"):
        assert m[seg]["15m"]["n_ts"] >= 500
        assert abs(m[seg]["15m"]["ic_mean"]) < 0.02
        assert abs(m[seg]["5m"]["ic_mean"]) < 0.02


# ---------------------------------------------------------------------------------------------------------------------
# 4.2 forward returns: no look-ahead, gaps -> NaN
# ---------------------------------------------------------------------------------------------------------------------


def test_forward_returns_hand_built_with_gaps() -> None:
    t0 = pd.Timestamp("2026-03-01 00:00:00")
    a_times = [t0 + pd.Timedelta(minutes=m) for m in (0, 5, 10, 20, 25)]  # 00:15 missing
    b_times = [t0 + pd.Timedelta(minutes=m) for m in (5, 10, 15, 20)]  # late listed, no gaps
    rows = [(t, "AAA") for t in a_times] + [(t, "BBB") for t in b_times]
    idx = pd.MultiIndex.from_tuples(rows, names=["datetime", "instrument"]).sort_values()
    values = {("AAA", 0): 100.0, ("AAA", 5): 101.0, ("AAA", 10): 102.0, ("AAA", 20): 104.0, ("AAA", 25): 105.0}
    values.update({("BBB", 5): 10.0, ("BBB", 10): 11.0, ("BBB", 15): 12.0, ("BBB", 20): 13.0})
    close = pd.Series([values[(i, int((t - t0).total_seconds() // 60))] for t, i in idx], index=idx)
    r = forward_returns(close, CFG)
    assert list(r.columns) == ["ret_5m", "ret_15m"]
    assert r.index.equals(close.index)

    def ret(inst: str, minute: int, col: str) -> float:
        return r.loc[(t0 + pd.Timedelta(minutes=minute), inst), col]

    assert ret("AAA", 0, "ret_5m") == pytest.approx(101.0 / 100.0 - 1)
    assert ret("AAA", 5, "ret_5m") == pytest.approx(102.0 / 101.0 - 1)
    assert np.isnan(ret("AAA", 10, "ret_5m"))  # 00:15 missing -> NaN, not the 00:20 bar
    assert ret("AAA", 20, "ret_5m") == pytest.approx(105.0 / 104.0 - 1)
    assert np.isnan(ret("AAA", 25, "ret_5m"))  # end of series
    assert np.isnan(ret("AAA", 0, "ret_15m"))  # 00:15 missing
    assert ret("AAA", 5, "ret_15m") == pytest.approx(104.0 / 101.0 - 1)
    assert ret("AAA", 10, "ret_15m") == pytest.approx(105.0 / 102.0 - 1)
    assert ret("BBB", 5, "ret_15m") == pytest.approx(13.0 / 10.0 - 1)
    assert np.isnan(ret("BBB", 10, "ret_15m"))
    assert ret("BBB", 15, "ret_5m") == pytest.approx(13.0 / 12.0 - 1)


def test_forward_returns_random_walk_matches_direct_computation(close: pd.Series, rets: pd.DataFrame) -> None:
    wide = close.unstack("instrument")
    direct5 = (wide.shift(-1) / wide - 1).stack(future_stack=True).reindex(close.index)
    direct15 = (wide.shift(-3) / wide - 1).stack(future_stack=True).reindex(close.index)
    pd.testing.assert_series_equal(rets["ret_5m"], direct5, check_names=False)
    pd.testing.assert_series_equal(rets["ret_15m"], direct15, check_names=False)


def test_shifted_factor_does_not_leak(rets: pd.DataFrame) -> None:
    # using ret_5m at t+1 as the factor at t must not correlate with ret_5m at t (targets use only t and t+h)
    lead = rets["ret_5m"].groupby(level="instrument").shift(-1)
    ic = rankic_by_timestamp(lead, rets["ret_5m"], CFG.min_assets)
    assert abs(float(ic.mean())) < 0.05


# ---------------------------------------------------------------------------------------------------------------------
# 4.2 segments and purge
# ---------------------------------------------------------------------------------------------------------------------


def test_segment_masks_purge_and_disjoint() -> None:
    dts = pd.date_range("2026-03-01", "2026-03-05 23:55:00", freq="5min")
    cfg = EvalConfig(
        train=("2026-03-01", "2026-03-02"), valid=("2026-03-03", "2026-03-04"), test=("2026-03-05", "2026-03-05")
    )
    masks = segment_masks(dts, cfg)
    assert set(masks) == {"train", "valid", "test"}
    total = masks["train"].astype(int) + masks["valid"].astype(int) + masks["test"].astype(int)
    assert total.max() == 1  # no origin in two segments
    train_dts = dts[masks["train"]]
    assert train_dts.min() == pd.Timestamp("2026-03-01 00:00:00")
    assert train_dts.max() == pd.Timestamp("2026-03-02 23:25:00")  # 23:30 + 15m + 15m purge > end of day
    assert pd.Timestamp("2026-03-02 23:30:00") not in train_dts
    valid_dts = dts[masks["valid"]]
    assert valid_dts.min() == pd.Timestamp("2026-03-03 00:00:00")
    assert valid_dts.max() == pd.Timestamp("2026-03-04 23:25:00")
    assert dts[masks["test"]].max() == pd.Timestamp("2026-03-05 23:25:00")
    # explicit datetime end bound is used as-is
    cfg2 = EvalConfig(
        train=("2026-03-01", "2026-03-02 12:00:00"),
        valid=("2026-03-03", "2026-03-04"),
        test=("2026-03-05", "2026-03-05"),
    )
    m2 = segment_masks(dts, cfg2)
    assert dts[m2["train"]].max() == pd.Timestamp("2026-03-02 11:30:00")
    # tz-aware inputs are converted to naive UTC
    m3 = segment_masks(dts.tz_localize("UTC"), cfg)
    assert np.array_equal(m3["train"], masks["train"])


def test_segment_masks_on_multiindex_rows(close: pd.Series) -> None:
    dt_rows = close.index.get_level_values("datetime")
    masks = segment_masks(dt_rows, CFG)
    assert masks["train"].shape == (len(close),)
    assert masks["train"].sum() == (18 * BARS_PER_DAY - 6) * N_INST
    assert masks["valid"].sum() == (6 * BARS_PER_DAY - 6) * N_INST
    assert masks["test"].sum() == (6 * BARS_PER_DAY - 6) * N_INST


# ---------------------------------------------------------------------------------------------------------------------
# 4.2 min_assets and rankic
# ---------------------------------------------------------------------------------------------------------------------


def test_rankic_matches_scipy_style_spearman() -> None:
    rng = np.random.default_rng(3)
    dts = pd.date_range("2026-03-01", periods=3, freq="5min")
    inst = [f"I{i}" for i in range(12)]
    idx = pd.MultiIndex.from_product([dts, inst], names=["datetime", "instrument"])
    x = pd.Series(rng.normal(size=len(idx)), index=idx)
    y = pd.Series(rng.normal(size=len(idx)), index=idx)
    x.iloc[3] = np.nan  # one missing pair on the first datetime
    x.iloc[5] = x.iloc[6]  # a tie
    ic = rankic_by_timestamp(x, y, min_assets=5)
    assert list(ic.index) == list(dts)
    for t in dts:
        xs, ys = x.loc[t], y.loc[t]
        ok = xs.notna() & ys.notna()
        expected = np.corrcoef(xs[ok].rank(method="average"), ys[ok].rank(method="average"))[0, 1]
        assert ic.loc[t] == pytest.approx(expected, abs=1e-12)


def test_min_assets_skips_sparse_timestamps(close: pd.Series, rets: pd.DataFrame) -> None:
    f = _planted(rets)
    dt_rows = close.index.get_level_values("datetime")
    # on every 3rd datetime of the panel only 5 instruments carry a value
    all_dts = dt_rows.unique()
    sparse = all_dts[::3]
    inst_rows = close.index.get_level_values("instrument")
    keep_inst = inst_rows.isin([f"C{i:02d}USDT" for i in range(5)])
    f = f.where(~dt_rows.isin(sparse) | keep_inst)
    ic = rankic_by_timestamp(f, rets["ret_15m"], min_assets=10)
    assert not ic.index.isin(sparse).any()
    m = factor_metrics(f, rets, EvalConfig(**{**CFG.__dict__, "min_assets": 10}))
    masks = segment_masks(all_dts, CFG)
    expected_valid = int((masks["valid"] & ~all_dts.isin(sparse)).sum())
    assert m["valid"]["15m"]["n_ts"] == expected_valid
    assert m["valid"]["15m"]["coverage"] < 1.0
    # datetimes with fewer than min_assets are skipped (n_ts counts only the evaluated ones)
    assert m["valid"]["15m"]["n_ts"] < (6 * BARS_PER_DAY - 6)


def test_rankic_empty_and_constant() -> None:
    empty = pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"]))
    assert rankic_by_timestamp(empty, empty, 2).empty
    dts = pd.date_range("2026-03-01", periods=2, freq="5min")
    idx = pd.MultiIndex.from_product([dts, list("abcde")], names=["datetime", "instrument"])
    const = pd.Series(1.0, index=idx)
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    assert rankic_by_timestamp(const, y, 2).empty  # undefined correlation is skipped
    m = factor_metrics(const, pd.DataFrame({"ret_5m": y, "ret_15m": y}), CFG)
    assert m["train"]["15m"]["n_ts"] == 0 and np.isnan(m["train"]["15m"]["ic_mean"])


# ---------------------------------------------------------------------------------------------------------------------
# 4.2 rank / composite
# ---------------------------------------------------------------------------------------------------------------------


def test_cross_sectional_rank_properties(rets: pd.DataFrame) -> None:
    f = _noise(rets.index)
    f.iloc[7] = np.nan
    r = cross_sectional_rank(f)
    assert r.index.equals(f.index)
    assert np.isnan(r.iloc[7])
    ok = r.dropna()
    assert ok.max() == pytest.approx(0.5)
    assert ok.min() > -0.5
    dts = f.index.get_level_values("datetime").unique()
    first = r.loc[dts[0]]  # one NaN -> 39 ranked instruments
    assert first.max() == pytest.approx(0.5)
    assert first.min() == pytest.approx(1.0 / (N_INST - 1) - 0.5)
    second = r.loc[dts[1]]
    assert second.min() == pytest.approx(1.0 / N_INST - 0.5)
    assert sorted(second.to_numpy()) == pytest.approx(np.arange(1, N_INST + 1) / N_INST - 0.5)


def test_composite_single_and_duplicate(rets: pd.DataFrame) -> None:
    f = _planted(rets)
    lib1 = pd.DataFrame({"f": f})
    c_plus = composite(lib1, {"f": 1})
    pd.testing.assert_series_equal(c_plus, cross_sectional_rank(f), check_names=False)
    c_minus = composite(lib1, {"f": -1})
    pd.testing.assert_series_equal(c_minus, -cross_sectional_rank(f), check_names=False)
    lib2 = pd.DataFrame({"f": f, "g": f})
    c2 = composite(lib2, {"f": 1, "g": 1})
    pd.testing.assert_series_equal(c2, c_plus, check_names=False)
    empty = composite(pd.DataFrame(index=rets.index), {})
    assert empty.empty


# ---------------------------------------------------------------------------------------------------------------------
# 4.2 correlation with library, gate
# ---------------------------------------------------------------------------------------------------------------------


def test_max_abs_corr_with_library(close: pd.Series, rets: pd.DataFrame) -> None:
    f = _planted(rets)
    dt_rows = close.index.get_level_values("datetime")
    masks = segment_masks(dt_rows, CFG)
    tv = masks["train"] | masks["valid"]
    assert max_abs_corr_with_library(f, pd.DataFrame(index=close.index), tv) == (0.0, None)
    lib = pd.DataFrame({"same": f, "neg": -f, "noise": _noise(close.index)})
    corr, name = max_abs_corr_with_library(f, lib, tv, CFG.min_assets)
    assert corr == pytest.approx(1.0)
    assert name in ("same", "neg")
    corr2, name2 = max_abs_corr_with_library(f, lib[["noise"]], tv, CFG.min_assets)
    assert corr2 < 0.05 and name2 == "noise"


def test_metrics_report_naive_and_dynamic_tstats(rets: pd.DataFrame) -> None:
    m = factor_metrics(_planted(rets), rets, CFG)
    for seg in ("train", "valid", "test"):
        for h in ("5m", "15m"):
            d = m[seg][h]
            assert {"ic_tstat", "ic_tstat_naive", "ic_tstat_dynamic"} <= set(d)
            assert d["ic_tstat_naive"] == pytest.approx(d["ic_ir"] * np.sqrt(d["n_ts"]))
            assert np.sign(d["ic_tstat"]) == np.sign(d["ic_tstat_naive"])
        assert np.isnan(m[seg]["5m"]["ic_tstat_dynamic"])  # defined at the primary horizon only
        assert m[seg]["15m"]["ic_tstat_dynamic"] > 3.0  # the planted signal is purely dynamic
    assert CFG.nw_lags == 6


def test_newey_west_tstat_deflates_autocorrelated_ic() -> None:
    rng = np.random.default_rng(5)
    n = 5000
    iid = rng.normal(0.01, 0.1, size=n)
    d = _ic_stats(iid, nw_lags=6)
    assert d["ic_tstat_naive"] == pytest.approx(d["ic_ir"] * np.sqrt(n))
    assert 0.7 < d["ic_tstat"] / d["ic_tstat_naive"] < 1.3  # no autocorrelation: NW ~ naive
    ar = np.empty(n)
    ar[0] = 0.0
    eps = rng.normal(size=n)
    for i in range(1, n):
        ar[i] = 0.6 * ar[i - 1] + eps[i]
    ar = ar * 0.1 + 0.01
    d = _ic_stats(ar, nw_lags=6)
    assert abs(d["ic_tstat"]) < 0.6 * abs(d["ic_tstat_naive"])  # positive autocorrelation: NW t-stat is smaller
    assert _ic_stats(np.full(50, 0.3), nw_lags=6)["ic_std"] == 0.0
    assert np.isnan(_ic_stats(np.full(50, 0.3), nw_lags=6)["ic_tstat"])


def test_static_ranking_fails_gate_r1_dynamic() -> None:
    # instruments with a persistent drift: a static ranking (per-instrument constant) has a persistently positive IC
    # and a huge naive t-stat, but carries no time-varying information -> R1 (ic_tstat_dynamic) must reject it
    rng = np.random.default_rng(3)
    n_inst, n_days = 40, 30
    dts = pd.date_range("2026-03-01", periods=n_days * BARS_PER_DAY, freq="5min")
    inst = [f"C{i:02d}USDT" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dts, inst], names=["datetime", "instrument"])
    mu = rng.normal(0.0, 0.0004, size=n_inst)
    lr = mu.reshape(1, -1) + rng.normal(0.0, 0.002, size=(len(dts), n_inst))
    close = pd.Series((100.0 * np.exp(np.cumsum(lr, axis=0))).ravel(), index=idx, name="$close")
    rets = forward_returns(close, CFG)
    static = pd.Series(np.tile(mu, len(dts)), index=idx, name="static")
    m = factor_metrics(static, rets, CFG)
    v = m["valid"]["15m"]
    assert v["ic_mean"] > 0.05 and v["ic_tstat_naive"] > 3.0
    assert v["rank_autocorr_1"] == pytest.approx(1.0)
    assert np.isnan(v["ic_tstat_dynamic"])  # the demeaned factor is constant -> no dynamic information
    passed, reasons = gate(m, 0.0, None, m, CFG)
    assert not passed and any("ic_tstat_dynamic" in r for r in reasons)
    # a static ranking plus a little iid noise: still rejected by the dynamic control
    noisy = static + pd.Series(rng.normal(0.0, 1e-5, size=len(idx)), index=idx)
    m2 = factor_metrics(noisy, rets, CFG)
    assert abs(m2["valid"]["15m"]["ic_tstat_dynamic"]) < 3.0
    assert not gate(m2, 0.0, None, m2, CFG)[0]


def test_gate_r0_rejects_exact_target_copy(rets: pd.DataFrame) -> None:
    leak = rets["ret_15m"].rename("leak")
    m = factor_metrics(leak, rets, CFG_CAP)
    assert m["valid"]["15m"]["ic_mean"] == pytest.approx(1.0) and m["valid"]["15m"]["ic_std"] == 0.0
    passed, reasons = gate(m, 0.0, None, m, CFG_CAP)
    assert not passed
    assert reasons[0].startswith("R0") and "ic_std=0" in reasons[0]
    assert any("look-ahead" in r for r in reasons)


def test_gate_r0_rejects_implausible_ic(rets: pd.DataFrame) -> None:
    strong = _planted(rets, seed=9, noise_scale=2.0)  # |IC| ~ 0.45: far too strong for a 5-minute cross-section
    m = factor_metrics(strong, rets, CFG_CAP)
    assert 0.3 < m["valid"]["15m"]["ic_mean"] < 0.6
    passed, reasons = gate(m, 0.0, None, m, CFG_CAP)
    assert not passed and any(r.startswith("R0") and "implausible" in r for r in reasons)
    ev = evaluate_experiment(pd.DataFrame({"strong": strong}), pd.DataFrame(index=rets.index), _panel(), CFG_CAP)
    assert ev["accepted"] == [] and any(r.startswith("R0") for r in ev["factors"]["strong"]["gate_reasons"])
    # a realistic-strength factor passes the cap
    weak = _planted(rets, seed=10, noise_scale=6.0)  # |IC| ~ 0.16
    mw = factor_metrics(weak, rets, CFG_CAP)
    assert gate(mw, 0.0, None, mw, CFG_CAP)[0], gate(mw, 0.0, None, mw, CFG_CAP)[1]


def test_inf_factor_values_are_missing(rets: pd.DataFrame) -> None:
    f = _planted(rets)
    vals = f.to_numpy(dtype=float).copy()
    vals[::7] = np.inf
    with_inf = pd.Series(vals, index=f.index, name="with_inf")
    vals_nan = f.to_numpy(dtype=float).copy()
    vals_nan[::7] = np.nan
    with_nan = pd.Series(vals_nan, index=f.index, name="with_nan")
    m_inf = factor_metrics(with_inf, rets, CFG)
    m_nan = factor_metrics(with_nan, rets, CFG)
    assert m_inf["valid"]["15m"]["coverage"] == pytest.approx(m_nan["valid"]["15m"]["coverage"])
    assert m_inf["valid"]["15m"]["coverage"] < 0.9
    assert m_inf["valid"]["15m"]["ic_mean"] == pytest.approx(m_nan["valid"]["15m"]["ic_mean"])
    assert cross_sectional_rank(with_inf).isna().sum() == cross_sectional_rank(with_nan).isna().sum()


def test_gate_rules(rets: pd.DataFrame) -> None:
    good = factor_metrics(_planted(rets), rets, CFG)
    noise = factor_metrics(_noise(rets.index), rets, CFG)
    passed, reasons = gate(good, 0.1, None, good, CFG)
    assert passed and reasons == []
    # R3 rejects a factor equal to a library factor
    passed, reasons = gate(good, 1.0, good, good, CFG)
    assert not passed and any(r.startswith("R3") for r in reasons)
    # R1 rejects noise
    passed, reasons = gate(noise, 0.0, None, noise, CFG)
    assert not passed and any(r.startswith("R1") for r in reasons)
    assert all("valid.15m" in r for r in reasons if r.startswith("R1"))
    # R2 coverage
    low_cov = json.loads(json.dumps(good))
    low_cov["valid"]["15m"]["coverage"] = 0.2
    passed, reasons = gate(low_cov, 0.0, None, low_cov, CFG)
    assert not passed and reasons and reasons[0].startswith("R2")
    # R1 sign inconsistency between train and valid
    flipped = json.loads(json.dumps(good))
    flipped["train"]["15m"]["ic_mean"] = -flipped["train"]["15m"]["ic_mean"]
    passed, reasons = gate(flipped, 0.0, None, flipped, CFG)
    assert not passed and any("sign" in r for r in reasons)
    # R4 composite ir must not decrease
    worse = json.loads(json.dumps(good))
    worse["valid"]["15m"]["ic_ir"] = good["valid"]["15m"]["ic_ir"] - 1.0
    passed, reasons = gate(good, 0.0, good, worse, CFG)
    assert not passed and reasons[0].startswith("R4")
    # R4 passes when there is no library before
    passed, _ = gate(good, 0.0, None, worse, CFG)
    assert passed


# ---------------------------------------------------------------------------------------------------------------------
# evaluate_experiment: greedy acceptance, dict shape, pickle/json safety, performance
# ---------------------------------------------------------------------------------------------------------------------


def _assert_plain(obj: object, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, str), f"non-str key at {path}: {k!r}"
            _assert_plain(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_plain(v, f"{path}[{i}]")
    else:
        assert obj is None or type(obj) in (str, float, int, bool), f"non-plain value at {path}: {type(obj)}"


def test_evaluate_experiment_greedy_acceptance(close: pd.Series, rets: pd.DataFrame) -> None:
    planted = _planted(rets)
    new = pd.DataFrame({"planted": planted, "planted_copy": planted * 2.0 + 1.0, "noise": _noise(close.index)})
    library = pd.DataFrame(index=close.index)
    t0 = time.perf_counter()
    ev = evaluate_experiment(new, library, close, CFG)
    elapsed = time.perf_counter() - t0
    assert elapsed < 20.0, f"evaluate_experiment took {elapsed:.1f}s"

    assert ev["accepted"] == ["planted"]
    assert ev["library_before"] == [] and ev["library_after"] == ["planted"]
    assert ev["composite_before"] is None
    assert ev["composite_after"]["valid"]["15m"]["ic_mean"] > 0.3
    assert ev["n_instruments"] == N_INST
    assert ev["panel_start"].startswith("2026-03-01") and ev["panel_end"].startswith("2026-03-30")
    assert ev["cfg"]["horizons_minutes"] == [5, 15] and ev["cfg"]["primary_horizon_minutes"] == 15

    f = ev["factors"]
    assert list(f) == ["planted", "planted_copy", "noise"]
    assert f["planted"]["accepted"] and f["planted"]["gate_passed"] and f["planted"]["gate_reasons"] == []
    assert f["planted"]["sign"] == 1 and f["planted"]["max_corr"] == 0.0 and f["planted"]["max_corr_with"] is None
    assert f["planted"]["composite_before"] is None
    # greedy: the copy is compared against the library that now contains "planted"
    assert not f["planted_copy"]["accepted"]
    assert f["planted_copy"]["max_corr"] == pytest.approx(1.0) and f["planted_copy"]["max_corr_with"] == "planted"
    assert any(r.startswith("R3") for r in f["planted_copy"]["gate_reasons"])
    assert f["planted_copy"]["composite_before"] == f["planted"]["composite_after"]
    assert not f["noise"]["accepted"] and any(r.startswith("R1") for r in f["noise"]["gate_reasons"])
    for name in f:
        for seg in ("train", "valid", "test"):
            for h in ("5m", "15m"):
                m = f[name]["metrics"][seg][h]
                assert set(m) == {
                    "n_ts", "ic_mean", "ic_std", "ic_ir", "ic_tstat", "ic_tstat_naive", "ic_tstat_dynamic",
                    "ic_pos_rate", "coverage", "rank_autocorr_1", "q_spread_bp_gross", "q_spread_bp_net",
                }  # fmt: skip
                assert m["n_ts"] > 500

    summary = ev["summary_text"]
    assert 5 <= len(summary.splitlines()) <= 10
    assert "test" not in summary.lower().replace("latest", "")
    assert "planted" in summary and "REJECTED" in summary and "ACCEPTED" in summary

    _assert_plain(ev)
    pickle.loads(pickle.dumps(ev))
    json.dumps(ev)


def test_evaluate_experiment_with_existing_library(close: pd.Series, rets: pd.DataFrame) -> None:
    planted = _planted(rets)
    other = _planted(rets, seed=11, noise_scale=3.0)  # weaker, distinct signal
    library = pd.DataFrame({"lib_a": -planted, "lib_b": _noise(close.index, seed=5)})  # negative sign library factor
    new = pd.DataFrame({"other": other, "dup_of_lib": planted})
    ev = evaluate_experiment(new, library, close, CFG)
    assert ev["library_before"] == ["lib_a", "lib_b"]
    assert ev["composite_before"] is not None
    assert ev["composite_before"]["valid"]["15m"]["ic_mean"] > 0  # library signs applied
    assert not ev["factors"]["dup_of_lib"]["accepted"]
    assert ev["factors"]["dup_of_lib"]["max_corr_with"] == "lib_a"
    assert ev["factors"]["other"]["composite_before"] == ev["composite_before"]
    assert ev["library_after"][:2] == ["lib_a", "lib_b"]
    assert set(ev["accepted"]) <= {"other"}
    _assert_plain(ev)


def test_evaluate_experiment_no_candidates_empty_library(close: pd.Series) -> None:
    ev = evaluate_experiment(pd.DataFrame(index=close.index), pd.DataFrame(index=close.index), close, CFG)
    assert ev["accepted"] == [] and ev["factors"] == {} and ev["composite_before"] is None
    assert ev["composite_after"]["valid"]["15m"]["n_ts"] == 0
    assert np.isnan(ev["composite_after"]["valid"]["15m"]["ic_mean"])
    assert ev["summary_text"]


def test_factor_aligned_to_panel_subset(close: pd.Series, rets: pd.DataFrame) -> None:
    # a factor covering only half the instruments: coverage ~0.5 and metrics are still computed on the panel grid
    f = _planted(rets)
    inst_rows = close.index.get_level_values("instrument")
    f = f[inst_rows.isin([f"C{i:02d}USDT" for i in range(N_INST // 2)])]
    cfg = EvalConfig(**{**CFG.__dict__, "min_assets": 10})
    m = factor_metrics(f, rets, cfg)
    assert m["valid"]["15m"]["coverage"] == pytest.approx(0.5)
    assert m["valid"]["15m"]["ic_mean"] > 0.3


def test_crypto_eval_settings_to_eval_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from rdagent.scenarios.crypto.conf import CryptoEvalSettings

    monkeypatch.setenv("CRYPTO_EVAL_HORIZONS_MINUTES", "5, 15,30")
    monkeypatch.setenv("CRYPTO_EVAL_PRIMARY_HORIZON_MINUTES", "30")
    monkeypatch.setenv("CRYPTO_EVAL_GATE_MIN_TSTAT", "2.5")
    monkeypatch.setenv("CRYPTO_EVAL_VALID_END", "2026-07-15 12:00:00")
    s = CryptoEvalSettings()
    assert s.panel_filename == "crypto_5m.h5"
    cfg = s.to_eval_config()
    assert isinstance(cfg, EvalConfig)
    assert cfg.horizons_minutes == (5, 15, 30)
    assert cfg.primary_horizon_minutes == 30 and cfg.primary_key == "30m"
    assert cfg.gate_min_tstat == 2.5
    assert cfg.gate_min_abs_ic == 0.01 and cfg.gate_max_abs_ic == 0.25
    monkeypatch.setenv("CRYPTO_EVAL_GATE_MAX_ABS_IC", "0.4")
    assert CryptoEvalSettings().to_eval_config().gate_max_abs_ic == 0.4
    assert cfg.train == ("2026-02-25", "2026-06-30")
    assert cfg.valid == ("2026-07-01", "2026-07-15 12:00:00")
    assert cfg.test == ("2026-08-01", "2026-08-23 23:59:00")
    assert cfg.to_dict()["horizons_minutes"] == [5, 15, 30]
    monkeypatch.setenv("CRYPTO_EVAL_PRIMARY_HORIZON_MINUTES", "10")
    with pytest.raises(ValueError):
        CryptoEvalSettings().to_eval_config()
