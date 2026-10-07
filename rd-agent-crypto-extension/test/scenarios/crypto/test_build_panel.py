"""Tests for rdagent.scenarios.crypto.data.build_panel (DESIGN.md section 3 / 11)."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.crypto.data.build_panel import aggregate_bars, main, read_1m_file

DAY1 = pd.Timestamp("2026-03-01 00:00:00")
DAY2 = pd.Timestamp("2026-03-02 00:00:00")
COLS = [
    "timestamps",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_amount",
]


def _minute_rows(start: pd.Timestamp, n_minutes: int, base_price: float, amount_scale: float) -> pd.DataFrame:
    """Deterministic 1-minute rows: price = base + k*0.01, volume = k+1, amount = amount_scale*(k+1)."""
    k = np.arange(n_minutes, dtype=float)
    ts = start + pd.to_timedelta(k, unit="min")
    price = base_price + 0.01 * k
    return pd.DataFrame(
        {
            "timestamps": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in ts],
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price + 0.1,
            "volume": k + 1,
            "amount": amount_scale * (k + 1),
            "trade_count": 10.0 + k,
            "taker_buy_volume": 0.5 * (k + 1),
            "taker_buy_amount": 0.25 * amount_scale * (k + 1),
        }
    )


def _write(dir_: Path, symbol: str, df: pd.DataFrame) -> None:
    with gzip.open(dir_ / f"{symbol}_1m.csv.gz", "wt") as f:
        df[COLS].to_csv(f, index=False)


@pytest.fixture()
def source_dir(tmp_path: Path) -> Path:
    src = tmp_path / "normalized"
    src.mkdir()
    two_days = 2 * 24 * 60
    # AAAUSDT: complete, largest turnover.
    _write(src, "AAAUSDT", _minute_rows(DAY1, two_days, base_price=100.0, amount_scale=1000.0))
    # BBBUSDT: complete except minutes 00:11 and 00:13 (inside bar 00:10) and the whole bar 00:20-00:24 of day 1.
    bbb = _minute_rows(DAY1, two_days, base_price=50.0, amount_scale=100.0)
    drop = {DAY1 + pd.Timedelta(minutes=m) for m in (11, 13, 20, 21, 22, 23, 24)}
    bbb = bbb[~pd.to_datetime(bbb["timestamps"]).dt.tz_localize(None).isin(drop)]
    _write(src, "BBBUSDT", bbb)
    # CCCUSDT: late listing, starts day 2 at 12:00, smallest turnover.
    _write(src, "CCCUSDT", _minute_rows(DAY2 + pd.Timedelta(hours=12), 12 * 60, base_price=1.0, amount_scale=1.0))
    return src


def _run(source_dir: Path, tmp_path: Path, **overrides: str) -> tuple[Path, Path]:
    full = tmp_path / "full"
    debug = tmp_path / "debug"
    argv = [
        "--source-dir",
        str(source_dir),
        "--out-dir",
        str(full),
        "--debug-out-dir",
        str(debug),
        "--bar-minutes",
        "5",
        "--debug-symbols",
        overrides.get("debug_symbols", "2"),
        "--debug-days",
        overrides.get("debug_days", "1"),
    ]
    assert main(argv) == 0
    return full, debug


def test_read_1m_file_is_tz_naive_float64(source_dir: Path) -> None:
    df = read_1m_file(source_dir / "AAAUSDT_1m.csv.gz")
    assert str(df["timestamps"].dtype) == "datetime64[ns]"
    assert df["timestamps"].dt.tz is None
    assert df["timestamps"].iloc[0] == DAY1
    for c in COLS[1:]:
        assert df[c].dtype == np.float64


def test_full_panel_structure_and_aggregation(source_dir: Path, tmp_path: Path) -> None:
    full, _ = _run(source_dir, tmp_path)
    assert (full / "crypto_5m.h5").exists()
    df = pd.read_hdf(full / "crypto_5m.h5", key="data")

    assert isinstance(df.index, pd.MultiIndex)
    assert list(df.index.names) == ["datetime", "instrument"]
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    assert str(df.index.get_level_values("datetime").dtype) == "datetime64[ns]"
    assert df.index.get_level_values("datetime").tz is None
    assert (df.index.get_level_values("datetime").minute % 5 == 0).all()
    assert list(df.columns) == [
        "$open",
        "$high",
        "$low",
        "$close",
        "$volume",
        "$amount",
        "$trade_count",
        "$taker_buy_volume",
        "$taker_buy_amount",
    ]
    assert all(dt == np.float64 for dt in df.dtypes)
    assert not df.isna().any().any()

    # AAAUSDT bar 00:05 of day 1: minutes k=5..9 -> price 100.05..100.09.
    row = df.loc[(DAY1 + pd.Timedelta(minutes=5), "AAAUSDT")]
    assert row["$open"] == pytest.approx(100.05)
    assert row["$high"] == pytest.approx(100.09 + 0.5)
    assert row["$low"] == pytest.approx(100.05 - 0.5)
    assert row["$close"] == pytest.approx(100.09 + 0.1)
    assert row["$volume"] == pytest.approx(6 + 7 + 8 + 9 + 10)
    assert row["$amount"] == pytest.approx(1000.0 * (6 + 7 + 8 + 9 + 10))
    assert row["$trade_count"] == pytest.approx(15 + 16 + 17 + 18 + 19)
    assert row["$taker_buy_volume"] == pytest.approx(0.5 * (6 + 7 + 8 + 9 + 10))
    assert row["$taker_buy_amount"] == pytest.approx(0.25 * 1000.0 * (6 + 7 + 8 + 9 + 10))

    # Row counts: AAA 576 bars, BBB 575 (one fully-missing bar), CCC 144 bars.
    counts = df.groupby(level="instrument").size()
    assert counts.to_dict() == {"AAAUSDT": 576, "BBBUSDT": 575, "CCCUSDT": 144}

    # BBBUSDT bar 00:10 with minutes 11 and 13 missing: k in {10, 12, 14}.
    row = df.loc[(DAY1 + pd.Timedelta(minutes=10), "BBBUSDT")]
    assert row["$open"] == pytest.approx(50.10)
    assert row["$close"] == pytest.approx(50.14 + 0.1)
    assert row["$high"] == pytest.approx(50.14 + 0.5)
    assert row["$low"] == pytest.approx(50.10 - 0.5)
    assert row["$volume"] == pytest.approx(11 + 13 + 15)
    assert row["$amount"] == pytest.approx(100.0 * (11 + 13 + 15))
    assert row["$trade_count"] == pytest.approx(20 + 22 + 24)

    # Fully missing bar is absent, not NaN.
    assert (DAY1 + pd.Timedelta(minutes=20), "BBBUSDT") not in df.index
    assert (DAY1 + pd.Timedelta(minutes=25), "BBBUSDT") in df.index

    # Late listing starts at its first bar.
    ccc_dts = df.xs("CCCUSDT", level="instrument").index
    assert ccc_dts.min() == DAY2 + pd.Timedelta(hours=12)
    assert ccc_dts.max() == DAY2 + pd.Timedelta(hours=23, minutes=55)
    assert df.index.get_level_values("datetime").min() == DAY1
    assert df.index.get_level_values("datetime").max() == DAY2 + pd.Timedelta(hours=23, minutes=55)


def test_debug_subset_top_n_by_amount_and_last_n_days(source_dir: Path, tmp_path: Path) -> None:
    full, debug = _run(source_dir, tmp_path, debug_symbols="2", debug_days="1")
    df_full = pd.read_hdf(full / "crypto_5m.h5", key="data")
    df_dbg = pd.read_hdf(debug / "crypto_5m.h5", key="data")

    assert list(df_dbg.index.names) == ["datetime", "instrument"]
    assert df_dbg.index.is_monotonic_increasing
    assert list(df_dbg.columns) == list(df_full.columns)
    # Top-2 by total $amount over the full panel: AAA (1000x) and BBB (100x), not CCC.
    assert sorted(df_dbg.index.get_level_values("instrument").unique()) == ["AAAUSDT", "BBBUSDT"]
    # Last 1 day ending at the panel's last bar: only day-2 bars.
    dts = df_dbg.index.get_level_values("datetime")
    assert dts.min() == DAY2
    assert dts.max() == DAY2 + pd.Timedelta(hours=23, minutes=55)
    assert len(df_dbg) == 288 * 2
    # Values are identical to the full panel on the subset.
    pd.testing.assert_frame_equal(df_dbg, df_full.loc[df_dbg.index])


def test_debug_symbols_larger_than_universe_keeps_all(source_dir: Path, tmp_path: Path) -> None:
    _, debug = _run(source_dir, tmp_path, debug_symbols="10", debug_days="5")
    df_dbg = pd.read_hdf(debug / "crypto_5m.h5", key="data")
    assert sorted(df_dbg.index.get_level_values("instrument").unique()) == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    assert len(df_dbg) == 576 + 575 + 144


def test_readme_written_in_both_folders(source_dir: Path, tmp_path: Path) -> None:
    full, debug = _run(source_dir, tmp_path, debug_symbols="2", debug_days="1")
    full_readme = (full / "README.md").read_text()
    dbg_readme = (debug / "README.md").read_text()
    assert "{{" not in full_readme and "{{" not in dbg_readme
    assert "3 instruments" in full_readme
    assert "2 instruments" in dbg_readme
    assert "crypto_5m.h5" in full_readme
    assert "2026-03-01 00:00:00" in full_readme and "2026-03-02 23:55:00" in full_readme
    assert "2026-03-02 00:00:00" in dbg_readme
    for banned in ("stock", "Qlib", "CSI300", "daily"):
        assert banned.lower() not in full_readme.lower(), banned
    assert "look-ahead" in full_readme
    # Only the two files RD-Agent's get_data_folder_intro can describe live in each folder.
    assert sorted(p.name for p in full.iterdir()) == ["README.md", "crypto_5m.h5"]
    assert sorted(p.name for p in debug.iterdir()) == ["README.md", "crypto_5m.h5"]


def test_aggregate_bars_unsorted_input_uses_time_order() -> None:
    df = _minute_rows(DAY1, 5, base_price=10.0, amount_scale=1.0)
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True).dt.tz_localize(None)
    shuffled = df.iloc[[3, 0, 4, 1, 2]]
    bars = aggregate_bars(shuffled, 5)
    assert len(bars) == 1
    assert bars.index[0] == DAY1
    assert bars["$open"].iloc[0] == pytest.approx(10.00)
    assert bars["$close"].iloc[0] == pytest.approx(10.04 + 0.1)
