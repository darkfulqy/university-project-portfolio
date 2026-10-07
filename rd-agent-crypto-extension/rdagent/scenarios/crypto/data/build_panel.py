"""Build the crypto 5-minute panel consumed by RD-Agent from Binance 1-minute csv.gz files.

Contract: docs/crypto_5m/DESIGN.md section 3. Usage::

    python -m rdagent.scenarios.crypto.data.build_panel \
        --source-dir <dir with SYMBOL_1m.csv.gz> --out-dir <full folder> --debug-out-dir <debug folder> \
        [--bar-minutes 5 --debug-symbols 10 --debug-days 14 --panel-filename crypto_5m.h5]

The builder processes one source file at a time (read, aggregate, append) so peak memory stays far below the
size of the raw 1-minute data. Output: ``<out-dir>/<panel-filename>`` (pandas HDF5, key ``"data"``, fixed
format) plus ``README.md`` rendered from ``README_template.md``; the same pair in ``<debug-out-dir>`` restricted
to the top ``--debug-symbols`` instruments by total ``$amount`` and the last ``--debug-days`` days.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_SUFFIX = "_1m.csv.gz"
SOURCE_NUMERIC_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_amount",
)
PANEL_COLUMNS: tuple[str, ...] = tuple(f"${c}" for c in SOURCE_NUMERIC_COLUMNS)
INDEX_NAMES: tuple[str, str] = ("datetime", "instrument")
HDF_KEY = "data"
_AGG: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "amount": "sum",
    "trade_count": "sum",
    "taker_buy_volume": "sum",
    "taker_buy_amount": "sum",
}
README_TEMPLATE_PATH = Path(__file__).with_name("README_template.md")


def instrument_from_path(path: Path) -> str:
    """Return the instrument symbol encoded in a source file name (``BTCUSDT_1m.csv.gz`` -> ``BTCUSDT``)."""
    name = path.name
    if name.endswith(SOURCE_SUFFIX):
        return name[: -len(SOURCE_SUFFIX)]
    return name.split(".", 1)[0].removesuffix("_1m")


def read_1m_file(path: Path) -> pd.DataFrame:
    """Read one 1-minute csv.gz file: tz-naive UTC ``timestamps`` column plus float64 numeric columns."""
    df = pd.read_csv(
        path,
        usecols=["timestamps", *SOURCE_NUMERIC_COLUMNS],
        dtype={c: np.float64 for c in SOURCE_NUMERIC_COLUMNS},
    )
    ts = pd.to_datetime(df["timestamps"], utc=True, format="ISO8601")
    df["timestamps"] = ts.dt.tz_localize(None).astype("datetime64[ns]")
    return df


def aggregate_bars(df_1m: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """Aggregate 1-minute rows to ``bar_minutes`` bars keyed by bar OPEN time.

    open=first, high=max, low=min, close=last, all volume-like columns summed. Only bins that contain at least
    one source row are emitted (no NaN padding). Returns a frame indexed by ``datetime`` with ``$``-prefixed
    float64 columns.
    """
    df = df_1m.sort_values("timestamps", kind="stable")
    bar_open = df["timestamps"].dt.floor(f"{bar_minutes}min")
    out = df[list(SOURCE_NUMERIC_COLUMNS)].groupby(bar_open, sort=True).agg(_AGG)
    out.index.name = "datetime"
    out.columns = [f"${c}" for c in out.columns]
    return out[list(PANEL_COLUMNS)].astype(np.float64)


def build_panel(source_dir: Path, bar_minutes: int, log=print) -> pd.DataFrame:
    """Read every ``*_1m.csv.gz`` in ``source_dir`` (one at a time) and return the sorted MultiIndex panel."""
    files = sorted(Path(source_dir).glob(f"*{SOURCE_SUFFIX}"))
    if not files:
        raise FileNotFoundError(f"no *{SOURCE_SUFFIX} files found in {source_dir}")
    parts: list[pd.DataFrame] = []
    for i, path in enumerate(files, 1):
        instrument = instrument_from_path(path)
        t0 = time.perf_counter()
        raw = read_1m_file(path)
        bars = aggregate_bars(raw, bar_minutes)
        bars["instrument"] = instrument
        bars = bars.set_index("instrument", append=True)
        parts.append(bars)
        log(
            f"[{i}/{len(files)}] {instrument}: {len(raw)} 1m rows -> {len(bars)} bars ({time.perf_counter() - t0:.1f}s)"
        )
        del raw
    panel = pd.concat(parts)
    del parts
    panel.index = panel.index.set_names(list(INDEX_NAMES))
    panel = panel.sort_index()
    return panel


def select_debug_panel(panel: pd.DataFrame, debug_symbols: int, debug_days: int) -> pd.DataFrame:
    """Top ``debug_symbols`` instruments by total ``$amount`` over the full panel, last ``debug_days`` days.

    The time window is ``(last_datetime - debug_days, last_datetime]``, i.e. exactly ``debug_days`` * 24h of bars
    ending at the panel's last bar.
    """
    totals = panel["$amount"].groupby(level="instrument").sum().sort_values(ascending=False)
    keep = list(totals.index[:debug_symbols])
    last_dt = panel.index.get_level_values("datetime").max()
    start = last_dt - pd.Timedelta(days=debug_days)
    dts = panel.index.get_level_values("datetime")
    mask = (dts > start) & panel.index.get_level_values("instrument").isin(keep)
    return panel[mask].sort_index()


def panel_summary(panel: pd.DataFrame) -> dict[str, object]:
    """Row count, instrument count and datetime span of a panel."""
    dts = panel.index.get_level_values("datetime")
    return {
        "n_rows": int(len(panel)),
        "n_instruments": int(panel.index.get_level_values("instrument").nunique()),
        "first_datetime": dts.min(),
        "last_datetime": dts.max(),
    }


def render_readme(panel: pd.DataFrame, bar_minutes: int, panel_filename: str, template_path: Path | None = None) -> str:
    """Render README_template.md with the panel's instrument count, row count, span and bar size."""
    s = panel_summary(panel)
    text = (template_path or README_TEMPLATE_PATH).read_text(encoding="utf-8")
    subs = {
        "{{PANEL_FILENAME}}": panel_filename,
        "{{BAR_MINUTES}}": str(bar_minutes),
        "{{N_INSTRUMENTS}}": str(s["n_instruments"]),
        "{{N_ROWS}}": str(s["n_rows"]),
        "{{FIRST_DATETIME}}": str(s["first_datetime"]),
        "{{LAST_DATETIME}}": str(s["last_datetime"]),
    }
    for k, v in subs.items():
        text = text.replace(k, v)
    return text


def write_folder(
    panel: pd.DataFrame, out_dir: Path, bar_minutes: int, panel_filename: str, label: str, log=print
) -> Path:
    """Write ``<out_dir>/<panel_filename>`` (key "data", fixed format) and ``README.md``; print a summary."""
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / panel_filename
    panel.to_hdf(h5_path, key=HDF_KEY, mode="w", format="fixed")
    (out_dir / "README.md").write_text(render_readme(panel, bar_minutes, panel_filename), encoding="utf-8")
    s = panel_summary(panel)
    log(
        f"[{label}] {h5_path}: rows={s['n_rows']} instruments={s['n_instruments']} "
        f"first={s['first_datetime']} last={s['last_datetime']} "
        f"size={h5_path.stat().st_size / 1e6:.1f}MB written in {time.perf_counter() - t0:.1f}s"
    )
    return h5_path


def build_parser() -> argparse.ArgumentParser:
    """CLI parser (see module docstring)."""
    p = argparse.ArgumentParser(
        description="Build the crypto 5-minute HDF5 panel (+debug panel) from 1-minute csv.gz files."
    )
    p.add_argument("--source-dir", type=Path, required=True, help="directory containing <SYMBOL>_1m.csv.gz files")
    p.add_argument("--out-dir", type=Path, required=True, help="full panel folder (FACTOR_CoSTEER_data_folder)")
    p.add_argument(
        "--debug-out-dir", type=Path, required=True, help="debug panel folder (FACTOR_CoSTEER_data_folder_debug)"
    )
    p.add_argument("--bar-minutes", type=int, default=5, help="bar size in minutes (default 5)")
    p.add_argument(
        "--debug-symbols", type=int, default=10, help="instruments kept in the debug panel (top by total $amount)"
    )
    p.add_argument("--debug-days", type=int, default=14, help="days kept in the debug panel, ending at the last bar")
    p.add_argument("--panel-filename", default="crypto_5m.h5", help="HDF5 file name written in both folders")
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point: build the full and debug panels and print a summary for both."""
    args = build_parser().parse_args(argv)
    if args.bar_minutes <= 0:
        raise SystemExit("--bar-minutes must be positive")
    t0 = time.perf_counter()
    panel = build_panel(args.source_dir, args.bar_minutes)
    print(f"panel built in {time.perf_counter() - t0:.1f}s")
    write_folder(panel, args.out_dir, args.bar_minutes, args.panel_filename, label="full")
    debug = select_debug_panel(panel, args.debug_symbols, args.debug_days)
    write_folder(debug, args.debug_out_dir, args.bar_minutes, args.panel_filename, label="debug")
    print(f"total {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
