# How to read files
For example, if you want to read `{{PANEL_FILENAME}}`
```Python
import pandas as pd
df = pd.read_hdf("{{PANEL_FILENAME}}", key="data")
```
NOTE: **key is always "data" for all hdf5 files**.

# Data description

| Filename | Description |
| -------- | ----------- |
| "{{PANEL_FILENAME}}" | Binance USDT spot {{BAR_MINUTES}}-minute OHLCV bars ({{N_INSTRUMENTS}} instruments, {{FIRST_DATETIME}} .. {{LAST_DATETIME}} UTC, {{N_ROWS}} rows). |

This file in this folder: {{N_INSTRUMENTS}} instruments, {{N_ROWS}} rows, from {{FIRST_DATETIME}} to {{LAST_DATETIME}} (UTC, bar open times).

## Index
- `MultiIndex` with level names exactly `["datetime", "instrument"]`, sorted lexicographically.
- `datetime`: tz-naive UTC `datetime64[ns]` on a {{BAR_MINUTES}}-minute grid (minute is always a multiple of {{BAR_MINUTES}}).
- `instrument`: the Binance spot symbol string, e.g. `"BTCUSDT"`, `"ETHUSDT"`.
- Example index entry: `(Timestamp('2026-03-01 00:05:00'), 'BTCUSDT')`.

## Timing semantics (read carefully, this is where look-ahead bugs come from)
- `datetime` is the bar OPEN time. Bar `t` covers the interval `[t, t + {{BAR_MINUTES}}min)`.
- Every column of row `t` becomes known at `t + {{BAR_MINUTES}}min` (bar close). A signal emitted for timestamp `t` may use panel rows with `datetime <= t` only; row `t` itself is usable for a signal emitted at `t + {{BAR_MINUTES}}min`.
- Never use rows with `datetime > t` when computing the factor value at `t`. Any use of future bars (e.g. `shift(-1)`, centered rolling windows, `bfill`, `.iloc[::-1]` cumulative operations, resampling that labels bars by their close but reads the whole future window) is look-ahead and invalidates the factor.
- Rolling / expanding computations must be done per instrument (`groupby(level="instrument")`) with windows that only look backwards.

## Columns (all `float64`)
- `$open`: first traded price inside the bar.
- `$high`: highest traded price inside the bar.
- `$low`: lowest traded price inside the bar.
- `$close`: last traded price inside the bar.
- `$volume`: base-asset volume traded inside the bar (e.g. BTC for BTCUSDT), summed over the one-minute bars.
- `$amount`: quote-asset (USDT) turnover inside the bar, summed over the one-minute bars.
- `$trade_count`: number of trades inside the bar, summed.
- `$taker_buy_volume`: base-asset volume where the taker side was a buyer (aggressive buying), summed.
- `$taker_buy_amount`: quote-asset (USDT) turnover where the taker side was a buyer, summed.
- Derived quantities that are often useful: taker-buy imbalance `$taker_buy_volume / $volume`, VWAP `$amount / $volume`, average trade size `$amount / $trade_count`, bar range `($high - $low) / $close`.

## Coverage and gaps
- A row exists only when at least one one-minute bar with trades existed inside the {{BAR_MINUTES}}-minute window. Missing bars are ABSENT rows, not NaN rows; there is no forward fill and no NaN padding.
- Instruments listed on the exchange after the panel start (e.g. GRAMUSDT, REUSDT) begin at their first traded bar, so the cross-section is smaller early in the panel.
- Because rows can be absent, per-instrument lags must be computed on the time index, not by positional `shift()` alone, if exact bar spacing matters. Reindexing an instrument onto the full {{BAR_MINUTES}}-minute grid (without filling values) is an acceptable way to get exact lags.
- The market trades 24/7; there are no sessions, no scheduled close, no weekends and no holidays. "Time of day" effects refer to UTC hours.

## Output expectation for factor code
- The factor result must be a DataFrame with the same `(datetime, instrument)` index style, `datetime` values on the panel grid, and exactly one float column named after the factor, saved to `result.h5` with key `"data"`.
