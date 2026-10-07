"""Built-in indicator set, registered on import."""

from __future__ import annotations

from decimal import Decimal

from ..marketdata.types import CandleSeries, SignalDirection
from . import core
from .registry import IndicatorOutput, register

FIFTY = Decimal(50)
SEVENTY = Decimal(70)
THIRTY = Decimal(30)
EIGHTY = Decimal(80)
TWENTY = Decimal(20)


def _direction(is_long: bool, is_short: bool) -> SignalDirection:
    if is_long and not is_short:
        return SignalDirection.LONG
    if is_short and not is_long:
        return SignalDirection.SHORT
    return SignalDirection.NEUTRAL


@register("ma", min_candles=50, description="SMA 20/50 trend", category="trend")
def _ma(series: CandleSeries) -> IndicatorOutput:
    closes = series.closes()
    fast = core.last_value(core.sma(closes, 20))
    slow = core.last_value(core.sma(closes, 50))
    values = {}
    if fast is not None:
        values["ma20"] = fast
    if slow is not None:
        values["ma50"] = slow
    direction = SignalDirection.NEUTRAL
    if fast is not None and slow is not None:
        direction = _direction(fast > slow, fast < slow)
    return IndicatorOutput(values=values, direction=direction)


@register("ema", min_candles=26, description="EMA 12/26 cross", category="trend")
def _ema(series: CandleSeries) -> IndicatorOutput:
    closes = series.closes()
    fast = core.last_value(core.ema(closes, 12))
    slow = core.last_value(core.ema(closes, 26))
    values = {}
    if fast is not None:
        values["ema12"] = fast
    if slow is not None:
        values["ema26"] = slow
    direction = SignalDirection.NEUTRAL
    if fast is not None and slow is not None:
        direction = _direction(fast > slow, fast < slow)
    return IndicatorOutput(values=values, direction=direction)


@register("vwap", min_candles=5, description="Session VWAP vs close", category="volume")
def _vwap(series: CandleSeries) -> IndicatorOutput:
    value = core.last_value(core.vwap(series.candles))
    last = series.last
    if value is None or last is None:
        return IndicatorOutput()
    return IndicatorOutput(
        values={"vwap": value},
        direction=_direction(last.close > value, last.close < value),
    )


@register("rsi", min_candles=15, description="Wilder RSI(14)", category="momentum")
def _rsi(series: CandleSeries) -> IndicatorOutput:
    value = core.last_value(core.rsi(series.closes(), 14))
    if value is None:
        return IndicatorOutput()
    note = "overbought" if value >= SEVENTY else "oversold" if value <= THIRTY else ""
    return IndicatorOutput(
        values={"rsi14": value},
        direction=_direction(value > FIFTY, value < FIFTY),
        note=note,
    )


@register("macd", min_candles=35, description="MACD 12/26/9", category="momentum")
def _macd(series: CandleSeries) -> IndicatorOutput:
    line, signal, histogram = core.macd(series.closes())
    macd_value = core.last_value(line)
    signal_value = core.last_value(signal)
    hist_value = core.last_value(histogram)
    values = {}
    if macd_value is not None:
        values["macd"] = macd_value
    if signal_value is not None:
        values["signal"] = signal_value
    if hist_value is not None:
        values["hist"] = hist_value
    direction = SignalDirection.NEUTRAL
    if hist_value is not None:
        direction = _direction(hist_value > 0, hist_value < 0)
    return IndicatorOutput(values=values, direction=direction)


@register("bbands", min_candles=20, description="Bollinger 20/2", category="volatility")
def _bbands(series: CandleSeries) -> IndicatorOutput:
    closes = series.closes()
    middle, upper, lower = core.bollinger(closes, 20, Decimal(2))
    mid = core.last_value(middle)
    top = core.last_value(upper)
    bottom = core.last_value(lower)
    if mid is None or top is None or bottom is None:
        return IndicatorOutput()
    last = closes[-1]
    width = top - bottom
    percent_b = (last - bottom) / width * Decimal(100) if width else Decimal(50)
    return IndicatorOutput(
        values={"upper": top, "middle": mid, "lower": bottom, "percent_b": percent_b},
        direction=_direction(last > mid, last < mid),
        note="upper band" if last >= top else "lower band" if last <= bottom else "",
    )


@register("atr", min_candles=15, description="ATR(14) and volatility", category="volatility")
def _atr(series: CandleSeries) -> IndicatorOutput:
    value = core.last_value(core.atr(series.candles, 14))
    volatility = core.realized_volatility(series.closes(), 20)
    values = {}
    if value is not None:
        values["atr14"] = value
        last = series.last
        if last is not None and last.close:
            values["atr_pct"] = value / last.close * Decimal(100)
    if volatility is not None:
        values["rv20_pct"] = volatility
    return IndicatorOutput(values=values)


@register("obv", min_candles=10, description="On-balance volume slope", category="volume")
def _obv(series: CandleSeries) -> IndicatorOutput:
    values_list = core.obv(series.candles)
    if len(values_list) < 6:
        return IndicatorOutput()
    latest = values_list[-1]
    reference = values_list[-6]
    return IndicatorOutput(
        values={"obv": latest, "obv_delta5": latest - reference},
        direction=_direction(latest > reference, latest < reference),
    )


@register("stoch", min_candles=17, description="Stochastic 14/3", category="momentum")
def _stoch(series: CandleSeries) -> IndicatorOutput:
    percent_k, percent_d = core.stochastic(series.candles, 14, 3)
    k_value = core.last_value(percent_k)
    d_value = core.last_value(percent_d)
    values = {}
    if k_value is not None:
        values["k"] = k_value
    if d_value is not None:
        values["d"] = d_value
    direction = SignalDirection.NEUTRAL
    note = ""
    if k_value is not None and d_value is not None:
        direction = _direction(k_value > d_value, k_value < d_value)
        note = "overbought" if k_value >= EIGHTY else "oversold" if k_value <= TWENTY else ""
    return IndicatorOutput(values=values, direction=direction, note=note)


@register("mfi", min_candles=15, description="Money flow index(14)", category="volume")
def _mfi(series: CandleSeries) -> IndicatorOutput:
    value = core.last_value(core.mfi(series.candles, 14))
    if value is None:
        return IndicatorOutput()
    note = "distribution" if value >= EIGHTY else "accumulation" if value <= TWENTY else ""
    return IndicatorOutput(
        values={"mfi14": value},
        direction=_direction(value > FIFTY, value < FIFTY),
        note=note,
    )
