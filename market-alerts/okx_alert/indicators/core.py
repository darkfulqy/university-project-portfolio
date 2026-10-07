"""Pure Decimal indicator math.

Every function takes and returns ``Decimal`` sequences and pads the warm-up
region with ``None`` so index ``i`` of the output always lines up with index
``i`` of the input.
"""

from __future__ import annotations

from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Sequence

from ..marketdata.types import ZERO, Candle, TradePrint, TradeSide

Series = Sequence[Decimal]
Optional = list[Decimal | None]

HUNDRED = Decimal(100)


def _safe_divide(numerator: Decimal, denominator: Decimal) -> Decimal | None:
    if denominator == ZERO:
        return None
    try:
        return numerator / denominator
    except (DivisionByZero, InvalidOperation):
        return None


def sma(values: Series, period: int) -> Optional:
    if period <= 0:
        raise ValueError("period must be positive")
    output: Optional = [None] * len(values)
    running = ZERO
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        if index >= period - 1:
            output[index] = running / Decimal(period)
    return output


def ema(values: Series, period: int) -> Optional:
    if period <= 0:
        raise ValueError("period must be positive")
    output: Optional = [None] * len(values)
    if len(values) < period:
        return output
    multiplier = Decimal(2) / Decimal(period + 1)
    seed = sum(values[:period], ZERO) / Decimal(period)
    output[period - 1] = seed
    previous = seed
    for index in range(period, len(values)):
        previous = (values[index] - previous) * multiplier + previous
        output[index] = previous
    return output


def stddev(values: Series, period: int) -> Optional:
    output: Optional = [None] * len(values)
    for index in range(period - 1, len(values)):
        window = values[index - period + 1 : index + 1]
        mean = sum(window, ZERO) / Decimal(period)
        variance = sum(((value - mean) ** 2 for value in window), ZERO) / Decimal(period)
        output[index] = variance.sqrt()
    return output


def bollinger(
    values: Series, period: int = 20, deviations: Decimal = Decimal(2)
) -> tuple[Optional, Optional, Optional]:
    middle = sma(values, period)
    spread = stddev(values, period)
    upper: Optional = [None] * len(values)
    lower: Optional = [None] * len(values)
    for index, (mid, dev) in enumerate(zip(middle, spread)):
        if mid is None or dev is None:
            continue
        upper[index] = mid + dev * deviations
        lower[index] = mid - dev * deviations
    return middle, upper, lower


def rsi(values: Series, period: int = 14) -> Optional:
    """Wilder's RSI."""
    output: Optional = [None] * len(values)
    if len(values) <= period:
        return output
    gains = ZERO
    losses = ZERO
    for index in range(1, period + 1):
        change = values[index] - values[index - 1]
        if change >= ZERO:
            gains += change
        else:
            losses -= change
    average_gain = gains / Decimal(period)
    average_loss = losses / Decimal(period)
    output[period] = _rsi_value(average_gain, average_loss)
    for index in range(period + 1, len(values)):
        change = values[index] - values[index - 1]
        gain = change if change > ZERO else ZERO
        loss = -change if change < ZERO else ZERO
        average_gain = (average_gain * Decimal(period - 1) + gain) / Decimal(period)
        average_loss = (average_loss * Decimal(period - 1) + loss) / Decimal(period)
        output[index] = _rsi_value(average_gain, average_loss)
    return output


def _rsi_value(average_gain: Decimal, average_loss: Decimal) -> Decimal:
    if average_loss == ZERO:
        return HUNDRED if average_gain > ZERO else Decimal(50)
    strength = average_gain / average_loss
    return HUNDRED - HUNDRED / (Decimal(1) + strength)


def macd(
    values: Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[Optional, Optional, Optional]:
    fast_line = ema(values, fast)
    slow_line = ema(values, slow)
    macd_line: Optional = [None] * len(values)
    for index, (quick, slowly) in enumerate(zip(fast_line, slow_line)):
        if quick is None or slowly is None:
            continue
        macd_line[index] = quick - slowly
    dense = [value for value in macd_line if value is not None]
    signal_dense = ema(dense, signal)
    signal_line: Optional = [None] * len(values)
    histogram: Optional = [None] * len(values)
    offset = len(values) - len(dense)
    for position, value in enumerate(signal_dense):
        if value is None:
            continue
        index = offset + position
        signal_line[index] = value
        line = macd_line[index]
        if line is not None:
            histogram[index] = line - value
    return macd_line, signal_line, histogram


def true_range(candles: Sequence[Candle]) -> list[Decimal]:
    ranges: list[Decimal] = []
    for index, candle in enumerate(candles):
        if index == 0:
            ranges.append(candle.high - candle.low)
            continue
        previous_close = candles[index - 1].close
        ranges.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        )
    return ranges


def atr(candles: Sequence[Candle], period: int = 14) -> Optional:
    ranges = true_range(candles)
    output: Optional = [None] * len(candles)
    if len(candles) < period:
        return output
    average = sum(ranges[:period], ZERO) / Decimal(period)
    output[period - 1] = average
    for index in range(period, len(candles)):
        average = (average * Decimal(period - 1) + ranges[index]) / Decimal(period)
        output[index] = average
    return output


def vwap(candles: Sequence[Candle]) -> Optional:
    """Session VWAP accumulated over the supplied window."""
    output: Optional = [None] * len(candles)
    cumulative_value = ZERO
    cumulative_volume = ZERO
    for index, candle in enumerate(candles):
        cumulative_value += candle.typical_price * candle.volume
        cumulative_volume += candle.volume
        output[index] = _safe_divide(cumulative_value, cumulative_volume)
    return output


def obv(candles: Sequence[Candle]) -> list[Decimal]:
    output: list[Decimal] = []
    running = ZERO
    for index, candle in enumerate(candles):
        if index == 0:
            output.append(running)
            continue
        previous_close = candles[index - 1].close
        if candle.close > previous_close:
            running += candle.volume
        elif candle.close < previous_close:
            running -= candle.volume
        output.append(running)
    return output


def stochastic(
    candles: Sequence[Candle], period: int = 14, smooth: int = 3
) -> tuple[Optional, Optional]:
    percent_k: Optional = [None] * len(candles)
    for index in range(period - 1, len(candles)):
        window = candles[index - period + 1 : index + 1]
        highest = max(candle.high for candle in window)
        lowest = min(candle.low for candle in window)
        ratio = _safe_divide(candles[index].close - lowest, highest - lowest)
        percent_k[index] = HUNDRED * ratio if ratio is not None else Decimal(50)
    dense = [value for value in percent_k if value is not None]
    smoothed = sma(dense, smooth)
    percent_d: Optional = [None] * len(candles)
    offset = len(candles) - len(dense)
    for position, value in enumerate(smoothed):
        if value is not None:
            percent_d[offset + position] = value
    return percent_k, percent_d


def mfi(candles: Sequence[Candle], period: int = 14) -> Optional:
    output: Optional = [None] * len(candles)
    if len(candles) <= period:
        return output
    flows: list[tuple[Decimal, Decimal]] = [(ZERO, ZERO)]
    for index in range(1, len(candles)):
        typical = candles[index].typical_price
        previous = candles[index - 1].typical_price
        raw = typical * candles[index].volume
        flows.append((raw, ZERO) if typical > previous else (ZERO, raw))
    for index in range(period, len(candles)):
        window = flows[index - period + 1 : index + 1]
        positive = sum((item[0] for item in window), ZERO)
        negative = sum((item[1] for item in window), ZERO)
        if negative == ZERO:
            output[index] = HUNDRED if positive > ZERO else Decimal(50)
            continue
        ratio = positive / negative
        output[index] = HUNDRED - HUNDRED / (Decimal(1) + ratio)
    return output


def realized_volatility(values: Series, period: int = 20) -> Decimal | None:
    """Standard deviation of simple returns over the window, in percent."""
    if len(values) < period + 1:
        return None
    returns: list[Decimal] = []
    for index in range(len(values) - period, len(values)):
        previous = values[index - 1]
        if previous == ZERO:
            continue
        returns.append((values[index] - previous) / previous)
    if len(returns) < 2:
        return None
    mean = sum(returns, ZERO) / Decimal(len(returns))
    variance = sum(((value - mean) ** 2 for value in returns), ZERO) / Decimal(len(returns))
    return variance.sqrt() * HUNDRED


def cumulative_volume_delta(trades: Sequence[TradePrint]) -> tuple[Decimal, Decimal, Decimal]:
    """Return (buy volume, sell volume, cumulative delta) for the trade window."""
    buys = sum((trade.size for trade in trades if trade.side is TradeSide.BUY), ZERO)
    sells = sum((trade.size for trade in trades if trade.side is TradeSide.SELL), ZERO)
    return buys, sells, buys - sells


def basis_percent(spot: Decimal, derivative: Decimal) -> Decimal | None:
    """Perp/futures premium over spot, in percent."""
    ratio = _safe_divide(derivative - spot, spot)
    return ratio * HUNDRED if ratio is not None else None


def last_value(values: Optional) -> Decimal | None:
    for value in reversed(values):
        if value is not None:
            return value
    return None
