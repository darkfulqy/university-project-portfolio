"""Indicator engine: registry specs applied to a candle series."""

from __future__ import annotations

from decimal import Decimal
from typing import Sequence

from ..marketdata.types import (
    CandleSeries,
    IndicatorSnapshot,
    OrderBookSnapshot,
    ProviderResult,
    SignalDirection,
    Timeframe,
    TradePrint,
    UnavailableReason,
)
from . import builtins as _builtins  # noqa: F401  (registers the built-in set)
from . import core
from .registry import registered, specs


class IndicatorEngine:
    """Implements ``IndicatorProvider`` on top of the plugin registry."""

    provider_id = "indicator-engine"

    def available(self) -> tuple[str, ...]:
        return registered()

    def compute(
        self, series: CandleSeries, names: tuple[str, ...] = ()
    ) -> ProviderResult[tuple[IndicatorSnapshot, ...]]:
        if not series.candles:
            return ProviderResult.fail(
                UnavailableReason.STALE_DATA,
                self.provider_id,
                f"no candles for {series.symbol} {series.timeframe.value}",
            )
        wanted = set(names) if names else None
        updated = series.candles[-1].start_ms
        snapshots: list[IndicatorSnapshot] = []
        for spec in specs():
            if wanted is not None and spec.name not in wanted:
                continue
            if len(series.candles) < spec.min_candles:
                snapshots.append(
                    IndicatorSnapshot(
                        identity=series.identity,
                        name=spec.name,
                        timeframe=series.timeframe,
                        values={},
                        updated_ms=updated,
                        note=f"需要 {spec.min_candles} 根K线，当前 {len(series.candles)}",
                    )
                )
                continue
            output = spec.compute(series)
            snapshots.append(
                IndicatorSnapshot(
                    identity=series.identity,
                    name=spec.name,
                    timeframe=series.timeframe,
                    values=dict(output.values),
                    updated_ms=updated,
                    direction=output.direction,
                    note=output.note,
                )
            )
        return ProviderResult.ok(tuple(snapshots))


def order_book_indicator(book: OrderBookSnapshot, levels: int = 10) -> IndicatorSnapshot:
    """Order book imbalance as an indicator snapshot."""
    imbalance = book.imbalance(levels)
    values: dict[str, Decimal] = {}
    direction = SignalDirection.NEUTRAL
    if imbalance is not None:
        values["imbalance"] = imbalance * Decimal(100)
        threshold = Decimal("0.08")
        if imbalance > threshold:
            direction = SignalDirection.LONG
        elif imbalance < -threshold:
            direction = SignalDirection.SHORT
    spread = book.spread
    if spread is not None:
        values["spread"] = spread
    return IndicatorSnapshot(
        identity=book.identity,
        name="book_imbalance",
        timeframe=Timeframe.M1,
        values=values,
        updated_ms=book.freshness.as_of_ms,
        direction=direction,
    )


def flow_indicator(
    identity: str, trades: Sequence[TradePrint], updated_ms: int
) -> IndicatorSnapshot:
    """Cumulative volume delta from the recent trade window."""
    buys, sells, delta = core.cumulative_volume_delta(trades)
    direction = SignalDirection.NEUTRAL
    if delta > 0:
        direction = SignalDirection.LONG
    elif delta < 0:
        direction = SignalDirection.SHORT
    return IndicatorSnapshot(
        identity=identity,
        name="cvd",
        timeframe=Timeframe.M1,
        values={"buy": buys, "sell": sells, "delta": delta},
        updated_ms=updated_ms,
        direction=direction,
        note=f"{len(trades)} trades",
    )
