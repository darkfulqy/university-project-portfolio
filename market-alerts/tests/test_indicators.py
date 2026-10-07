from __future__ import annotations

import unittest
from decimal import Decimal

from okx_alert.indicators import core
from okx_alert.indicators.engine import IndicatorEngine, flow_indicator, order_book_indicator
from okx_alert.indicators.registry import IndicatorOutput, register, registered, unregister
from okx_alert.marketdata.fake import FakeExchangeProvider
from okx_alert.marketdata.identity import instrument_from_id
from okx_alert.marketdata.types import (
    Candle,
    CandleSeries,
    Channel,
    Freshness,
    OrderBookLevel,
    OrderBookSnapshot,
    SignalDirection,
    Timeframe,
    TradePrint,
    TradeSide,
)

NOW = 1_756_000_000_000
INSTRUMENT = instrument_from_id("BTC-USDT")


def _decimals(values: list[str]) -> tuple[Decimal, ...]:
    return tuple(Decimal(value) for value in values)


def _series(closes: list[str], timeframe: Timeframe = Timeframe.M15) -> CandleSeries:
    candles = []
    for index, close in enumerate(closes):
        value = Decimal(close)
        candles.append(
            Candle(
                start_ms=NOW + index * timeframe.milliseconds,
                open=value,
                high=value + Decimal("1"),
                low=value - Decimal("1"),
                close=value,
                volume=Decimal("100"),
            )
        )
    return CandleSeries(
        identity="test",
        symbol="TEST",
        timeframe=timeframe,
        candles=tuple(candles),
        source="test",
        freshness=Freshness(as_of_ms=NOW, received_ms=NOW, channel=Channel.COMPUTED),
    )


class CoreMathTests(unittest.TestCase):
    def test_sma_pads_warmup_and_averages(self) -> None:
        values = _decimals(["1", "2", "3", "4", "5"])
        output = core.sma(values, 3)
        self.assertEqual([None, None, Decimal(2), Decimal(3), Decimal(4)], output)

    def test_ema_seeds_from_sma(self) -> None:
        values = _decimals(["1", "2", "3", "4", "5"])
        output = core.ema(values, 3)
        self.assertEqual(Decimal(2), output[2])
        self.assertEqual(Decimal(3), output[3])
        self.assertEqual(Decimal(4), output[4])

    def test_rsi_saturates_on_a_pure_uptrend(self) -> None:
        values = _decimals([str(value) for value in range(1, 20)])
        self.assertEqual(Decimal(100), core.last_value(core.rsi(values, 14)))

    def test_rsi_matches_known_series(self) -> None:
        closes = _decimals(
            [
                "44.34", "44.09", "44.15", "43.61", "44.33", "44.83", "45.10",
                "45.42", "45.84", "46.08", "45.89", "46.03", "45.61", "46.28",
                "46.28",
            ]
        )
        value = core.last_value(core.rsi(closes, 14))
        self.assertAlmostEqual(70.46, float(value), places=1)

    def test_bollinger_bands_are_symmetric(self) -> None:
        values = _decimals([str(v) for v in range(1, 25)])
        middle, upper, lower = core.bollinger(values, 20, Decimal(2))
        index = -1
        self.assertAlmostEqual(
            float(upper[index] - middle[index]), float(middle[index] - lower[index]), places=9
        )

    def test_atr_is_positive_and_wilder_smoothed(self) -> None:
        series = _series([str(100 + index) for index in range(30)])
        value = core.last_value(core.atr(series.candles, 14))
        self.assertGreater(value, Decimal(0))

    def test_obv_tracks_direction(self) -> None:
        series = _series(["10", "11", "9", "12"])
        self.assertEqual(
            [Decimal(0), Decimal(100), Decimal(0), Decimal(100)], core.obv(series.candles)
        )

    def test_stochastic_and_mfi_stay_in_range(self) -> None:
        series = _series([str(100 + (index % 7)) for index in range(40)])
        percent_k, percent_d = core.stochastic(series.candles)
        self.assertTrue(Decimal(0) <= core.last_value(percent_k) <= Decimal(100))
        self.assertTrue(Decimal(0) <= core.last_value(percent_d) <= Decimal(100))
        self.assertTrue(Decimal(0) <= core.last_value(core.mfi(series.candles)) <= Decimal(100))

    def test_vwap_weighs_by_volume(self) -> None:
        series = _series(["10", "20"])
        self.assertEqual(Decimal(10), core.vwap(series.candles)[0])
        self.assertEqual(Decimal(15), core.vwap(series.candles)[1])

    def test_macd_histogram_is_line_minus_signal(self) -> None:
        series = _series([str(100 + index * (1 if index % 3 else -2)) for index in range(60)])
        line, signal, histogram = core.macd(series.closes())
        self.assertEqual(
            core.last_value(line) - core.last_value(signal), core.last_value(histogram)
        )

    def test_cvd_and_basis(self) -> None:
        trades = (
            TradePrint("x", "1", Decimal("10"), Decimal("2"), TradeSide.BUY, NOW),
            TradePrint("x", "2", Decimal("10"), Decimal("5"), TradeSide.SELL, NOW),
        )
        buys, sells, delta = core.cumulative_volume_delta(trades)
        self.assertEqual((Decimal(2), Decimal(5), Decimal(-3)), (buys, sells, delta))
        self.assertEqual(Decimal(10), core.basis_percent(Decimal(100), Decimal(110)))
        self.assertIsNone(core.basis_percent(Decimal(0), Decimal(1)))

    def test_no_division_by_zero_on_flat_input(self) -> None:
        flat = _series(["5"] * 40)
        self.assertIsNotNone(core.last_value(core.rsi(flat.closes(), 14)))
        self.assertEqual(Decimal(0), core.realized_volatility(flat.closes(), 20))


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = IndicatorEngine()
        self.series = (
            FakeExchangeProvider(lambda: NOW).candles(INSTRUMENT, Timeframe.M15, 120).unwrap()
        )

    def test_all_builtin_indicators_are_registered(self) -> None:
        for name in ("ma", "ema", "vwap", "rsi", "macd", "bbands", "atr", "obv", "stoch", "mfi"):
            self.assertIn(name, self.engine.available())

    def test_compute_returns_one_snapshot_per_indicator(self) -> None:
        snapshots = self.engine.compute(self.series).unwrap()
        self.assertEqual(len(self.engine.available()), len(snapshots))
        self.assertTrue(all(snapshot.timeframe is Timeframe.M15 for snapshot in snapshots))
        rsi = next(snapshot for snapshot in snapshots if snapshot.name == "rsi")
        self.assertIn("rsi14", rsi.values)

    def test_short_history_reports_requirement_instead_of_guessing(self) -> None:
        snapshots = self.engine.compute(self.series.tail(5)).unwrap()
        macd = next(snapshot for snapshot in snapshots if snapshot.name == "macd")
        self.assertEqual({}, macd.values)
        self.assertIn("需要", macd.note)

    def test_empty_series_is_unavailable(self) -> None:
        empty = self.series.__class__(
            identity="x",
            symbol="X",
            timeframe=Timeframe.M1,
            candles=(),
            source="test",
            freshness=self.series.freshness,
        )
        result = self.engine.compute(empty)
        self.assertFalse(result.is_ok)

    def test_subset_selection(self) -> None:
        snapshots = self.engine.compute(self.series, ("rsi",)).unwrap()
        self.assertEqual(["rsi"], [snapshot.name for snapshot in snapshots])

    def test_microstructure_indicators(self) -> None:
        book = OrderBookSnapshot(
            identity="x",
            symbol="X",
            bids=(OrderBookLevel(Decimal(10), Decimal(9)),),
            asks=(OrderBookLevel(Decimal(11), Decimal(1)),),
            freshness=self.series.freshness,
            source="test",
        )
        snapshot = order_book_indicator(book)
        self.assertIs(SignalDirection.LONG, snapshot.direction)
        self.assertEqual(Decimal(80), snapshot.values["imbalance"])
        flow = flow_indicator(
            "x",
            (TradePrint("x", "1", Decimal(1), Decimal(3), TradeSide.SELL, NOW),),
            NOW,
        )
        self.assertIs(SignalDirection.SHORT, flow.direction)


class RegistryTests(unittest.TestCase):
    def tearDown(self) -> None:
        unregister("unittest_custom")

    def test_custom_indicator_plugs_in(self) -> None:
        @register("unittest_custom", min_candles=2, description="test plugin")
        def _custom(series):
            return IndicatorOutput(
                values={"last": series.candles[-1].close}, direction=SignalDirection.LONG
            )

        self.assertIn("unittest_custom", registered())
        series = _series(["1", "2", "3"])
        snapshots = IndicatorEngine().compute(series, ("unittest_custom",)).unwrap()
        self.assertEqual(Decimal(3), snapshots[0].values["last"])

    def test_duplicate_registration_is_rejected(self) -> None:
        register("unittest_custom")(lambda series: IndicatorOutput())
        with self.assertRaises(ValueError):
            register("unittest_custom")(lambda series: IndicatorOutput())


if __name__ == "__main__":
    unittest.main()
