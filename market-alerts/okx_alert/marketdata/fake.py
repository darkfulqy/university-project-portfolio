"""Deterministic offline providers.

These back the ``--demo`` dashboard and every unit test, so no test ever needs
the network. The generator is stateless: a given (identity, timeframe, bucket)
always yields the same candle, which keeps rendering stable while the window
slides forward.

Shape factors are computed with floats for the trigonometric mixing, then
quantized to ``Decimal`` before touching any price. Prices themselves are never
float.
"""

from __future__ import annotations

import hashlib
import math
import re
from decimal import Decimal
from typing import Callable

from .protocols import NullExchangeProvider, NullOnchainProvider
from .types import (
    Candle,
    CandleSeries,
    Channel,
    DataQuality,
    Freshness,
    FundingSnapshot,
    InstrumentRef,
    LiquidityRouteSnapshot,
    OnchainQuoteSnapshot,
    OnchainTokenSnapshot,
    OpenInterestSnapshot,
    OrderBookLevel,
    OrderBookSnapshot,
    ProviderResult,
    ProviderState,
    ProviderStatus,
    RouteHop,
    TickerSnapshot,
    Timeframe,
    TokenRef,
    TradePrint,
    TradeSide,
)

ANCHOR_PRICES: dict[str, Decimal] = {
    "BTC": Decimal("68000"),
    "ETH": Decimal("3450"),
    "SOL": Decimal("176.4"),
    "OKB": Decimal("48.2"),
    "DOGE": Decimal("0.1428"),
    "USDT": Decimal("1.0"),
    "USDC": Decimal("1.0"),
    "CYBERLEEK": Decimal("0.0198"),
}

SYMBOL_ALIASES: dict[str, str] = {"WSOL": "SOL", "WETH": "ETH", "WBTC": "BTC", "USD": "USDT"}


def _digest(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _unit(*parts: object) -> float:
    """Deterministic value in [0, 1)."""
    return (_digest(*parts) % 1_000_000) / 1_000_000


def _signed(*parts: object) -> float:
    return _unit(*parts) * 2 - 1


def _canonical(symbol: str) -> str:
    upper = symbol.strip().upper()
    return SYMBOL_ALIASES.get(upper, upper)


def split_pair(symbol: str) -> tuple[str, str]:
    """Split ``BASE/QUOTE``, ``BASE-QUOTE`` or ``BASE-QUOTE-SWAP`` into its legs."""
    parts = [part for part in re.split(r"[/\-_]", symbol.strip()) if part]
    if not parts:
        return "", ""
    return _canonical(parts[0]), _canonical(parts[1]) if len(parts) > 1 else ""


def usd_anchor(symbol: str) -> Decimal:
    """USD-denominated anchor for a single asset."""
    base = _canonical(symbol)
    known = ANCHOR_PRICES.get(base)
    if known is not None:
        return known
    exponent = _digest("exp", base) % 9 - 5
    mantissa = Decimal(1) + Decimal(_digest("man", base) % 900) / Decimal(100)
    return (mantissa * (Decimal(10) ** exponent)).normalize()


def anchor_price(symbol: str) -> Decimal:
    """Anchor for a market, expressed in its own quote asset.

    A micro-cap like CYBERLEEK quoted in SOL must land near 1e-4, otherwise the
    demo dashboard shows a derived CyberLeek/USDT price off by four orders of
    magnitude. Anchors are stored in USD and converted through the quote leg.
    """
    base, quote = split_pair(symbol)
    price = usd_anchor(base)
    quote_usd = ANCHOR_PRICES.get(quote) if quote else None
    if quote_usd is None or quote_usd == 1:
        return price
    return price / quote_usd


def _shape(identity: str, bucket: int) -> Decimal:
    """Smooth multiplicative factor around 1.0 for a given absolute bucket."""
    phase_a = _unit(identity, "a") * math.tau
    phase_b = _unit(identity, "b") * math.tau
    phase_c = _unit(identity, "c") * math.tau
    phase_d = _unit(identity, "d") * math.tau
    factor = (
        1.0
        + 0.048 * math.sin(bucket / 617.0 + phase_d)
        + 0.031 * math.sin(bucket / 41.0 + phase_a)
        + 0.017 * math.sin(bucket / 11.0 + phase_b)
        + 0.008 * math.sin(bucket / 3.0 + phase_c)
        + 0.004 * _signed(identity, bucket)
    )
    return Decimal(str(round(max(factor, 0.05), 9)))


def _tick(reference: Decimal) -> Decimal:
    """Tick size that suits the magnitude of ``reference``."""
    if reference >= 1000:
        return Decimal("0.01")
    if reference >= 1:
        return Decimal("0.0001")
    if reference >= Decimal("0.001"):
        return Decimal("0.00000001")
    return Decimal("0.000000000001")


def _quantize(value: Decimal, reference: Decimal) -> Decimal:
    """Round to a tick size that suits the magnitude of ``reference``."""
    return value.quantize(_tick(reference))


def build_candles(
    identity: str,
    symbol: str,
    timeframe: Timeframe,
    now_ms: int,
    limit: int,
) -> tuple[Candle, ...]:
    anchor = anchor_price(symbol)
    step_ms = timeframe.milliseconds
    current_bucket = now_ms // step_ms
    scale = Decimal(str(round(math.sqrt(timeframe.seconds / 60.0), 6)))
    candles: list[Candle] = []
    for offset in range(limit - 1, -1, -1):
        bucket = current_bucket - offset
        open_price = anchor * _shape(identity, bucket)
        close_price = anchor * _shape(identity, bucket + 1)
        span = abs(close_price - open_price) + anchor * Decimal("0.0016") * scale
        high = max(open_price, close_price) + span * Decimal(str(round(_unit(identity, bucket, "h"), 6)))
        low = min(open_price, close_price) - span * Decimal(str(round(_unit(identity, bucket, "l"), 6)))
        volume = (
            Decimal(str(round(400 + 900 * _unit(identity, bucket, "v"), 4)))
            * scale
        )
        candles.append(
            Candle(
                start_ms=bucket * step_ms,
                open=_quantize(open_price, anchor),
                high=_quantize(high, anchor),
                low=_quantize(max(low, anchor / Decimal(1000)), anchor),
                close=_quantize(close_price, anchor),
                volume=volume,
                quote_volume=volume * close_price,
                trades=int(80 + 400 * _unit(identity, bucket, "t")),
                complete=offset > 0,
            )
        )
    return tuple(candles)


def _stats_24h(identity: str, last: Decimal, now_ms: int) -> tuple[Decimal, Decimal, Decimal]:
    """Rolling-day open/high/low that stay consistent with ``last``.

    The daily candle lives on its own bucket scale and knows nothing about the
    minute close, so reading the statistics off it left ``last`` outside the
    day range and clamping collapsed one extreme onto ``last``. Everything is
    derived from ``last`` instead: the open sits within a few percent of it,
    then the band is widened outwards so both extremes are strictly beyond the
    open/last pair.
    """
    bucket = now_ms // Timeframe.D1.milliseconds
    tick = _tick(last)
    change = Decimal(str(round(_signed(identity, bucket, "chg24") * 0.06, 6)))
    open_24h = _quantize(last / (Decimal(1) + change), last)
    top = max(open_24h, last)
    bottom = min(open_24h, last)
    # Day range of 2%-8%, but never narrower than the open-to-last move itself.
    span = last * Decimal(str(round(0.02 + 0.06 * _unit(identity, bucket, "span24"), 6)))
    extra = max(span - (top - bottom), last * Decimal("0.004"))
    upper = Decimal(str(round(0.3 + 0.4 * _unit(identity, bucket, "skew24"), 6)))
    high_24h = max(_quantize(top + extra * upper, last), top + tick)
    low_24h = min(_quantize(bottom - extra * (Decimal(1) - upper), last), bottom - tick)
    return open_24h, high_24h, max(low_24h, tick)


class FakeExchangeProvider(NullExchangeProvider):
    """Full offline implementation of the exchange surface."""

    def __init__(
        self,
        clock_ms: Callable[[], int],
        provider_id: str = "fake-exchange",
        channel: Channel = Channel.WEBSOCKET,
    ) -> None:
        self.provider_id = provider_id
        self.channel = channel
        self.clock_ms = clock_ms
        self.request_count = 0

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider_id=self.provider_id,
            state=ProviderState.ONLINE,
            channel=self.channel,
            latency_ms=12,
            last_ok_ms=self.clock_ms(),
            detail="deterministic offline generator",
        )

    def _freshness(self, now_ms: int) -> Freshness:
        return Freshness(
            as_of_ms=now_ms, received_ms=now_ms, max_age_ms=15_000, channel=self.channel
        )

    def ticker(self, instrument: InstrumentRef) -> ProviderResult[TickerSnapshot]:
        self.request_count += 1
        now = self.clock_ms()
        day = build_candles(instrument.identity, instrument.display, Timeframe.D1, now, 2)
        minute = build_candles(instrument.identity, instrument.display, Timeframe.M1, now, 2)
        last = minute[-1].close
        open_24h, high_24h, low_24h = _stats_24h(instrument.identity, last, now)
        tick = last * Decimal("0.0002")
        return ProviderResult.ok(
            TickerSnapshot(
                identity=instrument.identity,
                symbol=instrument.display,
                last=last,
                freshness=self._freshness(now),
                source=self.provider_id,
                bid=_quantize(last - tick, last),
                ask=_quantize(last + tick, last),
                open_24h=open_24h,
                high_24h=high_24h,
                low_24h=low_24h,
                volume_24h_base=day[-1].volume,
                volume_24h_quote=day[-1].quote_volume,
                quality=DataQuality.LIVE,
            )
        )

    def candles(
        self, instrument: InstrumentRef, timeframe: Timeframe, limit: int = 200
    ) -> ProviderResult[CandleSeries]:
        self.request_count += 1
        now = self.clock_ms()
        return ProviderResult.ok(
            CandleSeries(
                identity=instrument.identity,
                symbol=instrument.display,
                timeframe=timeframe,
                candles=build_candles(
                    instrument.identity, instrument.display, timeframe, now, limit
                ),
                source=self.provider_id,
                freshness=self._freshness(now),
            )
        )

    def order_book(
        self, instrument: InstrumentRef, depth: int = 20
    ) -> ProviderResult[OrderBookSnapshot]:
        self.request_count += 1
        now = self.clock_ms()
        mid = self.ticker(instrument).unwrap().last
        tick = max(mid * Decimal("0.0001"), Decimal("0.00000001"))
        bids: list[OrderBookLevel] = []
        asks: list[OrderBookLevel] = []
        for level in range(depth):
            skew = Decimal(str(round(0.6 + _unit(instrument.identity, now // 5000, level), 4)))
            bids.append(
                OrderBookLevel(
                    price=_quantize(mid - tick * (level + 1), mid),
                    size=(Decimal(3) + skew) * Decimal(level + 1),
                    orders=level + 2,
                )
            )
            asks.append(
                OrderBookLevel(
                    price=_quantize(mid + tick * (level + 1), mid),
                    size=(Decimal(3) + Decimal(1) / skew) * Decimal(level + 1),
                    orders=level + 2,
                )
            )
        return ProviderResult.ok(
            OrderBookSnapshot(
                identity=instrument.identity,
                symbol=instrument.display,
                bids=tuple(bids),
                asks=tuple(asks),
                freshness=self._freshness(now),
                source=self.provider_id,
            )
        )

    def trades(
        self, instrument: InstrumentRef, limit: int = 50
    ) -> ProviderResult[tuple[TradePrint, ...]]:
        self.request_count += 1
        now = self.clock_ms()
        mid = self.ticker(instrument).unwrap().last
        prints: list[TradePrint] = []
        for index in range(limit):
            seed = now // 1000 - index
            side = TradeSide.BUY if _unit(instrument.identity, seed, "side") > 0.5 else TradeSide.SELL
            drift = Decimal(str(round(_signed(instrument.identity, seed, "px") * 0.0004, 8)))
            prints.append(
                TradePrint(
                    identity=instrument.identity,
                    trade_id=str(_digest(instrument.identity, seed) % 10_000_000),
                    price=_quantize(mid * (Decimal(1) + drift), mid),
                    size=Decimal(str(round(0.05 + 4 * _unit(instrument.identity, seed, "sz"), 4))),
                    side=side,
                    timestamp_ms=now - index * 1_200,
                    source=self.provider_id,
                )
            )
        return ProviderResult.ok(tuple(prints))

    def funding(self, instrument: InstrumentRef) -> ProviderResult[FundingSnapshot]:
        now = self.clock_ms()
        rate = Decimal(str(round(_signed(instrument.identity, "funding") * 0.0004, 8)))
        return ProviderResult.ok(
            FundingSnapshot(
                identity=instrument.identity,
                funding_rate=rate,
                next_funding_ms=now + 3_600_000,
                freshness=self._freshness(now),
                source=self.provider_id,
                predicted_rate=rate * Decimal("1.1"),
                interval_hours=1,
            )
        )

    def open_interest(
        self, instrument: InstrumentRef
    ) -> ProviderResult[OpenInterestSnapshot]:
        now = self.clock_ms()
        value = Decimal(str(round(1_000_000 * (1 + _unit(instrument.identity, "oi")), 2)))
        return ProviderResult.ok(
            OpenInterestSnapshot(
                identity=instrument.identity,
                open_interest=value,
                open_interest_ccy=value / Decimal(10),
                freshness=self._freshness(now),
                source=self.provider_id,
            )
        )


class FakeOnchainProvider(NullOnchainProvider):
    """Offline pool prices, read-only swap quotes and routes."""

    def __init__(
        self,
        clock_ms: Callable[[], int],
        provider_id: str = "fake-onchain",
        channel: Channel = Channel.REST,
    ) -> None:
        self.provider_id = provider_id
        self.channel = channel
        self.clock_ms = clock_ms

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider_id=self.provider_id,
            state=ProviderState.ONLINE,
            channel=self.channel,
            latency_ms=140,
            last_ok_ms=self.clock_ms(),
            detail="deterministic offline generator",
        )

    def _freshness(self, now_ms: int) -> Freshness:
        return Freshness(
            as_of_ms=now_ms, received_ms=now_ms, max_age_ms=45_000, channel=self.channel
        )

    def token(
        self, token: TokenRef, quote_symbol: str = ""
    ) -> ProviderResult[OnchainTokenSnapshot]:
        now = self.clock_ms()
        identity = f"{token.identity}:{quote_symbol}" if quote_symbol else token.identity
        candles = build_candles(identity, token.display, Timeframe.M1, now, 1)
        price = candles[-1].close
        usd_rate = ANCHOR_PRICES.get(quote_symbol.upper(), Decimal(1))
        return ProviderResult.ok(
            OnchainTokenSnapshot(
                identity=identity,
                token=token,
                price_native=price,
                price_usd=_quantize(price * usd_rate, price * usd_rate),
                freshness=self._freshness(now),
                source=self.provider_id,
                quote_symbol=quote_symbol,
                pool_address=f"pool{_digest(identity) % 10**8:08d}",
                dex="raydium" if token.chain == "solana" else "uniswap-v3",
                liquidity_usd=Decimal(str(round(50_000 + 900_000 * _unit(identity, "liq"), 2))),
                volume_24h_usd=Decimal(str(round(20_000 + 400_000 * _unit(identity, "vol"), 2))),
                fdv_usd=Decimal(str(round(1_000_000 + 40_000_000 * _unit(identity, "fdv"), 2))),
            )
        )

    def candles(
        self,
        token: TokenRef,
        quote_symbol: str = "",
        timeframe: Timeframe = Timeframe.M15,
        limit: int = 200,
    ) -> ProviderResult[CandleSeries]:
        now = self.clock_ms()
        identity = f"{token.identity}:{quote_symbol}" if quote_symbol else token.identity
        return ProviderResult.ok(
            CandleSeries(
                identity=identity,
                symbol=f"{token.display}/{quote_symbol}" if quote_symbol else token.display,
                timeframe=timeframe,
                candles=build_candles(identity, token.display, timeframe, now, limit),
                source=self.provider_id,
                freshness=self._freshness(now),
            )
        )

    def quote(
        self, base: TokenRef, quote: TokenRef, amount_in: Decimal
    ) -> ProviderResult[OnchainQuoteSnapshot]:
        now = self.clock_ms()
        identity = f"{base.identity}->{quote.identity}"
        rate = anchor_price(base.display) / max(anchor_price(quote.display), Decimal("0.000001"))
        impact = Decimal(str(round(4 + 60 * _unit(identity, "impact"), 2)))
        amount_out = amount_in * rate * (Decimal(1) - impact / Decimal(10_000))
        return ProviderResult.ok(
            OnchainQuoteSnapshot(
                identity=identity,
                base=base,
                quote=quote,
                amount_in=amount_in,
                amount_out=amount_out,
                price_impact_bps=impact,
                freshness=self._freshness(now),
                source=self.provider_id,
                slippage_bps=Decimal("50"),
                gas_estimate=Decimal("180000"),
                gas_usd=Decimal(str(round(0.2 + 3 * _unit(identity, "gas"), 4))),
            )
        )

    def liquidity_route(
        self, base: TokenRef, quote: TokenRef, amount_in: Decimal
    ) -> ProviderResult[LiquidityRouteSnapshot]:
        now = self.clock_ms()
        identity = f"{base.identity}->{quote.identity}"
        primary = Decimal(str(round(55 + 30 * _unit(identity, "hop"), 2)))
        hops = (
            RouteHop(dex="raydium", pool_address=f"pool{_digest(identity, 1) % 10**8:08d}", share_percent=primary),
            RouteHop(
                dex="orca",
                pool_address=f"pool{_digest(identity, 2) % 10**8:08d}",
                share_percent=Decimal(100) - primary,
            ),
        )
        return ProviderResult.ok(
            LiquidityRouteSnapshot(
                identity=identity,
                hops=hops,
                total_liquidity_usd=Decimal(str(round(120_000 + 800_000 * _unit(identity, "tl"), 2))),
                freshness=self._freshness(now),
                source=self.provider_id,
                estimated_slippage_bps=Decimal(str(round(8 + 40 * _unit(identity, "slip"), 2))),
            )
        )
