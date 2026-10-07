"""Read-only market data transfer objects shared by providers and the TUI.

Every price, size and ratio is a :class:`~decimal.Decimal`. Floats are never used
for monetary values because binary rounding silently corrupts small-cap token
prices such as ``0.00000004217``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import StrEnum
from typing import Generic, TypeVar

T = TypeVar("T")

ZERO = Decimal(0)


class MarketKind(StrEnum):
    """How an asset is quoted, which decides the provider family used."""

    SPOT = "spot"
    SWAP = "swap"
    FUTURES = "futures"
    ONCHAIN = "onchain"
    DERIVED = "derived"


class Channel(StrEnum):
    """Transport a snapshot arrived on. WebSocket wins, REST compensates."""

    REST = "rest"
    WEBSOCKET = "websocket"
    CACHE = "cache"
    COMPUTED = "computed"
    REPLAY = "replay"


class Timeframe(StrEnum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1H"
    H4 = "4H"
    D1 = "1D"

    @property
    def seconds(self) -> int:
        return {
            Timeframe.M1: 60,
            Timeframe.M5: 300,
            Timeframe.M15: 900,
            Timeframe.H1: 3_600,
            Timeframe.H4: 14_400,
            Timeframe.D1: 86_400,
        }[self]

    @property
    def milliseconds(self) -> int:
        return self.seconds * 1_000

    def bucket_start_ms(self, timestamp_ms: int) -> int:
        return timestamp_ms - (timestamp_ms % self.milliseconds)


TIMEFRAME_ORDER: tuple[Timeframe, ...] = (
    Timeframe.M1,
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.H1,
    Timeframe.H4,
    Timeframe.D1,
)


class DataQuality(StrEnum):
    """Trust level of a snapshot, rendered as a colour in the dashboard."""

    LIVE = "live"
    DELAYED = "delayed"
    STALE = "stale"
    PARTIAL = "partial"
    SYNTHETIC = "synthetic"
    UNAVAILABLE = "unavailable"


class UnavailableReason(StrEnum):
    """Why a provider could not answer. Never fail silently: always say which."""

    NOT_IMPLEMENTED = "not_implemented"
    NO_CREDENTIALS = "no_credentials"
    UNSUPPORTED_MARKET = "unsupported_market"
    NETWORK_ERROR = "network_error"
    RATE_LIMITED = "rate_limited"
    UPSTREAM_ERROR = "upstream_error"
    INVALID_RESPONSE = "invalid_response"
    STALE_DATA = "stale_data"
    DISABLED = "disabled"


class ProviderState(StrEnum):
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    RATE_LIMITED = "rate_limited"
    NOT_IMPLEMENTED = "not_implemented"
    DISABLED = "disabled"


class SignalDirection(StrEnum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class TradeSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Unavailable:
    """A typed, explicit "no data" answer carrying a sanitized detail string."""

    reason: UnavailableReason
    provider_id: str
    detail: str = ""
    retry_after_seconds: float | None = None

    def __str__(self) -> str:
        text = f"{self.provider_id}: {self.reason.value}"
        return f"{text} ({self.detail})" if self.detail else text


@dataclass(frozen=True, slots=True)
class ProviderResult(Generic[T]):
    """Either a value or an :class:`Unavailable`. Constructed by the helpers."""

    value: T | None = None
    error: Unavailable | None = None

    @classmethod
    def ok(cls, value: T) -> "ProviderResult[T]":
        return cls(value=value)

    @classmethod
    def fail(
        cls,
        reason: UnavailableReason,
        provider_id: str,
        detail: str = "",
        retry_after_seconds: float | None = None,
    ) -> "ProviderResult[T]":
        return cls(
            error=Unavailable(
                reason=reason,
                provider_id=provider_id,
                detail=detail,
                retry_after_seconds=retry_after_seconds,
            )
        )

    @classmethod
    def not_implemented(cls, provider_id: str, feature: str) -> "ProviderResult[T]":
        return cls.fail(
            UnavailableReason.NOT_IMPLEMENTED,
            provider_id,
            f"{feature} is not wired to a real endpoint yet",
        )

    @property
    def is_ok(self) -> bool:
        return self.error is None and self.value is not None

    def unwrap(self) -> T:
        if self.value is None:
            raise LookupError(str(self.error) if self.error else "empty result")
        return self.value

    def or_none(self) -> T | None:
        return self.value


@dataclass(frozen=True, slots=True)
class Freshness:
    """Age bookkeeping for one snapshot."""

    as_of_ms: int
    received_ms: int
    max_age_ms: int = 30_000
    channel: Channel = Channel.REST

    def age_ms(self, now_ms: int) -> int:
        return max(0, now_ms - self.as_of_ms)

    def quality(self, now_ms: int) -> DataQuality:
        age = self.age_ms(now_ms)
        if age <= self.max_age_ms:
            return DataQuality.LIVE
        if age <= self.max_age_ms * 4:
            return DataQuality.DELAYED
        return DataQuality.STALE

    def age_text(self, now_ms: int) -> str:
        seconds = self.age_ms(now_ms) / 1000
        if seconds < 10:
            return f"{seconds:.1f}s"
        if seconds < 600:
            return f"{seconds:.0f}s"
        if seconds < 7200:
            return f"{seconds / 60:.0f}m"
        return f"{seconds / 3600:.0f}h"


@dataclass(frozen=True, slots=True)
class InstrumentRef:
    """A centralized-exchange instrument, e.g. OKX ``BTC-USDT`` spot."""

    exchange: str
    instrument_id: str
    base: str = ""
    quote: str = ""
    kind: MarketKind = MarketKind.SPOT

    @property
    def identity(self) -> str:
        return f"cex:{self.exchange.lower()}:{self.instrument_id.upper()}"

    @property
    def display(self) -> str:
        return self.instrument_id.upper()


@dataclass(frozen=True, slots=True)
class TokenRef:
    """An on-chain token identified by chain plus contract address."""

    chain: str
    address: str
    symbol: str = ""
    decimals: int | None = None

    @property
    def identity(self) -> str:
        address = self.address.lower() if self.address.startswith("0x") else self.address
        return f"onchain:{self.chain.lower()}:{address}"

    @property
    def display(self) -> str:
        return self.symbol.upper() or f"{self.chain}:{self.address[:6]}"


@dataclass(frozen=True, slots=True)
class TickerSnapshot:
    """Last price plus the 24h statistics rendered in the detail pane."""

    identity: str
    symbol: str
    last: Decimal
    freshness: Freshness
    source: str
    bid: Decimal | None = None
    ask: Decimal | None = None
    open_24h: Decimal | None = None
    high_24h: Decimal | None = None
    low_24h: Decimal | None = None
    volume_24h_base: Decimal | None = None
    volume_24h_quote: Decimal | None = None
    quality: DataQuality = DataQuality.LIVE

    @property
    def change_24h(self) -> Decimal | None:
        if self.open_24h is None or self.open_24h == ZERO:
            return None
        return (self.last - self.open_24h) / self.open_24h * Decimal(100)

    @property
    def spread(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @property
    def spread_bps(self) -> Decimal | None:
        spread = self.spread
        if spread is None or self.ask is None or self.ask == ZERO:
            return None
        mid = (self.ask + (self.bid or self.ask)) / Decimal(2)
        if mid == ZERO:
            return None
        return spread / mid * Decimal(10_000)


@dataclass(frozen=True, slots=True)
class Candle:
    start_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = ZERO
    quote_volume: Decimal | None = None
    trades: int | None = None
    complete: bool = True

    @property
    def is_up(self) -> bool:
        return self.close >= self.open

    @property
    def range(self) -> Decimal:
        return self.high - self.low

    @property
    def typical_price(self) -> Decimal:
        return (self.high + self.low + self.close) / Decimal(3)


@dataclass(frozen=True, slots=True)
class CandleSeries:
    identity: str
    symbol: str
    timeframe: Timeframe
    candles: tuple[Candle, ...]
    source: str
    freshness: Freshness
    quality: DataQuality = DataQuality.LIVE

    def __len__(self) -> int:
        return len(self.candles)

    @property
    def last(self) -> Candle | None:
        return self.candles[-1] if self.candles else None

    def closes(self) -> tuple[Decimal, ...]:
        return tuple(candle.close for candle in self.candles)

    def tail(self, count: int) -> "CandleSeries":
        if count >= len(self.candles):
            return self
        return replace(self, candles=self.candles[-count:])


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    price: Decimal
    size: Decimal
    orders: int | None = None


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    identity: str
    symbol: str
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]
    freshness: Freshness
    source: str
    quality: DataQuality = DataQuality.LIVE

    @property
    def best_bid(self) -> OrderBookLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> OrderBookLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / Decimal(2)

    @property
    def spread(self) -> Decimal | None:
        if not self.bids or not self.asks:
            return None
        return self.asks[0].price - self.bids[0].price

    def depth(self, side: str, levels: int = 10) -> Decimal:
        book = self.bids if side == "bid" else self.asks
        return sum((level.size for level in book[:levels]), ZERO)

    def imbalance(self, levels: int = 10) -> Decimal | None:
        """(bid - ask) / (bid + ask) over the top ``levels``; range [-1, 1]."""
        bid = self.depth("bid", levels)
        ask = self.depth("ask", levels)
        total = bid + ask
        if total == ZERO:
            return None
        return (bid - ask) / total


@dataclass(frozen=True, slots=True)
class TradePrint:
    identity: str
    trade_id: str
    price: Decimal
    size: Decimal
    side: TradeSide
    timestamp_ms: int
    source: str = ""

    @property
    def notional(self) -> Decimal:
        return self.price * self.size


@dataclass(frozen=True, slots=True)
class FundingSnapshot:
    identity: str
    funding_rate: Decimal
    next_funding_ms: int | None
    freshness: Freshness
    source: str
    predicted_rate: Decimal | None = None
    # OKX escalates 8h contracts to 4h/2h/1h while the rate pegs its clamp, so
    # the interval is derived from settlement timestamps; None means unknown.
    interval_hours: int | None = None


@dataclass(frozen=True, slots=True)
class OpenInterestSnapshot:
    identity: str
    open_interest: Decimal
    open_interest_ccy: Decimal | None
    freshness: Freshness
    source: str


@dataclass(frozen=True, slots=True)
class OnchainTokenSnapshot:
    """Pool-level view of a token: price, liquidity and the pool it came from."""

    identity: str
    token: TokenRef
    price_native: Decimal | None
    price_usd: Decimal | None
    freshness: Freshness
    source: str
    quote_symbol: str = ""
    pool_address: str | None = None
    dex: str | None = None
    liquidity_usd: Decimal | None = None
    volume_24h_usd: Decimal | None = None
    fdv_usd: Decimal | None = None
    quality: DataQuality = DataQuality.LIVE


@dataclass(frozen=True, slots=True)
class OnchainQuoteSnapshot:
    """A read-only swap quote. Quotes are never executed by this project."""

    identity: str
    base: TokenRef
    quote: TokenRef
    amount_in: Decimal
    amount_out: Decimal
    price_impact_bps: Decimal | None
    freshness: Freshness
    source: str
    slippage_bps: Decimal | None = None
    gas_estimate: Decimal | None = None
    gas_usd: Decimal | None = None


@dataclass(frozen=True, slots=True)
class RouteHop:
    dex: str
    pool_address: str
    share_percent: Decimal


@dataclass(frozen=True, slots=True)
class LiquidityRouteSnapshot:
    identity: str
    hops: tuple[RouteHop, ...]
    total_liquidity_usd: Decimal | None
    freshness: Freshness
    source: str
    estimated_slippage_bps: Decimal | None = None


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    provider_id: str
    state: ProviderState
    channel: Channel = Channel.REST
    latency_ms: int | None = None
    error_count: int = 0
    reconnects: int = 0
    rate_limit_remaining: int | None = None
    last_ok_ms: int | None = None
    last_error: str = ""
    detail: str = ""

    @property
    def healthy(self) -> bool:
        return self.state is ProviderState.ONLINE


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    """One indicator evaluated on one timeframe."""

    identity: str
    name: str
    timeframe: Timeframe
    values: dict[str, Decimal] = field(default_factory=dict)
    updated_ms: int = 0
    direction: SignalDirection = SignalDirection.NEUTRAL
    note: str = ""

    @property
    def primary(self) -> Decimal | None:
        if not self.values:
            return None
        return next(iter(self.values.values()))


@dataclass(frozen=True, slots=True)
class SignalSnapshot:
    identity: str
    strategy: str
    direction: SignalDirection
    confidence: Decimal
    reasons: tuple[str, ...]
    updated_ms: int
    timeframes: tuple[Timeframe, ...] = ()


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Reserved read-only shape. This build never requests account endpoints."""

    identity: str
    equity_usd: Decimal
    freshness: Freshness
    source: str
