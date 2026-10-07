"""Provider contracts.

Providers are read-only. There is deliberately no order, transfer or wallet
method anywhere in this module, so no implementation can accidentally trade.

Anything a provider cannot answer must come back as
:class:`~okx_alert.marketdata.types.Unavailable` inside a ``ProviderResult``.
The ``Null*`` base classes below default every capability to
``NOT_IMPLEMENTED`` so a partial adapter can only ever under-promise.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol, runtime_checkable

from .types import (
    CandleSeries,
    Channel,
    FundingSnapshot,
    IndicatorSnapshot,
    InstrumentRef,
    LiquidityRouteSnapshot,
    OnchainQuoteSnapshot,
    OnchainTokenSnapshot,
    OpenInterestSnapshot,
    OrderBookSnapshot,
    ProviderResult,
    ProviderState,
    ProviderStatus,
    SignalSnapshot,
    TickerSnapshot,
    Timeframe,
    TokenRef,
    TradePrint,
)


@runtime_checkable
class MarketDataProvider(Protocol):
    provider_id: str

    def status(self) -> ProviderStatus: ...


@runtime_checkable
class ExchangeMarketDataProvider(MarketDataProvider, Protocol):
    """Centralized exchange surface: ticker, candles, book, trades, derivatives."""

    def ticker(self, instrument: InstrumentRef) -> ProviderResult[TickerSnapshot]: ...

    def candles(
        self, instrument: InstrumentRef, timeframe: Timeframe, limit: int = 200
    ) -> ProviderResult[CandleSeries]: ...

    def order_book(
        self, instrument: InstrumentRef, depth: int = 20
    ) -> ProviderResult[OrderBookSnapshot]: ...

    def trades(
        self, instrument: InstrumentRef, limit: int = 50
    ) -> ProviderResult[tuple[TradePrint, ...]]: ...

    def funding(self, instrument: InstrumentRef) -> ProviderResult[FundingSnapshot]: ...

    def open_interest(
        self, instrument: InstrumentRef
    ) -> ProviderResult[OpenInterestSnapshot]: ...


@runtime_checkable
class OnchainMarketDataProvider(MarketDataProvider, Protocol):
    """On-chain surface: pool price, read-only swap quote, liquidity route."""

    def token(self, token: TokenRef, quote_symbol: str = "") -> ProviderResult[
        OnchainTokenSnapshot
    ]: ...

    def candles(
        self,
        token: TokenRef,
        quote_symbol: str = "",
        timeframe: Timeframe = Timeframe.M15,
        limit: int = 200,
    ) -> ProviderResult[CandleSeries]: ...

    def quote(
        self, base: TokenRef, quote: TokenRef, amount_in: Decimal
    ) -> ProviderResult[OnchainQuoteSnapshot]: ...

    def liquidity_route(
        self, base: TokenRef, quote: TokenRef, amount_in: Decimal
    ) -> ProviderResult[LiquidityRouteSnapshot]: ...


@runtime_checkable
class IndicatorProvider(Protocol):
    provider_id: str

    def available(self) -> tuple[str, ...]: ...

    def compute(
        self, series: CandleSeries, names: tuple[str, ...] = ()
    ) -> ProviderResult[tuple[IndicatorSnapshot, ...]]: ...


@runtime_checkable
class SignalProvider(Protocol):
    provider_id: str

    def evaluate(
        self, identity: str, indicators: tuple[IndicatorSnapshot, ...], now_ms: int
    ) -> ProviderResult[SignalSnapshot]: ...


class NullExchangeProvider:
    """Every capability unavailable until a subclass overrides it."""

    provider_id = "null-exchange"
    channel: Channel = Channel.REST

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider_id=self.provider_id,
            state=ProviderState.NOT_IMPLEMENTED,
            channel=self.channel,
            detail="no endpoint bound",
        )

    def ticker(self, instrument: InstrumentRef) -> ProviderResult[TickerSnapshot]:
        return ProviderResult.not_implemented(self.provider_id, "ticker")

    def candles(
        self, instrument: InstrumentRef, timeframe: Timeframe, limit: int = 200
    ) -> ProviderResult[CandleSeries]:
        return ProviderResult.not_implemented(self.provider_id, "candles")

    def order_book(
        self, instrument: InstrumentRef, depth: int = 20
    ) -> ProviderResult[OrderBookSnapshot]:
        return ProviderResult.not_implemented(self.provider_id, "order book")

    def trades(
        self, instrument: InstrumentRef, limit: int = 50
    ) -> ProviderResult[tuple[TradePrint, ...]]:
        return ProviderResult.not_implemented(self.provider_id, "trades")

    def funding(self, instrument: InstrumentRef) -> ProviderResult[FundingSnapshot]:
        return ProviderResult.not_implemented(self.provider_id, "funding rate")

    def open_interest(
        self, instrument: InstrumentRef
    ) -> ProviderResult[OpenInterestSnapshot]:
        return ProviderResult.not_implemented(self.provider_id, "open interest")


class NullOnchainProvider:
    provider_id = "null-onchain"
    channel: Channel = Channel.REST

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider_id=self.provider_id,
            state=ProviderState.NOT_IMPLEMENTED,
            channel=self.channel,
            detail="no endpoint bound",
        )

    def token(
        self, token: TokenRef, quote_symbol: str = ""
    ) -> ProviderResult[OnchainTokenSnapshot]:
        return ProviderResult.not_implemented(self.provider_id, "token price")

    def candles(
        self,
        token: TokenRef,
        quote_symbol: str = "",
        timeframe: Timeframe = Timeframe.M15,
        limit: int = 200,
    ) -> ProviderResult[CandleSeries]:
        return ProviderResult.not_implemented(self.provider_id, "pool candles")

    def quote(
        self, base: TokenRef, quote: TokenRef, amount_in: Decimal
    ) -> ProviderResult[OnchainQuoteSnapshot]:
        return ProviderResult.not_implemented(self.provider_id, "swap quote")

    def liquidity_route(
        self, base: TokenRef, quote: TokenRef, amount_in: Decimal
    ) -> ProviderResult[LiquidityRouteSnapshot]:
        return ProviderResult.not_implemented(self.provider_id, "liquidity route")
