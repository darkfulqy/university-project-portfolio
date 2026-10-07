from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class Source(StrEnum):
    OKX_DEX = "okx_dex"
    OKX_CEX = "okx_cex"
    DEXSCREENER = "dexscreener"
    GECKOTERMINAL = "geckoterminal"
    DERIVED = "derived"


class Direction(StrEnum):
    ABOVE = "above"
    BELOW = "below"


def normalize_contract(address: str) -> str:
    """EVM addresses are case-insensitive; non-EVM addresses are not."""
    return address.lower() if address.startswith("0x") else address


def decimal_text(value: Decimal) -> str:
    """Render prices without scientific notation."""
    return format(value, "f")


@dataclass(frozen=True, slots=True)
class AlertRule:
    id: str
    name: str
    source: Source
    direction: Direction
    target_price: Decimal | None
    enabled: bool
    notify: bool = True
    chain_index: str | None = None
    chain_id: str | None = None
    token_contract_address: str | None = None
    instrument_id: str | None = None
    quote_symbol: str | None = None
    pair_address: str | None = None
    multiply_rule_ids: tuple[str, ...] = ()

    @property
    def market_key(self) -> tuple[str, ...]:
        if self.source is Source.OKX_DEX:
            return (
                self.source.value,
                self.chain_index or "",
                normalize_contract(self.token_contract_address or ""),
            )
        if self.source in (Source.DEXSCREENER, Source.GECKOTERMINAL):
            return (
                self.source.value,
                self.chain_id or "",
                normalize_contract(self.token_contract_address or ""),
                (self.quote_symbol or "").upper(),
                self.pair_address or "",
            )
        if self.source is Source.DERIVED:
            return (self.source.value, *self.multiply_rule_ids)
        return (self.source.value, self.instrument_id or "")

    def condition_met(self, price: Decimal) -> bool:
        if self.target_price is None:
            return False
        if self.direction is Direction.ABOVE:
            return price >= self.target_price
        return price <= self.target_price


@dataclass(frozen=True, slots=True)
class AppConfig:
    poll_interval_seconds: float
    configuration_retry_seconds: float
    request_timeout_seconds: float
    max_price_age_seconds: int
    notification_sound: str
    alerts: tuple[AlertRule, ...]

    @property
    def enabled_alerts(self) -> tuple[AlertRule, ...]:
        return tuple(rule for rule in self.alerts if rule.enabled)


@dataclass(frozen=True, slots=True)
class DexCredentials:
    api_key: str
    secret_key: str
    passphrase: str

    @property
    def complete(self) -> bool:
        return bool(self.api_key and self.secret_key and self.passphrase)


@dataclass(frozen=True, slots=True)
class PricePoint:
    price: Decimal
    timestamp_ms: int
    source: Source
    pair_address: str | None = None
    liquidity_usd: Decimal | None = None

    def freshness_error(self, now_ms: int, max_age_seconds: int) -> str | None:
        age_ms = now_ms - self.timestamp_ms
        if age_ms < -30_000:
            return "price timestamp is more than 30 seconds in the future"
        if age_ms > max_age_seconds * 1_000:
            return f"price is stale by {age_ms // 1_000} seconds"
        return None


@dataclass(frozen=True, slots=True)
class Candle:
    timestamp_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    volume_usd: Decimal
    confirmed: bool


@dataclass(frozen=True, slots=True)
class FundingRate:
    """One row of ``/api/v5/public/funding-rate``.

    ``funding_rate`` is the venue's running estimate for the *upcoming*
    settlement at ``funding_time_ms``. The settlement interval is dynamic on
    OKX (8h contracts escalate to 4h/2h/1h while the rate pegs its clamp), so
    it must always be derived from the timestamps, never assumed.
    """

    instrument_id: str
    funding_rate: Decimal
    funding_time_ms: int
    next_funding_time_ms: int
    prev_funding_time_ms: int | None = None
    min_rate: Decimal | None = None
    max_rate: Decimal | None = None
    premium: Decimal | None = None
    interest_rate: Decimal | None = None
    settled_rate: Decimal | None = None
    settlement_state: str = ""
    formula_type: str = ""
    timestamp_ms: int = 0

    @property
    def interval_hours(self) -> int | None:
        """Settlement interval derived from the venue's own timestamps."""
        span_ms = self.next_funding_time_ms - self.funding_time_ms
        if span_ms <= 0 or span_ms % 3_600_000:
            return None
        return span_ms // 3_600_000

    @property
    def capped(self) -> bool:
        """True when the rate sits at ≥95% of its clamp: realized carry is
        below the quote and OKX may escalate the settlement frequency."""
        if self.max_rate is not None and self.max_rate > 0:
            if self.funding_rate >= self.max_rate * Decimal("0.95"):
                return True
        if self.min_rate is not None and self.min_rate < 0:
            if self.funding_rate <= self.min_rate * Decimal("0.95"):
                return True
        return False


@dataclass(frozen=True, slots=True)
class FundingSettlement:
    """One row of ``/api/v5/public/funding-rate-history`` (a settled period)."""

    instrument_id: str
    funding_time_ms: int
    funding_rate: Decimal
    realized_rate: Decimal | None = None
    formula_type: str = ""


@dataclass(frozen=True, slots=True)
class OpenInterest:
    instrument_id: str
    contracts: Decimal
    base_ccy: Decimal | None
    usd: Decimal | None
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class Liquidation:
    """One forced close: a position of ``size`` contracts on ``position_side``
    wiped out at ``price``.

    ``position_side`` is the half of the perpetual that was killed — the one
    observable that separates the long and the short sitting in the same open
    interest. ``price`` is the venue's bankruptcy price, not the fill.
    """

    instrument_id: str
    timestamp_ms: int
    position_side: str
    price: Decimal
    size: Decimal
    loss: Decimal | None = None


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Sizing and lifecycle fields of ``/api/v5/public/instruments``.

    ``min_sz`` counts contracts for derivatives but base currency for spot;
    ``ct_val`` (face value per contract) converts between the two.
    """

    instrument_id: str
    instrument_type: str
    state: str
    base_ccy: str = ""
    quote_ccy: str = ""
    settle_ccy: str = ""
    ct_val: Decimal | None = None
    ct_val_ccy: str = ""
    ct_type: str = ""
    min_sz: Decimal | None = None
    lot_sz: Decimal | None = None
    tick_sz: Decimal | None = None
    list_time_ms: int | None = None


# The two OKX announcement feed types the carry pipeline consumes. They share
# one storage table, so consumers must filter by type: a listing headline
# naming a coin must never satisfy the scanner's delist gate.
ANN_TYPE_DELISTINGS = "announcements-delistings"
ANN_TYPE_NEW_LISTINGS = "announcements-new-listings"


@dataclass(frozen=True, slots=True)
class Announcement:
    announcement_type: str
    title: str
    url: str
    published_ms: int


@dataclass(frozen=True, slots=True)
class CexTicker:
    """A full OKX v5 ticker row. Only ``last`` and ``timestamp_ms`` are required;
    every 24h statistic stays ``None`` when the venue omits or blanks it."""

    instrument_id: str
    price: Decimal
    timestamp_ms: int
    bid: Decimal | None = None
    ask: Decimal | None = None
    open_24h: Decimal | None = None
    high_24h: Decimal | None = None
    low_24h: Decimal | None = None
    volume_24h_base: Decimal | None = None
    volume_24h_quote: Decimal | None = None

    @property
    def price_point(self) -> "PricePoint":
        return PricePoint(
            price=self.price, timestamp_ms=self.timestamp_ms, source=Source.OKX_CEX
        )
