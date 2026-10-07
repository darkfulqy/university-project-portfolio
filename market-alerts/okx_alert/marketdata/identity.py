"""Identity normalization.

DexScreener, GeckoTerminal, OKX CEX and OKX DEX each name the same asset
differently. Everything downstream (cache keys, indicator state, event log,
watchlist rows) keys off the single ``identity`` string produced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ..models import AlertRule, Source
from .types import InstrumentRef, MarketKind, TokenRef

QUOTE_ALIASES: dict[str, str] = {"WSOL": "SOL", "WETH": "ETH", "WBTC": "BTC"}


def normalize_symbol(symbol: str) -> str:
    upper = symbol.strip().upper()
    return QUOTE_ALIASES.get(upper, upper)


def normalize_address(address: str) -> str:
    """EVM addresses are case-insensitive; Solana addresses are not."""
    text = address.strip()
    return text.lower() if text.startswith("0x") else text


def instrument_from_id(instrument_id: str, exchange: str = "okx") -> InstrumentRef:
    parts = instrument_id.upper().split("-")
    kind = MarketKind.SPOT
    if len(parts) >= 3 and parts[2] == "SWAP":
        kind = MarketKind.SWAP
    elif len(parts) >= 3:
        kind = MarketKind.FUTURES
    return InstrumentRef(
        exchange=exchange,
        instrument_id=instrument_id.upper(),
        base=parts[0] if parts else "",
        quote=parts[1] if len(parts) > 1 else "",
        kind=kind,
    )


def token_from_rule(rule: AlertRule) -> TokenRef | None:
    address = rule.token_contract_address
    if not address:
        return None
    chain = rule.chain_id or rule.chain_index or ""
    return TokenRef(chain=chain, address=normalize_address(address), symbol=_base_symbol(rule))


def _base_symbol(rule: AlertRule) -> str:
    """Derive a display base symbol from the human rule name, e.g. ``A/B`` -> ``A``."""
    name = rule.name.strip()
    for separator in ("/", "-", " "):
        if separator in name:
            return normalize_symbol(name.split(separator)[0])
    return normalize_symbol(name)


@dataclass(frozen=True, slots=True)
class WatchTarget:
    """One watchlist row: a rule bound to a normalized market identity."""

    rule_id: str
    identity: str
    display: str
    kind: MarketKind
    source: Source
    instrument: InstrumentRef | None = None
    token: TokenRef | None = None
    quote_symbol: str = ""
    components: tuple[str, ...] = ()
    notify: bool = False
    target_price: Decimal | None = None
    direction: str = ""

    @property
    def provider_family(self) -> str:
        if self.kind is MarketKind.ONCHAIN:
            return "onchain"
        if self.kind is MarketKind.DERIVED:
            return "derived"
        return "exchange"


def target_from_rule(rule: AlertRule) -> WatchTarget:
    common = {
        "rule_id": rule.id,
        "source": rule.source,
        "notify": rule.notify,
        "target_price": rule.target_price,
        "direction": rule.direction.value,
    }
    if rule.source is Source.OKX_CEX:
        instrument = instrument_from_id(rule.instrument_id or "")
        return WatchTarget(
            identity=instrument.identity,
            display=instrument.display,
            kind=instrument.kind,
            instrument=instrument,
            quote_symbol=instrument.quote,
            **common,
        )
    if rule.source is Source.DERIVED:
        return WatchTarget(
            identity=f"derived:{rule.id}",
            display=rule.name,
            kind=MarketKind.DERIVED,
            components=rule.multiply_rule_ids,
            **common,
        )
    token = token_from_rule(rule)
    quote_symbol = normalize_symbol(rule.quote_symbol or "")
    identity = token.identity if token else f"onchain:{rule.id}"
    display = f"{token.display}/{quote_symbol}" if token and quote_symbol else rule.name
    return WatchTarget(
        identity=f"{identity}:{quote_symbol}" if quote_symbol else identity,
        display=display,
        kind=MarketKind.ONCHAIN,
        token=token,
        quote_symbol=quote_symbol,
        **common,
    )


def targets_from_rules(rules: tuple[AlertRule, ...]) -> tuple[WatchTarget, ...]:
    return tuple(target_from_rule(rule) for rule in rules)
