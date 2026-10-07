"""Indicator plugin registry.

Third-party or project-local indicators register themselves with
:func:`register` and immediately become available to the engine, the dashboard
Indicators tab and the signal provider — no changes to the TUI required.

    from okx_alert.indicators.registry import register, IndicatorOutput

    @register("my_zscore", min_candles=50, description="Close z-score")
    def _zscore(series):
        ...
        return IndicatorOutput(values={"z": value})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable

from ..marketdata.types import CandleSeries, SignalDirection


@dataclass(frozen=True, slots=True)
class IndicatorOutput:
    values: dict[str, Decimal] = field(default_factory=dict)
    direction: SignalDirection = SignalDirection.NEUTRAL
    note: str = ""


ComputeFn = Callable[[CandleSeries], IndicatorOutput]


@dataclass(frozen=True, slots=True)
class IndicatorSpec:
    name: str
    compute: ComputeFn
    min_candles: int
    description: str
    category: str


_REGISTRY: dict[str, IndicatorSpec] = {}


def register(
    name: str,
    *,
    min_candles: int = 1,
    description: str = "",
    category: str = "custom",
    overwrite: bool = False,
) -> Callable[[ComputeFn], ComputeFn]:
    def decorator(function: ComputeFn) -> ComputeFn:
        if name in _REGISTRY and not overwrite:
            raise ValueError(f"indicator '{name}' is already registered")
        _REGISTRY[name] = IndicatorSpec(
            name=name,
            compute=function,
            min_candles=min_candles,
            description=description,
            category=category,
        )
        return function

    return decorator


def unregister(name: str) -> bool:
    return _REGISTRY.pop(name, None) is not None


def get(name: str) -> IndicatorSpec | None:
    return _REGISTRY.get(name)


def registered() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def specs() -> tuple[IndicatorSpec, ...]:
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))
