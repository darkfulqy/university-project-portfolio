"""Settings for the crypto 5m factor evaluation (env prefix ``CRYPTO_EVAL_``, DESIGN.md section 10)."""

from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings
from rdagent.scenarios.crypto.evaluation import EvalConfig


class CryptoEvalSettings(ExtendedBaseSettings):
    """Evaluation / gate settings. Every field can be overridden with ``CRYPTO_EVAL_<FIELD>``."""

    model_config = SettingsConfigDict(env_prefix="CRYPTO_EVAL_")

    panel_filename: str = "crypto_5m.h5"
    """File name of the panel inside FACTOR_CoSTEER_data_folder / data_folder_debug"""

    bar_minutes: int = 5
    horizons_minutes: str = "5,15"
    """Comma-separated forward-return horizons in minutes (each a multiple of bar_minutes)"""
    primary_horizon_minutes: int = 15

    train_start: str = "2026-02-25"
    train_end: str = "2026-06-30"
    valid_start: str = "2026-07-01"
    valid_end: str = "2026-07-31"
    test_start: str = "2026-08-01"
    test_end: str = "2026-08-23 23:59:00"
    """Inclusive tz-naive UTC bounds; a date-only end bound covers the whole day"""

    purge_minutes: int = 15
    min_assets: int = 30
    n_quantiles: int = 5
    fee_bp_per_side: float = 10.0
    slippage_bp_per_side: float = 5.0

    gate_min_tstat: float = 3.0
    """Minimum |Newey-West ic_tstat| (and |ic_tstat_dynamic|) on valid at the primary horizon (gate R1)"""
    gate_min_abs_ic: float = 0.01
    gate_min_coverage: float = 0.5
    gate_max_corr: float = 0.7
    gate_min_ir_gain: float = 0.0
    gate_max_abs_ic: float = 0.25
    """Implausibility cap on valid |ic_mean| at the primary horizon: above it the factor is rejected (gate R0)"""

    ledger_path: str = "./crypto_factor_ledger.jsonl"
    """JSONL ledger appended by the runner after every evaluation"""

    def horizons(self) -> tuple[int, ...]:
        """Parse ``horizons_minutes`` (``"5,15"``) into a tuple of ints."""
        parts = [p.strip() for p in str(self.horizons_minutes).split(",")]
        horizons = tuple(int(p) for p in parts if p)
        if not horizons:
            raise ValueError("CRYPTO_EVAL_HORIZONS_MINUTES must list at least one horizon, e.g. '5,15'")
        return horizons

    def to_eval_config(self) -> EvalConfig:
        """Build the frozen :class:`EvalConfig` consumed by ``evaluation.py``."""
        horizons = self.horizons()
        if self.primary_horizon_minutes not in horizons:
            raise ValueError(
                f"primary_horizon_minutes={self.primary_horizon_minutes} not in horizons_minutes={list(horizons)}"
            )
        return EvalConfig(
            bar_minutes=int(self.bar_minutes),
            horizons_minutes=horizons,
            primary_horizon_minutes=int(self.primary_horizon_minutes),
            train=(self.train_start, self.train_end),
            valid=(self.valid_start, self.valid_end),
            test=(self.test_start, self.test_end),
            purge_minutes=int(self.purge_minutes),
            min_assets=int(self.min_assets),
            n_quantiles=int(self.n_quantiles),
            fee_bp_per_side=float(self.fee_bp_per_side),
            slippage_bp_per_side=float(self.slippage_bp_per_side),
            gate_min_tstat=float(self.gate_min_tstat),
            gate_min_abs_ic=float(self.gate_min_abs_ic),
            gate_min_coverage=float(self.gate_min_coverage),
            gate_max_corr=float(self.gate_max_corr),
            gate_min_ir_gain=float(self.gate_min_ir_gain),
            gate_max_abs_ic=float(self.gate_max_abs_ic),
        )


CRYPTO_EVAL_SETTINGS = CryptoEvalSettings()
