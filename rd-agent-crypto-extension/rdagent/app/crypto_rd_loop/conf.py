"""Loop settings for the crypto 5m factor scenario (env prefix ``CRYPTO_LOOP_``)."""

from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from rdagent.app.qlib_rd_loop.conf import FactorBasePropSetting


class CryptoFactorPropSetting(FactorBasePropSetting):
    """Component class paths for the crypto factor loop. Override any of them with ``CRYPTO_LOOP_<field>``."""

    model_config = SettingsConfigDict(env_prefix="CRYPTO_LOOP_", protected_namespaces=())

    scen: str = "rdagent.scenarios.crypto.experiment.CryptoFactorScenario"
    """Scenario class for crypto 5m factors"""

    hypothesis_gen: str = "rdagent.scenarios.crypto.proposal.CryptoFactorHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.crypto.proposal.CryptoFactorHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"
    """Coder class (upstream CoSTEER factor coder, unchanged)"""

    runner: str = "rdagent.scenarios.crypto.runner.CryptoRankICRunner"
    """Runner class (RankIC evaluation, no Qlib backtest)"""

    summarizer: str = "rdagent.scenarios.crypto.feedback.CryptoFactorExperiment2Feedback"
    """Summarizer class (deterministic gate decision + LLM commentary)"""


CRYPTO_PROP_SETTING = CryptoFactorPropSetting()
