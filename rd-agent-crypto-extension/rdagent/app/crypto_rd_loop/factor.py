"""
Crypto 5m factor workflow with session control (mirrors rdagent/app/qlib_rd_loop/factor.py).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional

import fire

from rdagent.app.crypto_rd_loop.conf import CRYPTO_PROP_SETTING
from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
from rdagent.log import rdagent_logger as logger


class CryptoFactorRDLoop(FactorRDLoop):
    """Same step machinery as ``FactorRDLoop``; only the component classes differ (``CRYPTO_PROP_SETTING``)."""


def _maybe_install_local_embedding() -> None:
    if os.environ.get("CRYPTO_LOCAL_EMBEDDING") == "1":
        from rdagent.scenarios.crypto import local_embedding

        local_embedding.install()
        logger.info("CRYPTO_LOCAL_EMBEDDING=1: using hashed-token local embedding for CoSTEER knowledge retrieval")


def main(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: str | None = None,
    checkout: bool = True,
    checkout_path: Optional[str] = None,
    base_features_path: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Auto R&D evolving loop for crypto 5m/15m cross-sectional factors.

    Continue a session with

    .. code-block:: bash

        dotenv run -- python rdagent/app/crypto_rd_loop/factor.py $LOG_PATH/__session__/1/0_propose --step_n 1

    """
    if checkout_path is not None:
        checkout = Path(checkout_path)

    _maybe_install_local_embedding()

    if path is None:
        factor_loop = CryptoFactorRDLoop(CRYPTO_PROP_SETTING)
    else:
        factor_loop = CryptoFactorRDLoop.load(path, checkout=checkout)

    factor_loop._init_base_features(base_features_path)
    if "user_interaction_queues" in kwargs and kwargs["user_interaction_queues"] is not None:
        factor_loop._set_interactor(*kwargs["user_interaction_queues"])
        factor_loop._interact_init_params()
    asyncio.run(factor_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
