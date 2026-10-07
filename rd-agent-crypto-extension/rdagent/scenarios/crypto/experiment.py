"""Crypto 5-minute factor scenario and experiment (docs/crypto_5m/DESIGN.md section 6).

``CryptoFactorExperiment`` reuses ``QlibFactorExperiment`` so that upstream ``process_factor_data`` keeps working; it
adds the crypto evaluation bookkeeping (``crypto_accepted`` / ``crypto_eval``).  ``CryptoFactorScenario`` is the
LLM-visible description of the data, the coding interface and the RankIC evaluator; all its text comes from
``rdagent/scenarios/crypto/prompts.yaml``.
"""

from __future__ import annotations

import platform
import re
from copy import deepcopy
from importlib import metadata
from pathlib import Path
from typing import Any

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario
from rdagent.scenarios.crypto.conf import CRYPTO_EVAL_SETTINGS
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.utils import get_data_folder_intro
from rdagent.utils.agent.tpl import T

BUILD_PANEL_COMMAND = (
    "python -m rdagent.scenarios.crypto.data.build_panel --source-dir <normalized 1m csv.gz dir> "
    "--out-dir <FACTOR_CoSTEER_data_folder> --debug-out-dir <FACTOR_CoSTEER_data_folder_debug>"
)


class CryptoFactorExperiment(QlibFactorExperiment):
    """Factor experiment evaluated by the crypto RankIC runner.

    Attributes
    ----------
    crypto_accepted:
        Names of the sub-task factors that passed the acceptance gate in this experiment (set by the runner).
    crypto_eval:
        The pickle-safe ``ExperimentEval`` dict produced by ``evaluation.evaluate_experiment`` (DESIGN 4.3), or ``None``
        when the experiment has not been evaluated (or failed).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.crypto_accepted: list[str] = []
        self.crypto_eval: dict | None = None


def parse_horizons(value: Any) -> tuple[int, ...]:
    """Normalise ``CryptoEvalSettings.horizons_minutes`` (comma string, list or tuple) to a tuple of ints."""
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        return tuple(int(p) for p in parts if p)
    return tuple(int(v) for v in value)


def _dist_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def crypto_runtime_environment() -> str:
    """Static runtime description (no subprocess, no conda): python + core library versions and execution rules."""
    timeout = getattr(FACTOR_COSTEER_SETTINGS, "file_based_execution_timeout", 3600)
    lines = [
        f"Python {platform.python_version()} ({platform.system()} {platform.machine()})",
        f"pandas {_dist_version('pandas')}, numpy {_dist_version('numpy')}, "
        f"tables (PyTables) {_dist_version('tables')}",
        "CPU only (no GPU, no deep learning frameworks). Available libraries: pandas, numpy, scipy (optional), tables.",
        f"Your code is executed as a subprocess `python factor.py` with a timeout of {timeout} s; "
        "the source data file is symlinked into the working directory.",
    ]
    return "\n".join(lines)


def assert_panel_folders_ready(panel_filename: str | None = None) -> tuple[Path, Path]:
    """Return ``(full_folder, debug_folder)`` after checking both contain the panel file.

    Raises
    ------
    FileNotFoundError
        Naming the missing path and the ``build_panel`` command, so that the upstream Qlib/Docker data generation in
        ``get_data_folder_intro`` is never triggered.
    """
    panel_filename = panel_filename or CRYPTO_EVAL_SETTINGS.panel_filename
    full = Path(FACTOR_COSTEER_SETTINGS.data_folder)
    debug = Path(FACTOR_COSTEER_SETTINGS.data_folder_debug)
    missing = [str(folder / panel_filename) for folder in (full, debug) if not (folder / panel_filename).is_file()]
    if missing:
        raise FileNotFoundError(
            "Crypto panel file missing: "
            + ", ".join(missing)
            + ". Set FACTOR_CoSTEER_data_folder / FACTOR_CoSTEER_data_folder_debug and build the panel first with: "
            + BUILD_PANEL_COMMAND
        )
    return full, debug


class CryptoFactorScenario(Scenario):
    """LLM-facing description of the crypto 5-minute factor task (same property set as ``QlibFactorScenario``)."""

    def __init__(self) -> None:
        super().__init__()
        settings = CRYPTO_EVAL_SETTINGS
        panel_filename = settings.panel_filename
        self._background = deepcopy(
            T(".prompts:crypto_factor_background").r(runtime_environment=self.get_runtime_environment())
        )
        assert_panel_folders_ready(panel_filename)
        # Only the two contract files are described: any stray file (.DS_Store, *.bak, ...) in the debug folder would
        # make the upstream get_file_desc raise NotImplementedError.
        self._source_data = deepcopy(get_data_folder_intro(fname_reg=rf"^({re.escape(panel_filename)}|README\.md)$"))
        self._output_format = deepcopy(T(".prompts:crypto_factor_output_format").r())
        self._interface = deepcopy(T(".prompts:crypto_factor_interface").r(panel_filename=panel_filename))
        self._simulator = deepcopy(T(".prompts:crypto_factor_simulator").r())
        self._rich_style_description = deepcopy(T(".prompts:crypto_factor_rich_style_description").r())
        horizons = parse_horizons(settings.horizons_minutes)
        self._experiment_setting = deepcopy(
            T(".prompts:crypto_factor_experiment_setting").r(
                panel_filename=panel_filename,
                bar_minutes=settings.bar_minutes,
                n_instruments_text="Binance USDT spot pairs (about 43 instruments)",
                horizons_text=", ".join(f"{h}m" for h in horizons),
                primary_horizon_minutes=settings.primary_horizon_minutes,
                train_start=settings.train_start,
                train_end=settings.train_end,
                valid_start=settings.valid_start,
                valid_end=settings.valid_end,
                test_start=settings.test_start,
                test_end=settings.test_end,
                purge_minutes=settings.purge_minutes,
                min_assets=settings.min_assets,
                n_quantiles=settings.n_quantiles,
                fee_bp_per_side=settings.fee_bp_per_side,
                slippage_bp_per_side=settings.slippage_bp_per_side,
                gate_min_tstat=settings.gate_min_tstat,
                gate_min_abs_ic=settings.gate_min_abs_ic,
                gate_min_coverage=settings.gate_min_coverage,
                gate_max_corr=settings.gate_max_corr,
                gate_min_ir_gain=settings.gate_min_ir_gain,
                gate_max_abs_ic=settings.gate_max_abs_ic,
            )
        )

    @property
    def background(self) -> str:
        return self._background

    def get_source_data_desc(self, task: Task | None = None) -> str:  # noqa: ARG002
        return self._source_data

    @property
    def output_format(self) -> str:
        return self._output_format

    @property
    def interface(self) -> str:
        return self._interface

    @property
    def simulator(self) -> str:
        return self._simulator

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description

    @property
    def experiment_setting(self) -> str:
        return self._experiment_setting

    def get_scenario_all_desc(
        self, task: Task | None = None, filtered_tag: str | None = None, simple_background: bool | None = None
    ) -> str:  # noqa: ARG002
        """A static scenario describer (same layout as the upstream factor scenario, plus the evaluation setting)."""
        if simple_background:
            return f"""Background of the scenario:
{self.background}"""
        return f"""Background of the scenario:
{self.background}
The source data you can use:
{self.get_source_data_desc(task)}
The interface you should follow to write the runnable code:
{self.interface}
The output of your code should be in the format:
{self.output_format}
The evaluator used to score your factor:
{self.simulator}
The experiment setting:
{self.experiment_setting}
"""

    def get_runtime_environment(self) -> str:
        return crypto_runtime_environment()
