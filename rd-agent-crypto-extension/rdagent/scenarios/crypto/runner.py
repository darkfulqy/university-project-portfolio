"""Runner for the crypto 5m factor scenario: cross-sectional RankIC evaluation (DESIGN.md section 5).

The runner replaces the upstream Qlib/LightGBM backtest. It executes the accepted factor code of the
new experiment (and of the accepted library experiments), reads ``$close`` from the full panel and calls
:func:`rdagent.scenarios.crypto.evaluation.evaluate_experiment`. No Docker, no Qlib.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.runner import CachedRunner
from rdagent.core.exception import FactorEmptyError
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.crypto.conf import CRYPTO_EVAL_SETTINGS
from rdagent.scenarios.crypto.evaluation import EvalConfig, evaluate_experiment, summary_text
from rdagent.scenarios.crypto.experiment import CryptoFactorExperiment
from rdagent.scenarios.crypto.ledger import append_record, record_from_eval
from rdagent.scenarios.qlib.developer.utils import process_factor_data

SEGMENTS: tuple[str, ...] = ("train", "valid", "test")
METRIC_NAMES: tuple[str, ...] = (
    "n_ts",
    "ic_mean",
    "ic_std",
    "ic_ir",
    "ic_tstat",
    "ic_tstat_naive",
    "ic_tstat_dynamic",
    "ic_pos_rate",
    "coverage",
    "rank_autocorr_1",
    "q_spread_bp_gross",
    "q_spread_bp_net",
)
N_LIBRARY_KEY = "n_library_factors"
LOOKAHEAD_CUT_DAYS = 2
"""Days dropped from the panel end by the look-ahead re-run; every value up to the cut is compared."""
LOOKAHEAD_REASON = "R0 look-ahead: values change when future rows are removed"


def _eval_config() -> EvalConfig:
    """Build the EvalConfig from the (possibly monkeypatched) module settings at call time."""
    return CRYPTO_EVAL_SETTINGS.to_eval_config()


def _horizon_keys(cfg: EvalConfig) -> list[str]:
    return [f"{h}m" for h in cfg.horizons_minutes]


def standard_result_keys(cfg: EvalConfig | None = None) -> list[str]:
    """Return the ordered list of keys of ``exp.result``: ``"{segment}.{h}.{metric}"`` plus ``n_library_factors``."""
    cfg = cfg or _eval_config()
    keys = [f"{seg}.{h}.{m}" for seg in SEGMENTS for h in _horizon_keys(cfg) for m in METRIC_NAMES]
    keys.append(N_LIBRARY_KEY)
    return keys


def baseline_result(cfg: EvalConfig | None = None) -> pd.Series:
    """Result Series of the empty-library baseline: every standard key NaN and ``n_library_factors=0``."""
    keys = standard_result_keys(cfg)
    res = pd.Series(np.nan, index=keys, dtype=float)
    res[N_LIBRARY_KEY] = 0.0
    return res


def baseline_eval() -> dict[str, Any]:
    """Minimal ExperimentEval dict assigned to the empty baseline experiment."""
    return {
        "cfg": None,
        "library_before": [],
        "library_after": [],
        "composite_before": None,
        "composite_after": None,
        "factors": {},
        "accepted": [],
        "summary_text": "baseline: empty factor library (no evaluation performed)",
    }


def result_series(ev: dict[str, Any], cfg: EvalConfig | None = None) -> pd.Series:
    """Flatten ``ev["composite_after"]`` into a float Series keyed ``"{segment}.{h}.{metric}"``.

    Missing entries stay NaN so the key set is identical to :func:`baseline_result`. ``n_library_factors``
    is ``len(ev["library_after"])``.
    """
    res = baseline_result(cfg)
    composite = ev.get("composite_after") or {}
    for key in res.index:
        if key == N_LIBRARY_KEY:
            continue
        seg, h, metric = key.split(".", 2)
        val = composite.get(seg, {}).get(h, {}).get(metric)
        if val is not None:
            try:
                res[key] = float(val)
            except (TypeError, ValueError):
                res[key] = np.nan
    res[N_LIBRARY_KEY] = float(len(ev.get("library_after") or []))
    return res


def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated(keep="last")]


def _dedup_rows(df: pd.DataFrame, what: str) -> pd.DataFrame:
    """Drop duplicated ``(datetime, instrument)`` rows (keep last); ``evaluate_experiment`` needs a unique index."""
    if df.index.has_duplicates:
        n = int(df.index.duplicated(keep="last").sum())
        logger.warning(f"Crypto runner: {what} has {n} duplicated (datetime, instrument) rows; keeping last")
        df = df[~df.index.duplicated(keep="last")]
    return df


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Make the ``datetime`` level tz-naive UTC ``datetime64[ns]`` so factor output aligns with the panel grid."""
    idx = df.index
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels != 2:
        return df
    level = idx.get_level_values(0)
    tz_aware = getattr(level, "tz", None) is not None
    if not tz_aware and level.dtype == "datetime64[ns]" and list(idx.names) == ["datetime", "instrument"]:
        return df
    dt = pd.DatetimeIndex(level)
    if dt.tz is not None:
        dt = dt.tz_convert("UTC").tz_localize(None)
    df = df.copy()
    df.index = pd.MultiIndex.from_arrays(
        [dt.astype("datetime64[ns]"), idx.get_level_values(1)], names=["datetime", "instrument"]
    )
    return df


def _on_panel_grid(index: pd.Index, panel_index: pd.Index, n_samples: int = 64) -> bool:
    """Cheap alignment probe: does any of ``n_samples`` evenly spaced keys of ``index`` exist in ``panel_index``?"""
    n = len(index)
    if n == 0:
        return False
    for pos in np.unique(np.linspace(0, n - 1, num=min(n, n_samples)).astype(int)):
        try:
            if index[pos] in panel_index:
                return True
        except (TypeError, KeyError, ValueError):
            continue
    return False


def _factor_code_by_name(exp: CryptoFactorExperiment) -> dict[str, str]:
    """``factor_name -> factor.py source`` for every sub-workspace that holds code."""
    out: dict[str, str] = {}
    for i, task in enumerate(exp.sub_tasks):
        name = getattr(task, "factor_name", None)
        ws = exp.sub_workspace_list[i] if i < len(exp.sub_workspace_list) else None
        code = (getattr(ws, "file_dict", None) or {}).get("factor.py") if ws is not None else None
        if name and isinstance(code, str) and code.strip():
            out[name] = code
    return out


def run_factor_code(code: str, data_dir: Path, workdir: Path, python_bin: str, timeout: float) -> pd.DataFrame:
    """Run ``factor.py`` in ``workdir`` with every file of ``data_dir`` linked in (like ``FactorFBWorkspace.execute``).

    Returns the ``result.h5`` frame.  Raises ``RuntimeError`` on a non-zero exit, a timeout or a missing/unreadable
    output file.  No pickle cache is involved.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    for f in data_dir.iterdir():
        target = workdir / f.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(f.resolve())
    (workdir / "factor.py").write_text(code)
    try:
        proc = subprocess.run(
            [python_bin, "factor.py"], cwd=workdir, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"timeout after {timeout} s") from e
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-5:]
        raise RuntimeError(f"exit code {proc.returncode}: " + " | ".join(tail)[:800])
    out = workdir / "result.h5"
    if not out.is_file():
        raise RuntimeError("result.h5 not written")
    try:
        df = pd.read_hdf(out)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"result.h5 unreadable: {e}"[:800]) from e
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df


def _task_by_name(exp: CryptoFactorExperiment) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, task in enumerate(exp.sub_tasks):
        name = getattr(task, "factor_name", None)
        if name is None:
            continue
        ws = exp.sub_workspace_list[i] if i < len(exp.sub_workspace_list) else None
        out[name] = {
            "description": getattr(task, "factor_description", ""),
            "formulation": getattr(task, "factor_formulation", ""),
            "workspace_path": str(ws.workspace_path) if ws is not None and getattr(ws, "workspace_path", None) else "",
        }
    return out


class CryptoRankICRunner(CachedRunner[CryptoFactorExperiment]):
    """Evaluate the new factors of an experiment by cross-sectional RankIC against forward returns.

    No pickle cache decorator on ``develop``: the evaluation is cheap and the upstream cache key ignores
    the data path. Factor code execution itself is cached by ``FactorFBWorkspace.execute``.
    """

    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        self._library_cache: dict[int, pd.DataFrame] = {}
        self._close_cache: tuple[str, float, pd.Series] | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Drop the process-lifetime caches: ``LoopBase.dump()`` pickles the runner after every step, the ``id()``
        keys of ``_library_cache`` are meaningless in another process and the full-panel frames bloat every dump."""
        state = self.__dict__.copy()
        state["_library_cache"] = {}
        state["_close_cache"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("_library_cache", {})
        self.__dict__.setdefault("_close_cache", None)

    # ------------------------------------------------------------------ data
    def _panel_path(self) -> Path:
        return Path(FACTOR_COSTEER_SETTINGS.data_folder).expanduser() / CRYPTO_EVAL_SETTINGS.panel_filename

    def load_close(self) -> pd.Series:
        """Read ``$close`` from the full panel (cached per path + mtime for the process lifetime)."""
        path = self._panel_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Panel file {path} not found. Build it with "
                "`python -m rdagent.scenarios.crypto.data.build_panel --source-dir <normalized dir> "
                "--out-dir <full folder> --debug-out-dir <debug folder>`."
            )
        mtime = path.stat().st_mtime
        if self._close_cache is not None and self._close_cache[0] == str(path) and self._close_cache[1] == mtime:
            return self._close_cache[2]
        panel = pd.read_hdf(path, key="data")
        close = panel["$close"].astype(float)
        close.name = "$close"
        self._close_cache = (str(path), mtime, close)
        return close

    def new_factor_frame(self, exp: CryptoFactorExperiment) -> pd.DataFrame:
        """Execute the new experiment's factor code (full data) and keep only sub-task columns."""
        logger.info("Crypto runner: new factor processing ...")
        new_df = process_factor_data(exp)
        if new_df is None or new_df.empty:
            raise FactorEmptyError("Factors failed to run on the full sample, this round of experiment failed.")
        wanted = [t.factor_name for t in exp.sub_tasks if getattr(t, "factor_name", None)]
        produced = [str(c) for c in new_df.columns]
        cols = [c for c in new_df.columns if c in wanted]
        new_df = _dedup_columns(new_df.loc[:, cols]) if cols else new_df.iloc[:, :0]
        new_df = new_df.dropna(axis=1, how="all")
        if new_df.shape[1] == 0:
            raise FactorEmptyError(
                "No usable factor column matched a sub-task factor_name "
                f"(got columns={produced}, expected one of {wanted}); the result.h5 column must be named exactly "
                "like the factor."
            )
        new_df = _normalize_index(new_df.astype(float)).sort_index()
        return _dedup_rows(new_df, "new factor frame")

    def library_frame(self, exp: CryptoFactorExperiment) -> pd.DataFrame:
        """Concatenate the accepted columns of every based experiment with a non-empty ``crypto_accepted``."""
        frames: list[pd.DataFrame] = []
        for based_exp in exp.based_experiments:
            accepted = getattr(based_exp, "crypto_accepted", None)
            if not accepted:
                continue
            key = id(based_exp)
            df = self._library_cache.get(key)
            if df is None:
                try:
                    full = process_factor_data(based_exp)
                except FactorEmptyError as e:
                    logger.warning(f"Crypto runner: library experiment produced no data, skipped: {e}")
                    continue
                cols = [c for c in full.columns if c in accepted]
                df = _normalize_index(_dedup_columns(full.loc[:, cols]).astype(float)).sort_index()
                df = _dedup_rows(df, "library frame")
                self._library_cache[key] = df
            if df.shape[1]:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        lib = pd.concat(frames, axis=1)
        return _dedup_columns(lib)

    # ------------------------------------------------------------------ look-ahead check
    def lookahead_check(self, exp: CryptoFactorExperiment, new_df: pd.DataFrame) -> dict[str, str]:
        """Re-run every new factor's ``factor.py`` on a truncated panel and flag columns whose past values change.

        The panel is cut ``LOOKAHEAD_CUT_DAYS`` days before its end and written to a temporary data folder; each
        factor is executed there (fresh subprocess, no pickle cache) and its values at every datetime up to and
        including the cut are compared with ``new_df``.  A causal factor (DESIGN 3.3: the value at ``t`` only uses
        rows with ``datetime <= t``) is identical there by construction; any change means the code looked at rows
        after ``t`` (a ``shift(-1)`` differs at the last truncated bar, a full-sample normalisation differs
        everywhere).  Returns
        ``{factor_name: gate reason}`` for the flagged columns; factors without code (no workspace) are skipped.
        Execution failures on the truncated panel are flagged too (fail closed).
        """
        codes = _factor_code_by_name(exp)
        names = [str(c) for c in new_df.columns if str(c) in codes]
        if not names:
            logger.warning("Crypto runner: look-ahead check skipped (no factor.py source for the new columns)")
            return {}
        panel_path = self._panel_path()
        panel = pd.read_hdf(panel_path, key="data")
        dt = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
        t_cut = dt.max() - pd.Timedelta(days=LOOKAHEAD_CUT_DAYS)
        if t_cut <= dt.min():
            logger.warning("Crypto runner: look-ahead check skipped (panel shorter than the truncation window)")
            return {}
        flagged: dict[str, str] = {}
        python_bin = str(FACTOR_COSTEER_SETTINGS.python_bin)
        timeout = float(getattr(FACTOR_COSTEER_SETTINGS, "file_based_execution_timeout", 3600))
        with tempfile.TemporaryDirectory(prefix="crypto_lookahead_") as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            panel[dt <= t_cut].to_hdf(data_dir / panel_path.name, key="data", mode="w", format="fixed")
            readme = panel_path.parent / "README.md"
            if readme.is_file():
                shutil.copy(readme, data_dir / "README.md")
            del panel
            for i, name in enumerate(names):
                try:
                    trunc = run_factor_code(codes[name], data_dir, Path(tmp) / f"ws_{i}", python_bin, timeout)
                    trunc = _dedup_rows(_normalize_index(trunc), f"truncated output of {name}")
                    col = name if name in trunc.columns else (trunc.columns[0] if trunc.shape[1] == 1 else None)
                    if col is None:
                        raise RuntimeError(f"column {name!r} missing in truncated output {list(trunc.columns)}")
                    full_col = new_df[name]
                    rows = full_col.index[pd.DatetimeIndex(full_col.index.get_level_values("datetime")) <= t_cut]
                    a = full_col.loc[rows].to_numpy(dtype=float)
                    b = trunc[col].reindex(rows).to_numpy(dtype=float)
                except Exception as e:  # noqa: BLE001 - fail closed: an unverifiable factor is not accepted
                    flagged[name] = f"R0 look-ahead check: factor code failed on the truncated panel ({e})"[:600]
                    logger.warning(f"Crypto runner: {name}: {flagged[name]}")
                    continue
                same = np.isclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)
                if not same.all():
                    n_bad = int((~same).sum())
                    flagged[name] = f"{LOOKAHEAD_REASON} ({n_bad} of {len(a)} values differ up to {t_cut})"
                    logger.warning(f"Crypto runner: {name}: {flagged[name]}")
                else:
                    logger.info(f"Crypto runner: {name}: look-ahead check passed ({len(a)} values unchanged)")
        return flagged

    @staticmethod
    def _rejected_entry(reason: str) -> dict[str, Any]:
        return {
            "metrics": {},
            "sign": 1,
            "max_corr": 0.0,
            "max_corr_with": None,
            "gate_passed": False,
            "gate_reasons": [reason],
            "accepted": False,
            "composite_before": None,
            "composite_after": None,
        }

    # ------------------------------------------------------------------ outputs
    @staticmethod
    def write_workspace_files(exp: CryptoFactorExperiment, ev: dict[str, Any]) -> None:
        """Write ``crypto_eval.json`` and ``factor_metrics.csv`` into the experiment workspace."""
        ws_path = Path(exp.experiment_workspace.workspace_path)
        ws_path.mkdir(parents=True, exist_ok=True)
        (ws_path / "crypto_eval.json").write_text(json.dumps(ev, default=str, indent=2, ensure_ascii=False))
        fieldnames = ["factor", "segment", "horizon", *METRIC_NAMES, "sign", "max_corr", "gate_passed", "accepted"]
        with (ws_path / "factor_metrics.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for name, info in (ev.get("factors") or {}).items():
                metrics = info.get("metrics") or {}
                for seg, by_h in metrics.items():
                    for h, m in (by_h or {}).items():
                        row = {"factor": name, "segment": seg, "horizon": h}
                        row.update({k: (m or {}).get(k) for k in METRIC_NAMES})
                        row.update(
                            {
                                "sign": info.get("sign"),
                                "max_corr": info.get("max_corr"),
                                "gate_passed": info.get("gate_passed"),
                                "accepted": info.get("accepted"),
                            }
                        )
                        writer.writerow(row)

    @staticmethod
    def ledger_record(exp: CryptoFactorExperiment, ev: dict[str, Any]) -> dict[str, Any]:
        """Build the JSONL ledger record (DESIGN.md section 9) via ``ledger.record_from_eval``."""
        hyp = exp.hypothesis
        return record_from_eval(
            ev,
            hypothesis=getattr(hyp, "hypothesis", "") if hyp is not None else "",
            reason=getattr(hyp, "reason", "") if hyp is not None else "",
            factor_info=_task_by_name(exp),
        )

    # ------------------------------------------------------------------ main entry
    def develop(self, exp: CryptoFactorExperiment) -> CryptoFactorExperiment:
        """Evaluate ``exp``; assign ``result``, ``crypto_eval``, ``crypto_accepted``, ``stdout``; write files."""
        cfg = _eval_config()

        # Baseline: the empty-library placeholder gets a NaN result instead of a recursive develop().
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            base = exp.based_experiments[-1]
            logger.info("Crypto runner: assigning empty baseline result ...")
            base.result = baseline_result(cfg)
            base.crypto_eval = baseline_eval()
            if not hasattr(base, "crypto_accepted"):
                base.crypto_accepted = []

        new_df = self.new_factor_frame(exp)
        library = self.library_frame(exp)
        close = self.load_close()
        if not _on_panel_grid(new_df.index, close.index):
            raise FactorEmptyError(
                "result.h5 index does not match the panel grid (datetime dtype/tz or off-grid timestamps): "
                f"first rows {list(new_df.index[:2])} vs panel {list(close.index[:2])}"
            )

        order = [str(c) for c in new_df.columns]
        flagged = self.lookahead_check(exp, new_df)
        if flagged:
            new_df = new_df.drop(columns=list(flagged))

        logger.info(f"Crypto runner: evaluating {list(new_df.columns)} against library {list(library.columns)} ...")
        try:
            ev = evaluate_experiment(new_df, library, close, cfg)
        except (ValueError, KeyError, TypeError) as e:
            raise FactorEmptyError(f"Factor evaluation failed on malformed factor output: {e}") from e
        if flagged:
            factors = dict(ev.get("factors") or {})
            for name, reason in flagged.items():
                factors[name] = self._rejected_entry(reason)
            ev["factors"] = {name: factors[name] for name in order if name in factors}
            ev["summary_text"] = summary_text(ev, cfg)

        exp.crypto_eval = ev
        exp.crypto_accepted = list(ev.get("accepted") or [])
        exp.result = result_series(ev, cfg)
        exp.stdout = str(ev.get("summary_text", ""))

        try:
            self.write_workspace_files(exp, ev)
        except Exception as e:  # noqa: BLE001 - workspace I/O must not kill the loop
            logger.warning(f"Crypto runner: failed to write workspace files: {e}")

        try:
            append_record(Path(CRYPTO_EVAL_SETTINGS.ledger_path).expanduser(), self.ledger_record(exp, ev))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Crypto runner: failed to append ledger record: {e}")

        logger.info(f"Crypto runner: accepted={exp.crypto_accepted} library_after={ev.get('library_after')}")
        return exp
