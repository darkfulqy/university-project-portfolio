# Crypto 5m/15m factor loop on RD-Agent: design contract (v1)

Status: binding contract for the implementation agents. Every agent reads this file first.
Fork root: `/path/to/rdagent-crypto` (branch `crypto-5m`, shallow clone of microsoft/RD-Agent HEAD 32b3d395).
Dev venv on the Mac: `/path/to/rdagent-crypto/.venv` (python 3.11, `pip install -e .` + pandas numpy tables pyarrow scipy pytest).
Production host: a Linux desktop with an RTX 3060 (conda env `rdagent-crypto`). Never run the LLM loop on the Mac.

## 1. Goal and what we keep / replace

Keep from RD-Agent unchanged:
- `FactorRDLoop` (rdagent/app/qlib_rd_loop/factor.py) step machinery, Trace / SOTA bookkeeping, session dump/resume.
- `QlibFactorCoSTEER` coder and the whole CoSTEER evolve/evaluate/knowledge loop (rdagent/components/coder/factor_coder, rdagent/components/coder/CoSTEER).
- `FactorTask`, `FactorFBWorkspace` (factor.py is executed by subprocess `{FACTOR_CoSTEER_python_bin} factor.py` inside a workspace where every file of `FACTOR_CoSTEER_data_folder` (full) or `FACTOR_CoSTEER_data_folder_debug` (debug) is symlinked; output `result.h5`).
- `process_factor_data` from rdagent/scenarios/qlib/developer/utils.py (executes accepted sub-workspaces with data_type "All" and concatenates the frames). Do NOT import `rdagent.scenarios.qlib.developer.factor_runner` (it calls `pandarallel.initialize()` at import).

Replace (new code, all under the fork):
- Scenario (LLM-visible description of data, interface, simulator), hypothesis generation prompts, runner (evaluation), feedback (SOTA decision), app entry + CLI.
- Evaluation = cross-sectional Spearman RankIC of the factor at 5-minute origins against forward simple returns at horizons 5m and 15m, on Binance USDT spot 5m bars, 43 instruments, 2026-02-25..2026-08-23.

Why: the upstream runner backtests daily CSI300 with Qlib/LightGBM; its metric keys ("IC", "1day.excess_return_...") are hard-coded in three prompt templates (rdagent/scenarios/qlib/prompts.yaml:14-16, 34-36, 55-57), in feedback.py IMPORTANT_METRICS, and the Streamlit UI. We fork the prompts instead of padding fake keys. The UI is a non-goal for v1.

## 2. Package layout (new files unless stated)

```
rdagent/scenarios/crypto/__init__.py
rdagent/scenarios/crypto/conf.py            CryptoEvalSettings (env prefix CRYPTO_EVAL_)
rdagent/scenarios/crypto/data/__init__.py
rdagent/scenarios/crypto/data/build_panel.py   CLI: 1m csv.gz dir -> panel h5 (+ debug h5, README.md)
rdagent/scenarios/crypto/data/README_template.md   LLM-facing data description (rendered into README.md next to the h5)
rdagent/scenarios/crypto/evaluation.py      pure pandas/numpy metrics; NO rdagent imports
rdagent/scenarios/crypto/experiment.py      CryptoFactorExperiment, CryptoFactorScenario
rdagent/scenarios/crypto/runner.py          CryptoRankICRunner
rdagent/scenarios/crypto/proposal.py        CryptoFactorHypothesisGen, CryptoFactorHypothesis2Experiment
rdagent/scenarios/crypto/feedback.py        CryptoFactorExperiment2Feedback
rdagent/scenarios/crypto/prompts.yaml       all crypto prompt templates (keys in section 8)
rdagent/scenarios/crypto/ledger.py          JSONL ledger append + `report` subcommand
rdagent/scenarios/crypto/local_embedding.py optional hashing-embedding fallback (section 10)
rdagent/app/crypto_rd_loop/__init__.py
rdagent/app/crypto_rd_loop/conf.py          CryptoFactorPropSetting (env prefix CRYPTO_LOOP_)
rdagent/app/crypto_rd_loop/factor.py        CryptoFactorRDLoop + main()
rdagent/app/cli.py                          (EDIT upstream) add `crypto_factor` command mirroring fin_factor
deploy/crypto/env.example                   every env var with comments
deploy/crypto/setup_3060.sh                 miniconda (user-level, no sudo) + conda env + pip install -e .
deploy/crypto/sync_from_mac.sh              tar|ssh sync of the fork (exclude .venv .git git_ignore_folder log pickle_cache)
deploy/crypto/run_loop.sh                   dotenv + `rdagent crypto_factor --loop-n N`
docs/crypto_5m/DESIGN.md                    this file
docs/crypto_5m/RUNBOOK.md                   how to build data, run tests, run the loop, read the ledger
test/scenarios/crypto/test_build_panel.py
test/scenarios/crypto/test_evaluation.py
test/scenarios/crypto/test_runner_feedback.py
test/scenarios/crypto/test_prompts.py
```

Coding conventions: Python 3.11, `from __future__ import annotations`, black line length 120 (upstream style), type hints, no new third-party dependencies beyond pandas/numpy/tables/pyarrow/scipy (scipy optional; implement Spearman by ranking + Pearson in numpy so scipy is not required). Minimal edits to upstream files: only `rdagent/app/cli.py`. Do not modify anything under rdagent/scenarios/qlib or rdagent/components.

## 3. Data contract

### 3.1 Panel file
- File name: `crypto_5m.h5` (constant `CryptoEvalSettings.panel_filename`), pandas HDF5, key `"data"`, fixed format.
- Index: `MultiIndex` with names exactly `["datetime", "instrument"]`, sorted lexicographically. `datetime` is tz-naive UTC `datetime64[ns]` on a 5-minute grid; `instrument` is the Binance symbol string, e.g. `"BTCUSDT"`.
- `datetime` is the bar OPEN time. Bar `t` covers `[t, t+5min)`. Every column of row `t` is known at `t + 5min` (bar close). Factor code must treat row `t` as usable for a signal emitted at `t + 5min`.
- Columns (all float64, `$`-prefixed so upstream `get_file_desc` groups them):
  `$open $high $low $close $volume $amount $trade_count $taker_buy_volume $taker_buy_amount`
  - `$volume` base-asset volume, `$amount` quote (USDT) volume, `$trade_count` number of trades, `$taker_buy_volume` / `$taker_buy_amount` taker-buy base/quote volume. Sums over the 5 one-minute bars; open = first, high = max, low = min, close = last.
- Rows exist only where at least one 1-minute row exists inside the bar (no forward fill, no NaN padding). Instruments listed later (GRAMUSDT, REUSDT) start at their first bar.
- Source: `/path/to/market-data/quant_assist/data/binance_spot_1m_core43_180d/normalized/<SYMBOL>_1m.csv.gz`, columns `timestamps,open,high,low,close,volume,amount,trade_count,taker_buy_volume,taker_buy_amount`, `timestamps` ISO-8601 UTC (`2026-02-25T00:00:00Z`). 43 symbols, 2026-02-25..2026-08-23 23:59.

### 3.2 Folders consumed by RD-Agent
- Full folder = `FACTOR_CoSTEER_data_folder` (upstream setting, env `FACTOR_CoSTEER_data_folder`): contains `crypto_5m.h5` (all 43 symbols, full span) + `README.md`.
- Debug folder = `FACTOR_CoSTEER_data_folder_debug`: contains `crypto_5m.h5` (top 10 symbols by total `$amount`, last 14 days of the panel) + the same `README.md`. Used by CoSTEER during code evolution; the full folder is used by the runner ("All").
- The runner reads `$close` from the full folder's panel to build targets. There is no separate target file.
- Build both with `python -m rdagent.scenarios.crypto.data.build_panel --source-dir <normalized dir> --out-dir <full folder> --debug-out-dir <debug folder> [--bar-minutes 5 --debug-symbols 10 --debug-days 14]`. Must print row counts and the date span.
- Default output location on the Mac: `/path/to/rdagent-crypto-data/5m/full` and `/path/to/rdagent-crypto-data/5m/debug` (outside the git repo). On the 3060: `~/rdagent-crypto-data/5m/{full,debug}`.

### 3.3 Factor output contract (unchanged upstream contract, restated)
`result.h5`, key `"data"`, MultiIndex `(datetime, instrument)`, exactly one float column named as the factor. `datetime` values must lie on the panel grid. The value at `(t, i)` may only use panel rows with `datetime <= t` (bars closed by `t + 5min`). Anything else is look-ahead and is a bug the evaluation must be robust to (see 4.2 shift test).

## 4. Evaluation contract (`evaluation.py`, no rdagent imports)

### 4.1 Functions (exact names)
```python
@dataclass(frozen=True)
class EvalConfig:
    bar_minutes: int = 5
    horizons_minutes: tuple[int, ...] = (5, 15)
    primary_horizon_minutes: int = 15
    train: tuple[str, str]; valid: tuple[str, str]; test: tuple[str, str]   # ISO date/datetime strings, inclusive, UTC
    purge_minutes: int = 15
    min_assets: int = 30
    n_quantiles: int = 5
    fee_bp_per_side: float = 10.0
    slippage_bp_per_side: float = 5.0
    gate_min_tstat: float = 3.0
    gate_min_abs_ic: float = 0.01      # v1.1: raised from 0.005 (below the static-ranking null noise level)
    gate_min_coverage: float = 0.5
    gate_max_corr: float = 0.7
    gate_min_ir_gain: float = 0.0
    gate_max_abs_ic: float = 0.25      # v1.1: implausibility cap (gate R0)
    # property nw_lags = 2 * max(horizons_minutes) // bar_minutes   (Newey-West lags, 6 by default)

def forward_returns(close: pd.Series, cfg: EvalConfig) -> pd.DataFrame
    # close: Series indexed (datetime, instrument). Returns DataFrame same index, columns "ret_5m", "ret_15m", ...
    # ret_h[t,i] = close[t + h, i] / close[t, i] - 1 computed on a per-instrument full 5-minute grid (reindex, shift by h/bar_minutes, no ffill). NaN where the future bar is missing.

def segment_masks(datetimes: pd.Index, cfg: EvalConfig) -> dict[str, np.ndarray]
    # keys "train","valid","test". An origin t belongs to a segment if start <= t <= end AND t + max(horizons) + purge_minutes <= segment end (so no target crosses into the next segment).

def rankic_by_timestamp(factor: pd.Series, ret: pd.Series, min_assets: int) -> pd.Series
    # Spearman per datetime across instruments with both non-NaN; skip datetimes with < min_assets pairs. Implement as average-rank then Pearson (numpy). Index = datetime.

def factor_metrics(factor: pd.Series, rets: pd.DataFrame, cfg: EvalConfig) -> dict
    # {segment: {f"{h}m": {n_ts, ic_mean, ic_std, ic_ir, ic_tstat, ic_tstat_naive, ic_tstat_dynamic, ic_pos_rate, coverage, rank_autocorr_1, q_spread_bp_gross, q_spread_bp_net}}}
    # ic_ir = ic_mean / ic_std; ic_tstat_naive = ic_ir * sqrt(n_ts)
    # v1.1: ic_tstat = Newey-West t-statistic of ic_mean (Bartlett kernel, cfg.nw_lags lags) because per-bar ICs of
    #   overlapping 15m targets on 5m origins are autocorrelated and the naive statistic is inflated ~3x.
    # v1.1: ic_tstat_dynamic = Newey-West t-statistic of the IC of (factor - per-instrument mean over the train rows),
    #   the time-varying part of the factor; defined at the primary horizon only (NaN at the others).
    # only finite factor values are observations (inf counts as missing)
    # coverage = mean over datetimes of (# non-NaN instruments / # instruments present in panel at that datetime)
    # rank_autocorr_1 = mean over datetimes of cross-sectional Spearman between factor[t] and factor[t - 1 bar] (turnover proxy, 1.0 = static)
    # q_spread_bp_gross = mean over datetimes of (mean ret of top quantile - mean ret of bottom quantile) * 1e4; q_spread_bp_net = gross - turnover_estimate * 2 * (fee + slippage) where turnover_estimate = (1 - rank_autocorr_1) clipped to [0,1] (document this approximation in the docstring).

def cross_sectional_rank(factor: pd.Series) -> pd.Series
    # per datetime: pct rank in (0,1] minus 0.5; NaN stays NaN.

def composite(library: pd.DataFrame, signs: dict[str, int]) -> pd.Series
    # library columns = factor names; composite = mean over columns of sign * cross_sectional_rank(col), skipping NaN; if library is empty -> empty Series.

def factor_sign(metrics: dict, cfg: EvalConfig) -> int
    # sign of train ic_mean at the primary horizon (0 -> +1).

def max_abs_corr_with_library(new: pd.Series, library: pd.DataFrame, datetimes_mask: np.ndarray) -> tuple[float, str | None]
    # mean over datetimes (train+valid) of cross-sectional Spearman with each library column; return (max |corr|, name). (0.0, None) when library empty.

def gate(new_metrics: dict, max_corr: float, composite_before: dict | None, composite_after: dict, cfg: EvalConfig) -> tuple[bool, list[str]]
    # rules (all on segment "valid", horizon primary):
    #   R0 (v1.1) degenerate / look-ahead suspected: ic_std == 0 (constant IC series) or |ic_mean| > gate_max_abs_ic
    #   R1 |ic_mean| >= gate_min_abs_ic and ic_tstat sign-consistent: sign(valid ic_mean) == sign(train ic_mean) and |ic_tstat| >= gate_min_tstat
    #      and (v1.1) |ic_tstat_dynamic| >= gate_min_tstat (a static per-instrument ranking cannot pass on persistence alone)
    #   R2 coverage >= gate_min_coverage
    #   R3 max_corr < gate_max_corr
    #   R4 composite_after valid ic_ir (signed) >= composite_before valid ic_ir + gate_min_ir_gain (composite_before None or empty library -> passes)
    # returns (passed, reasons) where reasons lists every failed rule with numbers, e.g. "R1 valid.15m ic_tstat=1.8 < 3.0".

def evaluate_experiment(new_factors: pd.DataFrame, library: pd.DataFrame, close: pd.Series, cfg: EvalConfig) -> dict
    # orchestrates: rets = forward_returns; per new factor -> metrics, sign, max_corr, composite_before (library) / composite_after (library + this factor with its sign), gate.
    # Greedy acceptance in column order: an accepted factor is added to the library before evaluating the next one.
    # returns the ExperimentEval dict (section 4.3).
```

### 4.2 Required behaviours (tests must cover)
- Planted signal: factor = ret_15m + noise -> valid ic_mean > 0.3; reversed sign -> negative; pure noise -> |ic_mean| < 0.02 with n_ts >= 500.
- No look-ahead in targets: shifting the factor forward by one bar (using ret at t+1 as factor at t) must NOT give IC ~1 for horizon 5m of the same construction ... simpler: `forward_returns` uses only close at t and t+h; assert ret_5m[t] == close[t+5m]/close[t]-1 on a hand-built panel with gaps (a missing t+5m bar -> NaN, not the next available bar).
- Segment purge: origins within `max(horizons)+purge` of a segment end are excluded; no origin appears in two segments.
- min_assets: datetimes with fewer pairs are skipped and n_ts reflects that.
- composite of a single factor equals its own rank (sign applied); composite of two identical factors equals one.
- gate R3 rejects a factor equal to a library factor; R1 rejects noise.
- (v1.1) R0 rejects a factor equal to ret_15m (ic_std=0) with an explicit "look-ahead suspected" reason and a factor with valid |ic_mean| > gate_max_abs_ic; R1 rejects a static per-instrument ranking (ic_tstat_dynamic).
- (v1.1) ic_tstat is smaller than ic_tstat_naive for a positively autocorrelated IC series; inf factor values count as missing.

### 4.3 ExperimentEval dict (pickle-safe: plain dict/list/str/float/bool/None only)
```
{
  "cfg": {...EvalConfig as dict...},
  "n_instruments": int, "panel_start": str, "panel_end": str,
  "library_before": [names], "library_after": [names],
  "composite_before": {segment: {"15m": {...metrics...}, "5m": {...}}} | None,
  "composite_after": {...same...},
  "factors": {
     name: {"metrics": {segment: {h: {...}}}, "sign": +1|-1, "max_corr": float, "max_corr_with": str|None,
            "gate_passed": bool, "gate_reasons": [str], "accepted": bool}
  },
  "accepted": [names],
  "summary_text": str   # 5-10 lines, train+valid only, used as exp.stdout
}
```

## 5. Runner (`runner.py`)
```python
class CryptoRankICRunner(CachedRunner[CryptoFactorExperiment]):
    def __init__(self, scen: Scenario): ...
    def develop(self, exp: CryptoFactorExperiment) -> CryptoFactorExperiment
```
- Baseline: if `exp.based_experiments` and `exp.based_experiments[-1].result is None`, set that experiment's `result` to `baseline_result()` (Series of NaN for all standard keys, `n_library_factors=0`) and its `crypto_eval` to `{"accepted": [], "library_after": [], ...}`; do not call `develop` recursively.
- New factors: `new_df = process_factor_data(exp)` (import from `rdagent.scenarios.qlib.developer.utils`). Keep only columns whose name matches a sub-task factor_name; drop duplicate columns. (v1.1) Normalise the index (tz-aware datetime -> naive UTC `datetime64[ns]`), drop duplicated `(datetime, instrument)` rows (keep last, log a warning), and raise `FactorEmptyError` when no row of the frame lies on the panel grid. The `FactorEmptyError` for a name mismatch lists the columns actually produced.
- (v1.1) Look-ahead re-run (gate R0, `CryptoRankICRunner.lookahead_check`): before evaluation, write the panel truncated `LOOKAHEAD_CUT_DAYS=2` days before its end into a temporary data folder, execute every new sub-workspace's `factor.py` there in a fresh subprocess (`FACTOR_CoSTEER_python_bin factor.py`, no pickle cache) and compare all values with `datetime <= cut` against the full-panel result (`np.isclose(rtol=1e-6, atol=1e-9, equal_nan=True)`). A causal factor is identical there; any difference (or an execution failure on the truncated panel: fail closed) drops the column from evaluation and records `factors[name] = {"metrics": {}, "sign": 1, "max_corr": 0.0, "max_corr_with": None, "gate_passed": False, "gate_reasons": ["R0 look-ahead: values change when future rows are removed (...)"], "accepted": False}` (same shape as evaluated factors, in the original column order). Factors without a workspace/`factor.py` (unit tests with a monkeypatched `process_factor_data`) skip the check with a warning.
- Library: iterate `exp.based_experiments` (skip those without `crypto_accepted` or with empty list); for each, `process_factor_data(based_exp)` and keep the accepted columns (upstream pickle cache on `FactorFBWorkspace.execute` makes this cheap). Cache in an instance dict keyed by `id(based_exp)` for the process lifetime. (v1.1) The runner defines `__getstate__`/`__setstate__` that drop `_library_cache` and `_close_cache`, because `LoopBase.dump()` pickles the whole loop (runner included) after every step and the `id()` keys are meaningless after a resume.
- Close: read `FACTOR_COSTEER_SETTINGS.data_folder / panel_filename`, column `$close`.
- `ev = evaluate_experiment(...)` (v1.1: a `ValueError`/`KeyError`/`TypeError` raised by malformed factor output is re-raised as `FactorEmptyError` so `FactorRDLoop.skip_loop_error` routes it to feedback instead of killing the process); set `exp.crypto_eval = ev`, `exp.crypto_accepted = ev["accepted"]`, `exp.result = result_series(ev)` where `result_series` flattens `composite_after` into a float Series with keys `"{segment}.{h}.{metric}"` plus `"n_library_factors"`; `exp.stdout = ev["summary_text"]`.
- Write into `exp.experiment_workspace.workspace_path`: `crypto_eval.json` (the dict) and `factor_metrics.csv` (one row per factor × segment × horizon).
- Append a ledger record (section 9). Never raise on ledger I/O errors (log a warning).
- No runner-level pickle cache decorator (evaluation is cheap; upstream cache key ignores the data path).
- Raise `FactorEmptyError` if `new_df` has no usable column (mirrors upstream so FactorRDLoop skips to feedback).

## 6. Scenario and experiment (`experiment.py`)
- `class CryptoFactorExperiment(QlibFactorExperiment)`: adds `self.crypto_accepted: list[str] = []` and `self.crypto_eval: dict | None = None`. (Reusing the Qlib experiment keeps `process_factor_data`'s isinstance checks happy; its Qlib template files in the workspace are unused.)
- `class CryptoFactorScenario(Scenario)`: same property set as `QlibFactorScenario` (background, get_source_data_desc, output_format, interface, simulator, rich_style_description, experiment_setting, get_scenario_all_desc, get_runtime_environment). Templates from `rdagent/scenarios/crypto/prompts.yaml` via `T(".prompts:crypto_factor_background")` etc.
  - `get_runtime_environment()` returns a static string built in-process (python version, pandas/numpy/tables versions, "CPU only, subprocess `python factor.py`, timeout 3600 s"). No subprocess, no conda.
  - `get_source_data_desc()` calls `get_data_folder_intro(fname_reg=r"^(crypto_5m\.h5|README\.md)$")` from rdagent/scenarios/qlib/experiment/utils **only after** asserting both `FACTOR_COSTEER_SETTINGS.data_folder` and `data_folder_debug` exist and contain `crypto_5m.h5`; otherwise raise `FileNotFoundError` with the build_panel command in the message (never trigger the Qlib/Docker generation). (v1.1) The `fname_reg` restricts the description to the two contract files so a stray `.DS_Store`/`*.bak` in the folder cannot make upstream `get_file_desc` raise `NotImplementedError`.
  - `experiment_setting` renders the segment dates, horizons, min_assets and the gate thresholds from `CryptoEvalSettings` as a markdown table.

## 7. Proposal (`proposal.py`)
- `class CryptoFactorHypothesisGen(QlibFactorHypothesisGen)`: override `prepare_context` to render `scenarios.crypto.prompts:hypothesis_and_feedback`, `last_hypothesis_and_feedback`, `sota_library_summary` (accepted factor names + valid metrics; v1.1: also passed as `sota_hypothesis_and_feedback`, the only key of the two that upstream `LLMHypothesisGen.gen` forwards into the rendered prompt, so `RAG` carries the stage guidance and hard rules only), `factor_hypothesis_output_format`, `factor_hypothesis_specification`, and a crypto `RAG` string (rounds < 8: simple microstructure factors: short-horizon reversal/momentum, taker-buy imbalance, volume/amount shocks vs rolling mean, realized volatility, range/ATR normalisation, BTC-relative return, time-of-day effects; rounds >= 8: interactions and regime-conditioned factors; always: only past bars, prefer low turnover (rank_autocorr_1 high), 1-3 factors per round).
- `class CryptoFactorHypothesis2Experiment(QlibFactorHypothesis2Experiment)`: same as upstream but templates from `scenarios.crypto.prompts`, builds `CryptoFactorExperiment`, (v1.1) appends the hard rules + `sota_library_summary` to the `hypothesis_and_feedback` context value because the upstream `hypothesis2experiment` templates render neither `RAG` nor `target_list` (both keys are still supplied: `LLMHypothesis2Experiment.convert` reads them unconditionally), `based_experiments = [CryptoFactorExperiment(sub_tasks=[])] + [accepted experiments]`, and actually drops tasks whose `factor_name` already exists in the accepted library (upstream sets `exp.tasks`, which is dead code; construct the experiment from the filtered task list instead).
- Templates render metrics from `experiment.crypto_eval` with Jinja `.get()` chains, never `.loc[...]`. Only `train` and `valid` segments are rendered, never `test`.

## 8. Prompt keys in `rdagent/scenarios/crypto/prompts.yaml`
`crypto_factor_background`, `crypto_factor_interface`, `crypto_factor_output_format`, `crypto_factor_simulator`, `crypto_factor_rich_style_description`, `crypto_factor_experiment_setting`,
`hypothesis_and_feedback`, `last_hypothesis_and_feedback`, `sota_library_summary`, `factor_hypothesis_output_format`, `factor_hypothesis_specification`, `factor_experiment_output_format`,
`factor_feedback_generation` (with `system` and `user`).
Content rules: describe 5-minute Binance USDT spot bars, the `(datetime, instrument)` index, the `$` columns, the "row t known at t+5min" rule, the RankIC evaluation with horizons 5m/15m, the gate rules, and the fee model. Interface/output format keep the upstream contract (calculate_{name} function, `result.h5`, one column) but with a crypto example index (`Timestamp('2026-03-01 00:05:00'), 'BTCUSDT'`). No "stock", "day", "CSI300", "Qlib", "LightGBM", "portfolio" wording.

## 9. Feedback (`feedback.py`) and ledger (`ledger.py`)
- `class CryptoFactorExperiment2Feedback(Experiment2Feedback)`: `decision = bool(exp.crypto_eval and exp.crypto_eval["accepted"])` (deterministic gate; the LLM cannot flip it). Calls the LLM with `scenarios.crypto.prompts:factor_feedback_generation.system/user` (json_mode) to get `Observations`, `Feedback for Hypothesis`, `New Hypothesis`, `Reasoning`; on any exception return the feedback with those fields set to a short error note and the gate decision unchanged. `reason` must begin with the gate outcome per factor (name: accepted / rejected: reasons).
- Ledger: `append_record(path, record: dict)` writes one JSON line; record = `{"ts_utc", "hypothesis", "reason", "factors": [{name, description, formulation, workspace_path, metrics, sign, max_corr, gate_passed, gate_reasons, accepted}], "composite_before", "composite_after", "accepted", "library_after"}`. `python -m rdagent.scenarios.crypto.ledger report --path X` prints: accepted library table (name, valid 15m ic_mean/ic_ir/ic_tstat, rank_autocorr_1, test 15m ic_mean) and the 10 best rejected by valid ic_tstat with reasons.

## 10. Settings and env
`CryptoEvalSettings(BaseSettings)` env prefix `CRYPTO_EVAL_`: `panel_filename="crypto_5m.h5"`, `bar_minutes=5`, `horizons_minutes="5,15"` (comma string parsed to tuple), `primary_horizon_minutes=15`, `train_start="2026-02-25"`, `train_end="2026-06-30"`, `valid_start="2026-07-01"`, `valid_end="2026-07-31"`, `test_start="2026-08-01"`, `test_end="2026-08-23 23:59:00"`, `purge_minutes=15`, `min_assets=30`, `n_quantiles=5`, `fee_bp_per_side=10`, `slippage_bp_per_side=5`, `gate_min_tstat=3.0`, `gate_min_abs_ic=0.01` (v1.1, was 0.005), `gate_min_coverage=0.5`, `gate_max_corr=0.7`, `gate_min_ir_gain=0.0`, `gate_max_abs_ic=0.25` (v1.1), `ledger_path="./crypto_factor_ledger.jsonl"`. Provide `to_eval_config()`.
`CryptoFactorPropSetting(FactorBasePropSetting)` env prefix `CRYPTO_LOOP_`: `scen`, `hypothesis_gen`, `hypothesis2experiment`, `runner`, `summarizer` default to the crypto classes; `coder` stays `rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER`.
`rdagent/app/crypto_rd_loop/factor.py`: `class CryptoFactorRDLoop(FactorRDLoop)` constructed with `CRYPTO_PROP_SETTING`; `main(path, step_n, loop_n, all_duration, checkout, checkout_path)` identical in shape to upstream `main` (call `_init_base_features(None)` as upstream does; the crypto runner ignores `exp.base_features`).
`rdagent/app/cli.py`: `crypto_factor` command with the same options as `fin_factor`.
Local embedding fallback (`local_embedding.py`): when env `CRYPTO_LOCAL_EMBEDDING=1`, `rdagent.app.crypto_rd_loop.factor` monkeypatches `rdagent.oai.llm_utils.calculate_embedding_distance_between_str_list` with a hashed-token cosine similarity (numpy only, 4096 buckets, unigram+bigram). Document that this only degrades CoSTEER's knowledge retrieval quality and lets the loop run without an embedding provider.
`deploy/crypto/run_loop.sh` / RUNBOOK (v1.1): `loop_n` is the absolute loop count of the session (`LoopBase.run` restarts the loop index at 0 and counts completed loops of a resumed session), not "N more loops".
`deploy/crypto/sync_from_mac.sh` (v1.1): the `log` excludes must be anchored to the fork root (bsdtar `--exclude='^./log'`); an unanchored `--exclude=log` strips `rdagent/log/`.
`deploy/crypto/env.example` documents: `CHAT_MODEL`, `OPENAI_API_BASE`, `OPENAI_API_KEY`, alternative `ANTHROPIC_API_KEY` + `CHAT_MODEL=anthropic/<model>`, `EMBEDDING_MODEL` (or `CRYPTO_LOCAL_EMBEDDING=1`), `REASONING_EFFORT=` (must stay empty for openai/ Claude routes), `FACTOR_CoSTEER_data_folder`, `FACTOR_CoSTEER_data_folder_debug`, `FACTOR_CoSTEER_python_bin` (absolute path of the conda env python), `FACTOR_CoSTEER_new_knowledge_base_path`, `PICKLE_CACHE_FOLDER_PATH_STR` (bump when the panel changes), `LOG_TRACE_PATH`, all `CRYPTO_EVAL_*`, `CRYPTO_LOOP_*`.

## 11. Tests (pytest, `test/scenarios/crypto/`, run with `.venv/bin/python -m pytest test/scenarios/crypto -q`)
- `test_build_panel.py`: synthetic 1m csv.gz for 3 symbols × 2 days (one late-listed, one with missing minutes) -> assert 5m OHLC/sum aggregation exactly, index names/dtypes, debug subset selection (top-N by `$amount`, last-N days), README.md written in both folders.
- `test_evaluation.py`: the behaviours in 4.2 plus `gate` and `evaluate_experiment` greedy acceptance.
- `test_runner_feedback.py`: monkeypatch `process_factor_data` to return synthetic frames and `FACTOR_COSTEER_SETTINGS.data_folder` to a tmp panel; run `CryptoRankICRunner.develop` on a fake `CryptoFactorExperiment` with `based_experiments=[CryptoFactorExperiment(sub_tasks=[])]`; assert baseline assignment, `exp.result` keys, workspace files, ledger line; then `CryptoFactorExperiment2Feedback.generate_feedback` with `APIBackend` monkeypatched to return a canned JSON -> decision equals gate; and with `APIBackend` raising -> decision still equals gate.
- `test_prompts.py`: instantiate `CryptoFactorScenario` against a tmp data folder pair; render every template with a fake `Trace` (hist with/without `crypto_eval`); assert no exception, and that none of `"1day."`, `"CSI300"`, `"Qlib"`, `"stock"` appear, and that no `test` segment numbers appear in the hypothesis prompts.

## 12. Non-goals for v1
Streamlit UI, funding-rate / open-interest features, multi-frequency panels, live trading, model (fin_model) scenario.
