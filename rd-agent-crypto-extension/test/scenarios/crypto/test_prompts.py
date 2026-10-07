"""Prompt rendering tests for the crypto 5-minute factor scenario (DESIGN.md section 11, ``test_prompts.py``).

Everything runs offline: the scenario is built against a tiny synthetic ``crypto_5m.h5`` folder pair, the LLM backend
is monkeypatched in the feedback tests and no network access is needed.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.proposal import Hypothesis, HypothesisFeedback, Trace
from rdagent.scenarios.crypto import experiment, feedback, proposal
from rdagent.scenarios.crypto.conf import CRYPTO_EVAL_SETTINGS  # noqa: F401 - a broken conf.py must fail here
from rdagent.utils.agent.tpl import T

PANEL_FILENAME = "crypto_5m.h5"
FORBIDDEN = ("1day.", "CSI300", "Qlib", "LightGBM", "stock")
TEST_SENTINEL = 0.987654
TEST_SENTINEL_STR = "0.987654"
TEST_SENTINEL_N_TS = 7777777  # n_ts placed only in the test segment (rendered as an int, not %.4f)


# --------------------------------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------------------------------
def _write_panel(folder: Path, instruments: list[str], n_bars: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2026-03-01 00:00:00", periods=n_bars, freq="5min"), instruments],
        names=["datetime", "instrument"],
    )
    close = 100.0 + rng.standard_normal(len(idx)).cumsum() * 0.1
    volume = rng.uniform(1.0, 10.0, len(idx))
    df = pd.DataFrame(
        {
            "$open": close - 0.01,
            "$high": close + 0.05,
            "$low": close - 0.05,
            "$close": close,
            "$volume": volume,
            "$amount": volume * close,
            "$trade_count": rng.integers(10, 100, len(idx)).astype(float),
            "$taker_buy_volume": volume * 0.5,
            "$taker_buy_amount": volume * close * 0.5,
        },
        index=idx,
    ).sort_index()
    df.to_hdf(folder / PANEL_FILENAME, key="data", mode="w", format="fixed")
    (folder / "README.md").write_text(
        "# crypto_5m.h5\n\nBinance USDT spot 5-minute bars indexed by (datetime, instrument); "
        "datetime is the bar open time (UTC) and every column of row t is known at t + 5min.\n"
    )


@pytest.fixture(scope="module")
def mods():
    """The real scenario modules (no stub: a broken conf.py must fail these tests, not be papered over)."""
    return types.SimpleNamespace(experiment=experiment, feedback=feedback, proposal=proposal)


@pytest.fixture()
def data_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    full = tmp_path / "full"
    debug = tmp_path / "debug"
    _write_panel(full, ["BTCUSDT", "ETHUSDT", "SOLUSDT"], 24)
    _write_panel(debug, ["BTCUSDT", "ETHUSDT"], 12)
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder", str(full))
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder_debug", str(debug))
    return full, debug


@pytest.fixture()
def scen(mods, data_folders):
    return mods.experiment.CryptoFactorScenario()


def _metrics(ic: float, n_ts: int = 1000) -> dict:
    return {
        "n_ts": n_ts,
        "ic_mean": ic,
        "ic_std": 0.05,
        "ic_ir": ic / 0.05,
        "ic_tstat": ic / 0.05 * np.sqrt(n_ts),
        "ic_pos_rate": 0.55,
        "coverage": 0.9,
        "rank_autocorr_1": 0.8,
        "q_spread_bp_gross": 3.0,
        "q_spread_bp_net": 1.0,
    }


def _segments(train_ic: float, valid_ic: float) -> dict:
    """Metrics for all segments; the test segment carries the sentinel number that must never be rendered."""
    return {
        "train": {"5m": _metrics(train_ic * 0.5), "15m": _metrics(train_ic)},
        "valid": {"5m": _metrics(valid_ic * 0.5), "15m": _metrics(valid_ic)},
        "test": {"5m": _metrics(TEST_SENTINEL, TEST_SENTINEL_N_TS), "15m": _metrics(TEST_SENTINEL, TEST_SENTINEL_N_TS)},
    }


def _cfg() -> dict:
    return {
        "bar_minutes": 5,
        "horizons_minutes": [5, 15],
        "primary_horizon_minutes": 15,
        "train": ["2026-02-25", "2026-06-30"],
        "valid": ["2026-07-01", "2026-07-31"],
        "test": ["2026-08-01", "2026-08-23 23:59:00"],
        "purge_minutes": 15,
        "min_assets": 30,
        "n_quantiles": 5,
        "fee_bp_per_side": 10.0,
        "slippage_bp_per_side": 5.0,
        "gate_min_tstat": 3.0,
        "gate_min_abs_ic": 0.005,
        "gate_min_coverage": 0.5,
        "gate_max_corr": 0.7,
        "gate_min_ir_gain": 0.0,
    }


def _hypothesis(text: str) -> Hypothesis:
    return Hypothesis(text, "reason", "concise reason", "concise observation", "concise justification", "knowledge")


def _task(name: str) -> FactorTask:
    return FactorTask(
        factor_name=name,
        factor_description=f"[Reversal Factor] {name} description",
        factor_formulation=r"f_t = -\frac{close_t}{close_{t-6}} + 1",
        variables={"close_t": "close of bar t"},
    )


@pytest.fixture()
def trace(mods, scen) -> Trace:
    """Trace with (a) an evaluated experiment with one accepted and one rejected factor, (b) a failed experiment."""
    exp_cls = mods.experiment.CryptoFactorExperiment

    exp_a = exp_cls([_task("rev_6"), _task("noise_x")], hypothesis=_hypothesis("Short-horizon reversal works."))
    exp_a.crypto_eval = {
        "cfg": _cfg(),
        "n_instruments": 43,
        "panel_start": "2026-02-25 00:00:00",
        "panel_end": "2026-08-23 23:55:00",
        "library_before": [],
        "library_after": ["rev_6"],
        "composite_before": None,
        "composite_after": _segments(0.02, 0.018),
        "factors": {
            "rev_6": {
                "metrics": _segments(0.02, 0.018),
                "sign": -1,
                "max_corr": 0.0,
                "max_corr_with": None,
                "gate_passed": True,
                "gate_reasons": [],
                "accepted": True,
            },
            "noise_x": {
                "metrics": _segments(0.001, -0.0005),
                "sign": 1,
                "max_corr": 0.1,
                "max_corr_with": "rev_6",
                "gate_passed": False,
                "gate_reasons": ["R1 valid.15m |ic_mean|=0.0005 < 0.005", "R1 valid.15m ic_tstat=-0.3 < 3.0"],
                "accepted": False,
            },
        },
        "accepted": ["rev_6"],
        "summary_text": "rev_6 accepted; noise_x rejected (train/valid summary)",
    }
    exp_a.crypto_accepted = ["rev_6"]
    exp_a.stdout = exp_a.crypto_eval["summary_text"]
    fb_a = HypothesisFeedback(
        observations="rev_6 is significant on valid.",
        hypothesis_evaluation="Supported.",
        new_hypothesis="Try taker-buy imbalance next.",
        reason="rev_6: accepted\nnoise_x: rejected: R1",
        decision=True,
    )

    exp_b = exp_cls([_task("broken_factor")], hypothesis=_hypothesis("Volume shocks predict returns."))
    exp_b.crypto_eval = None
    exp_b.crypto_accepted = []
    fb_b = HypothesisFeedback(
        observations="The code failed.",
        hypothesis_evaluation="Not verifiable.",
        new_hypothesis="Fix the implementation.",
        reason="broken_factor: rejected: not evaluated",
        decision=False,
    )

    tr = Trace(scen)
    tr.hist.append((exp_a, fb_a))
    tr.hist.append((exp_b, fb_b))
    return tr


# --------------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------------
def _assert_clean(text: str, *, check_sentinel: bool = True) -> None:
    assert isinstance(text, str) and text.strip()
    for word in FORBIDDEN:
        assert word not in text, f"forbidden substring {word!r} found in rendered prompt"
    if check_sentinel:
        assert TEST_SENTINEL_STR not in text, "test-segment metric leaked into a prompt"
        assert str(TEST_SENTINEL_N_TS) not in text, "test-segment n_ts leaked into a prompt"
        assert "test.15m" not in text and "test.5m" not in text


# --------------------------------------------------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------------------------------------------------
def test_prompts_loadable_by_both_uri_forms(mods):
    from_pkg = T("scenarios.crypto.prompts:crypto_factor_background").r(runtime_environment=None)
    assert "5-minute" in from_pkg or "5 minute" in from_pkg
    for key in (
        "crypto_factor_background",
        "crypto_factor_interface",
        "crypto_factor_output_format",
        "crypto_factor_simulator",
        "crypto_factor_rich_style_description",
        "crypto_factor_experiment_setting",
        "hypothesis_and_feedback",
        "last_hypothesis_and_feedback",
        "sota_library_summary",
        "factor_hypothesis_output_format",
        "factor_hypothesis_specification",
        "factor_experiment_output_format",
        "factor_feedback_generation.system",
        "factor_feedback_generation.user",
    ):
        assert isinstance(T(f"scenarios.crypto.prompts:{key}").template, str), key


def test_scenario_requires_panel_files(mods, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder", str(tmp_path / "missing_full"))
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder_debug", str(tmp_path / "missing_debug"))
    with pytest.raises(FileNotFoundError) as excinfo:
        mods.experiment.CryptoFactorScenario()
    assert "build_panel" in str(excinfo.value)


def test_scenario_ignores_stray_files_in_data_folders(mods, data_folders):
    full, debug = data_folders
    for folder in (full, debug):
        (folder / ".DS_Store").write_bytes(b"\x00\x01junk")
        (folder / "crypto_5m.h5.bak").write_bytes(b"old panel")
    scen = mods.experiment.CryptoFactorScenario()  # must not raise NotImplementedError for .DS_Store
    desc = scen.get_source_data_desc()
    assert PANEL_FILENAME in desc and ".DS_Store" not in desc and ".bak" not in desc


def test_scenario_description(mods, scen):
    desc = scen.get_scenario_all_desc()
    _assert_clean(desc)
    assert PANEL_FILENAME in desc
    assert "calculate_{function_name}" in desc
    assert "result.h5" in desc
    assert "(datetime, instrument)" in desc or '("datetime", "instrument")' in desc
    assert "t + 5min" in desc
    assert "RankIC" in desc
    assert "Timestamp('2026-03-01 00:05:00'), 'BTCUSDT'" in desc
    assert "$taker_buy_volume" in desc
    # experiment setting renders the gate thresholds and the segments as a table
    setting = scen.experiment_setting
    _assert_clean(setting)
    assert "|" in setting and "2026-07-01" in setting and "15m" in setting
    assert "Gate R0" in setting and "0.25" in setting and "ic_tstat_dynamic" in setting
    assert "R0" in scen.simulator and "Newey-West" in scen.simulator and "truncated" in scen.simulator
    # static runtime environment: no subprocess, mentions the interpreter and libraries
    env = scen.get_runtime_environment()
    assert "Python" in env and "pandas" in env and "factor.py" in env
    _assert_clean(scen.rich_style_description)
    _assert_clean(scen.get_scenario_all_desc(simple_background=True))


def test_hypothesis_gen_context(mods, scen, trace):
    gen = mods.proposal.CryptoFactorHypothesisGen(scen)
    ctx, json_flag = gen.prepare_context(trace)
    assert json_flag is True
    for key in (
        "hypothesis_and_feedback",
        "last_hypothesis_and_feedback",
        "sota_library_summary",
        "RAG",
        "hypothesis_output_format",
        "hypothesis_specification",
    ):
        _assert_clean(ctx[key])
    # train/valid metrics of the evaluated experiment are visible, test metrics are not
    haf = ctx["hypothesis_and_feedback"]
    assert "rev_6" in haf and "ACCEPTED" in haf and "REJECTED" in haf
    assert "train.15m" in haf and "valid.15m" in haf
    assert "R1 valid.15m" in haf
    assert "No evaluation result is available" in haf  # experiment (b) with crypto_eval None
    assert "rev_6" in ctx["sota_library_summary"]
    assert "0.0180" in ctx["sota_library_summary"]  # valid ic_mean of the accepted factor
    assert ctx["sota_hypothesis_and_feedback"] == ctx["sota_library_summary"]  # the key upstream renders
    assert "time-of-day" in ctx["RAG"] and "Hard rules" in ctx["RAG"]  # early-stage guidance + hard rules
    # full prompts through the upstream generic templates
    system_prompt = T("components.proposal.prompts:hypothesis_gen.system_prompt").r(
        targets="factors",
        scenario=scen.get_scenario_all_desc(filtered_tag="factors"),
        hypothesis_output_format=ctx["hypothesis_output_format"],
        hypothesis_specification=ctx["hypothesis_specification"],
        user_instruction=None,
    )
    user_prompt = T("components.proposal.prompts:hypothesis_gen.user_prompt").r(
        targets="factors",
        hypothesis_and_feedback=ctx["hypothesis_and_feedback"],
        last_hypothesis_and_feedback=ctx["last_hypothesis_and_feedback"],
        sota_hypothesis_and_feedback=ctx["sota_hypothesis_and_feedback"],
        RAG=ctx["RAG"],
    )
    _assert_clean(system_prompt)
    _assert_clean(user_prompt)
    # the accepted-library table (with its metrics) and the hard rules reach the LLM in the rendered prompt
    assert "Accepted factor library (SOTA" in user_prompt and "0.0180" in user_prompt
    assert "do NOT re-implement" in user_prompt and "Hard rules" in user_prompt


def test_hypothesis_gen_context_first_round(mods, scen):
    gen = mods.proposal.CryptoFactorHypothesisGen(scen)
    ctx, _ = gen.prepare_context(Trace(scen))
    _assert_clean(ctx["hypothesis_and_feedback"])
    assert "first round" in ctx["hypothesis_and_feedback"]
    assert "empty" in ctx["sota_library_summary"]
    # late-stage RAG after >= 8 rounds
    long_trace = Trace(scen)
    exp_cls = mods.experiment.CryptoFactorExperiment
    for i in range(8):
        e = exp_cls([_task(f"f{i}")], hypothesis=_hypothesis(f"h{i}"))
        long_trace.hist.append((e, HypothesisFeedback(reason="x", decision=False)))
    ctx_late, _ = gen.prepare_context(long_trace)
    assert "regime" in ctx_late["RAG"]
    _assert_clean(ctx_late["hypothesis_and_feedback"])


def test_hypothesis2experiment_context_and_dedup(mods, scen, trace):
    h2e = mods.proposal.CryptoFactorHypothesis2Experiment()
    hypothesis = _hypothesis("Taker-buy imbalance predicts 15m returns.")
    ctx, json_flag = h2e.prepare_context(hypothesis, trace)
    assert json_flag is True
    for key in ("target_hypothesis", "scenario", "hypothesis_and_feedback", "experiment_output_format", "RAG"):
        _assert_clean(ctx[key])
    assert "rev_6" in ctx["RAG"]
    system_prompt = T("components.proposal.prompts:hypothesis2experiment.system_prompt").r(
        targets="factors", scenario=ctx["scenario"], experiment_output_format=ctx["experiment_output_format"]
    )
    user_prompt = T("components.proposal.prompts:hypothesis2experiment.user_prompt").r(
        targets="factors",
        target_hypothesis=ctx["target_hypothesis"],
        hypothesis_and_feedback=ctx["hypothesis_and_feedback"],
        last_hypothesis_and_feedback="",
        sota_hypothesis_and_feedback="",
        target_list=ctx["target_list"],
        RAG=ctx["RAG"],
    )
    _assert_clean(system_prompt)
    _assert_clean(user_prompt)
    # upstream renders neither RAG nor target_list: the library summary and hard rules must arrive through
    # hypothesis_and_feedback
    assert "rev_6" in user_prompt and "do NOT re-implement" in user_prompt and "Hard rules:" in user_prompt
    assert "Accepted factor library (SOTA" in user_prompt and "0.0180" in user_prompt

    # output format keeps the upstream JSON shape (factor name -> description/formulation/variables)
    fmt = ctx["experiment_output_format"]
    assert '"description"' in fmt and '"formulation"' in fmt and '"variables"' in fmt

    # convert_response drops the task already in the accepted library, keeps the new one
    response = json.dumps(
        {
            "rev_6": {"description": "dup", "formulation": "x", "variables": {"x": "x"}},
            "tbi_12": {
                "description": "[Order Flow Imbalance Factor] taker-buy share over 12 bars",
                "formulation": r"\frac{\sum tbv}{\sum v}",
                "variables": {"tbv": "$taker_buy_volume", "v": "$volume"},
            },
        }
    )
    exp = h2e.convert_response(response, hypothesis, trace)
    assert isinstance(exp, mods.experiment.CryptoFactorExperiment)
    assert [t.factor_name for t in exp.sub_tasks] == ["tbi_12"]
    assert len(exp.sub_workspace_list) == 1
    assert exp.hypothesis is hypothesis
    assert exp.crypto_accepted == [] and exp.crypto_eval is None
    # based_experiments = [empty baseline] + accepted experiments only
    assert isinstance(exp.based_experiments[0], mods.experiment.CryptoFactorExperiment)
    assert exp.based_experiments[0].sub_tasks == []
    assert [b.crypto_accepted for b in exp.based_experiments[1:]] == [["rev_6"]]


def _install_fake_backend(monkeypatch: pytest.MonkeyPatch, mods, *, raise_error: bool) -> dict:
    captured: dict = {}

    class FakeBackend:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def build_messages_and_create_chat_completion(self, *args, **kwargs) -> str:
            captured["user_prompt"] = kwargs.get("user_prompt", args[0] if args else "")
            captured["system_prompt"] = kwargs.get("system_prompt", args[1] if len(args) > 1 else "")
            captured["json_mode"] = kwargs.get("json_mode")
            if raise_error:
                raise RuntimeError("no API key configured")
            return json.dumps(
                {
                    "Observations": "canned observations",
                    "Feedback for Hypothesis": "canned evaluation",
                    "New Hypothesis": "canned new hypothesis",
                    "Reasoning": "canned reasoning",
                }
            )

    monkeypatch.setattr(mods.feedback, "APIBackend", FakeBackend)
    return captured


def test_feedback_prompts_and_decision_with_canned_llm(mods, scen, trace, monkeypatch: pytest.MonkeyPatch):
    captured = _install_fake_backend(monkeypatch, mods, raise_error=False)
    summarizer = mods.feedback.CryptoFactorExperiment2Feedback(scen)

    exp_a, _ = trace.hist[0]
    fb = summarizer.generate_feedback(exp_a, trace)
    assert fb.decision is True
    assert fb.reason.startswith("rev_6: accepted\nnoise_x: rejected: R1 valid.15m")
    assert fb.observations == "canned observations"
    assert fb.new_hypothesis == "canned new hypothesis"
    assert captured["json_mode"] is True
    _assert_clean(captured["system_prompt"])
    _assert_clean(captured["user_prompt"])
    assert "rev_6" in captured["user_prompt"] and "valid.15m" in captured["user_prompt"]
    assert "Gate outcome" in captured["user_prompt"]

    exp_b, _ = trace.hist[1]
    fb_b = summarizer.generate_feedback(exp_b, trace)
    assert fb_b.decision is False
    assert fb_b.reason.startswith("broken_factor: rejected: not evaluated")
    _assert_clean(captured["user_prompt"])
    assert "No evaluation result is available" in captured["user_prompt"]


def test_feedback_decision_survives_llm_failure(mods, scen, trace, monkeypatch: pytest.MonkeyPatch):
    _install_fake_backend(monkeypatch, mods, raise_error=True)
    summarizer = mods.feedback.CryptoFactorExperiment2Feedback(scen)

    exp_a, _ = trace.hist[0]
    fb = summarizer.generate_feedback(exp_a, trace)
    assert fb.decision is True  # equals the gate even though the LLM raised
    assert fb.reason.startswith("rev_6: accepted")
    assert mods.feedback.LLM_ERROR_NOTE in fb.observations
    assert mods.feedback.LLM_ERROR_NOTE in fb.reason

    exp_b, _ = trace.hist[1]
    fb_b = summarizer.generate_feedback(exp_b, trace, exception=RuntimeError("boom"))
    assert fb_b.decision is False
    assert fb_b.reason.startswith("broken_factor: rejected")
    assert "boom" in fb_b.reason


def test_feedback_templates_render_directly(mods, scen, trace):
    exp_a, _ = trace.hist[0]
    system_prompt = T("scenarios.crypto.prompts:factor_feedback_generation.system").r(
        scenario=scen.get_scenario_all_desc()
    )
    user_prompt = T("scenarios.crypto.prompts:factor_feedback_generation.user").r(
        hypothesis_text=exp_a.hypothesis.hypothesis,
        task_details=[t.get_task_information_and_implementation_result() for t in exp_a.sub_tasks],
        gate_summary=mods.feedback.gate_outcome_text(exp_a),
        crypto_eval=exp_a.crypto_eval,
        summary_text=exp_a.stdout,
    )
    _assert_clean(system_prompt)
    _assert_clean(user_prompt)
    assert "Observations" in system_prompt and "New Hypothesis" in system_prompt
    assert "0.0180" in user_prompt  # valid ic_mean rendered
    assert "0.0200" in user_prompt  # train ic_mean rendered
