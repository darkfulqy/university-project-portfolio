"""Hypothesis generation and hypothesis-to-experiment conversion for the crypto 5-minute factor loop (DESIGN section 7).

Both classes reuse the upstream LLM plumbing (``LLMHypothesisGen.gen`` / ``LLMHypothesis2Experiment.convert``) and only
replace the scenario-specific context: crypto prompt templates, a crypto RAG string, a compact summary of the accepted
factor library and real de-duplication of proposed factors against that library.
"""

from __future__ import annotations

import json
from typing import Any, Tuple

from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask
from rdagent.core.proposal import Hypothesis, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.crypto.conf import CRYPTO_EVAL_SETTINGS
from rdagent.scenarios.crypto.experiment import CryptoFactorExperiment
from rdagent.scenarios.qlib.proposal.factor_proposal import (
    QlibFactorHypothesis2Experiment,
    QlibFactorHypothesisGen,
)
from rdagent.utils.agent.tpl import T

PROMPTS = "scenarios.crypto.prompts"
NO_HISTORY = "No previous hypothesis and feedback available since it's the first round."

RAG_ALWAYS = (
    "Hard rules: a factor value at bar t may only use panel rows with datetime <= t (rolling windows over past bars, "
    "shift(k) with k >= 1, no centered windows, no negative shifts). Prefer low turnover: a high rank_autocorr_1 "
    "(slowly changing cross-sectional ranking, e.g. smoothed with an EWM or a rolling mean over several bars) is "
    "needed to survive the 10 bp fee + 5 bp slippage model. Propose 1-3 factors per round and give every window in "
    "bars."
)
RAG_EARLY = (
    "Stage: early exploration. Try simple microstructure factors first, one mechanism per round: short-horizon "
    "reversal and momentum (1-48 bars), taker-buy imbalance ($taker_buy_volume / $volume or $taker_buy_amount / "
    "$amount, optionally smoothed), volume / amount shocks versus a rolling mean, realized volatility over a window of "
    "bars, range or ATR normalisation of the last move, return relative to BTCUSDT, average trade size "
    "($amount / $trade_count) and time-of-day (UTC hour) effects."
)
RAG_LATE = (
    "Stage: refinement. The simple mechanisms have been explored; now try interactions and regime-conditioned factors "
    "(e.g. imbalance conditioned on a volatility regime, momentum conditioned on a volume shock, reversal only when "
    "the range is wide, BTC-relative signals conditioned on BTC's own momentum) and combinations of the accepted "
    "library directions that keep |correlation| with existing factors low."
)
EARLY_ROUNDS = 8


def primary_horizon_key(ev: dict | None = None) -> str:
    """Return the metrics key of the primary horizon (e.g. ``"15m"``) from an eval dict or the settings."""
    cfg = (ev or {}).get("cfg") or {}
    minutes = cfg.get("primary_horizon_minutes", CRYPTO_EVAL_SETTINGS.primary_horizon_minutes)
    return f"{int(minutes)}m"


def accepted_library_names(trace: Trace) -> list[str]:
    """Names of every factor accepted into the library so far (from ``crypto_accepted`` of accepted experiments)."""
    names: list[str] = []
    for exp, feedback in trace.hist:
        if not feedback:
            continue
        for name in getattr(exp, "crypto_accepted", None) or []:
            if name not in names:
                names.append(name)
    return names


def build_library_summary(trace: Trace) -> dict[str, Any]:
    """Collect the compact SOTA library summary rendered by ``sota_library_summary``.

    Returns ``{"library": [rows], "composite": metrics | None, "primary_horizon_key": str}`` where each row holds the
    factor name and its valid-segment metrics at the primary horizon (``ic_mean``, ``ic_ir``, ``ic_tstat``,
    ``rank_autocorr_1``).  Only ``train``/``valid`` data is touched; ``test`` metrics are never read.
    """
    rows: dict[str, dict[str, Any]] = {}
    composite: dict | None = None
    ph = primary_horizon_key(None)
    for exp, feedback in trace.hist:
        if not feedback:
            continue
        ev = getattr(exp, "crypto_eval", None) or {}
        accepted = getattr(exp, "crypto_accepted", None) or ev.get("accepted") or []
        if not accepted:
            continue
        ph = primary_horizon_key(ev)
        factors = ev.get("factors") or {}
        for name in accepted:
            metrics = (factors.get(name) or {}).get("metrics") or {}
            valid = (metrics.get("valid") or {}).get(ph) or {}
            rows[name] = {
                "name": name,
                "ic_mean": valid.get("ic_mean"),
                "ic_ir": valid.get("ic_ir"),
                "ic_tstat": valid.get("ic_tstat"),
                "rank_autocorr_1": valid.get("rank_autocorr_1"),
            }
        composite_after = ((ev.get("composite_after") or {}).get("valid") or {}).get(ph)
        if composite_after:
            composite = composite_after
    return {"library": list(rows.values()), "composite": composite, "primary_horizon_key": ph}


def render_library_summary(trace: Trace) -> str:
    """Render the ``sota_library_summary`` template for the trace."""
    return T(f"{PROMPTS}:sota_library_summary").r(**build_library_summary(trace))


def crypto_rag(trace: Trace) -> str:
    """Stage-dependent guidance (DESIGN 7) plus the hard rules.

    The accepted library summary is passed separately under ``sota_hypothesis_and_feedback`` (the key the upstream
    ``hypothesis_gen`` template renders), so it is not duplicated here.
    """
    stage = RAG_EARLY if len(trace.hist) < EARLY_ROUNDS else RAG_LATE
    return f"{stage}\n{RAG_ALWAYS}"


class CryptoFactorHypothesisGen(QlibFactorHypothesisGen):
    """Hypothesis generator using the crypto prompt templates and library summary."""

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        has_hist = len(trace.hist) > 0
        hypothesis_and_feedback = T(f"{PROMPTS}:hypothesis_and_feedback").r(trace=trace) if has_hist else NO_HISTORY
        last_hypothesis_and_feedback = (
            T(f"{PROMPTS}:last_hypothesis_and_feedback").r(experiment=trace.hist[-1][0], feedback=trace.hist[-1][1])
            if has_hist
            else NO_HISTORY
        )
        library_summary = render_library_summary(trace)
        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "sota_library_summary": library_summary,
            # upstream LLMHypothesisGen.gen only forwards hypothesis_and_feedback / last_... / sota_... / RAG, so the
            # library summary travels under the key the template renders ("SOTA trail" block)
            "sota_hypothesis_and_feedback": library_summary,
            "RAG": crypto_rag(trace),
            "hypothesis_output_format": T(f"{PROMPTS}:factor_hypothesis_output_format").r(),
            "hypothesis_specification": T(f"{PROMPTS}:factor_hypothesis_specification").r(),
        }
        return context_dict, True


class CryptoFactorHypothesis2Experiment(QlibFactorHypothesis2Experiment):
    """Turn a hypothesis into a ``CryptoFactorExperiment`` whose tasks are not already in the accepted library."""

    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]:
        scenario = trace.scen.get_scenario_all_desc()
        experiment_output_format = T(f"{PROMPTS}:factor_experiment_output_format").r()

        if len(trace.hist) == 0:
            hypothesis_and_feedback = NO_HISTORY
        else:
            specific_trace = Trace(trace.scen)
            for exp, fb in trace.hist:
                action = getattr(getattr(exp, "hypothesis", None), "action", "factor")
                if action == "factor":
                    specific_trace.hist.append((exp, fb))
            hypothesis_and_feedback = (
                T(f"{PROMPTS}:hypothesis_and_feedback").r(trace=specific_trace)
                if specific_trace.hist
                else "No previous hypothesis and feedback available."
            )

        # The upstream hypothesis2experiment templates render neither RAG nor target_list, so the hard rules and the
        # accepted library summary ("do NOT re-implement") are folded into hypothesis_and_feedback, which is rendered.
        # Both keys must still exist: LLMHypothesis2Experiment.convert reads them unconditionally.
        library_block = f"{RAG_ALWAYS}\n{render_library_summary(trace)}"
        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": f"{hypothesis_and_feedback}\n\n{library_block}",
            "experiment_output_format": experiment_output_format,
            "target_list": [],
            "RAG": library_block,
        }, True

    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> CryptoFactorExperiment:
        response_dict = json.loads(response)
        based_experiments = [CryptoFactorExperiment(sub_tasks=[])] + [
            t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
        ]
        library: set[str] = set()
        for based_exp in based_experiments:
            library.update(getattr(based_exp, "crypto_accepted", None) or [])

        tasks: list[FactorTask] = []
        seen: set[str] = set()
        for factor_name, spec in response_dict.items():
            name = str(factor_name).strip()
            spec = spec if isinstance(spec, dict) else {}
            if name in library:
                logger.warning(f"Dropping proposed factor `{name}`: already in the accepted library.")
                continue
            if name in seen:
                continue
            seen.add(name)
            tasks.append(
                FactorTask(
                    factor_name=name,
                    factor_description=str(spec.get("description", "")),
                    factor_formulation=str(spec.get("formulation", "")),
                    variables=spec.get("variables") if isinstance(spec.get("variables"), dict) else {},
                )
            )
        if not tasks:
            logger.warning("All proposed factors were duplicates of the accepted library; the experiment has no task.")

        exp = CryptoFactorExperiment(tasks, hypothesis=hypothesis)
        exp.based_experiments = based_experiments
        return exp
