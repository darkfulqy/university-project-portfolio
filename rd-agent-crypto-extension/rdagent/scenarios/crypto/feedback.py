"""Feedback (SOTA decision) for the crypto 5-minute factor loop (DESIGN section 9).

The decision is deterministic: an experiment is "SOTA" iff at least one of its factors passed the acceptance gate
(``exp.crypto_eval["accepted"]`` non-empty).  The LLM only writes the narrative fields (observations, hypothesis
evaluation, new hypothesis, reasoning); any failure of the LLM call leaves the decision untouched.
"""

from __future__ import annotations

import json
from typing import Any

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T

PROMPTS = "scenarios.crypto.prompts"
LLM_ERROR_NOTE = "LLM feedback unavailable"


def gate_decision(exp: Experiment) -> bool:
    """Deterministic SOTA decision: True iff the runner accepted at least one factor of ``exp``."""
    ev = getattr(exp, "crypto_eval", None)
    return bool(ev and ev.get("accepted"))


def gate_outcome_text(exp: Experiment, exception: Exception | None = None) -> str:
    """One line per sub-task factor: ``name: accepted`` or ``name: rejected: <gate reasons>``."""
    ev = getattr(exp, "crypto_eval", None)
    task_names = [getattr(t, "factor_name", getattr(t, "name", str(t))) for t in getattr(exp, "sub_tasks", [])]
    if not ev:
        note = f"experiment failed: {exception!s}" if exception is not None else "no evaluation result"
        if not task_names:
            return f"no factor evaluated ({note})"
        return "\n".join(f"{name}: rejected: not evaluated ({note})" for name in task_names)
    factors: dict[str, dict[str, Any]] = ev.get("factors") or {}
    names = list(task_names) + [n for n in factors if n not in task_names]
    lines: list[str] = []
    for name in names:
        info = factors.get(name)
        if info is None:
            lines.append(f"{name}: rejected: not evaluated (no usable output column)")
        elif info.get("accepted"):
            lines.append(f"{name}: accepted")
        else:
            reasons = info.get("gate_reasons") or ["gate failed"]
            lines.append(f"{name}: rejected: " + "; ".join(str(r) for r in reasons))
    if not lines:
        lines.append("no factor evaluated")
    return "\n".join(lines)


class CryptoFactorExperiment2Feedback(Experiment2Feedback):
    """Gate-driven feedback with an LLM-written narrative."""

    def generate_feedback(
        self, exp: Experiment, trace: Trace, exception: Exception | None = None
    ) -> HypothesisFeedback:  # noqa: ARG002
        decision = gate_decision(exp)
        gate_text = gate_outcome_text(exp, exception)
        logger.info(f"Generating crypto feedback (gate decision={decision})...")

        hypothesis = getattr(exp, "hypothesis", None)
        hypothesis_text = getattr(hypothesis, "hypothesis", "") if hypothesis is not None else ""
        ev = getattr(exp, "crypto_eval", None)

        try:
            task_details = [
                task.get_task_information_and_implementation_result() for task in getattr(exp, "sub_tasks", [])
            ]
            sys_prompt = T(f"{PROMPTS}:factor_feedback_generation.system").r(scenario=self.scen.get_scenario_all_desc())
            usr_prompt = T(f"{PROMPTS}:factor_feedback_generation.user").r(
                hypothesis_text=hypothesis_text,
                task_details=task_details,
                gate_summary=gate_text,
                crypto_eval=ev,
                summary_text=getattr(exp, "stdout", "") or "",
            )
            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=usr_prompt,
                system_prompt=sys_prompt,
                json_mode=True,
                json_target_type=dict[str, str],
            )
            response_json = json.loads(response)
            if not isinstance(response_json, dict):
                raise ValueError(f"unexpected feedback payload type {type(response_json).__name__}")
            observations = str(response_json.get("Observations", "No observations provided"))
            hypothesis_evaluation = str(response_json.get("Feedback for Hypothesis", "No feedback provided"))
            new_hypothesis = str(response_json.get("New Hypothesis", "No new hypothesis provided"))
            reasoning = str(response_json.get("Reasoning", "No reasoning provided"))
        except Exception as e:  # noqa: BLE001 - the gate decision must survive any LLM/template failure
            note = f"{LLM_ERROR_NOTE}: {type(e).__name__}: {e!s}"[:300]
            logger.warning(note)
            observations = hypothesis_evaluation = new_hypothesis = reasoning = note

        reason = f"{gate_text}\nReasoning: {reasoning}"
        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )
