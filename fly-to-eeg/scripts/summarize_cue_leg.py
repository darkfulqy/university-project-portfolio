#!/usr/bin/env python3
"""Descriptive analysis of saved, frozen cue-leg evaluations; no fitting."""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cue_leg/v1_0_pilot"


def main():
    path = OUT / "training_summary.json"
    summary = json.loads(path.read_text())
    if summary["status"] != "completed_engineering_pilot":
        raise ValueError("Completed training required")
    config = summary["configuration"]
    bank = summary["test_response_bank"]
    stages = ("before", "after", "noncontingent_expected_reward_control")
    per_cue = []
    for cue_index, (cue, target) in enumerate(zip(config["cue_names"], config["assigned_target_legs"])):
        entry = {"cue_id": cue, "assigned_target_leg": target, "stages": {}}
        for stage in stages:
            # Rows are parameter ensembles; policy replicates share these rows.
            by_ensemble = []
            for ensemble in bank:
                policy_values = []
                for policy in summary["policies"]:
                    probabilities = policy[stage]["probabilities"][cue_index]
                    weighted = {key: 0.0 for key in ("correct", "wrong_single_leg", "omission", "multiple_legs", "other_failure", "blank_false_alarm")}
                    for probability, metrics in zip(probabilities, ensemble):
                        active = metrics["observed_active_legs"]
                        success = metrics["success_by_cue"][cue_index]
                        if success:
                            weighted["correct"] += probability
                        elif target is None:
                            weighted["blank_false_alarm"] += probability
                        elif not active:
                            weighted["omission"] += probability
                        elif len(active) > 1:
                            weighted["multiple_legs"] += probability
                        elif active[0] != target:
                            weighted["wrong_single_leg"] += probability
                        else:
                            weighted["other_failure"] += probability
                    policy_values.append(weighted)
                by_ensemble.append({key: float(np.mean([p[key] for p in policy_values])) for key in policy_values[0]})
            entry["stages"][stage] = {
                "mean": {key: float(np.mean([e[key] for e in by_ensemble])) for key in by_ensemble[0]},
                "by_parameter_ensemble": by_ensemble,
            }
        per_cue.append(entry)
    measured = []
    for e, seed in enumerate(config["test_ensemble_seeds"]):
        for policy in summary["policies"]:
            for cue_index in (0, 1):
                action = policy["after"]["greedy_actions"][cue_index]
                metrics = bank[e][action]
                measured.append({
                    "source_seed": seed, "policy_seed": policy["seed"],
                    "cue_id": config["cue_names"][cue_index],
                    "target_lift_mm": metrics["peak_foreleg_lift_mm"][cue_index],
                    "non_target_displacement_mm": metrics["non_target_max_displacement_mm"][cue_index],
                    "latency_s": metrics["latency_s"][cue_index],
                    "return_error_mm": max(metrics["return_error_mm"]),
                })
    ranges = {}
    for key in ("target_lift_mm", "non_target_displacement_mm", "latency_s", "return_error_mm"):
        values = [m[key] for m in measured if m[key] is not None]
        ranges[key] = [min(values), max(values)] if values else None
    output = dict(
        analysis="descriptive_probability_weighted_outcomes_no_fitting",
        training_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        independent_test_source_parameter_ensembles=config["test_ensemble_seeds"],
        policy_seeds=config["policy_seeds"], per_cue=per_cue,
        greedy_physical_ranges=ranges, greedy_physical_records=measured,
        interpretation="Exact probability-weighted outcomes on a fixed measured response bank, not sampled animal success rates; policy replicates are not additional source models",
    )
    destination = OUT / "descriptive_outcomes.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    for row in per_cue:
        print(row["cue_id"], {s: row["stages"][s]["mean"] for s in stages})
    print("GREEDY_PHYSICAL_RANGES", ranges)


if __name__ == "__main__":
    main()
