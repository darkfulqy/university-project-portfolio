#!/usr/bin/env python3
"""Reviewed local simulation/learning entry point. No network or cloud calls.

Stages: screen calibration ensemble, train with locked actions/solver, render.
Source graph is actual Male CNS v1.0. Learned policy is explicitly external to
the fixed recurrent graph. Physical responses are never assigned from cues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from flycue.connectome import load_connectome
from flycue.dynamics import RateNetwork
from flycue.learning import CuePolicy, train_policy, make_yoked_reward_bank
from flycue.readout import MotorReadout, candidate_inputs


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nuisance(config, seed):
    if seed == config["calibration_seed"]:
        return {"gain": 1.0, "onset_jitter_s": 0.0}
    rng = np.random.default_rng(seed + 7013)
    n = config["nuisance"]
    return {"gain": float(rng.uniform(*n["input_gain_range"])),
            "onset_jitter_s": float(rng.choice(n["onset_jitter_s"]))}


def neural_trial(graph, readout, config, action, seed, dt):
    nc = config["neural"]
    model = RateNetwork(graph.W, graph.sizes, seed=seed, b=nc["synaptic_scaling_b"])
    noise = nuisance(config, seed)
    onset = nc["onset_s"] + noise["onset_jitter_s"]
    offset = nc["offset_s"] + noise["onset_jitter_s"]
    if action["body_id"] is None:
        indices, stim = [], None
    else:
        indices = [int(np.where(graph.ids == action["body_id"])[0][0])]
        magnitude = action["magnitude_per_size"] * graph.sizes[indices[0]] * noise["gain"]
        stim = lambda t: np.array([magnitude if onset <= t < offset else 0.0])
    result = model.simulate(duration_s=nc["duration_s"], dt_s=dt,
                            stimulation_indices=indices, stimulation=stim,
                            monitor_indices=readout.monitor_indices,
                            output_every=int(round(nc["output_dt_s"] / dt)),
                            solver=nc["solver"])
    drive = readout.transform(result.activity, result.monitor_indices)
    return result, drive, model.parameters, {**noise, "onset_s": onset, "offset_s": offset}


def body_trial(body, times, drive, *, capture=False):
    observations = [body.reset(seed=0)]
    frames = []
    for k, delta in enumerate(np.diff(times)):
        observations.append(body.step(drive[k], float(delta)))
        if capture and k % 4 == 0:
            frames.append(body.render_image("front", width=800, height=600))
    fields = ("foreleg_lift_mm", "tip_lift_mm", "tip_body_mm", "tip_world_mm",
              "tip_head_mm", "joint_angles_rad", "joint_velocities_rad_s",
              "joint_targets_rad", "contact_forces_flygym", "thorax_world_mm")
    return {key: np.stack([o[key] for o in observations]) for key in fields}, frames


def movement_metrics(config, times, trajectory, onset):
    criteria = config["behavior_criteria"]
    lifts = trajectory["foreleg_lift_mm"]
    start, end = criteria["response_window_after_onset_s"]
    window = (times >= onset + start) & (times <= onset + end)
    peak = np.maximum(lifts[window].max(axis=0), 0)
    displacement = np.linalg.norm(trajectory["tip_body_mm"]-trajectory["tip_body_mm"][0], axis=-1)
    return_error = displacement[times >= times[-1]-0.1][:, [0,3]].max(axis=0)
    prepeak = displacement[times < onset][:, [0,3]].max(axis=0)
    non_target = [float(displacement[:, [j for j in range(6) if j != target]].max())
                  for target in (0, 3)]
    latency = []
    for side in (0, 1):
        hits = np.flatnonzero((lifts[:, side] >= criteria["target_lift_mm"]) & (times >= onset))
        latency.append(None if not len(hits) else float(times[hits[0]]-onset))
    successes = [bool(peak[s] >= criteria["target_lift_mm"]
                      and non_target[s] < criteria["non_target_max_displacement_mm"]
                      and return_error.max() < criteria["return_tolerance_mm"]
                      and prepeak.max() < criteria["blank_max_displacement_mm"]) for s in (0, 1)]
    blank = bool(displacement.max() < criteria["blank_max_displacement_mm"])
    active = [leg for leg, val in zip(("LF","LM","LH","RF","RM","RH"),displacement.max(axis=0))
              if val >= criteria["target_lift_mm"]]
    rewards = [float(np.clip(peak[s]/0.6, 0, 1)
                     - np.clip(non_target[s]/0.3, 0, 1)
                     - 0.5*np.clip(return_error.max()/criteria["return_tolerance_mm"], 0, 2)
                     - np.clip(prepeak.max()/criteria["blank_max_displacement_mm"], 0, 1))
               for s in (0, 1)]
    rewards.append(1.0 if blank else -1.0)
    return dict(peak_foreleg_lift_mm=peak.tolist(), return_error_mm=return_error.tolist(),
                pre_cue_peak_mm=prepeak.tolist(), latency_s=latency,
                observed_active_legs=active, success_by_cue=successes+[blank],
                non_target_max_displacement_mm=non_target,
                reward_by_cue=rewards,
                max_other_leg_displacement_mm=float(displacement[:, [1,2,4,5]].max()))


def screen(graph, readout, config, out):
    from flycue.body import ForelegBody
    if (out / "screen.json").exists():
        raise FileExistsError("Screen already exists; preserve it and choose a new output directory for a new pilot")
    inputs = candidate_inputs(graph, readout, config["candidate_inputs_per_side"])
    dt = config["neural"]["initial_dt_s"]
    seed = config["calibration_seed"]
    records = []
    for i in inputs:
        for mag in config["input_magnitudes_per_size"]:
            action = dict(body_id=int(graph.ids[i]), type=graph.nodes[i]["type"],
                          magnitude_per_size=mag)
            result, drive, _, _ = neural_trial(graph, readout, config, action, seed, dt)
            peaks = drive.max(axis=0)
            record = dict(action=action, peak_drive=peaks.tolist(), diagnostics=result.diagnostics)
            records.append(record)
            print("SCREEN", action, "peak_drive", np.round(peaks, 3).tolist(), flush=True)
    chosen = []
    for side in (0, 1):
        ordered = sorted(records, key=lambda r: r["peak_drive"][side]-r["peak_drive"][1-side], reverse=True)
        for record in ordered[:2]:
            if record["action"] not in chosen:
                chosen.append(record["action"])
    actions = [dict(body_id=None, type="no_neural_input", magnitude_per_size=0)] + chosen
    convergence = []
    while True:
        errors = []
        for action in actions[1:]:
            coarse, _, _, _ = neural_trial(graph, readout, config, action, seed, dt)
            fine, _, _, _ = neural_trial(graph, readout, config, action, seed, dt/2)
            error = float(np.sqrt(np.mean((coarse.activity-fine.activity)**2)) /
                          max(np.sqrt(np.mean(fine.activity**2)), 0.001))
            errors.append(error)
        convergence.append(dict(dt_s=dt, half_dt_s=dt/2, relative_rms_errors=errors))
        print("CONVERGENCE", convergence[-1], flush=True)
        if max(errors, default=0) <= config["neural"]["convergence_relative_rms_tolerance"]:
            break
        dt /= 2
        if dt < config["neural"]["minimum_dt_s"]:
            dump(out/"screen_failure.json", dict(reason="neural_dt_convergence_failed", checks=convergence))
            raise RuntimeError("Neural time step failed convergence; no training started")
    body = ForelegBody()
    calibrated = []
    try:
        for action in actions:
            result, drive, _, noise = neural_trial(graph, readout, config, action, seed, dt)
            trajectory, _ = body_trial(body, result.time_s, drive)
            metrics = movement_metrics(config, result.time_s, trajectory, noise["onset_s"])
            calibrated.append(dict(action=action, metrics=metrics))
            print("CALIBRATION", action, metrics, flush=True)
        ready = all(any(r["metrics"]["success_by_cue"][cue] for r in calibrated) for cue in range(3))
        report = dict(status="ready_for_training" if ready else "controllability_failed",
                      actions=actions, resolved_dt_s=dt, convergence=convergence,
                      neural_screen=records, calibration=calibrated,
                      body_metadata=body.metadata(), readout=readout.metadata(graph),
                      config_sha256=sha(ROOT/"configs/cue_leg.json"),
                      graph_manifest_sha256=sha(ROOT/config["graph_directory"]/"manifest.json"),
                      selection_uses_only_calibration_seed=seed)
        dump(out/"screen.json", report)
        if not ready:
            raise RuntimeError("Specified unilateral lift not controllable with candidate inputs; no training started")
    finally:
        body.close()


def train(graph, readout, config, out):
    from flycue.body import ForelegBody
    screen_report = json.loads((out/"screen.json").read_text())
    if screen_report["status"] != "ready_for_training":
        raise RuntimeError("Controllability screen must pass first")
    if screen_report["config_sha256"] != sha(ROOT/"configs/cue_leg.json"):
        raise RuntimeError("Config changed after screening; create a new experiment directory")
    if screen_report["graph_manifest_sha256"] != sha(ROOT/config["graph_directory"]/"manifest.json"):
        raise RuntimeError("Graph changed after screening")
    if (out/"training_summary.json").exists():
        raise FileExistsError("Completed training output exists; preserve it")
    seeds = config["training_ensemble_seeds"] + config["test_ensemble_seeds"]
    if len(set(seeds + [config["calibration_seed"]])) != len(seeds)+1:
        raise ValueError("Calibration/train/test parameter groups must be disjoint")
    actions = screen_report["actions"]
    banks = {"train": [], "test": []}
    body = ForelegBody()
    body_metadata = body.metadata()
    start = time.monotonic()
    try:
        for phase, ensemble_seeds in (("train", config["training_ensemble_seeds"]),
                                      ("test", config["test_ensemble_seeds"])):
            for seed in ensemble_seeds:
                rows = []
                for action_index, action in enumerate(actions):
                    result, drive, params, noise = neural_trial(graph, readout, config, action, seed, screen_report["resolved_dt_s"])
                    trajectory, _ = body_trial(body, result.time_s, drive)
                    metrics = movement_metrics(config, result.time_s, trajectory, noise["onset_s"])
                    trial_id = f"{phase}_ensemble_{seed}_action_{action_index}"
                    path = out/"trials"/trial_id
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path.with_suffix(".npz"), time_s=result.time_s,
                                        motor_activity_au=result.activity, motor_drive=drive,
                                        monitor_body_ids=graph.ids[result.monitor_indices], **trajectory)
                    ppath = out/"parameters"/f"ensemble_{seed}.npz"
                    ppath.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(ppath, body_ids=graph.ids, sizes=graph.sizes,
                                        gain=params.gain, threshold=params.threshold,
                                        rmax=params.rmax, tau_s=params.tau_s)
                    metadata = dict(trial_id=trial_id, parent_trial=trial_id,
                                    split=phase, split_group=f"{phase}_source_model_{seed}",
                                    connectome_version=config["dataset"],
                                    graph_manifest_sha256=screen_report["graph_manifest_sha256"],
                                    source_model_seed=seed, parameter_ensemble=str(ppath.relative_to(ROOT)),
                                    all_cell_identifiers=graph.ids.tolist(),
                                    stimulus=dict(neural_action=action, cue_encoding=config["cue_encoding"],
                                                  cue_id=None, note="cue-independent reset response bank; policy episode selects action"),
                                    nuisance_parameters=noise, observer_geometry=body_metadata,
                                    sampling_rate_hz=1/config["neural"]["output_dt_s"],
                                    units=dict(neural_activity="arbitrary_units", time="s", joint_angle="rad", foot_position="mm"),
                                    source_neural_metadata=result.metadata,
                                    dynamics_diagnostics=result.diagnostics,
                                    readout=readout.metadata(graph), metrics=metrics,
                                    body_reset_seed=0, direct_motor_prompt=False)
                    dump(path.with_suffix(".json"), metadata)
                    rows.append(metrics)
                    print("TRIAL", trial_id, metrics["observed_active_legs"], np.round(metrics["peak_foreleg_lift_mm"],3).tolist(), flush=True)
                banks[phase].append(rows)
    finally:
        body.close()
    rewards = np.array([[r["reward_by_cue"] for r in rows] for rows in banks["train"]]).transpose(0,2,1)
    test_success = np.array([[r["success_by_cue"] for r in rows] for rows in banks["test"]]).transpose(0,2,1)
    policies = []
    evaluation_records = []
    for policy_seed in config["policy_seeds"]:
        initial = CuePolicy.initialize(3, len(actions), policy_seed)
        learned, history = train_policy(rewards, seed=policy_seed, **config["training"])
        control, _ = train_policy(make_yoked_reward_bank(rewards), seed=policy_seed, **config["training"])
        # Evaluate exact expected success from frozen policies on physically measured test responses.
        def evaluate(policy):
            probs = np.stack([policy.probabilities(c) for c in range(3)])
            expected = np.einsum("eca,ca->ec",test_success.astype(float),probs)
            greedy = np.argmax(probs,axis=1)
            greedy_success = np.stack([test_success[:,c,a] for c,a in enumerate(greedy)],axis=1)
            return dict(probabilities=probs.tolist(), expected_success_by_ensemble_cue=expected.tolist(),
                        expected_success_mean=float(expected.mean()), greedy_actions=greedy.tolist(),
                        greedy_success_by_ensemble_cue=greedy_success.tolist(),
                        greedy_success_mean=float(greedy_success.mean()))
        entry = dict(seed=policy_seed, before=evaluate(initial), after=evaluate(learned),
                     noncontingent_expected_reward_control=evaluate(control))
        policies.append(entry)
        dump(out/"policies"/f"policy_{policy_seed}.json", dict(logits=learned.logits.tolist(),
             learning_locus=config["learning_locus"], action_bank=actions,
             cue_names=config["cue_names"], target_assignment_used_only_for_reward=config["assigned_target_legs"]))
        for ensemble_index, neural_seed in enumerate(config["test_ensemble_seeds"]):
            for cue_index, cue_id in enumerate(config["cue_names"]):
                probs = learned.probabilities(cue_index)
                action_index = int(np.argmax(probs))
                evaluation_records.append(dict(
                    cue_id=cue_id, assigned_target_leg=config["assigned_target_legs"][cue_index],
                    policy_seed=policy_seed, neural_ensemble_seed=neural_seed,
                    split="test", split_group=f"test_source_model_{neural_seed}",
                    policy_probabilities=probs.tolist(), selection="greedy_frozen_policy",
                    selected_neural_action=action_index,
                    parent_trial=f"test_ensemble_{neural_seed}_action_{action_index}",
                    observed_active_legs=banks["test"][ensemble_index][action_index]["observed_active_legs"],
                    success=bool(test_success[ensemble_index,cue_index,action_index]),
                    reinforcement_enabled=False, learning_updates_enabled=False,
                    externally_supplied_motor_target=False))
        # Each episode points to a real measured parent response; reused source trials stay in the same split.
        for h in history:
            h["cue_id"] = config["cue_names"][h["cue"]]
            h["assigned_target_leg"] = config["assigned_target_legs"][h["cue"]]
            h["parent_trial"] = f"train_ensemble_{config['training_ensemble_seeds'][h['ensemble_index']]}_action_{h['action']}"
            h["observed_active_legs"] = banks["train"][h["ensemble_index"]][h["action"]]["observed_active_legs"]
        dump(out/"policies"/f"history_{policy_seed}.json",history)
    summary = dict(status="completed_engineering_pilot", task=config["task"], dataset=config["dataset"],
                   configuration=config, actions=actions, policies=policies,
                   mean_before=float(np.mean([p["before"]["expected_success_mean"] for p in policies])),
                   mean_after=float(np.mean([p["after"]["expected_success_mean"] for p in policies])),
                   mean_noncontingent_control=float(np.mean([p["noncontingent_expected_reward_control"]["expected_success_mean"] for p in policies])),
                   independent_test_parameter_ensembles=len(config["test_ensemble_seeds"]),
                   independent_policy_seeds=len(policies), test_response_bank=banks["test"],
                   wall_time_s=time.monotonic()-start,
                   limitations=["symbolic cues, no retina/visual perception model",
                     "learning in added policy; recurrent connectome fixed",
                     "incoming-contact size proxy, not neuronal volume",
                     "unknown-MN outgoing central contacts omitted",
                     "female body template coupled to male neural graph",
                     "suspended foreleg assay, fixed servo readout, no empirical muscle calibration",
                     "no physiological validation, no topology superiority or MI transfer tested"],
                   source_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                     sorted((ROOT/"src/flycue").glob("*.py"))+[Path(__file__).resolve()]},
                   graph_manifest_sha256=screen_report["graph_manifest_sha256"])
    dump(out/"evaluation_trials.json",evaluation_records)
    dump(out/"training_summary.json",summary)
    print("RESULT", {k:summary[k] for k in ("mean_before","mean_after","mean_noncontingent_control","wall_time_s")},flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=["screen","train"])
    args=ap.parse_args()
    config=json.loads((ROOT/"configs/cue_leg.json").read_text())
    out=ROOT/config["output_directory"]
    out.mkdir(parents=True,exist_ok=True)
    graph=load_connectome(ROOT/config["graph_directory"])
    if graph.manifest["dataset"] != config["dataset"]:
        raise ValueError("Wrong source dataset")
    readout=MotorReadout.from_graph(graph,config["readout_rate_scale_au"])
    (screen if args.stage=="screen" else train)(graph,readout,config,out)


if __name__=="__main__":
    main()
