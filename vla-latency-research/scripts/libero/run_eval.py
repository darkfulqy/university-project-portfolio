"""Paired LIBERO rollouts via the Phase-0 websocket server (Python 3.8)."""

import argparse
from collections import deque
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import (DUMMY_ACTION, MAX_STEPS, add_identity_args, keyed_seed, make_env,
                     parse_indices, query_paired, run_jsonl, runtime_metadata, seed_globals, server_contract)


def run(args, writer):
    runtime = runtime_metadata(args)
    from libero.libero import benchmark
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    contract = server_contract(client)
    if args.expected_nfe is not None and contract["num_steps"] != args.expected_nfe:
        raise RuntimeError("Connected to wrong NFE server")
    if args.checkpoint_sha and contract["runtime"]["checkpoint"]["sha256"] != args.checkpoint_sha.lower():
        raise RuntimeError("Server checkpoint hash does not match --checkpoint-sha")
    if not contract["runtime"]["frozen_runtime_sources"] and not args.allow_source_mismatch:
        raise RuntimeError("Server does not use the pinned source commits")
    if args.replan_steps > contract["action_horizon"]:
        raise ValueError("replan_steps must not exceed the returned action horizon")
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task_ids = parse_indices(args.tasks, suite.n_tasks)
    max_steps = args.max_steps if args.max_steps is not None else MAX_STEPS[args.task_suite]
    episode_plan = []
    for task_id in task_ids:
        initial_states = suite.get_task_init_states(task_id)
        if args.init_indices is None:
            count = args.episodes_per_task if args.episodes_per_task is not None else 50
            if count > len(initial_states):
                raise ValueError("Not enough initial states; no modulo reuse is allowed")
            indices = list(range(count))
        else:
            indices = parse_indices(args.init_indices, len(initial_states))
            if args.episodes_per_task is not None and args.episodes_per_task != len(indices):
                raise ValueError("episodes_per_task must equal the explicit initial-state index count")
        episode_plan.append((task_id, indices, initial_states))
    writer.write("metadata", runtime=runtime, server=contract, task_suite=args.task_suite,
                 task_ids=task_ids, seed=args.seed, replan_steps=args.replan_steps, settle_steps=args.settle_steps,
                 max_steps=max_steps, initial_state_indices={str(task_id): indices for task_id, indices, _ in episode_plan},
                 render_resolution=256, input_resolution=224,
                 pairing="episode seed from run/suite/task/init; flow and env RNG keyed by absolute simulator step",
                 policy_latency_boundary="CPU prepared request -> msgpack/network/server infer -> decoded CPU response; image preprocessing separate",
                 gpu_active_seconds="not_measured", first_call_may_compile=True,
                 episode_limit_override=args.max_steps is not None, errors="record then abort; never silently drop or count as task failure")
    successes = completed = attempted = 0
    total_wall = time.perf_counter()
    for task_id, indices, initial_states in episode_plan:
        task = suite.get_task(task_id)
        env = None
        try:
            for initial_index in indices:
                attempted += 1
                episode_seed = keyed_seed(args.seed, args.task_suite, task_id, initial_index, "episode")
                latencies, calls = [], []
                steps = settled = 0
                success = False
                episode_error = None
                episode_started = time.perf_counter()
                rollout_started = None
                preparation_seconds = environment_seconds = 0.0
                try:
                    if env is None:
                        env = make_env(task, episode_seed)
                    seed_globals(episode_seed)
                    env.seed(episode_seed)
                    env.reset()
                    observation = env.set_init_state(initial_states[initial_index])
                    for absolute_step in range(args.settle_steps):
                        seed_globals(keyed_seed(episode_seed, absolute_step, "environment"))
                        observation, _, _, _ = env.step(DUMMY_ACTION)
                        settled += 1
                    action_queue = deque()
                    rollout_started = time.perf_counter()
                    while steps < max_steps:
                        absolute_step = args.settle_steps + steps
                        if not action_queue:
                            query_started = time.perf_counter()
                            try:
                                actions, elapsed, server_timing, noise_seed = query_paired(
                                    client, observation, task.language, contract, episode_seed, absolute_step)
                            except Exception as error:
                                if hasattr(error, "phase0_latency_ms"):
                                    latencies.append(error.phase0_latency_ms)
                                    calls.append({"absolute_step": absolute_step, "latency_ms": error.phase0_latency_ms,
                                                  "status": "error"})
                                raise
                            preparation_seconds += time.perf_counter() - query_started - elapsed / 1000
                            latencies.append(elapsed)
                            calls.append({"absolute_step": absolute_step, "latency_ms": elapsed,
                                          "server_timing": server_timing, "flow_noise_seed": noise_seed, "status": "ok"})
                            if len(actions) < args.replan_steps:
                                raise ValueError("Policy returned fewer actions than replan_steps")
                            action_queue.extend(actions[:args.replan_steps])
                        action = action_queue.popleft()
                        seed_globals(keyed_seed(episode_seed, absolute_step, "environment"))
                        step_started = time.perf_counter()
                        try:
                            observation, _, done, _ = env.step(action.tolist())
                        finally:
                            environment_seconds += time.perf_counter() - step_started
                        steps += 1
                        if done:
                            success = bool(env.check_success())
                            break
                except Exception as error:
                    episode_error = {"type": type(error).__name__, "message": str(error)}
                finished = time.perf_counter()
                if episode_error is None:
                    completed += 1
                    successes += int(success)
                writer.write("episode", task_suite=args.task_suite, task_id=task_id, task_name=task.name,
                             initial_state_index=initial_index, seed=episode_seed, run_seed=args.seed,
                             success=success if episode_error is None else None, status="complete" if episode_error is None else "error",
                             steps=steps, settle_steps=settled, total_sim_steps=steps + settled,
                             policy_calls=len(latencies), policy_latency_ms=latencies, calls=calls,
                             policy_call_seconds=sum(latencies) / 1000, preprocessing_seconds=preparation_seconds,
                             env_step_seconds=environment_seconds, wall_seconds_including_reset=finished - episode_started,
                             rollout_wall_seconds=None if rollout_started is None else finished - rollout_started,
                             nfe=contract["num_steps"], replan_steps=args.replan_steps,
                             checkpoint_sha256=contract["runtime"]["checkpoint"]["sha256"], error=episode_error)
                if episode_error is not None:
                    writer.write("summary", status="aborted_infrastructure_or_protocol_error", attempted=attempted,
                                 completed=completed, successes=successes, wall_seconds=time.perf_counter() - total_wall)
                    return 2
        finally:
            if env is not None:
                env.close()
    writer.write("summary", status="complete", attempted=attempted, completed=completed, successes=successes,
                 success_rate=successes / completed if completed else None,
                 wall_seconds=time.perf_counter() - total_wall, failures_included_in_cost=True,
                 noninferiority_conclusion="not_tested")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_identity_args(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--expected-nfe", type=int)
    parser.add_argument("--task-suite", choices=sorted(MAX_STEPS), default="libero_10")
    parser.add_argument("--tasks", help="0-based subset, e.g. 0,2-4; default all")
    parser.add_argument("--episodes-per-task", type=int, help="Default 50; with --init-indices must equal index count")
    parser.add_argument("--init-indices", help="0-based initial states, e.g. 0-49")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, help="Smoke test override; not the official protocol")
    args = parser.parse_args()
    if args.replan_steps < 1 or args.settle_steps < 0 or (args.episodes_per_task is not None and args.episodes_per_task < 1):
        parser.error("Positive replan/episode counts and nonnegative settle steps required")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max_steps must be positive")
    return run_jsonl(args, run)


if __name__ == "__main__":
    sys.exit(main())
