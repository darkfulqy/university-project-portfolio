"""Serve pi05_libero with explicit NFE and optional client-keyed flow noise."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import NOISE_FIELD, NOISE_PROTOCOL, add_identity_args, run_jsonl, runtime_metadata


class PairedNoisePolicy:
    def __init__(self, policy, horizon, action_dim):
        self.policy = policy
        self.horizon = horizon
        self.action_dim = action_dim

    def infer(self, request):
        import numpy as np
        observation = dict(request)
        noise = observation.pop(NOISE_FIELD, None)
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float32)
            if noise.shape != (self.horizon, self.action_dim) or not np.isfinite(noise).all():
                raise ValueError("Explicit flow noise must be finite, shape (%d,%d)" % (self.horizon, self.action_dim))
        return self.policy.infer(observation, noise=noise)

    def reset(self):
        return self.policy.reset()


def run(args, writer):
    runtime = runtime_metadata(args, require_checkpoint=True)
    sys.path.insert(0, str(args.openpi_root / "src"))
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config

    cfg = config.get_config("pi05_libero")
    if not cfg.model.pi05 or cfg.model.discrete_state_input or cfg.model.action_horizon != 10:
        raise RuntimeError("Frozen pi05_libero configuration mismatch")
    policy = policy_config.create_trained_policy(cfg, str(args.checkpoint),
                                                sample_kwargs={"num_steps": args.num_steps})
    if policy._is_pytorch_model:
        raise RuntimeError("This launcher requires the JAX/Orbax checkpoint")
    contract = {"config": "pi05_libero", "num_steps": args.num_steps,
                "action_horizon": cfg.model.action_horizon, "action_dim": cfg.model.action_dim,
                "noise_protocol": NOISE_PROTOCOL, "noise_field": NOISE_FIELD,
                "noise_absent_behavior": "official_internal_jax_rng",
                "runtime": runtime}
    metadata = dict(policy.metadata)
    metadata["phase0"] = contract
    writer.write("server_start", host=args.host, port=args.port, phase0=contract,
                 warmup="none; first client inference includes compilation")
    websocket_policy_server.WebsocketPolicyServer(
        policy=PairedNoisePolicy(policy, cfg.model.action_horizon, cfg.model.action_dim),
        host=args.host, port=args.port, metadata=metadata).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_identity_args(parser)
    parser.add_argument("--num-steps", type=int, default=10, help="Flow-matching NFE (usually 1, 2, 10)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.num_steps < 1 or not 1 <= args.port <= 65535:
        parser.error("Positive NFE and port in [1,65535] required")
    return run_jsonl(args, run)


if __name__ == "__main__":
    sys.exit(main())
