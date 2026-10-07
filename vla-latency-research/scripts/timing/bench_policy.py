"""Synchronized, in-process official pi05_libero timing; no environment rollout."""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import add_identity_args, run_jsonl, runtime_metadata


def measure(function, synchronize, warmup, samples):
    import numpy as np
    started = time.perf_counter_ns()
    result = function()
    synchronize(result)
    first_ms = (time.perf_counter_ns() - started) / 1e6
    for _ in range(warmup):
        result = function()
        synchronize(result)
    timings = []
    for _ in range(samples):
        synchronize(result)
        started = time.perf_counter_ns()
        result = function()
        synchronize(result)
        timings.append((time.perf_counter_ns() - started) / 1e6)
    return {"unit": "ms", "first_call_including_compile_ms": first_ms, "warmup": warmup,
            "samples": samples, "p50": float(np.percentile(timings, 50)),
            "p90": float(np.percentile(timings, 90)), "p99": float(np.percentile(timings, 99)),
            "mean": float(np.mean(timings)), "sample_ms": timings}


def array_difference(left, right):
    import numpy as np
    left, right = np.asarray(left), np.asarray(right)
    return {"shape_equal": left.shape == right.shape,
            "bitwise_equal": left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes(),
            "max_abs": float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))),
            "allclose_atol_1e-5_rtol_1e-5": bool(np.allclose(left, right, atol=1e-5, rtol=1e-5))}


def run(args, writer):
    runtime = runtime_metadata(args, require_checkpoint=True)
    sys.path.insert(0, str(args.openpi_root / "src"))
    import einops
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    from openpi.policies import policy_config
    from openpi.shared import nnx_utils
    from openpi.training import config

    cfg = config.get_config("pi05_libero")
    if not cfg.model.pi05 or cfg.model.discrete_state_input or cfg.model.action_horizon != 10:
        raise RuntimeError("pi05_libero model configuration differs from the audited baseline")
    writer.write("metadata", runtime=runtime, track="official_jax", batch_size=1,
                 jax_devices=[str(device) for device in jax.devices()],
                 config="pi05_libero", action_horizon=cfg.model.action_horizon,
                 action_dim=cfg.model.action_dim, seed=args.seed, nfe=args.nfe,
                 timing_boundary="CPU observation -> policy.infer transforms/H2D/model/D2H/output transforms -> synchronized CPU actions",
                 module_boundary="preloaded device operands, synchronized wall time including dispatch; not GPU-active seconds",
                 system_optimized_track="not_implemented; do not relabel this run")
    started = time.perf_counter()
    policy = policy_config.create_trained_policy(cfg, str(args.checkpoint), sample_kwargs={"num_steps": 10})
    if policy._is_pytorch_model:
        raise RuntimeError("Expected Orbax/JAX checkpoint, not model.safetensors")
    policy._rng = jax.random.key(args.seed)
    writer.write("load", elapsed_seconds=time.perf_counter() - started)
    random_state = np.random.RandomState(args.seed)
    raw = {"observation/image": random_state.randint(0, 256, (224, 224, 3), dtype=np.uint8),
           "observation/wrist_image": random_state.randint(0, 256, (224, 224, 3), dtype=np.uint8),
           "observation/state": random_state.standard_normal(8), "prompt": args.prompt}
    fixed_noise = random_state.standard_normal((cfg.model.action_horizon, cfg.model.action_dim)).astype(np.float32)
    transformed = policy._input_transform(jax.tree.map(lambda value: value, raw))
    batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
    observation = model_api.Observation.from_dict(batched)
    observation = model_api.preprocess_observation(None, observation, train=False)
    noise = jnp.asarray(fixed_noise)[None, ...]
    jax.block_until_ready((observation, noise))

    class Segments(nnx.Module):
        def __init__(self, model):
            self.model = model

        def images(self, obs):
            return {name: self.model.PaliGemma.img(image, train=False)[0] for name, image in obs.images.items()}

        def prefix(self, obs):
            tokens, mask, ar_mask = self.model.embed_prefix(obs)
            _, cache = self.model.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar_mask),
                                                positions=jnp.cumsum(mask, axis=1) - 1)
            return cache, mask

        def prefix_from_images(self, obs, encoded):
            tokens = [encoded[name] for name in obs.images]
            masks = [einops.repeat(obs.image_masks[name], "b -> b s", s=encoded[name].shape[1]) for name in obs.images]
            if obs.tokenized_prompt is not None:
                tokens.append(self.model.PaliGemma.llm(obs.tokenized_prompt, method="embed"))
                masks.append(obs.tokenized_prompt_mask)
            prefix_tokens = jnp.concatenate(tokens, axis=1)
            prefix_mask = jnp.concatenate(masks, axis=1)
            ar_mask = jnp.zeros((prefix_tokens.shape[1],), dtype=jnp.bool_)
            _, cache = self.model.PaliGemma.llm([prefix_tokens, None],
                                                mask=make_attn_mask(prefix_mask, ar_mask),
                                                positions=jnp.cumsum(prefix_mask, axis=1) - 1)
            return cache, prefix_mask

        def denoise(self, obs, cache, prefix_mask, initial_noise, num_steps):
            delta = -1.0 / num_steps
            batch_size = obs.state.shape[0]

            def step(carry):
                actions, current_time = carry
                tokens, mask, ar_mask, condition = self.model.embed_suffix(
                    obs, actions, jnp.broadcast_to(current_time, batch_size))
                prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=tokens.shape[1])
                full_mask = jnp.concatenate([prefix_attn, make_attn_mask(mask, ar_mask)], axis=-1)
                positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
                (_, output), _ = self.model.PaliGemma.llm(
                    [None, tokens], mask=full_mask, positions=positions, kv_cache=cache,
                    adarms_cond=[None, condition])
                velocity = self.model.action_out_proj(output[:, -self.model.action_horizon:])
                return actions + delta * velocity, current_time + delta

            return jax.lax.while_loop(lambda carry: carry[1] >= -delta / 2,
                                      step, (initial_noise, 1.0))[0]

    segments = Segments(policy._model)
    images_fn = nnx_utils.module_jit(segments.images)
    prefix_fn = nnx_utils.module_jit(segments.prefix)
    prefix_only_fn = nnx_utils.module_jit(segments.prefix_from_images)
    denoise_fn = nnx_utils.module_jit(segments.denoise)

    def record(name, function, sync=jax.block_until_ready, **extra):
        result = measure(function, sync, args.warmup, args.samples)
        writer.write("timing", segment=name, **extra, **result)

    record("siglip_all_three_slots", lambda: images_fn(observation))
    encoded = images_fn(observation)
    jax.block_until_ready(encoded)
    record("prefix_embed_and_gemma_including_siglip", lambda: prefix_fn(observation))
    cache, prefix_mask = prefix_fn(observation)
    jax.block_until_ready((cache, prefix_mask))
    record("prefix_embed_and_gemma_excluding_siglip", lambda: prefix_only_fn(observation, encoded))
    standalone_cache, standalone_mask = prefix_only_fn(observation, encoded)
    jax.block_until_ready((standalone_cache, standalone_mask))
    writer.write("diagnostic", name="prefix_split_equivalence",
                 cache_leaves=[array_difference(left, right) for left, right in
                               zip(jax.tree.leaves(cache), jax.tree.leaves(standalone_cache))],
                 mask=array_difference(prefix_mask, standalone_mask))
    equivalent = True
    for num_steps in args.nfe:
        policy._sample_kwargs["num_steps"] = num_steps
        record("cpu_to_cpu_policy_infer", lambda: policy.infer(raw),
               sync=lambda result: jax.block_until_ready((result, policy._rng)), nfe=num_steps,
               noise="official_jax_rng; no explicit noise supplied")
        record("action_expert_cached_prefix", lambda: denoise_fn(observation, cache, prefix_mask, noise, num_steps),
               nfe=num_steps, noise="fixed_device_noise; RNG generation excluded")
        official = policy._sample_actions(jax.random.key(args.seed), observation, num_steps=num_steps, noise=noise)
        split = denoise_fn(observation, cache, prefix_mask, noise, num_steps)
        jax.block_until_ready((official, split))
        difference = array_difference(official, split)
        equivalent = equivalent and difference["allclose_atol_1e-5_rtol_1e-5"]
        writer.write("diagnostic", name="split_vs_official_denoise", nfe=num_steps, **difference)
    record("input_transforms_cpu", lambda: policy._input_transform(jax.tree.map(lambda value: value, raw)),
           sync=lambda result: None)
    output_operand = {"state": np.asarray(observation.state[0]), "actions": np.asarray(official[0])}
    record("output_transforms_cpu", lambda: policy._output_transform(jax.tree.map(lambda value: value.copy(), output_operand)),
           sync=lambda result: None)

    policy._sample_kwargs["num_steps"] = 10
    altered = dict(raw, **{"observation/state": raw["observation/state"] + 0.25})
    original_result = policy.infer(raw, noise=fixed_noise)
    altered_result = policy.infer(altered, noise=fixed_noise)
    jax.block_until_ready((original_result, altered_result, policy._rng))
    writer.write("diagnostic", name="state_only_same_flow_noise", nfe=10,
                 discrete_state_input=cfg.model.discrete_state_input,
                 **array_difference(original_result["actions"], altered_result["actions"]))

    traced_shapes = []
    original_encoder = policy._model.PaliGemma.img

    class TracedEncoder(nnx.Module):
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, image, **kwargs):
            traced_shapes.append(list(image.shape))
            return self.inner(image, **kwargs)

    try:
        policy._model.PaliGemma.img = TracedEncoder(original_encoder)
        traced_prefix = nnx_utils.module_jit(policy._model.embed_prefix)
        jax.block_until_ready(traced_prefix(observation))
    finally:
        policy._model.PaliGemma.img = original_encoder
    writer.write("diagnostic", name="masked_third_camera_encoding",
                 camera_order=list(observation.images), traced_encoder_calls=traced_shapes,
                 image_masks={name: np.asarray(mask).tolist() for name, mask in observation.image_masks.items()},
                 third_slot_normalized_min=float(np.min(np.asarray(observation.images["right_wrist_0_rgb"]))),
                 third_slot_normalized_max=float(np.max(np.asarray(observation.images["right_wrist_0_rgb"]))),
                 conclusion="embed_prefix traces the masked slot through SigLIP; post-XLA executed-kernel cost requires profiling",
                 deletion_equivalence="not_tested; no slot removed")
    writer.write("summary", status="timing_complete" if equivalent else "inconclusive_split_mismatch",
                 split_allclose=equivalent, gpu_active_seconds="not_measured",
                 additive_percentile_attribution=False)
    return 0 if equivalent else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_identity_args(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="pick up the black bowl and place it on the plate")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 10])
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1 or any(value < 1 for value in args.nfe):
        parser.error("warmup >= 0, samples >= 1 and positive NFE required")
    return run_jsonl(args, run)


if __name__ == "__main__":
    sys.exit(main())
