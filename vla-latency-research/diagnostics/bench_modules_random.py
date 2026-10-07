"""Claude-side independent module timing for pi05_libero (JAX, openpi@215abfb).

Latency depends on the compute graph, not on weight values, so this uses a randomly
initialised Pi0(pi05=True, action_horizon=10, discrete_state_input=False) in the config's
dtype. Real-weight numbers come later from the policy-level benchmark.

Measures (device time, each call ends with block_until_ready, batch=1):
  img3      SigLIP on all 3 camera slots (LIBERO leaves the 3rd slot as a zero image)
  img2      SigLIP on 2 slots only (what would remain if the empty slot were dropped)
  prefix    embed_prefix + Gemma prefix forward that builds the KV cache (3 slots + prompt)
  full@k    sample_actions with num_steps=k, k in {1,2,10} (prefix + k denoise steps)
Derived:   per-step action expert = (full@10 - full@1) / 9

Usage: python bench_modules_random.py --out result.json [--warmup 50] [--iters 500]
"""
import argparse
import json
import platform
import statistics
import time

import jax
import jax.numpy as jnp
from flax import nnx

from openpi.models import model as _model
from openpi.models import pi0_config


def stats(xs):
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
    return {"n": len(xs), "mean_ms": statistics.fmean(xs), "p50_ms": q(0.5), "p90_ms": q(0.9), "p99_ms": q(0.99), "min_ms": xs[0]}


def bench(fn, args, warmup, iters):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    return stats(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    args = ap.parse_args()

    cfg = pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
    model = cfg.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    obs = cfg.fake_obs(batch_size=1)
    # LIBERO: third camera slot is an all-zero image with mask False (still encoded by SigLIP)
    names = list(obs.images.keys())
    obs = _model.Observation(
        images={n: (obs.images[n] if i < 2 else jnp.zeros_like(obs.images[n])) for i, n in enumerate(names)},
        image_masks={n: (obs.image_masks[n] if i < 2 else jnp.zeros_like(obs.image_masks[n])) for i, n in enumerate(names)},
        state=obs.state,
        tokenized_prompt=obs.tokenized_prompt,
        tokenized_prompt_mask=obs.tokenized_prompt_mask,
        token_ar_mask=obs.token_ar_mask,
        token_loss_mask=obs.token_loss_mask,
    )

    @jax.jit
    def img_fn(state, images):
        m = nnx.merge(graphdef, state)
        return [m.PaliGemma.img(im, train=False)[0] for im in images]

    @jax.jit
    def prefix_fn(state, obs):
        from openpi.models.pi0 import make_attn_mask
        m = nnx.merge(graphdef, state)
        o = _model.preprocess_observation(None, obs, train=False)
        toks, mask, ar = m.embed_prefix(o)
        attn = make_attn_mask(mask, ar)
        pos = jnp.cumsum(mask, axis=1) - 1
        _, kv = m.PaliGemma.llm([toks, None], mask=attn, positions=pos)
        return kv

    def make_full(k):
        @jax.jit
        def full_fn(state, obs, rng):
            m = nnx.merge(graphdef, state)
            return m.sample_actions(rng, obs, num_steps=k)
        return full_fn

    rng = jax.random.key(1)
    imgs3 = [obs.images[n] for n in names]
    res = {
        "meta": {
            "jax": jax.__version__,
            "device": str(jax.devices()[0]),
            "device_kind": jax.devices()[0].device_kind,
            "host": platform.node(),
            "dtype": str(cfg.dtype),
            "weights": "random-init (latency is weight-independent)",
            "camera_slots": names,
            "prefix_tokens": int(sum(int(x.shape[1]) for x in img_fn(state, imgs3)) + obs.tokenized_prompt.shape[1]),
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "img3": bench(img_fn, (state, imgs3), args.warmup, args.iters),
        "img2": bench(img_fn, (state, imgs3[:2]), args.warmup, args.iters),
        "prefix": bench(prefix_fn, (state, obs), args.warmup, args.iters),
    }
    for k in (1, 2, 10):
        res[f"full@{k}"] = bench(make_full(k), (state, obs, rng), args.warmup, args.iters)
    res["derived"] = {
        "expert_per_step_ms": (res["full@10"]["p50_ms"] - res["full@1"]["p50_ms"]) / 9,
        "prefix_share_of_full10": res["prefix"]["p50_ms"] / res["full@10"]["p50_ms"],
        "empty_slot_cost_ms": res["img3"]["p50_ms"] - res["img2"]["p50_ms"],
        "speedup_nfe1_vs_10": res["full@10"]["p50_ms"] / res["full@1"]["p50_ms"],
    }
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: (v if k in ("meta", "derived") else {kk: round(vv, 2) for kk, vv in v.items()}) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
