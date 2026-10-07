#!/usr/bin/env python
"""Thin launcher around the *unmodified* official LGM entry points
(third_party/linear-gradient-matching/src).  Nothing under third_party is edited; every
compatibility fix lives here as a monkeypatch and is listed in lgm/README.md (偏离表).

usage:
  official_entry.py [--timing] [--dry-run] <distill|eval|neighbors|centroids|random_reals|full_dataset> [official CLI args...]
  official_entry.py check-imports            # static import check (works on the Mac, no CUDA needed)

Patches applied (all outside third_party):
  P1  wandb offline: `wandb.run.name` is None in WANDB_MODE=offline, but official distill.py does
      os.path.join(..., wandb.run.name)  ->  we set run.name = run.id (= --run_name) after wandb.init.
  P2  official distill.py's skip_if_exists looks for "05000.pth", a file nothing ever writes
      -> we skip when <log_dir>/data.pth already exists (real resumability), and print the ckpt state.
  P3  --timing: wraps LinearGM.match_gradients to record seconds/iteration and prints
      mean s/it (excluding the first 2 warm-up its), peak CUDA memory, and a 5000-it projection.
  P4  os._exit hook: official distill.py ends with os._exit(0); we print the P3 summary first.
  P5  torch.hub offline: torch.hub.load("facebookresearch/dinov2", ...) always sends one HTTPS request to
      github.com to decide between the 'main'/'master' ref, even when the repo is already cached; any
      non-URLError network failure (e.g. http.client.RemoteDisconnected, seen on the server 2026-09-25
      13:28, timing attempt apb=5) crashes the job. When the cached repo dir exists we answer from the
      cache, which is exactly the directory the official code then loads from.
The official `__main__` blocks are executed verbatim through runpy (same seeds 3407, same Tap parsers).
"""
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "third_party", "linear-gradient-matching", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

MODULES = {
    "distill": "distillation.distill",
    "eval": "distillation.eval",
    "neighbors": "baselines.neighbors",
    "centroids": "baselines.centroids",
    "random_reals": "baselines.random_reals",
    "full_dataset": "baselines.full_dataset",
}

IMPORT_CHECK = [
    "config", "augmentation", "synsets", "my_utils.log_utils", "models", "data.dataloaders",
    "distillation.linear_gm", "distillation.distill", "distillation.eval",
    "baselines.neighbors", "baselines.centroids", "baselines.random_reals", "baselines.full_dataset",
]


def _arg(args, name, default=None):
    """Read `--name value` or `--name=value` from an official argv list (Tap style)."""
    for i, a in enumerate(args):
        if a == f"--{name}" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def check_imports() -> int:
    import importlib
    bad = 0
    for m in IMPORT_CHECK:
        try:
            importlib.import_module(m)
            print(f"[check-imports] ok   {m}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"[check-imports] FAIL {m}: {type(e).__name__}: {e}")
    import torch
    print(f"[check-imports] torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"{'all imports ok' if bad == 0 else str(bad) + ' failures'}")
    return 1 if bad else 0


def patch_wandb():
    """P1: give offline runs a name so official log_dir = logged_files/.../{run.name} works."""
    import wandb
    _orig_init = wandb.init

    def init(*a, **k):
        run = _orig_init(*a, **k)
        try:
            if run is not None and run.name is None:
                run.name = k.get("name") or run.id
                print(f"[official_entry] P1 wandb offline: run.name set to {run.name!r}")
        except Exception as e:  # noqa: BLE001
            print(f"[official_entry] P1 warning: could not set run.name: {e}")
        return run

    wandb.init = init


def patch_torch_hub_offline():
    """P5: answer torch.hub's main/master probe from the local hub cache instead of github.com."""
    import torch.hub as hub
    orig = getattr(hub, "_parse_repo_info", None)
    if orig is None:
        print("[official_entry] P5 warning: torch.hub._parse_repo_info not found; patch skipped")
        return

    def _parse_repo_info(github):
        if ":" not in github and github.count("/") == 1:
            owner, name = github.split("/")
            for ref in ("main", "master"):
                if os.path.isdir(os.path.join(hub.get_dir(), f"{owner}_{name}_{ref}")):
                    print(f"[official_entry] P5 torch.hub: cached repo {owner}_{name}_{ref} found -> "
                          f"skipping the github.com ref probe", flush=True)
                    return owner, name, ref
        return orig(github)

    hub._parse_repo_info = _parse_repo_info


class Timing:
    """P3: per-iteration timing of LinearGM.match_gradients (+ peak VRAM)."""

    def __init__(self, iterations, augs_per_batch, model):
        self.durations = []
        self.iterations = iterations
        self.augs_per_batch = augs_per_batch
        self.model = model
        self.t_start = time.time()

    def install(self):
        import torch
        from distillation import linear_gm
        orig = linear_gm.LinearGM.match_gradients
        timing = self

        def match_gradients(self_):
            torch.cuda.synchronize()
            t0 = time.time()
            out = orig(self_)
            torch.cuda.synchronize()
            timing.durations.append(time.time() - t0)
            if len(timing.durations) <= 3 or len(timing.durations) % 5 == 0:
                print(f"[timing] it {len(timing.durations)} {timing.durations[-1]:.2f}s "
                      f"alloc={torch.cuda.memory_allocated() / 2**30:.2f}GB "
                      f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GB", flush=True)
            return out

        linear_gm.LinearGM.match_gradients = match_gradients

    def summary(self):
        import torch
        d = self.durations
        if not d:
            print("[timing] no iterations recorded")
            return
        steady = d[2:] if len(d) > 4 else d
        mean = sum(steady) / len(steady)
        peak_alloc = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        peak_res = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0
        proj_h = mean * 5000 / 3600
        print(
            f"[timing] SUMMARY model={self.model} augs_per_batch={self.augs_per_batch} "
            f"iters_recorded={len(d)} first_it={d[0]:.2f}s mean_s_per_it={mean:.3f} "
            f"(excl. first 2) peak_alloc_GB={peak_alloc:.2f} peak_reserved_GB={peak_res:.2f} "
            f"wall_total_s={time.time() - self.t_start:.0f} "
            f"proj_5000it_h={proj_h:.2f} proj_cost_CNY={proj_h * 2.18:.2f}",
            flush=True,
        )


def main(argv) -> int:
    timing_flag = False
    dry_run = False
    while argv and argv[0].startswith("--"):
        flag = argv.pop(0)
        if flag == "--timing":
            timing_flag = True
        elif flag == "--dry-run":
            dry_run = True
        else:
            print(f"unknown launcher flag {flag}", file=sys.stderr)
            return 2
    if not argv:
        print(__doc__)
        return 2
    sub, rest = argv[0], argv[1:]
    if sub == "check-imports":
        return check_imports()
    if sub not in MODULES:
        print(f"unknown subcommand {sub}; choose from {sorted(MODULES)} or check-imports", file=sys.stderr)
        return 2
    module = MODULES[sub]
    print(f"[official_entry] cwd={os.getcwd()} module={module} argv={rest}", flush=True)
    if dry_run:
        return 0

    import torch
    if not torch.cuda.is_available():
        print("[official_entry] ERROR: the official code hard-codes .cuda(); run this on the GPU server "
              "(use --dry-run or check-imports on the Mac).", file=sys.stderr)
        return 2

    patch_torch_hub_offline()  # P5: every subcommand may load dinov2_vitb through torch.hub

    timing = None
    if sub == "distill":
        # P2: real skip-if-exists (official checks a never-written 05000.pth)
        job_tag = _arg(rest, "job_tag", "distillation")
        dataset = _arg(rest, "dataset")
        model = _arg(rest, "model")
        run_name = _arg(rest, "run_name")
        skip = _arg(rest, "skip_if_exists", "True")
        if run_name and dataset and model and str(skip).lower() != "false":
            log_dir = os.path.join("logged_files", job_tag, dataset, model, run_name)
            if os.path.exists(os.path.join(log_dir, "data.pth")):
                print(f"[official_entry] P2 {log_dir}/data.pth exists -> distillation already done, skipping.")
                return 0
            if os.path.exists(os.path.join(log_dir, "ckpt.pth")):
                print(f"[official_entry] P2 {log_dir}/ckpt.pth exists -> official code will resume from it.")
        patch_wandb()
        if timing_flag:
            timing = Timing(int(_arg(rest, "iterations", 5000)), int(_arg(rest, "augs_per_batch", 10)), model)
            timing.install()

    # P4: official distill.py calls os._exit(0) at the end -> print summary first
    _orig_exit = os._exit

    def _exit(code=0):
        if timing is not None:
            timing.summary()
        print(f"[official_entry] {module} finished (os._exit {code})", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        _orig_exit(code)

    os._exit = _exit

    sys.argv = [module.replace(".", "/") + ".py"] + rest
    code = 0
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as e:  # official code calls exit() on skip
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    if timing is not None:
        timing.summary()
    print(f"[official_entry] {module} finished (exit {code})", flush=True)
    return code


if __name__ == "__main__":
    rc = main(sys.argv[1:])
    sys.stdout.flush()
    sys.exit(rc)
