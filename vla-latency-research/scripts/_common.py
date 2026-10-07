"""Standard-library metadata helpers; simulator imports stay lazy for --help."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time


OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
LIBERO_COMMIT = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
NOISE_FIELD = "_phase0_noise"
NOISE_PROTOCOL = "phase0-explicit-flow-noise-v1"
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def default_openpi_root():
    server = Path(os.environ.get("VLA", "/path/to/external-workspace/vla")) / "src/openpi"
    return server if server.is_dir() else Path(__file__).resolve().parents[1] / "_vendor/openpi"


def add_identity_args(parser):
    parser.add_argument("--openpi-root", type=Path, default=default_openpi_root())
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/path/to/external-workspace/vla/ckpt/pi05_libero"))
    parser.add_argument("--checkpoint-sha", help="Previously verified manifest SHA256; recorded as declared, not rehashed")
    parser.add_argument("--allow-source-mismatch", action="store_true",
                        help="Diagnostic runs only; do not label these as the frozen baseline")
    parser.add_argument("--output", type=Path, help="New JSONL file (exclusive create); otherwise stdout")


def git_info(path):
    def git(*args):
        result = subprocess.run(["git", "-C", str(path)] + list(args),
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    try:
        status = git("status", "--porcelain", "--untracked-files=no")
        return {"path": str(Path(path).resolve()), "commit": git("rev-parse", "HEAD"),
                "tracked_dirty": bool(status), "tracked_status": status,
                "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest()}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"path": str(path), "commit": None, "error": str(error)}


def tracked_changes(path, paths):
    result = subprocess.run(["git", "-C", str(path), "diff", "HEAD", "--"] + paths,
                            capture_output=True, text=True, check=True)
    return result.stdout


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(path, declared_sha=None, required=False):
    path = Path(path).resolve()
    result = {"path": str(path), "sha256": None,
              "algorithm": "sha256-sorted-jsonl-relative-path-size-content-sha256-v1"}
    if declared_sha is not None:
        if len(declared_sha) != 64 or any(char not in "0123456789abcdef" for char in declared_sha.lower()):
            raise ValueError("--checkpoint-sha must be a 64-character SHA256 hex digest")
        result.update(sha256=declared_sha.lower(), verification="declared_not_recomputed")
        return result
    if not path.is_dir():
        if required:
            raise FileNotFoundError("Checkpoint directory missing: %s" % path)
        result["verification"] = "unavailable_in_this_environment"
        return result
    started = time.perf_counter()
    digest = hashlib.sha256()
    count = total = 0
    for filename in sorted(path.rglob("*")):
        relative = filename.relative_to(path)
        if any(part in (".git", ".cache", "__pycache__") for part in relative.parts) or not filename.is_file():
            continue
        size = filename.stat().st_size
        record = [relative.as_posix(), size, file_sha256(filename)]
        digest.update((json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n").encode())
        count += 1
        total += size
    if not count:
        raise ValueError("Empty checkpoint directory")
    result.update(sha256=digest.hexdigest(), verification="computed_from_all_checkpoint_files",
                  file_count=count, bytes=total, hash_seconds=time.perf_counter() - started)
    return result


def runtime_metadata(args, require_checkpoint=False):
    sources = {"openpi": git_info(args.openpi_root),
               "libero": git_info(args.openpi_root / "third_party/libero")}
    matched = sources["openpi"]["commit"] == OPENPI_COMMIT and sources["libero"]["commit"] == LIBERO_COMMIT
    if not matched and not args.allow_source_mismatch:
        raise RuntimeError("Pinned source commits do not match: %s" % sources)
    core_diff = tracked_changes(args.openpi_root, ["src/openpi", "packages/openpi-client/src"])
    libero_diff = tracked_changes(args.openpi_root / "third_party/libero", ["libero"])
    frozen = matched and not core_diff and not libero_diff
    if not frozen and not args.allow_source_mismatch:
        raise RuntimeError("Audited runtime sources are modified; inspect the diff or explicitly mark a diagnostic run")
    versions = {}
    for name in ("jax", "jaxlib", "flax", "torch", "numpy", "mujoco", "robosuite", "openpi-client", "libero"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        gpu = {"unavailable": str(error)}
    script_root = Path(__file__).resolve().parent
    return {"schema_version": 1, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version, "platform": platform.platform(), "sources": sources,
            "pinned_commits_match": matched, "frozen_runtime_sources": frozen, "versions": versions, "gpu": gpu,
            "checkpoint": checkpoint_identity(args.checkpoint, args.checkpoint_sha, require_checkpoint),
            "script_sha256": {str(path.relative_to(script_root)): file_sha256(path)
                              for path in sorted(script_root.rglob("*")) if path.suffix in (".py", ".sh")},
            "environment": {key: os.environ.get(key) for key in
                            ("MUJOCO_GL", "PYOPENGL_PLATFORM", "CUDA_VISIBLE_DEVICES", "XLA_FLAGS",
                             "XLA_PYTHON_CLIENT_PREALLOCATE", "OMP_NUM_THREADS")}}


class JsonlWriter:
    def __init__(self, path=None):
        self.handle = open(path, "x", encoding="utf-8") if path else sys.stdout
        self.owned = path is not None

    def write(self, event, **payload):
        self.handle.write(json.dumps(dict(event=event, **payload), ensure_ascii=False, allow_nan=False) + "\n")
        self.handle.flush()

    def close(self):
        if self.owned:
            self.handle.close()


def run_jsonl(args, function):
    writer = JsonlWriter(args.output)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = function(args, writer)
        return 0 if result is None else result
    except Exception as error:
        writer.write("error", error_type=type(error).__name__, message=str(error))
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        writer.close()


def keyed_seed(*parts):
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def seed_globals(seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)


def parse_indices(text, total):
    if text is None:
        return list(range(total))
    values = []
    for item in text.split(","):
        bounds = item.strip().split("-")
        if len(bounds) == 1:
            values.append(int(bounds[0]))
        elif len(bounds) == 2:
            first, last = map(int, bounds)
            if first > last:
                raise ValueError("Index ranges must be ascending")
            values.extend(range(first, last + 1))
        else:
            raise ValueError("Use comma-separated indices/ranges, e.g. 0,2-4")
    if not values or len(set(values)) != len(values) or any(value < 0 or value >= total for value in values):
        raise ValueError("Indices must be unique and inside [0,%d)" % total)
    return values


def make_env(task, seed, resolution=256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    seed_globals(seed)
    filename = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(filename), camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env


def policy_observation(observation, prompt):
    import math
    import numpy as np
    from openpi_client import image_tools
    quaternion = np.array(observation["robot0_eef_quat"], copy=True)
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quaternion[3] * quaternion[3])
    rotation = np.zeros(3) if math.isclose(denominator, 0.0) else quaternion[:3] * 2.0 * math.acos(quaternion[3]) / denominator
    images = [image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(observation[key][::-1, ::-1]), 224, 224))
        for key in ("agentview_image", "robot0_eye_in_hand_image")]
    return {"observation/image": images[0], "observation/wrist_image": images[1],
            "observation/state": np.concatenate((observation["robot0_eef_pos"], rotation,
                                                   observation["robot0_gripper_qpos"])), "prompt": str(prompt)}


def server_contract(client):
    metadata = client.get_server_metadata()
    phase0 = metadata.get("phase0", {})
    if phase0.get("noise_protocol") != NOISE_PROTOCOL:
        raise RuntimeError("Use serve_policy_nfe.sh: stock server cannot accept paired flow noise")
    if phase0.get("config") != "pi05_libero" or phase0.get("action_horizon") != 10:
        raise RuntimeError("Expected frozen pi05_libero with action horizon 10")
    if not phase0.get("runtime", {}).get("checkpoint", {}).get("sha256"):
        raise RuntimeError("Server did not provide checkpoint hash")
    return phase0


def query_paired(client, observation, prompt, contract, episode_seed, absolute_step):
    import numpy as np
    element = policy_observation(observation, prompt)
    noise_seed = keyed_seed(episode_seed, absolute_step, "flow_noise")
    element[NOISE_FIELD] = np.random.RandomState(noise_seed).standard_normal(
        (contract["action_horizon"], contract["action_dim"])).astype(np.float32)
    started = time.perf_counter_ns()
    try:
        result = client.infer(element)
    except Exception as error:
        error.phase0_latency_ms = (time.perf_counter_ns() - started) / 1e6
        raise
    elapsed = (time.perf_counter_ns() - started) / 1e6
    try:
        actions = np.asarray(result["actions"])
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError("Invalid policy action chunk: %s" % (actions.shape,))
    except Exception as error:
        error.phase0_latency_ms = elapsed
        raise
    return actions, elapsed, result.get("server_timing", {}), noise_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emit Phase-0 source/checkpoint/runtime identity as JSONL")
    add_identity_args(parser)
    arguments = parser.parse_args()
    sys.exit(run_jsonl(arguments, lambda args, writer: writer.write("metadata", runtime=runtime_metadata(args))))
