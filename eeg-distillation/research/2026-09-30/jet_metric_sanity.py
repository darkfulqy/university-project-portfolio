"""Tiny, training-free audit of the pinned upstream JET TS-FID implementation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "baseline/JET/code"
spec = importlib.util.spec_from_file_location("jet_upstream_metrics", CODE / "data/metrics.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)

torch.set_num_threads(2)
torch.manual_seed(20260930)
z = np.load(ROOT / "evaluation/repro/data/eegdlite_bciiv2a.npz", allow_pickle=False)
# Read real training-session examples only, never train a model or access weights.
idx = np.concatenate([
    np.flatnonzero((z["subject"] == 1) & (z["session"] == "0train") & (z["y"] == c))[:8]
    for c in range(4)
])
x = torch.from_numpy(z["X"][idx].copy()).float() / 100.0
labels = z["y"][idx]
fft = torch.fft.rfft(x.double(), dim=-1)
phase = torch.rand_like(fft.real) * (2 * np.pi)
phase[..., 0] = 0
phase[..., -1] = 0
scrambled = torch.fft.irfft(fft * torch.exp(1j * phase), n=x.shape[-1], dim=-1).float()
signs = torch.where(torch.arange(x.shape[1]) % 2 == 0, 1.0, -1.0)[None, :, None]
sign_flipped = x * signs

def normalized_rmse(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()) / torch.sqrt((b ** 2).mean()))

def channel_corr_change(a, b):
    return float(torch.stack([
        (torch.corrcoef(ai.double()) - torch.corrcoef(bi.double())).abs().mean()
        for ai, bi in zip(a, b)
    ]).mean())

ref_features = metrics._prepare_fid_features(x, feature_bins=256)
out = {
    "purpose": "metric invariance diagnostic; not generator quality or downstream utility",
    "commit": subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip(),
    "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
    "data": {"path": "evaluation/repro/data/eegdlite_bciiv2a.npz", "subject": 1, "session": "0train", "indices": idx.tolist(), "shape": list(x.shape), "scale": 0.01, "sfreq": int(z["sfreq"])},
    "seed": 20260930,
    "metric_args": {"feature_bins": 256, "spatial_bins": 4, "max_frequency_ratio": 0.5, "chunk_size": 16},
    "feature_dim": int(ref_features.shape[1]),
    "results": {},
}
for name, y in {"identity": x, "independent_random_phase": scrambled, "alternating_channel_sign": sign_flipped}.items():
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fid = metrics.compute_ts_fid_streaming(y, x, chunk_size=16, feature_bins=256)
    feats = metrics._prepare_fid_features(y, feature_bins=256)
    out["results"][name] = {
        "ts_fid_raw": float(fid),
        "max_abs_feature_difference": float(np.max(np.abs(feats - ref_features))),
        "normalized_waveform_rmse": normalized_rmse(y, x),
        "mean_abs_channel_correlation_change": channel_corr_change(y, x),
        "metric_wall_seconds": time.monotonic() - start,
        "warnings": sorted(set(str(w.message) for w in caught)),
    }

# Overall metric has no label argument. Relabeling exactly the same waveforms
# leaves it unchanged by construction. Per-class metrics are a separate audit.
out["label_permutation"] = {
    "operation": "y <- (y + 1) % 4, waveforms unchanged",
    "fraction_labels_changed": float(np.mean((labels + 1) % 4 != labels)),
    "overall_ts_fid": out["results"]["identity"]["ts_fid_raw"],
    "interpretation": "exact invariant because overall TS-FID has no labels; upstream inference also reports per-class FID, which can detect distribution differences",
}
out["limitations"] = [
    "Only 32 real training trials; FID covariance is singular and tiny negative values are numerical sqrtm error, not negative true distance.",
    "No model was trained or sampled; no downstream accuracy claim is established.",
    "Phase changes and channel sign inversions may alter task-relevant EEG relationships despite unchanged spectral magnitude.",
    "This exposes known scope of an amplitude-spectrum metric, not proof that JET samples are poor or that its paper results are invalid.",
]
target = Path(__file__).with_name("jet_metric_sanity.json")
target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in out.items() if k in {"results", "label_permutation", "feature_dim", "versions"}}, ensure_ascii=False, indent=2))
