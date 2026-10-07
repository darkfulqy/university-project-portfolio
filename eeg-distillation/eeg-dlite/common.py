"""Shared helpers for the EEG-DLite pilot-study reproduction (BCI IV-2a substitute).

Everything that touches the OFFICIAL code (third_party/EEG-DLite) goes through
`import_official()` below, which
  * puts third_party/EEG-DLite on sys.path,
  * disables wandb (WANDB_MODE=disabled; the official code calls wandb.init/log/save),
  * provides a stub `torcheeg.models.cnn.EEGNet` because `torcheeg` cannot be installed
    into this venv (it pins scipy==1.10, which would break mne/moabb).  The k-center /
    CRT path we use never touches EEGNet, the import only has to succeed.
No file under third_party is modified.
"""
from __future__ import annotations

import json
import os
import random
import sys
import types
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # $R/eeg-dlite
R = HERE.parent                                   # $R
THIRD_PARTY = R / "third_party" / "EEG-DLite"
DATA_NPZ = R / "data" / "eegdlite_bciiv2a.npz"
RUNS_ROOT = R / "runs" / "eegdlite"
RUNS_SMOKE_ROOT = R / "runs" / "eegdlite_smoke"
LOGS = R / "logs"

DATASET_NAME = "bciiv2a"
N_FOLDS = 3
SEEDS_FULL = [0, 1, 2, 3, 4]
ETAS_FULL = [0.05, 0.10, 0.25, 0.50]
TAUS_FULL = [0.0, 0.01]
METHODS = ["random", "pca_ds", "proposed"]      # + "full" (eta=1)

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
# let unsupported MPS ops fall back to CPU instead of crashing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MOABB_DOWNLOAD_PROVIDER", "upstream")


# ---------------------------------------------------------------------------
# official code import (shimmed, third_party untouched)
# ---------------------------------------------------------------------------
_official = None


def import_official():
    """Return the official `dataset_distillation.methods` module (cached)."""
    global _official
    if _official is not None:
        return _official
    import importlib

    if str(THIRD_PARTY) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY))

    try:
        import torcheeg.models.cnn  # noqa: F401
    except ImportError:
        # stub package torcheeg.models.cnn with our own EEGNet (never used by k-center path)
        from eegnet import EEGNet as _EEGNet  # local module in this directory

        torcheeg = types.ModuleType("torcheeg")
        models = types.ModuleType("torcheeg.models")
        cnn = types.ModuleType("torcheeg.models.cnn")
        cnn.EEGNet = _EEGNet
        models.cnn = cnn
        models.EEGNet = _EEGNet
        torcheeg.models = models
        sys.modules["torcheeg"] = torcheeg
        sys.modules["torcheeg.models"] = models
        sys.modules["torcheeg.models.cnn"] = cnn

    _official = importlib.import_module("dataset_distillation.methods")
    return _official


# ---------------------------------------------------------------------------
# data / splits
# ---------------------------------------------------------------------------
def load_data(path: Path = DATA_NPZ):
    d = np.load(path, allow_pickle=False)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    subject = d["subject"].astype(np.int64)
    session = d["session"]
    return X, y, subject, session


def make_folds(subject: np.ndarray, n_splits: int = N_FOLDS):
    """GroupKFold(3) by subject -> list of (train_idx, test_idx). Deterministic."""
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(subject))
    return [(tr, te) for tr, te in gkf.split(dummy, groups=subject)]


def zscore_per_sample(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """numpy twin of the official `normalize_mat` (flatten each sample, z-score with
    the *unbiased* std that torch.std uses, eps=1e-8)."""
    n = X.shape[0]
    flat = X.reshape(n, -1)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1, ddof=1)
    shape = (n,) + (1,) * (X.ndim - 1)
    return ((X - mean.reshape(shape)) / (std.reshape(shape) + eps)).astype(np.float32)


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except Exception:  # torch not needed by every script
        pass


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def fmt_eta(eta: float) -> str:
    return f"{eta:g}"


def fmt_tau(tau: float) -> str:
    return f"{tau:g}"


def selection_path(root: Path, dataset: str, fold: int, seed: int, method: str, eta: float, tau: float) -> Path:
    return root / "selections" / dataset / f"fold{fold}_seed{seed}_{method}_eta{fmt_eta(eta)}_tau{fmt_tau(tau)}.json"


def parse_selection_name(p: Path) -> dict:
    stem = p.stem  # fold0_seed0_proposed_eta0.25_tau0.01
    parts = stem.split("_")
    fold = int(parts[0][len("fold"):])
    seed = int(parts[1][len("seed"):])
    tau = float(parts[-1][len("tau"):])
    eta = float(parts[-2][len("eta"):])
    method = "_".join(parts[2:-2])
    return dict(fold=fold, seed=seed, method=method, eta=eta, tau=tau)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)
