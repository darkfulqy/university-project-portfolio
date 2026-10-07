"""BCI Competition IV-2a (BNCI 001-2014) loader.

Each .mat holds 9 runs: runs 0-2 are EOG calibration, runs 3-8 are the six
motor-imagery runs (48 trials each -> 288 trials per session). Session T and
session E were recorded on different days, which is what makes this a
cross-session benchmark rather than a cross-subject one.
"""
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt

FS = 250
N_EEG = 22          # first 22 channels are EEG, last 3 are EOG
TMIN, TMAX = 2.0, 6.0   # seconds from trial onset; cue is at 2.0 s


def _bandpass(x, lo=4.0, hi=38.0, fs=FS, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def load_session(path):
    """Return (X, y) with X of shape (n_trials, 22, n_samples), y in {0,1,2,3}."""
    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    runs = mat["data"]

    lo, hi = int(TMIN * FS), int(TMAX * FS)
    n_samples = hi - lo

    X, y = [], []
    for run in runs:
        # EOG calibration runs carry no class labels
        labels = np.atleast_1d(run.y)
        if labels.size == 0 or not np.any(labels):
            continue
        sig = np.asarray(run.X, dtype=np.float64).T      # (25, n_time)
        sig = sig[:N_EEG]
        sig = _bandpass(sig)

        starts = np.atleast_1d(run.trial).astype(int)
        artifacts = np.atleast_1d(run.artifacts).astype(bool)
        if artifacts.size != starts.size:
            artifacts = np.zeros(starts.size, dtype=bool)

        for k, s in enumerate(starts):
            if s + hi > sig.shape[1]:
                continue
            if artifacts[k]:                              # drop flagged trials
                continue
            X.append(sig[:, s + lo:s + hi])
            y.append(int(labels[k]) - 1)

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    assert X.shape[1:] == (N_EEG, n_samples), X.shape
    return X, y


def euclidean_align(X):
    """Euclidean Alignment (He & Wu, 2020) — unsupervised, per session.

    Whitens each session by the inverse square root of its mean spatial
    covariance, so sessions start from a common reference point.
    """
    cov = np.einsum("nct,ndt->cd", X, X) / (X.shape[0] * X.shape[2])
    w, V = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-12)
    R_inv_sqrt = (V * (w ** -0.5)) @ V.T
    return np.einsum("cd,ndt->nct", R_inv_sqrt, X).astype(np.float32)


def zscore(X, mean=None, std=None):
    """Per-channel standardisation. Statistics come from the training set."""
    if mean is None:
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return ((X - mean) / std).astype(np.float32), mean, std
