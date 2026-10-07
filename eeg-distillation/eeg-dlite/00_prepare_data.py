#!/usr/bin/env python
"""Step 00: BCI IV-2a (BNCI2014_001, 9 subjects x 2 sessions) -> $R/data/eegdlite_bciiv2a.npz

Pre-processing follows the EEG-DLite / LaBraM convention for pre-training corpora:
  band-pass 0.1-75 Hz  ->  50 Hz notch  ->  resample to 200 Hz  ->  epoch the MI segment [2, 6] s
Only the 22 EEG channels are kept (EOG dropped).  Values are stored in raw micro-volts;
the sample-wise z-score (official `normalize_mat`) is applied later inside the pipeline.

Resumable: if the output exists the script exits immediately (use --force to redo).
`--subjects 1,2` limits the subjects (for a quick check only; the real run uses all 9).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (sets MOABB_DOWNLOAD_PROVIDER=upstream)

FMIN, FMAX, NOTCH, SFREQ_NEW = 0.1, 75.0, 50.0, 200
TMIN, TMAX = 2.0, 6.0                      # MI segment relative to trial onset (dataset interval)
N_TIMES = int(round((TMAX - TMIN) * SFREQ_NEW))   # 800 samples
EVENT_ID = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}   # moabb BNCI2014_001 events


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(common.DATA_NPZ))
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        d = np.load(out)
        print(f"[00] {out} exists (X{d['X'].shape}); skip. Use --force to redo.")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    import mne
    mne.set_log_level("WARNING")
    print("[00] importing moabb (first import ~45 s) ...", flush=True)
    from moabb.datasets import BNCI2014_001

    subjects = [int(s) for s in args.subjects.split(",") if s.strip()]
    ds = BNCI2014_001()
    print(f"[00] moabb ready ({time.time() - t0:.0f}s); loading subjects {subjects}", flush=True)

    Xs, ys, subs, sess, runs = [], [], [], [], []
    for subj in subjects:
        data = ds.get_data(subjects=[subj])[subj]      # {session: {run: Raw}}
        for sess_name in sorted(data):
            for run_name in sorted(data[sess_name]):
                raw = data[sess_name][run_name]
                if raw is None:
                    continue
                events = mne.find_events(raw, stim_channel="STI", verbose=False)
                if len(events) == 0:
                    continue
                raw = raw.copy().load_data()
                raw.pick("eeg")                        # 22 EEG channels, EOG dropped
                raw.filter(FMIN, FMAX, verbose=False)  # at native 250 Hz (75 < Nyquist 125)
                raw.notch_filter(NOTCH, verbose=False)
                raw, events = raw.resample(SFREQ_NEW, events=events, verbose=False)
                keep = np.isin(events[:, 2], list(EVENT_ID.values()))
                events = events[keep]
                ep = mne.Epochs(raw, events, event_id=EVENT_ID, tmin=TMIN, tmax=TMIN + (N_TIMES - 1) / SFREQ_NEW,
                                baseline=None, preload=True, verbose=False)
                X = ep.get_data(units="uV").astype(np.float32)      # (n, 22, 800)
                assert X.shape[1:] == (22, N_TIMES), X.shape
                y = ep.events[:, 2] - 1                              # 0..3
                Xs.append(X); ys.append(y)
                subs.append(np.full(len(y), subj, dtype=np.int64))
                sess.append(np.array([sess_name] * len(y)))
                runs.append(np.array([run_name] * len(y)))
        print(f"[00] subject {subj}: {sum(len(v) for v in ys if True)} trials so far ({time.time() - t0:.0f}s)", flush=True)

    X = np.concatenate(Xs); y = np.concatenate(ys); subject = np.concatenate(subs)
    session = np.concatenate(sess); run = np.concatenate(runs)
    ch_names = np.array(ep.ch_names)
    np.savez_compressed(out, X=X, y=y, subject=subject, session=session, run=run, ch_names=ch_names,
                        sfreq=SFREQ_NEW, tmin=TMIN, tmax=TMAX, fmin=FMIN, fmax=FMAX, notch=NOTCH, unit="uV")
    print(f"[00] saved {out}: X{X.shape} y{np.bincount(y)} subjects={np.unique(subject).tolist()} "
          f"sessions={np.unique(session).tolist()} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
