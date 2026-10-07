"""Five adaptation actions from one frozen source model, per subject and seed.

  A0  no adaptation          -- the thing every other action must beat
  A1  Euclidean Alignment    -- statistical, input-level (needs its own source model)
  A2  AdaBN                  -- statistical, feature-level, no gradients
  A3  T3A                    -- prototype, optimization-free
  A4  Tent                   -- gradient, entropy minimisation

A0, A2, A3 and A4 all read the same trained weights and the same unlabelled
target session, so any difference between them is the action, not the run.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data_io import load_session, euclidean_align, zscore
from run_baseline import train_eval, device
from eegnet import EEGNet
from adapt import adabn, t3a_predict, tent, plain_predict

import torch.nn as nn

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results"


def fit_source(Xtr, ytr, seed, epochs, dev):
    """Same recipe as run_baseline, but returns the model instead of scores."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EEGNet(n_chans=Xtr.shape[1], n_samples=Xtr.shape[2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    X = torch.tensor(Xtr, device=dev)
    y = torch.tensor(ytr, device=dev)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            lossf(model(X[idx]), y[idx]).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--subjects", type=str, default="1-9")
    ap.add_argument("--tag", type=str, default="fam")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    dev = device()
    print(f"device: {dev}\n", flush=True)

    lo, hi = (int(v) for v in args.subjects.split("-"))
    rows = []
    for s in range(lo, hi + 1):
        sid = f"A{s:02d}"
        Xt, yt = load_session(DATA / f"{sid}T.mat")
        Xe, ye = load_session(DATA / f"{sid}E.mat")
        itr, ite = train_test_split(
            np.arange(len(Xt)), test_size=0.2, stratify=yt, random_state=0)

        for seed in range(args.seeds):
            t0 = time.time()

            # ---- source model on unaligned data; A0/A2/A3/A4 all branch from it
            Xtr, mu, sd = zscore(Xt[itr])
            Xin, _, _ = zscore(Xt[ite], mu, sd)
            Xcr, _, _ = zscore(Xe, mu, sd)
            src = fit_source(Xtr, yt[itr], seed, args.epochs, dev)

            acc = lambda p: float((p == ye).mean())
            r = dict(subject=sid, seed=seed)
            r["within"] = float((plain_predict(src, Xin, dev) == yt[ite]).mean())
            preds = {
                "A0_none": plain_predict(src, Xcr, dev),
                "A2_adabn": plain_predict(adabn(src, Xcr, dev), Xcr, dev),
                "A3_t3a": t3a_predict(src, Xcr, dev),
                "A4_tent": plain_predict(tent(src, Xcr, dev), Xcr, dev),
            }
            for k, v in preds.items():
                r[k] = acc(v)
            # keep per-trial correctness so ΔEA can be bootstrapped over trials,
            # which is the unit that actually carries the sampling error
            r["_hit"] = {k: (v == ye).astype(int).tolist() for k, v in preds.items()}

            # ---- EA needs a source model trained on aligned data
            Xt_ea, Xe_ea = euclidean_align(Xt), euclidean_align(Xe)
            Xtr2, mu2, sd2 = zscore(Xt_ea[itr])
            Xcr2, _, _ = zscore(Xe_ea, mu2, sd2)
            src_ea = fit_source(Xtr2, yt[itr], seed, args.epochs, dev)
            p_ea = plain_predict(src_ea, Xcr2, dev)
            r["A1_ea"] = acc(p_ea)
            r["_hit"]["A1_ea"] = (p_ea == ye).astype(int).tolist()

            rows.append(r)
            print(f"{sid} s{seed}  none {r['A0_none']:.3f}  EA {r['A1_ea']:.3f}  "
                  f"AdaBN {r['A2_adabn']:.3f}  T3A {r['A3_t3a']:.3f}  "
                  f"Tent {r['A4_tent']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    (OUT / f"{args.tag}.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote results/{args.tag}.json")


if __name__ == "__main__":
    main()
