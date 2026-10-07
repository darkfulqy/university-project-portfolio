"""Cross-session baseline on BCI IV-2a, plus a first look at H1.

Three conditions per subject, all sharing one training set (80% of session T):

  within     evaluate on the held-out 20% of session T   -> same-day ceiling
  cross      evaluate on session E (a different day)     -> the drop we care about
  cross+EA   Euclidean Alignment applied per session     -> one unsupervised adaptation

The point is not the mean accuracy. It is whether EA's benefit varies across
subjects by more than seed noise -- that is hypothesis H1 in miniature.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from data_io import load_session, euclidean_align, zscore
from eegnet import EEGNet

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results"


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_eval(Xtr, ytr, evals, seed, epochs=200, bs=32, lr=1e-3, dev=None):
    """Train once, then evaluate on every (name, X, y) in `evals`."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = dev or device()

    model = EEGNet(n_chans=Xtr.shape[1], n_samples=Xtr.shape[2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    Xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(ytr, device=dev)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    out = {}
    with torch.no_grad():
        for name, Xe, ye in evals:
            pred = model(torch.tensor(Xe, device=dev)).argmax(1).cpu().numpy()
            out[name] = float((pred == ye).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--subjects", type=str, default="1-9")
    ap.add_argument("--tag", type=str, default="raw")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    dev = device()
    print(f"device: {dev}\n")

    lo, hi = (int(v) for v in args.subjects.split("-"))
    rows = []
    for s in range(lo, hi + 1):
        sid = f"A{s:02d}"
        Xt, yt = load_session(DATA / f"{sid}T.mat")
        Xe, ye = load_session(DATA / f"{sid}E.mat")

        # one split, reused by every condition, so only the test set differs
        itr, ite = train_test_split(
            np.arange(len(Xt)), test_size=0.2, stratify=yt, random_state=0)

        for seed in range(args.seeds):
            t0 = time.time()

            # --- no adaptation ---
            Xtr, mu, sd = zscore(Xt[itr])
            Xin, _, _ = zscore(Xt[ite], mu, sd)
            Xcr, _, _ = zscore(Xe, mu, sd)
            base = train_eval(Xtr, yt[itr],
                              [("within", Xin, yt[ite]), ("cross", Xcr, ye)],
                              seed, args.epochs, dev=dev)

            # --- Euclidean Alignment, fitted per session, labels never used ---
            Xt_ea = euclidean_align(Xt)          # alignment uses session T as a whole
            Xe_ea = euclidean_align(Xe)          # and session E as a whole
            Xtr2, mu2, sd2 = zscore(Xt_ea[itr])
            Xcr2, _, _ = zscore(Xe_ea, mu2, sd2)
            ea = train_eval(Xtr2, yt[itr], [("cross_ea", Xcr2, ye)],
                            seed, args.epochs, dev=dev)

            rows.append(dict(subject=sid, seed=seed, **base, **ea,
                             n_train=len(itr), n_eval_E=len(ye)))
            print(f"{sid} seed{seed}  within {base['within']:.3f}  "
                  f"cross {base['cross']:.3f}  cross+EA {ea['cross_ea']:.3f}  "
                  f"({time.time()-t0:.0f}s)")

    (OUT / f"{args.tag}.json").write_text(json.dumps(rows, indent=1))
    if (hi - lo + 1) == 9:
        report(rows)


def report(rows):
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[r["subject"]].append(r)

    print("\n" + "=" * 74)
    print(f"{'subject':>8} {'within':>9} {'cross':>9} {'gap':>8} {'cross+EA':>10} {'ΔEA':>8}")
    print("-" * 74)
    summary = []
    for sid in sorted(by):
        rs = by[sid]
        w = np.mean([r["within"] for r in rs])
        c = np.mean([r["cross"] for r in rs])
        e = np.mean([r["cross_ea"] for r in rs])
        sd = np.std([r["cross"] for r in rs])
        summary.append(dict(subject=sid, within=w, cross=c, cross_ea=e,
                            gap=w - c, d_ea=e - c, seed_sd=sd))
        print(f"{sid:>8} {w:9.3f} {c:9.3f} {w-c:+8.3f} {e:10.3f} {e-c:+8.3f}")
    print("-" * 74)
    W = np.mean([s["within"] for s in summary])
    C = np.mean([s["cross"] for s in summary])
    E = np.mean([s["cross_ea"] for s in summary])
    print(f"{'mean':>8} {W:9.3f} {C:9.3f} {W-C:+8.3f} {E:10.3f} {E-C:+8.3f}")
    print("=" * 74)

    d = np.array([s["d_ea"] for s in summary])
    noise = np.mean([s["seed_sd"] for s in summary])
    print(f"\nEA per-subject effect: min {d.min():+.3f}, max {d.max():+.3f}, "
          f"sd {d.std():.3f}")
    print(f"mean seed-to-seed sd:  {noise:.3f}")
    print(f"subjects helped by EA: {(d > 0).sum()}/9   hurt: {(d < 0).sum()}/9")
    verdict = ("heterogeneous — EA's effect varies by more than seed noise"
               if d.std() > 2 * noise else
               "not clearly heterogeneous at this sample size")
    print(f"H1 read: {verdict}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
