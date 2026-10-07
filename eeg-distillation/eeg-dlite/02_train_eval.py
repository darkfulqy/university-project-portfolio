#!/usr/bin/env python
"""Step 02: train EEGNet-8,2 on the *labelled* selected subset of each selection file and evaluate on the
held-out test subjects of that fold (accuracy, macro-F1, Cohen's kappa).

Protocol (paper gives none for the pilot study; see README):
  EEGNet(F1=8, D=2, F2=16, kernLength=fs//2=100, dropout=0.5, 4 classes), Adam 1e-3, batch 64,
  fixed 100 epochs, cross-entropy, no early stopping / no validation on the test subjects,
  per-sample z-score of the input (same as the distillation side), model of the LAST epoch evaluated.

Every selection json under {out_root}/selections/{dataset}/ matching the filters becomes one row of
{out_root}/results.csv (append; a row whose key already exists is skipped -> resumable and safe to run
one process per fold in parallel:  --folds 0 | --folds 1 | --folds 2).
--smoke: fold 0, seed 0, eta 0.25 (+ full), 2 epochs, out_root runs/eegdlite_smoke.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from eegnet import EEGNet  # noqa: E402

FIELDS = ["dataset", "fold", "seed", "method", "eta", "tau", "n_train", "n_test", "epochs",
          "acc", "f1_macro", "kappa", "train_loss_last", "train_acc_last", "train_sec", "device", "finished_at"]
KEY = ["dataset", "fold", "seed", "method", "eta", "tau"]


def log(msg: str):
    print(f"[02 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def row_key(r: dict) -> tuple:
    return (r["dataset"], int(r["fold"]), int(r["seed"]), r["method"], f"{float(r['eta']):g}", f"{float(r['tau']):g}")


def existing_rows(csv_path: Path) -> dict:
    """key -> epochs of the rows already in results.csv (the key itself does not contain epochs)."""
    if not csv_path.exists():
        return {}
    with open(csv_path) as f:
        return {row_key(r): int(r["epochs"]) for r in csv.DictReader(f)}


def append_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            new = f.tell() == 0
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def train_eval(X_tr, y_tr, X_te, y_te, seed: int, epochs: int, device: str, batch_size: int = 64, lr: float = 1e-3):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    common.seed_everything(seed)
    C, T = X_tr.shape[1], X_tr.shape[2]
    fs = 200                                   # data sampling rate after step 00
    model = EEGNet(chunk_size=T, num_electrodes=C, F1=8, D=2, F2=16, num_classes=4,
                   kernel_1=fs // 2, kernel_2=16, dropout=0.5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(common.zscore_per_sample(X_tr)).to(device)
    yt = torch.from_numpy(y_tr).to(device)
    Xe = torch.from_numpy(common.zscore_per_sample(X_te)).to(device)
    g = torch.Generator().manual_seed(seed)
    n = len(Xt)
    last_loss = last_acc = float("nan")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        tot_loss = torch.zeros((), device=device)
        tot_correct = torch.zeros((), device=device, dtype=torch.long)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:          # BatchNorm needs >1 sample
                continue
            out = model(Xt[idx])
            loss = F.cross_entropy(out, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.apply_max_norm()
            tot_loss += loss.detach() * len(idx)          # accumulate on device: one host sync per epoch
            tot_correct += (out.argmax(1) == yt[idx]).sum()
        last_loss, last_acc = tot_loss.item() / n, tot_correct.item() / n
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xe), 256):
            preds.append(model(Xe[i:i + 256]).argmax(1).cpu())
    pred = torch.cat(preds).numpy()
    return dict(acc=accuracy_score(y_te, pred), f1_macro=f1_score(y_te, pred, average="macro"),
                kappa=cohen_kappa_score(y_te, pred), train_loss_last=last_loss, train_acc_last=last_acc,
                train_sec=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(common.DATA_NPZ))
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--methods", default="random,pca_ds,proposed,full")
    ap.add_argument("--etas", default=None, help="filter, e.g. 0.25,0.5 (default: all found)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.folds, args.seeds, args.epochs = "0", "0", 2
    out_root = Path(args.out_root) if args.out_root else (common.RUNS_SMOKE_ROOT if args.smoke else common.RUNS_ROOT)
    folds = {int(x) for x in args.folds.split(",") if x.strip()}
    seeds = {int(x) for x in args.seeds.split(",") if x.strip()}
    methods = {m for m in args.methods.split(",") if m}
    etas = {float(x) for x in args.etas.split(",")} if args.etas else None
    device = common.pick_device(args.device)
    csv_path = out_root / "results.csv"
    log(f"out_root={out_root} device={device} epochs={args.epochs} folds={sorted(folds)} seeds={sorted(seeds)}")

    X, y, subject, session = common.load_data(Path(args.data))
    splits = common.make_folds(subject)
    sel_dir = out_root / "selections" / common.DATASET_NAME
    files = sorted(sel_dir.glob("*.json"))
    jobs = []
    for p in files:
        m = common.parse_selection_name(p)
        if m["fold"] in folds and m["seed"] in seeds and m["method"] in methods and (etas is None or m["eta"] in etas or m["method"] == "full"):
            jobs.append((p, m))
    done = existing_rows(csv_path)
    log(f"{len(files)} selection files, {len(jobs)} match filters, {len(done)} rows already in {csv_path.name}")

    for p, m in jobs:
        key = (common.DATASET_NAME, m["fold"], m["seed"], m["method"], f"{m['eta']:g}", f"{m['tau']:g}")
        if key in done:
            if done[key] != args.epochs:
                log(f"WARNING: {p.stem} already in {csv_path.name} with epochs={done[key]} "
                    f"(requested {args.epochs}); skipping - delete that row to redo it")
            continue
        sel = common.read_json(p)
        tr_idx = np.asarray(sel["selected_global_idx"], dtype=int)
        te_idx = splits[m["fold"]][1]
        assert not set(subject[tr_idx]) & set(subject[te_idx]), "train/test subject leak"
        res = train_eval(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], m["seed"], args.epochs, device)
        row = dict(dataset=common.DATASET_NAME, fold=m["fold"], seed=m["seed"], method=m["method"],
                   eta=f"{m['eta']:g}", tau=f"{m['tau']:g}", n_train=len(tr_idx), n_test=len(te_idx),
                   epochs=args.epochs, device=device, finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                   **{k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in res.items()})
        append_row(csv_path, row)
        done[key] = args.epochs
        log(f"{p.stem}: n_train={len(tr_idx)} acc={res['acc']:.3f} f1={res['f1_macro']:.3f} "
            f"kappa={res['kappa']:.3f} ({res['train_sec']:.0f}s)")
    log("done")


if __name__ == "__main__":
    main()
