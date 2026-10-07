#!/usr/bin/env python
"""Reviewer sanity checks for the LGM->EEG pilot (server only, < 1 GB VRAM, ~3-4 min for all sections).

Reads the finished run in runs/lgm_eeg (phi + synsets) and re-evaluates a few things with
INDEPENDENT code paths so that the headline numbers do not rest on one implementation:
  A  phi.features() path == end-to-end forward path (the 47.0% test acc), feature scale
  B  RandomTimeCrop is the identity at scale=1/offset=0 (crop maths), stretch-factor statistics,
     8-30 Hz band fraction of real data before/after AugBasic
  C  Full-pool linear probe: official protocol (AugBasic) vs no augmentation vs sklearn LogisticRegression
  D  ipc=1 sets (LGM / no-pyramid / no-aug / Random x5 / Centroids) re-probed on EEGNet with a fresh head:
     official protocol (other probe seed), no augmentation, fixed lr, sklearn, cosine-to-prototype
  E  gradient-matching loss of the SAVED synsets vs Random real sets against fresh real batches / fresh heads
  F  band-power (8-30 Hz fraction) of pyramid vs no-pyramid synsets vs real data
  G  cross-model (ShallowConvNet, random EEGNet): ipc=1 sets + Random + Full, AugBasic probe vs no augmentation
Usage: python review_checks.py [--root DIR] [--only A,B,...]   (results merged into runs/lgm_eeg/results/review_checks.json)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lgm_eeg as P  # noqa: E402


def probe_noaug(phi, X, y, te_feat, y_te, epochs, lr, K=5, seed=3407):
    """Official-style probe (fresh heads, Adam, cosine) but WITHOUT augmentation: features precomputed once."""
    P.seed_all(seed)
    Z = P.features(phi, X)
    N, Fd = Z.shape
    bs = min(100, N)
    W = (torch.randn(K, P.NUM_CLASSES, Fd, device=Z.device) * 0.01).requires_grad_(True)
    b = torch.zeros(K, P.NUM_CLASSES, device=Z.device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        perms = torch.stack([torch.randperm(N, generator=g) for _ in range(K)]).to(Z.device)
        for i in range(0, N, bs):
            idx = perms[:, i:i + bs]
            z, yy = Z[idx], y[idx]                       # (K, m, F), (K, m)
            logits = torch.einsum("kmf,kcf->kmc", z, W) + b[:, None, :]
            loss = F.cross_entropy(logits.reshape(-1, P.NUM_CLASSES), yy.reshape(-1), reduction="none").view(K, -1).mean(1).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        pred = (torch.einsum("nf,kcf->knc", te_feat, W) + b[:, None, :]).argmax(2)
        return (pred == y_te[None]).float().mean(1).cpu().numpy()


def sklearn_probe(phi, X, y, te_feat, y_te, C=1.0):
    from sklearn.linear_model import LogisticRegression
    Z = P.features(phi, X).cpu().numpy()
    clf = LogisticRegression(C=C, max_iter=2000).fit(Z, y.cpu().numpy())
    return float((clf.predict(te_feat.cpu().numpy()) == y_te.cpu().numpy()).mean())


def cosine_prototype(phi, X, y, te_feat, y_te):
    with torch.no_grad():
        Z = P.features(phi, X)
        proto = torch.stack([Z[y == c].mean(0) for c in range(P.NUM_CLASSES)])
        pred = (F.normalize(te_feat, dim=1) @ F.normalize(proto, dim=1).T).argmax(1)
        return float((pred == y_te).float().mean())


def band_fraction(x: np.ndarray, fs=200.0, lo=8.0, hi=30.0):
    """fraction of total power (>0.5 Hz) in [lo, hi] Hz, averaged over samples and channels."""
    f = np.fft.rfftfreq(x.shape[-1], 1 / fs)
    p = np.abs(np.fft.rfft(x - x.mean(-1, keepdims=True), axis=-1)) ** 2
    tot = p[..., (f > 0.5)].sum(-1)
    band = p[..., (f >= lo) & (f <= hi)].sum(-1)
    return float((band / np.maximum(tot, 1e-12)).mean())


def load_sets(root, data, phi):
    sets = {}
    for name in ("lgm_ipc1", "lgm_ipc1_nopyramid", "lgm_ipc1_noaug"):
        d = torch.load(root / "synsets" / name / "data.pt", map_location="cpu")
        sets[name] = (d["x"].to(data.device), d["y"].to(data.device))
    for s_ in range(5):
        sets[f"random_ipc1_seed{s_}"] = P.random_reals(data, 1, s_)
    sets["centroids_ipc1"] = P.centroid_reals(data, phi, 1)
    return sets


def sec_A(out, data, phi, te_feat):
    with torch.no_grad():
        logits_fwd = torch.cat([phi(data.X_test[i:i + 256]) for i in range(0, len(data.X_test), 256)])
        logits_feat = phi.classifier(te_feat)
    out["A_end2end_acc_forward"] = float((logits_fwd.argmax(1) == data.y_test).float().mean())
    out["A_end2end_acc_via_features"] = float((logits_feat.argmax(1) == data.y_test).float().mean())
    out["A_max_abs_logit_diff"] = float((logits_fwd - logits_feat).abs().max())
    out["A_feat_mean_abs"] = float(te_feat.abs().mean())
    out["A_feat_mean_norm"] = float(te_feat.norm(dim=1).mean())
    out["A_test_class_counts"] = torch.bincount(data.y_test).tolist()


def sec_B(out, data):
    x = data.X_pool[:8]
    theta = torch.zeros(8, 2, 3, device=x.device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    grid = F.affine_grid(theta, (8, data.C, 1, P.T_LEN), align_corners=False)
    xi = F.grid_sample(x.unsqueeze(2), grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(2)
    out["B_crop_identity_max_abs_diff"] = float((xi - x).abs().max())
    s = torch.rand(100000) * 0.5 + 0.5
    out["B_mean_time_stretch_factor(1/s)"] = float((1 / s).mean())      # frequencies are divided by this
    out["B_P(stretch>1.5)"] = float(((1 / s) > 1.5).float().mean())
    aug = P.make_augmentor("basic").to(x.device)
    P.seed_all(0)
    xr = data.X_pool[:256]
    out["B_bandfrac_8_30_real"] = band_fraction(xr.cpu().numpy())
    out["B_bandfrac_8_30_real_after_AugBasic"] = band_fraction(aug(xr).cpu().numpy())


def sec_C(out, data, phi, te_feat, device):
    res = P.linear_probe(phi, [data.X_pool], [data.y_pool], te_feat, data.y_test, 100, device, lr_rule="linear")
    out["C_full_official_AugBasic_K1"] = res["acc"]
    res = P.linear_probe(phi, [data.X_pool], [data.y_pool], te_feat, data.y_test, 100, device, lr_rule="linear", aug="none")
    out["C_full_official_lr_noaug_K1"] = res["acc"]
    acc = probe_noaug(phi, data.X_pool, data.y_pool, te_feat, data.y_test, 100, 1e-3, K=3)
    out["C_full_noaug_fixed1e-3_K3"] = [float(a) for a in acc]
    out["C_full_sklearn_logreg_C1"] = sklearn_probe(phi, data.X_pool, data.y_pool, te_feat, data.y_test)
    out["C_full_sklearn_logreg_C0.01"] = sklearn_probe(phi, data.X_pool, data.y_pool, te_feat, data.y_test, C=0.01)


def sec_D(out, data, phi, te_feat, sets, device):
    D = {}
    for name, (X, y) in sets.items():
        r = {}
        rr = P.linear_probe(phi, [X] * 5, [y] * 5, te_feat, data.y_test, 1000, device, seed=777, lr_rule="linear")
        r["official_seed777_mean"], r["official_seed777_std"] = rr["acc"], rr["acc_std"]
        rr = P.linear_probe(phi, [X] * 5, [y] * 5, te_feat, data.y_test, 1000, device, lr_rule="linear", aug="none")
        r["noaug_linearlr_mean"] = rr["acc"]
        r["noaug_fixed1e-3_mean"] = float(probe_noaug(phi, X, y, te_feat, data.y_test, 1000, 1e-3, K=5).mean())
        r["sklearn_C1"] = sklearn_probe(phi, X, y, te_feat, data.y_test)
        r["sklearn_C0.01"] = sklearn_probe(phi, X, y, te_feat, data.y_test, C=0.01)
        r["cosine_prototype"] = cosine_prototype(phi, X, y, te_feat, data.y_test)
        D[name] = r
        print("D", name, r, flush=True)
    out["D"] = D
    rnd = [D[f"random_ipc1_seed{s_}"] for s_ in range(5)]
    out["D_random_mean"] = {k: float(np.mean([r[k] for r in rnd])) for k in rnd[0]}
    for name in ("lgm_ipc1", "lgm_ipc1_nopyramid", "lgm_ipc1_noaug"):
        d = torch.load(Path(out["_root"]) / "synsets" / name / "data.pt", map_location="cpu")
        out[f"D_{name}_x_absmax"], out[f"D_{name}_x_std"] = float(d["x"].abs().max()), float(d["x"].std())


def sec_E(out, data, phi, sets, device):
    P.seed_all(1)
    E = {}
    sampler = P.RealSampler(data.X_pool, data.y_pool, 40, 1)
    batches = [sampler.next() for _ in range(20)]
    for name in ("lgm_ipc1", "lgm_ipc1_nopyramid", "lgm_ipc1_noaug", "random_ipc1_seed0", "random_ipc1_seed1", "centroids_ipc1"):
        X, y = sets[name]
        with torch.no_grad():
            z_syn = phi.features(X)
        vals = []
        for xb, yb in batches:
            with torch.no_grad():
                z_real = phi.features(xb)
            fc = P.new_fc(z_real.shape[1], device)
            gr = torch.autograd.grad(F.cross_entropy(fc(z_real), yb), [fc.weight, fc.bias])
            gs = torch.autograd.grad(F.cross_entropy(fc(z_syn), y), [fc.weight, fc.bias])
            gr = torch.cat([g.flatten() for g in gr])
            gs = torch.cat([g.flatten() for g in gs])
            vals.append(float(1 - F.cosine_similarity(gr, gs, dim=0)))
        E[name] = dict(mean_1_minus_cos=float(np.mean(vals)), std=float(np.std(vals)))
    out["E"] = E


def sec_F(out, data, sets):
    Fo = {name: band_fraction(sets[name][0].cpu().numpy()) for name in ("lgm_ipc1", "lgm_ipc1_nopyramid", "lgm_ipc1_noaug")}
    Fo["real_pool_first256"] = band_fraction(data.X_pool[:256].cpu().numpy())
    out["F_bandfrac_8_30"] = Fo


def sec_G(out, root, data, sets, device):
    G = {}
    for m in ("shallow", "eegnet_random"):
        phi_m = P.load_phi(root, m, data)
        te_m = P.features(phi_m, data.X_test)
        Gm = {}
        for name in ("lgm_ipc1", "lgm_ipc1_nopyramid", "lgm_ipc1_noaug", "centroids_ipc1") + tuple(f"random_ipc1_seed{s_}" for s_ in range(5)):
            X, y = sets[name]
            r = {}
            rr = P.linear_probe(phi_m, [X] * 5, [y] * 5, te_m, data.y_test, 1000, device, lr_rule="linear")
            r["official_AugBasic_mean"], r["official_AugBasic_std"] = rr["acc"], rr["acc_std"]
            rr = P.linear_probe(phi_m, [X] * 5, [y] * 5, te_m, data.y_test, 1000, device, lr_rule="linear", aug="none")
            r["noaug_linearlr_mean"] = rr["acc"]
            r["sklearn_C1"] = sklearn_probe(phi_m, X, y, te_m, data.y_test)
            r["cosine_prototype"] = cosine_prototype(phi_m, X, y, te_m, data.y_test)
            Gm[name] = r
            print("G", m, name, r, flush=True)
        rnd = [Gm[f"random_ipc1_seed{s_}"] for s_ in range(5)]
        Gm["random_ipc1_mean"] = {k: float(np.mean([r[k] for r in rnd])) for k in rnd[0]}
        rr = P.linear_probe(phi_m, [data.X_pool], [data.y_pool], te_m, data.y_test, 100, device, lr_rule="linear")
        Gm["full"] = dict(official_AugBasic_K1=rr["acc"])
        rr = P.linear_probe(phi_m, [data.X_pool], [data.y_pool], te_m, data.y_test, 100, device, lr_rule="linear", aug="none")
        Gm["full"]["noaug_linearlr_K1"] = rr["acc"]
        Gm["full"]["sklearn_C1"] = sklearn_probe(phi_m, data.X_pool, data.y_pool, te_m, data.y_test)
        print("G", m, "full", Gm["full"], "random mean", Gm["random_ipc1_mean"], flush=True)
        G[m] = Gm
    out["G"] = G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(P.RUNS_ROOT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default="A,B,C,D,E,F,G")
    args = ap.parse_args()
    only = set(args.only.split(","))
    root = Path(args.root)
    res_json = root / "results" / "review_checks.json"
    out = json.loads(res_json.read_text()) if res_json.exists() else {}
    out["_root"] = str(root)
    t0 = time.time()
    data = P.Data(P.DATA_NPZ, args.device)
    phi = P.load_phi(root, "eegnet", data)
    te_feat = P.features(phi, data.X_test)
    sets = load_sets(root, data, phi)

    if "A" in only:
        sec_A(out, data, phi, te_feat)
        print("A", {k: v for k, v in out.items() if k.startswith("A_")}, flush=True)
    if "B" in only:
        sec_B(out, data)
        print("B", {k: v for k, v in out.items() if k.startswith("B_")}, flush=True)
    if "C" in only:
        sec_C(out, data, phi, te_feat, args.device)
        print("C", {k: v for k, v in out.items() if k.startswith("C_")}, flush=True)
    if "D" in only:
        sec_D(out, data, phi, te_feat, sets, args.device)
        print("D random mean", out["D_random_mean"], flush=True)
    if "E" in only:
        sec_E(out, data, phi, sets, args.device)
        print("E", out["E"], flush=True)
    if "F" in only:
        sec_F(out, data, sets)
        print("F", out["F_bandfrac_8_30"], flush=True)
    if "G" in only:
        sec_G(out, root, data, sets, args.device)

    out["seconds_last_run"] = time.time() - t0
    out["peak_mem_GB_last_run"] = torch.cuda.max_memory_allocated() / 2**30
    res_json.parent.mkdir(parents=True, exist_ok=True)
    res_json.write_text(json.dumps(out, indent=1))
    print("done", out["seconds_last_run"], "s, peak", out["peak_mem_GB_last_run"], "GB ->", res_json)


if __name__ == "__main__":
    main()
