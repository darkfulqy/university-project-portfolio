#!/usr/bin/env python
"""Step 01: label-free subset selection on the training-subject pool of every GroupKFold(3) fold.

Methods (all operate on the pool only, labels are never used):
  proposed : official EEG-DLite `KCenterDistillation` path  = CRT multi-view SSL autoencoder (50 ep)
             -> HBOS outlier removal (tau)  -> k-center greedy (eta).  We drive the official
             functions step by step so that we obtain *indices* instead of data copies.
  pca_ds   : official 'SVD' path = IncrementalPCA(64) on flattened (sample-z-scored) segments
             -> same k-center greedy (tau = 0).  The official `reduce_dim` SVD branch crashes on
             numpy input (`np.vstack([])`), so the IncrementalPCA fit/transform is re-done here
             with the official settings (batch 512, n_components 64).
  random   : uniform sampling WITHOUT replacement (official label-free branch samples with
             replacement, see README).
  full     : all pool samples (eta = 1).

Outputs  {out_root}/selections/{dataset}/fold{f}_seed{s}_{method}_eta{eta}_tau{tau}.json
Caches   {out_root}/cache/{dataset}/fold{f}_seed{s}/  (sslmodel_*.pth, ssl_features.pt, hbos_*.npy, kcenter order)
         {out_root}/cache/{dataset}/fold{f}/          (svdmodel_*.pth, pca_features.npy, kcenter order)
Resumable: existing selection files / caches are reused.  --smoke: 1 SSL epoch, fold 0, seed 0, eta 0.25.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

SSL_DIM = 64


def log(msg: str):
    print(f"[01 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# official k-center (with a numpy-2 compatibility monkeypatch, third_party untouched)
# ---------------------------------------------------------------------------
def kcenter_order(features: np.ndarray, n_centers: int, seed: int):
    """Run the official `kCenterGreedy.select_batch_` and return the greedy order (list of row ids).

    Official code prints `'%0.2f' % max(self.min_distances)` where min_distances is (N, 1); under
    numpy>=2 this raises TypeError *after* the selection is complete.  We shadow `max` inside the
    official module with a scalar-returning version; the selection logic itself is unchanged.
    """
    M = common.import_official()
    import dataset_distillation.kcenter_greedy as kg

    if not getattr(kg, "_repro_patched", False):
        kg.max = lambda x: float(np.max(x))
        kg._repro_patched = True
    solver = kg.kCenterGreedy(np.ascontiguousarray(features, dtype=np.float32), None, seed, "euclidean")
    order = solver.select_batch_(model=None, already_selected=[], N=int(n_centers))
    return [int(i) for i in order]


def hbos_outliers(features: np.ndarray, tau: float) -> np.ndarray:
    """Exactly the official `KCenterDistillation.apply` HBOS branch."""
    from pyod.models.hbos import HBOS
    from pyod.utils.utility import standardizer

    fm = standardizer(features)
    det = HBOS(n_bins="auto", contamination=tau)
    det.fit(fm)
    return det.predict(fm, return_confidence=False).astype(bool)


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------
def ssl_features(X_pool: np.ndarray, subj_pool: np.ndarray, cache_dir: Path, fold: int, seed: int,
                 ssl_epochs: int, device: str) -> tuple[np.ndarray, str]:
    """CRT self-supervised features of the pool via the official `KCenterDistillation.reduce_dim`.
    Cached per (fold, seed) as cache_dir/ssl_features.pt; the SSL weights as sslmodel_<suffix>.pth."""
    feat_path = cache_dir / "ssl_features.pt"
    meta_path = cache_dir / "ssl_meta.json"      # what the cached features were actually made with
    suffix = f"{common.DATASET_NAME}_fold{fold}_seed{seed}"
    model_path = cache_dir / f"sslmodel_{suffix}.pth"
    import torch

    if feat_path.exists():
        f = torch.load(feat_path, map_location="cpu")
        meta = common.read_json(meta_path) if meta_path.exists() else dict(ssl_epochs=None, ssl_device=None)
        if meta.get("ssl_epochs") is not None and meta["ssl_epochs"] != ssl_epochs:
            log(f"WARNING: cached SSL features in {cache_dir} were trained with ssl_epochs={meta['ssl_epochs']}, "
                f"--ssl-epochs {ssl_epochs} requested; reusing the cache (delete the directory to retrain)")
        return np.asarray(f.numpy(), dtype=np.float32), meta

    M = common.import_official()
    from torch.utils.data import DataLoader, TensorDataset

    C, T = X_pool.shape[1], X_pool.shape[2]
    ssl_kwargs = {"seq_len": 2 * T, "patch_len": (2 * T) // 20, "dim": SSL_DIM, "num_class": 2, "in_dim": C}
    loader = DataLoader(TensorDataset(torch.from_numpy(X_pool), torch.from_numpy(subj_pool)),
                        batch_size=512, shuffle=False)

    def _run(dev: str):
        common.seed_everything(seed)
        cwd = os.getcwd()
        os.chdir(cache_dir)          # official code writes sslmodel_*.pth and <ts>_ssl_features.pt into cwd
        try:
            kc = M.KCenterDistillation(
                reduction_ratio=1.0, contamination=0.0, feature_extraction="SSL", ssl_type="CRT",
                ssl_epoch_num=ssl_epochs, batch_size=512, contamination_method="HBOS", device=dev,
                ssl_kwargs=ssl_kwargs, model_name_suffix=suffix,
                cache_path=str(model_path) if model_path.exists() else "")
            feats, _ = kc.reduce_dim(loader, k=SSL_DIM, sample_num=len(X_pool))
        finally:
            os.chdir(cwd)
        return feats

    t0 = time.time()
    used = device
    had_model = model_path.exists()
    try:
        feats = _run(device)
    except (RuntimeError, NotImplementedError) as e:      # e.g. an MPS op without kernel
        if device == "cpu":
            raise
        log(f"SSL on {device} failed ({type(e).__name__}: {str(e)[:120]}); retrying on cpu")
        used = "cpu"
        feats = _run("cpu")
    feats = feats.detach().cpu().float()
    # new features -> HBOS labels / k-center orders derived from older features in this dir are stale
    for stale in list(cache_dir.glob("hbos_tau*.npy")) + list(cache_dir.glob("kcenter_order_tau*.json")):
        stale.unlink()
        log(f"  removed stale cache {stale.name}")
    torch.save(feats, feat_path)
    if had_model:      # weights came from disk: keep their recorded epochs if we have them, else unknown
        old = common.read_json(meta_path) if meta_path.exists() else {}
        meta = dict(ssl_epochs=old.get("ssl_epochs"), ssl_device=used, ssl_weights_trained_now=False)
    else:
        meta = dict(ssl_epochs=ssl_epochs, ssl_device=used, ssl_weights_trained_now=True)
    common.write_json(meta_path, meta)
    log(f"SSL features {tuple(feats.shape)} on {used} in {time.time() - t0:.0f}s "
        f"({'loaded cached weights' if had_model else 'trained'} {model_path.name})")
    return feats.numpy(), meta


def pca_features(X_pool: np.ndarray, cache_dir: Path, fold: int) -> np.ndarray:
    """Official 'SVD' path settings: IncrementalPCA(n_components=64), partial_fit/transform in chunks of 512
    on the flattened samples.  Samples are z-scored per sample first (same normalisation the SSL path uses)."""
    feat_path = cache_dir / "pca_features.npy"
    if feat_path.exists():
        return np.load(feat_path)
    import joblib
    from sklearn.decomposition import IncrementalPCA

    t0 = time.time()
    Xn = common.zscore_per_sample(X_pool).reshape(len(X_pool), -1)
    model_path = cache_dir / f"svdmodel_{common.DATASET_NAME}_fold{fold}.pth"
    bs = 512
    if model_path.exists():
        svd = joblib.load(model_path)
    else:
        svd = IncrementalPCA(n_components=SSL_DIM)
        for i in range(0, len(Xn), bs):
            svd.partial_fit(Xn[i:i + bs])
        joblib.dump(svd, model_path)
    feats = np.vstack([svd.transform(Xn[i:i + bs]) for i in range(0, len(Xn), bs)]).astype(np.float32)
    stale = cache_dir / "pca_kcenter_order.json"        # derived from older features -> stale
    if stale.exists():
        stale.unlink()
        log("  removed stale cache pca_kcenter_order.json")
    np.save(feat_path, feats)
    log(f"PCA features {feats.shape} in {time.time() - t0:.0f}s")
    return feats


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(common.DATA_NPZ))
    ap.add_argument("--out-root", default=None, help="default runs/eegdlite (runs/eegdlite_smoke with --smoke)")
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--etas", default=",".join(map(str, common.ETAS_FULL)))
    ap.add_argument("--taus", default=",".join(map(str, common.TAUS_FULL)), help="for 'proposed' only")
    ap.add_argument("--methods", default="random,pca_ds,proposed,full")
    ap.add_argument("--ssl-epochs", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.folds, args.seeds, args.etas, args.ssl_epochs = "0", "0", "0.25", 1
    out_root = Path(args.out_root) if args.out_root else (common.RUNS_SMOKE_ROOT if args.smoke else common.RUNS_ROOT)
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    etas = [float(x) for x in args.etas.split(",") if x.strip()]
    taus = [float(x) for x in args.taus.split(",") if x.strip()]
    methods = [m for m in args.methods.split(",") if m]
    device = common.pick_device(args.device)
    ds = common.DATASET_NAME
    log(f"out_root={out_root} device={device} folds={folds} seeds={seeds} etas={etas} taus={taus} "
        f"methods={methods} ssl_epochs={args.ssl_epochs}")

    X, y, subject, session = common.load_data(Path(args.data))
    splits = common.make_folds(subject)
    t_all = time.time()

    for fold in folds:
        tr, te = splits[fold]
        X_pool, subj_pool = X[tr], subject[tr]
        n_pool = len(tr)
        info = dict(dataset=ds, fold=fold, n_pool=int(n_pool),
                    train_subjects=sorted(set(subj_pool.tolist())), test_subjects=sorted(set(subject[te].tolist())))
        log(f"fold {fold}: pool {n_pool} samples from subjects {info['train_subjects']}, test {info['test_subjects']}")

        def save(method, eta, tau, seed, pool_pos, extra=None):
            p = common.selection_path(out_root, ds, fold, seed, method, eta, tau)
            pool_pos = [int(i) for i in pool_pos]
            obj = dict(info, seed=seed, method=method, eta=eta, tau=tau, n_selected=len(pool_pos),
                       selected_pool_pos=pool_pos, selected_global_idx=[int(tr[i]) for i in pool_pos])
            if extra:
                obj.update(extra)
            common.write_json(p, obj)
            log(f"  wrote {p.name} ({len(pool_pos)} samples)")

        def todo(method, eta, tau, seed):
            return not common.selection_path(out_root, ds, fold, seed, method, eta, tau).exists()

        # ---------------- full ----------------
        if "full" in methods:
            for seed in seeds:
                if todo("full", 1.0, 0.0, seed):
                    save("full", 1.0, 0.0, seed, np.arange(n_pool))

        # ---------------- random (without replacement) ----------------
        if "random" in methods:
            for seed in seeds:
                for eta in etas:
                    if todo("random", eta, 0.0, seed):
                        # one permutation per seed, prefixes for the different eta (nested subsets)
                        perm = np.random.RandomState(seed).permutation(n_pool)
                        save("random", eta, 0.0, seed, perm[: int(n_pool * eta)])

        # ---------------- PCA + DS (tau = 0, seed-independent) ----------------
        if "pca_ds" in methods and any(todo("pca_ds", eta, 0.0, s) for eta in etas for s in seeds):
            cdir = out_root / "cache" / ds / f"fold{fold}"
            cdir.mkdir(parents=True, exist_ok=True)
            feats = pca_features(X_pool, cdir, fold)
            n_max = int(n_pool * max(etas))
            order_path = cdir / "pca_kcenter_order.json"
            order = common.read_json(order_path) if order_path.exists() else None
            if order is None or len(order) < n_max:
                t0 = time.time()
                order = kcenter_order(feats, n_max, seed=0)
                common.write_json(order_path, order)
                log(f"  PCA k-center greedy order ({n_max}) in {time.time() - t0:.0f}s")
            for seed in seeds:
                for eta in etas:
                    if todo("pca_ds", eta, 0.0, seed):
                        save("pca_ds", eta, 0.0, seed, order[: int(n_pool * eta)],
                             extra=dict(n_outliers_removed=0, feature="IncrementalPCA64"))

        # ---------------- Proposed (SSL -> HBOS -> k-center) ----------------
        if "proposed" in methods:
            for seed in seeds:
                if not any(todo("proposed", eta, tau, seed) for eta in etas for tau in taus):
                    continue
                cdir = out_root / "cache" / ds / f"fold{fold}_seed{seed}"
                cdir.mkdir(parents=True, exist_ok=True)
                feats, ssl_meta = ssl_features(X_pool, subj_pool, cdir, fold, seed, args.ssl_epochs, device)
                for tau in taus:
                    if not any(todo("proposed", eta, tau, seed) for eta in etas):
                        continue
                    # HBOS on *all* pool features (official), cached
                    if tau > 0:
                        hp = cdir / f"hbos_tau{common.fmt_tau(tau)}.npy"
                        if hp.exists():
                            outl = np.load(hp)
                        else:
                            outl = hbos_outliers(feats, tau)
                            np.save(hp, outl)
                    else:
                        outl = np.zeros(n_pool, dtype=bool)
                    keep_pos = np.flatnonzero(~outl)
                    # official: num_centers = int(sample_num * eta) with sample_num = pool size BEFORE removal
                    n_max = int(n_pool * max(etas))
                    op = cdir / f"kcenter_order_tau{common.fmt_tau(tau)}.json"
                    order = common.read_json(op) if op.exists() else None
                    if order is None or len(order) < n_max:
                        t0 = time.time()
                        order = kcenter_order(feats[keep_pos], n_max, seed=seed)
                        common.write_json(op, order)
                        log(f"  SSL k-center greedy order ({n_max}, tau={tau}, removed {int(outl.sum())}) "
                            f"in {time.time() - t0:.0f}s")
                    for eta in etas:
                        if todo("proposed", eta, tau, seed):
                            sel = keep_pos[np.asarray(order[: int(n_pool * eta)], dtype=int)]
                            save("proposed", eta, tau, seed, sel,
                                 extra=dict(n_outliers_removed=int(outl.sum()), feature="CRT-SSL64",
                                            ssl_epochs=ssl_meta.get("ssl_epochs"),      # actual, from ssl_meta.json
                                            ssl_device=ssl_meta.get("ssl_device")))
    log(f"done in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
