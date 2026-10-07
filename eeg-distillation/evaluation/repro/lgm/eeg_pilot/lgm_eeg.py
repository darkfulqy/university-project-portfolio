#!/usr/bin/env python
"""LGM (Linear Gradient Matching, NeurIPS 2025) ported component-by-component to EEG waveforms.

BCI IV-2a, GroupKFold(3) fold 0 (6 train subjects / 3 test subjects, same split as eeg-dlite),
frozen feature extractor phi + linear probe, exactly mirroring the official
third_party/linear-gradient-matching/src/{distillation/linear_gm.py, synsets/pyramid.py,
synsets/base.py, augmentation/ops/*, distillation/eval.py, baselines/*}. Nothing under
third_party is imported or modified; the mapping of every component is in README.md.

Stages (all resumable: an output that already exists is skipped):
  phi        train EEGNet-8,2 and ShallowConvNet on the train pool (val subject for early stopping);
             a random (untrained) EEGNet is built on the fly from a fixed seed
  distill    LGM distillation (pyramid + channel decorrelation + differentiable augs), grid in run_all.sh
  eval       linear probe (official eval.py protocol) of every synset on every eval model
  baselines  Random (5 seeds) / Centroids / Full linear probes
  summarize  results.csv -> results/table.md + figures

Usage: python lgm_eeg.py <stage> [--smoke] [options]   (see run_all.sh)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent                 # $R/lgm/eeg_pilot
R = HERE.parent.parent                                 # $R
sys.path.insert(0, str(R / "eeg-dlite"))
import common  # noqa: E402  (eeg-dlite/common.py: load_data, make_folds, zscore_per_sample)
from eegnet import EEGNet  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

if Path("/path/to/external-workspace/eegdd").is_dir():
    DATA_NPZ = Path("/path/to/external-workspace/eegdd/data/eegdlite_bciiv2a.npz")
    RUNS_ROOT = Path("/path/to/external-workspace/eegdd/runs/lgm_eeg")
else:
    DATA_NPZ = R / "data" / "eegdlite_bciiv2a.npz"
    RUNS_ROOT = HERE / "runs" / "lgm_eeg"

NUM_CLASSES = 4
CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]
T_LEN = 800
PYRAMID_RES = [25, 50, 100, 200, 400, 800]
EVAL_MODELS = ["eegnet", "shallow", "eegnet_random"]
RANDOM_PHI_SEED = 12345


def log(msg: str):
    print(f"[lgm_eeg {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class Data:
    """Train pool / val subject / test subjects of GroupKFold(3) fold 0, per-sample z-scored, on GPU."""

    def __init__(self, path: Path, device: str, fold: int = 0):
        X, y, subject, _ = common.load_data(path)
        tr, te = common.make_folds(subject)[fold]
        pool_subjects = sorted(set(subject[tr].tolist()))
        self.val_subject = pool_subjects[-1]                 # deterministic: largest id of the pool
        self.pool_subjects, self.test_subjects = pool_subjects, sorted(set(subject[te].tolist()))
        Xz = common.zscore_per_sample(X)
        self.device = device
        self.X_pool = torch.from_numpy(Xz[tr]).to(device)      # (3456, 22, 800)  distillation source
        self.y_pool = torch.from_numpy(y[tr]).to(device)
        self.subj_pool = subject[tr]
        self.X_test = torch.from_numpy(Xz[te]).to(device)
        self.y_test = torch.from_numpy(y[te]).to(device)
        fit = self.subj_pool != self.val_subject
        self.X_fit, self.y_fit = self.X_pool[torch.from_numpy(fit).to(device)], self.y_pool[torch.from_numpy(fit).to(device)]
        self.X_val, self.y_val = self.X_pool[torch.from_numpy(~fit).to(device)], self.y_pool[torch.from_numpy(~fit).to(device)]
        self.C = self.X_pool.shape[1]
        d = np.load(path, allow_pickle=False)
        self.ch_names = [str(c) for c in d["ch_names"]]

    def summary(self):
        return dict(pool_subjects=self.pool_subjects, val_subject=self.val_subject, test_subjects=self.test_subjects,
                    n_pool=int(len(self.X_pool)), n_fit=int(len(self.X_fit)), n_val=int(len(self.X_val)),
                    n_test=int(len(self.X_test)))

    @torch.no_grad()
    def channel_decorrelation(self) -> torch.Tensor:
        """Cholesky factor L of the mean per-sample channel second-moment matrix (1/T) X X^T of the z-scored
        train pool (the per-sample z-score is over all C*T values, so per-channel means are NOT removed; this is
        the uncentred 'covariance'), divided by its max column norm (official: color_correlation_svd_sqrt /
        max_norm_svd_sqrt). Test subjects are never used here."""
        cov = torch.zeros(self.C, self.C, device=self.device, dtype=torch.float64)
        for i in range(0, len(self.X_pool), 256):
            xb = self.X_pool[i:i + 256].double()
            cov += torch.einsum("bct,bdt->cd", xb, xb) / xb.shape[-1]
        cov /= len(self.X_pool)
        L = torch.linalg.cholesky(cov)
        L = L / torch.linalg.norm(L, dim=0).max()
        return L.float()


# ---------------------------------------------------------------------------
# models (feature extractors)
# ---------------------------------------------------------------------------
class ShallowConvNet(nn.Module):
    """ShallowConvNet from third_party/TGA-repo/tga/runner.py (4 classes); features = flatten before classifier."""

    def __init__(self, C: int, T: int, n_classes: int = 4, F_: int = 40, dropout: float = 0.5):
        super().__init__()
        self.conv_time = nn.Conv2d(1, F_, kernel_size=(1, 25), padding=(0, 12), bias=False)
        self.conv_spat = nn.Conv2d(F_, F_, kernel_size=(C, 1), bias=False)
        self.bn = nn.BatchNorm2d(F_)
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.drop = nn.Dropout(dropout)
        with torch.no_grad():
            flat = self.features(torch.zeros(1, C, T)).shape[1]
        self.classifier = nn.Linear(flat, n_classes)

    def features(self, x):
        x = x.unsqueeze(1)
        x = self.conv_spat(self.conv_time(x))
        x = torch.square(self.bn(x))
        x = torch.log(torch.clamp(self.pool(x), min=1e-6))
        return torch.flatten(self.drop(x), start_dim=1)

    def forward(self, x):
        return self.classifier(self.features(x))


class EEGNetFeat(EEGNet):
    """eeg-dlite EEGNet-8,2 with a `features` method (flatten after block2 = F2*T/32 = 400 dims)."""

    def features(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.block2(self.block1(x)).flatten(1)


def build_model(name: str, C: int, T: int) -> nn.Module:
    if name in ("eegnet", "eegnet_random"):
        return EEGNetFeat(chunk_size=T, num_electrodes=C, F1=8, D=2, F2=16, num_classes=NUM_CLASSES,
                          kernel_1=100, kernel_2=16, dropout=0.5)
    if name == "shallow":
        return ShallowConvNet(C, T, n_classes=NUM_CLASSES)
    raise ValueError(name)


def phi_path(root: Path, name: str) -> Path:
    return root / "phi" / f"{name}.pt"


def load_phi(root: Path, name: str, data: Data) -> nn.Module:
    """Frozen feature extractor in eval mode (BatchNorm uses running stats, dropout off: official model.eval())."""
    model = build_model(name, data.C, T_LEN)
    if name == "eegnet_random":
        seed_all(RANDOM_PHI_SEED)
        model = build_model(name, data.C, T_LEN)          # fresh init under the fixed seed
    else:
        model.load_state_dict(torch.load(phi_path(root, name), map_location="cpu"))
    model.to(data.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def features(model: nn.Module, X: torch.Tensor, bs: int = 256) -> torch.Tensor:
    return torch.cat([model.features(X[i:i + bs]) for i in range(0, len(X), bs)])


def train_phi(root: Path, name: str, data: Data, max_epochs: int, seed: int = 0, patience: int = 15,
              batch_size: int = 64, lr: float = 1e-3):
    out = phi_path(root, name)
    if out.exists():
        log(f"phi {name}: exists, skip")
        return
    seed_all(seed)
    model = build_model(name, data.C, T_LEN).to(data.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = len(data.X_fit)
    best_acc, best_state, best_ep, bad = -1.0, None, -1, 0
    hist = []
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(data.device)
        tot = torch.zeros((), device=data.device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(model(data.X_fit[idx]), data.y_fit[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if hasattr(model, "apply_max_norm"):
                model.apply_max_norm()
            tot += loss.detach() * len(idx)
        model.eval()
        with torch.no_grad():
            pred_val = torch.cat([model(data.X_val[i:i + 256]).argmax(1) for i in range(0, len(data.X_val), 256)])
            val_acc = (pred_val == data.y_val).float().mean().item()
        hist.append(dict(epoch=ep, train_loss=tot.item() / n, val_acc=val_acc))
        if val_acc > best_acc:
            best_acc, best_ep, bad = val_acc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or bad == 0:
            log(f"phi {name} ep {ep:3d} loss {tot.item() / n:.3f} val_acc {val_acc:.3f} (best {best_acc:.3f} @ {best_ep})")
        if bad >= patience:
            log(f"phi {name}: early stop at epoch {ep} (patience {patience})")
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(data.X_test[i:i + 256]).argmax(1) for i in range(0, len(data.X_test), 256)])
    test_acc = (pred == data.y_test).float().mean().item()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in best_state.items()}, out)
    write_json(out.with_suffix(".json"), dict(name=name, best_val_acc=best_acc, best_epoch=best_ep,
                                              epochs_run=len(hist), end2end_test_acc=test_acc,
                                              feat_dim=int(model.features(data.X_val[:2]).shape[1]),
                                              train_sec=time.time() - t0, history=hist))
    log(f"phi {name}: best val {best_acc:.3f} @ ep {best_ep}, end-to-end test acc {test_acc:.3f}, {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# differentiable augmentations (official augmentation/ops/* -> 1-D)
# ---------------------------------------------------------------------------
class RandomTimeCrop(nn.Module):
    """1-D RandomResizedCrop: crop a random window of length ratio~U(lo,hi) at a random start and
    resample it to T (bilinear grid_sample on a (B,C,1,T) 'image', like the official affine_grid crop)."""

    def __init__(self, size: int = T_LEN, scale=(0.5, 1.0)):
        super().__init__()
        self.size, self.scale = size, scale

    def forward(self, x):
        B, C, T = x.shape
        with torch.no_grad():
            r = torch.rand(B, 2, device=x.device)
            s = r[:, 0] * (self.scale[1] - self.scale[0]) + self.scale[0]
            x1 = r[:, 1] * (1 - s)
            off = (x1 + s / 2 - 0.5) * 2
        theta = torch.zeros(B, 2, 3, dtype=x.dtype, device=x.device)
        theta[:, 0, 0] = s
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = off
        grid = F.affine_grid(theta, (B, C, 1, self.size), align_corners=False)
        out = F.grid_sample(x.unsqueeze(2), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return out.squeeze(2)


class RandomAmplitudeScale(nn.Module):
    """Per-sample gain ~U(lo, hi): the 'lossless' counterpart of RandomHorizontalFlip (EEG has no L/R flip)."""

    def __init__(self, lo: float = 0.8, hi: float = 1.2):
        super().__init__()
        self.lo, self.hi = lo, hi

    def forward(self, x):
        with torch.no_grad():
            g = torch.rand(x.shape[0], 1, 1, device=x.device) * (self.hi - self.lo) + self.lo
        return x * g


class RandomGaussianNoise(nn.Module):
    """Official RandomGaussianNoise (std 0.2, per-sample p=0.5), 1-D."""

    def __init__(self, std: float = 0.2, p: float = 0.5):
        super().__init__()
        self.std, self.p = std, p

    def forward(self, x):
        with torch.no_grad():
            apply = (torch.rand(x.shape[0], device=x.device) < self.p).float().view(-1, 1, 1)
        if apply.any():
            x = x + apply * torch.randn_like(x) * self.std
        return x


def make_augmentor(mode: str) -> nn.Module:
    if mode == "standard":      # official AugStandard: flip + RRC + noise
        return nn.Sequential(RandomAmplitudeScale(), RandomTimeCrop(), RandomGaussianNoise())
    if mode == "basic":         # official AugBasic: flip + RRC (used by the linear probe)
        return nn.Sequential(RandomAmplitudeScale(), RandomTimeCrop())
    if mode == "none":
        return nn.Identity()
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# synthetic set: time pyramid + channel decorrelation + soft clamp (official synsets/pyramid.py + base.py)
# ---------------------------------------------------------------------------
class PyramidSynset:
    def __init__(self, ipc: int, C: int, L: torch.Tensor, lr: float, device: str, pyramid: bool = True,
                 extend_it: int = 200, clamp: float = 3.0):
        self.ipc, self.C, self.L, self.lr, self.device = ipc, C, L, lr, device
        self.use_pyramid, self.extend_it, self.clamp = pyramid, extend_it, clamp
        self.labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(ipc)
        n = ipc * NUM_CLASSES
        self.res_list = list(PYRAMID_RES) if pyramid else [T_LEN]
        self.levels = [torch.randn(n, C, self.res_list[0], device=device)]      # N(0,1)/len(pyramid), len=1
        self.levels[0].requires_grad_(True)
        self.optimizer = self.init_optimizer()

    def init_optimizer(self):
        return torch.optim.Adam([{"params": p, "lr": self.lr} for p in self.levels])

    def extend(self) -> bool:
        old_len, new_len = len(self.levels), len(self.levels) + 1
        if old_len >= len(self.res_list):
            return False
        new_res = self.res_list[old_len]
        n = self.levels[0].shape[0]
        self.levels = [p.detach().clone() * old_len / new_len for p in self.levels]   # official re-normalisation
        self.levels.append(torch.randn(n, self.C, new_res, device=self.device) / new_len)
        for p in self.levels:
            p.requires_grad_(True)
        self.optimizer = self.init_optimizer()
        log(f"pyramid extended -> res {[p.shape[-1] for p in self.levels]}")
        return True

    def upkeep(self, step: int):
        if self.use_pyramid and step > 1 and (step - 1) % self.extend_it == 0:
            self.extend()

    def render(self) -> torch.Tensor:
        x = torch.stack([F.interpolate(p, size=T_LEN, mode="linear", align_corners=False) for p in self.levels]).sum(0)
        x = torch.einsum("cd,bdt->bct", self.L, x)                      # channel decorrelation (fixed linear map)
        return self.clamp * torch.tanh(x / self.clamp)                  # soft amplitude clamp (official sigmoid(2x))

    def get_data(self):
        return self.render(), self.labels

    def state(self):
        return dict(levels=[p.detach().cpu() for p in self.levels], opt=self.optimizer.state_dict())


# ---------------------------------------------------------------------------
# linear gradient matching (official distillation/linear_gm.py)
# ---------------------------------------------------------------------------
def new_fc(num_feats: int, device: str) -> nn.Linear:
    fc = nn.Linear(num_feats, NUM_CLASSES).to(device)     # official LinearClassifier init
    fc.weight.data.normal_(0.0, 0.01)
    fc.bias.data.zero_()
    return fc


class RealSampler:
    """Shuffled epochs with drop_last, like the official DataLoader(shuffle=True, drop_last=True)."""

    def __init__(self, X, y, batch_size: int, seed: int):
        self.X, self.y, self.bs = X, y, batch_size
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.perm, self.pos = None, 0

    def next(self):
        if self.perm is None or self.pos + self.bs > len(self.perm):
            self.perm = torch.randperm(len(self.X), generator=self.g).to(self.X.device)
            self.pos = 0
        idx = self.perm[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return self.X[idx], self.y[idx]


def distill(root: Path, name: str, data: Data, phi: nn.Module, ipc: int, iterations: int, lr: float,
            augs_per_batch: int = 10, pyramid: bool = True, aug: str = "standard", seed: int = 0,
            extend_it: int = 200):
    out_dir = root / "synsets" / name
    if (out_dir / "data.pt").exists():
        log(f"distill {name}: exists, skip")
        return
    seed_all(seed)
    L = data.channel_decorrelation()
    syn = PyramidSynset(ipc, data.C, L, lr, data.device, pyramid=pyramid, extend_it=extend_it)
    aug_syn, aug_real = make_augmentor(aug).to(data.device), make_augmentor(aug).to(data.device)
    real_bs = ipc * augs_per_batch * NUM_CLASSES
    sampler = RealSampler(data.X_pool, data.y_pool, real_bs, seed)
    num_feats = phi.features(data.X_pool[:2]).shape[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    losses = []
    t0 = time.time()
    for step in range(iterations + 1):
        syn.upkeep(step)
        fc = new_fc(num_feats, data.device)
        # --- real side: d l_real / d(W,b); phi's graph is not needed (grad only wrt the head) ---
        x_real, y_real = sampler.next()
        with torch.no_grad():
            z_real = phi.features(aug_real(x_real))
        loss_real = F.cross_entropy(fc(z_real), y_real)
        gw, gb = torch.autograd.grad(loss_real, [fc.weight, fc.bias], create_graph=False)
        grad_real = torch.cat([gw.detach().flatten(), gb.detach().flatten()])
        # --- synthetic side: augs_per_batch different augmentations of the rendered synset ---
        x_syn, y_syn = syn.get_data()
        x_syn = aug_syn(torch.cat([x_syn] * augs_per_batch))
        y_syn = torch.cat([y_syn] * augs_per_batch)
        loss_syn = F.cross_entropy(fc(phi.features(x_syn)), y_syn)
        gw, gb = torch.autograd.grad(loss_syn, [fc.weight, fc.bias], retain_graph=True, create_graph=True)
        grad_syn = torch.cat([gw.flatten(), gb.flatten()])
        match = 1 - F.cosine_similarity(grad_real, grad_syn, dim=0)     # fp32: no AMP_SCALE needed
        syn.optimizer.zero_grad(set_to_none=True)
        match.backward()
        syn.optimizer.step()
        if step % 10 == 0:
            losses.append((step, match.item()))
        if step % 100 == 0 or step == iterations:
            log(f"distill {name} step {step}/{iterations} loss {match.item():.4f} levels {len(syn.levels)} "
                f"({time.time() - t0:.0f}s, mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB)")
    with torch.no_grad():
        x_final, y_final = syn.get_data()
    torch.save(dict(x=x_final.detach().cpu(), y=y_final.cpu(), pyramid=syn.state()["levels"],
                    cfg=dict(name=name, ipc=ipc, iterations=iterations, lr=lr, augs_per_batch=augs_per_batch,
                             pyramid=pyramid, aug=aug, seed=seed, extend_it=extend_it, res=syn.res_list,
                             seconds=time.time() - t0, loss_first=losses[0][1], loss_last=losses[-1][1],
                             loss_mean_last10=float(np.mean([l for _, l in losses[-10:]])))),
               out_dir / "data.pt")
    with open(out_dir / "loss.csv", "w") as f:
        f.write("step,loss\n" + "".join(f"{s},{l:.6f}\n" for s, l in losses))
    log(f"distill {name}: done in {time.time() - t0:.0f}s, loss {losses[0][1]:.4f} -> {losses[-1][1]:.4f}")


# ---------------------------------------------------------------------------
# linear probe evaluation (official distillation/eval.py Evaluator)
# ---------------------------------------------------------------------------
def linear_probe(phi: nn.Module, X_sets, y_sets, X_te_feat, y_te, epochs: int, device: str,
                 seed: int = 3407, lr_rule: str = "linear", probe_lr: float = 1e-3, aug: str = "basic"):
    """Official Evaluator protocol for K independent linear probes run *vectorised* in one pass
    (K = num_eval repeats of the same set, or K different Random selections). Each probe k has its own
    head (W_k ~ N(0, 0.01), b_k = 0), its own shuffling, its own per-sample augmentation; the total loss is the
    sum of the K CE losses and Adam is element-wise, so the K runs are independent (identical to running them
    one after another up to RNG streams). Evaluation on the un-augmented test set after the last epoch only
    (official eval_it=-1)."""
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    K, N = len(X_sets), len(X_sets[0])
    assert all(len(x) == N for x in X_sets)
    num_feats = X_te_feat.shape[1]
    bs = min(100, N)
    lr = 0.001 * bs / 256.0 if lr_rule == "linear" else probe_lr
    seed_all(seed)
    W = (torch.randn(K, NUM_CLASSES, num_feats, device=device) * 0.01).requires_grad_(True)
    b = torch.zeros(K, NUM_CLASSES, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    augmentor = make_augmentor(aug).to(device)
    last = torch.zeros(K, device=device)
    # all shufflings drawn up front (same draw order as per-epoch) -> a single host->device copy, no per-step syncs
    perms_all = torch.stack([torch.randperm(N, generator=g) for _ in range(epochs * K)]).view(epochs, K, N).to(device)
    for ep in range(epochs):
        perms = perms_all[ep]
        for i in range(0, N, bs):
            idx = [pm[i:i + bs] for pm in perms]
            m = len(idx[0])
            x = torch.cat([X_sets[k][idx[k]] for k in range(K)])
            y = torch.stack([y_sets[k][idx[k]] for k in range(K)])                       # (K, m)
            with torch.no_grad():
                z = phi.features(augmentor(x)).view(K, m, num_feats)
            logits = torch.einsum("kmf,kcf->kmc", z, W) + b[:, None, :]
            per_k = F.cross_entropy(logits.reshape(K * m, NUM_CLASSES), y.reshape(-1), reduction="none").view(K, m).mean(1)
            loss = per_k.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ep == epochs - 1:
                last += per_k.detach() * m / N
        sched.step()
    with torch.no_grad():
        pred = (torch.einsum("nf,kcf->knc", X_te_feat, W) + b[:, None, :]).argmax(2).cpu().numpy()
    y_te_np = y_te.cpu().numpy()
    runs = [dict(acc=accuracy_score(y_te_np, pred[k]), f1=f1_score(y_te_np, pred[k], average="macro"),
                 kappa=cohen_kappa_score(y_te_np, pred[k]), train_loss_last=float(last[k])) for k in range(K)]
    res = {k: float(np.mean([r[k] for r in runs])) for k in ("acc", "f1", "kappa")}
    res.update({k + "_std": float(np.std([r[k] for r in runs])) for k in ("acc", "f1", "kappa")})
    res.update(train_loss_last=float(np.mean([r["train_loss_last"] for r in runs])), n_train=int(N),
               epochs=epochs, num_eval=K, lr=lr, lr_rule=lr_rule, runs=runs)
    return res


RESULT_FIELDS = ["tag", "method", "variant", "ipc", "distill_model", "eval_model", "n_train", "epochs", "num_eval",
                 "lr_rule", "probe_lr", "acc_mean", "acc_std", "f1_mean", "f1_std", "kappa_mean", "kappa_std",
                 "train_loss_last", "seconds"]


def parse_rule(rule: str):
    """Probe rule token -> (lr_rule, probe augmentation).  'linear' / 'fixed' = official AugBasic (scale + 1-D crop);
    'linear+noaug' / 'fixed+noaug' = same lr rule but NO train-time augmentation (reviewer check: the 1-D
    RandomResizedCrop time-stretches the signal -> frequency shift -> train/test mismatch on EEG, costs ~6 pts on Full).
    The full token is what goes into the eval json name and the results.csv `lr_rule` column."""
    base, _, augtok = rule.partition("+")
    if base not in ("linear", "fixed") or augtok not in ("", "noaug"):
        raise ValueError(f"unknown probe rule {rule!r}; use linear|fixed[+noaug]")
    return base, ("none" if augtok == "noaug" else "basic")


def rule_label(rule: str) -> str:
    base, aug = parse_rule(rule)
    return ("官方 0.001×batch/256" if base == "linear" else "固定 lr") + ("，无探针增强" if aug == "none" else "，AugBasic 增强")


def append_result(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def run_eval(root: Path, tag: str, method: str, variant: str, ipc: int, distill_model: str, eval_model: str,
             data: Data, phi_cache: dict, X_tr, y_tr, epochs: int, num_eval: int, lr_rule: str, probe_lr: float,
             seed: int = 3407):
    out = root / "evals" / f"{tag}__{eval_model}__{lr_rule}.json"
    if out.exists():
        log(f"eval {tag} on {eval_model} [{lr_rule}]: exists, skip")
        return
    if eval_model not in phi_cache:
        phi = load_phi(root, eval_model, data)
        phi_cache[eval_model] = (phi, features(phi, data.X_test))
    phi, te_feat = phi_cache[eval_model]
    base_rule, probe_aug = parse_rule(lr_rule)
    t0 = time.time()
    res = linear_probe(phi, [X_tr] * num_eval, [y_tr] * num_eval, te_feat, data.y_test, epochs, data.device,
                       seed=seed, lr_rule=base_rule, probe_lr=probe_lr, aug=probe_aug)
    res["lr_rule"] = lr_rule
    res.update(tag=tag, method=method, variant=variant, ipc=ipc, distill_model=distill_model, eval_model=eval_model,
               seconds=time.time() - t0)
    write_json(out, res)
    append_result(root / "results.csv", dict(tag=tag, method=method, variant=variant, ipc=ipc, distill_model=distill_model,
                                             eval_model=eval_model, n_train=res["n_train"], epochs=epochs,
                                             num_eval=num_eval, lr_rule=lr_rule, probe_lr=res["lr"],
                                             acc_mean=res["acc"], acc_std=res["acc_std"], f1_mean=res["f1"],
                                             f1_std=res["f1_std"], kappa_mean=res["kappa"], kappa_std=res["kappa_std"],
                                             train_loss_last=res["train_loss_last"], seconds=res["seconds"]))
    log(f"eval {tag} on {eval_model} [{lr_rule}]: acc {res['acc']:.3f}±{res['acc_std']:.3f} f1 {res['f1']:.3f} "
        f"kappa {res['kappa']:.3f} (train loss {res['train_loss_last']:.3f}, {res['seconds']:.0f}s)")


# ---------------------------------------------------------------------------
# baselines (official baselines/random_reals.py, centroids.py, full_dataset.py)
# ---------------------------------------------------------------------------
def random_reals(data: Data, ipc: int, seed: int):
    rng = np.random.RandomState(seed)
    y = data.y_pool.cpu().numpy()
    idx = np.concatenate([rng.choice(np.where(y == c)[0], ipc, replace=False) for c in range(NUM_CLASSES)])
    idx = torch.from_numpy(idx).to(data.device)
    return data.X_pool[idx], data.y_pool[idx]


@torch.no_grad()
def centroid_reals(data: Data, phi: nn.Module, ipc: int):
    feat = features(phi, data.X_pool)
    idx = []
    for c in range(NUM_CLASSES):
        cls = torch.where(data.y_pool == c)[0]
        d = torch.norm(feat[cls] - feat[cls].mean(0, keepdim=True), dim=1)
        idx.append(cls[torch.argsort(d)[:ipc]])
    idx = torch.cat(idx)
    return data.X_pool[idx], data.y_pool[idx]


# ---------------------------------------------------------------------------
# summary + figures
# ---------------------------------------------------------------------------
def summarize(root: Path, data: Data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res_dir = root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(root / "results.csv")))
    for r in rows:
        r["ipc"] = int(r["ipc"])
        for k in ("acc_mean", "acc_std", "f1_mean", "f1_std", "kappa_mean", "kappa_std"):
            r[k] = float(r[k])
    lines = []
    for rule in sorted({r["lr_rule"] for r in rows}):
        lines.append(f"### 线性探针规则：`{rule}`（{rule_label(rule)}）")
        lines.append("")
        lines.append("| tag | method | variant | ipc | distill φ | eval φ | acc (%) | macro-F1 (%) | κ (%) | n_train | num_eval |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        order = {"lgm": 0, "random": 1, "centroids": 2, "full": 3}
        for r in sorted([r for r in rows if r["lr_rule"] == rule],
                        key=lambda r: (order.get(r["method"], 9), r["variant"], r["ipc"], r["tag"], EVAL_MODELS.index(r["eval_model"]))):
            lines.append(f"| {r['tag']} | {r['method']} | {r['variant']} | {r['ipc']} | {r['distill_model']} | {r['eval_model']} | "
                         f"{100 * r['acc_mean']:.1f} ± {100 * r['acc_std']:.1f} | {100 * r['f1_mean']:.1f} ± {100 * r['f1_std']:.1f} | "
                         f"{100 * r['kappa_mean']:.1f} ± {100 * r['kappa_std']:.1f} | {r['n_train']} | {r['num_eval']} |")
        lines.append("")
    phi_info = {}
    for name in ("eegnet", "shallow"):
        p = phi_path(root, name).with_suffix(".json")
        if p.exists():
            j = json.load(open(p))
            phi_info[name] = j
            lines.append(f"- φ `{name}`：feat_dim {j['feat_dim']}，best val acc {100 * j['best_val_acc']:.1f}% @ epoch {j['best_epoch']}，"
                         f"端到端（含原分类头）测试 acc {100 * j['end2end_test_acc']:.1f}%，训练 {j['train_sec']:.0f}s")
    for p in sorted((root / "synsets").glob("*/data.pt")):
        cfg = torch.load(p, map_location="cpu")["cfg"]
        lines.append(f"- 蒸馏 `{cfg['name']}`：ipc {cfg['ipc']}，{cfg['iterations']} 步，lr {cfg['lr']}，pyramid {cfg['pyramid']}，aug {cfg['aug']}，"
                     f"loss {cfg['loss_first']:.3f} → {cfg['loss_mean_last10']:.3f}（末 10 次记录均值），{cfg['seconds']:.0f}s")
    (res_dir / "table.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {res_dir / 'table.md'}")

    # --- figures: distilled ipc=1 waveforms vs real samples, 4 channels ---
    chans = [data.ch_names.index(c) for c in ("C3", "Cz", "C4", "Pz")]
    t = np.arange(T_LEN) / 200.0 + 2.0
    def plot_set(X, y, title, path):
        fig, axes = plt.subplots(NUM_CLASSES, 1, figsize=(10, 9), sharex=True)
        for c in range(NUM_CLASSES):
            i = int(torch.where(y == c)[0][0])
            for k, ch in enumerate(chans):
                axes[c].plot(t, X[i, ch] + 6 * (len(chans) - 1 - k), lw=0.7, label=data.ch_names[ch])
            axes[c].set_ylabel(CLASS_NAMES[c])
            axes[c].set_yticks([])
        axes[0].legend(ncol=4, fontsize=8, loc="upper right")
        axes[-1].set_xlabel("time (s), MI window [2, 6] s, z-scored units (+6 offset per channel)")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
    p = root / "synsets" / "lgm_ipc1" / "data.pt"
    if p.exists():
        d = torch.load(p, map_location="cpu")
        plot_set(d["x"].numpy(), d["y"], "LGM distilled EEG (ipc=1, pyramid + decorrelation, phi = EEGNet)",
                 res_dir / "distilled_ipc1.png")
    for name, title in (("lgm_ipc1_nopyramid", "LGM distilled EEG (ipc=1, NO pyramid: direct C×T optimisation)"),
                        ("lgm_ipc1_noaug", "LGM distilled EEG (ipc=1, NO differentiable augmentation)")):
        p = root / "synsets" / name / "data.pt"
        if p.exists():
            d = torch.load(p, map_location="cpu")
            plot_set(d["x"].numpy(), d["y"], title, res_dir / f"distilled_{name[4:]}.png")
    Xr, yr = random_reals(data, 1, 0)
    plot_set(Xr.cpu().numpy(), yr.cpu(), "Real EEG samples (one per class, random seed 0, same z-scored scale)",
             res_dir / "real_ipc1.png")
    # --- loss curves ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for p in sorted((root / "synsets").glob("*/loss.csv")):
        a = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
        if len(a):
            ax.plot(a[:, 0], a[:, 1], lw=0.8, label=p.parent.name)
    ax.set_xlabel("distillation step")
    ax.set_ylabel("1 - cos(grad_real, grad_syn)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(res_dir / "loss_curves.png", dpi=130)
    plt.close(fig)
    log(f"figures written to {res_dir}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["phi", "distill", "eval", "baselines", "summarize", "info"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--data", default=str(DATA_NPZ))
    ap.add_argument("--device", default="cuda")
    # phi
    ap.add_argument("--phi-epochs", type=int, default=100)
    # distill
    ap.add_argument("--name", default=None, help="synset name (distill) / eval tag filter")
    ap.add_argument("--ipc", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--augs-per-batch", type=int, default=10)
    ap.add_argument("--no-pyramid", action="store_true")
    ap.add_argument("--aug", default="standard", choices=["standard", "none"])
    ap.add_argument("--extend-it", type=int, default=200)
    ap.add_argument("--distill-model", default="eegnet")
    # eval
    ap.add_argument("--eval-models", default=",".join(EVAL_MODELS))
    ap.add_argument("--eval-epochs", type=int, default=1000)
    ap.add_argument("--full-epochs", type=int, default=100)
    ap.add_argument("--num-eval", type=int, default=5)
    ap.add_argument("--random-seeds", default="0,1,2,3,4")
    ap.add_argument("--ipcs", default="1,5,10")
    ap.add_argument("--lr-rules", default="linear", help="comma list of linear|fixed[+noaug] (see parse_rule)")
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    args = ap.parse_args()

    root = Path(args.root) if args.root else (RUNS_ROOT.with_name("lgm_eeg_smoke") if args.smoke else RUNS_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.phi_epochs = min(args.phi_epochs, 3)
        args.iterations = min(args.iterations, 50)
        args.extend_it = min(args.extend_it, 10)
        args.eval_epochs = min(args.eval_epochs, 20)
        args.full_epochs = min(args.full_epochs, 2)
        args.num_eval = 1
        args.ipcs = "1"
    torch.backends.cudnn.benchmark = False
    t_start = time.time()
    data = Data(Path(args.data), args.device)
    log(f"stage={args.stage} root={root} data={data.summary()}")
    write_json(root / "split.json", data.summary())
    lr_rules = [r for r in args.lr_rules.split(",") if r]
    for r in lr_rules:
        parse_rule(r)                                     # fail early on a typo
    eval_models = [m for m in args.eval_models.split(",") if m]

    if args.stage == "info":
        return
    if args.stage == "phi":
        for name in ("eegnet", "shallow"):
            train_phi(root, name, data, args.phi_epochs)
    elif args.stage == "distill":
        phi = load_phi(root, args.distill_model, data)
        name = args.name or f"lgm_ipc{args.ipc}"
        distill(root, name, data, phi, args.ipc, args.iterations, args.lr, args.augs_per_batch,
                pyramid=not args.no_pyramid, aug=args.aug, extend_it=args.extend_it)
    elif args.stage == "eval":
        cache = {}
        for p in sorted((root / "synsets").glob("*/data.pt")):
            d = torch.load(p, map_location="cpu")
            cfg = d["cfg"]
            if args.name and cfg["name"] != args.name:
                continue
            variant = ("pyramid" if cfg["pyramid"] else "nopyramid") + "+" + ("aug" if cfg["aug"] != "none" else "noaug")
            X_tr, y_tr = d["x"].to(data.device), d["y"].to(data.device)
            for rule in lr_rules:
                for m in eval_models:
                    run_eval(root, cfg["name"], "lgm", variant, cfg["ipc"], args.distill_model, m, data, cache, X_tr, y_tr,
                             args.eval_epochs, args.num_eval, rule, args.probe_lr)
    elif args.stage == "baselines":
        cache = {}
        ipcs = [int(x) for x in args.ipcs.split(",")]
        seeds = [int(x) for x in args.random_seeds.split(",")]
        phi_d = load_phi(root, args.distill_model, data)
        for rule in lr_rules:
            for m in eval_models:
                for ipc in ipcs:
                    # Random: official = num_eval 1 per seed, 5 seeds -> we aggregate the seeds into mean ± std
                    tag = f"random_ipc{ipc}"
                    out = root / "evals" / f"{tag}__{m}__{rule}.json"
                    if out.exists():
                        log(f"eval {tag} on {m} [{rule}]: exists, skip")
                    else:
                        if m not in cache:
                            phi = load_phi(root, m, data)
                            cache[m] = (phi, features(phi, data.X_test))
                        phi, te_feat = cache[m]
                        base_rule, probe_aug = parse_rule(rule)
                        t0 = time.time()
                        sets = [random_reals(data, ipc, s) for s in seeds]          # one selection per seed
                        res = linear_probe(phi, [x for x, _ in sets], [y for _, y in sets], te_feat, data.y_test,
                                           args.eval_epochs, data.device, seed=seeds[0], lr_rule=base_rule,
                                           probe_lr=args.probe_lr, aug=probe_aug)
                        res.update(tag=tag, method="random", variant="real", ipc=ipc, distill_model="-", eval_model=m,
                                   seeds=seeds, seconds=time.time() - t0, lr_rule=rule)
                        write_json(out, res)
                        append_result(root / "results.csv", dict(tag=tag, method="random", variant="real", ipc=ipc, distill_model="-",
                                                                 eval_model=m, n_train=res["n_train"], epochs=args.eval_epochs,
                                                                 num_eval=len(seeds), lr_rule=rule, probe_lr=res["lr"],
                                                                 acc_mean=res["acc"], acc_std=res["acc_std"], f1_mean=res["f1"],
                                                                 f1_std=res["f1_std"], kappa_mean=res["kappa"],
                                                                 kappa_std=res["kappa_std"], train_loss_last=res["train_loss_last"],
                                                                 seconds=res["seconds"]))
                        log(f"eval {tag} on {m} [{rule}]: acc {res['acc']:.3f}±{res['acc_std']:.3f} ({res['seconds']:.0f}s)")
                    # Centroids: nearest-to-class-mean real samples in the distillation phi's feature space
                    X_tr, y_tr = centroid_reals(data, phi_d, ipc)
                    run_eval(root, f"centroids_ipc{ipc}", "centroids", "real", ipc, args.distill_model, m, data, cache,
                             X_tr, y_tr, args.eval_epochs, args.num_eval, rule, args.probe_lr)
                # Full: whole train pool, official FullDatasetCfg (100 epochs, batch 100)
                run_eval(root, "full", "full", "real", 0, "-", m, data, cache, data.X_pool, data.y_pool,
                         args.full_epochs, args.num_eval, rule, args.probe_lr)
    elif args.stage == "summarize":
        summarize(root, data)
    log(f"stage {args.stage} finished in {time.time() - t_start:.0f}s, peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
