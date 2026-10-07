"""Unsupervised test-time adaptation methods, all label-free.

Every method here takes a source model that is already trained and frozen, and
is given the same unlabelled target session. Nothing sees a target label. That
shared starting point is the point: it is what makes the per-method differences
attributable to the method rather than to a different training run.

EA is the exception and lives in data_io — it transforms the input, so its
source model has to be trained on aligned data in the first place.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bn_modules(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


@torch.no_grad()
def adabn(model, Xt, dev, bs=64):
    """AdaBN: recompute BatchNorm statistics on the target session.

    No gradients, no labels, no parameter updates -- only the running mean and
    variance are replaced by the target's own. The cheapest thing that can
    possibly work, and often the most robust.
    """
    m = copy.deepcopy(model)
    for bn in _bn_modules(m):
        bn.reset_running_stats()
        bn.momentum = None          # cumulative moving average over the pass
    m.train()
    for i in range(0, len(Xt), bs):
        m(torch.tensor(Xt[i:i + bs], device=dev))
    m.eval()
    return m


def _features(model, X, dev, bs=256):
    """Penultimate-layer features, i.e. the input to the linear head."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.tensor(X[i:i + bs], device=dev).unsqueeze(1)
            z = model.block2(model.block1(x)).flatten(1)
            outs.append(z)
    return torch.cat(outs)


@torch.no_grad()
def t3a_predict(model, Xt, dev, M=20):
    """T3A (Iwasawa & Matsuo, 2020): optimization-free template adjustment.

    The classifier's weight rows are the initial class templates. Confident
    target features are appended to their predicted class's support set, the
    set is pruned to the M lowest-entropy members, and the final prediction is
    a nearest-centroid decision. No backprop, so nothing can diverge.
    """
    z = _features(model, Xt, dev)                       # (N, D)
    W = model.head.weight.detach()                      # (K, D)
    K = W.shape[0]

    logits = z @ W.T + model.head.bias.detach()
    ent = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    yhat = logits.argmax(1)

    supports = [W[k:k + 1] for k in range(K)]           # seed with the templates
    ent_of = [torch.zeros(1, device=z.device) for _ in range(K)]
    for k in range(K):
        sel = yhat == k
        if sel.any():
            supports[k] = torch.cat([supports[k], z[sel]])
            ent_of[k] = torch.cat([ent_of[k], ent[sel]])

    centroids = []
    for k in range(K):
        keep = ent_of[k].argsort()[:M]                  # most confident only
        centroids.append(F.normalize(supports[k][keep].mean(0), dim=0))
    C = torch.stack(centroids)                          # (K, D)

    return (F.normalize(z, dim=1) @ C.T).argmax(1).cpu().numpy()


def tent(model, Xt, dev, lr=1e-3, steps=1, bs=64):
    """Tent (Wang et al., 2021): entropy minimisation on BN affine parameters.

    The gradient-based family. It can help, and it can also confidently walk a
    model off a cliff -- which is exactly the behaviour worth measuring against
    the optimization-free methods above.
    """
    m = copy.deepcopy(model)
    m.requires_grad_(False)

    params = []
    for bn in _bn_modules(m):
        bn.requires_grad_(True)
        bn.track_running_stats = False
        bn.running_mean = None
        bn.running_var = None
        params += [bn.weight, bn.bias]

    opt = torch.optim.Adam(params, lr=lr)
    m.train()
    for _ in range(steps):
        for i in range(0, len(Xt), bs):
            x = torch.tensor(Xt[i:i + bs], device=dev)
            if len(x) < 2:                              # BN needs >1 sample
                continue
            opt.zero_grad()
            out = m(x)
            loss = -(out.softmax(1) * out.log_softmax(1)).sum(1).mean()
            loss.backward()
            opt.step()
    return m


@torch.no_grad()
def plain_predict(model, X, dev, bs=256):
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        outs.append(model(torch.tensor(X[i:i + bs], device=dev)).argmax(1))
    return torch.cat(outs).cpu().numpy()
