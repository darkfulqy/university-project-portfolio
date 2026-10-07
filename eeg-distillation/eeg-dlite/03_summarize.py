#!/usr/bin/env python
"""Step 03: aggregate {out_root}/results.csv -> {out_root}/results/pilot_table.{csv,md} + fig_ood_removal.png

* mean +- sd over (fold x seed) of accuracy / macro-F1 / kappa per (method, eta, tau)
* Mann-Whitney U (one-sided, 'greater', as the paper marks "significantly better than Random at the same
  eta") of every method vs Random at the same eta, on the per-(fold, seed) values; dagger when p < 0.05
* qualitative check of the three paper claims (Proposed(tau=1%) >= Full at 25-50 %, OOD removal helps,
  PCA < SSL) on the substitute dataset
* fig_ood_removal.png: accuracy of Proposed vs tau for every eta (two points when only two taus were run)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

METRICS = ["acc", "f1_macro", "kappa"]
NAMES = {"random": "Random", "pca_ds": "PCA + DS", "proposed": "Proposed", "full": "Full data"}


def mwu_greater(a, b) -> float:
    from scipy.stats import mannwhitneyu

    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2 or (np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]):
        return float("nan")
    return float(mannwhitneyu(a, b, alternative="greater").pvalue)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_root = Path(args.out_root) if args.out_root else (common.RUNS_SMOKE_ROOT if args.smoke else common.RUNS_ROOT)
    csv_path = out_root / "results.csv"
    res_dir = out_root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    for c in ["eta", "tau"] + METRICS:
        df[c] = df[c].astype(float)
    df = df.drop_duplicates(subset=["dataset", "fold", "seed", "method", "eta", "tau"], keep="last")
    print(f"[03] {len(df)} rows, folds={sorted(df.fold.unique())}, seeds={sorted(df.seed.unique())}")

    rows = []
    for (method, eta, tau), g in df.groupby(["method", "eta", "tau"]):
        rnd = df[(df.method == "random") & (df.eta == eta)]
        r = dict(method=method, method_name=NAMES.get(method, method), eta=eta, eta_pct=int(round(eta * 100)),
                 tau=tau, n_runs=len(g), n_train_mean=g.n_train.mean())
        for m in METRICS:
            r[f"{m}_mean"] = 100 * g[m].mean()
            r[f"{m}_sd"] = 100 * g[m].std(ddof=1) if len(g) > 1 else float("nan")
            if method != "random" and len(rnd):
                # pair on (fold, seed) so the two samples cover the same runs, then unpaired MWU (paper's test)
                merged = g.merge(rnd, on=["fold", "seed"], suffixes=("", "_rnd"))
                r[f"{m}_p_vs_random"] = mwu_greater(merged[m], merged[f"{m}_rnd"])
            else:
                r[f"{m}_p_vs_random"] = float("nan")
        rows.append(r)
    tab = pd.DataFrame(rows).sort_values(["method", "tau", "eta"], key=lambda s: s.map(
        {"random": 0, "pca_ds": 1, "proposed": 2, "full": 3}) if s.name == "method" else s)
    tab.to_csv(res_dir / "pilot_table.csv", index=False, float_format="%.4f")

    # ----- markdown -----
    def cell(r, m):
        s = f"{r[f'{m}_mean']:.1f}"
        if not np.isnan(r[f"{m}_sd"]):
            s += f" ± {r[f'{m}_sd']:.1f}"
        p = r[f"{m}_p_vs_random"]
        if not np.isnan(p) and p < 0.05:
            s += " †"
        return s

    lines = [f"# EEG-DLite pilot-study 复现表（{common.DATASET_NAME}，EEGNet 跨被试，GroupKFold(3) × seeds）", "",
             f"来源：`{csv_path}`，{len(df)} 行；每格 mean ± sd（%），† = Mann–Whitney U（单侧 greater）相对同 η 的 Random p<0.05。", "",
             "| Method | τ | η (%) | n_train | Acc. | F1 (macro) | κ | runs |", "|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        lines.append(f"| {r.method_name} | {r.tau:g} | {r.eta_pct} | {r.n_train_mean:.0f} | {cell(r, 'acc')} | "
                     f"{cell(r, 'f1_macro')} | {cell(r, 'kappa')} | {int(r.n_runs)} |")

    # ----- qualitative checks vs paper Table tab:pilot-study -----
    def get(method, eta, tau, m="acc_mean"):
        q = tab[(tab.method == method) & (np.isclose(tab.eta, eta)) & (np.isclose(tab.tau, tau))]
        return float(q[m].iloc[0]) if len(q) else float("nan")

    full = get("full", 1.0, 0.0)
    checks = []
    for eta in [0.25, 0.5]:
        p1 = get("proposed", eta, 0.01)
        checks.append((f"Proposed(τ=1%) @η={eta:g} 不低于 Full", p1, full, (p1 >= full) if not np.isnan(p1 + full) else None))
    for eta in sorted(tab.eta.unique()):
        if eta >= 1:
            continue
        a, b = get("proposed", eta, 0.01), get("proposed", eta, 0.0)
        checks.append((f"OOD 去除有帮助 @η={eta:g}（τ=1% vs τ=0）", a, b, (a > b) if not np.isnan(a + b) else None))
        c, d = get("proposed", eta, 0.0), get("pca_ds", eta, 0.0)
        checks.append((f"SSL 优于 PCA @η={eta:g}（Proposed τ=0 vs PCA+DS）", c, d, (c > d) if not np.isnan(c + d) else None))
        e, f = get("proposed", eta, 0.01), get("random", eta, 0.0)
        checks.append((f"Proposed(τ=1%) 优于 Random @η={eta:g}", e, f, (e > f) if not np.isnan(e + f) else None))
    lines += ["", "## 与论文 pilot 表的定性对照（accuracy, %）", "", "| 论文结论 | 我们 A | 我们 B | 成立? |", "|---|---|---|---|"]
    for name, a, b, ok in checks:
        lines.append(f"| {name} | {a:.1f} | {b:.1f} | {'✅' if ok else ('❌' if ok is not None else 'n/a')} |")
    lines += ["", "论文（SEED, EEGNet）: Random 25/50% = 52.8/53.4, PCA+DS = 51.6/51.8, Proposed τ=0 = 54.6/54.1, "
              "Proposed τ=1% = 55.3/56.6, Full = 54.6（acc %）。本表为 **BCI IV-2a 替代数据集**，只做定性对照。"]
    (res_dir / "pilot_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ----- fig: accuracy vs tau (Proposed) -----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    prop = tab[tab.method == "proposed"]
    for eta, g in prop.groupby("eta"):
        g = g.sort_values("tau")
        ax.errorbar(g.tau * 100, g.acc_mean, yerr=g.acc_sd.fillna(0), marker="o", capsize=3, label=f"η={eta:g}")
    if not np.isnan(full):
        ax.axhline(full, ls="--", c="gray", label="Full data")
    ax.set_xlabel("OOD removal ratio τ (%)"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Proposed: effect of outlier removal ({common.DATASET_NAME}, EEGNet)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(res_dir / "fig_ood_removal.png", dpi=150)
    print(f"[03] wrote {res_dir / 'pilot_table.md'}, pilot_table.csv, fig_ood_removal.png")


if __name__ == "__main__":
    main()
