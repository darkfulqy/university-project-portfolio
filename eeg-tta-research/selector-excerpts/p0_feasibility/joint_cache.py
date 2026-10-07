"""Validated default-method predictions and joint trial resampling.

One bootstrap count matrix is shared by every seed, adapted method, and R0
within a subject-day. Seeds are fixed trained models, never independent trial
replicates. Resampling conditions on stored transductive predictions; it does
not retrain or rerun adaptation on the resampled EEG.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS = PROJECT.parent / "safe-to-adapt" / "results"
KEYS = ["subject_id", "seed", "target_session"]
REGIMES = {
    "bn_adapt": "R4_AdaBN", "ea": "R1_EA", "eata": "R10_EATA",
    "lame": "R6_LAME", "pseudo_label": "R9_PL", "riemannian_align": "R7_RA",
    "sar": "R11_SAR", "shot": "R12_SHOT", "t3a": "R2_T3A",
    "tent": "R3_Tent", "tent_pure": "R8_TentPure",
}
METHODS = tuple(sorted(REGIMES))
BASE_METHODS = {"bn_adapt", "ea", "t3a", "tent"}
EXCLUSIONS = {"Wang2026": (), "Zhou2020": ((20, 5), (20, 6), (20, 7))}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(a, b, context):
    if not np.array_equal(a, b):
        raise ValueError(f"Trial alignment mismatch: {context}")


def _close(a, b, context):
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(
            a, b, rtol=0., atol=1e-9):
        raise ValueError(f"BA/Delta mismatch: {context}")


def _point_ba(predictions, y, n_classes):
    """BA in percentage points, preserving the prediction leading dimensions."""
    if set(np.unique(y)) != set(range(n_classes)):
        raise ValueError("Every scored class must be present")
    return 100. * np.stack([(predictions[..., y == c] == c).mean(axis=-1)
                            for c in range(n_classes)]).mean(axis=0)


def _table(table_path, dataset, backbone, seeds):
    df = pd.read_csv(table_path)
    df = df[df.dataset.eq(dataset) & df.backbone.eq(backbone) & df.seed.isin(seeds)].copy()
    if "subject" in df and "subject_id" not in df:
        df = df.rename(columns={"subject": "subject_id", "day": "target_session",
                                "ba_R0": "ba_no_tta", "n_cls": "n_classes"})
    if "labelled" in df:
        flag = df.labelled.astype(str).str.lower()
        if not flag.isin(["true", "false", "1", "0"]).all():
            raise ValueError("Unknown labelled flags")
        df = df[flag.isin(["false", "0"])].copy()
    if "source_session" in df and not df.source_session.eq(1).all():
        raise ValueError("Expected day-1 source models")
    if df.method.str.startswith("R").all():
        df = df[df.method.ne("R5_head")].copy()
        df["method"] = df.method.map({regime: method for method, regime in REGIMES.items()})
    if df.empty or df.method.isna().any() or sorted(df.method.unique()) != list(METHODS):
        raise ValueError("Expected the complete 11-method default table")
    if df.duplicated(KEYS + ["method"]).any():
        raise ValueError("Duplicate cell/method rows")
    if not df.groupby(KEYS).method.nunique().eq(len(METHODS)).all():
        raise ValueError("Incomplete cell/method table")
    if (df.groupby(KEYS)[["ba_no_tta", "n_classes"]].nunique(dropna=False).to_numpy() != 1).any():
        raise ValueError("Inconsistent cell metadata")
    if not np.isfinite(df[["delta", "ba_no_tta", "n_classes"]].to_numpy()).all():
        raise ValueError("Non-finite table measurements")
    by_group = df.groupby(["subject_id", "target_session"]).seed.apply(lambda x: tuple(sorted(x.unique())))
    if any(value != tuple(seeds) for value in by_group):
        raise ValueError("Every subject-day must have all requested seeds")
    excluded = EXCLUSIONS[dataset]
    if any((int(s), int(d)) in excluded for s, d in df[["subject_id", "target_session"]].itertuples(index=False)):
        raise ValueError("Table contains an explicitly excluded subject-day")
    return df


def _inventory(roots, backbone, ending, seeds, excluded):
    files, omitted = {}, []
    for root in roots:
        for path in sorted((root / backbone).glob(f"S*/seed*/day*_{ending}.npz")):
            subject = int(path.parent.parent.name[1:])
            seed = int(path.parent.name[4:])
            day = int(re.fullmatch(r"day(\d+)_" + ending + r"\.npz", path.name).group(1))
            if seed not in seeds:
                continue  # Explicitly requested seed scope, recorded in the audit.
            if (subject, day) in excluded:
                omitted.append(str(path.resolve()))
                continue
            key = subject, seed, day
            if key in files:
                raise ValueError(f"Ambiguous duplicate cache file for {key}")
            files[key] = path
    return files, omitted


def load_joint_cache(dataset, backbone, table_path=None, base_roots=None, pool_root=None, seeds=None):
    """Load all requested cells in CSV order sorted by subject, seed, day.

    Dataset is ``Wang2026`` or ``Zhou2020``; backbone is ``eegnet`` or
    ``specialist``. Defaults use 10 EEGNet seeds or 3 ATCNet seeds. Explicit
    roots point to dataset directories that contain backbone subdirectories.

    Returns keys, methods, subjects, delta[cell,method], ba0[cell], groups and
    audit. Each subject-day group contains sorted seeds, global cell_indices,
    y/run/trial on evaluated trials, and predictions[seed,R0+method,trial].
    Full labels/run/trial/eval_mask are retained in full_metadata for audit.
    """
    if dataset not in EXCLUSIONS or backbone not in ("eegnet", "specialist"):
        raise ValueError("Unsupported dataset/backbone")
    seeds = tuple(range(10 if backbone == "eegnet" else 3)) if seeds is None else tuple(sorted(seeds))
    if not seeds or len(set(seeds)) != len(seeds) or any(int(s) != s or s < 0 for s in seeds):
        raise ValueError("Expected unique nonnegative integer seeds")
    table_path = Path(table_path or (PROJECT / "deltas_wang2026.csv" if dataset == "Wang2026"
                                    else HERE / "out" / "cells.csv"))
    default_roots = [RESULTS / "wang"] if dataset == "Wang2026" else [RESULTS / "main", RESULTS / "main_seeds10"]
    roots = default_roots if base_roots is None else ([base_roots] if isinstance(base_roots, (str, Path)) else base_roots)
    roots = [Path(root) for root in roots]
    pool_root = Path(pool_root or PROJECT / "results_pool" / ("wang" if dataset == "Wang2026" else "zhou"))
    df = _table(table_path, dataset, backbone, seeds)
    truth = df.pivot(index=KEYS, columns="method", values="delta")[list(METHODS)].sort_index()
    cells = df.drop_duplicates(KEYS).set_index(KEYS).reindex(truth.index)
    keys = truth.index.to_frame(index=False)
    expected = set(tuple(map(int, row)) for row in truth.index)
    base, excluded_base = _inventory(roots, backbone, "regimes", seeds, EXCLUSIONS[dataset])
    pool, excluded_pool = _inventory([pool_root], backbone, "extra", seeds, EXCLUSIONS[dataset])
    for label, inventory in [("base", base), ("pool", pool)]:
        actual = set(inventory)
        if actual != expected:
            raise ValueError(f"{label} cohort mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}")
    groups, input_files = [], []
    group_map = keys.groupby(["subject_id", "target_session"], sort=True).indices
    for (subject, day), indices in group_map.items():
        indices = np.array(sorted(indices, key=lambda i: int(keys.iloc[i].seed)), dtype=int)
        preds, reference, n_classes = [], None, None
        for i in indices:
            key = tuple(map(int, keys.iloc[i].to_numpy()))
            bp, ep = base[key], pool[key]
            with np.load(bp, allow_pickle=False) as b, np.load(ep, allow_pickle=False) as e:
                metadata = {field: b[field] for field in ("y", "run", "trial", "eval_mask")}
                for field in metadata:
                    _same(metadata[field], e[field], f"{key} base vs pool {field}")
                    if reference is not None:
                        _same(metadata[field], reference[field], f"{key} across seeds {field}")
                y, mask = metadata["y"], metadata["eval_mask"]
                if y.ndim != 1 or mask.dtype != bool or mask.shape != y.shape or not mask.any():
                    raise ValueError(f"Malformed labels/evaluation mask for {key}")
                if any(metadata[field].shape != y.shape for field in ("run", "trial")):
                    raise ValueError(f"Malformed run/trial metadata for {key}")
                nc = int(cells.iloc[i].n_classes)
                if n_classes is not None and nc != n_classes:
                    raise ValueError("Class count differs between seeds")
                n_classes = nc
                seed_predictions = []
                for method in ("R0",) + METHODS:
                    source = b if method == "R0" or method in BASE_METHODS else e
                    name = "logits_R0" if method == "R0" else "logits_" + REGIMES[method]
                    logits = source[name]
                    if logits.shape != (len(y), nc) or not np.isfinite(logits).all():
                        raise ValueError(f"Malformed/non-finite {name} for {key}")
                    seed_predictions.append(logits[mask].argmax(axis=1))
                pred = np.stack(seed_predictions).astype(np.int16)
                ba = _point_ba(pred, y[mask], nc)
                _close(ba[0], cells.iloc[i].ba_no_tta, f"{key} R0 vs CSV")
                _close(ba[1:] - ba[0], truth.iloc[i].to_numpy(), f"{key} method Delta vs CSV")
                preds.append(pred)
                if reference is None:
                    reference = metadata
            input_files.extend([dict(kind="base", path=str(bp.resolve()), sha256=_sha(bp)),
                                dict(kind="pool", path=str(ep.resolve()), sha256=_sha(ep))])
        mask = reference["eval_mask"]
        groups.append(dict(subject_id=int(subject), target_session=int(day),
                           cell_indices=indices, seeds=keys.iloc[indices].seed.to_numpy(int),
                           y=reference["y"][mask], run=reference["run"][mask],
                           trial=reference["trial"][mask], n_classes=n_classes,
                           predictions=np.stack(preds), full_metadata=reference))
    audit = dict(dataset=dataset, backbone=backbone, seeds=list(map(int, seeds)),
                 n_cells=len(keys), n_groups=len(groups), n_subjects=int(keys.subject_id.nunique()),
                 n_methods=len(METHODS), csv_delta_comparisons=len(keys)*len(METHODS),
                 base_pool_metadata_comparisons=len(keys)*4,
                 across_seed_metadata_comparisons=(len(keys)-len(groups))*4,
                 mismatch_count=0, explicitly_excluded_subject_days=EXCLUSIONS[dataset],
                 excluded_base_files=excluded_base, excluded_pool_files=excluded_pool,
                 table_path=str(table_path.resolve()), table_sha256=_sha(table_path),
                 input_files=input_files,
                 input_files_digest=hashlib.sha256(json.dumps(input_files, sort_keys=True).encode()).hexdigest(),
                 scoring_scope="stored eval_mask only; fixed trained models and stored transductive predictions",
                 cell_order=KEYS, method_order=list(METHODS), prediction_method_order=["R0"]+list(METHODS))
    return dict(keys=keys, methods=list(METHODS), subjects=keys.subject_id.to_numpy(),
                delta=truth.to_numpy(), ba0=cells.ba_no_tta.to_numpy(), groups=groups, audit=audit)


def moving_block_counts(run, y, n_classes, n_draws, block_length, rng,
                        max_attempt_factor=100, proposal_batch_size=512):
    """Joint within-run non-circular moving-block counts, with full-class BA.

    Runs retain their observed sample sizes. Blocks have length min(L,n_run);
    a short run is therefore sampled as one whole run, without switching to
    independent trials. A proposal lacking any scored class is rejected for
    the entire group, including every seed/method/R0. The attempted-draw limit
    is explicit, and rejection statistics are returned even when zero.
    """
    run, y = np.asarray(run), np.asarray(y)
    integers = (n_classes, n_draws, block_length, max_attempt_factor, proposal_batch_size)
    if any(int(x) != x or x <= 0 for x in integers):
        raise ValueError("Expected positive integer bootstrap parameters")
    n_classes, n_draws, block_length, max_attempt_factor, proposal_batch_size = map(int, integers)
    if run.ndim != 1 or y.shape != run.shape or not len(y) or pd.isna(run).any():
        raise ValueError("Invalid run/label vectors")
    if set(np.unique(y)) != set(range(n_classes)):
        raise ValueError("Original scored trials must contain every class")
    indices = [np.flatnonzero(run == value) for value in np.unique(run)]
    counts = np.empty((n_draws, len(y)), dtype=np.int32)
    accepted = attempted = rejected = 0
    max_attempts = n_draws * max_attempt_factor
    while accepted < n_draws:
        batch = min(proposal_batch_size, n_draws - accepted, max_attempts - attempted)
        if batch <= 0:
            raise RuntimeError(f"Joint bootstrap exhausted {max_attempts} attempts: "
                               f"accepted={accepted}, rejected={rejected}; "
                               "no draws were silently dropped")
        proposals = np.zeros((batch, len(y)), dtype=np.int32)
        offsets = np.arange(batch)[:, None] * len(y)
        for idx in indices:
            n = len(idx)
            length = min(block_length, n)
            n_blocks = (n + length - 1) // length
            starts = rng.integers(0, n-length+1, size=(batch, n_blocks))
            within = (starts[:, :, None] + np.arange(length)).reshape(batch, -1)[:, :n]
            encoded = offsets + idx[within]
            proposals += np.bincount(encoded.ravel(), minlength=batch*len(y)).reshape(batch, len(y))
        class_counts = np.stack([proposals[:, y == c].sum(axis=1) for c in range(n_classes)], axis=1)
        good = (class_counts > 0).all(axis=1)
        n_good = int(good.sum())
        counts[accepted:accepted+n_good] = proposals[good]
        accepted += n_good
        attempted += batch
        rejected += batch - n_good
    return counts, dict(n_draws=n_draws, block_length=block_length,
                        effective_block_lengths=[min(block_length, len(idx)) for idx in indices],
                        run_sizes=[len(idx) for idx in indices], attempts=attempted,
                        accepted=accepted, rejected=rejected,
                        rejection_rate=rejected/attempted, max_attempts=max_attempts,
                        short_run_rule="one whole run when n_run <= block_length",
                        absent_class_rule="reject complete joint draw and resample")


def deltas_from_counts(group, counts):
    """Compute every seed/method BA then subtract its own R0, in float64."""
    weights = np.asarray(counts, dtype=np.float64)
    y, predictions = group["y"], group["predictions"]
    n_classes = int(group["n_classes"])
    if weights.ndim != 2 or weights.shape[1] != len(y) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid bootstrap count matrix")
    if predictions.ndim != 3 or predictions.shape[-1] != len(y):
        raise ValueError("Expected predictions[seed,R0+method,trial]")
    ba = np.zeros((len(weights), int(np.prod(predictions.shape[:-1]))), dtype=np.float64)
    for c in range(n_classes):
        subset = y == c
        denominators = weights[:, subset].sum(axis=1)
        if (denominators <= 0).any():
            raise ValueError("A joint draw lacks a class; reject it before computing BA")
        hit = (predictions[..., subset] == c).reshape(-1, int(subset.sum())).astype(np.float64)
        ba += (weights[:, subset] @ hit.T) / denominators[:, None]
    ba = (ba * (100./n_classes)).reshape((len(weights),) + predictions.shape[:-1])
    return ba[..., 1:] - ba[..., :1]


def iter_joint_bootstrap(cache, n_draws, block_length=10, seed=20260910,
                         dtype=np.float32, max_attempt_factor=100, proposal_batch_size=512):
    """Yield one subject-day's joint draws and float64 covariance at a time.

    ``draws`` has shape [draw,seed,method], seed-major covariance columns are
    [seed0.method0,...,seed1.method0,...]. Covariances and draw means are
    computed before optional float32 archival conversion. Distinct subject-day
    groups have independent deterministic streams; iteration order is irrelevant.
    This generator keeps memory proportional to a single subject-day.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Use float32 or float64 draw storage")
    if int(n_draws) != n_draws or n_draws < 2 or int(seed) != seed or seed < 0:
        raise ValueError("At least two draws and a nonnegative integer RNG seed are required")
    source_audit = cache["audit"]
    for group in cache["groups"]:
        identity = f"{source_audit['dataset']}|{source_audit['backbone']}|{group['subject_id']}|{group['target_session']}"
        key_words = np.frombuffer(hashlib.sha256(identity.encode()).digest()[:16], dtype="<u4").tolist()
        rng = np.random.default_rng(np.random.SeedSequence([int(seed)] + key_words))
        counts, audit = moving_block_counts(group["run"], group["y"], group["n_classes"],
                                           n_draws, block_length, rng, max_attempt_factor,
                                           proposal_batch_size)
        draws = deltas_from_counts(group, counts)
        point = deltas_from_counts(group, np.ones((1, len(group["y"]))))[0]
        _close(point, cache["delta"][group["cell_indices"]], f"{identity} count-based point Delta")
        seed_mean = draws.mean(axis=1)
        audit.update(group_identity=identity, master_seed=int(seed), stream_key_words=key_words,
                     computation_dtype="float64", storage_dtype=dtype.name,
                     n_seeds=len(group["seeds"]), n_methods=len(cache["methods"]),
                     covariance_order="seed-major, then cache.methods",
                     resampling_scope="same within-run counts for all seeds, all methods, and R0")
        result = dict(subject_id=group["subject_id"], target_session=group["target_session"],
                      seeds=group["seeds"], cell_indices=group["cell_indices"],
                      point_delta=point, mean_delta=draws.mean(axis=0),
                      covariance=np.cov(draws.reshape(len(draws), -1), rowvar=False, ddof=1),
                      seed_mean_covariance=np.cov(seed_mean, rowvar=False, ddof=1),
                      seed_mean_draws=seed_mean.astype(dtype), draws=draws.astype(dtype), audit=audit)
        yield result


def joint_bootstrap(cache, n_draws, block_length=10, seed=20260910, dtype=np.float32,
                    max_attempt_factor=100, proposal_batch_size=512):
    """Collect the generator into draws[draw,cell,method]; use iter_* to stream."""
    draws = np.empty((n_draws, len(cache["keys"]), len(cache["methods"])), dtype=dtype)
    means = np.empty((n_draws, len(cache["groups"]), len(cache["methods"])), dtype=dtype)
    group_outputs = []
    for i, result in enumerate(iter_joint_bootstrap(cache, n_draws, block_length, seed, dtype,
                                                    max_attempt_factor, proposal_batch_size)):
        draws[:, result["cell_indices"], :] = result.pop("draws")
        means[:, i, :] = result.pop("seed_mean_draws")
        group_outputs.append(result)
    attempts = sum(result["audit"]["attempts"] for result in group_outputs)
    rejected = sum(result["audit"]["rejected"] for result in group_outputs)
    audit = dict(input=cache["audit"], n_draws=int(n_draws), block_length=int(block_length),
                 master_seed=int(seed), storage_dtype=np.dtype(dtype).name,
                 computation_dtype="float64", attempts=attempts, rejected=rejected,
                 rejection_rate=rejected/attempts if attempts else 0.,
                 covariance_order="per subject-day, seed-major then methods",
                 group_audits=[result["audit"] for result in group_outputs])
    return dict(draws=draws, seed_mean_draws=means, groups=group_outputs,
                keys=cache["keys"].copy(), methods=list(cache["methods"]), audit=audit)
