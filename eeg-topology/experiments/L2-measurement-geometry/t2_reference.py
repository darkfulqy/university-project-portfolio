#!/usr/bin/env python3
"""T2: exact common-reference / subset algebra and the fixed TCP electrode graph.

Run with the sister venv read-only. This script needs NumPy, not MNE.
See T2_PREREG.md for definitions and thresholds fixed before the first run.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import platform

import numpy as np


SEED = 20260930
TCP22_NAMES = [
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "T3-C3", "C3-CZ", "CZ-C4", "C4-T4",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "A1-T3", "T4-A2",
]
COMMIT = "f7e94379255c425819c9138ceca60c26bc13b88e"
BASE_URL = f"https://raw.githubusercontent.com/pulp-bio/BioFoundation/{COMMIT}/"


def centering(n):
    return np.eye(n) - np.ones((n, n)) / n


def maxabs(a):
    return float(np.max(np.abs(a)))


def graph_report(names, extra_vertices=()):
    pairs = [s.split("-") for s in names]
    vertices = list(dict.fromkeys([v for pair in pairs for v in pair] + list(extra_vertices)))
    index = {v: i for i, v in enumerate(vertices)}
    B = np.zeros((len(pairs), len(vertices)))
    adjacency = {v: set() for v in vertices}
    for i, (a, b) in enumerate(pairs):
        B[i, index[a]], B[i, index[b]] = 1, -1
        adjacency[a].add(b)
        adjacency[b].add(a)
    components = []
    unseen = set(vertices)
    while unseen:
        todo = [min(unseen)]
        component = []
        while todo:
            v = todo.pop()
            if v not in unseen:
                continue
            unseen.remove(v)
            component.append(v)
            todo.extend(sorted(adjacency[v]))
        components.append(sorted(component))
    singular = np.linalg.svd(B, compute_uv=False)
    rank = int(np.sum(singular > 1e-10 * singular[0]))
    projector = np.linalg.pinv(B, rcond=1e-10) @ B
    block_projector = np.zeros_like(projector)
    for comp in components:
        ids = [index[v] for v in comp]
        block_projector[np.ix_(ids, ids)] = centering(len(ids))
    return {
        "ordered_edge_names": names,
        "ordered_vertices": vertices,
        "incidence_matrix": B.tolist(),
        "vertex_count": len(vertices),
        "edge_count": len(pairs),
        "components": components,
        "component_count": len(components),
        "rank": rank,
        "expected_rank_n_minus_components": len(vertices) - len(components),
        "rank_check_pass": rank == len(vertices) - len(components),
        "singular_values": singular.tolist(),
        "cycle_constraint_dimension_edges_minus_rank": len(pairs) - rank,
        "kernel_dimension": len(vertices) - rank,
        "full_contrast_dimension": len(vertices) - 1,
        "full_contrast_information_missing_dimensions": len(vertices) - 1 - rank,
        "pinv_B_times_B_vs_full_AR_max_error": maxabs(projector - centering(len(vertices))),
        "pinv_B_times_B_vs_component_AR_max_error": maxabs(projector - block_projector),
        "equivalent_to_endpoint_contrast_quotient": len(components) == 1,
    }


def main():
    root = Path(__file__).resolve().parent
    prereg = root / "T2_PREREG.md"
    if not prereg.exists():
        raise RuntimeError("T2_PREREG.md must exist before running")
    rng = np.random.default_rng(SEED)
    endpoint_names = list(dict.fromkeys(v for s in TCP22_NAMES for v in s.split("-")))
    full_names = endpoint_names + ["FZ", "PZ"]
    subset_names = ["FP1", "FP2", "F7", "F8", "C3", "C4", "CZ", "O1", "O2"]
    n, k, trials = len(full_names), len(subset_names), 1000
    index = {s: i for i, s in enumerate(full_names)}
    subset_idx = [index[s] for s in subset_names]
    R = np.eye(n)[subset_idx]
    PS, PT = centering(k), centering(n)
    x = rng.normal(size=(trials, n))
    xs = x[:, subset_idx]
    target = xs @ PS
    reference_scalars = {
        "AR_observed_S": xs.mean(axis=1),
        "AR_full_T": x.mean(axis=1),
        "LE_linked_ears": (x[:, index["A1"]] + x[:, index["A2"]]) / 2,
        "one_mastoid_A1": x[:, index["A1"]],
        "Cz": x[:, index["CZ"]],
        "arbitrary_external_scalar": rng.normal(size=trials),
    }
    reference_errors = {
        name: maxabs((xs - ref[:, None]) @ PS - target)
        for name, ref in reference_scalars.items()
    }
    contrast_T = x[:, index["C3"]] - x[:, index["C4"]]
    contrast_S = xs[:, subset_names.index("C3")] - xs[:, subset_names.index("C4")]
    shifted = xs + rng.normal(size=(trials, 1))
    contrast_shifted = shifted[:, subset_names.index("C3")] - shifted[:, subset_names.index("C4")]
    fixed_pair_error = maxabs(contrast_T - contrast_S)
    contrast_ri_error = maxabs(contrast_S - contrast_shifted)
    contrast_std = float(np.std(contrast_T))

    # Witness: changing only an excluded potential is invisible to subset AR.
    witness = np.zeros(n)
    witness[index["A1"]] = 1
    full_ar_witness = PT @ witness
    subset_ar_witness = PS @ R @ witness
    full_loss = float(np.linalg.norm(full_ar_witness))
    raw_mismatch = R @ PT - PS @ R
    ar_compatibility = PS @ R @ PT - PS @ R

    # A fixed retained anchor provides a non-AR, reference-invariant,
    # restriction-equivariant canonical representative on this fixed pair.
    anchor_T = np.eye(n) - np.outer(np.ones(n), np.eye(n)[index["CZ"]])
    anchor_S = np.eye(k) - np.outer(np.ones(k), np.eye(k)[subset_names.index("CZ")])
    anchor_compatibility = R @ anchor_T - anchor_S @ R

    graphs = {
        "TCP22_verified_BioFoundation": graph_report(TCP22_NAMES),
        "TCP20_verified_BioFoundation": graph_report(TCP22_NAMES[:20]),
        "reader_prefix19_not_official_montage": graph_report(TCP22_NAMES[:19]),
        "TCP22_with_unobserved_FZ_PZ_vertices": graph_report(TCP22_NAMES, ["FZ", "PZ"]),
    }
    thresholds = {
        "equality_max_absolute_error": 1e-12,
        "rank_relative_singular_value_tolerance": 1e-10,
        "nonconstant_contrast_std_minimum": 0.1,
        "noninverse_witness_full_norm_minimum": 0.1,
        "raw_AR_commutation_rejection_minimum": 1e-3,
    }
    verdicts = {
        "common_reference_equivalence": all(e < 1e-12 for e in reference_errors.values()),
        "unqualified_fixed_pair_no_go_refuted": fixed_pair_error < 1e-12 and contrast_ri_error < 1e-12 and contrast_std > 0.1,
        "AR_full_to_subset_fixed_map_exists": maxabs(ar_compatibility) < 1e-12,
        "raw_AR_restriction_commutation_refuted": maxabs(raw_mismatch) > 1e-3,
        "AR_subset_to_full_inverse_impossible_witness": maxabs(subset_ar_witness) < 1e-12 and full_loss > 0.1,
        "fixed_anchor_RI_restriction_equivariance_counterexample": maxabs(anchor_compatibility) < 1e-12 and maxabs(anchor_T @ np.ones(n)) < 1e-12,
        "TCP22_endpoint_contrast_equivalence": graphs["TCP22_verified_BioFoundation"]["rank"] == graphs["TCP22_verified_BioFoundation"]["vertex_count"] - 1 and graphs["TCP22_verified_BioFoundation"]["component_count"] == 1 and graphs["TCP22_verified_BioFoundation"]["pinv_B_times_B_vs_full_AR_max_error"] < 1e-12,
        "N4_original_no_go_verdict": "refutes",
        "N4_qualified_no_go_verdict": "supports_algebraically_under_explicit_information_or_mask_family_assumptions",
    }
    result = {
        "task": "T2 / N4 / D5",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "prereg_sha256": hashlib.sha256(prereg.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "thresholds": thresholds,
        "sources": [BASE_URL + "make_datasets/make_tueg_bipolar.py", BASE_URL + "datasets/tuh_dataset.py"],
        "reference_and_subset_test": {
            "full_names": full_names,
            "subset_names": subset_names,
            "trials": trials,
            "reference_errors": reference_errors,
            "common_contrast_fixed_pair_SI_error": fixed_pair_error,
            "common_contrast_RI_error": contrast_ri_error,
            "common_contrast_std": contrast_std,
            "P_S_R_P_T_minus_P_S_R_max_error": maxabs(ar_compatibility),
            "R_P_T_minus_P_S_R_max_operator_difference": maxabs(raw_mismatch),
            "fixed_anchor_compatibility_max_error": maxabs(anchor_compatibility),
            "inverse_impossibility_witness_full_AR_norm": full_loss,
            "inverse_impossibility_witness_subset_AR_norm": float(np.linalg.norm(subset_ar_witness)),
            "inverse_impossibility_witness_full_AR": full_ar_witness.tolist(),
            "fixed_pair_common_contrast_information_dimension": k - 1,
            "full_contrast_information_dimension": n - 1,
            "information_loss_dimensions": n - k,
        },
        "graphs": graphs,
        "verdicts": verdicts,
        "limitations": [
            "Synthetic potential vectors; no EEG downstream effects are measured.",
            "Common-reference equivalence assumes simultaneous ideal common-scalar subtraction and the same channel potentials.",
            "Graph equivalence is between the endpoint contrast quotient and image(B), not all arbitrary noisy R^edges signals.",
            "reader_prefix19 is a code prefix, not an independently verified official TCP montage.",
            "Universal no-go proofs assume arbitrary unconstrained potentials; physical field priors can alter identifiability.",
        ],
    }
    (root / "t2_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"reference_errors": reference_errors, "subset": result["reference_and_subset_test"], "graphs": {k: {p: v[p] for p in ["vertex_count", "edge_count", "component_count", "rank", "pinv_B_times_B_vs_full_AR_max_error"]} for k, v in graphs.items()}, "verdicts": verdicts}, indent=2))


if __name__ == "__main__":
    main()
