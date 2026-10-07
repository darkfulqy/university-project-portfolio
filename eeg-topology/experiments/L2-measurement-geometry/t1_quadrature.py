"""T1: support-limited, real-layout quadrature drift and refinement checks.

Run with the read-only sibling Python; no packages or data are installed.
See T1_PREREG.md for decisions fixed before the first numerical run.
"""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import SphericalVoronoi, cKDTree

from common import SEED, layouts, real_sh, jsonable

ROOT = Path(__file__).resolve().parent
N_GRID = 65536
N_TRIALS = 512
SIGMAS = (0.0, 0.1, 0.3)
GROUPS = {"integral": [0], "low_spectrum": list(range(1, 9))}


def fibonacci(n, hemisphere=False, alpha=0.0):
    i = np.arange(n)
    t = (i + 0.5) / n
    if hemisphere:
        z = t
    elif alpha:
        z = (-1 + np.sqrt((1-alpha)**2 + 4*alpha*t))/alpha
    else:
        z = 2*t - 1
    phi = i * np.pi * (3-np.sqrt(5))
    r = np.sqrt(np.maximum(0, 1-z*z))
    return np.column_stack((r*np.cos(phi), r*np.sin(phi), z))


def farthest_subset(points, count, initial):
    selected = list(np.flatnonzero(initial))
    if not selected:
        selected = [int(np.argmax(points[:, 2]))]
    min_dist = np.min(1-points @ points[selected].T, axis=1)
    min_dist[selected] = -1
    while len(selected) < count:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, 1-points @ points[nxt])
        min_dist[selected] = -1
    return np.sort(selected)


def make_pairs():
    d = layouts()
    out = {}
    for key in ("ten_ten", "hydrocel129"):
        raw = d[key]
        cap = np.flatnonzero(raw["unit"][:, 2] >= 0)
        p = raw["unit"][cap]
        boundary = p[:, 2] <= 0.18
        if key == "ten_ten":
            retained = boundary | (p[:, 1] >= 0)
            count = int(retained.sum() + np.ceil((~retained).sum()/4))
            label = "ten_ten_frontal_dense"
        else:
            retained = boundary
            count = max(int(np.ceil(len(p)/2)), int(retained.sum())+1)
            label = "hydrocel_uniform_half"
        sub = farthest_subset(p, count, retained)
        out[label] = {
            "source_layout": key,
            "source_n": len(raw["names"]),
            "names": [raw["names"][i] for i in cap],
            "points": p,
            "source_indices": cap,
            "subset_indices": sub,
            "subset_names": [raw["names"][cap[i]] for i in sub],
            "boundary_names": [raw["names"][cap[i]] for i in np.flatnonzero(boundary)],
            "n_full": len(p), "n_subset": len(sub),
        }
    return out


def coverage(points, probes):
    dist, _ = cKDTree(points).query(probes)
    a = 2*np.arcsin(np.clip(dist/2, 0, 1))
    return {"sampled_fill_radians": float(a.max()),
            "mean_nearest_radians": float(a.mean()),
            "p90_nearest_radians": float(np.quantile(a, .9))}


def clipped_voronoi(points, probes):
    _, assignment = cKDTree(points).query(probes)
    return np.bincount(assignment, minlength=len(points)) / len(probes)


def full_voronoi(points):
    sv = SphericalVoronoi(points, radius=1, center=np.zeros(3))
    weights = sv.calculate_areas() / (4*np.pi)
    assert abs(weights.sum()-1) < 1e-10
    return weights, sv


def kde_inverse(points, probes, boundary_correct=True):
    # Division by the kernel's cap mass avoids treating the support boundary as
    # a reduction in sensor density. It is not a fitted field-dependent weight.
    tau = .35
    estimated = np.exp((points @ points.T - 1)/tau**2).mean(axis=1)
    if boundary_correct:
        kernel_mass = np.zeros(len(points))
        for q in np.array_split(probes, 16):
            kernel_mass += np.exp((points @ q.T - 1)/tau**2).sum(axis=1)
        estimated /= kernel_mass / len(probes)
    w = 1/estimated
    return w/w.sum()


def all_weights(points, probes, full_sphere=False):
    return {"uniform": np.full(len(points), 1/len(points)),
            "voronoi": full_voronoi(points)[0] if full_sphere else clipped_voronoi(points, probes),
            "kde_inverse": kde_inverse(points, probes, boundary_correct=not full_sphere)}


def group_rms(matrix, indices):
    return float(np.sqrt(np.mean(np.sum(matrix[indices]**2, axis=1))))


def evaluate_pair(pair, weights, transform, targets, pair_index, support):
    p, sub = pair["points"], pair["subset_indices"]
    basis = real_sh(p, 2) @ transform
    results = []
    for L in (6, 10):
        phi = real_sh(p, L)
        variance = 4*np.pi/(L+1)**2
        rng = np.random.default_rng(SEED + pair_index*1000 + L)
        coefficients = rng.normal(size=((L+1)**2, N_TRIALS)) * np.sqrt(variance)
        clean = phi @ coefficients
        noise = rng.normal(size=(len(p), N_TRIALS))
        truth = targets[L]
        for method, (wf, ws) in weights.items():
            Rf = (wf[:, None] * basis).T
            Rs = (ws[:, None] * basis[sub]).T
            Of, Os = Rf @ phi, Rs @ phi[sub]
            shared_noise_operator = Rf.copy()
            shared_noise_operator[:, sub] -= Rs
            delta_field_operator = (Of-Os) * np.sqrt(variance)
            for sigma in SIGMAS:
                y = clean + sigma*noise
                empirical_delta = Rf @ y - Rs @ y[sub]
                for group, inds in GROUPS.items():
                    field_drift = group_rms(delta_field_operator, inds)
                    noise_drift = sigma*group_rms(shared_noise_operator, inds)
                    independent_noise_drift = sigma*np.sqrt(group_rms(Rf, inds)**2 + group_rms(Rs, inds)**2)
                    errors = {}
                    for role, R, O in (("full", Rf, Of), ("subset", Rs, Os)):
                        discretization = np.sqrt(variance)*group_rms(O-truth, inds)
                        noise_error = sigma*group_rms(R, inds)
                        errors[role] = {
                            "rms_field_discretization_error": discretization,
                            "rms_observation_noise_error": noise_error,
                            "total_rmse": float(np.hypot(discretization, noise_error)),
                        }
                    results.append({
                        "support": support, "pair": pair["label"], "L": L,
                        "sigma": sigma, "method": method, "target": group,
                        "drift_population_rms": float(np.hypot(field_drift, noise_drift)),
                        "drift_field_rms": field_drift,
                        "drift_shared_noise_rms": noise_drift,
                        "drift_independent_acquisition_rms": float(np.hypot(field_drift, independent_noise_drift)),
                        "drift_monte_carlo_rms": float(np.sqrt(np.mean(empirical_delta[inds]**2))),
                        "target_errors": errors,
                    })
    return results


def aggregate(rows, support="upper_hemisphere", sigma=.1, pair=None, L=None, key="drift_population_rms"):
    groups = {}
    for m in ("uniform", "voronoi", "kde_inverse"):
        values = [r[key]**2 for r in rows if r["support"] == support and r["sigma"] == sigma
                  and r["method"] == m and (pair is None or r["pair"] == pair)
                  and (L is None or r["L"] == L)]
        groups[m] = float(np.sqrt(np.mean(values)))
    return {"rms": groups, "uniform_over_voronoi": groups["uniform"]/groups["voronoi"],
            "uniform_over_kde": groups["uniform"]/groups["kde_inverse"]}


def refinement():
    entries = []
    a = .8
    vnorm = np.sqrt(1.16)
    integral_exp = np.sinh(vnorm)/vnorm
    # E[z exp(v.x)] = v_z/||v|| times the radial derivative of sinh(r)/r.
    asymptotic_exp_bias = a*(vnorm*np.cosh(vnorm)-np.sinh(vnorm))/vnorm**3
    for alpha in (0., a):
        for n in (32, 64, 128, 256, 512, 1024):
            p = fibonacci(n, alpha=alpha)
            w, sv = full_voronoi(p)
            vertex_dist, _ = cKDTree(p).query(sv.vertices)
            h = float((2*np.arcsin(np.clip(vertex_dist/2, 0, 1))).max())
            for name, vals, truth, lip, asymptotic in (
                ("z", p[:, 2], 0., 1., alpha/3),
                ("exp_z_plus_point4x", np.exp(p[:, 2]+.4*p[:, 0]), integral_exp,
                 vnorm*np.exp(vnorm), asymptotic_exp_bias if alpha else 0.),
            ):
                eu = abs(vals.mean()-truth)
                ev = abs(w @ vals-truth)
                entries.append({"alpha": alpha, "n": n, "field": name,
                                "true_integral": float(truth),
                                "uniform_error": float(eu), "voronoi_error": float(ev),
                                "covering_radius_radians": h,
                                "voronoi_error_over_h": float(ev/h),
                                "lipschitz_bound": float(lip*h),
                                "bound_satisfied": bool(ev <= lip*h + 1e-9),
                                "uniform_asymptotic_bias": float(asymptotic)})
    uz = next(r for r in entries if r["alpha"] == 0 and r["n"] == 1024 and r["field"] == "z")
    nz = next(r for r in entries if r["alpha"] == a and r["n"] == 1024 and r["field"] == "z")
    passed = abs(nz["uniform_error"]-a/3)<.02 and nz["uniform_error"]>=.1 and nz["voronoi_error"]<.01 and uz["uniform_error"]<.01
    return {"rows": entries, "density_separation_pass": bool(passed),
            "all_lipschitz_bounds_satisfied": all(r["bound_satisfied"] for r in entries),
            "uniform_refinement_has_Omega1_bias": False,
            "interpretation": "Uniform refinement has no necessary persistent bias. Persistent unweighted bias requires a limiting sampling measure different from the target measure; Voronoi error is bounded by Lip(f)*h."}


def table_real(data):
    rows = ["| Pair | L | Uniform drift | Voronoi drift | KDE drift | a/b |",
            "|---|---:|---:|---:|---:|---:|"]
    for pair in data["pairs"]:
        for L in (6, 10):
            a = aggregate(data["rows"], pair=pair, L=L)
            r = a["rms"]
            rows.append(f"| {pair} | {L} | {r['uniform']:.6f} | {r['voronoi']:.6f} | {r['kde_inverse']:.6f} | {a['uniform_over_voronoi']:.3f} |")
    return "\n".join(rows)


def write_report(data):
    agg = data["primary"]
    r = agg["rms"]
    lines = [
        "# T1: real-layout quadrature drift — L2 synthetic evidence",
        "",
        "All thresholds and estimands were written in `T1_PREREG.md` before the first run. "
        "No EEG recordings or downstream labels are used; template electrodes are real, fields are synthetic.",
        "",
        f"**Predeclared primary result:** a/b = **{agg['uniform_over_voronoi']:.4f}** "
        f"at sigma = 0.1 (uniform {r['uniform']:.6f}, clipped Voronoi {r['voronoi']:.6f}, "
        f"boundary-corrected inverse KDE {r['kde_inverse']:.6f}). "
        f"Gate verdict: **{data['verdict']}**. The threshold is 1.5; "
        "passing establishes only this synthetic estimator mechanism, not EEG task benefit or novelty.",
        "",
        "**The aggregate pass is not uniform across layouts:** the 10-10 pair has a/b = "
        f"{aggregate(data['rows'], pair='ten_ten_frontal_dense')['uniform_over_voronoi']:.4f}, below 1.5. "
        "At L=10 its Voronoi drift is slightly larger than the unweighted drift. "
        "This negative cell must remain visible when deciding whether N5 can be a main contribution.",
        "",
        "## Reproduce",
        "",
        "```sh",
        "PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\",
        "  /path/to/local-workspaces/eeg_data_distilation/evaluation/repro/.venv/bin/python \\",
        "  experiments/L2-measurement-geometry/t1_quadrature.py",
        "```",
        "",
        "The script reads local MNE templates through `common.py`; it does not install packages or modify the borrowed environment. "
        "`t1_results.json` includes the names, unit coordinates, selection masks, all weights, and every condition. "
        f"Seed {SEED}; 512 Gaussian fields per pair/bandwidth; noise sigma 0/0.1/0.3; L = 6/10.",
        "",
        "## Geometry and target definition",
        "",
        "The common coordinate frame is MNE's fiducial head frame, with unit = xyz/||xyz|| and no layout-specific fitted sphere. "
        "The fixed integration domain is the upper hemisphere z >= 0, under normalized area. "
        "The 10-10 source is the explicit 86-name extended standard_1005 base grid, not a claim that there is one canonical universal 10-10 cap. "
        "Only electrodes lying in the domain are used. A common boundary belt z <= 0.18 is retained in each pair. "
        "The electrode sets do not have identical finite-resolution coverage: we measure the difference below.",
        "",
        "| Pair | Cap full/subset | Boundary count | Full fill (rad) | Subset fill (rad) | Fill ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, pair in data["pairs"].items():
        c = pair["coverage"]
        lines.append(f"| {name} | {pair['n_full']}/{pair['n_subset']} | {len(pair['boundary_names'])} | "
                     f"{c['full']['sampled_fill_radians']:.4f} | {c['subset']['sampled_fill_radians']:.4f} | {c['fill_ratio']:.3f} |")
    lines += [
        "",
        "Fill distances are maximum nearest-sensor angular distances over the fixed 65,536-point equal-area grid; "
        "they approximate, rather than certify, the continuous-domain fill distance. The coverage-confounding flag is ratio > 2.5. "
        "A retained boundary gap can dominate this maximum, so equal maximum fill distances do not imply identical spatial resolution everywhere. "
        "The 10-10 pair keeps the frontal region dense and thins interior posterior points to one quarter; "
        "HydroCel uses farthest-point half sampling.",
        "",
        "For field f = Phi_L a, a has independent N(0, 4*pi/(L+1)^2) coefficients; hence pointwise variance is one. "
        "Targets are mean integral plus the other eight coefficients in a cap-orthonormalized l <= 2 basis. "
        "These are cap coefficients, not a claim to recover full-sphere spectral coefficients from a partial cap. "
        "Readout j is sum_i w_i B_j(u_i) y_i. We compare uniform, clipped Voronoi area, and inverse KDE "
        "(spherical bandwidth 0.35 rad with support boundary correction).",
        "",
        "## Drift, accuracy, and noise",
        "",
        "The main statistic is population RMS full-minus-subset drift. Because the estimators are linear Gaussian, "
        "it is computed exactly from their operators; sampled fields independently check the implementation. "
        "Within each target family coefficients are equally weighted; the two families, two pairs, and two L settings then get equal mean-squared weight. "
        "Retained electrode noise is shared for channel deletion. Independent-acquisition noise is included in JSON.",
        "",
        table_real(data),
        "",
        "Per-pair aggregates (over L and target families):",
        "",
    ]
    for name in data["pairs"]:
        a = aggregate(data["rows"], pair=name)
        lines.append(f"- {name}: a/b = {a['uniform_over_voronoi']:.4f}; a/KDE = {a['uniform_over_kde']:.4f}.")
    lines += ["", "Noise sensitivity (same aggregation):", "",
              "| sigma | Uniform drift | Voronoi drift | KDE drift | a/b |",
              "|---:|---:|---:|---:|---:|"]
    for sigma in SIGMAS:
        a = aggregate(data["rows"], sigma=sigma)
        r = a["rms"]
        lines.append(f"| {sigma:g} | {r['uniform']:.6f} | {r['voronoi']:.6f} | {r['kde_inverse']:.6f} | {a['uniform_over_voronoi']:.4f} |")
    lines += [
        "",
        "Absolute target RMSE at sigma = 0.1 is required to distinguish stable wrong answers from useful quadrature:",
        "",
        "| Pair | Role | Uniform RMSE | Voronoi RMSE | KDE RMSE |",
        "|---|---|---:|---:|---:|",
    ]
    for name in data["pairs"]:
        for role in ("full", "subset"):
            vals = []
            for method in ("uniform", "voronoi", "kde_inverse"):
                rr = [x for x in data["rows"] if x["support"] == "upper_hemisphere" and x["sigma"] == .1
                      and x["pair"] == name and x["method"] == method]
                vals.append(np.sqrt(np.mean([x["target_errors"][role]["total_rmse"]**2 for x in rr])))
            lines.append(f"| {name} | {role} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} |")
    lines += [
        "",
        "JSON separately records field-discretization RMS and observation-noise RMS for each target/layout. "
        "The random-field ensemble has mean zero, so its signed ensemble bias is zero for all linear readers. "
        "The reported discretization term is the RMS conditional-on-field quadrature bias, not a nonzero signed ensemble mean.",
        "",
        f"Clipped-area grid doubled to 131,072: a/b = {data['grid_sensitivity']['ratio_fine']:.4f}, "
        f"relative change {100*data['grid_sensitivity']['relative_ratio_change']:.3f}%; preregistered tolerance 5%. "
        f"Monte Carlo aggregate a/b = {data['monte_carlo']['uniform_over_voronoi']:.4f}; "
        f"independent-acquisition-noise a/b = {data['independent_noise']['uniform_over_voronoi']:.4f}.",
        "",
        "## Full-sphere sensitivity: extrapolation is a separate problem",
        "",
        f"The same cap electrodes with *unclipped* spherical Voronoi areas give aggregate a/b = "
        f"{data['full_sphere_sensitivity']['uniform_over_voronoi']:.4f}. "
        "This assigns large areas of the unseen lower hemisphere to boundary electrodes and targets the full sphere. "
        "It is not the primary density result, and its apparent drift reduction must not be interpreted as reconstructing missing support.",
        "",
        "## D2: uniform refinement does not force persistent bias",
        "",
        "For normalized area mu and Voronoi cells V_i, w_i = mu(V_i). "
        "If f is K-Lipschitz in angular distance and h is the covering radius, "
        "|sum_i w_i f(u_i) - integral f dmu| <= sum_i integral_(V_i) K d(u,u_i) dmu <= K h. "
        "Thus Voronoi O(h) is a direct bound, not an extrapolated fit. Uniform empirical measures converging to mu also integrate "
        "continuous fields consistently; unweighted Omega(1) bias requires a different limiting sampling measure.",
        "",
        "The nonuniform refinement uses density p(u) = (1 + 0.8 z)/(4*pi). "
        "For f=z, uniform-sphere truth is zero, but the unweighted limit is 0.8/3. "
        "Fibonacci sequences are deterministic; full-sphere Voronoi areas and vertex covering radii are exact to numerical precision.",
        "",
        "| Density alpha | N | Field | Uniform error | Voronoi error | h |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for entry in data["D2"]["rows"]:
        if entry["n"] in (32, 1024):
            lines.append(f"| {entry['alpha']:g} | {entry['n']} | {entry['field']} | {entry['uniform_error']:.7f} | {entry['voronoi_error']:.7f} | {entry['covering_radius_radians']:.4f} |")
    lines += [
        "",
        f"Predeclared density-separation check: {data['D2']['density_separation_pass']}; "
        f"all Lip(f)*h bounds met: {data['D2']['all_lipschitz_bounds_satisfied']}. "
        "All N = 32/64/128/256/512/1024 values and error/h are in JSON.",
        "",
        "## Interpretation and limits",
        "",
        "A density-biased refinement can produce persistent unweighted integration error; genuinely uniform refinement need not. "
        "Finite EEG layouts additionally change coverage and sampling resolution, so the real-layout gate is deliberately stricter "
        "than the existence proof. Area weighting does not remove lost spatial bandwidth, missing support, or reference ambiguity. "
        "This is a linear uniform-attention limit; learned softmax attention can partly compensate or create other biases. "
        "No downstream EEG accuracy claim follows from these L2 outcomes.",
        "",
        f"Preregistration SHA256: `{data['preregistration_sha256']}`.",
    ]
    (ROOT/"T1.md").write_text("\n".join(lines)+"\n")


def main():
    probes = fibonacci(N_GRID, hemisphere=True)
    probes_fine = fibonacci(2*N_GRID, hemisphere=True)
    raw_basis = real_sh(probes, 2)
    gram = raw_basis.T @ raw_basis / N_GRID
    transform = np.linalg.inv(np.linalg.cholesky(gram).T)
    cap_basis = raw_basis @ transform
    targets = {L: cap_basis.T @ real_sh(probes, L)/N_GRID for L in (6, 10)}
    full_transform = np.eye(9)*np.sqrt(4*np.pi)
    full_targets = {}
    for L in (6, 10):
        full_targets[L] = np.zeros((9, (L+1)**2))
        full_targets[L][:9, :9] = np.eye(9)/np.sqrt(4*np.pi)
    pairs = make_pairs()
    rows, fine_rows = [], []
    for index, (label, pair) in enumerate(pairs.items()):
        print("T1 computing", label, pair["n_full"], pair["n_subset"], flush=True)
        pair["label"] = label
        p, sub = pair["points"], pair["subset_indices"]
        cf, cs = coverage(p, probes), coverage(p[sub], probes)
        ratio = cs["sampled_fill_radians"]/cf["sampled_fill_radians"]
        pair["coverage"] = {"full": cf, "subset": cs, "fill_ratio": ratio,
                            "coverage_confounded": bool(ratio>2.5)}
        wf, ws = all_weights(p, probes), all_weights(p[sub], probes)
        weights = {m:(wf[m], ws[m]) for m in wf}
        pair["cap_weights"] = weights
        rows.extend(evaluate_pair(pair, weights, transform, targets, index, "upper_hemisphere"))
        finer = dict(weights)
        finer["voronoi"] = (clipped_voronoi(p, probes_fine), clipped_voronoi(p[sub], probes_fine))
        fine_rows.extend(evaluate_pair(pair, finer, transform, targets, index, "upper_hemisphere"))
        wff, wsf = all_weights(p, probes, full_sphere=True), all_weights(p[sub], probes, full_sphere=True)
        full_weights = {m:(wff[m], wsf[m]) for m in wff}
        pair["full_sphere_weights"] = full_weights
        rows.extend(evaluate_pair(pair, full_weights, full_transform, full_targets, index, "full_sphere_extrapolation"))
    primary = aggregate(rows)
    fine = aggregate(fine_rows)
    change = abs(fine["uniform_over_voronoi"]/primary["uniform_over_voronoi"]-1)
    all_confounded = all(p["coverage"]["coverage_confounded"] for p in pairs.values())
    verdict = "inconclusive" if all_confounded or change>.05 else ("refutes_main_contribution_gate" if primary["uniform_over_voronoi"]<1.5 else "supports_synthetic_mechanism_gate")
    data = {
        "seed": SEED, "trials": N_TRIALS, "grid_n": N_GRID,
        "preregistration_sha256": hashlib.sha256((ROOT/"T1_PREREG.md").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_threshold": 1.5, "primary": primary, "verdict": verdict,
        "per_pair_gates": {name: {"ratio": aggregate(rows, pair=name)["uniform_over_voronoi"],
                                  "pass": aggregate(rows, pair=name)["uniform_over_voronoi"] >= 1.5}
                           for name in pairs},
        "grid_sensitivity": {"n_fine": 2*N_GRID, "ratio_fine": fine["uniform_over_voronoi"], "relative_ratio_change": change},
        "monte_carlo": aggregate(rows, key="drift_monte_carlo_rms"),
        "independent_noise": aggregate(rows, key="drift_independent_acquisition_rms"),
        "full_sphere_sensitivity": aggregate(rows, support="full_sphere_extrapolation"),
        "pairs": pairs, "cap_basis_transform": transform,
        "cap_basis_orthonormality_error": float(np.linalg.norm(cap_basis.T @ cap_basis/N_GRID-np.eye(9))),
        "cap_truth_operators": targets,
        "rows": rows, "D2": refinement(),
        "notes": ["Zero-mean random fields have zero signed ensemble bias; field discretization RMS is conditional-field quadrature bias.",
                  "No per-layout sphere fit, no dataset or training results, no empirical field-dependent weight selection.",
                  "Clipped areas approximate exact Voronoi cells by equal-area deterministic quadrature.",
                  "No claim that standard 10-10 or HydroCel sensors cover a full sphere."]
    }
    (ROOT/"t1_results.json").write_text(json.dumps(jsonable(data), indent=2)+"\n")
    write_report(data)
    print(json.dumps({"primary":primary, "verdict":verdict,
                      "grid_sensitivity":data["grid_sensitivity"],
                      "D2_pass": data["D2"]["density_separation_pass"]}, indent=2))


if __name__ == "__main__":
    main()
