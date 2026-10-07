"""Traceable, VNC-only Male CNS graph extraction and loading.

The graph uses contact counts from ``syn-partners.primary_post``. The supplied
whole-CNS weight table is deliberately not used. Transmitter signs and the
incoming-contact size proxy are model assumptions, not measured conductances
or neuronal volumes. Sparse matrices use postsynaptic rows/presynaptic columns.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy import sparse


DATASET = "male-cns:v1.0"
TRACED_STATUSES = {"Reviewed", "Roughly traced", "Prelim Roughly traced"}
NT_SIGNS = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
NAMED_CPG_TYPES = {"DNg100": "walking_descending_input", "DNb08": "rhythmic_leg_descending_input",
                   "IN17A001": "E1", "INXXX466": "E2", "IN16B036": "I1", "IN19A007": "I2",
                   "IN19B012": "E3", "INXXX464": "E5"}
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
TRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
PARTNERS = "syn-partners-male-cns-v1.0-minconf-0.5.feather"
POINTS = "syn-points-male-cns-v1.0-minconf-0.5.feather"


@dataclass
class Connectome:
    ids: np.ndarray
    sizes: np.ndarray
    W: sparse.csr_matrix
    nodes: list[dict]
    manifest: dict

    @property
    def body_ids(self) -> np.ndarray:
        return self.ids

    @property
    def weights(self) -> sparse.csr_matrix:
        return self.W

    @property
    def size_proxy(self) -> np.ndarray:
        return self.sizes

    def indices(self, **criteria: str) -> np.ndarray:
        return np.array([i for i, row in enumerate(self.nodes)
                         if all(row.get(k) == v for k, v in criteria.items())], dtype=int)


def _batches(path: Path, columns: list[str]) -> Iterator[pa.RecordBatch]:
    with pa.memory_map(str(path), "r") as source:
        schema = ipc.open_file(source).schema
        options = ipc.IpcReadOptions(included_fields=[schema.get_field_index(c) for c in columns])
        reader = ipc.open_file(source, options=options)
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(path: Path, expected: dict | None = None) -> dict:
    """Keep previously verified hashes; mark whether rehashed in this build."""
    verified_path = path.with_suffix(path.suffix + ".verified.json")
    verified = json.loads(verified_path.read_text()) if verified_path.exists() else {}
    size = path.stat().st_size
    if expected and size != expected["bytes"]:
        raise ValueError(f"Source size changed: {path}")
    if verified and size != verified["bytes"]:
        raise ValueError(f"Source differs from verification record: {path}")
    digest = verified.get("sha256") or sha256_file(path)
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest,
            "sha256_recomputed_this_build": not bool(verified.get("sha256")),
            "prior_verified_at": verified.get("verified_at"),
            "official": expected}


def derive_vnc_rois(points: Path, progress: Callable[[str], None] = print) -> tuple[list[str], dict]:
    """Derive ROI membership from actual PostSyn major/primary annotations.

    No dictionary-order or prefix heuristic is used: CRN, for example, belongs
    to CentralBrain despite occurring beside VNC labels in the dictionary.
    """
    counts: Counter = Counter()
    total = 0
    for i, batch in enumerate(_batches(points, ["kind", "major", "primary"])):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.equal(table["kind"], "PostSyn")).select(["major", "primary"])
        groups = table.group_by(["major", "primary"]).aggregate([([], "count_all")])
        for row in groups.to_pylist():
            counts[(row["major"], row["primary"])] += row["count_all"]
        total += table.num_rows
        if (i + 1) % 1000 == 0:
            progress(f"ROI mapping: {i + 1} point batches")
    vnc_candidates = sorted({primary for (major, primary) in counts if major == "VNC"})
    ambiguous = [roi for roi in vnc_candidates if any(major != "VNC" and p == roi for major, p in counts)]
    # syn-partners does not include major_post. Conservatively omit complete
    # primary labels that cross major boundaries rather than admit non-VNC
    # contacts. This deliberately loses some true VNC contacts in these nerves.
    vnc = [roi for roi in vnc_candidates if roi not in ambiguous]
    if not {"LegNp(T1)(L)", "LegNp(T1)(R)"}.issubset(vnc):
        raise ValueError("Front-leg VNC ROIs missing")
    return vnc, {"post_synaptic_points_scanned": total,
                 "primary_major_point_counts": [
                     {"major": major, "primary": primary, "n": n}
                     for (major, primary), n in sorted(counts.items())],
                 "ambiguous_primary_rois": ambiguous,
                 "ambiguous_roi_policy": "exclude_entire_primary_ROI",
                 "ambiguous_roi_points_omitted": [
                     {"major": major, "primary": primary, "n": n}
                     for (major, primary), n in sorted(counts.items()) if primary in ambiguous]}


def count_vnc_edges(
    partners: Path,
    vnc_rois: list[str],
    pre_ids: set[int],
    post_ids: set[int],
    *,
    collect_all_incoming: bool = False,
    progress: Callable[[str], None] = print,
) -> tuple[Counter, Counter, dict]:
    """Count contacts without thresholding before aggregation across batches.

    Incoming proxy counts include all presynaptic bodies, including unselected
    ones, whose postsynaptic contact lies in VNC and whose target is selected.
    """
    pre_set = pa.array(sorted(pre_ids), type=pa.int64())
    post_set = pa.array(sorted(post_ids), type=pa.int64())
    roi_set = pa.array(vnc_rois, type=pa.string())
    edges: Counter = Counter()
    incoming: Counter = Counter()
    scanned = contacts = 0
    for i, batch in enumerate(_batches(partners, ["body_pre", "body_post", "primary_post"])):
        table = pa.Table.from_batches([batch])
        scanned += table.num_rows
        mask = pc.and_(pc.is_in(table["body_post"], value_set=post_set),
                       pc.is_in(table["primary_post"], value_set=roi_set))
        table = table.filter(mask)
        if collect_all_incoming and table.num_rows:
            group = table.group_by("body_post").aggregate([([], "count_all")])
            incoming.update({r["body_post"]: r["count_all"] for r in group.to_pylist()})
        table = table.filter(pc.is_in(table["body_pre"], value_set=pre_set))
        contacts += table.num_rows
        if table.num_rows:
            group = table.group_by(["body_pre", "body_post"]).aggregate([([], "count_all")])
            edges.update({(r["body_pre"], r["body_post"]): r["count_all"]
                          for r in group.to_pylist()})
        if (i + 1) % 1000 == 0:
            progress(f"Contact counting: {i + 1} batches, {len(edges):,} distinct pairs")
    return edges, incoming, {"contacts_scanned": scanned, "contacts_matching_filters": contacts,
                             "pairs_before_threshold": len(edges)}


def build_graph(project: Path, output: Path, min_synapses: int = 5,
                progress: Callable[[str], None] = print,
                include_unknown_motor_readouts: bool = True) -> dict:
    if min_synapses < 1:
        raise ValueError("min_synapses must be positive")
    output.mkdir(parents=True, exist_ok=True)
    raw = project / "data/raw/male_cns/v1.0"
    source_manifest = json.loads((project / "configs/male_cns_v1.0_manifest.json").read_text())
    official = {r["name"]: r for r in source_manifest["files"]}
    sources = {name: source_record(raw / name, official[name])
               for name in (ANNOTATIONS, TRANSMITTERS, PARTNERS, POINTS)}
    annotations = {int(row["bodyId"]): row
                   for row in feather.read_table(raw / ANNOTATIONS).to_pylist()}
    nts = {int(row["body"]): row for row in feather.read_table(raw / TRANSMITTERS).to_pylist()}
    eligible = {bid for bid, row in annotations.items()
                if row["statusLabel"] in TRACED_STATUSES
                and nts.get(bid, {}).get("consensus_nt") in NT_SIGNS}
    all_motors = {bid for bid, row in annotations.items()
                  if row["superclass"] == "vnc_motor"}
    front_annotated = {bid for bid in all_motors if annotations[bid]["subclass"] == "fl"}
    traced_front = {bid for bid in front_annotated if annotations[bid]["statusLabel"] in TRACED_STATUSES}
    motors = traced_front if include_unknown_motor_readouts else front_annotated & eligible
    unknown_motor_readouts = motors - eligible
    descending = {bid for bid in eligible if annotations[bid]["superclass"] == "descending_neuron"}
    roi_path = output / "roi_audit.json"
    roi_fingerprint = sources[POINTS]["sha256"]
    if roi_path.exists():
        roi_record = json.loads(roi_path.read_text())
        if roi_record.get("source_sha256") != roi_fingerprint:
            raise ValueError("Cached ROI audit has a different source hash")
        vnc_rois = roi_record["vnc_primary_rois"]
    else:
        progress("Deriving exact VNC primary ROIs from postsynaptic point annotations")
        vnc_rois, audit = derive_vnc_rois(raw / POINTS, progress)
        roi_record = dict(audit, source_sha256=roi_fingerprint, vnc_primary_rois=vnc_rois)
        roi_path.write_text(json.dumps(roi_record, indent=2) + "\n")
    progress(f"Selected {len(motors)} front-leg motors; using {len(vnc_rois)} VNC ROIs")
    to_motors, _, first_audit = count_vnc_edges(raw / PARTNERS, vnc_rois, eligible, motors, progress=progress)
    premotor = {pre for (pre, _), count in to_motors.items()
                if count >= min_synapses and pre not in all_motors and pre not in descending}
    # Direct DN-to-MN contacts remain eligible even if there is no qualifying
    # premotor contact; their separate entry role is recorded below.
    direct_dns = {pre for (pre, _), count in to_motors.items()
                  if count >= min_synapses and pre in descending}
    progress(f"Found {len(premotor)} premotor nodes; querying their descending inputs")
    dn_edges, _, second_audit = count_vnc_edges(raw / PARTNERS, vnc_rois, descending, premotor, progress=progress)
    input_dns = {pre for (pre, _), count in dn_edges.items() if count >= min_synapses} | direct_dns
    selected = motors | premotor | input_dns
    progress(f"Retaining all recurrent VNC connections among {len(selected)} selected nodes")
    recurrent, incoming, final_audit = count_vnc_edges(raw / PARTNERS, vnc_rois, selected & eligible, selected,
                                                      collect_all_incoming=True, progress=progress)
    kept = {edge: count for edge, count in recurrent.items() if count >= min_synapses}
    ids = np.array(sorted(selected), dtype=np.int64)
    index = {int(bid): i for i, bid in enumerate(ids)}
    incoming_counts = np.array([incoming[int(bid)] for bid in ids], dtype=np.int64)
    positive = incoming_counts[incoming_counts > 0]
    median_incoming = float(np.median(positive)) if len(positive) else 1.0
    sizes = np.maximum(incoming_counts, 1) / median_incoming
    author_path = project / "data/raw/pugliese_reference/20260210/wTable_20260210_vncRoisOnly.csv"
    older = {}
    if author_path.exists():
        with author_path.open(newline="") as handle:
            older = {int(r["bodyId"]): r for r in csv.DictReader(handle)}
    nodes = []
    module_matches = volume_matches = 0
    for i, bid_value in enumerate(ids):
        bid = int(bid_value)
        a = annotations[bid]
        old = older.get(bid, {})
        exact_match = bool(old and old.get("type") == a.get("type")
                           and old.get("somaSide") == a.get("somaSide"))
        old_module = old.get("motor module") if exact_match else None
        old_size = float(old["size"]) if exact_match and old.get("size") else None
        module_matches += bool(old_module)
        volume_matches += old_size is not None
        motor_side = motor_side_source = None
        if bid in motors:
            if a.get("rootSide") in {"L", "R"}:
                motor_side, motor_side_source = a["rootSide"], "v1.0_rootSide"
            elif a.get("instance") and a["instance"].endswith(("_L", "_R")):
                suffix = a["instance"][-1]
                if not a.get("somaSide") or suffix == a["somaSide"]:
                    motor_side, motor_side_source = suffix, "v1.0_motor_instance_suffix"
        nodes.append({"index": i, "body_id": bid, "bodyId": bid, "type": a.get("type"),
                      "instance": a.get("instance"), "superclass": a.get("superclass"),
                      "subclass": a.get("subclass"), "status": a.get("statusLabel"),
                      "soma_side": a.get("somaSide"), "root_side": a.get("rootSide"),
                      "soma_neuromere": a.get("somaNeuromere"), "exit_nerve": a.get("exitNerve"),
                      "role": "motor" if bid in motors else "descending" if bid in input_dns else "premotor",
                      "paper_cpg_type_label": NAMED_CPG_TYPES.get(a.get("type")),
                      "is_front_leg_motor": bid in motors, "motor_side": motor_side,
                      "motor_side_source": motor_side_source,
                      "motor_module": old_module or None,
                      "motor_module_source": "Pugliese_20260210_exact_id_type_somaSide_match" if old_module else None,
                      "consensus_nt": nts.get(bid, {}).get("consensus_nt"),
                      "sign": NT_SIGNS.get(nts.get(bid, {}).get("consensus_nt"), 0),
                      "unknown_nt_terminal_readout": bid in unknown_motor_readouts,
                      "outgoing_central_edges_enabled": bid not in unknown_motor_readouts,
                      "vnc_incoming_contacts_all_sources": int(incoming_counts[i]),
                      "size_proxy": float(sizes[i]), "size_proxy_source": "v1.0_VNC_incoming_contact_count",
                      "older_snapshot_volume": old_size,
                      "older_snapshot_volume_source": "Pugliese_20260210_not_v1.0_morphology" if old_size is not None else None})
    ordered_edges = sorted(kept.items())
    pre = np.array([index[a] for (a, _), _ in ordered_edges], dtype=np.int32)
    post = np.array([index[b] for (_, b), _ in ordered_edges], dtype=np.int32)
    counts = np.array([count for _, count in ordered_edges], dtype=np.int64)
    signs = np.array([node["sign"] for node in nodes], dtype=np.int8)
    signed = counts * signs[pre]
    W = sparse.csr_matrix((signed.astype(float), (post, pre)), shape=(len(ids), len(ids)))
    sparse.save_npz(output / "W.npz", W)
    np.savez_compressed(output / "graph_arrays.npz", ids=ids, sizes=sizes,
                        pre=pre, post=post, raw_weights=counts, signed_weights=signed,
                        incoming_contacts=incoming_counts)
    feather.write_feather(pa.Table.from_pylist(nodes), output / "nodes.feather")
    with (output / "nodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(nodes[0]))
        writer.writeheader()
        writer.writerows(nodes)
    feather.write_feather(pa.table({"body_pre": ids[pre], "body_post": ids[post],
                                   "synapse_count": counts, "signed_weight": signed}), output / "edges.feather")
    manifest = {
        "dataset": DATASET, "created_at": datetime.now(timezone.utc).isoformat(),
        "graph_kind": "bilateral_front_leg_VNC_unambiguous_primary_ROI_subgraph", "independent_v1_0_extraction": True,
        "is_author_model_replication": False,
        "source_license": source_manifest["license"], "sources": sources,
        "extraction": {
            "statuses": sorted(TRACED_STATUSES), "transmitters": NT_SIGNS,
            "motor_selector": {"superclass": "vnc_motor", "subclass": "fl"},
            "premotor_selector": "non-vnc-motor, non-descending eligible neuron with >= min_synapses VNC contacts to a front MN",
            "descending_selector": "eligible descending neuron with >= min_synapses VNC contacts to a selected premotor or front MN",
            "retained_edges": "all recurrent pairs within selected nodes, not only feedforward selection edges",
            "min_synapses_per_directed_pair": min_synapses,
            "threshold_applied_after_aggregation": True,
            "roi_field": "syn-partners.primary_post", "vnc_primary_rois": vnc_rois,
            "roi_membership_source": "PostSyn major=VNC in syn-points; all batches checked",
            "ambiguous_primary_rois_excluded": roi_record["ambiguous_primary_rois"],
            "ambiguous_roi_points_omitted": roi_record["ambiguous_roi_points_omitted"],
            "roi_boundary": "postsynaptic location only; no independent presynaptic ROI check",
            "brain_unspecified_and_unmapped_contacts_included": False,
            "uses_whole_CNS_weight_table": False,
            "include_unknown_motor_readouts": include_unknown_motor_readouts,
            "unknown_motor_nt_policy": "retain measured incoming contacts; remove all outgoing central edges; sign=0 is an unknown-output boundary, not an inhibitory or excitatory assignment",
        },
        "matrix": {"orientation": "postsynaptic rows, presynaptic columns", "units": "signed synaptic contact count",
                   "global_gain_applied": False, "shape": list(W.shape), "nonzero_edges": W.nnz,
                   "raw_contact_total": int(counts.sum()), "positive_edges": int(np.sum(signed > 0)),
                   "negative_edges": int(np.sum(signed < 0)), "self_edges": int(np.sum(pre == post))},
        "size_normalization": {
            "method": "max(all-source VNC incoming contacts,1) / median_positive_contacts_of_selected_nodes",
            "median_contacts": median_incoming, "is_morphological_volume": False,
            "new_model_assumption": True, "physiological_calibration": False,
            "zero_contact_floor": 1, "zero_incoming_nodes": int(np.sum(incoming_counts == 0)),
            "min_proxy": float(sizes.min()), "max_proxy": float(sizes.max()),
        },
        "counts": {"nodes": len(ids), "eligible_nodes_in_annotations": len(eligible),
                   "front_motors_annotated": len(front_annotated), "front_motors_retained": len(motors),
                   "front_motors_excluded": sorted(front_annotated - motors), "premotor_nodes": len(premotor),
                   "descending_nodes": len(input_dns), "direct_motor_descending_nodes": len(direct_dns),
                   "roles": dict(Counter(n["role"] for n in nodes)),
                   "transmitters": dict(Counter(n["consensus_nt"] or "missing" for n in nodes)),
                   "motor_sides": dict(Counter(n["motor_side"] or "unresolved" for n in nodes if n["is_front_leg_motor"])),
                   "motor_module_matches": module_matches, "older_volume_matches": volume_matches,
                   "zero_retained_incoming_nodes": int(np.sum(np.diff(W.indptr) == 0)),
                   "zero_retained_incoming_motor_ids": [int(ids[i]) for i in range(len(ids))
                       if nodes[i]["role"] == "motor" and W.indptr[i] == W.indptr[i + 1]]},
        "unknown_motor_readout_body_ids": sorted(unknown_motor_readouts),
        "named_candidates": [n for n in nodes if n.get("type") in NAMED_CPG_TYPES],
        "named_cpg_type_source": "Pugliese v2 Supplementary Table 5 and Results; type label matching does not establish identical circuitry",
        "named_candidate_note": "All descending and premotor IDs remain available in nodes.csv for screening; no named input is assumed to selectively lift one leg.",
        "motor_laterality_note": "motor_side uses rootSide or explicit motor instance suffix; unresolved types are not assigned from somaSide alone. Projection/target geometry has not independently been validated.",
        "optional_older_annotation": {"path": str(author_path), "sha256": sha256_file(author_path) if older else None,
                                     "join": "bodyId AND type AND somaSide exact", "used_for_default_sizes": False,
                                     "source_commit": "faee4b06869855ae0164cbf217fb6ec28ef3521b",
                                     "source_public_release": "not_confirmed_not_assumed_v1.0"},
        "counting_passes": [first_audit, second_audit, final_audit],
        "limitations": ["No visual or learned cue circuit", "No simulated activity or biological validation in graph extraction",
                        "Three ambiguous primary nerve labels are conservatively omitted, including their true VNC contacts",
                        "Unknown-transmitter motor neurons may be retained as terminal readouts; their outgoing central connections are omitted without neurotransmitter imputation",
                        "Glutamate inhibitory sign follows a VNC model assumption, not cell-specific receptor validation",
                        "Incoming-contact proxy differs from author's morphology normalization",
                        "Annotated circuit subset is not the full CNS"],
    }
    strict_baseline = output / "strict_known_nt_baseline/manifest.json"
    if strict_baseline.exists():
        manifest["prior_strict_known_nt_baseline"] = {"path": str(strict_baseline),
                                                       "manifest_sha256": sha256_file(strict_baseline)}
    manifest["outputs"] = {name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
                           for name in ["W.npz", "graph_arrays.npz", "nodes.csv", "nodes.feather", "edges.feather", "roi_audit.json"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    progress(json.dumps({"output": str(output), "counts": manifest["counts"], "matrix": manifest["matrix"]}, indent=2))
    return manifest


def load_connectome(directory: str | Path) -> Connectome:
    directory = Path(directory)
    arrays = np.load(directory / "graph_arrays.npz", allow_pickle=False)
    nodes = feather.read_table(directory / "nodes.feather").to_pylist()
    W = sparse.load_npz(directory / "W.npz").tocsr()
    manifest = json.loads((directory / "manifest.json").read_text())
    ids, sizes = arrays["ids"], arrays["sizes"]
    if W.shape != (len(ids), len(ids)) or [n["body_id"] for n in nodes] != ids.tolist():
        raise ValueError("Matrix, IDs, and node annotations have inconsistent order")
    if not np.isfinite(W.data).all() or not np.all(sizes > 0):
        raise ValueError("Graph has nonfinite weights or invalid size proxies")
    return Connectome(ids=ids, sizes=sizes, W=W, nodes=nodes, manifest=manifest)


load_graph = load_connectome
