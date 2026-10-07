"""Whole traced Male CNS graph, with explicit structural and sign assumptions.

Matrices have postsynaptic rows and presynaptic columns. The unsigned matrix
preserves every positive published pair count between retained bodies. The
signed matrix is a separate model assumption; it is not a conductance matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import re
from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy import sparse


DATASET = "male-cns:v1.0"
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
TRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NT_SIGNS = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}
PHOTORECEPTOR_PATTERN = re.compile(r"R1-R6|R[78](?:[pyd]|_unclear)|R7R8_unclear")


@dataclass
class FullConnectome:
    ids: np.ndarray
    nodes: pd.DataFrame
    unsigned_counts: sparse.csr_matrix
    W_normalized: sparse.csr_matrix
    input_indices: np.ndarray
    groups: list[dict]
    manifest: dict


def _source_record(path: Path, expected: dict) -> dict:
    verified = json.loads(path.with_suffix(path.suffix + ".verified.json").read_text())
    if path.stat().st_size != expected["bytes"] or path.stat().st_size != verified["bytes"]:
        raise ValueError(f"Source size differs from verified manifest: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": verified["sha256"], "sha256_recomputed_this_build": False,
            "prior_verified_at": verified.get("verified_at"), "official": expected}


def _side(row: pd.Series) -> tuple[str | None, str]:
    if row.get("somaSide") in {"L", "R", "M"}:
        return row["somaSide"], "somaSide"
    instance = row.get("instance")
    match = re.search(r"_([LR])$", instance) if isinstance(instance, str) else None
    if match:
        return match.group(1), "instance_suffix"
    return None, "unavailable"


def _input_group(cell_type: str | None) -> str | None:
    if not isinstance(cell_type, str):
        return None
    if cell_type == "R1-R6":
        return "R1_R6"
    if cell_type == "R7R8_unclear":
        return "R7R8_ambiguous"
    if PHOTORECEPTOR_PATTERN.fullmatch(cell_type):
        return "R7" if cell_type.startswith("R7") else "R8"
    return None


def _groups(nodes: pd.DataFrame) -> list[dict]:
    groups = []
    definitions = []
    for superclass in ["ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal",
                       "cb_intrinsic", "descending_neuron", "vnc_intrinsic"]:
        definitions.append((superclass, nodes.superclass.eq(superclass),
                            {"superclass": superclass}))
    for cell_type in ["L1", "L2", "L3", "L4", "L5", "Mi1", "Tm1"]:
        definitions.append((cell_type, nodes.type.eq(cell_type), {"type": cell_type}))
    for family in ["T4", "T5"]:
        mask = nodes.type.fillna("").str.fullmatch(f"{family}[abcd]")
        definitions.append((family, mask, {"type_regex": f"{family}[abcd]"}))
    for group in ["R1_R6", "R7", "R8", "R7R8_ambiguous"]:
        definitions.append((group, nodes.visual_input_group.eq(group),
                            {"visual_input_group": group}))
    for name, mask, criteria in definitions:
        for side in ["L", "R", "M", None]:
            smask = nodes.observer_side.isna() if side is None else nodes.observer_side.eq(side)
            indices = np.flatnonzero((mask & smask).to_numpy()).tolist()
            if indices:
                groups.append({"name": f"{name}_{side or 'unknown_side'}", "indices": indices,
                               "n": len(indices), "criteria": dict(criteria, observer_side=side),
                               "interpretation": "mean of labeled cell states; not an electrode or anatomical layer"})
    return groups


def build_full_graph(project: Path, output: Path, progress: Callable[[str], None] = print) -> dict:
    project, output = Path(project).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = project / "data/raw/male_cns/v1.0"
    official = {f["name"]: f for f in json.loads((project / "configs/male_cns_v1.0_manifest.json").read_text())["files"]}
    sources = {name: _source_record(raw / name, official[name]) for name in [ANNOTATIONS, TRANSMITTERS, WEIGHTS]}
    annotations = feather.read_table(raw / ANNOTATIONS).to_pandas()
    transmitters = feather.read_table(raw / TRANSMITTERS).to_pandas()
    if not annotations.bodyId.is_unique or not transmitters.body.is_unique:
        raise ValueError("Duplicate annotation/transmitter body IDs")
    all_nodes = annotations.merge(transmitters, left_on="bodyId", right_on="body", how="left", validate="one_to_one")
    nodes = all_nodes.loc[all_nodes.status.eq("Traced")].sort_values("bodyId").reset_index(drop=True).copy()
    nodes.insert(0, "index", np.arange(len(nodes), dtype=np.int32))
    nodes["nt_annotation_present"] = nodes.body.notna()
    nodes["output_sign"] = nodes.consensus_nt.map(NT_SIGNS).fillna(0).astype(np.int8)
    nodes["output_sign_rule"] = nodes.consensus_nt.map({k: "explicit_consensus_nt_model_assumption" for k in NT_SIGNS}).fillna("unsupported_or_missing_nt_output_disabled")
    sides = nodes.apply(_side, axis=1)
    nodes["observer_side"] = [s[0] for s in sides]
    nodes["observer_side_source"] = [s[1] for s in sides]
    nodes["visual_input_group"] = nodes.type.map(_input_group)
    # Achromatic first-pass input uses retained R1-R6 terminals only. R7/R8
    # photoreceptors and HB eyelets remain ordinary graph nodes.
    nodes["is_visual_input"] = nodes.type.eq("R1-R6") & nodes.superclass.eq("ol_sensory")
    ids = nodes.bodyId.to_numpy(np.int64)
    input_indices = np.flatnonzero(nodes.is_visual_input.to_numpy()).astype(np.int32)
    signs = nodes.output_sign.to_numpy(np.int8)
    n = len(ids)
    rows, cols, values = [], [], []
    counts = Counter()
    incoming_all_bodies = np.zeros(n, np.int64)
    outgoing_all_bodies = np.zeros(n, np.int64)
    progress(f"Retaining {n:,} status=Traced nodes; {len(input_indices):,} photoreceptor input nodes")
    with pa.memory_map(str(raw / WEIGHTS), "r") as source:
        reader = ipc.open_file(source)
        if reader.schema.names != ["body_pre", "body_post", "weight"]:
            raise ValueError(f"Unexpected weight-table schema: {reader.schema}")
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            pre = batch.column(0).to_numpy()
            post = batch.column(1).to_numpy()
            weight = batch.column(2).to_numpy()
            pi, qi = np.searchsorted(ids, pre), np.searchsorted(ids, post)
            pvalid = (pi < n) & (ids[np.minimum(pi, n-1)] == pre)
            qvalid = (qi < n) & (ids[np.minimum(qi, n-1)] == post)
            positive = weight >= 1
            keep = pvalid & qvalid & positive
            counts["source_pair_rows"] += len(weight)
            counts["source_contact_weight"] += int(weight.sum())
            counts["nonpositive_source_pair_rows"] += int((~positive).sum())
            counts["retained_pair_rows"] += int(keep.sum())
            counts["retained_contact_weight"] += int(weight[keep].sum())
            counts["incoming_boundary_pair_rows"] += int((qvalid & ~pvalid & positive).sum())
            counts["outgoing_boundary_pair_rows"] += int((pvalid & ~qvalid & positive).sum())
            counts["neither_endpoint_retained_pair_rows"] += int((~pvalid & ~qvalid & positive).sum())
            np.add.at(incoming_all_bodies, qi[qvalid & positive], weight[qvalid & positive])
            np.add.at(outgoing_all_bodies, pi[pvalid & positive], weight[pvalid & positive])
            rows.append(qi[keep].astype(np.int32))
            cols.append(pi[keep].astype(np.int32))
            values.append(weight[keep])
            if (batch_index + 1) % 500 == 0:
                progress(f"Scanned {batch_index+1}/{reader.num_record_batches} batches; {counts['retained_pair_rows']:,} retained pairs")
    unsigned = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n), dtype=np.int64).tocsr()
    unsigned.sum_duplicates()
    del rows, cols, values
    incoming = np.asarray(unsigned.sum(axis=1)).ravel()
    outgoing = np.asarray(unsigned.sum(axis=0)).ravel()
    nodes["retained_incoming_contacts"] = incoming
    nodes["retained_outgoing_contacts"] = outgoing
    nodes["all_source_incoming_contacts"] = incoming_all_bodies
    nodes["all_source_outgoing_contacts"] = outgoing_all_bodies
    nodes["incoming_contacts_from_excluded_bodies"] = incoming_all_bodies - incoming
    nodes["outgoing_contacts_to_excluded_bodies"] = outgoing_all_bodies - outgoing
    normalized = unsigned.astype(np.float32)
    normalized.data *= signs[normalized.indices]
    normalized.data /= np.repeat(np.maximum(incoming, 1), np.diff(normalized.indptr))
    normalized.eliminate_zeros()
    normalized.sort_indices()
    counts["retained_unique_pairs"] = unsigned.nnz
    counts["duplicate_pair_rows_aggregated"] = counts["retained_pair_rows"] - unsigned.nnz
    counts["signed_nonzero_pairs"] = normalized.nnz
    counts["disabled_output_pairs"] = unsigned.nnz - normalized.nnz
    counts["disabled_output_contact_weight"] = int(outgoing[signs == 0].sum())
    counts["self_pairs"] = int(np.count_nonzero(unsigned.diagonal()))
    counts["self_contact_weight"] = int(unsigned.diagonal().sum())
    counts["nodes_without_retained_input"] = int(np.count_nonzero(incoming == 0))
    counts["nodes_without_retained_output"] = int(np.count_nonzero(outgoing == 0))
    groups = _groups(nodes)
    observer = sparse.lil_matrix((len(groups), n), dtype=np.float32)
    for i, group in enumerate(groups):
        observer[i, group["indices"]] = 1 / len(group["indices"])
    sparse.save_npz(output / "unsigned_counts.npz", unsigned)
    sparse.save_npz(output / "W_normalized.npz", normalized)
    sparse.save_npz(output / "group_means.npz", observer.tocsr())
    feather.write_feather(pa.Table.from_pandas(nodes, preserve_index=False), output / "nodes.feather")
    np.savez_compressed(output / "arrays.npz", ids=ids, input_indices=input_indices,
                        incoming_denominator=incoming, output_sign=signs)
    np.save(output / "input_indices.npy", input_indices)
    (output / "groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    missing_cols = ["index", "bodyId", "type", "superclass", "statusLabel", "consensus_nt",
                    "predicted_nt", "ground_truth", "nt_annotation_present", "output_sign_rule",
                    "retained_outgoing_contacts", "retained_incoming_contacts", "is_visual_input"]
    nodes.loc[nodes.output_sign.eq(0), missing_cols].to_json(output / "disabled_output_nodes.json", orient="records", indent=2)
    input_cols = ["index", "bodyId", "type", "instance", "somaSide", "observer_side", "observer_side_source",
                  "statusLabel", "consensus_nt", "predicted_nt", "ground_truth", "output_sign", "visual_input_group",
                  "retained_incoming_contacts", "retained_outgoing_contacts"]
    nodes.loc[nodes.is_visual_input, input_cols].to_json(output / "visual_input_nodes.json", orient="records", indent=2)
    excluded = all_nodes.loc[~all_nodes.status.eq("Traced"), ["bodyId", "type", "superclass", "status", "statusLabel", "consensus_nt"]]
    feather.write_feather(pa.Table.from_pandas(excluded, preserve_index=False), output / "excluded_annotated_nodes.feather")
    def count_dict(series):
        return {str(k) if pd.notna(k) else "missing": int(v) for k,v in series.value_counts(dropna=False).items() if v}
    nt_summary = []
    for nt, group in nodes.groupby("consensus_nt", dropna=False):
        ix = group.index.to_numpy()
        nt_summary.append({"consensus_nt": nt if pd.notna(nt) else None, "nodes": len(group),
                           "output_sign": int(group.output_sign.iloc[0]),
                           "retained_outgoing_contacts": int(outgoing[ix].sum())})
    visual_summary = nodes.loc[nodes.is_visual_input].groupby(["type", "observer_side", "consensus_nt"], dropna=False).size().reset_index(name="n")
    manifest = {"dataset": DATASET, "built_at": datetime.now(timezone.utc).isoformat(), "sources": sources,
                "selection": {"node_rule": "status == Traced", "node_count": n,
                              "annotation_count": len(annotations), "excluded_annotation_count": len(excluded),
                              "edge_rule": "both endpoints retained and published weight >= 1",
                              "scope": "whole CNS; no visual/VNC/ROI/hop restriction",
                              "threshold_pruning": "none beyond positivity; upstream minconf-0.5 retained",
                              "self_connections": "retained", "status_label_counts": count_dict(nodes.statusLabel)},
                "matrix_orientation": "postsynaptic rows, presynaptic columns",
                "matrix_dtype": {"unsigned_counts": "int64", "W_normalized": "float32"},
                "normalization": "signed count divided by all retained incoming contact counts in that postsynaptic row; zero denominator replaced with 1",
                "nt_signs": NT_SIGNS, "nt_override": "none; consensus_nt retained unchanged",
                "unsupported_nt_policy": "preserve node and all unsigned edges; set modeled outgoing sign to zero",
                "nt_summary": nt_summary, "counts": dict(counts),
                "visual_input": {"rule": "ol_sensory and type == R1-R6; R7/R8, ambiguous R7R8 and HBeyelet retained in graph but not directly driven",
                                 "n": len(input_indices), "types_sides_nt": json.loads(visual_summary.to_json(orient="records")),
                                 "nt_disabled_inputs": int(nodes.loc[nodes.is_visual_input, "output_sign"].eq(0).sum()),
                                 "drive_assumption": "indices only; any uniform luminance-to-terminal drive is an added sensory model, not measured phototransduction"},
                "observer": {"file": "group_means.npz", "groups": len(groups), "units": "same units as underlying cell state",
                             "meaning": "overlapping cell-label population means, not spatial LFP channels",
                             "side_policy": "somaSide where known, otherwise instance _L/_R suffix; no coordinate or ROI inference",
                             "geometry": "original somaLocation retained in 8-nm dataset voxels; missing coordinates remain missing",
                             "layer_policy": "no layer assignment inferred from soma or cell type"},
                "limitations": ["Traced includes RT Hard to trace, Leaves, PRT Orphan and RT Orphan; retained per requested status rule.",
                                "Non-Traced fragments and bodies absent from annotations are outside modeled nodes; boundary contact counts recorded.",
                                "All glutamate outputs inhibitory is a simplifying assumption; receptor-specific sign is unavailable here.",
                                "All histamine outputs inhibitory simplifies R8 acetylcholine co-transmission and target-specific effects.",
                                "Synapse counts are not conductances, membrane currents, or extracellular voltages.",
                                "Uniform R1-R6 terminal stimulation does not model spectral sensitivity, adaptation, optics, or retinal geometry."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    progress(f"Saved {n:,} nodes, {unsigned.nnz:,} structural pairs, {normalized.nnz:,} modeled signed pairs")
    return manifest


def load_graph(path: Path) -> FullConnectome:
    path = Path(path)
    arrays = np.load(path / "arrays.npz", allow_pickle=False)
    return FullConnectome(arrays["ids"], feather.read_table(path / "nodes.feather").to_pandas(),
                          sparse.load_npz(path / "unsigned_counts.npz"), sparse.load_npz(path / "W_normalized.npz"),
                          arrays["input_indices"], json.loads((path / "groups.json").read_text()),
                          json.loads((path / "manifest.json").read_text()))
