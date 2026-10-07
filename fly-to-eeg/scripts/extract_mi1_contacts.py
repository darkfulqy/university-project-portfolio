#!/usr/bin/env python3
"""Extract actual synaptic input locations for the retained Mi1_R 17871 body.

This reads the original partner table and does not change the graph or model.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy import sparse


PROJECT = Path(__file__).resolve().parents[1]
BODY_ID = 17871
NT_SIGNS = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_source(path):
    record = json.loads(path.with_suffix(path.suffix + ".verified.json").read_text())
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"File size changed: {path}")
    return dict(record, path=str(path.resolve()), sha256_recomputed_this_run=False)


def main():
    raw = PROJECT / "data/raw/male_cns/v1.0"
    graph = PROJECT / "data/processed/male_cns_v1_0_visual_full"
    output = PROJECT / "data/processed/fly_observer_mi1"
    output.mkdir(parents=True, exist_ok=True)
    source = raw / "syn-partners-male-cns-v1.0-minconf-0.5.feather"
    ann_path = raw / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    nt_path = raw / "body-neurotransmitters-male-cns-v1.0.feather"
    sources = {p.name: verified_source(p) for p in [source, ann_path, nt_path]}
    nodes = feather.read_table(graph / "nodes.feather").to_pandas()
    target = nodes.loc[nodes.bodyId.eq(BODY_ID)].iloc[0]
    selected = []
    offset = 0
    with pa.memory_map(str(source), "r") as memory:
        reader = ipc.open_file(memory)
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            mask = pc.equal(batch.column("body_post"), BODY_ID)
            positions = np.flatnonzero(mask.to_numpy(zero_copy_only=False))
            if len(positions):
                found = pa.Table.from_batches([batch]).filter(mask)
                found = found.append_column("source_row_index", pa.array(positions + offset, type=pa.int64()))
                found = found.append_column("source_batch_index", pa.array(np.full(len(positions), batch_index), type=pa.int32()))
                selected.append(found)
            offset += batch.num_rows
            if (batch_index + 1) % 1000 == 0:
                print(f"Scanned {batch_index+1}/{reader.num_record_batches} batches", flush=True)
    if not selected:
        raise ValueError("Target has no contacts in original partner table")
    contacts = pa.concat_tables(selected).to_pandas()
    pre_ids = pa.array(contacts.body_pre.unique(), type=pa.int64())
    annotations = feather.read_table(ann_path, columns=["bodyId", "type", "instance", "status", "statusLabel"])
    annotations = annotations.filter(pc.is_in(annotations["bodyId"], value_set=pre_ids)).to_pandas().set_index("bodyId")
    nts = feather.read_table(nt_path, columns=["body", "consensus_nt", "predicted_nt", "ground_truth"])
    nts = nts.filter(pc.is_in(nts["body"], value_set=pre_ids)).to_pandas().set_index("body")
    node_index = nodes.set_index("bodyId")
    contacts["pre_in_graph"] = contacts.body_pre.isin(node_index.index)
    contacts["post_in_graph"] = True
    contacts["pre_graph_index"] = contacts.body_pre.map(node_index["index"]).astype("Int64")
    contacts["post_graph_index"] = int(target["index"])
    for field in ["type", "instance", "status", "statusLabel"]:
        contacts[f"pre_annotation_{field}"] = contacts.body_pre.map(annotations[field])
    for field in ["consensus_nt", "predicted_nt", "ground_truth"]:
        contacts[f"pre_{field}"] = contacts.body_pre.map(nts[field])
    contacts["pre_nominal_consensus_sign"] = contacts.pre_consensus_nt.map(NT_SIGNS).fillna(0).astype(np.int8)
    contacts["pre_model_output_sign"] = contacts.body_pre.map(node_index.output_sign).fillna(0).astype(np.int8)
    contacts["included_in_unsigned_graph"] = contacts.pre_in_graph
    contacts["included_in_signed_model"] = contacts.pre_in_graph & contacts.pre_model_output_sign.ne(0)
    contacts["contact_weight"] = np.int8(1)
    for axis in "xyz":
        contacts[f"{axis}_post_um"] = contacts[f"{axis}_post"].astype(np.float64) * 0.008
    contacts["source_file_name"] = source.name
    contacts["source_sha256"] = sources[source.name]["sha256"]
    contacts["connectome_version"] = "male-cns:v1.0"
    contacts = contacts.sort_values("source_row_index").reset_index(drop=True)
    unsigned = sparse.load_npz(graph / "unsigned_counts.npz")
    row = unsigned.getrow(int(target["index"]))
    expected = {int(nodes.iloc[i].bodyId): int(value) for i, value in zip(row.indices, row.data)}
    actual = {int(body): int(count) for body, count in contacts.loc[contacts.pre_in_graph].groupby("body_pre").size().items()}
    checks = {"all_post_body_ids_match": bool(contacts.body_post.eq(BODY_ID).all()),
              "raw_total_matches_node_all_source_incoming": len(contacts) == int(target.all_source_incoming_contacts),
              "retained_total_matches_unsigned_row": int(contacts.pre_in_graph.sum()) == int(row.sum()),
              "every_presynaptic_pair_count_matches_unsigned_row": actual == expected,
              "unique_source_row_indices": bool(contacts.source_row_index.is_unique),
              "retained_pre_status_is_Traced": bool(contacts.loc[contacts.pre_in_graph, "pre_annotation_status"].eq("Traced").all())}
    if not all(checks.values()):
        raise ValueError(json.dumps(checks))
    table = pa.Table.from_pandas(contacts, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update({b"dataset": b"male-cns:v1.0", b"raw_xyz_unit": b"8 nm voxel, original unmirrored Male CNS EM space",
                     b"post_xyz_um_conversion": b"raw xyz multiplied by 0.008", b"source_sha256": sources[source.name]["sha256"].encode()})
    feather.write_feather(table.replace_schema_metadata(metadata), output / "contacts.feather")
    manifest = {"dataset": "male-cns:v1.0", "created_at": datetime.now(timezone.utc).isoformat(),
                "script": str(Path(__file__).resolve()), "target_body_id": BODY_ID,
                "target_type": target.type, "target_instance": target.instance, "target_graph_index": int(target["index"]),
                "sources": sources,
                "graph_sources": {name: {"path": str((graph / name).resolve()), "sha256": sha256(graph / name)}
                                  for name in ["manifest.json", "nodes.feather", "unsigned_counts.npz"]},
                "selection": "body_post == 17871; retain every row in the original minconf-0.5 partner table",
                "graph_selection_match": "both endpoints have status=Traced; all positive aggregated pair counts; no extra confidence, ROI, or contact-count threshold",
                "coordinate_space": "original unmirrored Male CNS EM", "raw_coordinate_unit_nm": 8,
                "coordinate_source": "https://male-cns.janelia.org/download/", "same_space_as_native_swc": True,
                "counts": {"source_rows_scanned": offset, "target_input_contacts_all_bodies": len(contacts),
                           "target_input_contacts_in_unsigned_graph": int(contacts.pre_in_graph.sum()),
                           "target_input_contacts_from_excluded_bodies": int((~contacts.pre_in_graph).sum()),
                           "target_input_contacts_in_signed_model": int(contacts.included_in_signed_model.sum()),
                           "distinct_presynaptic_bodies_all": int(contacts.body_pre.nunique()),
                           "distinct_presynaptic_bodies_retained": len(actual)},
                "confidence_range": {field: [float(contacts[field].min()), float(contacts[field].max())]
                                     for field in ["conf_pre", "conf_post"]},
                "roi_counts": {str(k): int(v) for k, v in contacts.primary_post.value_counts().items() if v},
                "retained_pair_counts": [{"body_pre": key, "contact_count": value} for key, value in sorted(actual.items())],
                "checks": checks, "all_checks_pass": all(checks.values()),
                "fields": [{"name": field.name, "type": str(field.type)} for field in table.schema],
                "output": {"path": str((output / "contacts.feather").resolve()),
                           "bytes": (output / "contacts.feather").stat().st_size,
                           "sha256": sha256(output / "contacts.feather")},
                "limitations": ["These are observed synapse locations, not measured synaptic conductances or currents.",
                                "No synapse-to-SWC projection or artificial contacts were added.",
                                "Disabled or excluded presynaptic outputs remain explicit rows for audit."]}
    (output / "contacts_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"counts": manifest["counts"], "checks": checks, "output": manifest["output"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
