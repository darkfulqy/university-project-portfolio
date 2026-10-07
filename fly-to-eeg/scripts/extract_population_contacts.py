#!/usr/bin/env python3
"""Extract original synaptic input rows for several observer bodies.

One streaming Arrow pass over the original partner table; no pandas. Every row
whose body_post is a configured observer is kept, including rows from excluded
bodies and rows with disabled output signs. The graph and source are unchanged.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
NT_SIGNS = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}
CODE = ["scripts/extract_population_contacts.py"]
REGRESSION_COLUMNS = ["source_row_index", "body_pre", "body_post", "x_post", "y_post", "z_post",
                      "x_post_um", "y_post_um", "z_post_um", "conf_pre", "conf_post",
                      "pre_in_graph", "pre_graph_index", "pre_model_output_sign",
                      "pre_nominal_consensus_sign", "included_in_unsigned_graph",
                      "included_in_signed_model"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_source(path):
    record = json.loads(path.with_suffix(path.suffix + ".verified.json").read_text())
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"File size changed: {path}")
    return dict(record, path=str(path.relative_to(ROOT)), sha256_recomputed_this_run=False)


def unique_lookup(table, key, keys):
    column = pc.cast(table[key], pa.int64()).combine_chunks()
    if len(pc.unique(column)) != len(column):
        raise ValueError(f"Duplicate {key} rows in lookup table")
    return column, pc.index_in(keys, value_set=column)


def annotate(sub, target, graph, ann, nts, source_name, source_sha):
    pre = sub["body_pre"].combine_chunks()
    graph_ids = pa.array(graph["ids"], pa.int64())
    node = pc.index_in(pre, value_set=graph_ids)
    in_graph = pc.is_valid(node)
    _, ann_index = unique_lookup(ann, "bodyId", pre)
    _, nt_index = unique_lookup(nts, "body", pre)
    n = sub.num_rows
    columns = {
        "pre_in_graph": in_graph,
        "post_in_graph": pa.array(np.ones(n, bool)),
        "pre_graph_index": pc.take(pa.array(graph["index"], pa.int64()), node),
        "post_graph_index": pa.array(np.full(n, target["index"], np.int64)),
    }
    for field in ["type", "instance", "status", "statusLabel"]:
        columns[f"pre_annotation_{field}"] = pc.take(ann[field].combine_chunks(), ann_index)
    for field in ["consensus_nt", "predicted_nt", "ground_truth"]:
        columns[f"pre_{field}"] = pc.take(nts[field].combine_chunks(), nt_index)
    consensus = columns["pre_consensus_nt"].to_pylist()
    columns["pre_nominal_consensus_sign"] = pa.array(
        [NT_SIGNS.get(v, 0) if isinstance(v, str) else 0 for v in consensus], pa.int8())
    columns["pre_model_output_sign"] = pc.cast(
        pc.fill_null(pc.take(pa.array(graph["output_sign"], pa.int8()), node), 0), pa.int8())
    columns["included_in_unsigned_graph"] = in_graph
    columns["included_in_signed_model"] = pc.and_(in_graph, pc.not_equal(columns["pre_model_output_sign"], 0))
    columns["contact_weight"] = pa.array(np.ones(n, np.int8))
    for axis in "xyz":
        columns[f"{axis}_post_um"] = pc.multiply(pc.cast(sub[f"{axis}_post"], pa.float64()), 0.008)
    columns["source_file_name"] = pa.array([source_name] * n, pa.string())
    columns["source_sha256"] = pa.array([source_sha] * n, pa.string())
    columns["connectome_version"] = pa.array(["male-cns:v1.0"] * n, pa.string())
    for name, values in columns.items():
        sub = sub.append_column(name, values)
    return sub


def target_checks(sub, target, row, graph):
    body_pre = sub["body_pre"].to_numpy()
    in_graph = np.asarray(sub["pre_in_graph"].to_pylist(), dtype=bool)
    expected = {int(graph["ids"][i]): int(v) for i, v in zip(row.indices, row.data)}
    pre, count = np.unique(body_pre[in_graph], return_counts=True)
    actual = {int(b): int(c) for b, c in zip(pre, count)}
    status = np.array(sub["pre_annotation_status"].to_pylist(), dtype=object)[in_graph]
    rows = sub["source_row_index"].to_numpy()
    checks = {"all_post_body_ids_match": bool(np.all(sub["body_post"].to_numpy() == target["body_id"])),
              "raw_total_matches_node_all_source_incoming": sub.num_rows == int(target["all_source_incoming_contacts"]),
              "retained_total_matches_unsigned_row": int(in_graph.sum()) == int(row.sum()),
              "every_presynaptic_pair_count_matches_unsigned_row": actual == expected,
              "unique_source_row_indices": len(np.unique(rows)) == len(rows),
              "retained_pre_status_is_Traced": bool(np.all(status == "Traced")),
              "target_status_is_Traced": target["status"] == "Traced",
              "target_type_and_instance_match_config": bool(target["type_matches"] and target["instance_matches"])}
    return checks, actual


def regression(sub, path):
    old = feather.read_table(path)
    record = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
              "rows_old": old.num_rows, "rows_new": sub.num_rows, "columns": {}}
    for name in REGRESSION_COLUMNS:
        record["columns"][name] = (name in old.column_names and old.num_rows == sub.num_rows
                                   and old[name].to_pylist() == sub[name].to_pylist())
    record["passed"] = old.num_rows == sub.num_rows and all(record["columns"].values())
    return record


def main(config_path):
    config_path = Path(config_path)
    config_bytes = config_path.read_bytes(); cfg = json.loads(config_bytes)
    frozen = {p: (ROOT / p).read_bytes() for p in CODE}
    out = ROOT / cfg["contacts_output"]
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    raw = {k: ROOT / v for k, v in cfg["raw"].items()}
    sources = {p.name: verified_source(p) for p in raw.values()}
    graph_dir = ROOT / cfg["graph"]
    nodes = feather.read_table(graph_dir / "nodes.feather",
                               columns=["index", "bodyId", "type", "instance", "status",
                                        "output_sign", "all_source_incoming_contacts"])
    arrays = np.load(graph_dir / "arrays.npz")
    graph = {"ids": nodes["bodyId"].to_numpy(), "index": nodes["index"].to_numpy().astype(np.int64),
             "output_sign": nodes["output_sign"].to_numpy()}
    if not (np.array_equal(graph["index"], np.arange(nodes.num_rows)) and
            np.array_equal(graph["ids"], arrays["ids"]) and len(np.unique(graph["ids"])) == len(graph["ids"])):
        raise ValueError("Graph node order, IDs or uniqueness check failed")
    targets = []
    for cell in cfg["cells"]:
        hit = np.flatnonzero(graph["ids"] == cell["body_id"])
        if len(hit) != 1:
            raise ValueError(f"Observer {cell['body_id']} is not a retained graph node")
        i = int(hit[0])
        targets.append({"body_id": int(cell["body_id"]), "index": i, "status": nodes["status"][i].as_py(),
                        "type": nodes["type"][i].as_py(), "instance": nodes["instance"][i].as_py(),
                        "type_matches": nodes["type"][i].as_py() == cell["cell_type"],
                        "instance_matches": nodes["instance"][i].as_py() == cell["instance"],
                        "all_source_incoming_contacts": int(nodes["all_source_incoming_contacts"][i].as_py()),
                        "output": ROOT / cell["contacts"]})
    body_ids = pa.array([t["body_id"] for t in targets], pa.int64())
    unsigned = sparse.load_npz(graph_dir / "unsigned_counts.npz").tocsr()
    source_name = raw["partners"].name; source_sha = sources[source_name]["sha256"]
    manifest = {"dataset": cfg["dataset"], "created_at": datetime.now(timezone.utc).isoformat(),
                "script": CODE[0], "config": str(config_path),
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "code_sha256": {p: hashlib.sha256(d).hexdigest() for p, d in frozen.items()},
                "sources": sources,
                "graph_sources": {name: {"path": str((graph_dir / name).relative_to(ROOT)), "sha256": sha256(graph_dir / name)}
                                  for name in ["manifest.json", "nodes.feather", "unsigned_counts.npz", "arrays.npz"]},
                "selection": "body_post in configured observers; every row of the original minconf-0.5 partner table retained",
                "graph_selection_match": "both endpoints status=Traced; all positive aggregated pair counts; no extra confidence, ROI or contact-count threshold",
                "coordinate_space": "original unmirrored Male CNS EM", "raw_coordinate_unit_nm": 8,
                "same_space_as_native_swc": True, "pandas_used": False, "targets": {}}
    with pa.memory_map(str(raw["partners"]), "r") as memory:
        reader = ipc.open_file(memory)
        pieces = []; offset = 0
        for b in range(reader.num_record_batches):
            batch = reader.get_batch(b)
            mask = pc.is_in(batch.column("body_post"), value_set=body_ids)
            positions = np.flatnonzero(mask.to_numpy(zero_copy_only=False))
            if len(positions):
                found = pa.Table.from_batches([batch.filter(mask)])
                found = found.append_column("source_row_index", pa.array(positions + offset, pa.int64()))
                found = found.append_column("source_batch_index", pa.array(np.full(len(positions), b), pa.int32()))
                pieces.append(found)
            offset += batch.num_rows
            if (b + 1) % 500 == 0:
                print(f"Scanned {b + 1}/{reader.num_record_batches} batches", flush=True)
        manifest["rows_scanned"] = offset; manifest["batches_scanned"] = reader.num_record_batches
        if not pieces:
            raise ValueError("No observer contacts in original partner table")
        found = pa.concat_tables(pieces).unify_dictionaries().combine_chunks()
        pre_ids = pc.unique(found["body_pre"])
        ann = feather.read_table(raw["annotations"], columns=["bodyId", "type", "instance", "status", "statusLabel"])
        ann = ann.filter(pc.is_in(pc.cast(ann["bodyId"], pa.int64()), value_set=pre_ids)).combine_chunks()
        nts = feather.read_table(raw["neurotransmitters"], columns=["body", "consensus_nt", "predicted_nt", "ground_truth"])
        nts = nts.filter(pc.is_in(pc.cast(nts["body"], pa.int64()), value_set=pre_ids)).combine_chunks()
        tables = {}
        for target in targets:
            sub = found.filter(pc.equal(found["body_post"], target["body_id"]))
            sub = sub.take(pc.sort_indices(sub, sort_keys=[("source_row_index", "ascending")]))
            sub = annotate(sub, target, graph, ann, nts, source_name, source_sha).combine_chunks()
            row = unsigned.getrow(target["index"])
            checks, actual = target_checks(sub, target, row, graph)
            roi = sub["primary_post"].combine_chunks()
            if pa.types.is_dictionary(roi.type):
                roi = roi.dictionary_decode()
            record = {"body_id": target["body_id"], "type": target["type"], "instance": target["instance"],
                      "graph_index": target["index"], "checks": checks,
                      "counts": {"input_contacts_all_bodies": sub.num_rows,
                                 "input_contacts_in_unsigned_graph": int(pc.sum(pc.cast(sub["pre_in_graph"], pa.int64())).as_py()),
                                 "input_contacts_from_excluded_bodies": int(sub.num_rows - pc.sum(pc.cast(sub["pre_in_graph"], pa.int64())).as_py()),
                                 "input_contacts_in_signed_model": int(pc.sum(pc.cast(sub["included_in_signed_model"], pa.int64())).as_py()),
                                 "distinct_presynaptic_bodies_all": len(pc.unique(sub["body_pre"])),
                                 "distinct_presynaptic_bodies_retained": len(actual)},
                      "confidence_range": {f: [float(pc.min(sub[f]).as_py()), float(pc.max(sub[f]).as_py())]
                                           for f in ["conf_pre", "conf_post"]},
                      "roi_counts": {str(d["values"]): int(d["counts"]) for d in pc.value_counts(roi).to_pylist()},
                      "retained_pair_counts": [{"body_pre": k, "contact_count": v} for k, v in sorted(actual.items())]}
            if cfg.get("regression") and target["body_id"] == cfg["regression"]["body_id"]:
                record["regression_to_existing_single_cell_contacts"] = regression(sub, ROOT / cfg["regression"]["v1_contacts"])
            tables[target["body_id"]] = sub
            manifest["targets"][str(target["body_id"])] = record
    passed = all(all(r["checks"].values()) and r.get("regression_to_existing_single_cell_contacts", {"passed": True})["passed"]
                 for r in manifest["targets"].values())
    manifest["all_checks_pass"] = bool(passed)
    out.mkdir(parents=True, exist_ok=False)
    for target in targets:
        table = tables[target["body_id"]]
        metadata = {b"dataset": b"male-cns:v1.0", b"observer_body_id": str(target["body_id"]).encode(),
                    b"raw_xyz_unit": b"8 nm voxel, original unmirrored Male CNS EM space",
                    b"post_xyz_um_conversion": b"raw xyz multiplied by 0.008", b"source_sha256": source_sha.encode()}
        path = target["output"]
        if path.parent != out or path.exists():
            raise ValueError(f"Contact output must be a new file inside {out}: {path}")
        feather.write_feather(table.replace_schema_metadata(metadata), path)
        manifest["targets"][str(target["body_id"])]["output"] = {
            "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest["limitations"] = ["These are observed synapse locations, not measured synaptic conductances or currents.",
                               "No synapse-to-SWC projection or artificial contacts were added.",
                               "Disabled or excluded presynaptic outputs remain explicit rows for audit.",
                               "The partner table SHA-256 was not recomputed; its size was checked against the prior verification record."]
    snapshot = out / "code_snapshot"; snapshot.mkdir()
    for p, data in frozen.items():
        (snapshot / p.replace("/", "__")).write_bytes(data)
    (snapshot / "config.json").write_bytes(config_bytes)
    (out / "contacts_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({t: {"counts": r["counts"], "checks": r["checks"]} for t, r in manifest["targets"].items()}, indent=2), flush=True)
    if not passed:
        raise AssertionError("Contact extraction gate failed; evidence retained in " + str(out))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/fly_observer_population.json"))
    main(parser.parse_args().config)
