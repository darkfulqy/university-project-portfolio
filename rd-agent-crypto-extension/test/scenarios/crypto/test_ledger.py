"""Tests for rdagent.scenarios.crypto.ledger (DESIGN.md section 9)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rdagent.scenarios.crypto.ledger import append_record, build_report, load_records, main, record_from_eval


def _metrics(ic_mean: float, tstat: float, cov: float = 1.0) -> dict:
    d = {
        "n_ts": 1000,
        "ic_mean": ic_mean,
        "ic_std": 0.1,
        "ic_ir": ic_mean / 0.1,
        "ic_tstat": tstat,
        "ic_pos_rate": 0.6,
        "coverage": cov,
        "rank_autocorr_1": 0.8,
        "q_spread_bp_gross": 5.0,
        "q_spread_bp_net": 2.0,
    }
    return {seg: {"5m": dict(d), "15m": dict(d)} for seg in ("train", "valid", "test")}


def _record(i: int, accepted_name: str | None, rejected: list[tuple[str, float, list[str]]]) -> dict:
    factors = []
    if accepted_name:
        factors.append(
            {
                "name": accepted_name,
                "description": "desc",
                "formulation": "f = x",
                "workspace_path": f"/tmp/ws/{accepted_name}",
                "metrics": _metrics(0.02, 4.5),
                "sign": 1,
                "max_corr": 0.1,
                "max_corr_with": None,
                "gate_passed": True,
                "gate_reasons": [],
                "accepted": True,
            }
        )
    for name, tstat, reasons in rejected:
        factors.append(
            {
                "name": name,
                "description": "desc",
                "formulation": "f = y",
                "workspace_path": f"/tmp/ws/{name}",
                "metrics": _metrics(0.001, tstat, 0.9),
                "sign": -1,
                "max_corr": 0.3,
                "max_corr_with": accepted_name,
                "gate_passed": False,
                "gate_reasons": reasons,
                "accepted": False,
            }
        )
    return {
        "ts_utc": f"2026-09-13T00:0{i}:00+00:00",
        "hypothesis": f"hypothesis {i} 中文",
        "reason": "gate outcome",
        "factors": factors,
        "composite_before": None if i == 0 else _metrics(0.02, 4.5),
        "composite_after": _metrics(0.021, 4.7),
        "accepted": [accepted_name] if accepted_name else [],
        "library_after": [n for n in ("alpha", "beta") if accepted_name and n <= accepted_name],
    }


def test_append_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "ledger.jsonl"
    rec = _record(0, "alpha", [("noise1", 1.2, ["R1 valid.15m |ic_tstat|=1.20 < 3.0"])])
    rec["extra"] = {"np_float": np.float64(1.5), "np_int": np.int64(3), "np_bool": np.bool_(True), "path": Path("/x")}
    rec["nan_value"] = float("nan")
    out = append_record(path, rec)
    assert out == path and path.exists()
    append_record(path, {"hypothesis": "second", "factors": []})  # ts_utc filled in
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "中文" in lines[0]  # ensure_ascii=False
    loaded = load_records(path)
    assert len(loaded) == 2
    assert loaded[0]["hypothesis"] == rec["hypothesis"]
    assert loaded[0]["factors"][0]["name"] == "alpha"
    assert loaded[0]["extra"]["np_float"] == 1.5 and loaded[0]["extra"]["np_int"] == 3
    assert loaded[0]["extra"]["np_bool"] is True and loaded[0]["extra"]["path"] == "/x"
    assert math.isnan(loaded[0]["nan_value"])
    assert loaded[1]["ts_utc"] and loaded[1]["hypothesis"] == "second"
    assert load_records(tmp_path / "missing.jsonl") == []
    # blank / corrupt lines are skipped
    path.write_text(lines[0] + "\n\nnot json\n" + lines[1] + "\n", encoding="utf-8")
    assert len(load_records(path)) == 2


def test_record_from_eval_shape() -> None:
    ev = {
        "factors": {
            "f1": {
                "metrics": _metrics(0.01, 3.5),
                "sign": 1,
                "max_corr": 0.2,
                "max_corr_with": None,
                "gate_passed": True,
                "gate_reasons": [],
                "accepted": True,
            }
        },
        "composite_before": None,
        "composite_after": _metrics(0.01, 3.5),
        "accepted": ["f1"],
        "library_after": ["f1"],
    }
    rec = record_from_eval(ev, "h", "r", {"f1": {"description": "d", "formulation": "x", "workspace_path": Path("/w")}})
    assert set(rec) == {
        "ts_utc", "hypothesis", "reason", "factors", "composite_before", "composite_after", "accepted", "library_after"
    }  # fmt: skip
    f = rec["factors"][0]
    assert f["name"] == "f1" and f["workspace_path"] == "/w" and f["accepted"] is True and f["description"] == "d"
    json.dumps(rec)


def test_report_on_two_record_ledger(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "ledger.jsonl"
    append_record(path, _record(0, "alpha", [("noise1", 1.2, ["R1 valid.15m |ic_tstat|=1.20 < 3.0"])]))
    append_record(
        path,
        _record(
            1,
            "beta",
            [("noise2", -2.4, ["R1 valid.15m |ic_mean|=0.0010 < 0.005"]), ("dup", 4.0, ["R3 max_corr=0.950 >= 0.7"])],
        ),
    )
    rc = main(["report", "--path", str(path), "--top", "10"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Accepted library" in out and "alpha" in out and "beta" in out
    assert "rejected" in out.lower()
    # best rejected sorted by |valid ic_tstat|: dup (4.0) first, then noise2 (-2.4), then noise1 (1.2)
    rej_section = out.split("rejected by")[1]
    assert rej_section.index("dup") < rej_section.index("noise2") < rej_section.index("noise1")
    assert "R3 max_corr=0.950 >= 0.7" in out
    assert "valid.15m.ic_tstat" in out and "test.15m.ic_mean" in out and "rank_autocorr_1" in out
    # --top limits the rejected rows
    main(["report", "--path", str(path), "--top", "1"])
    out1 = capsys.readouterr().out
    assert "dup" in out1 and "noise1" not in out1.split("rejected by")[1]
    # build_report on an empty ledger does not fail
    assert "(none)" in build_report([])


def test_report_module_cli(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_record(path, _record(0, "alpha", []))
    append_record(path, _record(1, None, [("noise", 0.5, ["R1 x"])]))
    proc = subprocess.run(
        [sys.executable, "-m", "rdagent.scenarios.crypto.ledger", "report", "--path", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "alpha" in proc.stdout and "noise" in proc.stdout
