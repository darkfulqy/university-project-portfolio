"""Runner / loop-app tests for the crypto 5m scenario (DESIGN.md section 11, runner part).

Feedback (``CryptoFactorExperiment2Feedback``) tests are owned by the scenario agent and live elsewhere.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import FactorEmptyError
from rdagent.core.proposal import Hypothesis
from rdagent.core.utils import import_class
from rdagent.scenarios.crypto import local_embedding
from rdagent.scenarios.crypto.conf import CRYPTO_EVAL_SETTINGS

import rdagent.scenarios.crypto.runner as runner_mod
from rdagent.scenarios.crypto.runner import (
    N_LIBRARY_KEY,
    CryptoRankICRunner,
    baseline_result,
    result_series,
    standard_result_keys,
)

N_INST = 35
SEG = {
    "train_start": "2026-03-01",
    "train_end": "2026-03-03 23:59:00",
    "valid_start": "2026-03-04",
    "valid_end": "2026-03-05 23:59:00",
    "test_start": "2026-03-06",
    "test_end": "2026-03-07 23:59:00",
    "gate_max_abs_ic": 1.0,  # the synthetic planted factor has |IC| ~ 0.9; R0's real-data cap is tested separately
}

LEAK_CODE = """import pandas as pd


def calculate_leak_next():
    df = pd.read_hdf("crypto_5m.h5", key="data")
    close = df["$close"]
    factor = close.groupby(level="instrument").shift(-1) / close - 1.0
    factor.to_frame("leak_next").astype("float64").to_hdf("result.h5", key="data")


if __name__ == "__main__":
    calculate_leak_next()
"""

MOM_CODE = """import pandas as pd


def calculate_mom_3():
    df = pd.read_hdf("crypto_5m.h5", key="data")
    close = df["$close"]
    factor = close / close.groupby(level="instrument").shift(3) - 1.0
    factor.to_frame("mom_3").astype("float64").to_hdf("result.h5", key="data")


if __name__ == "__main__":
    calculate_mom_3()
"""

ZSCORE_CODE = """import pandas as pd


def calculate_zscore_full():
    df = pd.read_hdf("crypto_5m.h5", key="data")
    close = df["$close"]
    mom = close / close.groupby(level="instrument").shift(3) - 1.0
    g = mom.groupby(level="instrument")
    factor = (mom - g.transform("mean")) / g.transform("std")  # full-sample normalisation = look-ahead
    factor.to_frame("zscore_full").astype("float64").to_hdf("result.h5", key="data")


if __name__ == "__main__":
    calculate_zscore_full()
"""

BROKEN_CODE = """raise SystemExit(3)
"""


# --------------------------------------------------------------------------- fixtures
def _make_panel() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    ts = pd.date_range("2026-03-01", "2026-03-07 23:55:00", freq="5min")
    inst = [f"C{i:02d}USDT" for i in range(N_INST)]
    logp = np.cumsum(rng.normal(0, 0.002, size=(len(ts), N_INST)), axis=0) + 5.0
    close_wide = pd.DataFrame(np.exp(logp), index=ts, columns=inst)
    fwd15 = close_wide.shift(-3) / close_wide - 1.0
    panel = close_wide.stack()
    panel.index.names = ["datetime", "instrument"]
    panel = panel.sort_index()
    df = pd.DataFrame({"$close": panel.values}, index=panel.index)
    for col in ["$open", "$high", "$low"]:
        df[col] = df["$close"]
    for col in ["$volume", "$amount", "$trade_count", "$taker_buy_volume", "$taker_buy_amount"]:
        df[col] = 1.0
    fwd = fwd15.stack(future_stack=True)
    fwd.index.names = ["datetime", "instrument"]
    return df, fwd.reindex(df.index)


def _frame_from_code(df: pd.DataFrame, name: str) -> pd.Series:
    """What the corresponding factor.py above produces on the full panel (same formulas)."""
    close = df["$close"]
    g = close.groupby(level="instrument")
    if name == "leak_next":
        return (g.shift(-1) / close - 1.0).astype(float)
    if name == "mom_3":
        return (close / g.shift(3) - 1.0).astype(float)
    if name == "zscore_full":
        mom = close / g.shift(3) - 1.0
        gm = mom.groupby(level="instrument")
        return ((mom - gm.transform("mean")) / gm.transform("std")).astype(float)
    if name == "broken":
        return (close / g.shift(1) - 1.0).astype(float)
    raise KeyError(name)


@pytest.fixture(scope="module")
def panel():
    return _make_panel()


@pytest.fixture(scope="module")
def frames(panel):
    df, fwd = panel
    rng = np.random.default_rng(11)
    planted = fwd + rng.normal(0, fwd.std() * 0.5, size=len(fwd))
    noise = pd.Series(rng.normal(size=len(df)), index=df.index)
    return {
        "planted": planted.astype(float),
        "noise": noise,
        "planted_copy": planted.astype(float),
        "extra_junk": noise * 2,
        **{n: _frame_from_code(df, n) for n in ("leak_next", "mom_3", "zscore_full", "broken")},
    }


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, panel, frames):
    """Temp full folder with crypto_5m.h5, patched settings, patched process_factor_data."""
    df, _ = panel
    full = tmp_path / "full"
    full.mkdir()
    df.to_hdf(full / "crypto_5m.h5", key="data", mode="w", format="fixed")
    (full / "README.md").write_text("synthetic panel\n")
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder", str(full))
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder_debug", str(full))
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "python_bin", sys.executable)
    monkeypatch.setattr(RD_AGENT_SETTINGS, "workspace_path", tmp_path / "ws")
    for k, v in SEG.items():
        monkeypatch.setattr(CRYPTO_EVAL_SETTINGS, k, v)
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(CRYPTO_EVAL_SETTINGS, "ledger_path", str(ledger))

    calls: list[list[str]] = []

    def fake_process_factor_data(exp_or_list):
        exp = exp_or_list if not isinstance(exp_or_list, list) else exp_or_list[0]
        names = [t.factor_name for t in exp.sub_tasks]
        calls.append(names)
        cols = {n: frames[n] for n in names if n in frames}
        if getattr(exp, "_emit_junk", False):
            cols["extra_junk"] = frames["extra_junk"]
        if not cols:
            raise FactorEmptyError("no factor data")
        df = pd.DataFrame(cols)
        transform = getattr(exp, "_transform", None)
        return transform(df) if transform is not None else df

    monkeypatch.setattr(runner_mod, "process_factor_data", fake_process_factor_data)
    return {"full": full, "ledger": ledger, "calls": calls}


def _exp(names: list[str], based=None, hypothesis: bool = True):
    from rdagent.scenarios.crypto.experiment import CryptoFactorExperiment

    tasks = [FactorTask(factor_name=n, factor_description=f"desc {n}", factor_formulation=f"f_{n}") for n in names]
    hyp = Hypothesis(f"hyp {names}", "reason text", "cr", "co", "cj", "ck") if hypothesis else None
    return CryptoFactorExperiment(sub_tasks=tasks, based_experiments=based or [], hypothesis=hyp)


def _runner() -> CryptoRankICRunner:
    return CryptoRankICRunner(scen=None)


def _with_code(exp, codes: dict[str, str]):
    """Attach a real FactorFBWorkspace (with factor.py) to every sub-task named in ``codes``."""
    for i, task in enumerate(exp.sub_tasks):
        if task.factor_name in codes:
            ws = FactorFBWorkspace(target_task=task)
            ws.inject_files(**{"factor.py": codes[task.factor_name]})
            exp.sub_workspace_list[i] = ws
    return exp


# --------------------------------------------------------------------------- pure helpers
def test_result_series_and_baseline_key_parity():
    cfg = CRYPTO_EVAL_SETTINGS.to_eval_config()
    base = baseline_result(cfg)
    keys = standard_result_keys(cfg)
    assert list(base.index) == keys
    assert base[N_LIBRARY_KEY] == 0 and base.drop(N_LIBRARY_KEY).isna().all()
    assert "valid.15m.ic_mean" in keys and "train.5m.rank_autocorr_1" in keys and "test.15m.q_spread_bp_net" in keys

    ev = {
        "composite_after": {"valid": {"15m": {"ic_mean": 0.02, "ic_ir": 0.5}}, "train": {"5m": {"n_ts": 100}}},
        "library_after": ["a", "b"],
    }
    res = result_series(ev, cfg)
    assert list(res.index) == keys
    assert res.dtype == float
    assert res["valid.15m.ic_mean"] == pytest.approx(0.02)
    assert res["train.5m.n_ts"] == 100.0
    assert np.isnan(res["test.15m.ic_mean"])
    assert res[N_LIBRARY_KEY] == 2.0

    empty = result_series({"composite_after": None, "library_after": []}, cfg)
    assert empty.drop(N_LIBRARY_KEY).isna().all() and empty[N_LIBRARY_KEY] == 0.0


# --------------------------------------------------------------------------- develop()
def test_develop_first_round(env):
    base = _exp([], hypothesis=False)
    exp = _exp(["planted", "noise"], based=[base])
    exp._emit_junk = True
    r = _runner()
    out = r.develop(exp)
    assert out is exp

    # baseline assignment on based_experiments[-1]
    assert isinstance(base.result, pd.Series)
    assert list(base.result.index) == standard_result_keys()
    assert base.result[N_LIBRARY_KEY] == 0 and base.result.drop(N_LIBRARY_KEY).isna().all()
    assert base.crypto_eval["accepted"] == [] and base.crypto_eval["library_after"] == []
    assert base.crypto_accepted == []

    # experiment outputs
    ev = exp.crypto_eval
    assert isinstance(ev, dict)
    expected_keys = (
        "cfg",
        "library_before",
        "library_after",
        "composite_before",
        "composite_after",
        "factors",
        "accepted",
        "summary_text",
    )
    for k in expected_keys:
        assert k in ev, k
    assert set(ev["factors"]) == {"planted", "noise"}, "only sub-task columns must be evaluated"
    assert "extra_junk" not in ev["factors"]
    assert ev["library_before"] == []
    assert ev["factors"]["planted"]["accepted"] is True and ev["factors"]["planted"]["gate_passed"] is True
    assert ev["factors"]["noise"]["accepted"] is False and ev["factors"]["noise"]["gate_reasons"]
    assert exp.crypto_accepted == ["planted"] and ev["library_after"] == ["planted"]
    assert exp.stdout == ev["summary_text"] and exp.stdout

    assert isinstance(exp.result, pd.Series)
    assert list(exp.result.index) == standard_result_keys()
    assert exp.result[N_LIBRARY_KEY] == 1.0
    assert exp.result["valid.15m.ic_mean"] > 0.3
    assert exp.result["valid.15m.ic_mean"] == pytest.approx(ev["composite_after"]["valid"]["15m"]["ic_mean"])

    # workspace files
    ws = Path(exp.experiment_workspace.workspace_path)
    ev_json = json.loads((ws / "crypto_eval.json").read_text())
    assert ev_json["accepted"] == ["planted"]
    csv = pd.read_csv(ws / "factor_metrics.csv")
    assert set(csv["factor"]) == {"planted", "noise"}
    assert len(csv) == 2 * 3 * 2  # factor x segment x horizon
    assert {"ic_mean", "ic_tstat", "rank_autocorr_1", "q_spread_bp_net", "accepted"} <= set(csv.columns)

    # ledger line
    lines = env["ledger"].read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    record_keys = (
        "ts_utc",
        "hypothesis",
        "reason",
        "factors",
        "composite_before",
        "composite_after",
        "accepted",
        "library_after",
    )
    for k in record_keys:
        assert k in rec, k
    assert rec["hypothesis"].startswith("hyp") and rec["reason"] == "reason text"
    assert rec["accepted"] == ["planted"] and rec["library_after"] == ["planted"]
    names = {f["name"]: f for f in rec["factors"]}
    assert names["planted"]["accepted"] is True and names["planted"]["description"] == "desc planted"
    assert names["noise"]["gate_reasons"]


def test_develop_with_library_rejects_duplicate_and_caches(env):
    r = _runner()
    base0 = _exp([], hypothesis=False)
    exp1 = _exp(["planted"], based=[base0])
    r.develop(exp1)
    assert exp1.crypto_accepted == ["planted"]

    exp2 = _exp(["planted_copy"], based=[base0, exp1])
    r.develop(exp2)  # exp1.result is set, so no new baseline assignment
    ev = ev2 = exp2.crypto_eval
    assert ev["library_before"] == ["planted"]
    assert exp2.crypto_accepted == []
    assert ev["factors"]["planted_copy"]["accepted"] is False
    assert ev["factors"]["planted_copy"]["max_corr"] > 0.99
    assert any(reason.startswith("R3") for reason in ev["factors"]["planted_copy"]["gate_reasons"])
    assert exp2.result[N_LIBRARY_KEY] == 1.0 and ev2["library_after"] == ["planted"]

    # library cache: a third experiment must not re-execute exp1
    n_calls = len(env["calls"])
    exp3 = _exp(["noise"], based=[base0, exp1])
    r.develop(exp3)
    new_calls = env["calls"][n_calls:]
    assert new_calls == [["noise"]], new_calls
    assert exp3.crypto_eval["library_before"] == ["planted"]

    assert len(env["ledger"].read_text().strip().splitlines()) == 3


def test_develop_raises_factor_empty(env):
    base = _exp([], hypothesis=False)
    exp = _exp(["unknown_factor"], based=[base])
    with pytest.raises(FactorEmptyError):
        _runner().develop(exp)


def test_develop_ledger_failure_does_not_raise(env, monkeypatch):
    monkeypatch.setattr(CRYPTO_EVAL_SETTINGS, "ledger_path", str(env["full"] / "crypto_5m.h5" / "impossible.jsonl"))
    exp = _exp(["planted"], based=[_exp([], hypothesis=False)])
    _runner().develop(exp)
    assert exp.crypto_accepted == ["planted"]


# --------------------------------------------------------------------------- look-ahead re-run (gate R0)
def test_lookahead_check_flags_future_and_full_sample_code(env):
    exp = _exp(["leak_next", "mom_3", "zscore_full"], based=[_exp([], hypothesis=False)])
    _with_code(exp, {"leak_next": LEAK_CODE, "mom_3": MOM_CODE, "zscore_full": ZSCORE_CODE})
    r = _runner()
    new_df = r.new_factor_frame(exp)
    flagged = r.lookahead_check(exp, new_df)
    assert set(flagged) == {"leak_next", "zscore_full"}
    assert flagged["leak_next"].startswith("R0 look-ahead: values change when future rows are removed")
    assert "values differ" in flagged["zscore_full"]

    out = r.develop(exp)
    ev = out.crypto_eval
    assert list(ev["factors"]) == ["leak_next", "mom_3", "zscore_full"]  # original column order kept
    for name in ("leak_next", "zscore_full"):
        info = ev["factors"][name]
        assert info["accepted"] is False and info["gate_passed"] is False
        assert info["gate_reasons"] == [flagged[name]]
        assert info["metrics"] == {} and info["sign"] == 1 and info["max_corr"] == 0.0
    honest = ev["factors"]["mom_3"]
    assert honest["metrics"]["valid"]["15m"]["n_ts"] > 100
    assert not any(reason.startswith("R0 look-ahead") for reason in honest["gate_reasons"])
    assert exp.crypto_accepted == [] and ev["library_after"] == []
    assert "leak_next" in ev["summary_text"] and "R0 look-ahead" in ev["summary_text"]
    # downstream shapes: workspace files + ledger carry the R0 entries
    ws = Path(exp.experiment_workspace.workspace_path)
    ev_json = json.loads((ws / "crypto_eval.json").read_text())
    assert ev_json["factors"]["leak_next"]["gate_reasons"][0].startswith("R0")
    rec = json.loads(env["ledger"].read_text().strip().splitlines()[-1])
    names = {f["name"]: f for f in rec["factors"]}
    assert names["leak_next"]["accepted"] is False and names["leak_next"]["gate_reasons"][0].startswith("R0")


def test_lookahead_check_fails_closed_on_broken_code_and_skips_without_code(env):
    exp = _exp(["broken", "planted"], based=[_exp([], hypothesis=False)])
    _with_code(exp, {"broken": BROKEN_CODE})  # "planted" keeps no workspace -> no code -> skipped
    r = _runner()
    flagged = r.lookahead_check(exp, r.new_factor_frame(exp))
    assert set(flagged) == {"broken"} and "failed on the truncated panel" in flagged["broken"]
    r.develop(exp)
    assert exp.crypto_accepted == ["planted"]
    assert exp.crypto_eval["factors"]["broken"]["gate_reasons"][0].startswith("R0 look-ahead check")


# --------------------------------------------------------------------------- malformed factor output
def test_develop_dedups_duplicate_rows(env):
    exp = _exp(["planted"], based=[_exp([], hypothesis=False)])
    exp._transform = lambda df: pd.concat([df, df.iloc[:50]])
    _runner().develop(exp)
    assert exp.crypto_accepted == ["planted"]


def test_develop_malformed_output_raises_factor_empty(env, monkeypatch):
    def boom(*a, **k):
        raise ValueError("cannot handle a non-unique multi-index!")

    monkeypatch.setattr(runner_mod, "evaluate_experiment", boom)
    exp = _exp(["planted"], based=[_exp([], hypothesis=False)])
    with pytest.raises(FactorEmptyError, match="non-unique"):
        _runner().develop(exp)


def test_develop_normalises_tz_aware_index(env):
    def to_utc(df):
        dt = df.index.get_level_values("datetime").tz_localize("UTC")
        return df.set_axis(pd.MultiIndex.from_arrays([dt, df.index.get_level_values("instrument")]))

    exp = _exp(["planted"], based=[_exp([], hypothesis=False)])
    exp._transform = to_utc
    _runner().develop(exp)
    assert exp.crypto_accepted == ["planted"]
    assert exp.crypto_eval["factors"]["planted"]["metrics"]["valid"]["15m"]["coverage"] > 0.9


def test_develop_off_grid_index_raises_factor_empty(env):
    def shift_one_minute(df):
        dt = df.index.get_level_values("datetime") + pd.Timedelta(minutes=1)
        return df.set_axis(
            pd.MultiIndex.from_arrays([dt, df.index.get_level_values("instrument")], names=["datetime", "instrument"])
        )

    exp = _exp(["planted"], based=[_exp([], hypothesis=False)])
    exp._transform = shift_one_minute
    with pytest.raises(FactorEmptyError, match="panel grid"):
        _runner().develop(exp)


def test_factor_empty_message_names_produced_columns(env):
    exp = _exp(["planted"], based=[_exp([], hypothesis=False)])
    exp._transform = lambda df: df.rename(columns={"planted": "other_name"})
    with pytest.raises(FactorEmptyError) as excinfo:
        _runner().develop(exp)
    assert "other_name" in str(excinfo.value) and "planted" in str(excinfo.value)


# --------------------------------------------------------------------------- session pickling
def test_runner_pickle_drops_data_caches(env):
    r = _runner()
    base = _exp([], hypothesis=False)
    exp = _exp(["planted"], based=[base])
    r.develop(exp)
    exp2 = _exp(["noise"], based=[base, exp])
    r.develop(exp2)
    assert r._close_cache is not None and len(r._library_cache) == 1
    blob = pickle.dumps(r)
    assert len(blob) < 20_000, len(blob)
    restored = pickle.loads(blob)
    assert restored._close_cache is None and restored._library_cache == {}
    assert isinstance(restored, CryptoRankICRunner)
    # the caches are rebuilt transparently
    exp3 = _exp(["noise"], based=[base, exp])
    restored.develop(exp3)
    assert exp3.crypto_eval["library_before"] == ["planted"]


# --------------------------------------------------------------------------- loop app wiring
def test_prop_setting_class_paths_import():
    from rdagent.app.crypto_rd_loop.conf import CryptoFactorPropSetting

    setting = CryptoFactorPropSetting()
    assert setting.model_config["env_prefix"] == "CRYPTO_LOOP_"
    for field in ("scen", "hypothesis_gen", "hypothesis2experiment", "coder", "runner", "summarizer"):
        path = getattr(setting, field)
        try:
            cls = import_class(path)
        except (ModuleNotFoundError, ImportError, AttributeError) as e:
            pytest.fail(f"{field}={path} is not importable (other agents' module missing?): {e!r}")
        assert isinstance(cls, type), path
    assert setting.runner == "rdagent.scenarios.crypto.runner.CryptoRankICRunner"
    assert setting.coder == "rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"


def test_loop_class_and_cli_registration():
    import rdagent.app.crypto_rd_loop.factor as crypto_factor_mod
    from rdagent.app.crypto_rd_loop.factor import CryptoFactorRDLoop, main
    from rdagent.app.qlib_rd_loop.factor import FactorRDLoop

    assert issubclass(CryptoFactorRDLoop, FactorRDLoop)
    assert callable(main)

    # Source-level check: importing rdagent.app.cli pulls in the whole data-science stack (~90 s), so read the file.
    cli_src = (Path(crypto_factor_mod.__file__).resolve().parents[1] / "cli.py").read_text()
    assert "from rdagent.app.crypto_rd_loop.factor import main as crypto_factor" in cli_src
    assert '@app.command(name="crypto_factor")' in cli_src
    call = "crypto_factor(path=path, step_n=step_n, loop_n=loop_n, all_duration=all_duration, checkout=checkout)"
    assert call in cli_src


def test_local_embedding_shape_and_install(monkeypatch):
    assert local_embedding.local_embedding_distance([], ["a"]) == [[]]
    assert local_embedding.local_embedding_distance(["a"], []) == [[]]
    sim = local_embedding.local_embedding_distance(
        ["rolling mean of volume", "taker buy ratio"], ["volume rolling mean", "x"]
    )
    assert len(sim) == 2 and len(sim[0]) == 2
    assert all(isinstance(v, float) for row in sim for v in row)
    assert sim[0][0] > sim[0][1] and sim[0][0] > sim[1][0]
    assert local_embedding.local_embedding_distance(["abc def"], ["abc def"])[0][0] == pytest.approx(1.0)
    assert local_embedding.local_embedding_distance([""], ["abc"])[0][0] == 0.0

    import rdagent.components.coder.CoSTEER.knowledge_management as km
    import rdagent.oai.llm_utils as llm_utils

    orig = (llm_utils.calculate_embedding_distance_between_str_list, km.calculate_embedding_distance_between_str_list)
    try:
        patched = local_embedding.install()
        assert set(patched) == {"rdagent.oai.llm_utils", "rdagent.components.coder.CoSTEER.knowledge_management"}
        assert llm_utils.calculate_embedding_distance_between_str_list is local_embedding.local_embedding_distance
        assert km.calculate_embedding_distance_between_str_list is local_embedding.local_embedding_distance
        assert local_embedding.install() == patched  # idempotent
    finally:
        llm_utils.calculate_embedding_distance_between_str_list = orig[0]
        km.calculate_embedding_distance_between_str_list = orig[1]
