"""JSONL ledger of crypto factor evaluations plus a ``report`` subcommand (DESIGN.md section 9).

Usage::

    python -m rdagent.scenarios.crypto.ledger report --path crypto_factor_ledger.jsonl [--top 10] [--horizon 15m]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

RECORD_KEYS = (
    "ts_utc",
    "hypothesis",
    "reason",
    "factors",
    "composite_before",
    "composite_after",
    "accepted",
    "library_after",
)


def _json_default(obj: Any) -> Any:
    """JSON fallback: numpy scalars/arrays -> python numbers/lists, everything else -> ``str``."""
    try:
        import numpy as np  # local import: keep the module importable without numpy

        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except (TypeError, ValueError):
            pass
    return str(obj)


def append_record(path: str | Path, record: dict) -> Path:
    """Append ``record`` as one JSON line to ``path`` (parent dirs are created).

    ``ts_utc`` is filled in when missing.  Numpy scalars become python numbers, other unknown objects are serialised
    with ``str`` (``default=str`` semantics) and non-ASCII is kept as-is (``ensure_ascii=False``).  Returns the path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(record)
    rec.setdefault("ts_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    line = json.dumps(rec, ensure_ascii=False, default=_json_default, allow_nan=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return p


def load_records(path: str | Path) -> list[dict]:
    """Load every record of the JSONL ledger (missing file -> empty list; blank/corrupt lines are skipped)."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def record_from_eval(
    ev: dict,
    hypothesis: str = "",
    reason: str = "",
    factor_info: dict[str, dict] | None = None,
) -> dict:
    """Build a ledger record from an ``evaluate_experiment`` dict.

    ``factor_info`` maps factor name -> ``{"description", "formulation", "workspace_path"}`` (all optional).
    """
    info = factor_info or {}
    factors = []
    for name, f in (ev.get("factors") or {}).items():
        extra = info.get(name, {})
        factors.append(
            {
                "name": name,
                "description": extra.get("description", ""),
                "formulation": extra.get("formulation", ""),
                "workspace_path": str(extra.get("workspace_path", "")),
                "metrics": f.get("metrics"),
                "sign": f.get("sign"),
                "max_corr": f.get("max_corr"),
                "max_corr_with": f.get("max_corr_with"),
                "gate_passed": f.get("gate_passed"),
                "gate_reasons": list(f.get("gate_reasons") or []),
                "accepted": f.get("accepted"),
            }
        )
    return {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hypothesis": hypothesis,
        "reason": reason,
        "factors": factors,
        "composite_before": ev.get("composite_before"),
        "composite_after": ev.get("composite_after"),
        "accepted": list(ev.get("accepted") or []),
        "library_after": list(ev.get("library_after") or []),
    }


# --------------------------------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------------------------------


def _metric(f: dict, segment: str, horizon: str, key: str) -> float:
    try:
        v = f["metrics"][segment][horizon][key]
    except (KeyError, TypeError):
        return float("nan")
    try:
        return float("nan") if v is None else float(v)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(v: Any, nd: int = 4) -> str:
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(fv):
        return "nan"
    return f"{fv:.{nd}f}"


def format_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a plain-text table (no third-party dependencies)."""
    rows = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    sep = "-+-".join("-" * w for w in widths)
    lines = [" | ".join(h.ljust(w) for h, w in zip(headers, widths)), sep]
    for r in rows:
        lines.append(" | ".join(c.ljust(w) for c, w in zip(r, widths)))
    if not rows:
        lines.append("(none)")
    return "\n".join(lines)


def build_report(records: list[dict], top: int = 10, horizon: str = "15m") -> str:
    """Accepted-library table plus the ``top`` best rejected factors by |valid ic_tstat| with their reasons."""
    accepted: dict[str, tuple[int, str, dict]] = {}
    rejected: list[tuple[int, str, dict]] = []
    for i, rec in enumerate(records):
        ts = str(rec.get("ts_utc", ""))
        for f in rec.get("factors") or []:
            name = str(f.get("name", ""))
            if f.get("accepted"):
                accepted[name] = (i, ts, f)  # latest acceptance wins
            else:
                rejected.append((i, ts, f))
    library_after = None
    for rec in reversed(records):
        if rec.get("library_after") is not None:
            library_after = list(rec.get("library_after") or [])
            break

    acc_rows = []
    for name, (i, ts, f) in accepted.items():
        acc_rows.append(
            [
                name,
                i,
                _fmt(_metric(f, "valid", horizon, "ic_mean")),
                _fmt(_metric(f, "valid", horizon, "ic_ir"), 3),
                _fmt(_metric(f, "valid", horizon, "ic_tstat"), 2),
                _fmt(_metric(f, "valid", horizon, "rank_autocorr_1"), 3),
                _fmt(_metric(f, "test", horizon, "ic_mean")),
                "yes" if library_after is None or name in library_after else "no",
            ]
        )
    acc_headers = [
        "name",
        "round",
        f"valid.{horizon}.ic_mean",
        f"valid.{horizon}.ic_ir",
        f"valid.{horizon}.ic_tstat",
        "rank_autocorr_1",
        f"test.{horizon}.ic_mean",
        "in_library",
    ]

    def _abs_tstat(item: tuple[int, str, dict]) -> float:
        v = _metric(item[2], "valid", horizon, "ic_tstat")
        return -1.0 if math.isnan(v) else abs(v)

    rejected.sort(key=_abs_tstat, reverse=True)
    rej_rows = []
    for i, ts, f in rejected[: max(int(top), 0)]:
        rej_rows.append(
            [
                str(f.get("name", "")),
                i,
                _fmt(_metric(f, "valid", horizon, "ic_mean")),
                _fmt(_metric(f, "valid", horizon, "ic_tstat"), 2),
                _fmt(_metric(f, "valid", horizon, "coverage"), 3),
                _fmt(f.get("max_corr"), 3),
                "; ".join(str(r) for r in (f.get("gate_reasons") or [])),
            ]
        )
    rej_headers = [
        "name",
        "round",
        f"valid.{horizon}.ic_mean",
        f"valid.{horizon}.ic_tstat",
        "coverage",
        "max_corr",
        "reasons",
    ]

    lines = [
        f"Ledger records: {len(records)}; accepted factors: {len(accepted)}; rejected candidates: {len(rejected)}",
        f"Current library: {library_after if library_after is not None else '(unknown)'}",
        "",
        "Accepted library",
        format_table(acc_headers, acc_rows),
        "",
        f"Best {top} rejected by |valid {horizon} ic_tstat|",
        format_table(rej_headers, rej_rows),
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``report --path X [--top 10] [--horizon 15m]``."""
    parser = argparse.ArgumentParser(prog="python -m rdagent.scenarios.crypto.ledger")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="print the accepted library and the best rejected factors")
    rep.add_argument("--path", required=True, help="path of the JSONL ledger")
    rep.add_argument("--top", type=int, default=10, help="number of rejected factors to show (default 10)")
    rep.add_argument("--horizon", default="15m", help="horizon key to report, e.g. 15m (default)")
    args = parser.parse_args(argv)
    if args.command == "report":
        records = load_records(args.path)
        print(build_report(records, top=args.top, horizon=args.horizon))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
