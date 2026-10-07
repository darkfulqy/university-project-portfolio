from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_MOJIBAKE_RE = re.compile(r"[ÃÂÐÑØæçèéêëìíîïðñòóôõöøåäþÿ]")


def _text_quality(value: str) -> tuple[int, int, int, int]:
    cjk_count = len(_CJK_RE.findall(value))
    mojibake_count = len(_MOJIBAKE_RE.findall(value))
    replacement_count = value.count("\ufffd") + value.count("?")
    printable_count = sum(ch.isprintable() and not ch.isspace() for ch in value)
    return (cjk_count, printable_count, -mojibake_count, -replacement_count)


def _repair_once(value: str) -> str:
    best = value
    best_score = _text_quality(value)
    for source_encoding in ("latin1", "cp1252"):
        try:
            candidate = value.encode(source_encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        score = _text_quality(candidate)
        if score > best_score:
            best = candidate
            best_score = score
    return best


def repair_mojibake(value: str | None, max_rounds: int = 3) -> str | None:
    text = (value or "").strip()
    if not text:
        return value

    current = text
    for _ in range(max_rounds):
        repaired = _repair_once(current)
        if repaired == current:
            break
        current = repaired
    return current
