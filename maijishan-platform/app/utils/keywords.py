import re


_SPLIT_PATTERN = re.compile(r"[，,;；、|/\n\r\t ]+")


def split_keywords(text: str | None) -> list[str]:
    if not text:
        return []
    cleaned = text.replace("关键词", "").replace("关键字", "").replace(":", "").replace("：", "")
    parts = [item.strip() for item in _SPLIT_PATTERN.split(cleaned) if item.strip()]
    deduped = []
    seen = set()
    for item in parts:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped
