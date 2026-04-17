from __future__ import annotations

import re
from typing import Any, Dict, List


_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")
_CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_corrupted_hint_text(text: str) -> Dict[str, Any]:
    raw = str(text or "")
    reasons: List[str] = []
    lowered = raw.lower()

    if "unicodeencodeerror" in lowered or ("traceback" in lowered and "encode" in lowered):
        reasons.append("encoding_traceback")
    if _EMOJI_RE.search(raw):
        reasons.append("emoji")
    if _CJK_RE.search(raw):
        reasons.append("cjk_script")
    if _LATIN_RE.search(raw) and _CYRILLIC_RE.search(raw):
        reasons.append("latin_cyrillic_mix")
    if _LATIN_RE.search(raw) and _ARABIC_RE.search(raw):
        reasons.append("latin_arabic_mix")

    for token in ("вЂ", "â€™", "â€œ", "â€", "Ã", "�"):
        if token in raw:
            reasons.append(f"mojibake:{token}")

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "is_corrupted": bool(unique_reasons),
        "reasons": unique_reasons,
    }
