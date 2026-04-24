from __future__ import annotations

import re
from typing import List, TYPE_CHECKING

from .logging_utils import StructuredLogger
from .prompts import build_socratic_messages
from .schemas import PythonTask, SocraticHint
from .text_quality import detect_corrupted_hint_text

if TYPE_CHECKING:
    from .modeling import RoleSession


_FALLBACK_HINT = "Which assertion fails first, and what concrete value do you see right before it?"
_DIRECT_FIX_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bthe fix is\b",
        r"\bthe bug is\b",
        r"\bfix (?:it|this) by\b",
        r"\breplace\b.+\bwith\b",
        r"\bchange\b.+\bto\b",
        r"\buse\s+[^.\n]{0,80}\s+instead\b",
        r"\badd a missing\b",
        r"\brename\b.+\bto\b",
    )
]
_CODE_LINE_PATTERNS = [
    re.compile(r"^\s*(?:def|class|if|elif|else|for|while|try|except|finally|with|return|raise|import|from|assert|print)\b"),
    re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^=]"),
]


def socratic_contract_violation(text: str) -> bool:
    raw = str(text or "")
    lowered = raw.lower()
    if "<think" in lowered or "</think" in lowered:
        return True
    if "```" in raw:
        return True
    if any(pattern.search(raw) for pattern in _DIRECT_FIX_PATTERNS):
        return True
    code_like_lines = 0
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped and any(pattern.search(stripped) for pattern in _CODE_LINE_PATTERNS):
            code_like_lines += 1
    return code_like_lines >= 2


def sanitize_socratic_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"(?is)<think\b[^>]*>.*?</think>", "", cleaned)
    cleaned = re.sub(r"(?is)<think\b[^>]*>.*$", "", cleaned)
    cleaned = re.sub(r"(?is)</think>", "", cleaned)
    cleaned = re.sub(r"(?i)^\s*assistant\b[:\s-]*", "", cleaned).strip()
    cleaned = re.sub(r"(?s)```.*?```", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return _FALLBACK_HINT
    trimmed = "\n".join(lines[:4]).strip()
    words = trimmed.split()
    if len(words) > 120:
        trimmed = " ".join(words[:120]).strip()
    if socratic_contract_violation(trimmed):
        return _FALLBACK_HINT
    return trimmed


def generate_socratic_hint(session: RoleSession, task: PythonTask, logger: StructuredLogger) -> SocraticHint:
    return generate_socratic_hints(session, task, count=1, logger=logger)[0]


def generate_socratic_hints(
    session: RoleSession,
    task: PythonTask,
    *,
    count: int,
    logger: StructuredLogger,
) -> List[SocraticHint]:
    messages = build_socratic_messages(task)
    raw_outputs = session.generate([messages for _ in range(max(1, int(count)))])
    hints: List[SocraticHint] = []
    for candidate_index, raw in enumerate(raw_outputs):
        cleaned = sanitize_socratic_text(raw)
        corruption = detect_corrupted_hint_text(raw)
        contract_violation = socratic_contract_violation(raw)
        hint = SocraticHint(
            task_id=task.task_id,
            text=cleaned,
            raw_text=raw,
            metadata={
                "topic": task.topic,
                "candidate_index": candidate_index,
                "socratic_contract_violation": contract_violation,
                "sanitized_to_fallback": cleaned == _FALLBACK_HINT and bool(str(raw or "").strip()),
                "is_corrupted": corruption["is_corrupted"],
                "corruption_reasons": corruption["reasons"],
            },
        )
        logger.debug_dump("socratic_hint", task=task, hint=hint)
        hints.append(hint)
    return hints
