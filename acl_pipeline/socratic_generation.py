from __future__ import annotations

import re
from typing import List, TYPE_CHECKING

from .logging_utils import StructuredLogger
from .prompts import build_socratic_messages
from .schemas import PythonTask, SocraticHint
from .text_quality import detect_corrupted_hint_text

if TYPE_CHECKING:
    from .modeling import RoleSession


def sanitize_socratic_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", cleaned)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    cleaned = re.sub(r"(?s)```.*?```", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return "Which assertion fails first, and what concrete value do you see right before it?"
    trimmed = "\n".join(lines[:4]).strip()
    words = trimmed.split()
    if len(words) > 120:
        trimmed = " ".join(words[:120]).strip()
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
        hint = SocraticHint(
            task_id=task.task_id,
            text=cleaned,
            raw_text=raw,
            metadata={
                "topic": task.topic,
                "candidate_index": candidate_index,
                "is_corrupted": corruption["is_corrupted"],
                "corruption_reasons": corruption["reasons"],
            },
        )
        logger.debug_dump("socratic_hint", task=task, hint=hint)
        hints.append(hint)
    return hints
