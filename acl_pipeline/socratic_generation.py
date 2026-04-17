from __future__ import annotations

import re
from typing import Optional

from .logging_utils import StructuredLogger
from .modeling import RoleSession
from .prompts import build_socratic_messages
from .schemas import PythonTask, SocraticHint


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
    raw = session.generate([build_socratic_messages(task)])[0]
    cleaned = sanitize_socratic_text(raw)
    hint = SocraticHint(
        task_id=task.task_id,
        text=cleaned,
        raw_text=raw,
        metadata={"topic": task.topic},
    )
    logger.debug_dump("socratic_hint", task=task, hint=hint)
    return hint
