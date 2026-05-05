from __future__ import annotations

import dataclasses
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .logging_utils import StructuredLogger
from .prompts import build_socratic_messages
from .schemas import PythonTask, SocraticHint
from .text_quality import detect_corrupted_hint_text

if TYPE_CHECKING:
    from .config import SocraticDiversitySettings
    from .modeling import RoleSession


_DEFAULT_DIVERSITY_OVERRIDES: List[Dict[str, Any]] = [
    {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.0},
    {"temperature": 1.0, "top_p": 0.95, "repetition_penalty": 1.05},
    {"temperature": 1.3, "top_p": 1.0, "repetition_penalty": 1.10},
    {"temperature": 1.0, "top_p": 1.0, "repetition_penalty": 1.15},
]
_DEFAULT_DIVERSITY_SALTS: List[str] = [
    "Focus your questions on data flow: the inputs, intermediate values, and what each computation produces.",
    "Focus your questions on control flow: which branch is taken, how loops terminate, how exceptions propagate.",
    "Focus your questions on state and mutation: what is changed when, and how aliasing or shared references could affect it.",
    "Focus your questions on the failing assertion: what value the test expects versus what the code actually produces, and where they first diverge.",
]


_FALLBACK_HINT = "Which assertion fails first, and what concrete value do you see right before it?"
_DIRECT_FIX_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bthe fix is\b",
        r"\bthe bug is\b",
        r"\bfix (?:it|this) by\b",
        r"\bcorrect approach is\b",
        r"\bcurrent code uses\b.+\bshould use\b",
        r"\breplace\b.+\bwith\b",
        r"\bchange\b.+\bto\b",
        r"\buse\s+[^.\n]{0,80}\s+instead\b",
        r"\buse\s+[`']?[^`'\n]{1,40}(?:==|!=|<=|>=|<|>|//|/|\+|-|\*|%| and | or )[^`'\n]{0,40}[`']?\s+instead of\s+[`']?[^`'\n]{1,80}[`']?",
        r"\buse\s+[`']?(?:==|!=|<=|>=|<|>|//|/|\+|-|\*|%|and|or)[`']?\s+instead of\s+[`']?(?:==|!=|<=|>=|<|>|//|/|\+|-|\*|%|and|or)[`']?",
        r"\badd a missing\b",
        r"\brename\b.+\bto\b",
        r"\bfinal code should\b",
        r"\bmake\s+[A-Za-z_][A-Za-z0-9_]*\s+an instance variable\b",
        r"\b(?:swap|switch)\s+(?:the\s+)?(?:operator|comparison)?\s*[`']?(?:==|!=|<=|>=|<|>|//|/|\+|-|\*|%|and|or)[`']?\s+(?:to|for|with)\s+[`']?(?:==|!=|<=|>=|<|>|//|/|\+|-|\*|%|and|or)[`']?",
        r"\b(?:change|replace)\s+(?:the\s+)?(?:operator|comparison)\s+(?:from\s+)?[`']?(?:==|!=|<=|>=|<|>|//|/|\+|-|\*|%|and|or)[`']?",
        r"\b(?:return|set|compute)\s+[`']?[^.`'\n]{1,100}(?:==|!=|<=|>=|<|>|//|/|\+|-|\*|%)[^.`'\n]{1,100}[`']?",
        r"\b(?:return|set|compute|call)\s+[`']?[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?\([^)\n]{0,120}\)[`']?",
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
    if len(words) > 100:
        trimmed = " ".join(words[:100]).strip()
    if socratic_contract_violation(trimmed):
        return _FALLBACK_HINT
    return trimmed


def generate_socratic_hint(session: RoleSession, task: PythonTask, logger: StructuredLogger) -> SocraticHint:
    return generate_socratic_hints(session, task, count=1, logger=logger)[0]


_DIVERSITY_NUMERIC_FIELDS = {"temperature", "top_p", "repetition_penalty"}


def _override_for_index(overrides: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
    if not overrides:
        return {}
    raw = overrides[index % len(overrides)]
    cleaned: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in _DIVERSITY_NUMERIC_FIELDS:
            try:
                cleaned[key] = float(value)
            except (TypeError, ValueError):
                continue
        elif key == "do_sample":
            cleaned[key] = bool(value)
    return cleaned


def _salt_for_index(salts: List[str], index: int) -> Optional[str]:
    if not salts:
        return None
    return salts[index % len(salts)]


def _build_candidate_generation(base: Any, override: Dict[str, Any]) -> Any:
    if not override:
        return base
    fields = {f.name for f in dataclasses.fields(base)}
    safe_override = {k: v for k, v in override.items() if k in fields}
    if not safe_override:
        return base
    return dataclasses.replace(base, **safe_override)


def generate_socratic_hints(
    session: RoleSession,
    task: PythonTask,
    *,
    count: int,
    logger: StructuredLogger,
    diversity: Optional["SocraticDiversitySettings"] = None,
) -> List[SocraticHint]:
    candidate_count = max(1, int(count))
    use_diversity = bool(diversity and diversity.enabled and candidate_count > 1)

    if not use_diversity:
        messages = build_socratic_messages(task)
        raw_outputs = session.generate([messages for _ in range(candidate_count)])
        salts_used: List[Optional[str]] = [None] * candidate_count
        gen_used: List[Optional[Dict[str, Any]]] = [None] * candidate_count
    else:
        overrides = list(diversity.candidate_overrides) or _DEFAULT_DIVERSITY_OVERRIDES
        salts = list(diversity.candidate_focus_salts) or _DEFAULT_DIVERSITY_SALTS
        raw_outputs = []
        salts_used = []
        gen_used = []
        for candidate_index in range(candidate_count):
            override = _override_for_index(overrides, candidate_index)
            salt = _salt_for_index(salts, candidate_index)
            messages = build_socratic_messages(task, focus_salt=salt)
            candidate_generation = _build_candidate_generation(session.generation, override)
            outputs = session.generate(
                [messages],
                generation=candidate_generation if candidate_generation is not session.generation else None,
            )
            raw_outputs.extend(outputs)
            salts_used.append(salt)
            gen_used.append(override or None)

    hints: List[SocraticHint] = []
    for candidate_index, raw in enumerate(raw_outputs):
        cleaned = sanitize_socratic_text(raw)
        corruption = detect_corrupted_hint_text(raw)
        contract_violation = socratic_contract_violation(raw)
        metadata: Dict[str, Any] = {
            "topic": task.topic,
            "candidate_index": candidate_index,
            "text_source": "raw_text",
            "sanitized_text": cleaned,
            "socratic_contract_violation": contract_violation,
            "sanitized_to_fallback": cleaned == _FALLBACK_HINT and bool(str(raw or "").strip()),
            "is_corrupted": corruption["is_corrupted"],
            "corruption_reasons": corruption["reasons"],
        }
        if use_diversity:
            metadata["diversity_focus_salt"] = salts_used[candidate_index]
            metadata["diversity_generation_overrides"] = gen_used[candidate_index]
        hint = SocraticHint(
            task_id=task.task_id,
            text=raw,
            raw_text=raw,
            metadata=metadata,
        )
        logger.debug_dump("socratic_hint", task=task, hint=hint)
        hints.append(hint)
    return hints
