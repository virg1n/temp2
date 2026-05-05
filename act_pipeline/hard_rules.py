from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from .config import JudgeConfig


DIRECT_FIX_PATTERNS = [
    re.compile(r"\b(replace|change|set|modify|rewrite)\b.{0,60}\b(with|to|as)\b", re.IGNORECASE),
    re.compile(r"\b(the\s+)?fix\s+is\b", re.IGNORECASE),
    re.compile(r"\bhere\s+is\s+the\s+(corrected|fixed)\s+code\b", re.IGNORECASE),
    re.compile(r"\buse\s+this\s+code\b", re.IGNORECASE),
    re.compile(r"\breturn\s+.+\b(instead|rather)\b", re.IGNORECASE),
]

CODE_LIKE_PATTERNS = [
    re.compile(r"```"),
    re.compile(r"^\s*(def|class|return|for|while|if|elif|else|assert)\b", re.MULTILINE),
    re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=", re.MULTILINE),
]

GENERIC_PATTERNS = [
    re.compile(r"\b(check|review|look at)\s+(your\s+)?(code|logic|loop)\b", re.IGNORECASE),
    re.compile(r"\bprint\s+(some\s+)?variables\b", re.IGNORECASE),
]


@dataclass
class RuleAdjustment:
    score: float
    flags: list[str] = field(default_factory=list)
    score_ceiling: float | None = None


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _alphabetic_word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z][A-Za-z'-]*[A-Za-z]\b", text))


def _has_real_question_mark(text: str) -> bool:
    if "?" not in text:
        return False
    # Reject punctuation-only question runs such as "????".
    return bool(re.search(r"[A-Za-z0-9_`')\]]\s*\?", text))


def _dominance_ratios(text: str) -> tuple[float, float]:
    non_space = [char for char in text if not char.isspace()]
    if not non_space:
        return 1.0, 1.0
    punctuation = sum(1 for char in non_space if not char.isalnum())
    markdown = sum(1 for char in non_space if char in "`*_#[]()>:-")
    total = len(non_space)
    return punctuation / total, markdown / total


def _identifier_set(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.add(node.name)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
    return identifiers


def _invented_identifier_penalty(hint: str, code: str) -> bool:
    known = _identifier_set(code)
    if not known:
        return False
    code_tokens = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", hint))
    code_tokens.update(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|\[|=)", hint))
    ignored = {
        "AssertionError",
        "IndexError",
        "KeyError",
        "None",
        "True",
        "False",
        "len",
        "range",
        "sorted",
        "sum",
        "min",
        "max",
    }
    invented = {token for token in code_tokens if token not in known and token not in ignored}
    return bool(invented)


def _is_repetitive(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    if len(words) < 12:
        return False
    trigrams = list(zip(words, words[1:], words[2:]))
    return len(trigrams) - len(set(trigrams)) >= 3


def apply_hard_rules(
    hint: str,
    code: str,
    raw_score: float,
    judge_json: dict,
    cfg: JudgeConfig,
) -> RuleAdjustment:
    score = float(raw_score)
    flags: list[str] = []
    stripped = hint.strip()

    if not stripped:
        return RuleAdjustment(score=0.0, flags=["empty_response"])

    punctuation_ratio, markdown_ratio = _dominance_ratios(stripped)
    if (
        punctuation_ratio >= cfg.punctuation_markdown_zero_threshold
        or markdown_ratio >= cfg.markdown_zero_threshold
    ):
        return RuleAdjustment(
            score=0.0,
            flags=["punctuation_or_markdown_dominated"],
            score_ceiling=0.0,
        )

    if any(pattern.search(stripped) for pattern in DIRECT_FIX_PATTERNS):
        return RuleAdjustment(
            score=cfg.direct_fix_forced_score,
            flags=["direct_fix_pattern"],
            score_ceiling=cfg.direct_fix_forced_score,
        )

    if "no_solution_reveal" in judge_json and not bool(judge_json["no_solution_reveal"]):
        score *= cfg.leakage_multiplier
        flags.append("judge_solution_leak")

    if any(pattern.search(stripped) for pattern in CODE_LIKE_PATTERNS):
        score -= cfg.code_block_penalty
        flags.append("code_like_hint")

    words = _word_count(stripped)
    if words > cfg.hard_max_hint_words:
        score = min(score, 2.0)
        flags.append("far_too_long")
    elif words > cfg.max_hint_words:
        score -= cfg.too_long_penalty
        flags.append("too_long")

    question_count = stripped.count("?")
    score_ceiling: float | None = None
    if not _has_real_question_mark(stripped):
        score = min(score - cfg.no_question_penalty, cfg.no_question_score_ceiling)
        flags.append("no_question")
    elif question_count > cfg.max_questions:
        score -= cfg.too_many_questions_penalty * (question_count - cfg.max_questions)
        flags.append("too_many_questions")

    alpha_words = _alphabetic_word_count(stripped)
    if alpha_words < cfg.min_alphabetic_words:
        score = min(score, cfg.low_alpha_words_score_ceiling)
        flags.append("too_few_alphabetic_words")

    if any(pattern.search(stripped) for pattern in GENERIC_PATTERNS) and words < 35:
        score -= cfg.generic_penalty
        flags.append("generic_hint")

    if _invented_identifier_penalty(stripped, code):
        score -= 1.0
        flags.append("possible_invented_identifier")

    if _is_repetitive(stripped):
        score = min(score, 2.0)
        flags.append("repetitive")

    non_ascii = sum(1 for char in stripped if ord(char) > 127)
    if non_ascii / max(len(stripped), 1) > 0.15:
        score = min(score, 3.0)
        flags.append("mixed_or_non_ascii_text")

    final_score = max(0.0, min(10.0, score))
    if flags:
        # Any future normalization layer must clamp flagged outputs at this
        # ceiling instead of raising them back into the preferred range.
        score_ceiling = final_score
    return RuleAdjustment(score=final_score, flags=flags, score_ceiling=score_ceiling)
