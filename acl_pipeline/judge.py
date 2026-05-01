from __future__ import annotations

import builtins
import json
import keyword
import re
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Set

from .logging_utils import StructuredLogger
from .modeling import ModelPool
from .prompts import build_judge_batch_messages, build_socratic_messages
from .schemas import JudgeOutput, PythonTask, SocraticHint
from .text_quality import detect_corrupted_hint_text


_CODE_BLOCK_RE = re.compile(r"## (?P<section>Task|Code|Error)\n```[^\n]*\n(?P<body>.*?)```", re.DOTALL)
_TASK_SECTION_RE = re.compile(r"## Task\n(?P<body>.*?)\n\n## Code", re.DOTALL)
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_REFERENCE_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]\n]{1,32}\]|\.[A-Za-z_][A-Za-z0-9_]*|\([^)\n]{0,32}\))+")
_DEF_RE = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_SNAKE_CASE_RE = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")
_CAMEL_CASE_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}\b")
_BACKTICK_RE = re.compile(r"`([^`\n]{1,64})`")
_GENERIC_HINT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bwalk through\b",
        r"\bcheck the logic\b",
        r"\bthink about\b",
        r"\blook carefully\b",
        r"\bstep through\b",
        r"\btrace the values\b",
        r"\bwhat do you expect\b",
        r"\bdoes it match\b",
        r"\bcompare the expected\b",
        r"\bwhere does it go wrong\b",
    )
]
_STOPWORDS = {
    "about",
    "actual",
    "after",
    "again",
    "around",
    "before",
    "because",
    "between",
    "branch",
    "branches",
    "check",
    "code",
    "compare",
    "concrete",
    "consider",
    "condition",
    "conditions",
    "control",
    "debug",
    "different",
    "during",
    "each",
    "edge",
    "error",
    "fails",
    "failure",
    "focus",
    "first",
    "flow",
    "function",
    "given",
    "helper",
    "hint",
    "index",
    "inspect",
    "input",
    "line",
    "likely",
    "logic",
    "match",
    "maybe",
    "name",
    "notice",
    "output",
    "path",
    "paths",
    "passed",
    "point",
    "question",
    "reason",
    "reproduced",
    "return",
    "right",
    "running",
    "same",
    "seems",
    "should",
    "specific",
    "state",
    "step",
    "student",
    "tests",
    "trace",
    "using",
    "value",
    "values",
    "variable",
    "variables",
    "walk",
    "what",
    "when",
    "where",
    "whether",
    "which",
    "while",
    "why",
}
_BUILTIN_NAMES = {name.lower() for name in dir(builtins)}
_EXTRA_ALLOWED_IDENTIFIERS = {
    "assert",
    "assertion",
    "bug",
    "debug",
    "false",
    "none",
    "python",
    "runtime",
    "true",
}
_PASS_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bno failing assertion\b",
        r"\bno runtime error\b",
        r"\bno bug (?:was )?reproduced\b",
        r"\bthe (?:current )?run (?:passes?|passed)\b",
        r"\ball tests? pass(?:ed)?\b",
        r"\bprogram (?:exited|runs?) successfully\b",
        r"\bdoes not reproduce (?:the |a )?bug\b",
        r"\bnothing is failing\b",
        r"\bno (?:errors?|exceptions?|failures?)\b",
        r"\bthere (?:is|are) no (?:errors?|exceptions?|failures?|failing assertions?)\b",
    )
]
_MALFORMED_PATTERNS = [
    re.compile(r"\\\?{2,}"),
    re.compile(r"[?؟]{3,}"),
    re.compile(r"`[^`\n]*\{[^}\n]*`"),
    re.compile(r"`[^`\n]*\[[^\]\n]*`"),
]
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
        r"\b(?:it|this|that|the (?:answer|result|output|value|code|line|function|return value|expected))\s+should be\b",
        r"\b(?:only|just|always|never|every|all|any)\s+[^.\n]{0,80}\s+should\s+(?:be|return|raise|equal|produce|yield|contain|include|match)\b",
        r"\bshould\s+(?:return|raise|equal|produce|yield|contain|include|match)\b",
    )
]
_CODE_OUTPUT_PATTERNS = [
    re.compile(r"^\s*(?:def|class|if|elif|else|for|while|try|except|finally|with|return|raise|import|from|assert|print)\b"),
    re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^=]"),
]
_SYNTAX_OR_INDENT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bsyntax(?:error)?\b",
        r"\bindent(?:ation|ed|ing)?\b",
        r"\btaberror\b",
        r"\bparse\b",
        r"\bparser\b",
        r"\bdelimiter\b",
        r"\bcolon\b",
        r"\bparenthes",
        r"\bbracket\b",
        r"\bquote\b",
        r"\btoken\b",
        r"\binvalid python\b",
    )
]
_NAME_ERROR_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bnameerror\b",
        r"\bnot defined\b",
        r"\bundefined (?:name|variable|identifier)\b",
        r"\bunknown (?:name|variable|identifier)\b",
        r"\bwrong variable\b",
        r"\btypo\b",
        r"\bmisspell",
        r"\brename\b",
    )
]
_PASSED_EXECUTION_REWARD = 0.5


def _hint_text_for_judge(hint: SocraticHint) -> str:
    # Score the model's original output, not the sanitized hint, so verbosity,
    # code fences, direct fixes, and <think> leakage affect the reward.
    return str(getattr(hint, "raw_text", "") or hint.text or "")


def _extract_json(text: str) -> Optional[Any]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass

    stripped = raw.replace("```json", "").replace("```", "").strip()
    for left, right in (("[", "]"), ("{", "}")):
        start = stripped.find(left)
        end = stripped.rfind(right)
        if 0 <= start < end:
            try:
                return json.loads(stripped[start : end + 1])
            except Exception:
                continue
    return None


def _judge_items_from_parsed(parsed: Any, expected_count: int) -> List[Any]:
    if isinstance(parsed, list):
        return list(parsed)
    if isinstance(parsed, dict):
        maybe_items = parsed.get("items") or parsed.get("scores") or parsed.get("results")
        if isinstance(maybe_items, list):
            return list(maybe_items)
        if expected_count == 1 and any(
            key in parsed
            for key in (
                "no_solution_reveal",
                "bug_localization",
                "usefulness",
                "socratic_style",
                "technical_accuracy",
                "task_quality",
                "task_is_valid_for_socratic",
                "hint_is_valid_for_socratic",
            )
        ):
            return [parsed]
    return []


def _extract_prompt_sections(prompt_text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    for match in _CODE_BLOCK_RE.finditer(str(prompt_text or "")):
        sections[match.group("section").lower()] = match.group("body").strip()
    return sections


def _infer_execution_status(error_text: str) -> str:
    text = str(error_text or "")
    if "No failing assertion or runtime error was reproduced." in text:
        return "passed"
    if "IndentationError" in text or "TabError" in text:
        return "indentation_error"
    if "SyntaxError" in text:
        return "syntax_error"
    if "NameError" in text:
        return "nameerror"
    return "failed"


def _normalize_identifier(token: str) -> str:
    return str(token or "").strip().lower()


def _identifier_set(text: str) -> Set[str]:
    return {_normalize_identifier(token) for token in _IDENTIFIER_RE.findall(str(text or ""))}


def _code_like_identifier_set(text: str) -> Set[str]:
    tokens: Set[str] = set()
    for token in _SNAKE_CASE_RE.findall(str(text or "")):
        normalized = _normalize_identifier(token)
        if normalized:
            tokens.add(normalized)
    for token in _CAMEL_CASE_RE.findall(str(text or "")):
        normalized = _normalize_identifier(token)
        if normalized:
            tokens.add(normalized)
    for token in _BACKTICK_RE.findall(str(text or "")):
        root = _root_identifier(token)
        if root:
            tokens.add(root)
        normalized = _normalize_identifier(token)
        if normalized and _IDENTIFIER_RE.fullmatch(token) and ("_" in normalized or any(ch.isupper() for ch in token)):
            tokens.add(normalized)
    for reference in _REFERENCE_RE.findall(str(text or "")):
        root = _root_identifier(reference)
        if root:
            tokens.add(root)
    return {token for token in tokens if _is_trackable_identifier(token)}


def _definition_names(code: str) -> Set[str]:
    return {_normalize_identifier(token) for token in _DEF_RE.findall(str(code or ""))}


def _assert_lines(code: str) -> List[str]:
    return [line.strip() for line in str(code or "").splitlines() if line.strip().startswith("assert ")]


def _error_signal_tokens(error_text: str) -> Set[str]:
    generic = {
        "assertionerror",
        "error",
        "exception",
        "false",
        "file",
        "last",
        "line",
        "most",
        "recent",
        "runtimeerror",
        "traceback",
        "true",
    }
    return {
        token
        for token in _identifier_set(error_text)
        if token and token not in generic and len(token) >= 3
    }


def _is_trackable_identifier(token: str) -> bool:
    normalized = _normalize_identifier(token)
    if len(normalized) < 3:
        return False
    if keyword.iskeyword(normalized):
        return False
    if normalized in _STOPWORDS or normalized in _BUILTIN_NAMES or normalized in _EXTRA_ALLOWED_IDENTIFIERS:
        return False
    return True


def _root_identifier(reference: str) -> str:
    match = _IDENTIFIER_RE.match(reference.strip())
    if not match:
        return ""
    return _normalize_identifier(match.group(0))


def _hint_malformed_reasons(hint_text: str) -> List[str]:
    reasons: List[str] = []
    if hint_text.count("`") % 2 == 1:
        reasons.append("unbalanced_backticks")
    for pattern in _MALFORMED_PATTERNS:
        if pattern.search(hint_text):
            reasons.append(f"malformed:{pattern.pattern}")
    if any(abs(hint_text.count(left) - hint_text.count(right)) >= 2 for left, right in (("(", ")"), ("[", "]"), ("{", "}"))):
        reasons.append("unbalanced_delimiters")
    return list(dict.fromkeys(reasons))


def _contains_code_output(hint_text: str) -> bool:
    if "```" in hint_text:
        return True
    lines = [line.rstrip() for line in str(hint_text or "").splitlines() if line.strip()]
    code_like_lines = 0
    for line in lines:
        stripped = line.strip()
        if any(pattern.search(stripped) for pattern in _CODE_OUTPUT_PATTERNS):
            code_like_lines += 1
    if len(lines) == 1:
        stripped = lines[0].strip()
        return code_like_lines == 1 and "?" not in stripped and bool(re.match(r"^(?:def|class|return|raise|assert|print)\b|^[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^=]", stripped))
    return code_like_lines >= 2


def _contains_direct_fix(hint_text: str) -> bool:
    text = str(hint_text or "")
    lowered = text.lower()
    if "<think" in lowered or "</think>" in lowered:
        return True
    return any(pattern.search(text) for pattern in _DIRECT_FIX_PATTERNS)


def _intended_bug_signal_text(task: Optional[PythonTask]) -> str:
    if task is None:
        return ""
    spec = dict(task.metadata.get("red_spec") or {})
    chunks = [
        str(spec.get("target_function") or ""),
        str(spec.get("intended_bug") or ""),
        str(spec.get("expected_first_failure") or ""),
        str(task.metadata.get("failure_mode") or ""),
        str((spec.get("metadata") or {}).get("failure_mode") or ""),
    ]
    return "\n".join(chunk for chunk in chunks if chunk).lower()


def _execution_error_matches_intended_bug(task: Optional[PythonTask], execution_status: str) -> Optional[bool]:
    if execution_status not in {"syntax_error", "indentation_error", "nameerror"}:
        return None
    signal = _intended_bug_signal_text(task)
    if not signal:
        return None
    patterns = _NAME_ERROR_PATTERNS if execution_status == "nameerror" else _SYNTAX_OR_INDENT_PATTERNS
    return any(pattern.search(signal) for pattern in patterns)


def _build_row(prompt_text: str, completion: str, task: Optional[PythonTask] = None) -> Dict[str, Any]:
    sections = _extract_prompt_sections(prompt_text)
    task_match = _TASK_SECTION_RE.search(str(prompt_text or ""))
    error_text = sections.get("error", "")
    execution_status = _infer_execution_status(error_text)
    row = {
        "statement": task_match.group("body").strip() if task_match else sections.get("task", ""),
        "code": sections.get("code", ""),
        "observed_failure": error_text,
        "execution_status": execution_status,
        "assistant_response": completion[:1800],
    }
    if task is not None:
        spec = dict(task.metadata.get("red_spec") or {})
        row["red_spec"] = {
            "topic": spec.get("topic", task.topic),
            "target_function": spec.get("target_function", ""),
            "intended_bug": spec.get("intended_bug", ""),
            "expected_first_failure": spec.get("expected_first_failure", ""),
            "metadata": dict(spec.get("metadata") or {}),
        }
    return row


def _hint_quality_features(row: Dict[str, Any]) -> Dict[str, Any]:
    code = row.get("code", "")
    error_text = row.get("observed_failure", "")
    hint_text = str(row.get("assistant_response") or "")
    hint_lower = hint_text.lower()

    available_identifiers = _identifier_set(code) | _identifier_set(error_text) | _definition_names(code)
    assert_token_set = _identifier_set("\n".join(_assert_lines(code)))
    definition_names = _definition_names(code)
    error_tokens = _error_signal_tokens(error_text)

    hint_identifiers = _code_like_identifier_set(hint_text)
    invented_identifiers = sorted(token for token in hint_identifiers if token not in available_identifiers)

    invented_references: List[str] = []
    for reference in _REFERENCE_RE.findall(hint_text):
        root = _root_identifier(reference)
        if not root or not _is_trackable_identifier(root):
            continue
        if root not in available_identifiers:
            invented_references.append(reference)
    invented_references = list(dict.fromkeys(invented_references))

    grounding_hits: List[str] = []
    if definition_names & hint_identifiers:
        grounding_hits.append("function_or_class")
    if error_tokens & hint_identifiers:
        grounding_hits.append("error_token")

    generic_hits = sum(1 for pattern in _GENERIC_HINT_PATTERNS if pattern.search(hint_text))
    references_passed_execution = any(pattern.search(hint_text) for pattern in _PASS_PATTERNS)

    malformed_reasons = _hint_malformed_reasons(hint_text)

    word_count = len(hint_text.split())
    question_count = hint_text.count("?")

    delta = 0.0
    reasons: List[str] = []
    if grounding_hits:
        delta += min(0.6, 0.3 * len(grounding_hits))
        reasons.extend(f"grounded:{name}" for name in grounding_hits)
    if invented_identifiers:
        delta -= min(0.6, 0.25 * len(invented_identifiers))
        reasons.append(f"invented_identifiers:{len(invented_identifiers)}")
    if invented_references:
        delta -= min(0.6, 0.8 * len(invented_references))
        reasons.append(f"invented_references:{len(invented_references)}")
    if malformed_reasons:
        delta -= min(2.0, 0.8 * len(malformed_reasons))
        reasons.extend(malformed_reasons)

    if word_count > 180:
        length_penalty = min(0.6, 0.01 * (word_count - 180))
        delta -= length_penalty
        reasons.append(f"too_long:{word_count}w")
    if question_count > 4:
        question_penalty = min(0.6, 0.3 * (question_count - 4))
        delta -= question_penalty
        reasons.append(f"too_many_questions:{question_count}")

    severe_hint_failure = False
    if malformed_reasons and invented_references:
        severe_hint_failure = True
    if len(invented_references) >= 2:
        severe_hint_failure = True
    if len(invented_identifiers) >= 4 and not grounding_hits:
        severe_hint_failure = True

    if row.get("execution_status") == "passed":
        if references_passed_execution:
            delta += 0.35
            reasons.append("noticed_passed_execution")
        else:
            delta -= 0.6
            reasons.append("missed_passed_execution")

    return {
        "delta": delta,
        "reasons": reasons,
        "grounding_hits": grounding_hits,
        "generic_hits": generic_hits,
        "invented_identifiers": invented_identifiers,
        "invented_references": invented_references,
        "malformed_reasons": malformed_reasons,
        "references_passed_execution": references_passed_execution,
        "severe_hint_failure": severe_hint_failure,
        "execution_status": row.get("execution_status"),
    }


class JudgeService:
    def __init__(self, model_pool: ModelPool, logger: StructuredLogger) -> None:
        self.model_pool = model_pool
        self.logger = logger
        window_size = max(1, int(getattr(self.model_pool.config.judge, "normalize_across_batches", 8)))
        self._normalization_window: Deque[float] = deque(maxlen=window_size)

    def _weights(self) -> Dict[str, float]:
        return {
            key: float(value)
            for key, value in dict(self.model_pool.config.judge.reward_weights).items()
            if key != "no_solution_reveal"
        }

    def _coerce_criteria_scores(self, item: Any) -> Dict[str, float]:
        weights = self._weights()

        def coerce(raw: Any) -> float:
            if isinstance(raw, bool):
                return 1.0 if raw else 0.0
            try:
                return max(0.0, float(raw))
            except Exception:
                return 0.0

        if isinstance(item, dict):
            return {key: coerce(item.get(key, 0.0)) for key in weights}
        return {key: coerce(item) for key in weights}

    def _coerce_no_solution_reveal(self, item: Any, gate: Dict[str, Any]) -> bool:
        if gate.get("contains_direct_fix") or gate.get("contains_code_output"):
            return False
        if not isinstance(item, dict) or "no_solution_reveal" not in item:
            return True
        raw = item.get("no_solution_reveal")
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return float(raw) >= 5.0
        text = str(raw or "").strip().lower()
        if text in {"true", "yes", "no leak", "no leakage", "clean"}:
            return True
        if text in {"false", "no", "leak", "leaked", "solution leak"}:
            return False
        return True

    def _weighted_score(self, criteria_scores: Dict[str, float]) -> float:
        weights = self._weights()
        total_weight = sum(max(0.0, float(value)) for value in weights.values())
        if total_weight <= 0:
            return 0.0
        total = 0.0
        for key, weight in weights.items():
            total += float(criteria_scores.get(key, 0.0)) * float(weight)
        return max(0.0, total / total_weight)

    def _normalize_post_scores(self, pre_normalize_scores: List[float]) -> List[float]:
        window_size = max(1, int(getattr(self.model_pool.config.judge, "normalize_across_batches", 8)))
        episode_batch_size = max(1, int(getattr(self.model_pool.config.judge, "episode_batch_size", 1)))
        if self._normalization_window.maxlen != window_size:
            self._normalization_window = deque(self._normalization_window, maxlen=window_size)
        active_window: Deque[float]
        if window_size <= episode_batch_size:
            active_window = deque(maxlen=window_size)
        else:
            active_window = self._normalization_window

        post_scores: List[float] = []
        for pre_normalize in pre_normalize_scores:
            pre = float(pre_normalize)
            active_window.append(pre)
            window = list(active_window)
            if not window:
                post_scores.append(max(0.0, min(10.0, pre)))
                continue
            observed_max = max(window) if window else pre
            max_norm = min(10.0, observed_max)
            scale = (max_norm / observed_max) if observed_max > 0 else 1.0
            post_scores.append(max(0.0, min(10.0, pre * scale)))
        return post_scores

    def _hard_rule_gate(
        self,
        *,
        row: Dict[str, Any],
        task: Optional[PythonTask],
        corruption: Dict[str, Any],
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        weights = self._weights()
        zero_criteria = {key: 0.0 for key in weights}
        hint_text = str(row.get("assistant_response") or "")
        malformed_reasons = list(dict.fromkeys(list(features.get("malformed_reasons") or []) + list(corruption.get("reasons") or [])))
        code_output = _contains_code_output(hint_text)
        direct_fix = _contains_direct_fix(hint_text)
        pass_aware = bool(features.get("references_passed_execution"))
        gate = {
            "skip_llm": False,
            "forced_criteria": None,
            "forced_score": None,
            "task_quality_override": None,
            "force_task_valid": None,
            "force_hint_valid": None,
            "red_rejection_reason": "",
            "hint_rejection_reason": "",
            "reasons": [],
            "contains_code_output": code_output,
            "contains_direct_fix": direct_fix,
        }

        related_error = _execution_error_matches_intended_bug(task, str(row.get("execution_status") or ""))
        if related_error is False:
            gate["task_quality_override"] = 2.0
            gate["force_task_valid"] = False
            gate["red_rejection_reason"] = f"unrelated_{row['execution_status']}"
            gate["reasons"].append(f"task_invalid:unrelated_{row['execution_status']}")

        if malformed_reasons:
            gate["skip_llm"] = True
            gate["forced_criteria"] = dict(zero_criteria)
            gate["forced_score"] = 0.0
            gate["force_hint_valid"] = False
            gate["hint_rejection_reason"] = "malformed_hint_output"
            gate["reasons"].append("hint_zeroed:malformed_output")

        if direct_fix or code_output:
            gate["skip_llm"] = True
            gate["forced_criteria"] = dict(zero_criteria)
            gate["forced_score"] = 0.0
            gate["force_hint_valid"] = False
            gate["hint_rejection_reason"] = "solution_reveal"
            if code_output:
                gate["reasons"].append("hint_zeroed:code_output")
            if direct_fix:
                gate["reasons"].append("hint_zeroed:direct_fix")

        if row.get("execution_status") == "passed":
            gate["task_quality_override"] = 2.0
            gate["force_task_valid"] = False
            gate["red_rejection_reason"] = "already_correct_code"
            gate["reasons"].append("task_invalid:already_correct_code")

        if row.get("execution_status") == "passed" and not pass_aware:
            gate["reasons"].append("hint_review:missed_passed_execution")

        return gate

    def _task_and_hint_assessment(
        self,
        item: Any,
        *,
        score: float,
        severe_hint_failure: bool,
        corruption_detected: bool,
        hard_gate: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        threshold = float(self.model_pool.config.judge.bad_task_threshold)
        if not isinstance(item, dict):
            assessment = {
                "task_quality": 5.0,
                "task_is_valid_for_socratic": True,
                "hint_is_valid_for_socratic": not (severe_hint_failure or corruption_detected),
                "red_rejection_reason": "",
                "hint_rejection_reason": "",
            }
            if hard_gate:
                if hard_gate.get("task_quality_override") is not None:
                    assessment["task_quality"] = float(hard_gate["task_quality_override"])
                if hard_gate.get("force_task_valid") is not None:
                    assessment["task_is_valid_for_socratic"] = bool(hard_gate["force_task_valid"])
                if hard_gate.get("force_hint_valid") is not None:
                    assessment["hint_is_valid_for_socratic"] = bool(hard_gate["force_hint_valid"])
                if hard_gate.get("red_rejection_reason"):
                    assessment["red_rejection_reason"] = str(hard_gate["red_rejection_reason"])
                if hard_gate.get("hint_rejection_reason"):
                    assessment["hint_rejection_reason"] = str(hard_gate["hint_rejection_reason"])
            return assessment

        try:
            task_quality = max(0.0, min(10.0, float(item.get("task_quality", 5.0))))
        except Exception:
            task_quality = 5.0

        explicit_task_valid = item.get("task_is_valid_for_socratic")
        if explicit_task_valid is None and "use_for_socratic" in item:
            explicit_task_valid = item.get("use_for_socratic")
        if explicit_task_valid is None:
            task_is_valid = task_quality > threshold
        else:
            task_is_valid = bool(explicit_task_valid)

        explicit_hint_valid = item.get("hint_is_valid_for_socratic")
        if explicit_hint_valid is None:
            hint_is_valid = not (severe_hint_failure or corruption_detected) and score > 1.5
        else:
            hint_is_valid = bool(explicit_hint_valid)
            if severe_hint_failure or corruption_detected:
                hint_is_valid = False

        red_rejection_reason = str(item.get("red_rejection_reason") or "").strip()
        if not task_is_valid and not red_rejection_reason:
            red_rejection_reason = "judge_bad_task"

        hint_rejection_reason = str(item.get("hint_rejection_reason") or "").strip()
        if not hint_is_valid and not hint_rejection_reason:
            if corruption_detected or severe_hint_failure:
                hint_rejection_reason = "corrupted_or_hallucinated_hint"
            elif score <= 1.5:
                hint_rejection_reason = "very_low_hint_score"

        if hard_gate:
            if hard_gate.get("task_quality_override") is not None:
                task_quality = max(0.0, min(10.0, float(hard_gate["task_quality_override"])))
            if hard_gate.get("force_task_valid") is not None:
                task_is_valid = bool(hard_gate["force_task_valid"])
            if hard_gate.get("force_hint_valid") is not None:
                hint_is_valid = bool(hard_gate["force_hint_valid"])
            if hard_gate.get("red_rejection_reason"):
                red_rejection_reason = str(hard_gate["red_rejection_reason"])
            if hard_gate.get("hint_rejection_reason"):
                hint_rejection_reason = str(hard_gate["hint_rejection_reason"])

        return {
            "task_quality": task_quality,
            "task_is_valid_for_socratic": task_is_valid,
            "hint_is_valid_for_socratic": hint_is_valid,
            "red_rejection_reason": red_rejection_reason,
            "hint_rejection_reason": hint_rejection_reason,
        }

    def _apply_batch_spread(self, scores: List[float]) -> List[float]:
        strength = float(self.model_pool.config.judge.batch_spread_strength)
        if len(scores) < 2 or strength <= 0:
            return list(scores)

        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        std = variance ** 0.5
        if std <= 1e-6:
            return list(scores)

        adjusted: List[float] = []
        for score in scores:
            z = (score - mean) / std
            spread_score = score + (strength * 2.0 * z)
            adjusted.append(max(0.0, spread_score))
        return adjusted

    def score_pair_details(
        self,
        prompt_texts: List[str],
        completions: List[str],
        *,
        apply_batch_spread: bool,
        tasks: Optional[List[Optional[PythonTask]]] = None,
    ) -> List[Dict[str, Any]]:
        if not prompt_texts:
            return []

        task_items: List[Optional[PythonTask]] = list(tasks or [])
        if len(task_items) < len(prompt_texts):
            task_items.extend([None] * (len(prompt_texts) - len(task_items)))
        rows = [
            _build_row(prompt, completion, task)
            for prompt, completion, task in zip(prompt_texts, completions, task_items)
        ]
        corruption_flags = [detect_corrupted_hint_text(text) for text in completions]
        quality_features = [_hint_quality_features(row) for row in rows]
        hard_gates = [
            self._hard_rule_gate(
                row=row,
                task=task,
                corruption=corruption,
                features=features,
            )
            for row, task, corruption, features in zip(rows, task_items, corruption_flags, quality_features)
        ]

        judge_indexes = [index for index, gate in enumerate(hard_gates) if not gate["skip_llm"]]
        raw_items: List[Any] = [{} for _ in rows]
        raw_responses: List[str] = ["" for _ in rows]
        if judge_indexes:
            judge_rows = [rows[index] for index in judge_indexes]
            session = self.model_pool.get_judge()
            messages = build_judge_batch_messages(
                judge_rows,
                self._weights(),
                examples=getattr(self.model_pool.config.judge, "examples", []),
            )
            max_attempts = max(2, int(getattr(self.model_pool.config.judge, "vllm_max_retries", 2)) + 1)
            raw = ""
            parsed_items: List[Any] = []
            last_parsed_type = "None"
            for attempt in range(1, max_attempts + 1):
                raw = session.generate([messages])[0]
                parsed = _extract_json(raw)
                last_parsed_type = type(parsed).__name__ if parsed is not None else "None"
                parsed_items = _judge_items_from_parsed(parsed, expected_count=len(judge_rows))
                if len(parsed_items) == len(judge_rows):
                    break
                self.logger.warning(
                    "judge_malformed_response_retry",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    expected_items=len(judge_rows),
                    parsed_items=len(parsed_items),
                    parsed_type=last_parsed_type,
                    raw_chars=len(raw),
                    raw_preview=str(raw or "")[:500],
                )

            if len(parsed_items) != len(judge_rows):
                self.logger.error(
                    "judge_malformed_response_error",
                    expected_items=len(judge_rows),
                    parsed_items=len(parsed_items),
                    parsed_type=last_parsed_type,
                    raw_chars=len(raw),
                    raw_preview=str(raw or "")[:500],
                )
                raise RuntimeError(
                    "Judge returned malformed or truncated JSON after "
                    f"{max_attempts} attempt(s): expected {len(judge_rows)} item(s), "
                    f"parsed {len(parsed_items)}. Increase judge.generation.max_new_tokens "
                    "or reduce judge.episode_batch_size / socratic.dpo.num_hint_candidates."
                )
            for index, item in zip(judge_indexes, parsed_items):
                raw_items[index] = item
                raw_responses[index] = raw

        criteria_list: List[Dict[str, float]] = []
        for item, gate in zip(raw_items, hard_gates):
            if gate["forced_criteria"] is not None:
                criteria_list.append(dict(gate["forced_criteria"]))
            else:
                criteria_list.append(self._coerce_criteria_scores(item))

        raw_unclamped_scores: List[float] = []
        pre_normalize_scores: List[float] = []
        no_solution_reveals: List[bool] = []
        no_solution_multipliers: List[float] = []
        assessments: List[Dict[str, Any]] = []
        zero_criteria = {key: 0.0 for key in self._weights()}
        for index, (criteria, item, corruption, features, gate) in enumerate(zip(criteria_list, raw_items, corruption_flags, quality_features, hard_gates)):
            zero_out = bool(corruption["is_corrupted"] or features["severe_hint_failure"])
            forced_score = gate.get("forced_score")
            if zero_out and forced_score is None:
                criteria = dict(zero_criteria)
                criteria_list[index] = criteria
            no_solution_reveal = self._coerce_no_solution_reveal(item, gate)
            no_solution_multiplier = 1.0 if no_solution_reveal else 0.1
            if forced_score is not None:
                raw_unclamped = max(0.0, float(forced_score))
                pre_normalize = raw_unclamped * no_solution_multiplier
            else:
                base_score = self._weighted_score(criteria)
                if zero_out:
                    raw_unclamped = 0.0
                else:
                    delta = float(features["delta"])
                    raw_unclamped = max(0.0, base_score + delta)
                pre_normalize = raw_unclamped * no_solution_multiplier
            assessment = self._task_and_hint_assessment(
                item,
                score=pre_normalize,
                severe_hint_failure=bool(features["severe_hint_failure"]),
                corruption_detected=bool(corruption["is_corrupted"]),
                hard_gate=gate,
            )
            features["hard_gate"] = {
                "applied": bool(gate["reasons"]),
                "reasons": list(gate["reasons"]),
                "forced_score": gate["forced_score"],
                "contains_code_output": gate["contains_code_output"],
                "contains_direct_fix": gate["contains_direct_fix"],
            }
            raw_unclamped_scores.append(raw_unclamped)
            pre_normalize_scores.append(pre_normalize)
            no_solution_reveals.append(no_solution_reveal)
            no_solution_multipliers.append(no_solution_multiplier)
            assessments.append(assessment)

        post_normalize_scores = self._normalize_post_scores(pre_normalize_scores)
        adjusted_scores = self._apply_batch_spread(post_normalize_scores) if apply_batch_spread else list(post_normalize_scores)
        for index, (corruption, features, assessment, gate) in enumerate(zip(corruption_flags, quality_features, assessments, hard_gates)):
            if corruption["is_corrupted"] or features["severe_hint_failure"] or gate.get("forced_score") == 0.0:
                raw_unclamped_scores[index] = 0.0
                pre_normalize_scores[index] = 0.0
                post_normalize_scores[index] = 0.0
                adjusted_scores[index] = 0.0
                assessment["hint_is_valid_for_socratic"] = False

        return [
            {
                "criteria_scores": criteria,
                "raw_score": pre_normalize,
                "raw_unclamped": raw_unclamped,
                "pre_normalize": pre_normalize,
                "post_normalize": post_normalize,
                "adjusted_score": adjusted_score,
                "no_solution_reveal": no_solution_reveal,
                "no_solution_reveal_multiplier": no_solution_multiplier,
                "raw_response": raw_response,
                "task_quality": assessment["task_quality"],
                "task_is_valid_for_socratic": assessment["task_is_valid_for_socratic"],
                "hint_is_valid_for_socratic": assessment["hint_is_valid_for_socratic"],
                "use_for_socratic": assessment["hint_is_valid_for_socratic"],
                "red_rejection_reason": assessment["red_rejection_reason"],
                "hint_rejection_reason": assessment["hint_rejection_reason"],
                "hint_corruption": corruption,
                "local_tiebreak": features,
            }
            for criteria, raw_unclamped, pre_normalize, post_normalize, adjusted_score, no_solution_reveal, no_solution_multiplier, raw_response, assessment, corruption, features in zip(
                criteria_list,
                raw_unclamped_scores,
                pre_normalize_scores,
                post_normalize_scores,
                adjusted_scores,
                no_solution_reveals,
                no_solution_multipliers,
                raw_responses,
                assessments,
                corruption_flags,
                quality_features,
            )
        ]

    def score_pairs(
        self,
        prompt_texts: List[str],
        completions: List[str],
        *,
        apply_batch_spread: bool = True,
    ) -> List[float]:
        details = self.score_pair_details(
            prompt_texts,
            completions,
            apply_batch_spread=apply_batch_spread,
        )
        return [float(item["adjusted_score"]) for item in details]

    def _output_from_details(self, task: PythonTask, hint: SocraticHint, details: Dict[str, Any]) -> JudgeOutput:
        pre_normalize = float(details["pre_normalize"])
        post_normalize = float(details["post_normalize"])
        adjusted_score = float(details["adjusted_score"])
        scored_text = _hint_text_for_judge(hint)
        scored_text_source = "raw_text" if str(getattr(hint, "raw_text", "") or "").strip() else "text"
        return JudgeOutput(
            task_id=task.task_id,
            score=pre_normalize,
            normalized_reward=post_normalize / 10.0,
            raw_text=str(details["raw_response"]),
            criteria_scores=dict(details["criteria_scores"]),
            metadata={
                "topic": task.topic,
                "raw_score": pre_normalize,
                "raw_unclamped": float(details["raw_unclamped"]),
                "pre_normalize": pre_normalize,
                "post_normalize": post_normalize,
                "adjusted_score": adjusted_score,
                "no_solution_reveal": bool(details["no_solution_reveal"]),
                "no_solution_reveal_multiplier": float(details["no_solution_reveal_multiplier"]),
                "task_quality": float(details["task_quality"]),
                "task_is_valid_for_socratic": bool(details["task_is_valid_for_socratic"]),
                "hint_is_valid_for_socratic": bool(details["hint_is_valid_for_socratic"]),
                "use_for_socratic": bool(details["use_for_socratic"]),
                "red_rejection_reason": str(details["red_rejection_reason"]),
                "hint_rejection_reason": str(details["hint_rejection_reason"]),
                "hint_corruption": dict(details["hint_corruption"]),
                "hint_is_corrupted": bool(details["hint_corruption"].get("is_corrupted")),
                "hint_clean_text": str(hint.metadata.get("sanitized_text") or hint.text),
                "hint_scored_text_source": scored_text_source,
                "hint_scored_text": scored_text,
                "local_tiebreak": dict(details["local_tiebreak"]),
            },
        )

    def rank_hint_candidates(
        self,
        tasks: List[PythonTask],
        hint_groups: List[List[SocraticHint]],
        *,
        apply_group_spread: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        if len(tasks) != len(hint_groups):
            raise ValueError("tasks and hint_groups must have the same length")
        if not tasks:
            return []

        flat_tasks: List[PythonTask] = []
        flat_hints: List[SocraticHint] = []
        group_sizes: List[int] = []
        for task, hints in zip(tasks, hint_groups):
            usable_hints = list(hints)
            group_sizes.append(len(usable_hints))
            for hint in usable_hints:
                flat_tasks.append(task)
                flat_hints.append(hint)

        if not flat_hints:
            return [[] for _ in tasks]

        prompt_texts = [build_socratic_messages(task)[-1]["content"] for task in flat_tasks]
        hint_texts = [_hint_text_for_judge(hint) for hint in flat_hints]
        details_list = self.score_pair_details(
            prompt_texts,
            hint_texts,
            apply_batch_spread=False,
            tasks=flat_tasks,
        )

        ranked_groups: List[List[Dict[str, Any]]] = []
        offset = 0
        for task, group_size in zip(tasks, group_sizes):
            group_hints = flat_hints[offset : offset + group_size]
            group_details = [dict(item) for item in details_list[offset : offset + group_size]]
            offset += group_size

            if apply_group_spread and len(group_details) > 1:
                adjusted_scores = self._apply_batch_spread([float(item["post_normalize"]) for item in group_details])
                for details, adjusted_score in zip(group_details, adjusted_scores):
                    details["adjusted_score"] = adjusted_score

            ranked: List[Dict[str, Any]] = []
            for candidate_index, (hint, details) in enumerate(zip(group_hints, group_details)):
                judge = self._output_from_details(task, hint, details)
                judge.metadata["candidate_index"] = int(hint.metadata.get("candidate_index", candidate_index))
                ranked.append(
                    {
                        "hint": hint,
                        "judge": judge,
                        "candidate_index": int(hint.metadata.get("candidate_index", candidate_index)),
                    }
                )

            ranked.sort(
                key=lambda item: (
                    bool(item["judge"].metadata.get("task_is_valid_for_socratic", True)),
                    bool(item["judge"].metadata.get("hint_is_valid_for_socratic", True)),
                    float(item["judge"].metadata.get("adjusted_score") or 0.0),
                    float(item["judge"].metadata.get("post_normalize") or 0.0),
                ),
                reverse=True,
            )
            for rank, item in enumerate(ranked):
                item["rank"] = rank
                item["judge"].metadata["candidate_rank"] = rank
            ranked_groups.append(ranked)

        return ranked_groups

    def evaluate(self, task: PythonTask, hint_text: str) -> JudgeOutput:
        hint = SocraticHint(task_id=task.task_id, text=hint_text, raw_text=hint_text)
        return self.evaluate_batch([task], [hint], apply_batch_spread=False)[0]

    def evaluate_batch(
        self,
        tasks: List[PythonTask],
        hints: List[SocraticHint],
        *,
        apply_batch_spread: bool,
    ) -> List[JudgeOutput]:
        if not tasks:
            return []
        prompt_texts = [build_socratic_messages(task)[-1]["content"] for task in tasks]
        hint_texts = [_hint_text_for_judge(hint) for hint in hints]
        details_list = self.score_pair_details(
            prompt_texts,
            hint_texts,
            apply_batch_spread=apply_batch_spread,
            tasks=tasks,
        )
        outputs: List[JudgeOutput] = []
        for task, hint, details in zip(tasks, hints, details_list):
            judge = self._output_from_details(task, hint, details)
            self.logger.debug_dump("judge_eval", task=task, judge=judge)
            outputs.append(judge)
        return outputs
