from __future__ import annotations

import builtins
import json
import keyword
import re
from typing import Any, Dict, Iterable, List, Optional, Set

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


def _build_row(prompt_text: str, completion: str) -> Dict[str, Any]:
    sections = _extract_prompt_sections(prompt_text)
    task_match = _TASK_SECTION_RE.search(str(prompt_text or ""))
    error_text = sections.get("error", "")
    execution_status = _infer_execution_status(error_text)
    return {
        "statement": task_match.group("body").strip() if task_match else sections.get("task", ""),
        "code": sections.get("code", ""),
        "observed_failure": error_text,
        "execution_status": execution_status,
        "assistant_response": completion[:1800],
    }


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
    if assert_token_set & hint_identifiers:
        grounding_hits.append("assertion_token")
    if error_tokens & hint_identifiers:
        grounding_hits.append("error_token")

    generic_hits = sum(1 for pattern in _GENERIC_HINT_PATTERNS if pattern.search(hint_text))
    references_passed_execution = any(pattern.search(hint_text) for pattern in _PASS_PATTERNS)

    malformed_reasons = _hint_malformed_reasons(hint_text)

    delta = 0.0
    reasons: List[str] = []
    if grounding_hits:
        delta += min(1.25, 0.45 * len(grounding_hits))
        reasons.extend(f"grounded:{name}" for name in grounding_hits)
        if "?" in hint_text:
            delta += 0.25
            reasons.append("precise_question")
    if generic_hits:
        generic_penalty = 0.35 * generic_hits
        if not grounding_hits:
            generic_penalty += 0.55
        delta -= min(1.25, generic_penalty)
        reasons.append(f"generic:{generic_hits}")
    if invented_identifiers:
        delta -= min(1.25, 0.25 * len(invented_identifiers))
        reasons.append(f"invented_identifiers:{len(invented_identifiers)}")
    if invented_references:
        delta -= min(2.0, 0.8 * len(invented_references))
        reasons.append(f"invented_references:{len(invented_references)}")
    if malformed_reasons:
        delta -= min(2.0, 0.8 * len(malformed_reasons))
        reasons.extend(malformed_reasons)

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
            delta -= 2.25
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

    def _weights(self) -> Dict[str, float]:
        return dict(self.model_pool.config.judge.reward_weights)

    def _coerce_criteria_scores(self, item: Any) -> Dict[str, float]:
        weights = self._weights()
        if isinstance(item, dict):
            return {
                key: max(0.0, min(10.0, float(item.get(key, 0.0))))
                for key in weights
            }
        try:
            value = max(0.0, min(10.0, float(item)))
        except Exception:
            value = 0.0
        return {key: value for key in weights}

    def _weighted_score(self, criteria_scores: Dict[str, float]) -> float:
        weights = self._weights()
        total_weight = sum(max(0.0, float(value)) for value in weights.values())
        if total_weight <= 0:
            return 0.0
        total = 0.0
        for key, weight in weights.items():
            total += float(criteria_scores.get(key, 0.0)) * float(weight)
        return max(0.0, min(10.0, total / total_weight))

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
            gate["skip_llm"] = True
            gate["forced_criteria"] = dict(zero_criteria)
            gate["forced_score"] = 0.0
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

        if row.get("execution_status") == "passed" and not pass_aware and gate["forced_score"] is None:
            gate["skip_llm"] = True
            gate["forced_criteria"] = {
                "no_solution_reveal": 10.0,
                "bug_localization": 0.0,
                "usefulness": 0.0,
                "socratic_style": 1.0 if "?" in hint_text else 0.0,
                "technical_accuracy": 0.0,
            }
            gate["forced_score"] = _PASSED_EXECUTION_REWARD
            gate["force_hint_valid"] = False
            gate["hint_rejection_reason"] = "missed_passed_execution"
            gate["reasons"].append("hint_capped:missed_passed_execution")

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
            adjusted.append(max(0.0, min(10.0, spread_score)))
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

        rows = [_build_row(prompt, completion) for prompt, completion in zip(prompt_texts, completions)]
        task_items: List[Optional[PythonTask]] = list(tasks or [])
        if len(task_items) < len(rows):
            task_items.extend([None] * (len(rows) - len(task_items)))
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
            messages = build_judge_batch_messages(judge_rows, self._weights())
            raw = session.generate([messages])[0]
            parsed = _extract_json(raw)
            parsed_items: List[Any] = []
            if isinstance(parsed, list):
                parsed_items = list(parsed)
            elif isinstance(parsed, dict):
                maybe_items = parsed.get("items") or parsed.get("scores") or parsed.get("results")
                if isinstance(maybe_items, list):
                    parsed_items = list(maybe_items)
            if len(parsed_items) != len(judge_rows):
                parsed_items = [{} for _ in judge_rows]
            for index, item in zip(judge_indexes, parsed_items):
                raw_items[index] = item
                raw_responses[index] = raw

        criteria_list: List[Dict[str, float]] = []
        for item, gate in zip(raw_items, hard_gates):
            if gate["forced_criteria"] is not None:
                criteria_list.append(dict(gate["forced_criteria"]))
            else:
                criteria_list.append(self._coerce_criteria_scores(item))

        raw_scores: List[float] = []
        assessments: List[Dict[str, Any]] = []
        zero_criteria = {key: 0.0 for key in self._weights()}
        for index, (criteria, item, corruption, features, gate) in enumerate(zip(criteria_list, raw_items, corruption_flags, quality_features, hard_gates)):
            zero_out = bool(corruption["is_corrupted"] or features["severe_hint_failure"])
            forced_score = gate.get("forced_score")
            if zero_out and forced_score is None:
                criteria = dict(zero_criteria)
                criteria_list[index] = criteria
            if forced_score is not None:
                score = max(0.0, min(10.0, float(forced_score)))
            else:
                base_score = self._weighted_score(criteria)
                score = 0.0 if zero_out else max(0.0, min(10.0, base_score + float(features["delta"])))
            assessment = self._task_and_hint_assessment(
                item,
                score=score,
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
            raw_scores.append(score)
            assessments.append(assessment)

        adjusted_scores = self._apply_batch_spread(raw_scores) if apply_batch_spread else list(raw_scores)
        for index, (corruption, features, assessment, gate) in enumerate(zip(corruption_flags, quality_features, assessments, hard_gates)):
            if corruption["is_corrupted"] or features["severe_hint_failure"] or gate.get("forced_score") == 0.0:
                raw_scores[index] = 0.0
                adjusted_scores[index] = 0.0
                assessment["hint_is_valid_for_socratic"] = False
            elif gate.get("forced_score") is not None:
                cap = max(0.0, min(10.0, float(gate["forced_score"])))
                raw_scores[index] = min(raw_scores[index], cap)
                adjusted_scores[index] = min(adjusted_scores[index], cap)

        return [
            {
                "criteria_scores": criteria,
                "raw_score": raw_score,
                "adjusted_score": adjusted_score,
                "raw_response": raw_response,
                "task_quality": assessment["task_quality"],
                "task_is_valid_for_socratic": assessment["task_is_valid_for_socratic"],
                "hint_is_valid_for_socratic": assessment["hint_is_valid_for_socratic"],
                "use_for_socratic": assessment["task_is_valid_for_socratic"] and assessment["hint_is_valid_for_socratic"],
                "red_rejection_reason": assessment["red_rejection_reason"],
                "hint_rejection_reason": assessment["hint_rejection_reason"],
                "hint_corruption": corruption,
                "local_tiebreak": features,
            }
            for criteria, raw_score, adjusted_score, raw_response, assessment, corruption, features in zip(
                criteria_list,
                raw_scores,
                adjusted_scores,
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
        hint_texts = [hint.raw_text or hint.text for hint in hints]
        details_list = self.score_pair_details(
            prompt_texts,
            hint_texts,
            apply_batch_spread=apply_batch_spread,
            tasks=tasks,
        )
        outputs: List[JudgeOutput] = []
        for task, hint, details in zip(tasks, hints, details_list):
            raw_score = float(details["raw_score"])
            adjusted_score = float(details["adjusted_score"])
            judge = JudgeOutput(
                task_id=task.task_id,
                score=raw_score,
                normalized_reward=adjusted_score / 10.0,
                raw_text=str(details["raw_response"]),
                criteria_scores=dict(details["criteria_scores"]),
                metadata={
                    "topic": task.topic,
                    "raw_score": raw_score,
                    "adjusted_score": adjusted_score,
                    "task_quality": float(details["task_quality"]),
                    "task_is_valid_for_socratic": bool(details["task_is_valid_for_socratic"]),
                    "hint_is_valid_for_socratic": bool(details["hint_is_valid_for_socratic"]),
                    "use_for_socratic": bool(details["use_for_socratic"]),
                    "red_rejection_reason": str(details["red_rejection_reason"]),
                    "hint_rejection_reason": str(details["hint_rejection_reason"]),
                    "hint_corruption": dict(details["hint_corruption"]),
                    "hint_is_corrupted": bool(details["hint_corruption"].get("is_corrupted")),
                    "hint_clean_text": hint.text,
                    "local_tiebreak": dict(details["local_tiebreak"]),
                },
            )
            self.logger.debug_dump("judge_eval", task=task, judge=judge)
            outputs.append(judge)
        return outputs
