from __future__ import annotations

import builtins
import json
import keyword
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .logging_utils import StructuredLogger
from .modeling import ModelPool
from .prompts import build_judge_batch_messages, build_socratic_messages
from .schemas import JudgeOutput, PythonTask, SocraticHint
from .text_quality import detect_corrupted_hint_text


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_FUNC_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_ERROR_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b")
_CODE_BLOCK_RE = re.compile(r"## Code\n```python\n(.*?)\n```", re.S)
_ERROR_BLOCK_RE = re.compile(r"## Error\n```text\n(.*?)\n```", re.S)
_IGNORED_IDENTIFIERS = set(keyword.kwlist) | set(dir(builtins)) | {
    "python",
    "task",
    "code",
    "error",
    "debugging",
    "debug",
    "student",
    "assistant",
    "hint",
    "question",
    "value",
    "values",
    "result",
    "results",
    "issue",
    "problem",
    "logic",
    "function",
    "variable",
    "branch",
    "state",
    "print",
    "repr",
    "type",
}
_GENERIC_HINT_PHRASES = (
    "what do you expect",
    "trace through",
    "step by step",
    "small experiment",
    "what do you notice",
    "what happens if",
    "try printing",
)
_COMPILE_ERROR_NAMES = {"SyntaxError", "IndentationError", "TabError"}
_COMPILE_AWARE_HINT_TOKENS = (
    "syntaxerror",
    "indentationerror",
    "taberror",
    "does not compile",
    "doesn't compile",
    "compile",
    "compiles",
    "parser",
    "parsing",
    "syntax",
    "indent",
    "indentation",
    "unexpected indent",
    "invalid syntax",
    "expected an indented block",
    "line ",
    "colon",
)
_PASSED_EXECUTION_TOKENS = (
    "no failing assertion",
    "no runtime error",
    "program exited successfully",
    "code already runs",
    "code already passes",
    "the code passes",
    "the asserts pass",
    "no reproduced error",
    "it passes",
)


def _is_compile_failure(error_name: str, observed_failure: str) -> bool:
    if error_name in _COMPILE_ERROR_NAMES:
        return True
    lowered = str(observed_failure or "").lower()
    return any(
        token in lowered
        for token in (
            "syntaxerror",
            "indentationerror",
            "taberror",
            "invalid syntax",
            "unexpected indent",
            "expected an indented block",
            "unindent does not match",
            "can't compile",
            "does not compile",
        )
    )


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


def _extract_identifiers(text: str) -> List[str]:
    return [match.group(0) for match in _IDENTIFIER_RE.finditer(str(text or ""))]


def _normalized_identifier_set(text: str) -> set[str]:
    return {token for token in _extract_identifiers(text) if token and token not in _IGNORED_IDENTIFIERS}


def _first_error_name(text: str) -> str:
    match = _ERROR_NAME_RE.search(str(text or ""))
    return match.group(1) if match else ""


def _slice_assert_context(asserts: Sequence[str]) -> str:
    cleaned = [str(item).strip() for item in asserts if str(item).strip()]
    return "\n".join(cleaned[:3])


def _prompt_context(prompt_text: str) -> Dict[str, Any]:
    prompt = str(prompt_text or "")
    code_match = _CODE_BLOCK_RE.search(prompt)
    error_match = _ERROR_BLOCK_RE.search(prompt)
    code = code_match.group(1) if code_match else prompt
    error_text = error_match.group(1) if error_match else prompt
    execution_status = "passed" if "No failing assertion or runtime error was reproduced." in prompt else "failed"
    return {
        "allowed_identifiers": _normalized_identifier_set(code + "\n" + error_text),
        "target_function": "",
        "assert_context": "",
        "error_name": _first_error_name(error_text),
        "execution_status": execution_status,
        "observed_failure": error_text,
        "is_compile_failure": _is_compile_failure(_first_error_name(error_text), error_text),
    }


def _task_context(task: PythonTask) -> Dict[str, Any]:
    code = task.combined_program()
    error_text = task.observed_failure()
    spec = dict(task.metadata.get("red_spec") or {})
    target_function = str(spec.get("target_function") or "").strip()
    allowed = _normalized_identifier_set(code + "\n" + error_text + "\n" + _slice_assert_context(task.failing_asserts))
    if target_function:
        allowed.add(target_function)
    return {
        "allowed_identifiers": allowed,
        "target_function": target_function,
        "assert_context": _slice_assert_context(task.failing_asserts),
        "error_name": _first_error_name(error_text),
        "execution_status": str(task.metadata.get("execution_status") or "failed"),
        "observed_failure": error_text,
        "is_compile_failure": _is_compile_failure(_first_error_name(error_text), error_text),
    }


def _hallucinated_identifiers(hint_text: str, allowed_identifiers: set[str]) -> List[str]:
    candidates: set[str] = set()
    for snippet in _BACKTICK_RE.findall(hint_text):
        candidates.update(_normalized_identifier_set(snippet))
    for token in _FUNC_CALL_RE.findall(hint_text):
        if token:
            candidates.add(token)
    for token in _normalized_identifier_set(hint_text):
        if "." in token:
            candidates.update(_normalized_identifier_set(token))

    bad: List[str] = []
    for token in sorted(candidates):
        if token in allowed_identifiers or token in _IGNORED_IDENTIFIERS or len(token) <= 1:
            continue
        bad.append(token)
    return bad


def _local_hint_quality(hint_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    raw = str(hint_text or "")
    lowered = raw.lower()
    allowed_identifiers = set(context.get("allowed_identifiers") or set())
    mentioned_identifiers = [token for token in _normalized_identifier_set(raw) if token in allowed_identifiers]
    hallucinated = _hallucinated_identifiers(raw, allowed_identifiers)
    reasons: List[str] = []
    delta = 0.0

    if raw.count("`") % 2 == 1:
        reasons.append("unbalanced_backticks")
        delta -= 1.1

    target_function = str(context.get("target_function") or "")
    if target_function and target_function in raw:
        delta += 0.45

    error_name = str(context.get("error_name") or "")
    if error_name and error_name in raw:
        delta += 0.35

    assert_context = str(context.get("assert_context") or "")
    assert_hits = 0
    if assert_context:
        for token in _normalized_identifier_set(assert_context):
            if token in _IGNORED_IDENTIFIERS:
                continue
            if re.search(rf"\b{re.escape(token)}\b", raw):
                assert_hits += 1
        if assert_hits:
            delta += min(0.55, 0.14 * assert_hits)

    if mentioned_identifiers:
        delta += min(0.55, 0.12 * len(set(mentioned_identifiers)))

    generic_hits = sum(1 for phrase in _GENERIC_HINT_PHRASES if phrase in lowered)
    if generic_hits and len(set(mentioned_identifiers)) < 2:
        reasons.append("generic_hint")
        delta -= min(0.8, 0.25 * generic_hits)

    if hallucinated:
        reasons.append("hallucinated_identifiers")
        delta -= min(2.6, 1.0 * len(hallucinated))

    execution_status = str(context.get("execution_status") or "").strip().lower()
    if execution_status == "passed":
        if any(token in lowered for token in _PASSED_EXECUTION_TOKENS):
            delta += 0.35
        else:
            reasons.append("missed_passed_execution")
            delta -= 3.0

    if bool(context.get("is_compile_failure")):
        if error_name and error_name.lower() in lowered:
            delta += 0.25
        if any(token in lowered for token in _COMPILE_AWARE_HINT_TOKENS):
            delta += 0.45
        else:
            reasons.append("missed_compile_failure")
            delta -= 2.4

    severe = False
    if execution_status == "passed" and "missed_passed_execution" in reasons:
        severe = True
    if "hallucinated_identifiers" in reasons and "unbalanced_backticks" in reasons:
        severe = True

    hint_is_valid = not severe
    return {
        "delta": delta,
        "severe": severe,
        "hint_is_valid": hint_is_valid,
        "reasons": list(dict.fromkeys(reasons)),
        "hallucinated_identifiers": hallucinated,
        "mentioned_identifiers": sorted(set(mentioned_identifiers)),
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

    def _assessment(self, item: Any) -> Dict[str, Any]:
        threshold = float(self.model_pool.config.judge.bad_task_threshold)
        if not isinstance(item, dict):
            return {
                "task_quality": 5.0,
                "task_is_valid_for_socratic": True,
                "task_rejection_reason": "",
                "hint_is_valid_for_socratic": True,
                "hint_rejection_reason": "",
            }
        try:
            task_quality = max(0.0, min(10.0, float(item.get("task_quality", 5.0))))
        except Exception:
            task_quality = 5.0

        task_explicit = item.get("task_is_valid_for_socratic")
        if task_explicit is None:
            task_is_valid = task_quality > threshold
        else:
            task_is_valid = bool(task_explicit)

        hint_explicit = item.get("hint_is_valid_for_socratic")
        legacy_use = item.get("use_for_socratic")
        if hint_explicit is None:
            hint_is_valid = bool(legacy_use) if legacy_use is not None else True
        else:
            hint_is_valid = bool(hint_explicit)

        task_rejection_reason = str(
            item.get("task_rejection_reason")
            or item.get("red_rejection_reason")
            or ""
        ).strip()
        hint_rejection_reason = str(item.get("hint_rejection_reason") or "").strip()

        return {
            "task_quality": task_quality,
            "task_is_valid_for_socratic": task_is_valid,
            "task_rejection_reason": task_rejection_reason,
            "hint_is_valid_for_socratic": hint_is_valid,
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

    def _query_judge_session(self, rows: List[Dict[str, str]], *, session) -> Tuple[List[Any], str]:
        messages = build_judge_batch_messages(rows, self._weights())
        raw = session.generate([messages])[0]
        parsed = _extract_json(raw)
        raw_items: List[Any] = []
        if isinstance(parsed, list):
            raw_items = list(parsed)
        elif isinstance(parsed, dict):
            maybe_items = parsed.get("items") or parsed.get("scores") or parsed.get("results")
            if isinstance(maybe_items, list):
                raw_items = list(maybe_items)
        if len(raw_items) != len(rows):
            raw_items = [0.0] * len(rows)
        return raw_items, raw

    def _query_rows(self, rows: List[Dict[str, str]]) -> Tuple[List[Any], List[str]]:
        if not rows:
            return [], []
        session = self.model_pool.get_judge()
        items, raw = self._query_judge_session(rows, session=session)
        return items, [raw] * len(rows)

    def _score_rows(
        self,
        *,
        rows: List[Dict[str, str]],
        completions: List[str],
        contexts: List[Dict[str, Any]],
        apply_batch_spread: bool,
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []

        raw_items, raw_texts = self._query_rows(rows)
        criteria_list = [self._coerce_criteria_scores(item) for item in raw_items]
        assessments = [self._assessment(item) for item in raw_items]
        corruption_flags = [detect_corrupted_hint_text(text) for text in completions]
        local_quality = [_local_hint_quality(text, context) for text, context in zip(completions, contexts)]

        model_raw_scores = [self._weighted_score(criteria) for criteria in criteria_list]
        local_scores: List[float] = []
        for index, (criteria, assessment, corruption, quality) in enumerate(
            zip(criteria_list, assessments, corruption_flags, local_quality)
        ):
            if corruption["is_corrupted"]:
                criteria_list[index] = {key: 0.0 for key in self._weights()}
                assessment["hint_is_valid_for_socratic"] = False
                assessment["hint_rejection_reason"] = "assistant output is corrupted and contains gibberish"
                local_scores.append(0.0)
                continue

            if quality["severe"]:
                criteria_list[index] = {key: 0.0 for key in self._weights()}
                assessment["hint_is_valid_for_socratic"] = False
                if not assessment["hint_rejection_reason"]:
                    assessment["hint_rejection_reason"] = ", ".join(quality["reasons"]) or "severe_malformed_hint"
                local_scores.append(0.0)
                continue

            if not assessment["task_is_valid_for_socratic"]:
                local_scores.append(0.0)
                continue

            score = max(0.0, min(10.0, model_raw_scores[index] + float(quality["delta"])))
            if not assessment["hint_is_valid_for_socratic"]:
                score = min(score, 2.5)
            local_scores.append(score)

            if not assessment["hint_rejection_reason"] and quality["reasons"] and not quality["hint_is_valid"]:
                assessment["hint_rejection_reason"] = ", ".join(quality["reasons"])

        adjusted_scores = self._apply_batch_spread(local_scores) if apply_batch_spread else list(local_scores)
        for index, assessment in enumerate(assessments):
            if not assessment["task_is_valid_for_socratic"] or corruption_flags[index]["is_corrupted"] or local_quality[index]["severe"]:
                adjusted_scores[index] = 0.0

        return [
            {
                "criteria_scores": criteria,
                "model_raw_score": model_raw,
                "raw_score": raw_score,
                "adjusted_score": adjusted_score,
                "raw_response": raw_response,
                "task_quality": assessment["task_quality"],
                "task_is_valid_for_socratic": assessment["task_is_valid_for_socratic"],
                "task_rejection_reason": assessment["task_rejection_reason"],
                "hint_is_valid_for_socratic": assessment["hint_is_valid_for_socratic"],
                "hint_rejection_reason": assessment["hint_rejection_reason"],
                "hint_corruption": corruption,
                "hint_quality": quality,
            }
            for criteria, model_raw, raw_score, adjusted_score, raw_response, assessment, corruption, quality in zip(
                criteria_list,
                model_raw_scores,
                local_scores,
                adjusted_scores,
                raw_texts,
                assessments,
                corruption_flags,
                local_quality,
            )
        ]

    def score_pair_details(
        self,
        prompt_texts: List[str],
        completions: List[str],
        *,
        apply_batch_spread: bool,
    ) -> List[Dict[str, Any]]:
        rows = [
            {
                "student_prompt": prompt[:2200],
                "assistant_response": completion[:1800],
            }
            for prompt, completion in zip(prompt_texts, completions)
        ]
        contexts = [_prompt_context(prompt) for prompt in prompt_texts]
        return self._score_rows(
            rows=rows,
            completions=completions,
            contexts=contexts,
            apply_batch_spread=apply_batch_spread,
        )

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

        rows = [
            {
                "topic": task.topic,
                "statement": task.statement[:600],
                "buggy_code": task.combined_program()[:2200],
                "observed_failure": task.observed_failure()[:1200],
                "execution_status": str(task.metadata.get("execution_status") or "failed"),
                "assistant_response": (hint.raw_text or hint.text)[:1800],
            }
            for task, hint in zip(tasks, hints)
        ]
        details_list = self._score_rows(
            rows=rows,
            completions=[hint.raw_text or hint.text for hint in hints],
            contexts=[_task_context(task) for task in tasks],
            apply_batch_spread=apply_batch_spread,
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
                    "model_raw_score": float(details["model_raw_score"]),
                    "raw_score": raw_score,
                    "adjusted_score": adjusted_score,
                    "task_quality": float(details["task_quality"]),
                    "task_is_valid_for_socratic": bool(details["task_is_valid_for_socratic"]),
                    "task_rejection_reason": str(details["task_rejection_reason"]),
                    "hint_is_valid_for_socratic": bool(details["hint_is_valid_for_socratic"]),
                    "hint_rejection_reason": str(details["hint_rejection_reason"]),
                    "hint_corruption": dict(details["hint_corruption"]),
                    "hint_is_corrupted": bool(details["hint_corruption"].get("is_corrupted")),
                    "hint_quality": dict(details["hint_quality"]),
                    "hint_clean_text": hint.text,
                },
            )
            self.logger.debug_dump("judge_eval", task=task, judge=judge)
            outputs.append(judge)
        return outputs
