from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .logging_utils import StructuredLogger
from .modeling import ModelPool
from .prompts import build_judge_batch_messages, build_socratic_messages
from .schemas import JudgeOutput, PythonTask, SocraticHint
from .text_quality import detect_corrupted_hint_text


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

    def _task_assessment(self, item: Any) -> Dict[str, Any]:
        threshold = float(self.model_pool.config.judge.bad_task_threshold)
        if not isinstance(item, dict):
            return {
                "task_quality": 5.0,
                "use_for_socratic": True,
                "red_rejection_reason": "",
            }
        try:
            task_quality = max(0.0, min(10.0, float(item.get("task_quality", 5.0))))
        except Exception:
            task_quality = 5.0
        explicit = item.get("use_for_socratic")
        if explicit is None:
            use_for_socratic = task_quality > threshold
        else:
            use_for_socratic = bool(explicit)
        return {
            "task_quality": task_quality,
            "use_for_socratic": use_for_socratic,
            "red_rejection_reason": str(item.get("red_rejection_reason") or "").strip(),
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
    ) -> List[Dict[str, Any]]:
        if not prompt_texts:
            return []

        session = self.model_pool.get_judge()
        rows = [
            {
                "student_prompt": prompt[:1800],
                "assistant_response": completion[:1800],
            }
            for prompt, completion in zip(prompt_texts, completions)
        ]
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

        criteria_list = [self._coerce_criteria_scores(item) for item in raw_items]
        assessments = [self._task_assessment(item) for item in raw_items]
        corruption_flags = [detect_corrupted_hint_text(text) for text in completions]
        for index, corruption in enumerate(corruption_flags):
            if corruption["is_corrupted"]:
                criteria_list[index] = {key: 0.0 for key in self._weights()}
        raw_scores = [self._weighted_score(criteria) for criteria in criteria_list]
        adjusted_scores = self._apply_batch_spread(raw_scores) if apply_batch_spread else list(raw_scores)
        for index, corruption in enumerate(corruption_flags):
            if corruption["is_corrupted"]:
                raw_scores[index] = 0.0
                adjusted_scores[index] = 0.0

        return [
            {
                "criteria_scores": criteria,
                "raw_score": raw_score,
                "adjusted_score": adjusted_score,
                "raw_response": raw,
                "task_quality": assessment["task_quality"],
                "use_for_socratic": assessment["use_for_socratic"],
                "red_rejection_reason": assessment["red_rejection_reason"],
                "hint_corruption": corruption,
            }
            for criteria, raw_score, adjusted_score, assessment, corruption in zip(
                criteria_list,
                raw_scores,
                adjusted_scores,
                assessments,
                corruption_flags,
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
                    "use_for_socratic": bool(details["use_for_socratic"]),
                    "red_rejection_reason": str(details["red_rejection_reason"]),
                    "hint_corruption": dict(details["hint_corruption"]),
                    "hint_is_corrupted": bool(details["hint_corruption"].get("is_corrupted")),
                    "hint_clean_text": hint.text,
                },
            )
            self.logger.debug_dump("judge_eval", task=task, judge=judge)
            outputs.append(judge)
        return outputs
