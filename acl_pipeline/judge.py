from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .logging_utils import StructuredLogger
from .modeling import ModelPool
from .prompts import build_judge_batch_messages, build_socratic_messages
from .schemas import JudgeOutput, PythonTask


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

    def score_pairs(self, prompt_texts: List[str], completions: List[str]) -> List[float]:
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
        messages = build_judge_batch_messages(rows)
        raw = session.generate([messages])[0]
        parsed = _extract_json(raw)
        scores: List[float] = []
        if isinstance(parsed, list):
            for item in parsed:
                try:
                    scores.append(float(item))
                except Exception:
                    scores.append(0.0)
        if len(scores) != len(rows):
            scores = [0.0] * len(rows)
        return [max(0.0, min(10.0, value)) for value in scores]

    def evaluate(self, task: PythonTask, hint_text: str) -> JudgeOutput:
        prompt_text = build_socratic_messages(task)[-1]["content"]
        raw_score = self.score_pairs([prompt_text], [hint_text])[0]
        judge = JudgeOutput(
            task_id=task.task_id,
            score=raw_score,
            normalized_reward=raw_score / 10.0,
            raw_text=str(raw_score),
            metadata={"topic": task.topic},
        )
        self.logger.debug_dump("judge_eval", task=task, judge=judge)
        return judge
