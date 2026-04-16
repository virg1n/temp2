from __future__ import annotations

import logging

from environment_engine import EnvironmentEngine
from logging_utils import get_logger, log_event
from models_factory import extract_json
from prompts import build_hint_assessment_messages, render_hint_batch
from schemas import HintCandidate, HintCriterionScores, HintEvaluation, PipelineSettings, dataclass_to_dict
from storage import StorageManager


LOGGER = get_logger(__name__)


def _clamp_score(value: object) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


class HintJudgeAssessor:
    def __init__(self, environment: EnvironmentEngine, settings: PipelineSettings, storage: StorageManager | None = None) -> None:
        self.environment = environment
        self.settings = settings
        self.storage = storage

    def _score_source(self, payload: dict[str, object]) -> dict[str, object]:
        nested_scores = payload.get("scores")
        if isinstance(nested_scores, dict):
            merged = dict(nested_scores)
            merged.update(payload)
            return merged
        return payload

    def _get_score(self, payload: dict[str, object], *keys: str, fallback: float = 0.0) -> float:
        for key in keys:
            if key in payload:
                return _clamp_score(payload.get(key))
        return fallback

    def _combine_scores(self, payload: dict[str, object], hint_text: str) -> HintCriterionScores:
        weights = self.settings.judge.reward_weights
        source = self._score_source(payload)
        scalar_fallback = self._get_score(source, "overall_score", "score", fallback=0.0)
        no_solution_reveal = self._get_score(
            source,
            "no_solution_reveal",
            "solution_reveal_score",
            "non_reveal",
            fallback=scalar_fallback,
        )
        bug_localization = self._get_score(
            source,
            "bug_localization",
            "bug_location",
            "bug_identification",
            fallback=scalar_fallback,
        )
        usefulness = self._get_score(source, "usefulness", "helpfulness", fallback=scalar_fallback)
        socratic_style = self._get_score(source, "socratic_style", "style", fallback=scalar_fallback)
        technical_accuracy = self._get_score(
            source,
            "technical_accuracy",
            "accuracy",
            fallback=scalar_fallback,
        )

        weighted = (
            no_solution_reveal * weights.no_solution_reveal
            + bug_localization * weights.bug_localization
            + usefulness * weights.usefulness
            + socratic_style * weights.socratic_style
            + technical_accuracy * weights.technical_accuracy
        )
        penalty = 1.0
        lowered = hint_text.lower()
        if "```" in hint_text:
            penalty *= 0.3
        if "<think>" in lowered or "</think>" in lowered:
            penalty *= 0.0

        final_reward = max(0.0, min(1.0, weighted * penalty))
        return HintCriterionScores(
            no_solution_reveal=no_solution_reveal,
            bug_localization=bug_localization,
            usefulness=usefulness,
            socratic_style=socratic_style,
            technical_accuracy=technical_accuracy,
            final_reward=final_reward,
        )

    def score_prompt_completion_pairs(
        self,
        items: list[tuple[str, str, str]],
        *,
        context: str = "default",
        round_index: int | None = None,
    ) -> dict[str, HintEvaluation]:
        if not items:
            return {}
        judge = self.environment.load_judge()
        evaluations: dict[str, HintEvaluation] = {}
        batch_size = self.settings.judge.hint_assessment_batch_size
        total_batches = (len(items) + batch_size - 1) // batch_size

        log_event(
            LOGGER,
            logging.INFO,
            "hint_assessment_started",
            "Starting hint assessment",
            total_items=len(items),
            batch_size=batch_size,
            total_batches=total_batches,
        )

        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            batch_index = (start // batch_size) + 1
            log_event(
                LOGGER,
                logging.INFO,
                "hint_assessment_batch_started",
                "Starting hint assessment batch",
                batch_index=batch_index,
                total_batches=total_batches,
                batch_items=len(batch),
            )
            messages = build_hint_assessment_messages(batch)
            raw = judge.generate(messages, generation=self.settings.models.judge.generation, num_return_sequences=1)
            raw_text = raw[0] if raw else ""
            payload = extract_json(raw_text)
            verdicts = []
            if isinstance(payload, list):
                verdicts = payload
            elif isinstance(payload, dict):
                verdicts = payload.get("items") or payload.get("results") or payload.get("scores") or []
            verdict_map: dict[str, dict[str, object]] = {}

            if isinstance(verdicts, list):
                for index, item in enumerate(verdicts):
                    if isinstance(item, (int, float)) and len(verdicts) == len(batch):
                        score = _clamp_score(item)
                        verdict_map[batch[index][0]] = {
                            "item_id": batch[index][0],
                            "no_solution_reveal": score,
                            "bug_localization": score,
                            "usefulness": score,
                            "socratic_style": score,
                            "technical_accuracy": score,
                            "feedback": "Scalar score fallback",
                        }
                        continue
                    if not isinstance(item, dict):
                        continue
                    item_id = item.get("item_id") or item.get("task_id")
                    if item_id is None and len(verdicts) == len(batch):
                        item_id = batch[index][0]
                    if item_id is not None:
                        verdict_map[str(item_id)] = item

            if not verdict_map:
                preview = raw_text[:300].replace("\n", "\\n")
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "hint_assessment_parse_miss",
                    "Judge hint assessment did not return usable item ids",
                    batch_size=len(batch),
                    raw_preview=preview,
                )

            for item_id, prompt_text, hint_text in batch:
                verdict = verdict_map.get(item_id, {})
                scores = self._combine_scores(verdict, hint_text)
                evaluations[item_id] = HintEvaluation(
                    hint_id=item_id,
                    task_id=item_id.split("-hint-")[0] if "-hint-" in item_id else item_id,
                    scores=scores,
                    judge_feedback=str(verdict.get("feedback", "Judge response missing")).strip(),
                    raw_payload={
                        "prompt_text": prompt_text,
                        "hint_text": hint_text,
                        "judge_payload": verdict,
                    },
                )

            if self.storage is not None:
                self.storage.append_event(
                    "hint_assessment_batch_dump",
                    {
                        "context": context,
                        "round_index": round_index,
                        "batch_index": batch_index,
                        "total_batches": total_batches,
                        "raw_judge_response": raw_text,
                        "parsed_judge_response": payload,
                        "items": [
                            {
                                "item_id": item_id,
                                "task_prompt": prompt_text,
                                "hint_text": hint_text,
                                "judge_feedback": evaluations[item_id].judge_feedback,
                                "scores": dataclass_to_dict(evaluations[item_id].scores),
                                "judge_payload": evaluations[item_id].raw_payload.get("judge_payload", {}),
                            }
                            for item_id, prompt_text, hint_text in batch
                            if item_id in evaluations
                        ],
                    },
                )

            log_event(
                LOGGER,
                logging.INFO,
                "hint_assessment_batch_finished",
                "Finished hint assessment batch",
                batch_index=batch_index,
                total_batches=total_batches,
                accumulated_evaluations=len(evaluations),
            )

        log_event(
            LOGGER,
            logging.INFO,
            "hint_assessment_finished",
            "Completed hint assessment",
            evaluated_count=len(evaluations),
        )
        return evaluations

    def assess_hints(self, hints: list[HintCandidate], *, round_index: int | None = None) -> list[HintEvaluation]:
        results = self.score_prompt_completion_pairs(
            render_hint_batch(hints),
            context="pipeline_hint_assessment",
            round_index=round_index,
        )
        return [results[hint.hint_id] for hint in hints if hint.hint_id in results]
