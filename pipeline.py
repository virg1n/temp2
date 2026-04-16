from __future__ import annotations

import logging
import random
from dataclasses import replace

from curriculum import CurriculumManager
from data_buffer import ReplayBuffer, classify_record_label
from environment_engine import EnvironmentEngine
from judge_assess import HintJudgeAssessor
from judge_validation import TaskJudgeValidator
from logging_utils import get_logger, log_event
from red_dpo import RedTrainer
from red_generation import RedTaskGenerator
from schemas import (
    HintEvaluation,
    PipelineSettings,
    RejectedTask,
    ReplayRecord,
    TaskCandidate,
    TaskOutcome,
    ValidatedTask,
    ValidationResult,
    dataclass_to_dict,
)
from socratic_generation import SocraticHintGenerator
from socratic_grpo import SocraticGRPOTrainerWrapper
from storage import StorageManager


LOGGER = get_logger(__name__)


class AdversarialCurriculumPipeline:
    def __init__(self, settings: PipelineSettings, storage: StorageManager) -> None:
        self.settings = settings
        self.storage = storage
        self.state = storage.load_state()
        self.random = random.Random(settings.runtime.seed + self.state.round_index)
        self.environment = EnvironmentEngine(settings, self.state)
        self.curriculum = CurriculumManager(settings.curriculum, self.state.curriculum)
        self.buffer = ReplayBuffer(settings.buffer)
        self.validator = TaskJudgeValidator(self.environment, settings, storage)
        self.assessor = HintJudgeAssessor(self.environment, settings)
        self.red_generator = RedTaskGenerator(self.environment, settings)
        self.socratic_generator = SocraticHintGenerator(self.environment, settings)
        self.socratic_trainer = SocraticGRPOTrainerWrapper(settings, self.assessor)
        self.red_trainer = RedTrainer(settings)
        self.pending_grpo_tasks: list[TaskCandidate] = []
        self.topic_descriptions = {topic.name: topic.description for topic in settings.curriculum.topics}

    def _reset_red_state(self, reason: str) -> None:
        initial_adapter = (
            self.state.red.initial_adapter_path
            or self.settings.models.red.initial_adapter_path
            or self.settings.models.red.adapter_path
        )
        self.state.red = replace(
            self.state.red,
            active_adapter_path=initial_adapter,
            reset_count=self.state.red.reset_count + 1,
            last_reset_reason=reason,
        )
        self.environment.unload_red()
        log_event(
            LOGGER,
            logging.WARNING,
            "red_reset",
            "Red adaptation reset to initial state",
            reason=reason,
            active_adapter_path=self.state.red.active_adapter_path,
            reset_count=self.state.red.reset_count,
        )

    def _handle_repetition_reset(self) -> None:
        self._reset_red_state("repeated_topic_mode_collapse")
        self.curriculum.reset_to_initial()
        self.storage.append_event(
            "mode_collapse_reset",
            {
                "round_index": self.state.round_index,
                "reason": "repeated_topic_mode_collapse",
                "threshold": self.settings.curriculum.repeated_topic_reset_threshold,
            },
        )

    def _merge_validation_results(self, base: ValidationResult, extra: ValidationResult) -> ValidationResult:
        seen_valid_ids = {item.task.task_id for item in base.valid_tasks}
        seen_rejected_ids = {item.task.task_id for item in base.rejected_tasks}
        for item in extra.valid_tasks:
            if item.task.task_id not in seen_valid_ids:
                base.valid_tasks.append(item)
                seen_valid_ids.add(item.task.task_id)
        for item in extra.rejected_tasks:
            if item.task.task_id not in seen_rejected_ids and item.task.task_id not in seen_valid_ids:
                base.rejected_tasks.append(item)
                seen_rejected_ids.add(item.task.task_id)
        return base

    def _generate_valid_tasks(self, topics) -> ValidationResult:
        result = self.validator.validate_candidates(self.red_generator.generate_for_topics(topics))
        if len(result.valid_tasks) >= self.settings.judge.min_valid_tasks:
            return result

        if self.settings.runtime.retry_invalid_generation_once:
            retry_result = self.validator.validate_candidates(self.red_generator.generate_for_topics(topics))
            result = self._merge_validation_results(result, retry_result)
            if len(result.valid_tasks) >= self.settings.judge.min_valid_tasks:
                return result

        self._reset_red_state("insufficient_valid_tasks_after_retry")
        fallback_result = self.validator.validate_candidates(
            self.red_generator.generate_for_topics(topics, use_base_model=True)
        )
        return self._merge_validation_results(result, fallback_result)

    def _build_outcomes(
        self,
        validated_tasks: list[ValidatedTask],
        evaluations: list[HintEvaluation],
    ) -> tuple[list[TaskOutcome], list[ReplayRecord]]:
        by_task: dict[str, list[HintEvaluation]] = {}
        for evaluation in evaluations:
            by_task.setdefault(evaluation.task_id, []).append(evaluation)

        outcomes: list[TaskOutcome] = []
        records: list[ReplayRecord] = []
        for validated in validated_tasks:
            task_evaluations = by_task.get(validated.task.task_id, [])
            rewards = [item.scores.final_reward for item in task_evaluations]
            average_reward = sum(rewards) / len(rewards) if rewards else 0.0
            best_reward = max(rewards) if rewards else 0.0
            label = classify_record_label(
                average_reward,
                failure_threshold=self.settings.buffer.failure_reward_threshold,
                easy_threshold=self.settings.buffer.easy_reward_threshold,
                valid=validated.judge_passed,
            )
            outcome = TaskOutcome(
                task_id=validated.task.task_id,
                topic=validated.task.topic,
                average_reward=average_reward,
                best_reward=best_reward,
                hint_count=len(task_evaluations),
                label=label,
                validation_score=validated.judge_score,
                validation_feedback=validated.judge_feedback,
            )
            outcomes.append(outcome)
            records.append(ReplayRecord(task=validated.task, outcome=outcome, label=label))
        return outcomes, records

    def _build_rejected_records(self, rejected_tasks: list[RejectedTask]) -> list[ReplayRecord]:
        records: list[ReplayRecord] = []
        for rejected in rejected_tasks:
            label = classify_record_label(
                0.0,
                failure_threshold=self.settings.buffer.failure_reward_threshold,
                easy_threshold=self.settings.buffer.easy_reward_threshold,
                valid=False,
            )
            outcome = TaskOutcome(
                task_id=rejected.task.task_id,
                topic=rejected.task.topic,
                average_reward=0.0,
                best_reward=0.0,
                hint_count=0,
                label=label,
                validation_score=rejected.judge_score,
                validation_feedback=rejected.judge_feedback,
            )
            records.append(ReplayRecord(task=rejected.task, outcome=outcome, label=label))
        return records

    def _maybe_train_socratic(self, round_index: int) -> None:
        threshold = self.settings.training.socratic.update_every_tasks
        if len(self.pending_grpo_tasks) < threshold:
            return
        tasks = list(self.pending_grpo_tasks)
        self.state.socratic = self.socratic_trainer.train(tasks, self.state.socratic, round_index)
        self.pending_grpo_tasks.clear()
        self.environment.unload_socratic()

    def _maybe_train_red(self, round_index: int) -> None:
        if (round_index + 1) % self.settings.training.red.update_every_rounds != 0:
            return
        sft_examples = self.buffer.build_red_sft_examples(self.topic_descriptions)
        dpo_pairs = self.buffer.build_dpo_pairs(self.topic_descriptions)
        previous_adapter = self.state.red.active_adapter_path
        self.state.red = self.red_trainer.train(sft_examples, dpo_pairs, self.state.red, round_index)
        if self.state.red.active_adapter_path != previous_adapter:
            self.environment.unload_red()

    def run(self) -> None:
        try:
            self.environment.load_judge()
            for round_index in range(self.state.round_index, self.settings.runtime.max_rounds):
                topics = self.curriculum.sample_topics(round_index, self.random)
                if self.curriculum.register_topics(topics):
                    self._handle_repetition_reset()
                    topics = self.curriculum.sample_topics(round_index, self.random)
                    self.curriculum.register_topics(topics)

                validation_result = self._generate_valid_tasks(topics)
                if self.settings.runtime.unload_red_after_generation:
                    self.environment.unload_red()

                valid_tasks = validation_result.valid_tasks
                if not valid_tasks:
                    rejected_records = self._build_rejected_records(validation_result.rejected_tasks)
                    if rejected_records:
                        self.buffer.add_many(rejected_records)
                    self.state.round_index = round_index + 1
                    self.storage.save_state(self.state)
                    continue

                hints = self.socratic_generator.generate_hints([item.task for item in valid_tasks])
                evaluations = self.assessor.assess_hints(hints)
                if self.settings.runtime.unload_socratic_after_use:
                    self.environment.unload_socratic()

                outcomes, replay_records = self._build_outcomes(valid_tasks, evaluations)
                rejected_records = self._build_rejected_records(validation_result.rejected_tasks)
                self.buffer.add_many(replay_records + rejected_records)
                self.curriculum.update_from_outcomes(outcomes)
                self.pending_grpo_tasks.extend([item.task for item in valid_tasks])
                self._maybe_train_socratic(round_index)
                self._maybe_train_red(round_index)

                self.state.round_index = round_index + 1
                self.state.total_tasks_seen += len(valid_tasks)
                snapshot = {
                    "round_index": round_index,
                    "topics": [topic.name for topic in topics],
                    "valid_tasks": [item.task.task_id for item in valid_tasks],
                    "rejected_tasks": [item.task.task_id for item in validation_result.rejected_tasks],
                    "outcomes": [dataclass_to_dict(outcome) for outcome in outcomes],
                    "socratic_state": dataclass_to_dict(self.state.socratic),
                    "red_state": dataclass_to_dict(self.state.red),
                }
                self.storage.save_round_snapshot(round_index, snapshot)
                self.storage.append_event(
                    "round_completed",
                    {
                        "round_index": round_index,
                        "topics": [topic.name for topic in topics],
                        "valid_task_count": len(valid_tasks),
                        "rejected_task_count": len(validation_result.rejected_tasks),
                        "average_rewards": {outcome.task_id: outcome.average_reward for outcome in outcomes},
                    },
                )
                self.storage.save_state(self.state)
        finally:
            self.storage.save_state(self.state)
            self.environment.close()
