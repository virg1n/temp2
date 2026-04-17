from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .config import PipelineConfig
from .curriculum import CurriculumManager
from .judge import JudgeService
from .logging_utils import build_logger
from .modeling import ModelPool
from .prompts import (
    build_red_messages,
    build_red_repair_message,
    build_red_spec_repair_message,
    build_red_task_from_spec_message,
    build_red_training_prompt,
)
from .red_generation import RedTaskGenerator
from .red_update import RedUpdater, serialize_task_json
from .schemas import EpisodeRecord, RedTaskSpec, RedTrainingExample
from .socratic_generation import generate_socratic_hint
from .socratic_grpo import SocraticGrpoUpdater
from .storage import SimpleStorage
from .task_execution import execute_task


class AdversarialCurriculumPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = build_logger(
            config.storage.root_dir,
            level=config.runtime.log_level,
            debug_all=config.runtime.debug_all,
        )
        self.storage = SimpleStorage(
            config.storage.root_dir,
            keep_last_n_checkpoints=config.storage.keep_last_n_checkpoints,
            hard_buffer_max_size=config.storage.hard_buffer_max_size,
        )
        self.curriculum = CurriculumManager(config.curriculum)
        saved_state = self.storage.load_curriculum_state()
        if saved_state is not None:
            self.curriculum.restore(saved_state)

        self.model_pool = ModelPool(config, self.logger)
        self.judge = JudgeService(self.model_pool, self.logger)
        self.red_generator = RedTaskGenerator(self.logger)
        self.red_updater = RedUpdater(config, self.model_pool, self.storage, self.logger)
        self.socratic_updater = SocraticGrpoUpdater(config, self.model_pool, self.judge, self.storage, self.logger)
        self.rng = random.Random(config.runtime.seed)

        pointers = self.storage.load_pointers()
        self.current_socratic_model = str(pointers.get("socratic_model_path") or config.socratic.model_name_or_path)
        self.current_socratic_adapter = pointers.get("socratic_adapter_path") or config.socratic.base_adapter_path
        self.current_red_adapter = pointers.get("red_adapter_path") or config.red.base_adapter_path

    def _iteration_size(self) -> int:
        return max(1, int(self.config.runtime.iteration_size or self.config.red.update.update_every_episodes or 1))

    def _attach_execution(self, task, execution_result) -> None:
        task.metadata["execution"] = execution_result.to_dict()
        task.metadata["execution_status"] = execution_result.status
        task.metadata["observed_failure"] = execution_result.error_message

    def _normalize_topic(self, topic: str) -> str:
        return " ".join(str(topic).lower().replace("_", " ").split())

    def _task_spec_from_metadata(self, task) -> Optional[RedTaskSpec]:
        payload = dict(task.metadata.get("red_spec") or {})
        if not payload:
            return None
        topic = str(payload.get("topic") or task.topic)
        target_function = str(payload.get("target_function") or "").strip()
        intended_bug = str(payload.get("intended_bug") or "").strip()
        expected_first_failure = str(payload.get("expected_first_failure") or "").strip()
        if not (target_function or intended_bug or expected_first_failure):
            return None
        return RedTaskSpec(
            topic=topic,
            target_function=target_function,
            intended_bug=intended_bug,
            expected_first_failure=expected_first_failure,
            metadata=dict(payload.get("metadata") or {}),
        )

    def _record_red_rejection(
        self,
        *,
        topic: str,
        prompt: str,
        rejected_completion: str,
        rejection_reason: str,
        spec: Optional[RedTaskSpec] = None,
        task_quality: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        completion = str(rejected_completion or "").strip()
        if not completion:
            return
        example = RedRejectedExample(
            example_id=uuid4().hex[:16],
            topic=topic,
            prompt=prompt,
            rejected_completion=completion,
            rejection_reason=rejection_reason,
            task_quality=task_quality,
            spec=spec.to_dict() if spec is not None else None,
            metadata=dict(metadata or {}),
        )
        self.storage.append_red_rejected_example(example)
        self.logger.warning(
            "red_rejected_example_added",
            topic=topic,
            rejection_reason=rejection_reason,
            task_quality=task_quality,
        )

    def _candidate_rejection_reasons(
        self,
        *,
        requested_topic: str,
        task,
        execution_result,
    ) -> List[str]:
        reasons: List[str] = []
        if self._normalize_topic(task.topic) != self._normalize_topic(requested_topic):
            reasons.append("wrong topic")

        repair_probability = float(self.config.task_execution.probabilistic_repair_probability)
        if task.non_empty_line_count() < int(self.config.task_execution.min_code_lines_for_repair):
            if self.rng.random() < repair_probability:
                reasons.append("too short")

        if execution_result is not None and execution_result.status == "passed":
            if self.rng.random() < repair_probability:
                reasons.append("already correct code, there are no errors in asserts")
        return reasons

    def _generate_task_with_red_session(self, red_session, topic: str, weakness_summary: str):
        max_attempts = int(self.config.task_execution.max_red_generation_attempts)
        spec_messages = build_red_messages(topic, weakness_summary)
        spec: Optional[RedTaskSpec] = None
        last_rejection_reasons: List[str] = []

        for attempt in range(1, max_attempts + 1):
            spec_raw = self.red_generator.generate_raw_response(red_session, spec_messages)
            spec_messages.append({"role": "assistant", "content": spec_raw})
            spec, parse_reasons = self.red_generator.parse_spec_response(spec_raw, requested_topic=topic)
            if spec is not None and self._normalize_topic(spec.topic) == self._normalize_topic(topic):
                break
            last_rejection_reasons = list(parse_reasons)
            if spec is not None and self._normalize_topic(spec.topic) != self._normalize_topic(topic):
                last_rejection_reasons.append("wrong topic")
            spec = None
            self._record_red_rejection(
                topic=topic,
                prompt=build_red_training_prompt(topic, weakness_summary),
                rejected_completion=spec_raw,
                rejection_reason=", ".join(last_rejection_reasons or ["invalid spec"]),
                metadata={"stage": "spec", "attempt": attempt, "weakness_summary": weakness_summary},
            )
            self.logger.warning(
                "red_spec_repair_requested",
                topic=topic,
                attempt=attempt,
                rejection_reasons=last_rejection_reasons or ["invalid spec"],
            )
            spec_messages.append(build_red_spec_repair_message(topic, last_rejection_reasons or ["invalid spec"]))

        if spec is None:
            self.logger.warning(
                "red_task_generation_failed",
                topic=topic,
                weakness_summary=weakness_summary,
                rejection_reasons=last_rejection_reasons or ["invalid spec"],
            )
            return None

        task_prompt = build_red_training_prompt(topic, weakness_summary, spec=spec)
        messages = list(spec_messages)
        messages.append(build_red_task_from_spec_message(spec))

        for attempt in range(1, max_attempts + 1):
            raw = self.red_generator.generate_raw_response(red_session, messages)
            task, parse_reasons = self.red_generator.parse_task_response(raw, requested_topic=topic, spec=spec)
            messages.append({"role": "assistant", "content": raw})

            execution = None
            rejection_reasons = list(parse_reasons)

            if task is not None:
                task.metadata["red_prompt"] = task_prompt
                task.metadata["weakness_summary"] = weakness_summary
                if self.config.task_execution.enabled:
                    execution = execute_task(task, self.config.task_execution)
                    self._attach_execution(task, execution)
                rejection_reasons.extend(
                    self._candidate_rejection_reasons(
                        requested_topic=topic,
                        task=task,
                        execution_result=execution,
                    )
                )

            rejection_reasons = list(dict.fromkeys(reason for reason in rejection_reasons if reason))
            if task is not None and not rejection_reasons:
                return task

            last_rejection_reasons = rejection_reasons or ["unspecified issue"]
            self._record_red_rejection(
                topic=topic,
                prompt=task_prompt,
                rejected_completion=raw,
                rejection_reason=", ".join(last_rejection_reasons),
                spec=spec,
                metadata={
                    "stage": "task",
                    "attempt": attempt,
                    "weakness_summary": weakness_summary,
                    "execution_status": execution.status if execution is not None else None,
                },
            )
            self.logger.warning(
                "red_task_repair_requested",
                topic=topic,
                attempt=attempt,
                rejection_reasons=last_rejection_reasons,
                execution_status=execution.status if execution is not None else None,
            )
            messages.append(build_red_repair_message(topic, last_rejection_reasons, spec=spec))

        self.logger.warning(
            "red_task_generation_failed",
            topic=topic,
            weakness_summary=weakness_summary,
            rejection_reasons=last_rejection_reasons or ["unspecified issue"],
        )
        return None

    def _build_hard_example(self, episode: EpisodeRecord, weakness_summary: str) -> RedTrainingExample:
        spec = self._task_spec_from_metadata(episode.task)
        prompt = (
            build_red_training_prompt(episode.topic, weakness_summary, spec=spec)
            + f" Reproduced failure: {episode.task.observed_failure()[:600]}"
        )
        return RedTrainingExample(
            example_id=uuid4().hex[:16],
            topic=episode.topic,
            prompt=prompt,
            chosen_completion=serialize_task_json(episode.task),
            rejected_completion=None,
            reward=episode.judge.normalized_reward,
            task=episode.task,
            metadata={
                "episode_id": episode.episode_id,
                "socratic_score": episode.judge.score,
                "weakness_summary": weakness_summary,
                "observed_failure": episode.task.observed_failure(),
            },
        )

    def _store_hard_examples_for_batch(self, batch_records: List[EpisodeRecord]) -> None:
        if not batch_records:
            return
        bottom_fraction = float(self.config.red.update.mining_bottom_fraction)
        keep_count = max(1, math.ceil(len(batch_records) * bottom_fraction))
        selected = sorted(batch_records, key=lambda episode: episode.judge.normalized_reward)[:keep_count]
        self.logger.event(
            "hard_example_batch_selection",
            selected_episode_ids=[episode.episode_id for episode in selected],
            selected_rewards=[episode.judge.normalized_reward for episode in selected],
            batch_episode_ids=[episode.episode_id for episode in batch_records],
        )
        for episode in selected:
            weakness_summary = str(episode.metadata.get("weakness_summary") or "")
            example = self._build_hard_example(episode, weakness_summary)
            self.storage.append_hard_example(example)
            self.logger.event(
                "hard_example_added",
                episode_id=episode.episode_id,
                topic=episode.topic,
                reward=episode.judge.normalized_reward,
                batch_bottom_fraction=bottom_fraction,
            )

    def _log_episode_debug(self, episode: EpisodeRecord, weakness_summary: str) -> None:
        self.logger.debug_dump(
            "episode_debug",
            episode_id=episode.episode_id,
            topic=episode.topic,
            weakness_summary=weakness_summary,
            broken_code=episode.task.combined_program(),
            execution=episode.task.metadata.get("execution"),
            socratic_hint=episode.hint.text,
            socratic_hint_raw=episode.hint.raw_text,
            judge_grade=episode.judge.score,
            judge_adjusted_score=episode.judge.metadata.get("adjusted_score"),
            judge_criteria=episode.judge.criteria_scores,
            judge_task_quality=episode.judge.metadata.get("task_quality"),
            judge_use_for_socratic=episode.judge.metadata.get("use_for_socratic"),
            judge_red_rejection_reason=episode.judge.metadata.get("red_rejection_reason"),
            hint_corruption=episode.judge.metadata.get("hint_corruption"),
            task_metadata=episode.task.metadata,
        )

    def _reset_red_and_curriculum(self, episode_id: int) -> None:
        self.current_red_adapter = self.config.red.base_adapter_path
        self.storage.save_pointer("red_adapter_path", self.current_red_adapter)
        snapshot = self.curriculum.snapshot()
        self.storage.save_curriculum_state(snapshot)
        self.logger.warning(
            "curriculum_reset",
            episode_id=episode_id,
            reason="same_topic_repeat_threshold",
            red_adapter_path=self.current_red_adapter,
            curriculum_weights=snapshot.weights,
        )

    def _flush_pending_batch(
        self,
        pending_batch: List[Dict[str, Any]],
        next_episode_id: int,
    ) -> Tuple[List[EpisodeRecord], int]:
        tasks = [item["task"] for item in pending_batch]
        hints = [item["hint"] for item in pending_batch]
        judge_outputs = self.judge.evaluate_batch(
            tasks,
            hints,
            apply_batch_spread=True,
        )
        self.logger.event(
            "judge_batch_complete",
            candidate_count=len(pending_batch),
            topics=[item["task"].topic for item in pending_batch],
            raw_scores=[output.score for output in judge_outputs],
            adjusted_scores=[output.metadata.get("adjusted_score") for output in judge_outputs],
            adjusted_rewards=[output.normalized_reward for output in judge_outputs],
            task_quality=[output.metadata.get("task_quality") for output in judge_outputs],
            use_for_socratic=[output.metadata.get("use_for_socratic") for output in judge_outputs],
        )

        records: List[EpisodeRecord] = []
        for item, judge_output in zip(pending_batch, judge_outputs):
            task = item["task"]
            weakness_summary = item["weakness_summary"]
            use_for_socratic = bool(judge_output.metadata.get("use_for_socratic", True))
            if not use_for_socratic:
                spec = self._task_spec_from_metadata(task)
                rejection_reason = str(judge_output.metadata.get("red_rejection_reason") or "judge_bad_task")
                self._record_red_rejection(
                    topic=task.topic,
                    prompt=str(task.metadata.get("red_prompt") or build_red_training_prompt(task.topic, weakness_summary, spec=spec)),
                    rejected_completion=serialize_task_json(task),
                    rejection_reason=rejection_reason,
                    spec=spec,
                    task_quality=float(judge_output.metadata.get("task_quality") or 0.0),
                    metadata={
                        "stage": "judge",
                        "weakness_summary": weakness_summary,
                        "observed_failure": task.observed_failure(),
                    },
                )
                self.logger.warning(
                    "red_task_rejected_by_judge",
                    topic=task.topic,
                    task_quality=judge_output.metadata.get("task_quality"),
                    rejection_reason=rejection_reason,
                    observed_failure=task.observed_failure(),
                )
                continue

            next_episode_id += 1
            episode = EpisodeRecord(
                episode_id=next_episode_id,
                topic=task.topic,
                task=task,
                hint=item["hint"],
                judge=judge_output,
                metadata={
                    "weakness_summary": weakness_summary,
                    "socratic_model": self.current_socratic_model,
                    "socratic_adapter": self.current_socratic_adapter,
                    "red_adapter": self.current_red_adapter,
                },
            )
            self.storage.append_episode(episode)
            should_reset, snapshot = self.curriculum.observe(
                episode.topic,
                judge_output.normalized_reward,
            )
            self.storage.save_curriculum_state(snapshot)
            self.logger.event(
                "episode_complete",
                episode_id=episode.episode_id,
                topic=episode.topic,
                reward=judge_output.normalized_reward,
                score=judge_output.score,
                adjusted_score=judge_output.metadata.get("adjusted_score"),
                curriculum_weights=snapshot.weights,
            )
            self._log_episode_debug(episode, weakness_summary)
            if should_reset:
                self._reset_red_and_curriculum(episode.episode_id)
            records.append(episode)

        self._store_hard_examples_for_batch(records)
        return records, next_episode_id

    def _generate_iteration_tasks(self, target_count: int, iteration_index: int) -> List[Dict[str, Any]]:
        generated: List[Dict[str, Any]] = []
        max_generation_attempts = max(1, target_count * max(2, self.config.task_execution.max_red_generation_attempts))
        generation_attempt = 0
        red_session = self.model_pool.load_red_generation(adapter_path=self.current_red_adapter)
        try:
            while len(generated) < target_count and generation_attempt < max_generation_attempts:
                generation_attempt += 1
                topic = self.curriculum.sample_topic(self.rng)
                weakness_summary = self.curriculum.weakness_summary(topic)
                task = self._generate_task_with_red_session(red_session, topic, weakness_summary)
                if task is None:
                    self.logger.warning(
                        "episode_skipped_red_failure",
                        iteration=iteration_index,
                        generation_attempt=generation_attempt,
                        topic=topic,
                        weakness_summary=weakness_summary,
                    )
                    continue
                generated.append(
                    {
                        "task": task,
                        "weakness_summary": weakness_summary,
                    }
                )
        finally:
            red_session.unload()

        self.logger.event(
            "iteration_red_generation_complete",
            iteration=iteration_index,
            requested_tasks=target_count,
            generated_tasks=len(generated),
            attempts=generation_attempt,
        )
        return generated

    def _generate_socratic_hints_for_iteration(self, items: List[Dict[str, Any]], iteration_index: int) -> List[Dict[str, Any]]:
        if not items:
            return []
        socratic_session = self.model_pool.get_socratic(
            model_source=self.current_socratic_model,
            adapter_path=self.current_socratic_adapter,
        )
        try:
            for item in items:
                item["hint"] = generate_socratic_hint(socratic_session, item["task"], self.logger)
        finally:
            if not self.config.socratic.hardware.persistent:
                socratic_session.unload()
        self.logger.event(
            "iteration_socratic_generation_complete",
            iteration=iteration_index,
            hint_count=len(items),
        )
        return items

    def _run_iteration_updates(self, accepted_records: List[EpisodeRecord], iteration_index: int) -> None:
        step = accepted_records[-1].episode_id if accepted_records else self.storage.episode_count()
        if step <= 0:
            return

        if accepted_records:
            recent_episodes = self.storage.load_recent_episodes(self.config.socratic.grpo.max_training_examples)
            socratic_result = self.socratic_updater.run(
                episodes=recent_episodes,
                step=step,
                model_source=self.current_socratic_model,
                adapter_path=self.current_socratic_adapter,
            )
            if socratic_result is not None:
                self.current_socratic_model = socratic_result.model_source
                self.current_socratic_adapter = socratic_result.adapter_path
                self.storage.save_pointer("socratic_model_path", self.current_socratic_model)
                self.storage.save_pointer("socratic_adapter_path", self.current_socratic_adapter)

        hard_examples = self.storage.load_hard_examples(self.config.red.update.max_sft_examples)
        rejected_examples = self.storage.load_red_rejected_examples(self.config.red.update.max_dpo_pairs)
        recent_for_red = self.storage.load_recent_episodes(max(self.config.red.update.max_sft_examples, 256))
        red_result = self.red_updater.run(
            hard_examples=hard_examples,
            rejected_examples=rejected_examples,
            recent_episodes=recent_for_red,
            step=step,
            adapter_path=self.current_red_adapter,
        )
        if red_result.adapter_path:
            self.current_red_adapter = red_result.adapter_path
            self.storage.save_pointer("red_adapter_path", self.current_red_adapter)

        self.logger.event(
            "iteration_updates_complete",
            iteration=iteration_index,
            step=step,
            accepted_records=len(accepted_records),
            socratic_adapter=self.current_socratic_adapter,
            red_adapter=self.current_red_adapter,
        )

    def _apply_iteration_curriculum_focus(self, iteration_index: int) -> None:
        weakest_topic, snapshot = self.curriculum.apply_iteration_focus_boost()
        self.storage.save_curriculum_state(snapshot)
        self.logger.event(
            "curriculum_iteration_focus",
            iteration=iteration_index,
            weakest_topic=weakest_topic,
            curriculum_weights=snapshot.weights,
            running_topic_rewards=snapshot.running_topic_rewards,
        )

    def run(self) -> None:
        start_episode = self.storage.episode_count()
        self.logger.event(
            "pipeline_start",
            start_episode=start_episode,
            total_episodes=self.config.runtime.total_episodes,
            iteration_size=self._iteration_size(),
            socratic_model=self.current_socratic_model,
            socratic_adapter=self.current_socratic_adapter,
            red_adapter=self.current_red_adapter,
        )
        self.model_pool.get_judge()

        try:
            target_episode = start_episode + self.config.runtime.total_episodes
            next_episode_id = start_episode
            iteration_index = 0
            stalled_iterations = 0

            while next_episode_id < target_episode:
                iteration_index += 1
                remaining = target_episode - next_episode_id
                requested_tasks = min(self._iteration_size(), remaining)
                self.logger.event(
                    "iteration_start",
                    iteration=iteration_index,
                    start_episode=next_episode_id,
                    requested_tasks=requested_tasks,
                )

                generated = self._generate_iteration_tasks(requested_tasks, iteration_index)
                generated = self._generate_socratic_hints_for_iteration(generated, iteration_index)

                accepted_records: List[EpisodeRecord] = []
                pending_batch: List[Dict[str, Any]] = []
                for item in generated:
                    pending_batch.append(item)
                    if len(pending_batch) >= self.config.judge.episode_batch_size:
                        batch_records, next_episode_id = self._flush_pending_batch(pending_batch, next_episode_id)
                        accepted_records.extend(batch_records)
                        pending_batch = []

                if pending_batch:
                    batch_records, next_episode_id = self._flush_pending_batch(pending_batch, next_episode_id)
                    accepted_records.extend(batch_records)

                self._run_iteration_updates(accepted_records, iteration_index)
                self._apply_iteration_curriculum_focus(iteration_index)

                if accepted_records and accepted_records[-1].episode_id % self.config.runtime.checkpoint_every_episodes == 0:
                    self.storage.save_curriculum_state(self.curriculum.snapshot())
                    self.logger.event("checkpoint_marker", episode_id=accepted_records[-1].episode_id)

                self.logger.event(
                    "iteration_complete",
                    iteration=iteration_index,
                    accepted_episodes=len(accepted_records),
                    total_episodes=next_episode_id,
                )
                if accepted_records:
                    stalled_iterations = 0
                else:
                    stalled_iterations += 1
                    if stalled_iterations >= 5:
                        self.logger.error(
                            "pipeline_stalled",
                            iteration=iteration_index,
                            reason="five_iterations_without_accepted_episodes",
                        )
                        break
        finally:
            self.storage.save_curriculum_state(self.curriculum.snapshot())
            self.logger.event("pipeline_stop", model_pool=self.model_pool.debug_summary())
            self.model_pool.close()
