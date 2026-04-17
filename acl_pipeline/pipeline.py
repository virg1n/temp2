from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .config import PipelineConfig
from .curriculum import CurriculumManager
from .judge import JudgeService
from .logging_utils import build_logger
from .modeling import ModelPool
from .prompts import build_red_repair_message, build_red_messages, build_red_training_prompt
from .red_generation import RedTaskGenerator
from .red_update import RedUpdater, serialize_task_json
from .schemas import EpisodeRecord, RedTrainingExample
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

    def _attach_execution(self, task, execution_result) -> None:
        task.metadata["execution"] = execution_result.to_dict()
        task.metadata["execution_status"] = execution_result.status
        task.metadata["observed_failure"] = execution_result.error_message

    def _normalize_topic(self, topic: str) -> str:
        return " ".join(str(topic).lower().replace("_", " ").split())

    def _candidate_rejection_reasons(
        self,
        *,
        requested_topic: str,
        task,
        execution_result,
    ) -> List[str]:
        reasons: List[str] = []
        code_lines = [line for line in task.buggy_solution.splitlines() if line.strip()]
        assert_count = sum(1 for line in task.failing_asserts if line.strip().startswith("assert "))
        fn_markers = sum(1 for line in code_lines if line.lstrip().startswith(("def ", "class ")))

        if self._normalize_topic(task.topic) != self._normalize_topic(requested_topic):
            reasons.append("wrong topic")
        if len(code_lines) <= 10:
            reasons.append("too short")
        if assert_count < 3:
            reasons.append(f"only {assert_count} asserts")
        if fn_markers <= 1 and len(code_lines) <= 14:
            reasons.append("trivial")
        if execution_result is not None and execution_result.status == "passed":
            reasons.append("passed execution")
        return reasons

    def _generate_repair_or_skip(self, topic: str, weakness_summary: str) -> Optional[Any]:
        red_session = self.model_pool.load_red_generation(adapter_path=self.current_red_adapter)
        messages = build_red_messages(topic, weakness_summary)
        last_rejection_reasons: List[str] = []
        try:
            for attempt in range(1, self.config.task_execution.max_red_generation_attempts + 1):
                raw = self.red_generator.generate_raw_response(red_session, messages)
                task, parse_reasons = self.red_generator.parse_task_response(raw, requested_topic=topic)
                messages.append({"role": "assistant", "content": raw})
                rejection_reasons = list(parse_reasons)
                execution = None

                if task is not None and self.config.task_execution.enabled:
                    execution = execute_task(task, self.config.task_execution)
                    self._attach_execution(task, execution)

                if task is not None:
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
                self.logger.warning(
                    "red_task_repair_requested",
                    topic=topic,
                    attempt=attempt,
                    rejection_reasons=last_rejection_reasons,
                    execution_status=execution.status if execution is not None else None,
                )
                messages.append(build_red_repair_message(topic, last_rejection_reasons))
        finally:
            red_session.unload()

        self.logger.warning(
            "red_task_generation_failed",
            topic=topic,
            weakness_summary=weakness_summary,
            rejection_reasons=last_rejection_reasons,
        )
        return None

    def _build_hard_example(self, episode: EpisodeRecord, weakness_summary: str) -> RedTrainingExample:
        prompt = (
            build_red_training_prompt(episode.topic, weakness_summary)
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
            judge_grade=episode.judge.score,
            judge_adjusted_score=episode.judge.metadata.get("adjusted_score"),
            judge_criteria=episode.judge.criteria_scores,
            task_metadata=episode.task.metadata,
        )

    def _reset_red_and_curriculum(self, episode_id: int) -> None:
        self.current_red_adapter = self.config.red.base_adapter_path
        self.storage.save_pointer("red_adapter_path", self.current_red_adapter)
        self.storage.save_curriculum_state(self.curriculum.snapshot())
        self.logger.warning(
            "curriculum_reset",
            episode_id=episode_id,
            reason="same_topic_repeat_threshold",
            red_adapter_path=self.current_red_adapter,
            curriculum_weights=self.curriculum.snapshot().weights,
        )

    def _flush_pending_batch(self, pending_batch: List[Dict[str, Any]]) -> List[EpisodeRecord]:
        tasks = [item["task"] for item in pending_batch]
        hint_texts = [item["hint"].text for item in pending_batch]
        judge_outputs = self.judge.evaluate_batch(
            tasks,
            hint_texts,
            apply_batch_spread=True,
        )
        self.logger.event(
            "judge_batch_complete",
            episode_ids=[item["episode_id"] for item in pending_batch],
            raw_scores=[output.score for output in judge_outputs],
            adjusted_scores=[output.metadata.get("adjusted_score") for output in judge_outputs],
            adjusted_rewards=[output.normalized_reward for output in judge_outputs],
        )
        records: List[EpisodeRecord] = []
        for item, judge_output in zip(pending_batch, judge_outputs):
            episode = EpisodeRecord(
                episode_id=item["episode_id"],
                topic=item["task"].topic,
                task=item["task"],
                hint=item["hint"],
                judge=judge_output,
                metadata={
                    "weakness_summary": item["weakness_summary"],
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
            self._log_episode_debug(episode, item["weakness_summary"])
            if should_reset:
                self._reset_red_and_curriculum(episode.episode_id)
            records.append(episode)

        self._store_hard_examples_for_batch(records)
        return records

    def _run_due_updates(self, batch_records: List[EpisodeRecord]) -> None:
        if not batch_records:
            return
        socratic_due = [
            record.episode_id
            for record in batch_records
            if record.episode_id % self.config.socratic.grpo.update_every_episodes == 0
        ]
        if socratic_due:
            step = socratic_due[-1]
            recent_episodes = self.storage.load_recent_episodes(self.config.socratic.grpo.max_training_examples)
            result = self.socratic_updater.run(
                episodes=recent_episodes,
                step=step,
                model_source=self.current_socratic_model,
                adapter_path=self.current_socratic_adapter,
            )
            if result is not None:
                self.current_socratic_model = result.model_source
                self.current_socratic_adapter = result.adapter_path
                self.storage.save_pointer("socratic_model_path", self.current_socratic_model)
                self.storage.save_pointer("socratic_adapter_path", self.current_socratic_adapter)

        red_due = [
            record.episode_id
            for record in batch_records
            if record.episode_id % self.config.red.update.update_every_episodes == 0
        ]
        if red_due:
            step = red_due[-1]
            hard_examples = self.storage.load_hard_examples(self.config.red.update.max_sft_examples)
            recent_episodes = self.storage.load_recent_episodes(max(self.config.red.update.max_sft_examples, 256))
            red_result = self.red_updater.run(
                hard_examples=hard_examples,
                recent_episodes=recent_episodes,
                step=step,
                adapter_path=self.current_red_adapter,
            )
            if red_result.adapter_path:
                self.current_red_adapter = red_result.adapter_path
                self.storage.save_pointer("red_adapter_path", self.current_red_adapter)

    def run(self) -> None:
        start_episode = self.storage.episode_count()
        self.logger.event(
            "pipeline_start",
            start_episode=start_episode,
            total_episodes=self.config.runtime.total_episodes,
            socratic_model=self.current_socratic_model,
            socratic_adapter=self.current_socratic_adapter,
            red_adapter=self.current_red_adapter,
        )
        self.model_pool.get_judge()

        try:
            target_episode = start_episode + self.config.runtime.total_episodes
            episode_id = start_episode
            generation_attempt = 0
            pending_batch: List[Dict[str, Any]] = []
            while episode_id < target_episode:
                generation_attempt += 1
                topic = self.curriculum.sample_topic(self.rng)
                weakness_summary = self.curriculum.weakness_summary(topic)
                task = self._generate_repair_or_skip(topic, weakness_summary)
                if task is None:
                    self.logger.warning(
                        "episode_skipped_red_failure",
                        generation_attempt=generation_attempt,
                        topic=topic,
                        weakness_summary=weakness_summary,
                    )
                    continue

                socratic_session = self.model_pool.get_socratic(
                    model_source=self.current_socratic_model,
                    adapter_path=self.current_socratic_adapter,
                )
                try:
                    hint = generate_socratic_hint(socratic_session, task, self.logger)
                finally:
                    if not self.config.socratic.hardware.persistent:
                        socratic_session.unload()
                episode_id += 1
                pending_batch.append(
                    {
                        "episode_id": episode_id,
                        "task": task,
                        "hint": hint,
                        "weakness_summary": weakness_summary,
                    }
                )

                if len(pending_batch) >= self.config.judge.episode_batch_size:
                    batch_records = self._flush_pending_batch(pending_batch)
                    self._run_due_updates(batch_records)
                    if batch_records and batch_records[-1].episode_id % self.config.runtime.checkpoint_every_episodes == 0:
                        self.storage.save_curriculum_state(self.curriculum.snapshot())
                        self.logger.event("checkpoint_marker", episode_id=batch_records[-1].episode_id)
                    pending_batch = []

            if pending_batch:
                batch_records = self._flush_pending_batch(pending_batch)
                self._run_due_updates(batch_records)
        finally:
            self.storage.save_curriculum_state(self.curriculum.snapshot())
            self.logger.event("pipeline_stop", model_pool=self.model_pool.debug_summary())
            self.model_pool.close()
