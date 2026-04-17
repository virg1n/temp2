from __future__ import annotations

import random
from uuid import uuid4

from .config import PipelineConfig
from .curriculum import CurriculumManager
from .judge import JudgeService
from .logging_utils import build_logger
from .modeling import ModelPool
from .prompts import build_red_training_prompt
from .red_generation import RedTaskGenerator
from .red_update import RedUpdater, serialize_task_json
from .schemas import EpisodeRecord, RedTrainingExample
from .socratic_generation import generate_socratic_hint
from .socratic_grpo import SocraticGrpoUpdater
from .storage import SimpleStorage


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

    def _build_hard_example(self, episode: EpisodeRecord, weakness_summary: str) -> RedTrainingExample:
        prompt = build_red_training_prompt(episode.topic, weakness_summary)
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
            },
        )

    def _maybe_store_hard_example(self, episode: EpisodeRecord, weakness_summary: str) -> None:
        threshold = self.config.red.update.hard_reward_threshold
        if episode.judge.normalized_reward > threshold:
            return
        example = self._build_hard_example(episode, weakness_summary)
        self.storage.append_hard_example(example)
        self.logger.event(
            "hard_example_added",
            episode_id=episode.episode_id,
            topic=episode.topic,
            reward=episode.judge.normalized_reward,
        )

    def _log_episode_debug(self, episode: EpisodeRecord, weakness_summary: str) -> None:
        self.logger.debug_dump(
            "episode_debug",
            episode_id=episode.episode_id,
            topic=episode.topic,
            weakness_summary=weakness_summary,
            broken_code=episode.task.combined_program(),
            socratic_hint=episode.hint.text,
            judge_grade=episode.judge.score,
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
            final_episode = start_episode + self.config.runtime.total_episodes
            for episode_id in range(start_episode + 1, final_episode + 1):
                topic = self.curriculum.sample_topic(self.rng)
                weakness_summary = self.curriculum.weakness_summary(topic)

                red_session = self.model_pool.load_red_generation(adapter_path=self.current_red_adapter)
                try:
                    task = self.red_generator.generate_task(
                        red_session,
                        topic=topic,
                        weakness_summary=weakness_summary,
                    )
                finally:
                    red_session.unload()

                socratic_session = self.model_pool.get_socratic(
                    model_source=self.current_socratic_model,
                    adapter_path=self.current_socratic_adapter,
                )
                try:
                    hint = generate_socratic_hint(socratic_session, task, self.logger)
                finally:
                    if not self.config.socratic.hardware.persistent:
                        socratic_session.unload()

                judge_output = self.judge.evaluate(task, hint.text)
                episode = EpisodeRecord(
                    episode_id=episode_id,
                    topic=task.topic,
                    task=task,
                    hint=hint,
                    judge=judge_output,
                    metadata={
                        "weakness_summary": weakness_summary,
                        "socratic_model": self.current_socratic_model,
                        "socratic_adapter": self.current_socratic_adapter,
                        "red_adapter": self.current_red_adapter,
                    },
                )
                self.storage.append_episode(episode)
                self._maybe_store_hard_example(episode, weakness_summary)

                should_reset, snapshot = self.curriculum.observe(task.topic, judge_output.normalized_reward)
                self.storage.save_curriculum_state(snapshot)
                self.logger.event(
                    "episode_complete",
                    episode_id=episode_id,
                    topic=task.topic,
                    reward=judge_output.normalized_reward,
                    score=judge_output.score,
                    curriculum_weights=snapshot.weights,
                )
                self._log_episode_debug(episode, weakness_summary)

                if should_reset:
                    self._reset_red_and_curriculum(episode_id)

                if episode_id % self.config.socratic.grpo.update_every_episodes == 0:
                    recent_episodes = self.storage.load_recent_episodes(self.config.socratic.grpo.max_training_examples)
                    result = self.socratic_updater.run(
                        episodes=recent_episodes,
                        step=episode_id,
                        model_source=self.current_socratic_model,
                        adapter_path=self.current_socratic_adapter,
                    )
                    if result is not None:
                        self.current_socratic_model = result.model_source
                        self.current_socratic_adapter = result.adapter_path
                        self.storage.save_pointer("socratic_model_path", self.current_socratic_model)
                        self.storage.save_pointer("socratic_adapter_path", self.current_socratic_adapter)

                if episode_id % self.config.red.update.update_every_episodes == 0:
                    hard_examples = self.storage.load_hard_examples(self.config.red.update.max_sft_examples)
                    recent_episodes = self.storage.load_recent_episodes(max(self.config.red.update.max_sft_examples, 256))
                    red_result = self.red_updater.run(
                        hard_examples=hard_examples,
                        recent_episodes=recent_episodes,
                        step=episode_id,
                        adapter_path=self.current_red_adapter,
                    )
                    if red_result.adapter_path:
                        self.current_red_adapter = red_result.adapter_path
                        self.storage.save_pointer("red_adapter_path", self.current_red_adapter)

                if episode_id % self.config.runtime.checkpoint_every_episodes == 0:
                    self.storage.save_curriculum_state(self.curriculum.snapshot())
                    self.logger.event("checkpoint_marker", episode_id=episode_id)
        finally:
            self.storage.save_curriculum_state(self.curriculum.snapshot())
            self.logger.event("pipeline_stop", model_pool=self.model_pool.debug_summary())
            self.model_pool.close()
