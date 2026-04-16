from __future__ import annotations

import logging
import random

from logging_utils import get_logger, log_event
from schemas import CurriculumSettings, CurriculumState, CurriculumTopic, TaskOutcome, TopicStats


LOGGER = get_logger(__name__)


class CurriculumManager:
    def __init__(self, settings: CurriculumSettings, state: CurriculumState) -> None:
        self.settings = settings
        self.state = state
        self.topic_map = {topic.name: topic for topic in settings.topics}
        self._initial_weights = {topic.name: topic.initial_weight for topic in settings.topics}

    def _weighted_sample_without_replacement(self, rng: random.Random, count: int) -> list[CurriculumTopic]:
        available = list(self.settings.topics)
        chosen: list[CurriculumTopic] = []
        while available and len(chosen) < count:
            weights = []
            for topic in available:
                stats = self.state.topic_stats.setdefault(topic.name, TopicStats())
                weakness_bonus = max(0.0, self.settings.weak_reward_threshold - stats.average_reward)
                weights.append(self.state.topic_weights.get(topic.name, topic.initial_weight) * (1.0 + weakness_bonus))
            picked = rng.choices(available, weights=weights, k=1)[0]
            chosen.append(picked)
            available = [topic for topic in available if topic.name != picked.name]
        return chosen

    def sample_topics(self, round_index: int, rng: random.Random) -> list[CurriculumTopic]:
        count = min(self.settings.sample_topics_per_round, len(self.settings.topics))
        if round_index < self.settings.random_sampling_rounds:
            topics = rng.sample(self.settings.topics, k=count)
        else:
            topics = self._weighted_sample_without_replacement(rng, count)
        log_event(
            LOGGER,
            logging.INFO,
            "curriculum_sampled",
            "Sampled curriculum topics",
            round_index=round_index,
            topics=[topic.name for topic in topics],
        )
        return topics

    def register_topics(self, topics: list[CurriculumTopic]) -> bool:
        reset_required = False
        for topic in topics:
            if topic.name == self.state.last_topic:
                self.state.consecutive_topic_repetitions += 1
            else:
                self.state.last_topic = topic.name
                self.state.consecutive_topic_repetitions = 1
            if self.state.consecutive_topic_repetitions >= self.settings.repeated_topic_reset_threshold:
                reset_required = True

        log_event(
            LOGGER,
            logging.INFO,
            "topic_repetition_updated",
            "Updated topic repetition tracker",
            last_topic=self.state.last_topic,
            consecutive=self.state.consecutive_topic_repetitions,
            threshold=self.settings.repeated_topic_reset_threshold,
        )
        return reset_required

    def reset_to_initial(self) -> None:
        self.state.topic_weights = dict(self._initial_weights)
        self.state.topic_stats = {topic.name: TopicStats() for topic in self.settings.topics}
        self.state.last_topic = None
        self.state.consecutive_topic_repetitions = 0
        log_event(LOGGER, logging.WARNING, "curriculum_reset", "Curriculum reset to initial weights")

    def update_from_outcomes(self, outcomes: list[TaskOutcome]) -> None:
        for outcome in outcomes:
            stats = self.state.topic_stats.setdefault(outcome.topic, TopicStats())
            stats.attempts += 1
            stats.valid_tasks += 1
            stats.cumulative_reward += outcome.average_reward

            current_weight = self.state.topic_weights.get(outcome.topic, self._initial_weights.get(outcome.topic, 1.0))
            if outcome.average_reward <= self.settings.weak_reward_threshold:
                current_weight *= self.settings.weak_topic_boost
            elif outcome.average_reward >= self.settings.strong_reward_threshold:
                current_weight *= self.settings.strong_topic_decay
            self.state.topic_weights[outcome.topic] = max(
                self.settings.min_weight,
                min(self.settings.max_weight, current_weight),
            )

        log_event(
            LOGGER,
            logging.INFO,
            "curriculum_updated",
            "Updated curriculum weights from outcomes",
            topic_weights=self.state.topic_weights,
        )

    def topic_description(self, topic_name: str) -> str:
        topic = self.topic_map.get(topic_name)
        return topic.description if topic is not None else topic_name
