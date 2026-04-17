from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

from .config import CurriculumConfig
from .schemas import CurriculumState


class CurriculumManager:
    def __init__(self, config: CurriculumConfig) -> None:
        self.config = config
        self.initial_weights = {topic.name: float(topic.weight) for topic in config.topics}
        self.weights = dict(self.initial_weights)
        self.running_topic_rewards = {name: 0.5 for name in self.initial_weights}
        self.recent_topics: deque[str] = deque(maxlen=32)
        self.consecutive_topic: Optional[str] = None
        self.consecutive_count = 0
        self.resets = 0

    def sample_topic(self, rng: random.Random) -> str:
        topics = list(self.weights)
        values = [max(1e-6, float(self.weights[name])) for name in topics]
        return rng.choices(topics, weights=values, k=1)[0]

    def weakness_summary(self, topic: str) -> str:
        score = self.running_topic_rewards.get(topic, 0.5)
        return (
            f"Recent normalized Socratic reward for '{topic}' is {score:.3f}. "
            "Generate a task that is likely to expose the same weakness more clearly."
        )

    def weakest_topic(self) -> Optional[str]:
        if not self.running_topic_rewards:
            return None
        return min(
            self.running_topic_rewards,
            key=lambda name: (self.running_topic_rewards.get(name, 0.5), name),
        )

    def apply_iteration_focus_boost(self) -> Tuple[Optional[str], CurriculumState]:
        weakest = self.weakest_topic()
        if weakest is None:
            return None, self.snapshot()

        self.weights[weakest] = max(
            1e-3,
            float(self.weights.get(weakest, self.initial_weights.get(weakest, 1.0)))
            * (1.0 + float(self.config.iteration_weak_topic_boost)),
        )
        return weakest, self.snapshot()

    def observe(self, topic: str, reward: float) -> Tuple[bool, CurriculumState]:
        reward = max(0.0, min(1.0, float(reward)))
        alpha = self.config.reward_ema_alpha
        previous = self.running_topic_rewards.get(topic, 0.5)
        ema = ((1.0 - alpha) * previous) + (alpha * reward)
        self.running_topic_rewards[topic] = ema

        base = self.initial_weights.get(topic, 1.0)
        weakness = max(0.0, 1.0 - ema)
        self.weights[topic] = max(1e-3, base * (1.0 + self.config.low_reward_boost * weakness))

        self.recent_topics.append(topic)
        if topic == self.consecutive_topic:
            self.consecutive_count += 1
        else:
            self.consecutive_topic = topic
            self.consecutive_count = 1

        should_reset = self.consecutive_count >= self.config.repeat_topic_reset_threshold
        if should_reset:
            self.reset()

        return should_reset, self.snapshot()

    def reset(self) -> None:
        self.weights = dict(self.initial_weights)
        self.running_topic_rewards = {name: 0.5 for name in self.initial_weights}
        self.recent_topics.clear()
        self.consecutive_topic = None
        self.consecutive_count = 0
        self.resets += 1

    def restore(self, state: CurriculumState) -> None:
        self.weights = dict(state.weights)
        self.running_topic_rewards = dict(state.running_topic_rewards)
        self.recent_topics = deque(state.recent_topics, maxlen=32)
        self.consecutive_topic = state.consecutive_topic
        self.consecutive_count = int(state.consecutive_count)
        self.resets = int(state.resets)

    def snapshot(self) -> CurriculumState:
        return CurriculumState(
            weights=dict(self.weights),
            running_topic_rewards=dict(self.running_topic_rewards),
            recent_topics=list(self.recent_topics),
            consecutive_topic=self.consecutive_topic,
            consecutive_count=self.consecutive_count,
            resets=self.resets,
        )
