from __future__ import annotations

import random
from dataclasses import dataclass

from .config import ACTConfig, TopicConfig


@dataclass
class TopicState:
    ema_reward: float
    count: int = 0


class TopicSampler:
    def __init__(self, cfg: ACTConfig, rng: random.Random) -> None:
        self.cfg = cfg
        self.rng = rng
        self.states = {
            topic.name: TopicState(ema_reward=cfg.curriculum.initial_ema_reward)
            for topic in cfg.curriculum.topics
        }
        self.last_topic: str | None = None
        self.same_topic_streak = 0

    def sample(self) -> TopicConfig:
        candidates = list(self.cfg.curriculum.topics)
        if (
            self.last_topic is not None
            and self.same_topic_streak >= self.cfg.curriculum.max_same_topic_streak
            and len(candidates) > 1
        ):
            candidates = [topic for topic in candidates if topic.name != self.last_topic]

        weights = []
        for topic in candidates:
            state = self.states[topic.name]
            low_reward_boost = 1.0 + self.cfg.curriculum.low_reward_boost_lambda * (1.0 - state.ema_reward)
            weights.append(max(0.01, topic.base_weight * low_reward_boost))

        chosen = self.rng.choices(candidates, weights=weights, k=1)[0]
        if chosen.name == self.last_topic:
            self.same_topic_streak += 1
        else:
            self.last_topic = chosen.name
            self.same_topic_streak = 1
        return chosen

    def update(self, topic_name: str, mean_score_0_to_10: float) -> None:
        normalized = max(0.0, min(1.0, mean_score_0_to_10 / 10.0))
        state = self.states[topic_name]
        alpha = self.cfg.curriculum.ema_alpha
        state.ema_reward = alpha * normalized + (1.0 - alpha) * state.ema_reward
        state.count += 1

    def state_json(self) -> dict:
        return {
            name: {"ema_reward": state.ema_reward, "count": state.count}
            for name, state in self.states.items()
        }
