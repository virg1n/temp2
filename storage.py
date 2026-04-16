from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from schemas import (
    CurriculumSettings,
    CurriculumState,
    PipelineSettings,
    PipelineState,
    RedAdaptationState,
    SocraticState,
    TopicStats,
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_settings(path: str | Path) -> PipelineSettings:
    settings_path = Path(path)
    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected YAML mapping at {settings_path}, got {type(raw)}")
    return PipelineSettings.from_dict(raw)


def build_initial_state(curriculum: CurriculumSettings, initial_red_adapter: str | None) -> PipelineState:
    topic_weights = {topic.name: topic.initial_weight for topic in curriculum.topics}
    topic_stats = {topic.name: TopicStats() for topic in curriculum.topics}
    return PipelineState(
        round_index=0,
        total_tasks_seen=0,
        curriculum=CurriculumState(topic_weights=topic_weights, topic_stats=topic_stats),
        red=RedAdaptationState(
            active_adapter_path=initial_red_adapter,
            initial_adapter_path=initial_red_adapter,
        ),
        socratic=SocraticState(),
    )


class StorageManager:
    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings
        self.root_dir = Path(settings.storage.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root_dir / settings.storage.state_path
        self.seen_tasks_path = self.root_dir / settings.storage.seen_tasks_path
        self.events_path = self.root_dir / settings.storage.events_path
        self.round_snapshots_dir = self.root_dir / settings.storage.round_snapshots_dir
        self.round_snapshots_dir.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> PipelineState:
        raw = _read_json(self.state_path, None)
        if raw is None:
            return build_initial_state(
                self.settings.curriculum,
                self.settings.models.red.initial_adapter_path or self.settings.models.red.adapter_path,
            )

        topic_stats = {
            topic: TopicStats(
                attempts=int(stats.get("attempts", 0)),
                valid_tasks=int(stats.get("valid_tasks", 0)),
                cumulative_reward=float(stats.get("cumulative_reward", 0.0)),
            )
            for topic, stats in raw["curriculum"]["topic_stats"].items()
        }
        curriculum_state = CurriculumState(
            topic_weights={k: float(v) for k, v in raw["curriculum"]["topic_weights"].items()},
            topic_stats=topic_stats,
            last_topic=raw["curriculum"].get("last_topic"),
            last_topic_batch_signature=raw["curriculum"].get("last_topic_batch_signature"),
            consecutive_topic_repetitions=int(raw["curriculum"].get("consecutive_topic_repetitions", 0)),
        )
        return PipelineState(
            round_index=int(raw.get("round_index", 0)),
            total_tasks_seen=int(raw.get("total_tasks_seen", 0)),
            curriculum=curriculum_state,
            red=RedAdaptationState(
                active_adapter_path=raw["red"].get("active_adapter_path"),
                initial_adapter_path=raw["red"].get("initial_adapter_path"),
                update_count=int(raw["red"].get("update_count", 0)),
                reset_count=int(raw["red"].get("reset_count", 0)),
                last_reset_reason=raw["red"].get("last_reset_reason"),
            ),
            socratic=SocraticState(
                active_model_path=raw["socratic"].get("active_model_path"),
                active_adapter_path=raw["socratic"].get("active_adapter_path"),
                update_count=int(raw["socratic"].get("update_count", 0)),
            ),
        )

    def save_state(self, state: PipelineState) -> None:
        _write_json(self.state_path, asdict(state))

    def load_seen_tasks(self) -> set[str]:
        raw = _read_json(self.seen_tasks_path, [])
        return {str(item) for item in raw}

    def save_seen_tasks(self, fingerprints: set[str]) -> None:
        _write_json(self.seen_tasks_path, sorted(fingerprints))

    def append_event(self, event: str, payload: dict[str, Any]) -> None:
        _ensure_parent(self.events_path)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n")

    def save_round_snapshot(self, round_index: int, payload: dict[str, Any]) -> Path:
        path = self.round_snapshots_dir / f"round_{round_index:05d}.json"
        _write_json(path, payload)
        return path
