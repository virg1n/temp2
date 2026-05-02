from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import (
    CurriculumState,
    EpisodeRecord,
    PythonTask,
    RedRejectedExample,
    RedTrainingExample,
    SocraticPreferenceExample,
)


def _jsonl_line(payload: Dict[str, Any]) -> str:
    # ensure_ascii escapes U+2028/U+2029 and other generated non-ASCII
    # separators so a JSON object remains one physical LF-delimited record.
    return json.dumps(payload, ensure_ascii=True) + "\n"


def _jsonl_physical_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _load_jsonl_payloads(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    lines = _jsonl_physical_lines(path)
    if limit is not None:
        limit_value = max(0, int(limit))
        if limit_value <= 0:
            return []
        lines = lines[-limit_value:]

    payloads: List[Dict[str, Any]] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _counts_by_id(example_ids: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in example_ids:
        key = str(value or "").strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


class SimpleStorage:
    def __init__(self, root_dir: str, *, keep_last_n_checkpoints: int = 3, hard_buffer_max_size: int = 2048) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n_checkpoints = int(keep_last_n_checkpoints)
        self.hard_buffer_max_size = int(hard_buffer_max_size)

        self.state_dir = self.root_dir / "state"
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.episodes_path = self.root_dir / "episodes.jsonl"
        self.hard_examples_path = self.root_dir / "hard_examples.jsonl"
        self.red_rejections_path = self.root_dir / "red_rejections.jsonl"
        self.socratic_preferences_path = self.root_dir / "socratic_preferences.jsonl"
        self.pointers_path = self.state_dir / "pointers.json"
        self.curriculum_path = self.state_dir / "curriculum.json"

    def append_episode(self, episode: EpisodeRecord) -> None:
        with self.episodes_path.open("a", encoding="utf-8") as fh:
            fh.write(_jsonl_line(episode.to_dict()))

    def episode_count(self) -> int:
        return len(_load_jsonl_payloads(self.episodes_path))

    def load_recent_episodes(self, limit: int) -> List[EpisodeRecord]:
        return [EpisodeRecord.from_dict(item) for item in _load_jsonl_payloads(self.episodes_path, limit)]

    def append_hard_example(self, example: RedTrainingExample) -> None:
        rows = self.load_hard_examples(limit=self.hard_buffer_max_size - 1)
        rows.append(example)
        self.save_hard_examples(rows[-self.hard_buffer_max_size :])

    def save_hard_examples(self, rows: List[RedTrainingExample]) -> None:
        with self.hard_examples_path.open("w", encoding="utf-8") as fh:
            for row in rows[-self.hard_buffer_max_size :]:
                fh.write(_jsonl_line(row.to_dict()))

    def load_hard_examples(self, limit: Optional[int] = None) -> List[RedTrainingExample]:
        rows: List[RedTrainingExample] = []
        for payload in _load_jsonl_payloads(self.hard_examples_path, limit):
            payload["task"] = PythonTask(**payload["task"])
            rows.append(RedTrainingExample(**payload))
        return rows

    def append_red_rejected_example(self, example: RedRejectedExample) -> None:
        rows = self.load_red_rejected_examples(limit=self.hard_buffer_max_size - 1)
        rows.append(example)
        self.save_red_rejected_examples(rows[-self.hard_buffer_max_size :])

    def save_red_rejected_examples(self, rows: List[RedRejectedExample]) -> None:
        with self.red_rejections_path.open("w", encoding="utf-8") as fh:
            for row in rows[-self.hard_buffer_max_size :]:
                fh.write(_jsonl_line(row.to_dict()))

    def load_red_rejected_examples(self, limit: Optional[int] = None) -> List[RedRejectedExample]:
        return [RedRejectedExample(**payload) for payload in _load_jsonl_payloads(self.red_rejections_path, limit)]

    def append_socratic_preference(self, example: SocraticPreferenceExample) -> None:
        rows = self.load_socratic_preferences(limit=self.hard_buffer_max_size - 1)
        rows.append(example)
        self.save_socratic_preferences(rows[-self.hard_buffer_max_size :])

    def save_socratic_preferences(self, rows: List[SocraticPreferenceExample]) -> None:
        with self.socratic_preferences_path.open("w", encoding="utf-8") as fh:
            for row in rows[-self.hard_buffer_max_size :]:
                fh.write(_jsonl_line(row.to_dict()))

    def load_socratic_preferences(self, limit: Optional[int] = None) -> List[SocraticPreferenceExample]:
        return [
            SocraticPreferenceExample.from_dict(payload)
            for payload in _load_jsonl_payloads(self.socratic_preferences_path, limit)
        ]

    def increment_hard_example_use_counts(self, example_ids: List[str], metadata_key: str) -> None:
        counts = _counts_by_id(example_ids)
        if not counts:
            return
        rows = self.load_hard_examples()
        for row in rows:
            increment = counts.get(str(row.example_id))
            if not increment:
                continue
            metadata = dict(row.metadata or {})
            metadata[metadata_key] = int(metadata.get(metadata_key) or 0) + increment
            row.metadata = metadata
        self.save_hard_examples(rows)

    def increment_red_rejection_use_counts(self, example_ids: List[str], metadata_key: str) -> None:
        counts = _counts_by_id(example_ids)
        if not counts:
            return
        rows = self.load_red_rejected_examples()
        for row in rows:
            increment = counts.get(str(row.example_id))
            if not increment:
                continue
            metadata = dict(row.metadata or {})
            metadata[metadata_key] = int(metadata.get(metadata_key) or 0) + increment
            row.metadata = metadata
        self.save_red_rejected_examples(rows)

    def increment_socratic_preference_use_counts(self, example_ids: List[str], metadata_key: str) -> None:
        counts = _counts_by_id(example_ids)
        if not counts:
            return
        rows = self.load_socratic_preferences()
        for row in rows:
            increment = counts.get(str(row.example_id))
            if not increment:
                continue
            metadata = dict(row.metadata or {})
            metadata[metadata_key] = int(metadata.get(metadata_key) or 0) + increment
            row.metadata = metadata
        self.save_socratic_preferences(rows)

    def save_curriculum_state(self, state: CurriculumState) -> None:
        self.curriculum_path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")

    def load_curriculum_state(self) -> Optional[CurriculumState]:
        if not self.curriculum_path.exists():
            return None
        payload = json.loads(self.curriculum_path.read_text(encoding="utf-8"))
        return CurriculumState(**payload)

    def load_pointers(self) -> Dict[str, Any]:
        if not self.pointers_path.exists():
            return {}
        return json.loads(self.pointers_path.read_text(encoding="utf-8"))

    def save_pointer(self, name: str, value: Any) -> None:
        payload = self.load_pointers()
        payload[name] = value
        self.pointers_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def checkpoint_dir(self, role: str, step: int) -> Path:
        target = self.checkpoints_dir / role / f"step_{step:06d}"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def prune_role_checkpoints(self, role: str) -> None:
        role_dir = self.checkpoints_dir / role
        if not role_dir.exists():
            return
        items = sorted([path for path in role_dir.iterdir() if path.is_dir()])
        for stale in items[:-self.keep_last_n_checkpoints]:
            shutil.rmtree(stale, ignore_errors=True)
