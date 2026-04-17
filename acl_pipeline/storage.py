from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import CurriculumState, EpisodeRecord, PythonTask, RedRejectedExample, RedTrainingExample


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
        self.pointers_path = self.state_dir / "pointers.json"
        self.curriculum_path = self.state_dir / "curriculum.json"

    def append_episode(self, episode: EpisodeRecord) -> None:
        with self.episodes_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")

    def episode_count(self) -> int:
        if not self.episodes_path.exists():
            return 0
        return len(self.episodes_path.read_text(encoding="utf-8").splitlines())

    def load_recent_episodes(self, limit: int) -> List[EpisodeRecord]:
        if not self.episodes_path.exists():
            return []
        lines = self.episodes_path.read_text(encoding="utf-8").splitlines()
        items = [json.loads(line) for line in lines[-max(0, int(limit)) :]]
        return [EpisodeRecord.from_dict(item) for item in items]

    def append_hard_example(self, example: RedTrainingExample) -> None:
        rows = self.load_hard_examples(limit=self.hard_buffer_max_size - 1)
        rows.append(example)
        with self.hard_examples_path.open("w", encoding="utf-8") as fh:
            for row in rows[-self.hard_buffer_max_size :]:
                fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    def load_hard_examples(self, limit: Optional[int] = None) -> List[RedTrainingExample]:
        if not self.hard_examples_path.exists():
            return []
        lines = self.hard_examples_path.read_text(encoding="utf-8").splitlines()
        if limit is not None:
            lines = lines[-int(limit) :]
        rows: List[RedTrainingExample] = []
        for line in lines:
            payload = json.loads(line)
            payload["task"] = PythonTask(**payload["task"])
            rows.append(RedTrainingExample(**payload))
        return rows

    def append_red_rejected_example(self, example: RedRejectedExample) -> None:
        rows = self.load_red_rejected_examples(limit=self.hard_buffer_max_size - 1)
        rows.append(example)
        with self.red_rejections_path.open("w", encoding="utf-8") as fh:
            for row in rows[-self.hard_buffer_max_size :]:
                fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    def load_red_rejected_examples(self, limit: Optional[int] = None) -> List[RedRejectedExample]:
        if not self.red_rejections_path.exists():
            return []
        lines = self.red_rejections_path.read_text(encoding="utf-8").splitlines()
        if limit is not None:
            lines = lines[-int(limit) :]
        return [RedRejectedExample(**json.loads(line)) for line in lines]

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
