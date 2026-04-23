from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskExecutionResult:
    status: str
    returncode: int
    error_message: str
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PythonTask:
    task_id: str
    topic: str
    statement: str
    buggy_solution: str
    failing_asserts: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def non_empty_line_count(self) -> int:
        return len([line for line in (self.buggy_solution or "").splitlines() if line.strip()])

    def combined_program(self) -> str:
        body = (self.buggy_solution or "").rstrip()
        tests = "\n".join(x.strip() for x in self.failing_asserts if x.strip())
        if not body:
            return tests
        if not tests:
            return body
        return f"{body}\n\n{tests}".rstrip()

    def observed_failure(self) -> str:
        return str(self.metadata.get("observed_failure") or "AssertionError")

    def to_grpo_example(self) -> Dict[str, Any]:
        return {
            "problem": self.statement,
            "code": self.combined_program(),
            "observed_failure": self.observed_failure(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SocraticHint:
    task_id: str
    text: str
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeOutput:
    task_id: str
    score: float
    normalized_reward: float
    raw_text: str
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CurriculumState:
    weights: Dict[str, float]
    running_topic_rewards: Dict[str, float]
    recent_topics: List[str]
    consecutive_topic: Optional[str]
    consecutive_count: int
    resets: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RedTaskSpec:
    topic: str
    target_function: str
    intended_bug: str
    expected_first_failure: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RedTrainingExample:
    example_id: str
    topic: str
    prompt: str
    chosen_completion: str
    rejected_completion: Optional[str]
    reward: float
    task: PythonTask
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["task"] = self.task.to_dict()
        return payload


@dataclass
class RedRejectedExample:
    example_id: str
    topic: str
    prompt: str
    rejected_completion: str
    rejection_reason: str
    task_quality: Optional[float] = None
    spec: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SocraticPreferenceExample:
    example_id: str
    topic: str
    task: PythonTask
    chosen_hint: str
    rejected_hint: str
    chosen_score: float
    rejected_score: float
    chosen_judge: Dict[str, Any] = field(default_factory=dict)
    rejected_judge: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["task"] = self.task.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SocraticPreferenceExample":
        row = dict(payload)
        row["task"] = PythonTask(**row["task"])
        return cls(**row)


@dataclass
class EpisodeRecord:
    episode_id: int
    topic: str
    task: PythonTask
    hint: SocraticHint
    judge: JudgeOutput
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "topic": self.topic,
            "task": self.task.to_dict(),
            "hint": self.hint.to_dict(),
            "judge": self.judge.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EpisodeRecord":
        task = PythonTask(**payload["task"])
        hint = SocraticHint(**payload["hint"])
        judge = JudgeOutput(**payload["judge"])
        return cls(
            episode_id=int(payload["episode_id"]),
            topic=str(payload["topic"]),
            task=task,
            hint=hint,
            judge=judge,
            metadata=dict(payload.get("metadata") or {}),
        )
