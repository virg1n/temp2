from __future__ import annotations

import json
from pathlib import Path

from prompts import build_red_training_prompt, serialize_task_for_training
from schemas import (
    BufferSettings,
    DpoTrainingPair,
    RecordLabel,
    RedSFTExample,
    ReplayRecord,
    TaskCandidate,
    TaskOutcome,
    dataclass_to_dict,
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _task_from_dict(data: dict) -> TaskCandidate:
    return TaskCandidate(
        task_id=str(data["task_id"]),
        topic=str(data["topic"]),
        task_statement=str(data["task_statement"]),
        buggy_python=str(data["buggy_python"]),
        asserts=[str(item) for item in data.get("asserts", [])],
        bug_summary=str(data.get("bug_summary", "")),
        educational_value=str(data.get("educational_value", "")),
        source_model=str(data.get("source_model", "")),
        source_adapter_path=data.get("source_adapter_path"),
        observed_failure=data.get("observed_failure"),
        raw_payload=dict(data.get("raw_payload", {})),
    )


def _outcome_from_dict(data: dict) -> TaskOutcome:
    return TaskOutcome(
        task_id=str(data["task_id"]),
        topic=str(data["topic"]),
        average_reward=float(data["average_reward"]),
        best_reward=float(data["best_reward"]),
        hint_count=int(data["hint_count"]),
        label=data["label"],
        validation_score=float(data["validation_score"]),
        validation_feedback=str(data["validation_feedback"]),
    )


class ReplayBuffer:
    def __init__(self, settings: BufferSettings) -> None:
        self.settings = settings
        self.path = Path(settings.replay_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[ReplayRecord] = []
        for row in _load_jsonl(self.path):
            self.records.append(
                ReplayRecord(
                    task=_task_from_dict(row["task"]),
                    outcome=_outcome_from_dict(row["outcome"]),
                    label=row["label"],
                )
            )

    def add(self, record: ReplayRecord) -> None:
        self.records.append(record)
        rewrite_required = False
        if len(self.records) > self.settings.max_records:
            self.records = self.records[-self.settings.max_records :]
            rewrite_required = True
        if rewrite_required:
            self._rewrite_file()
        else:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dataclass_to_dict(record), ensure_ascii=False) + "\n")

    def add_many(self, records: list[ReplayRecord]) -> None:
        for record in records:
            self.add(record)

    def _rewrite_file(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(dataclass_to_dict(record), ensure_ascii=False) + "\n")

    def hard_positive_records(self) -> list[ReplayRecord]:
        return [record for record in self.records if record.label == "hard_positive"]

    def negative_records(self) -> list[ReplayRecord]:
        return [record for record in self.records if record.label in {"easy_negative", "invalid_negative"}]

    def build_red_sft_examples(self, topic_descriptions: dict[str, str]) -> list[RedSFTExample]:
        examples: list[RedSFTExample] = []
        for record in self.records:
            if record.label not in {"hard_positive", "useful_positive"}:
                continue
            examples.append(
                RedSFTExample(
                    prompt=build_red_training_prompt(record.task.topic, topic_descriptions.get(record.task.topic, record.task.topic)),
                    response=serialize_task_for_training(record.task),
                    topic=record.task.topic,
                    task_id=record.task.task_id,
                )
            )
        return examples

    def build_dpo_pairs(self, topic_descriptions: dict[str, str]) -> list[DpoTrainingPair]:
        positives = self.hard_positive_records()
        negatives = self.negative_records()
        negatives_by_topic: dict[str, list[ReplayRecord]] = {}
        for negative in negatives:
            negatives_by_topic.setdefault(negative.task.topic, []).append(negative)

        pairs: list[DpoTrainingPair] = []
        fallback_negatives = negatives[:]
        for positive in positives:
            rejected = None
            same_topic = negatives_by_topic.get(positive.task.topic, [])
            if same_topic:
                rejected = same_topic.pop(0)
            elif fallback_negatives:
                rejected = fallback_negatives.pop(0)
            if rejected is None:
                continue

            prompt = build_red_training_prompt(
                positive.task.topic,
                topic_descriptions.get(positive.task.topic, positive.task.topic),
            )
            pairs.append(
                DpoTrainingPair(
                    prompt=prompt,
                    chosen=serialize_task_for_training(positive.task),
                    rejected=serialize_task_for_training(rejected.task),
                    topic=positive.task.topic,
                    chosen_task_id=positive.task.task_id,
                    rejected_task_id=rejected.task.task_id,
                )
            )
        return pairs


def classify_record_label(
    average_reward: float,
    *,
    failure_threshold: float,
    easy_threshold: float,
    valid: bool,
) -> RecordLabel:
    if not valid:
        return "invalid_negative"
    if average_reward <= failure_threshold:
        return "hard_positive"
    if average_reward >= easy_threshold:
        return "easy_negative"
    return "useful_positive"
