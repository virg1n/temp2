from __future__ import annotations

import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "judge_lora_dataset"

from acl_pipeline.prompts import build_judge_batch_messages

FULL_PATH = DATA_DIR / "judge_lora_dataset_700.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"

RAW_FULL_PATH = DATA_DIR / "source_samples_700_raw.jsonl"
RAW_TRAIN_PATH = DATA_DIR / "source_train_630_raw.jsonl"
RAW_VAL_PATH = DATA_DIR / "source_val_70_raw.jsonl"

CONFIG_PATH = ROOT / "configs" / "default.yaml"

RUNTIME_KEYS = (
    "no_solution_reveal",
    "bug_localization",
    "usefulness",
    "socratic_style",
    "technical_accuracy",
    "task_quality",
    "task_is_valid_for_socratic",
    "hint_is_valid_for_socratic",
    "red_rejection_reason",
    "hint_rejection_reason",
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_raw_backups() -> None:
    backups = (
        (FULL_PATH, RAW_FULL_PATH),
        (TRAIN_PATH, RAW_TRAIN_PATH),
        (VAL_PATH, RAW_VAL_PATH),
    )
    for src, dst in backups:
        if not dst.exists():
            shutil.copyfile(src, dst)


def reward_weights() -> Dict[str, float]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return dict(config["judge"]["reward_weights"])


def infer_execution_status(observed_failure: str) -> str:
    text = str(observed_failure or "")
    if "No failing assertion or runtime error was reproduced." in text:
        return "passed"
    if "IndentationError" in text or "TabError" in text:
        return "indentation_error"
    if "SyntaxError" in text:
        return "syntax_error"
    if "NameError" in text:
        return "nameerror"
    return "failed"


def runtime_item(sample: Dict[str, Any]) -> Dict[str, Any]:
    observed_failure = str(sample.get("observed_failure") or "")
    return {
        "statement": str(sample.get("statement") or ""),
        "code": str(sample.get("code") or ""),
        "observed_failure": observed_failure,
        "execution_status": infer_execution_status(observed_failure),
        "assistant_response": str(sample.get("hint") or "")[:1800],
    }


def runtime_labels(sample: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(sample.get("labels") or {})
    labels: Dict[str, Any] = {
        "no_solution_reveal": float(raw.get("no_solution_reveal", 0.0)),
        "bug_localization": float(raw.get("bug_localization", 0.0)),
        "usefulness": float(raw.get("usefulness", 0.0)),
        "socratic_style": float(raw.get("socratic_style", 0.0)),
        "technical_accuracy": float(raw.get("technical_accuracy", 0.0)),
        "task_quality": float(raw.get("task_quality", 0.0)),
        "task_is_valid_for_socratic": bool(raw.get("task_is_valid_for_socratic", False)),
        "hint_is_valid_for_socratic": bool(raw.get("hint_is_valid_for_socratic", False)),
        "red_rejection_reason": str(raw.get("red_rejection_reason") or ""),
        "hint_rejection_reason": str(raw.get("hint_rejection_reason") or ""),
    }

    status = infer_execution_status(str(sample.get("observed_failure") or ""))
    if status == "passed":
        labels["task_quality"] = min(labels["task_quality"], 2.0) if labels["task_quality"] else 2.0
        labels["task_is_valid_for_socratic"] = False
        if not labels["red_rejection_reason"]:
            labels["red_rejection_reason"] = "already_correct_code"
    return labels


def sample_view(sample: Dict[str, Any]) -> Dict[str, Any]:
    view = {
        "sample_id": str(sample["sample_id"]),
        "source_group": str(sample.get("source_group") or ""),
        "source_detail": sample.get("source_detail"),
        "topic": str(sample.get("topic") or ""),
        "quality_bucket": str(sample.get("quality_bucket") or ""),
        "label_notes": list(sample.get("label_notes") or []),
        "code_hash": str(sample.get("code_hash") or ""),
        "hint_hash": str(sample.get("hint_hash") or ""),
        "original_execution_status": str(sample.get("execution_status") or ""),
        "runtime_item": runtime_item(sample),
        "runtime_labels": runtime_labels(sample),
    }
    return view


def build_batch_records(
    split_name: str,
    samples: List[Dict[str, Any]],
    weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    assert len(samples) % 4 == 0, f"{split_name} split must be divisible by 4"
    batches: List[Dict[str, Any]] = []
    for batch_index in range(0, len(samples), 4):
        group = samples[batch_index : batch_index + 4]
        payload = [sample["runtime_item"] for sample in group]
        label_array = [sample["runtime_labels"] for sample in group]
        messages = build_judge_batch_messages(payload, weights)
        assistant = json.dumps(label_array, ensure_ascii=False)
        message_rows = [messages[0], messages[1], {"role": "assistant", "content": assistant}]
        batches.append(
            {
                "batch_id": f"{split_name}-{(batch_index // 4) + 1:04d}",
                "split": split_name,
                "batch_size": 4,
                "sample_ids": [sample["sample_id"] for sample in group],
                "topics": [sample["topic"] for sample in group],
                "source_groups": [sample["source_group"] for sample in group],
                "quality_buckets": [sample["quality_bucket"] for sample in group],
                "items": group,
                "messages": message_rows,
            }
        )
    return batches


def overlap_count(left: Iterable[Dict[str, Any]], right: Iterable[Dict[str, Any]], key: str) -> int:
    left_values = {item[key] for item in left}
    right_values = {item[key] for item in right}
    return len(left_values & right_values)


def main() -> None:
    ensure_raw_backups()

    source_samples = load_jsonl(RAW_FULL_PATH)
    raw_train = load_jsonl(RAW_TRAIN_PATH)
    raw_val = load_jsonl(RAW_VAL_PATH)

    source_by_id = {str(sample["sample_id"]): sample_view(sample) for sample in source_samples}
    train_ids = [str(sample["sample_id"]) for sample in raw_train]
    val_ids = [str(sample["sample_id"]) for sample in raw_val]

    if len(train_ids) % 4 != 0:
        move_count = len(train_ids) % 4
        moved = train_ids[-move_count:]
        train_ids = train_ids[:-move_count]
        val_ids = val_ids + moved
    else:
        moved = []

    train_samples = [source_by_id[sample_id] for sample_id in train_ids]
    val_samples = [source_by_id[sample_id] for sample_id in val_ids]

    rng = random.Random(42)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    weights = reward_weights()
    train_batches = build_batch_records("train", train_samples, weights)
    val_batches = build_batch_records("val", val_samples, weights)
    full_batches = train_batches + val_batches

    write_jsonl(FULL_PATH, full_batches)
    write_jsonl(TRAIN_PATH, train_batches)
    write_jsonl(VAL_PATH, val_batches)

    normalized_status_counts = Counter(item["runtime_item"]["execution_status"] for item in source_by_id.values())
    source_counts = Counter(item["source_group"] for item in source_by_id.values())
    quality_counts = Counter(item["quality_bucket"] for item in source_by_id.values())
    task_valid_counts = Counter(str(item["runtime_labels"]["task_is_valid_for_socratic"]).lower() for item in source_by_id.values())
    hint_valid_count = sum(1 for item in source_by_id.values() if item["runtime_labels"]["hint_is_valid_for_socratic"])
    passed_invalid_count = sum(
        1
        for item in source_by_id.values()
        if item["runtime_item"]["execution_status"] == "passed"
        and not item["runtime_labels"]["task_is_valid_for_socratic"]
    )

    manifest = {
        "source_sample_count": len(source_samples),
        "runtime_batch_size": 4,
        "total_batch_records": len(full_batches),
        "train_batch_count": len(train_batches),
        "val_batch_count": len(val_batches),
        "train_item_count": len(train_samples),
        "val_item_count": len(val_samples),
        "source_counts": dict(source_counts),
        "normalized_execution_status_counts": dict(normalized_status_counts),
        "quality_bucket_counts": dict(quality_counts),
        "task_valid_counts": dict(task_valid_counts),
        "hint_valid_rate": round(hint_valid_count / max(1, len(source_samples)), 4),
        "passed_tasks_marked_invalid": passed_invalid_count,
        "train_val_code_hash_overlap": overlap_count(train_samples, val_samples, "code_hash"),
        "train_val_hint_hash_overlap": overlap_count(train_samples, val_samples, "hint_hash"),
        "weights_used_in_prompt": weights,
        "moved_train_ids_to_val_for_divisible_batches": moved,
        "source_backups": {
            "full": str(RAW_FULL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "train": str(RAW_TRAIN_PATH.relative_to(ROOT)).replace("\\", "/"),
            "val": str(RAW_VAL_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
        "notes": [
            "messages now match acl_pipeline.prompts.build_judge_batch_messages exactly",
            "training targets are strict JSON arrays of 4 judge objects to match N=4 runtime batches",
            "execution_status now uses acl_pipeline runtime vocabulary: passed, syntax_error, indentation_error, nameerror, failed",
            "passed tasks are marked task_is_valid_for_socratic=false with task_quality <= 2.0",
            "hint validity remains separate from task validity so passed-aware hints can still be labeled as good hints on invalid tasks",
        ],
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
