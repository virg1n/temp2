from __future__ import annotations

import json
from typing import Iterable

from schemas import CurriculumTopic, HintCandidate, TaskCandidate


SOCRATIC_SYSTEM_PROMPT = (
    "You are Socratic, a Python debugging tutor. "
    "Respond with short guiding questions and small hints only. "
    "Do not reveal the full corrected solution, final code, or direct step-by-step fix. "
    "Do not output chain-of-thought, hidden reasoning, or <think> tags. "
    "Keep the tone instructional and concise."
)

RED_SYSTEM_PROMPT = (
    "You are Red, an adversarial curriculum generator for Python debugging practice. "
    "Generate educational tasks with realistic buggy code. "
    "Return strict JSON only. Each task must include a topic, a clear task statement, buggy Python code, "
    "failing asserts, a short bug summary for teachers, and a brief note about educational value. "
    "The buggy code must look plausible for a learner at the target topic."
)

JUDGE_TASK_VALIDATION_SYSTEM_PROMPT = (
    "You are Judge, a frozen evaluator for Python tutoring tasks. "
    "Evaluate whether each task is on-topic, educationally valid, and suitable for debugging practice. "
    "Do not rewrite the task. Return strict JSON only."
)

JUDGE_HINT_ASSESS_SYSTEM_PROMPT = (
    "You are Judge, a strict evaluator of Socratic Python debugging hints. "
    "For each hint, score it from 0.0 to 1.0 on these dimensions: "
    "no_solution_reveal, bug_localization, usefulness, socratic_style, technical_accuracy. "
    "Give lower scores if the hint gives away the answer, provides corrected code, or uses hidden reasoning markers. "
    "Return strict JSON only."
)


def build_red_generation_messages(topic: CurriculumTopic, num_candidates: int) -> list[dict[str, str]]:
    user_prompt = (
        f"Generate {num_candidates} distinct Python debugging tasks for the topic '{topic.name}'.\n\n"
        f"Topic description:\n{topic.description}\n\n"
        "Return a JSON array. Each item must have this schema:\n"
        "{\n"
        '  "topic": "string",\n'
        '  "task_statement": "string",\n'
        '  "buggy_python": "full buggy Python program without markdown fences",\n'
        '  "asserts": ["assert ...", "assert ..."],\n'
        '  "bug_summary": "short teacher-only bug description",\n'
        '  "educational_value": "why this is useful for learning"\n'
        "}\n\n"
        "Requirements:\n"
        "- The asserts must fail on the buggy program as written.\n"
        "- Keep tasks self-contained.\n"
        "- Do not include explanations outside JSON.\n"
        "- Prefer bugs that expose reasoning weaknesses, edge cases, or misconceptions within the topic."
    )
    return [
        {"role": "system", "content": RED_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_socratic_user_prompt(task: TaskCandidate) -> str:
    assert_block = "\n".join(task.asserts) if task.asserts else "# no asserts provided"
    return (
        "You are helping a student debug Python code.\n\n"
        f"## Topic\n{task.topic}\n\n"
        f"## Task\n{task.task_statement}\n\n"
        f"## Buggy Code\n```python\n{task.buggy_python.rstrip()}\n```\n\n"
        f"## Failing Asserts\n```python\n{assert_block}\n```\n\n"
        "Respond with 2-3 short Socratic hints or leading questions. "
        "Do not reveal the full corrected answer or corrected code."
    )


def build_socratic_messages(task: TaskCandidate) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SOCRATIC_SYSTEM_PROMPT},
        {"role": "user", "content": build_socratic_user_prompt(task)},
    ]


def build_task_validation_messages(tasks: Iterable[TaskCandidate]) -> list[dict[str, str]]:
    payload = []
    for task in tasks:
        payload.append(
            {
                "task_id": task.task_id,
                "topic": task.topic,
                "task_statement": task.task_statement,
                "buggy_python": task.buggy_python,
                "asserts": task.asserts,
                "educational_value": task.educational_value,
            }
        )
    user_prompt = (
        "Evaluate the following candidate debugging tasks.\n"
        "For each item, return JSON with fields:\n"
        '[{"task_id":"...", "passed": true, "score": 0.0, "feedback": "short reason"}]\n\n'
        "Scoring guidance:\n"
        "- passed=true only if the task is clearly on-topic and educationally valid.\n"
        "- score should be between 0.0 and 1.0.\n"
        "- feedback should be short and specific.\n\n"
        f"Tasks:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": JUDGE_TASK_VALIDATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_hint_assessment_messages(items: Iterable[tuple[str, str, str]]) -> list[dict[str, str]]:
    payload = []
    for item_id, prompt_text, hint_text in items:
        payload.append(
            {
                "item_id": item_id,
                "task_prompt": prompt_text,
                "hint": hint_text,
            }
        )
    user_prompt = (
        "Evaluate the following Socratic hints for Python debugging tasks.\n"
        "For each item, return JSON with fields:\n"
        "[{"
        '"item_id":"...", '
        '"no_solution_reveal":0.0, '
        '"bug_localization":0.0, '
        '"usefulness":0.0, '
        '"socratic_style":0.0, '
        '"technical_accuracy":0.0, '
        '"feedback":"short reason"'
        "}]\n\n"
        "All scores must be in [0.0, 1.0]. Penalize direct answers, corrected code, and hidden reasoning markers.\n\n"
        f"Items:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": JUDGE_HINT_ASSESS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def serialize_task_for_training(task: TaskCandidate) -> str:
    payload = {
        "topic": task.topic,
        "task_statement": task.task_statement,
        "buggy_python": task.buggy_python,
        "asserts": task.asserts,
        "bug_summary": task.bug_summary,
        "educational_value": task.educational_value,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_red_training_prompt(topic: str, topic_description: str) -> str:
    return (
        "Generate one Python debugging task as strict JSON for the following topic.\n\n"
        f"Topic: {topic}\n"
        f"Description: {topic_description}\n\n"
        "Return the same schema used during generation: "
        "topic, task_statement, buggy_python, asserts, bug_summary, educational_value."
    )


def render_hint_batch(items: Iterable[HintCandidate]) -> list[tuple[str, str, str]]:
    return [(item.hint_id, item.prompt_text, item.text) for item in items]
