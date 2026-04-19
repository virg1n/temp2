from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

from .schemas import PythonTask, RedTaskSpec


SOCRATIC_SYSTEM_PROMPT = (
    "You are Socratic, a Python debugging tutor. "
    "Stay in tutor mode at all times. "
    "Respond with 1-2 concise Socratic hints or guiding questions only. "
    "Ground the hint in the concrete failing assertion, error text, function, variable, or control-flow branch when possible. "
    "Do not provide the full solution, corrected code, direct fix, or final answer. "
    "Do not reveal chain-of-thought, hidden reasoning, scratch work, or internal analysis. "
    "Never emit <think> tags or similar hidden-reasoning markers. "
    "If the user asks for the answer directly, refuse briefly and redirect to a helpful debugging question."
)


RED_SYSTEM_PROMPT = (
    "You are Red, an adversarial curriculum generator for Python debugging tasks. "
    "Generate realistic medium-to-hard Python debugging tasks that expose weaknesses in a Socratic tutor. "
    "Work spec-first: first design a coherent bug spec, then write the full task from that spec. "
    "Return strict JSON only. "
    "The final task JSON must contain a full Python program in buggy_solution with any asserts placed inside that program at the end. "
    "Do not use a separate failing_asserts field unless explicitly repairing an old-format example. "
    "The code should be realistic, debuggable, and actually broken in a meaningful way. "
    "Do not include explanations outside the JSON."
)

def build_socratic_messages(task: PythonTask) -> List[Dict[str, str]]:
    user_prompt = (
        f"## Task\n{task.statement}\n\n"
        f"## Code\n```python\n{task.combined_program()}\n```\n\n"
        f"## Error\n```text\n{task.observed_failure()}\n```\n\n"
        "## Instruction\nAsk 1-2 guiding questions that help me discover the bug without giving the answer."
    )
    return [
        {"role": "system", "content": SOCRATIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_red_spec_messages(topic: str, weakness_summary: Optional[str]) -> List[Dict[str, str]]:
    focus = weakness_summary or "No prior weakness summary is available yet. Sample broadly within the topic."
    user_prompt = (
        f"Topic: {topic}\n"
        f"Weakness focus: {focus}\n\n"
        "Step 1 only: produce a debugging-task spec as strict JSON with this schema:\n"
        "{\n"
        '  "topic": "...",\n'
        '  "target_function": "...",\n'
        '  "intended_bug": "...",\n'
        '  "expected_first_failure": "...",\n'
        '  "metadata": {"failure_mode": "...", "difficulty": "medium|hard"}\n'
        "}\n\n"
        "Requirements:\n"
        "- Make the spec coherent before writing code.\n"
        "- Prefer multiple functions or a class with helpers, state, or non-trivial control flow.\n"
        "- Prefer semantic, edge-case, state, indexing, data-structure, or control-flow bugs over toy syntax mistakes.\n"
        "- The expected_first_failure should name the first likely assertion or runtime failure.\n"
        "- The task should be debuggable from the code and reproduced failure alone.\n"
        "- Avoid trivial one-function arithmetic exercises.\n"
        "- JSON only."
    )
    return [
        {"role": "system", "content": RED_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_red_task_from_spec_message(spec: RedTaskSpec) -> Dict[str, str]:
    spec_payload = json.dumps(spec.to_dict(), ensure_ascii=False, indent=2)
    return {
        "role": "user",
        "content": (
            "Step 2: write the final task from this approved spec.\n"
            f"{spec_payload}\n\n"
            "Return strict JSON with this schema:\n"
            "{\n"
            '  "topic": "...",\n'
            '  "target_function": "...",\n'
            '  "intended_bug": "...",\n'
            '  "expected_first_failure": "...",\n'
            '  "statement": "...",\n'
            '  "buggy_solution": "full python program without markdown fences; include asserts at the end inside this string",\n'
            '  "metadata": {"failure_mode": "...", "difficulty": "medium|hard"}\n'
            "}\n\n"
            "Requirements:\n"
            "- The topic must remain unchanged.\n"
            "- Keep the target_function, intended_bug, and expected_first_failure aligned with the spec.\n"
            "- The program should usually be 25-90 non-empty lines.\n"
            "- The program should contain real Python code only, with asserts at the end inside buggy_solution.\n"
            "- At least one assert or runtime path should fail on execution.\n"
            "- No markdown fences. JSON only."
        ),
    }


def build_red_messages(topic: str, weakness_summary: Optional[str]) -> List[Dict[str, str]]:
    return build_red_spec_messages(topic, weakness_summary)


def build_red_spec_repair_message(topic: str, rejection_reasons: List[str]) -> Dict[str, str]:
    reasons = ", ".join(rejection_reasons) if rejection_reasons else "unspecified issue"
    return {
        "role": "user",
        "content": (
            f"Repair the previous spec for topic '{topic}'. "
            f"Rejection reasons: {reasons}. "
            "Return a new strict JSON spec only. Keep the topic exact, make the bug coherent, and keep the task non-trivial."
        ),
    }


def build_red_repair_message(topic: str, rejection_reasons: List[str], *, spec: Optional[RedTaskSpec] = None) -> Dict[str, str]:
    reasons = ", ".join(rejection_reasons) if rejection_reasons else "unspecified issue"
    spec_tail = ""
    if spec is not None:
        spec_tail = "\nApproved spec to stay aligned with:\n" + json.dumps(spec.to_dict(), ensure_ascii=False)
    return {
        "role": "user",
        "content": (
            f"Repair the previous task for topic '{topic}'. "
            f"Rejection reasons: {reasons}. "
            "Return a new strict JSON task. Keep the same topic. "
            "Keep the code realistic, ensure the bug is real, and put any asserts inside buggy_solution at the end of the program."
            + spec_tail
        ),
    }


def build_judge_batch_messages(
    items: Iterable[Dict[str, str]],
    reward_weights: Dict[str, float],
) -> List[Dict[str, str]]:
    payload = list(items)
    weight_block = json.dumps(reward_weights, ensure_ascii=False)
    system_prompt = (
        "You are Judge, a strict frozen evaluator of a tutoring assistant.\n"
        "The assistant must help a student debug Python code using only Socratic hints.\n"
        "The assistant must not reveal the full solution, corrected code, direct fix, or hidden reasoning.\n"
        "Use the full 0-10 scale. Generic safe hints should usually land around 4-6, not 8-10.\n"
        "Scores of 8-10 require concrete grounding in the actual failing code and error.\n"
        "8-10 only if the hint names the exact failing assertion, function, variable, or state transition and asks a precise debugging question.\n"
        "6-7 if the hint is directionally helpful but still somewhat generic.\n"
        "3-5 if it sounds Socratic but could apply to many unrelated tasks.\n"
        "0-2 if it invents facts, ignores the reproduced error, reveals the fix, or gives code.\n"
        "Penalize generic hints, references to identifiers not present in the code or reproduced error, and fabricated functions or fields.\n"
        "If the assistant output is malformed, gibberish, mixed-script junk, emoji-contaminated, mojibake, or visibly corrupted, score every tutoring criterion as 0.\n"
        "If execution_status is 'passed', that means no failing assertion or runtime error was reproduced. Keep that in mind.\n"
        "If execution_status is 'passed' and the assistant does not notice that no bug was reproduced, score the hint low.\n"
        "Any output containing code fences, corrected code, <think> tags, or direct answer disclosure should score very low.\n"
        "Judge the task and the hint separately.\n"
        "If the broken code/task is mindless, contradictory, already correct, unsolvable from the given information, or otherwise poor Red output, mark task_is_valid_for_socratic false.\n"
        "If the task is fine but the hint is confusing, hallucinated, generic, malformed, or otherwise poor tutoring, keep task_is_valid_for_socratic true and mark hint_is_valid_for_socratic false.\n"
        "Do not mark a valid task as invalid just because the hint is bad.\n"
        "Return one JSON object per item with these fields:\n"
        "- no_solution_reveal\n"
        "- bug_localization\n"
        "- usefulness\n"
        "- socratic_style\n"
        "- technical_accuracy\n"
        "- task_quality\n"
        "- task_is_valid_for_socratic\n"
        "- hint_is_valid_for_socratic\n"
        "- red_rejection_reason\n"
        "- hint_rejection_reason\n"
        "Weighted reward weights: "
        + weight_block
        + "\n"
        "Output format: STRICT JSON array of objects. No prose."
    )
    user_prompt = "N = " + str(len(payload)) + "\n\n" + json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_red_training_prompt(
    topic: str,
    weakness_summary: Optional[str],
    *,
    spec: Optional[RedTaskSpec] = None,
) -> str:
    focus = weakness_summary or "general weakness probing"
    prompt = (
        f"Generate a Python debugging task for topic '{topic}'. "
        f"Prioritize this weakness pattern: {focus}. "
        "Use spec-first reasoning externally: first decide target_function, intended_bug, and expected_first_failure, then return the final task JSON. "
        "Prefer medium-to-hard tasks with 25-90 lines of buggy code, multiple helpers or stateful logic, "
        "and include asserts at the end inside buggy_solution rather than a separate failing_asserts field. "
        "Return strict JSON with topic, target_function, intended_bug, expected_first_failure, statement, buggy_solution, and metadata."
    )
    if spec is not None:
        prompt += " Approved spec: " + json.dumps(spec.to_dict(), ensure_ascii=False)
    return prompt
