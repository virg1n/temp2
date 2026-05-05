from __future__ import annotations

import random
import statistics
import uuid
from pathlib import Path
from typing import Any

from .config import ACTConfig
from .curriculum import TopicSampler
from .execution import validate_task
from .judge import SocraticJudge
from .openai_client import OpenAIChatClient
from .prompts import format_socratic_prompt
from .red_generation import canonical_red_completion, generate_red_candidate, repair_buggy_solution
from .schemas import RedTask
from .socratic_generation import SocraticGenerator
from .storage import append_jsonl, ensure_dir, write_json


class ACTPipeline:
    def __init__(self, cfg: ACTConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.run.seed)
        self.output_dir = ensure_dir(cfg.run.output_dir)
        self.red_client = OpenAIChatClient(
            base_url=cfg.models.vllm_base_url,
            api_key=cfg.models.vllm_api_key,
            model=cfg.models.red_model,
        )
        judge_client = OpenAIChatClient(
            base_url=cfg.models.vllm_base_url,
            api_key=cfg.models.vllm_api_key,
            model=cfg.models.judge_model,
        )
        self.judge = SocraticJudge(judge_client, cfg)
        self.socratic = SocraticGenerator(cfg)
        self.topic_sampler = TopicSampler(cfg, self.rng)

    @property
    def paths(self) -> dict[str, Path]:
        return {
            "tasks": self.output_dir / "tasks.jsonl",
            "red_rejections": self.output_dir / "red_rejections.jsonl",
            "rollouts": self.output_dir / "judged_rollouts.jsonl",
            "socratic_preferences": self.output_dir / "socratic_preferences.jsonl",
            "red_kpo": self.output_dir / "red_kpo_groups.jsonl",
            "topic_state": self.output_dir / "topic_state.json",
        }

    def run(self, iterations: int | None = None) -> None:
        total = iterations if iterations is not None else self.cfg.run.max_iterations
        for iteration in range(1, total + 1):
            topic = self.topic_sampler.sample()
            task_scores: list[float] = []
            red_group: dict[str, Any] | None = None

            for candidate_idx in range(self.cfg.generation.red_candidates_per_topic):
                task, candidate_record = generate_red_candidate(self.red_client, topic, self.rng, self.cfg)
                if red_group is None:
                    red_group = {
                        "iteration": iteration,
                        "topic": topic.name,
                        "prompt": candidate_record["prompt"],
                        "messages": candidate_record["messages"],
                        "candidates": [],
                    }

                if task is None:
                    red_group["candidates"].append(candidate_record)
                    append_jsonl(self.paths["red_rejections"], candidate_record)
                    continue

                validation = validate_task(task, self.cfg.execution)
                if not validation.valid:
                    if "buggy_solution passes all asserts" in validation.rejection_reasons:
                        repair_code, repair_record = repair_buggy_solution(self.red_client, task, self.rng, self.cfg)
                        candidate_record["repair"] = repair_record
                        if repair_code is not None:
                            task.buggy_solution = repair_code
                            candidate_record["completion"] = canonical_red_completion(task)
                            validation = validate_task(task, self.cfg.execution)
                        if not validation.valid:
                            candidate_record["repair_rejection_reasons"] = validation.rejection_reasons

                if not validation.valid:
                    candidate_record["valid"] = False
                    candidate_record["score"] = 0.0
                    candidate_record["rejection_reasons"] = validation.rejection_reasons
                    red_group["candidates"].append(candidate_record)
                    append_jsonl(self.paths["red_rejections"], candidate_record)
                    continue

                task_id = str(uuid.uuid4())
                task_record = {
                    "task_id": task_id,
                    "iteration": iteration,
                    "candidate_idx": candidate_idx,
                    "task": task.to_json(),
                    "validation": validation.to_json(),
                }
                append_jsonl(self.paths["tasks"], task_record)

                mean_score = self._run_socratic_rollouts(task_id, task, validation.error_message)
                task_scores.append(mean_score)

                red_reward = self._red_reward(mean_score)
                candidate_record["valid"] = True
                candidate_record["score"] = red_reward
                candidate_record["task_id"] = task_id
                candidate_record["mean_socratic_score"] = mean_score
                red_group["candidates"].append(candidate_record)

            if red_group and len(red_group["candidates"]) >= 2:
                append_jsonl(self.paths["red_kpo"], red_group)

            if task_scores:
                self.topic_sampler.update(topic.name, statistics.mean(task_scores))
            write_json(self.paths["topic_state"], self.topic_sampler.state_json())

            mean_text = f"{statistics.mean(task_scores):.2f}" if task_scores else "no valid task"
            print(f"[ACT] iteration={iteration} topic={topic.name!r} mean_socratic_score={mean_text}", flush=True)

    def _run_socratic_rollouts(self, task_id: str, task: RedTask, error_message: str) -> float:
        prompt = format_socratic_prompt(task.buggy_solution, error_message)
        hints = self.socratic.generate(prompt, self.cfg.generation.socratic_candidates)
        judged: list[dict[str, Any]] = []

        for hint in hints:
            judge_result = self.judge.score(task.buggy_solution, error_message, hint)
            record = {
                "task_id": task_id,
                "topic": task.topic,
                "prompt": prompt,
                "hint": hint,
                "judge": judge_result.to_json(),
            }
            judged.append(record)
            append_jsonl(self.paths["rollouts"], record)

        scores = [record["judge"]["final_score"] for record in judged]
        best_idx = max(range(len(judged)), key=lambda idx: scores[idx])
        worst_idx = min(range(len(judged)), key=lambda idx: scores[idx])
        gap = scores[best_idx] - scores[worst_idx]
        if gap >= self.cfg.curriculum.min_score_gap:
            append_jsonl(
                self.paths["socratic_preferences"],
                {
                    "task_id": task_id,
                    "topic": task.topic,
                    "prompt": prompt,
                    "chosen": judged[best_idx]["hint"],
                    "rejected": judged[worst_idx]["hint"],
                    "chosen_score": scores[best_idx],
                    "rejected_score": scores[worst_idx],
                    "score_gap": gap,
                },
            )

        return statistics.mean(scores)

    def _red_reward(self, mean_socratic_score: float) -> float:
        hardness = 1.0 - max(0.0, min(1.0, mean_socratic_score / 10.0))
        validity_floor = 0.35
        reward = validity_floor + (1.0 - validity_floor) * hardness
        if mean_socratic_score <= self.cfg.curriculum.hard_reward_max:
            reward += 0.1
        return max(0.0, min(1.0, reward))
