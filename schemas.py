from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


JsonDict = dict[str, Any]
TopicLabel = str
RecordLabel = Literal["hard_positive", "useful_positive", "easy_negative", "invalid_negative"]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass(slots=True)
class QuantizationSettings:
    load_in_8bit: bool = False
    llm_int8_threshold: float = 6.0

    @classmethod
    def from_dict(cls, data: JsonDict | None) -> "QuantizationSettings":
        payload = data or {}
        return cls(
            load_in_8bit=bool(payload.get("load_in_8bit", False)),
            llm_int8_threshold=float(payload.get("llm_int8_threshold", 6.0)),
        )


@dataclass(slots=True)
class GenerationSettings:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_return_sequences: int = 1
    enable_thinking: bool = False
    stop_strings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: JsonDict | None) -> "GenerationSettings":
        payload = data or {}
        return cls(
            max_new_tokens=int(payload.get("max_new_tokens", 512)),
            temperature=float(payload.get("temperature", 0.7)),
            top_p=float(payload.get("top_p", 0.95)),
            repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
            do_sample=bool(payload.get("do_sample", True)),
            num_return_sequences=int(payload.get("num_return_sequences", 1)),
            enable_thinking=bool(payload.get("enable_thinking", False)),
            stop_strings=[str(item) for item in _as_list(payload.get("stop_strings"))],
        )


@dataclass(slots=True)
class ModelSettings:
    name: str
    model_name_or_path: str
    adapter_path: str | None = None
    initial_adapter_path: str | None = None
    trust_remote_code: bool = True
    device_map: str | JsonDict = "auto"
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    max_input_length: int = 4096
    quantization: QuantizationSettings = field(default_factory=QuantizationSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)

    @classmethod
    def from_dict(cls, name: str, data: JsonDict) -> "ModelSettings":
        return cls(
            name=name,
            model_name_or_path=str(data["model_name_or_path"]),
            adapter_path=data.get("adapter_path"),
            initial_adapter_path=data.get("initial_adapter_path"),
            trust_remote_code=bool(data.get("trust_remote_code", True)),
            device_map=data.get("device_map", "auto"),
            torch_dtype=str(data.get("torch_dtype", "bfloat16")),
            attn_implementation=data.get("attn_implementation"),
            max_input_length=int(data.get("max_input_length", 4096)),
            quantization=QuantizationSettings.from_dict(data.get("quantization")),
            generation=GenerationSettings.from_dict(data.get("generation")),
        )


@dataclass(slots=True)
class ModelsSettings:
    socratic: ModelSettings
    red: ModelSettings
    judge: ModelSettings

    @classmethod
    def from_dict(cls, data: JsonDict) -> "ModelsSettings":
        return cls(
            socratic=ModelSettings.from_dict("socratic", data["socratic"]),
            red=ModelSettings.from_dict("red", data["red"]),
            judge=ModelSettings.from_dict("judge", data["judge"]),
        )


@dataclass(slots=True)
class CurriculumTopic:
    name: str
    description: str
    initial_weight: float = 1.0

    @classmethod
    def from_dict(cls, data: JsonDict) -> "CurriculumTopic":
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", data["name"])),
            initial_weight=float(data.get("initial_weight", 1.0)),
        )


@dataclass(slots=True)
class CurriculumSettings:
    topics: list[CurriculumTopic]
    sample_topics_per_round: int = 2
    candidates_per_topic: int = 5
    random_sampling_rounds: int = 3
    weak_reward_threshold: float = 0.45
    strong_reward_threshold: float = 0.8
    weak_topic_boost: float = 1.15
    strong_topic_decay: float = 0.92
    min_weight: float = 0.25
    max_weight: float = 5.0
    repeated_topic_reset_threshold: int = 5

    @classmethod
    def from_dict(cls, data: JsonDict) -> "CurriculumSettings":
        topics = [CurriculumTopic.from_dict(item) for item in data.get("topics", [])]
        if not topics:
            raise ValueError("settings.curriculum.topics must contain at least one topic")
        return cls(
            topics=topics,
            sample_topics_per_round=int(data.get("sample_topics_per_round", 2)),
            candidates_per_topic=int(data.get("candidates_per_topic", 5)),
            random_sampling_rounds=int(data.get("random_sampling_rounds", 3)),
            weak_reward_threshold=float(data.get("weak_reward_threshold", 0.45)),
            strong_reward_threshold=float(data.get("strong_reward_threshold", 0.8)),
            weak_topic_boost=float(data.get("weak_topic_boost", 1.15)),
            strong_topic_decay=float(data.get("strong_topic_decay", 0.92)),
            min_weight=float(data.get("min_weight", 0.25)),
            max_weight=float(data.get("max_weight", 5.0)),
            repeated_topic_reset_threshold=int(data.get("repeated_topic_reset_threshold", 5)),
        )


@dataclass(slots=True)
class HintRewardWeights:
    no_solution_reveal: float = 0.3
    bug_localization: float = 0.25
    usefulness: float = 0.2
    socratic_style: float = 0.15
    technical_accuracy: float = 0.1

    @classmethod
    def from_dict(cls, data: JsonDict | None) -> "HintRewardWeights":
        payload = data or {}
        return cls(
            no_solution_reveal=float(payload.get("no_solution_reveal", 0.3)),
            bug_localization=float(payload.get("bug_localization", 0.25)),
            usefulness=float(payload.get("usefulness", 0.2)),
            socratic_style=float(payload.get("socratic_style", 0.15)),
            technical_accuracy=float(payload.get("technical_accuracy", 0.1)),
        )


@dataclass(slots=True)
class JudgeSettings:
    min_valid_tasks: int = 4
    task_validation_batch_size: int = 4
    hint_assessment_batch_size: int = 6
    reward_weights: HintRewardWeights = field(default_factory=HintRewardWeights)

    @classmethod
    def from_dict(cls, data: JsonDict | None) -> "JudgeSettings":
        payload = data or {}
        return cls(
            min_valid_tasks=int(payload.get("min_valid_tasks", 4)),
            task_validation_batch_size=int(payload.get("task_validation_batch_size", 4)),
            hint_assessment_batch_size=int(payload.get("hint_assessment_batch_size", 6)),
            reward_weights=HintRewardWeights.from_dict(payload.get("reward_weights")),
        )


@dataclass(slots=True)
class SocraticTrainingSettings:
    output_dir: str
    update_every_tasks: int = 8
    num_hints_per_task: int = 4
    learning_rate: float = 1e-5
    epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    optim: str = "adamw_torch"
    beta: float = 0.02
    max_prompt_length: int = 2048
    max_completion_length: int = 256
    log_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    full_ft: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=list)
    deepspeed: str | None = None

    @classmethod
    def from_dict(cls, data: JsonDict) -> "SocraticTrainingSettings":
        return cls(
            output_dir=str(data["output_dir"]),
            update_every_tasks=int(data.get("update_every_tasks", 8)),
            num_hints_per_task=int(data.get("num_hints_per_task", 4)),
            learning_rate=float(data.get("learning_rate", 1e-5)),
            epochs=int(data.get("epochs", 1)),
            per_device_batch_size=int(data.get("per_device_batch_size", 1)),
            gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 4)),
            warmup_ratio=float(data.get("warmup_ratio", 0.03)),
            weight_decay=float(data.get("weight_decay", 0.0)),
            optim=str(data.get("optim", "adamw_torch")),
            beta=float(data.get("beta", 0.02)),
            max_prompt_length=int(data.get("max_prompt_length", 2048)),
            max_completion_length=int(data.get("max_completion_length", 256)),
            log_steps=int(data.get("log_steps", 10)),
            save_steps=int(data.get("save_steps", 100)),
            save_total_limit=int(data.get("save_total_limit", 2)),
            bf16=bool(data.get("bf16", True)),
            fp16=bool(data.get("fp16", False)),
            gradient_checkpointing=bool(data.get("gradient_checkpointing", True)),
            full_ft=bool(data.get("full_ft", False)),
            lora_r=int(data.get("lora_r", 32)),
            lora_alpha=int(data.get("lora_alpha", 64)),
            lora_dropout=float(data.get("lora_dropout", 0.05)),
            lora_target_modules=[str(item) for item in _as_list(data.get("lora_target_modules"))],
            deepspeed=data.get("deepspeed"),
        )


@dataclass(slots=True)
class RedTrainingSettings:
    output_dir: str
    update_every_rounds: int = 2
    learning_rate: float = 1e-4
    epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    optim: str = "adamw_torch"
    max_length: int = 3072
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=list)
    dpo_beta: float = 0.1
    log_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    min_positive_examples: int = 12
    min_dpo_pairs: int = 6

    @classmethod
    def from_dict(cls, data: JsonDict) -> "RedTrainingSettings":
        return cls(
            output_dir=str(data["output_dir"]),
            update_every_rounds=int(data.get("update_every_rounds", 2)),
            learning_rate=float(data.get("learning_rate", 1e-4)),
            epochs=int(data.get("epochs", 1)),
            per_device_batch_size=int(data.get("per_device_batch_size", 1)),
            gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 4)),
            warmup_ratio=float(data.get("warmup_ratio", 0.03)),
            weight_decay=float(data.get("weight_decay", 0.0)),
            optim=str(data.get("optim", "adamw_torch")),
            max_length=int(data.get("max_length", 3072)),
            bf16=bool(data.get("bf16", True)),
            fp16=bool(data.get("fp16", False)),
            gradient_checkpointing=bool(data.get("gradient_checkpointing", True)),
            lora_r=int(data.get("lora_r", 64)),
            lora_alpha=int(data.get("lora_alpha", 128)),
            lora_dropout=float(data.get("lora_dropout", 0.05)),
            lora_target_modules=[str(item) for item in _as_list(data.get("lora_target_modules"))],
            dpo_beta=float(data.get("dpo_beta", 0.1)),
            log_steps=int(data.get("log_steps", 10)),
            save_steps=int(data.get("save_steps", 100)),
            save_total_limit=int(data.get("save_total_limit", 2)),
            min_positive_examples=int(data.get("min_positive_examples", 12)),
            min_dpo_pairs=int(data.get("min_dpo_pairs", 6)),
        )


@dataclass(slots=True)
class TrainingSettings:
    socratic: SocraticTrainingSettings
    red: RedTrainingSettings

    @classmethod
    def from_dict(cls, data: JsonDict) -> "TrainingSettings":
        return cls(
            socratic=SocraticTrainingSettings.from_dict(data["socratic"]),
            red=RedTrainingSettings.from_dict(data["red"]),
        )


@dataclass(slots=True)
class BufferSettings:
    replay_path: str
    max_records: int = 5000
    failure_reward_threshold: float = 0.45
    easy_reward_threshold: float = 0.85

    @classmethod
    def from_dict(cls, data: JsonDict) -> "BufferSettings":
        return cls(
            replay_path=str(data["replay_path"]),
            max_records=int(data.get("max_records", 5000)),
            failure_reward_threshold=float(data.get("failure_reward_threshold", 0.45)),
            easy_reward_threshold=float(data.get("easy_reward_threshold", 0.85)),
        )


@dataclass(slots=True)
class StorageSettings:
    root_dir: str
    state_path: str
    seen_tasks_path: str
    events_path: str
    round_snapshots_dir: str

    @classmethod
    def from_dict(cls, data: JsonDict) -> "StorageSettings":
        return cls(
            root_dir=str(data["root_dir"]),
            state_path=str(data["state_path"]),
            seen_tasks_path=str(data["seen_tasks_path"]),
            events_path=str(data["events_path"]),
            round_snapshots_dir=str(data["round_snapshots_dir"]),
        )


@dataclass(slots=True)
class RuntimeSettings:
    max_rounds: int = 10
    seed: int = 42
    retry_invalid_generation_once: bool = True
    unload_red_after_generation: bool = True
    unload_socratic_after_use: bool = True
    empty_cuda_cache_on_unload: bool = True

    @classmethod
    def from_dict(cls, data: JsonDict | None) -> "RuntimeSettings":
        payload = data or {}
        return cls(
            max_rounds=int(payload.get("max_rounds", 10)),
            seed=int(payload.get("seed", 42)),
            retry_invalid_generation_once=bool(payload.get("retry_invalid_generation_once", True)),
            unload_red_after_generation=bool(payload.get("unload_red_after_generation", True)),
            unload_socratic_after_use=bool(payload.get("unload_socratic_after_use", True)),
            empty_cuda_cache_on_unload=bool(payload.get("empty_cuda_cache_on_unload", True)),
        )


@dataclass(slots=True)
class PipelineSettings:
    run_name: str
    models: ModelsSettings
    curriculum: CurriculumSettings
    judge: JudgeSettings
    training: TrainingSettings
    buffer: BufferSettings
    storage: StorageSettings
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "PipelineSettings":
        return cls(
            run_name=str(data.get("run_name", "adversarial-curriculum")),
            models=ModelsSettings.from_dict(data["models"]),
            curriculum=CurriculumSettings.from_dict(data["curriculum"]),
            judge=JudgeSettings.from_dict(data.get("judge")),
            training=TrainingSettings.from_dict(data["training"]),
            buffer=BufferSettings.from_dict(data["buffer"]),
            storage=StorageSettings.from_dict(data["storage"]),
            runtime=RuntimeSettings.from_dict(data.get("runtime")),
        )


@dataclass(slots=True)
class TopicStats:
    attempts: int = 0
    valid_tasks: int = 0
    cumulative_reward: float = 0.0

    @property
    def average_reward(self) -> float:
        if self.attempts <= 0:
            return 0.0
        return self.cumulative_reward / self.attempts


@dataclass(slots=True)
class CurriculumState:
    topic_weights: dict[TopicLabel, float]
    topic_stats: dict[TopicLabel, TopicStats]
    last_topic: TopicLabel | None = None
    consecutive_topic_repetitions: int = 0


@dataclass(slots=True)
class RedAdaptationState:
    active_adapter_path: str | None = None
    initial_adapter_path: str | None = None
    update_count: int = 0
    reset_count: int = 0
    last_reset_reason: str | None = None


@dataclass(slots=True)
class SocraticState:
    active_model_path: str | None = None
    active_adapter_path: str | None = None
    update_count: int = 0


@dataclass(slots=True)
class PipelineState:
    round_index: int
    total_tasks_seen: int
    curriculum: CurriculumState
    red: RedAdaptationState
    socratic: SocraticState


@dataclass(slots=True)
class TaskCandidate:
    task_id: str
    topic: str
    task_statement: str
    buggy_python: str
    asserts: list[str]
    bug_summary: str
    educational_value: str
    source_model: str
    source_adapter_path: str | None
    raw_payload: JsonDict = field(default_factory=dict)

    @property
    def buggy_program(self) -> str:
        segments = [self.buggy_python.rstrip()]
        if self.asserts:
            segments.append("\n".join(self.asserts))
        return "\n\n".join(item for item in segments if item)


@dataclass(slots=True)
class ValidatedTask:
    task: TaskCandidate
    dedupe_key: str
    judge_passed: bool
    judge_score: float
    judge_feedback: str


@dataclass(slots=True)
class HintCandidate:
    hint_id: str
    task_id: str
    topic: str
    prompt_text: str
    text: str
    sample_index: int
    model_name: str


@dataclass(slots=True)
class HintCriterionScores:
    no_solution_reveal: float
    bug_localization: float
    usefulness: float
    socratic_style: float
    technical_accuracy: float
    final_reward: float


@dataclass(slots=True)
class HintEvaluation:
    hint_id: str
    task_id: str
    scores: HintCriterionScores
    judge_feedback: str
    raw_payload: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class TaskOutcome:
    task_id: str
    topic: str
    average_reward: float
    best_reward: float
    hint_count: int
    label: RecordLabel
    validation_score: float
    validation_feedback: str


@dataclass(slots=True)
class ReplayRecord:
    task: TaskCandidate
    outcome: TaskOutcome
    label: RecordLabel


@dataclass(slots=True)
class RedSFTExample:
    prompt: str
    response: str
    topic: str
    task_id: str


@dataclass(slots=True)
class DpoTrainingPair:
    prompt: str
    chosen: str
    rejected: str
    topic: str
    chosen_task_id: str
    rejected_task_id: str


def dataclass_to_dict(instance: Any) -> JsonDict:
    return asdict(instance)
