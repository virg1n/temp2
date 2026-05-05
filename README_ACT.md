# ACT Training Pipeline

This repo now contains a fresh ACT-only pipeline based on `main.tex`. It does not use the deleted pipeline files.

## What Is Implemented

- Red: `Qwen/Qwen3-32B` through a vLLM OpenAI-compatible endpoint.
- Judge: `Qwen/Qwen3-32B` through the same endpoint, with few-shot rubric examples and hard local penalties.
- Hard gates zero punctuation/markdown-dominated hints, cap no-question or too-short hints, and expose a `score_ceiling` so flagged outputs cannot be raised by later normalization.
- SocraticAI: `Qwen/Qwen3-1.7B` base model.
- Curriculum: topic EMA sampling weighted toward low Socratic scores.
- Validation: generated reference code must pass asserts; buggy code must fail.
- Socratic update: DPO pairs from best/worst judged hints, matching the ACT paper loop.
- Red update: KPO, not DPO. Each Red prompt stores all candidates with scalar rewards, so invalid, easy, and valid-hard samples all train the Red adapter.

The Socratic prompt is the paper template:

````text
## Code
```python
{code}
```
## Error
{error}

## Task
Ask guiding questions that help me discover the mistake.
````

## Start Training

Install dependencies in your training environment:

```bash
pip install -r requirements-act.txt
```

Run the 32B vLLM server on the GPU reserved for judge and Red. A 32B BF16 model usually will not fit on one 48 GB GPU, so use a quantized vLLM load or a compatible quantized Qwen3-32B checkpoint.

```bash
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-32B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  --dtype bfloat16 \
  --quantization bitsandbytes \
  --load-format bitsandbytes
```

Collect the first ACT batch on the other three GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m act_pipeline.cli collect \
  --config act_configs/qwen3_1_7b_act.yaml \
  --iterations 500
```

Train the SocraticAI LoRA adapter from ACT preference pairs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 3 scripts/train_socratic_dpo.py \
  --config act_configs/qwen3_1_7b_act.yaml \
  --train-file runs/act_qwen3_1_7b/socratic_preferences.jsonl \
  --output-dir runs/act_qwen3_1_7b/socratic_lora_round1
```

Continue ACT collection using the updated Socratic adapter:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m act_pipeline.cli collect \
  --config act_configs/qwen3_1_7b_act.yaml \
  --socratic-adapter runs/act_qwen3_1_7b/socratic_lora_round1 \
  --iterations 500
```

Train Red with KPO from all Red candidates:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 3 scripts/train_red_kpo.py \
  --config act_configs/qwen3_1_7b_act.yaml \
  --train-file runs/act_qwen3_1_7b/red_kpo_groups.jsonl \
  --output-dir runs/act_qwen3_1_7b/red_kpo_lora_round1
```

Repeat collect/train rounds. For the closest paper-like run, keep `socratic_candidates: 4`, topic EMA enabled, and do not add SFT/RLHF/GRPO stages.

## Outputs

- `tasks.jsonl`: valid Red tasks with reference/buggy code and execution errors.
- `judged_rollouts.jsonl`: every Socratic hint with judge score and hard-rule flags.
- `socratic_preferences.jsonl`: DPO chosen/rejected pairs for SocraticAI.
- `red_kpo_groups.jsonl`: listwise KPO groups for Red.
- `red_rejections.jsonl`: invalid Red outputs used as low-score KPO candidates.
- `topic_state.json`: curriculum EMA state.
