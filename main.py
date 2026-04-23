from __future__ import annotations

import argparse
import os
from typing import Iterable, List

from acl_pipeline.config import load_config


def _parse_cuda_visible_devices(value: str) -> List[int]:
    ids: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ids.append(int(item))
        except ValueError:
            return []
    return ids


def _unique_ints(values: Iterable[int]) -> List[int]:
    seen = set()
    result: List[int] = []
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _remap_hardware_to_visible_ordinals(hardware: object, visible_physical_ids: List[int]) -> None:
    gpu_ids = [int(item) for item in getattr(hardware, "gpu_ids", [])]
    if not gpu_ids or not visible_physical_ids:
        return
    physical_to_local = {physical_id: index for index, physical_id in enumerate(visible_physical_ids)}
    if all(gpu_id in physical_to_local for gpu_id in gpu_ids):
        setattr(hardware, "gpu_ids", [physical_to_local[gpu_id] for gpu_id in gpu_ids])


def _configure_cuda_visibility(config: object) -> None:
    existing_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_physical_ids = _parse_cuda_visible_devices(existing_visible) if existing_visible else []

    judge = getattr(config, "judge")
    red = getattr(config, "red")
    socratic = getattr(config, "socratic")
    judge_backend = str(getattr(judge, "inference_backend", "")).strip().lower()
    reserved_judge_gpus = set(getattr(judge.hardware, "gpu_ids", []) or []) if judge_backend == "vllm_server" else set()

    local_hardware = [socratic.hardware, red.hardware]
    if judge_backend != "vllm_server":
        local_hardware.append(judge.hardware)

    for hardware in local_hardware:
        gpu_ids = [int(item) for item in getattr(hardware, "gpu_ids", [])]
        filtered = [gpu_id for gpu_id in gpu_ids if gpu_id not in reserved_judge_gpus]
        if filtered:
            setattr(hardware, "gpu_ids", filtered)

    if not visible_physical_ids:
        requested = _unique_ints(
            gpu_id
            for hardware in local_hardware
            for gpu_id in getattr(hardware, "gpu_ids", [])
        )
        if requested:
            visible_physical_ids = requested
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in visible_physical_ids)

    for hardware in local_hardware:
        _remap_hardware_to_visible_ordinals(hardware, visible_physical_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial curriculum training for Socratic Python debugging.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config.")
    parser.add_argument("--debug-all", action="store_true", help="Print full task/hint/judge/reset details.")
    args = parser.parse_args()

    config = load_config(args.config, debug_all_override=args.debug_all)
    _configure_cuda_visibility(config)
    from acl_pipeline.pipeline import AdversarialCurriculumPipeline

    pipeline = AdversarialCurriculumPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
