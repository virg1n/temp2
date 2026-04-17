from __future__ import annotations

import argparse

from acl_pipeline.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial curriculum training for Socratic Python debugging.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config.")
    parser.add_argument("--debug-all", action="store_true", help="Print full task/hint/judge/reset details.")
    args = parser.parse_args()

    config = load_config(args.config, debug_all_override=args.debug_all)
    from acl_pipeline.pipeline import AdversarialCurriculumPipeline

    pipeline = AdversarialCurriculumPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
