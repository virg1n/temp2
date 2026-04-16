from __future__ import annotations

import argparse

from logging_utils import setup_logging
from pipeline import AdversarialCurriculumPipeline
from storage import StorageManager, load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="settings.yaml", help="Path to YAML settings file")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument("--max-rounds", type=int, default=None, help="Optional override for runtime.max_rounds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    settings = load_settings(args.settings)
    if args.max_rounds is not None:
        settings.runtime.max_rounds = args.max_rounds
    storage = StorageManager(settings)
    pipeline = AdversarialCurriculumPipeline(settings, storage)
    pipeline.run()


if __name__ == "__main__":
    main()
