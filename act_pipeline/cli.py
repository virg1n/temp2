from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import ACTPipeline
from .prompts import JUDGE_SYSTEM_PROMPT, RED_SYSTEM_PROMPT, format_socratic_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="ACT-only SocraticAI training pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Run ACT data collection loop.")
    collect.add_argument("--config", required=True)
    collect.add_argument("--iterations", type=int, default=None)
    collect.add_argument("--socratic-adapter", default=None)

    prompts = subparsers.add_parser("print-prompts", help="Print the core prompts.")
    prompts.add_argument("--sample", action="store_true")

    args = parser.parse_args()
    if args.command == "collect":
        cfg = load_config(args.config)
        if args.socratic_adapter:
            cfg.models.socratic_adapter = args.socratic_adapter
        ACTPipeline(cfg).run(iterations=args.iterations)
    elif args.command == "print-prompts":
        print("=== SOCRATIC PROMPT TEMPLATE ===")
        print(format_socratic_prompt("{code}", "{error}"))
        print("\n=== RED SYSTEM PROMPT ===")
        print(RED_SYSTEM_PROMPT)
        print("\n=== JUDGE SYSTEM PROMPT ===")
        print(JUDGE_SYSTEM_PROMPT)


if __name__ == "__main__":
    main()
