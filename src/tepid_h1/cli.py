from __future__ import annotations

import argparse
import json

from .config import TepidH1Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tepid-h1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="print the macro-block layer plan")
    plan.add_argument("--variant", choices=("prototype", "reference"), default="prototype")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plan":
        config = (
            TepidH1Config.prototype()
            if args.variant == "prototype"
            else TepidH1Config.reference_28b_a7b()
        )
        payload = {
            "config": config.to_dict(),
            "module_counts": config.module_counts(),
            "layers": [
                {
                    "index": layer.index + 1,
                    "macro_block": layer.macro_block + 1,
                    "sequence": layer.sequence.value,
                    "channel": layer.channel.value,
                }
                for layer in config.layer_plan
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

