"""Run stages explicitly; opening the dashboard never triggers API calls."""
import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .dataset import save_default
from .pipeline import generate, init_run, judge, manifest_for, position
from .providers import APIError


def main():
    parser = argparse.ArgumentParser(description="JudgeBench local evaluation lab")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("dataset", help="Write the included 40-question dataset")
    data.add_argument("--output", default="data/questions.json")
    init = sub.add_parser("init", help="Freeze an immutable experiment manifest; no API calls")
    init.add_argument("--run", required=True)
    init.add_argument("--config", default="config.json")
    init.add_argument("--dataset", default="data/questions.json")
    for name in ("plan", "generate", "judge", "position", "analyze"):
        command = sub.add_parser(name)
        command.add_argument("--run", required=True)
        if name in ("generate", "judge", "position"):
            command.add_argument("--limit", type=int, help="Maximum new records this invocation")
        if name == "judge":
            command.add_argument("--rubric", choices=["baseline", "strict_correctness"], default="baseline")
    args = parser.parse_args()
    try:
        if hasattr(args, "limit") and args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if args.command == "dataset":
            save_default(args.output)
            print(f"Wrote 40 questions to {args.output}.")
        elif args.command == "init":
            manifest = init_run(args.run, args.config, args.dataset)
            print(f"Initialized {args.run}; experiment {manifest['experiment_id'][:12]}. No API calls made.")
        elif args.command == "plan":
            m = manifest_for(args.run)
            n = len(m["questions"])
            repetitions = m["config"]["position_repeats"]
            plan = {"generation": 2*n, "baseline_judging": 2*n,
                    "position": 2*n*repetitions, "strict_judging": 2*n}
            print(json.dumps({"planned_calls_excluding_retries": plan, "total": sum(plan.values()),
                              "model_config": m["config"],
                              "note": "Model access/free quotas must be checked in provider consoles. "
                                      "Limits pause the run; no automatic paid fallback."}, indent=2))
        elif args.command == "generate":
            print(f"Saved {generate(args.run, args.limit)} new responses.")
        elif args.command == "judge":
            print(f"Saved {judge(args.run, args.rubric, args.limit)} new judgments.")
        elif args.command == "position":
            print(f"Saved {position(args.run, args.limit)} new position trials.")
        elif args.command == "analyze":
            from .analysis import analyze
            result = analyze(args.run)
            print(json.dumps(result["coverage"], indent=2))
            print(f"Exported metrics, CSV/JSON tables, and findings to {Path(args.run) / 'exports'}.")
    except (APIError, ValueError, FileNotFoundError, FileExistsError, ValidationError) as exc:
        print(f"JudgeBench: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
