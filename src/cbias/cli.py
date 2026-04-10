"""Command-line interface for the project scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from cbias.pipeline.run_all import run_all
from cbias.scoring import (
	DEFAULT_FALLBACK_MODEL,
	DEFAULT_MODEL,
	DEFAULT_OUTPUT_DIR,
	run_emotion_semantic_scoring,
)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="cbias",
		description="Cognitive-bias vulnerability pipeline scaffold",
	)
	subparsers = parser.add_subparsers(dest="command", required=True)

	subparsers.add_parser("run-all", help="Run preregistered stage scaffold")

	score_parser = subparsers.add_parser(
		"score-emotions",
		help="Create high-resolution binary semantic scores for each emotion",
	)
	score_parser.add_argument(
		"--env-file",
		type=Path,
		default=Path(".env"),
		help="Path to environment file with OPENROUTER_API_KEY",
	)
	score_parser.add_argument(
		"--emotion-file",
		type=Path,
		default=Path("src/ontologies/emotions/emotional_states.txt"),
		help="Path to source emotion lexicon file",
	)
	score_parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Directory where score outputs are written",
	)
	score_parser.add_argument(
		"--model",
		default=DEFAULT_MODEL,
		help="Primary OpenRouter model (default: Gemini Flash)",
	)
	score_parser.add_argument(
		"--fallback-model",
		default=DEFAULT_FALLBACK_MODEL,
		help="Fallback OpenRouter model used when primary call fails",
	)
	score_parser.add_argument(
		"--max-workers",
		type=int,
		default=50,
		help="ThreadPoolExecutor worker count",
	)
	score_parser.add_argument(
		"--retries",
		type=int,
		default=2,
		help="Retries per model call before fallback/failure",
	)
	score_parser.add_argument(
		"--timeout-seconds",
		type=int,
		default=120,
		help="Per-request HTTP timeout",
	)
	score_parser.add_argument(
		"--temperature",
		type=float,
		default=0.0,
		help="LLM sampling temperature",
	)
	score_parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Optional cap on number of emotions processed",
	)
	score_parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Generate deterministic synthetic outputs without API calls",
	)
	return parser


def main() -> int:
	args = build_parser().parse_args()
	if args.command == "run-all":
		run_all()
		return 0
	if args.command == "score-emotions":
		summary = run_emotion_semantic_scoring(
			env_file=args.env_file,
			emotion_file=args.emotion_file,
			output_dir=args.output_dir,
			model=args.model,
			fallback_model=args.fallback_model,
			max_workers=max(1, args.max_workers),
			retries=max(0, args.retries),
			timeout_seconds=max(1, args.timeout_seconds),
			temperature=args.temperature,
			limit=args.limit,
			dry_run=args.dry_run,
		)
		print(
			"emotion_scoring_summary "
			f"total={summary['total']} "
			f"success={summary['success']} "
			f"failed={summary['failed']} "
			f"model={summary['model']} "
			f"fallback={summary['fallback_model']} "
			f"dry_run={summary['dry_run']} "
			f"output_dir={summary['output_dir']}"
		)
		return 0
	return 1


if __name__ == "__main__":
	raise SystemExit(main())
