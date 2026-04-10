"""Utilities for preparing preregistered emotion-dimension calibration inputs."""

from __future__ import annotations

from pathlib import Path


DEFAULT_EMOTION_PATH = Path("src/ontologies/emotions/emotional_states.txt")
FALLBACK_EMOTION_PATHS = (
	DEFAULT_EMOTION_PATH,
	Path("src/data/emotions/emotional_states.txt"),
)
LATENT_DIMENSIONS = ("V", "A", "C", "U", "S")


def read_emotion_lexicon(path: Path | None = None) -> list[str]:
	"""Read the comma-separated emotion lexicon into a normalized list."""
	emotion_path = _resolve_emotion_path(path)
	text = emotion_path.read_text(encoding="utf-8")
	emotions = [item.strip() for item in text.replace("\n", " ").split(",")]
	return [item for item in emotions if item]


def _resolve_emotion_path(path: Path | None) -> Path:
	if path:
		return path
	for candidate in FALLBACK_EMOTION_PATHS:
		if candidate.exists():
			return candidate
	candidates = ", ".join(str(item) for item in FALLBACK_EMOTION_PATHS)
	raise FileNotFoundError(f"No emotion lexicon file found. Checked: {candidates}")


def build_dimension_template(path: Path | None = None) -> list[dict[str, str]]:
	"""Create empty V/A/C/U/S slots for each preregistered emotion label."""
	template: list[dict[str, str]] = []
	for emotion in read_emotion_lexicon(path):
		row = {"emotion": emotion}
		for dim in LATENT_DIMENSIONS:
			row[dim] = ""
		template.append(row)
	return template
