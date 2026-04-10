"""Helpers for reading and flattening the cognitive-bias taxonomy."""

from __future__ import annotations

import json
from pathlib import Path


DEFAULT_TAXONOMY_PATH = Path("src/ontologies/taxonomy/taxonomy_cognitive_biases_cyberrelevant.json")
FALLBACK_TAXONOMY_PATHS = (
	DEFAULT_TAXONOMY_PATH,
	Path("src/data/taxonomy/taxonomy_cognitive_biases_cyberrelevant.json"),
)


def load_taxonomy(path: Path | None = None) -> dict:
	"""Load taxonomy JSON into a nested dictionary."""
	taxonomy_path = _resolve_taxonomy_path(path)
	with taxonomy_path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _resolve_taxonomy_path(path: Path | None) -> Path:
	if path:
		return path
	for candidate in FALLBACK_TAXONOMY_PATHS:
		if candidate.exists():
			return candidate
	candidates = ", ".join(str(item) for item in FALLBACK_TAXONOMY_PATHS)
	raise FileNotFoundError(f"No taxonomy file found. Checked: {candidates}")


def flatten_leaf_biases(tree: dict, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
	"""Collect paths to all leaf bias terms in the nested taxonomy."""
	leaves: list[tuple[str, ...]] = []
	for key, value in tree.items():
		current = (*prefix, key)
		if isinstance(value, dict) and value:
			leaves.extend(flatten_leaf_biases(value, current))
		elif isinstance(value, dict):
			leaves.append(current)
	return leaves
