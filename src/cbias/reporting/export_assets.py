"""Utilities for report asset output scaffolding."""

from __future__ import annotations

from pathlib import Path


DEFAULT_ASSET_DIRS = (
	Path("paper/assets/figures"),
	Path("paper/assets/tables"),
	Path("paper/report/generated"),
)


def ensure_asset_dirs(paths: tuple[Path, ...] = DEFAULT_ASSET_DIRS) -> None:
	"""Create the default output directories used by the manuscript pipeline."""
	for path in paths:
		path.mkdir(parents=True, exist_ok=True)
