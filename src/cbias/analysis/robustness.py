"""Robustness analysis utilities."""

from __future__ import annotations


def agreement_rate(labels: list[str]) -> float:
	"""Return the proportion belonging to the modal label."""
	if not labels:
		return 0.0
	counts: dict[str, int] = {}
	for label in labels:
		counts[label] = counts.get(label, 0) + 1
	return max(counts.values()) / len(labels)
