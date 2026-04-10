"""Starter schema for preregistered scoring outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreCell:
	"""Single scoring cell across emotion, bias, and context factors."""

	emotion: str
	bias_leaf: str
	scenario_id: str
	persona_id: str
	judge_model: str
	direction: str = "no_shift"
	magnitude: int = 0
	confidence: float = 0.0
