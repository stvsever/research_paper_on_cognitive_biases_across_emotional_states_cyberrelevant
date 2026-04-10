"""Definitions for binary semantic emotion dimensions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinarySemanticDimension:
	"""One high-resolution binary semantic axis."""

	code: str
	name: str
	left_pole: str
	right_pole: str
	description: str


DIMENSIONS: tuple[BinarySemanticDimension, ...] = (
	BinarySemanticDimension(
		code="V",
		name="valence",
		left_pole="negative",
		right_pole="positive",
		description="Affective pleasantness from aversive to pleasant.",
	),
	BinarySemanticDimension(
		code="A",
		name="arousal",
		left_pole="low_activation",
		right_pole="high_activation",
		description="Physiological and mental activation intensity.",
	),
	BinarySemanticDimension(
		code="C",
		name="control",
		left_pole="low_control",
		right_pole="high_control",
		description="Perceived coping potential and agency.",
	),
	BinarySemanticDimension(
		code="U",
		name="uncertainty",
		left_pole="expected_or_certain",
		right_pole="uncertain_or_unpredictable",
		description="Expectedness and predictability of the state.",
	),
	BinarySemanticDimension(
		code="S",
		name="social_orientation",
		left_pole="self_oriented",
		right_pole="other_oriented",
		description="Orientation toward self versus others.",
	),
)

DIMENSION_BY_CODE = {dimension.code: dimension for dimension in DIMENSIONS}
