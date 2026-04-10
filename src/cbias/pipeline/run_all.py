"""Run the preregistered workflow stages in fixed order."""

from __future__ import annotations

PRE_REGISTERED_STAGES = [
	"01_protocol",
	"02_search_development",
	"03_database_search",
	"04_record_management",
	"05_title_abstract_screening",
	"06_full_text_eligibility",
	"07_concept_extraction",
	"08_synonym_consolidation",
	"09_cyber_relevance_tagging",
	"10_taxonomy_construction",
	"11_emotion_lexicon_finalization",
	"12_emotion_dimension_calibration",
	"13_scenario_persona_prompt_calibration",
	"14_llm_ensemble_scoring",
	"15_robustness_sensitivity_network",
	"16_confirmatory_output_freeze",
	"17_manuscript_preparation",
]


def run_all() -> None:
	"""Print preregistered stage order as a deterministic scaffold action."""
	for stage in PRE_REGISTERED_STAGES:
		print(f"[scaffold] ready: {stage}")
