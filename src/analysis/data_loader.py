"""Load and structure all core data assets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "src/ontologies/taxonomy/taxonomy_cognitive_biases_cyberrelevant.json"
EMOTION_SCORES_PATH = ROOT / "src/review_stages/12_emotion_dimension_calibration/outputs/emotion_semantic_scores_wide.csv"
EMOTION_LEXICON_PATH = ROOT / "src/ontologies/emotions/emotional_states.txt"


# ── Bias family descriptions for LLM prompting ─────────────────────────────
BIAS_FAMILY_DESCRIPTIONS = {
    "attention_salience_and_signal_detection_biases": (
        "Biases affecting what information captures attention, which signals are noticed vs. "
        "missed, and how vividness or threat-salience distorts detection. Includes salience bias, "
        "inattentional blindness, alarm fatigue, and banner blindness. Critically relevant to "
        "phishing detection and security alert triage."
    ),
    "trust_source_credibility_and_truth_judgment_biases": (
        "Biases causing over- or under-trust based on surface cues such as appearance, authority "
        "signals, familiarity, or fluency. Includes halo effect, illusory truth effect, and "
        "recognition heuristic. Relevant to evaluating email senders, websites, and security "
        "communications."
    ),
    "evidence_search_hypothesis_testing_and_belief_updating_biases": (
        "Biases in how people seek, interpret, and update evidence — including confirmatory "
        "processing, anchoring, and representativeness errors. Affects whether people correctly "
        "update threat assessments when confronted with new cybersecurity information."
    ),
    "memory_familiarity_and_source_monitoring_biases": (
        "Biases in how memory shapes judgment — including availability heuristic, source "
        "confusion, and hindsight bias. Affects whether people correctly attribute the origin "
        "of information (e.g., mistaking a phishing email for a legitimate past contact)."
    ),
    "risk_probability_uncertainty_and_outcome_valuation_biases": (
        "Biases distorting probability estimation, risk attitude, and loss/gain evaluation — "
        "including loss aversion, optimism bias, and probability neglect. Affects whether "
        "users correctly assess cyber threat likelihood and severity."
    ),
    "temporal_choice_default_action_and_commitment_biases": (
        "Biases toward present-focused choices, default actions, inertia, and sunk-cost "
        "commitment — including present bias, default effect, and sunk cost fallacy. Affects "
        "patch management, security update adoption, and escalation decisions."
    ),
    "social_influence_authority_affiliation_and_identity_biases": (
        "Biases from social proof, authority deference, group identity, and conformity — "
        "including bandwagon effect, authority bias, and in-group favoritism. Critical to "
        "social engineering susceptibility and phishing compliance."
    ),
    "self_assessment_attribution_and_metacognitive_biases": (
        "Biases in self-evaluation, confidence calibration, and attribution — including "
        "overconfidence, dunning-kruger effect, and fundamental attribution error. Affects "
        "whether users accurately assess their own cybersecurity skill and risk exposure."
    ),
    "interface_choice_architecture_automation_and_warning_response_biases": (
        "Biases from choice architecture, automation reliance, and routine interaction — "
        "including automation bias, habituation to security warnings, and position effects. "
        "Directly relevant to security UI design and warning dialog behavior."
    ),
    "privacy_disclosure_and_self_presentation_biases": (
        "Biases distorting decisions about sharing personal or organizational information — "
        "including the privacy paradox, online disinhibition effect, and illusion of "
        "transparency. Affects data disclosure, oversharing, and privacy protection behavior."
    ),
    "affective_evaluation_and_mood_congruent_judgment_biases": (
        "Biases where current affect directly colors judgment — including affect heuristic, "
        "mood-congruent memory, and hot-cold empathy gap. Foundational to understanding how "
        "emotional states modulate all other cognitive biases in cyber decisions."
    ),
}


def load_taxonomy() -> tuple[dict, pd.DataFrame]:
    """Load taxonomy JSON and return (raw_dict, flat_dataframe).

    DataFrame columns: leaf_bias, cluster, family, depth
    """
    with open(TAXONOMY_PATH) as f:
        raw = json.load(f)

    root = raw["cognitive_biases_in_cyber_relevant_contexts"]
    rows = []
    for family_key, clusters in root.items():
        for cluster_key, leaves in clusters.items():
            for leaf_key in leaves.keys():
                rows.append({
                    "leaf_bias": leaf_key,
                    "cluster": cluster_key,
                    "family": family_key,
                    "leaf_label": leaf_key.replace("_", " ").title(),
                    "cluster_label": cluster_key.replace("_", " ").title(),
                    "family_label": family_key.replace("_", " ").title(),
                })

    df = pd.DataFrame(rows)
    return root, df


def load_emotion_scores() -> pd.DataFrame:
    """Load emotion dimension scores from Stage 12 outputs."""
    df = pd.read_csv(EMOTION_SCORES_PATH)
    df = df[df["status"] == "ok"].copy()

    # Keep only emotion + signed scores
    dim_cols = {
        "V": "V_signed_score_m1000_to_p1000",
        "A": "A_signed_score_m1000_to_p1000",
        "C": "C_signed_score_m1000_to_p1000",
        "U": "U_signed_score_m1000_to_p1000",
        "S": "S_signed_score_m1000_to_p1000",
    }
    result = df[["emotion"] + list(dim_cols.values())].copy()
    result.columns = ["emotion", "V", "A", "C", "U", "S"]
    result = result.set_index("emotion")
    return result


def load_emotion_lexicon() -> list[str]:
    """Load the frozen emotion lexicon."""
    with open(EMOTION_LEXICON_PATH) as f:
        text = f.read()
    emotions = [e.strip() for e in text.replace("\n", ",").split(",") if e.strip()]
    return emotions


def get_family_keys() -> list[str]:
    """Return ordered list of bias family keys."""
    return list(BIAS_FAMILY_DESCRIPTIONS.keys())


def get_family_short_labels() -> dict[str, str]:
    """Short labels for bias families (for axes and legends)."""
    return {
        "attention_salience_and_signal_detection_biases": "Attention & Salience",
        "trust_source_credibility_and_truth_judgment_biases": "Trust & Credibility",
        "evidence_search_hypothesis_testing_and_belief_updating_biases": "Evidence & Belief",
        "memory_familiarity_and_source_monitoring_biases": "Memory & Source",
        "risk_probability_uncertainty_and_outcome_valuation_biases": "Risk & Probability",
        "temporal_choice_default_action_and_commitment_biases": "Temporal & Commitment",
        "social_influence_authority_affiliation_and_identity_biases": "Social & Authority",
        "self_assessment_attribution_and_metacognitive_biases": "Self-Assessment",
        "interface_choice_architecture_automation_and_warning_response_biases": "Interface & Automation",
        "privacy_disclosure_and_self_presentation_biases": "Privacy & Disclosure",
        "affective_evaluation_and_mood_congruent_judgment_biases": "Affect & Mood",
    }
