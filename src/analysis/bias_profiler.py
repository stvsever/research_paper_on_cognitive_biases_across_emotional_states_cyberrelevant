"""LLM-based bias family dimensional sensitivity profiling.

For each of the 11 bias families, we ask the LLM to rate sensitivity to
the five affective dimensions (V, A, C, U, S) on a -100 to +100 scale.
This creates a 'bias sensitivity profile' used in ECBSS computation.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from cbias.scoring.openrouter_client import OpenRouterClient
from analysis.data_loader import BIAS_FAMILY_DESCRIPTIONS

OUTPUT_PATH = ROOT / "src/review_stages/analysis_outputs/bias_sensitivity_profiles.json"

SYSTEM_PROMPT = """You are an expert cognitive psychologist specializing in cognitive biases and
their modulation by emotional states in cybersecurity decision contexts. You provide precise,
research-grade dimensional sensitivity assessments grounded in dual-process theory and appraisal
theory of emotions."""

USER_PROMPT_TEMPLATE = """For the cognitive bias family: "{family_name}"

Description: {family_description}

Rate how each of the following affective dimensions modulates susceptibility to biases in this
family when a person makes cyber-security relevant decisions (e.g., evaluating email legitimacy,
deciding on software installation, assessing digital trustworthiness, responding to security alerts,
managing access controls, handling urgent notifications).

Dimensions:
- V (Valence): Moving from negative emotional tone (-1000) to positive emotional tone (+1000)
- A (Arousal): Moving from low activation/calm (-1000) to high activation/agitation (+1000)
- C (Control): Moving from low perceived control/helplessness (-1000) to high control/mastery (+1000)
- U (Uncertainty): Moving from high certainty/predictability (-1000) to high uncertainty/unexpectedness (+1000)
- S (Social Orientation): Moving from self-focused (-1000) to other-focused (+1000)

For each dimension, rate: how does INCREASING the dimension value affect susceptibility to this
bias family?
- Positive score (+1 to +100): Increasing the dimension AMPLIFIES susceptibility
- Negative score (-100 to -1): Increasing the dimension ATTENUATES susceptibility
- Zero: Dimension has little/no effect on this bias family

Respond with valid JSON:
{{
  "V": <integer -100 to +100>,
  "A": <integer -100 to +100>,
  "C": <integer -100 to +100>,
  "U": <integer -100 to +100>,
  "S": <integer -100 to +100>,
  "rationale": "<3-4 sentences explaining the key dimensional drivers for this bias family>"
}}"""


def score_bias_family(client: OpenRouterClient, family_key: str, family_description: str) -> dict:
    """Score a single bias family's dimensional sensitivity."""
    family_name = family_key.replace("_", " ").title()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        family_name=family_name,
        family_description=family_description,
    )
    for attempt in range(3):
        try:
            result = client.chat_json(SYSTEM_PROMPT, user_prompt, temperature=0.0)
            # Validate expected keys
            for dim in ["V", "A", "C", "U", "S"]:
                assert dim in result, f"Missing dimension {dim}"
            result["family_key"] = family_key
            result["family_name"] = family_name
            return result
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  FAILED {family_key}: {e}")
                return {"family_key": family_key, "family_name": family_name,
                        "V": 0, "A": 0, "C": 0, "U": 0, "S": 0,
                        "rationale": f"Error: {e}"}


def run_bias_profiling(force: bool = False) -> dict[str, dict]:
    """Run bias family sensitivity profiling. Returns dict[family_key -> profile]."""
    if OUTPUT_PATH.exists() and not force:
        print(f"Loading existing bias profiles from {OUTPUT_PATH}")
        with open(OUTPUT_PATH) as f:
            return json.load(f)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
    client = OpenRouterClient(api_key=api_key, model=model, timeout_seconds=60)

    profiles = {}
    for i, (family_key, description) in enumerate(BIAS_FAMILY_DESCRIPTIONS.items(), 1):
        print(f"  [{i}/{len(BIAS_FAMILY_DESCRIPTIONS)}] Scoring: {family_key[:50]}...")
        profile = score_bias_family(client, family_key, description)
        profiles[family_key] = profile
        time.sleep(0.3)  # gentle rate limiting

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"Saved bias profiles to {OUTPUT_PATH}")
    return profiles


if __name__ == "__main__":
    import pprint
    profiles = run_bias_profiling(force=True)
    pprint.pprint({k: {d: v[d] for d in "VACUS"} for k, v in profiles.items()})
