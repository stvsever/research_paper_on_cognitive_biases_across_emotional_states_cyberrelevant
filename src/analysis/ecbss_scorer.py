"""Compute ECBSS (Emotion-Conditioned Bias Shift Score) using two methods:

1. Analytical: ECBSS = weighted dot product of emotion dimensions × bias sensitivity profile
2. LLM Direct: Parallel LLM scoring of all emotion × bias family pairs

Both methods produce a matrix of shape (N_emotions × N_families).
The analytical matrix covers all 200+ emotions.
The LLM direct matrix is used as primary result and for cross-validation.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from cbias.scoring.openrouter_client import OpenRouterClient
from analysis.data_loader import BIAS_FAMILY_DESCRIPTIONS, load_emotion_scores, get_family_short_labels

ANALYTICAL_OUTPUT = ROOT / "src/review_stages/analysis_outputs/ecbss_analytical.csv"
DIRECT_OUTPUT = ROOT / "src/review_stages/analysis_outputs/ecbss_direct.csv"
DIRECT_RAW_OUTPUT = ROOT / "src/review_stages/analysis_outputs/ecbss_direct_raw.jsonl"

FAMILY_SHORT = get_family_short_labels()

SYSTEM_PROMPT = """You are an expert in cognitive psychology, appraisal theory, and cybersecurity
decision-making. You rate how specific emotional states modulate susceptibility to cognitive bias
families in cyber-relevant decision contexts (phishing detection, software installation decisions,
trust assessment of digital communications, security alert responses, access control decisions).

The Emotion-Conditioned Bias Shift Score (ECBSS) quantifies this modulation:
- Range: -1000 to +1000
- +1000: The emotional state maximally AMPLIFIES susceptibility to this bias family
- 0: The emotional state has no effect on susceptibility
- -1000: The emotional state maximally ATTENUATES/suppresses susceptibility
Scores between these extremes reflect partial amplification or attenuation.

Base your judgment on appraisal theory (how the emotion's appraisal dimensions activate vs.
suppress the cognitive mechanisms underlying the bias), dual-process theory (System 1/2 balance),
and empirical evidence on emotion-cognition interactions."""

USER_PROMPT_TEMPLATE = """Rate the ECBSS for:

Emotional state: "{emotion}"
Cognitive bias family: "{family_name}"
Family description: {family_description}

Context: A person experiencing this emotional state is making cybersecurity-relevant decisions.

Respond with valid JSON:
{{
  "ecbss": <integer -1000 to +1000>,
  "direction": "amplify" | "attenuate" | "neutral",
  "confidence": <integer 0-100>,
  "rationale": "<2 sentences: mechanism and key appraisal dimensions driving this score>"
}}"""


# ── Analytical ECBSS ────────────────────────────────────────────────────────

def compute_analytical_ecbss(
    emotion_df: pd.DataFrame,
    bias_profiles: dict[str, dict],
) -> pd.DataFrame:
    """Compute ECBSS analytically as dot(emotion_dims, bias_sensitivity) / normalizer.

    Returns DataFrame[emotions × families] with ECBSS values in [-1000, +1000].
    """
    families = list(bias_profiles.keys())
    dims = ["V", "A", "C", "U", "S"]

    # Build bias sensitivity matrix: (N_families × 5)
    bias_mat = np.array([
        [bias_profiles[f][d] for d in dims] for f in families
    ], dtype=float)  # values in [-100, +100]

    # Normalize rows so max sensitivity = 1.0 magnitude
    row_norms = np.maximum(np.abs(bias_mat).sum(axis=1, keepdims=True), 1e-9)
    bias_mat_norm = bias_mat / row_norms  # unit-normalized sensitivity

    # Emotion matrix: (N_emotions × 5), values in [-1000, +1000]
    emotions = emotion_df.index.tolist()
    emotion_mat = emotion_df[dims].values.astype(float)  # (N_emotions × 5)

    # ECBSS = emotion_mat @ bias_mat_norm.T → (N_emotions × N_families)
    raw = emotion_mat @ bias_mat_norm.T  # values in roughly [-1000, +1000]

    # Clip to hard bounds
    ecbss_mat = np.clip(raw, -1000, 1000)

    result = pd.DataFrame(ecbss_mat, index=emotions, columns=families)
    result.index.name = "emotion"
    return result


# ── LLM Direct ECBSS ────────────────────────────────────────────────────────

def _score_single(
    client: OpenRouterClient,
    emotion: str,
    family_key: str,
    family_description: str,
    retries: int = 3,
) -> dict:
    """Score a single emotion × family pair via LLM."""
    family_name = family_key.replace("_", " ").title()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        emotion=emotion,
        family_name=family_name,
        family_description=family_description[:600],
    )
    for attempt in range(retries):
        try:
            result = client.chat_json(SYSTEM_PROMPT, user_prompt, temperature=0.0)
            assert "ecbss" in result
            ecbss = int(np.clip(result["ecbss"], -1000, 1000))
            return {
                "emotion": emotion,
                "family_key": family_key,
                "ecbss": ecbss,
                "direction": result.get("direction", "unknown"),
                "confidence": result.get("confidence", 50),
                "rationale": result.get("rationale", ""),
                "status": "ok",
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    "emotion": emotion,
                    "family_key": family_key,
                    "ecbss": 0,
                    "direction": "neutral",
                    "confidence": 0,
                    "rationale": f"Error: {e}",
                    "status": "error",
                }


def run_llm_direct_scoring(
    emotions: list[str],
    family_keys: list[str] | None = None,
    max_workers: int = 50,
    force: bool = False,
) -> pd.DataFrame:
    """Run parallel LLM direct scoring for all emotion × family pairs.

    Returns DataFrame[emotions × families] pivoted on ECBSS.
    """
    if DIRECT_OUTPUT.exists() and not force:
        print(f"Loading existing LLM direct ECBSS from {DIRECT_OUTPUT}")
        return pd.read_csv(DIRECT_OUTPUT, index_col="emotion")

    if family_keys is None:
        family_keys = list(BIAS_FAMILY_DESCRIPTIONS.keys())

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
    client = OpenRouterClient(api_key=api_key, model=model, timeout_seconds=90)

    tasks = [
        (emotion, fk)
        for emotion in emotions
        for fk in family_keys
    ]
    total = len(tasks)
    print(f"Running LLM direct ECBSS scoring: {len(emotions)} emotions × {len(family_keys)} families = {total} pairs")
    print(f"Using {max_workers} parallel workers with model {model}")

    results = []
    completed = 0
    errors = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_score_single, client, emotion, fk, BIAS_FAMILY_DESCRIPTIONS[fk]): (emotion, fk)
            for emotion, fk in tasks
        }
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            completed += 1
            if r["status"] == "error":
                errors += 1
            if completed % 100 == 0:
                elapsed = time.time() - start
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  {completed}/{total} pairs scored ({errors} errors) | ETA: {eta:.0f}s")

    elapsed = time.time() - start
    print(f"Completed {total} pairs in {elapsed:.1f}s ({errors} errors)")

    # Save raw JSONL
    DIRECT_RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DIRECT_RAW_OUTPUT, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Pivot to matrix form
    raw_df = pd.DataFrame(results)
    pivot = raw_df.pivot_table(index="emotion", columns="family_key", values="ecbss", aggfunc="first")
    pivot.columns.name = None
    pivot.index.name = "emotion"

    pivot.to_csv(DIRECT_OUTPUT)
    print(f"Saved LLM direct ECBSS matrix to {DIRECT_OUTPUT}")
    return pivot


def save_analytical_ecbss(df: pd.DataFrame) -> None:
    ANALYTICAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ANALYTICAL_OUTPUT)
    print(f"Saved analytical ECBSS matrix to {ANALYTICAL_OUTPUT}")


if __name__ == "__main__":
    from analysis.data_loader import load_emotion_scores, load_emotion_lexicon
    from analysis.bias_profiler import run_bias_profiling

    emotion_df = load_emotion_scores()
    profiles = run_bias_profiling()
    analytical = compute_analytical_ecbss(emotion_df, profiles)
    save_analytical_ecbss(analytical)
    print("Analytical ECBSS matrix shape:", analytical.shape)
    print(analytical.head())
