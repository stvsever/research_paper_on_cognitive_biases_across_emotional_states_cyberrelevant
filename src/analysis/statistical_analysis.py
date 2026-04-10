"""Statistical analyses: clustering, mixed-effects regression, PCA, composite emotions."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from dotenv import load_dotenv

from cbias.scoring.openrouter_client import OpenRouterClient
from analysis.data_loader import BIAS_FAMILY_DESCRIPTIONS

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "src/review_stages/analysis_outputs"
load_dotenv(ROOT / ".env", override=False)


# ── Emotion Clustering ──────────────────────────────────────────────────────

EMOTION_CLUSTER_NAMES = {
    0: "High-Arousal Negative\n(Fear/Panic/Alarm)",
    1: "Low-Valence Withdrawn\n(Grief/Desolation)",
    2: "Calm Positive\n(Trust/Contentment)",
    3: "High-Arousal Positive\n(Enthusiasm/Elation)",
    4: "Social Threat / Shame\n(Humiliation/Guilt)",
    5: "Hostile / Defiant\n(Anger/Contempt)",
}

N_EMOTION_CLUSTERS = 6


def cluster_emotions(
    emotion_df: pd.DataFrame,
    n_clusters: int = N_EMOTION_CLUSTERS,
    random_state: int = 42,
) -> tuple[pd.Series, np.ndarray]:
    """K-means cluster emotions based on V, A, C, U, S dimensional scores.

    Returns (cluster_labels Series, cluster_centers array).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(emotion_df[["V", "A", "C", "U", "S"]].values)

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels_raw = km.fit_predict(X)

    # Assign semantic cluster names: sort clusters by mean Valence then Arousal
    cluster_chars = []
    for cid in range(n_clusters):
        mask = labels_raw == cid
        mv = emotion_df["V"].values[mask].mean()
        ma = emotion_df["A"].values[mask].mean()
        cluster_chars.append((cid, mv, ma))

    # Sort: by valence ascending, then arousal descending
    sorted_clusters = sorted(cluster_chars, key=lambda x: (round(x[1] / 400), -x[2]))
    remap = {orig: new for new, (orig, _, _) in enumerate(sorted_clusters)}
    labels = pd.Series([remap[l] for l in labels_raw], index=emotion_df.index, name="emotion_cluster")

    centers = km.cluster_centers_
    return labels, centers


def get_emotion_cluster_profiles(
    emotion_df: pd.DataFrame,
    cluster_labels: pd.Series,
) -> pd.DataFrame:
    """Return mean V, A, C, U, S per emotion cluster."""
    joined = emotion_df.join(cluster_labels)
    return joined.groupby("emotion_cluster")[["V", "A", "C", "U", "S"]].mean()


def hierarchical_clustering_emotions(emotion_df: pd.DataFrame) -> np.ndarray:
    """Compute hierarchical linkage for emotion dendrogram."""
    X = StandardScaler().fit_transform(emotion_df[["V", "A", "C", "U", "S"]].values)
    return linkage(X, method="ward")


def hierarchical_clustering_biases(ecbss_df: pd.DataFrame) -> np.ndarray:
    """Compute hierarchical linkage for bias families based on emotion profiles."""
    # Each bias family described by its ECBSS profile across emotions
    X = ecbss_df.T.values  # (N_families × N_emotions)
    X_scaled = StandardScaler().fit_transform(X)
    return linkage(X_scaled, method="ward")


# ── PCA / UMAP ──────────────────────────────────────────────────────────────

def run_pca(emotion_df: pd.DataFrame) -> tuple[np.ndarray, PCA]:
    """PCA on V, A, C, U, S — return (coords_2d, fitted_pca)."""
    X = emotion_df[["V", "A", "C", "U", "S"]].values
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    return coords, pca


def run_umap(emotion_df: pd.DataFrame, n_neighbors: int = 20, min_dist: float = 0.3) -> np.ndarray:
    """UMAP embedding of emotions in 2D using V, A, C, U, S features."""
    import umap
    X = StandardScaler().fit_transform(emotion_df[["V", "A", "C", "U", "S"]].values)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42, n_components=2)
    return reducer.fit_transform(X)


# ── ECBSS Cluster Summaries ─────────────────────────────────────────────────

def compute_cluster_ecbss(
    ecbss_df: pd.DataFrame,
    cluster_labels: pd.Series,
) -> pd.DataFrame:
    """Average ECBSS per emotion cluster × bias family.

    Returns DataFrame[N_clusters × N_families].
    """
    aligned = ecbss_df.join(cluster_labels)
    return aligned.groupby("emotion_cluster").mean()


# ── Mixed-Effects Regression ─────────────────────────────────────────────────

def run_mixed_effects_regression(
    ecbss_df: pd.DataFrame,
    emotion_df: pd.DataFrame,
) -> dict:
    """Run mixed-effects model: ECBSS ~ V + A + C + U + S with random intercepts by family.

    Returns dict with overall model results, per-family OLS betas, and
    matched valence-only baseline comparisons for RQ4 reporting.
    """
    # Build long-format DataFrame
    records = []
    for family in ecbss_df.columns:
        for emotion in ecbss_df.index:
            if emotion in emotion_df.index:
                row = {
                    "emotion": emotion,
                    "family": family,
                    "ecbss": ecbss_df.loc[emotion, family],
                    "V": emotion_df.loc[emotion, "V"] / 1000.0,
                    "A": emotion_df.loc[emotion, "A"] / 1000.0,
                    "C": emotion_df.loc[emotion, "C"] / 1000.0,
                    "U": emotion_df.loc[emotion, "U"] / 1000.0,
                    "S": emotion_df.loc[emotion, "S"] / 1000.0,
                }
                records.append(row)

    long_df = pd.DataFrame(records)

    # Mixed-effects model with random intercepts by family
    try:
        md = smf.mixedlm("ecbss ~ V + A + C + U + S", long_df, groups=long_df["family"])
        mdf = md.fit(reml=False)
        overall_params = {
            "params": mdf.params.to_dict(),
            "pvalues": mdf.pvalues.to_dict(),
            "bse": mdf.bse.to_dict(),
            "aic": float(mdf.aic),
            "llf": float(mdf.llf),
            "converged": True,
        }
    except Exception as e:
        print(f"Mixed-effects model failed: {e}. Using OLS.")
        import statsmodels.api as sm
        X = sm.add_constant(long_df[["V", "A", "C", "U", "S"]].values)
        ols = sm.OLS(long_df["ecbss"].values, X).fit()
        overall_params = {
            "params": dict(zip(["const", "V", "A", "C", "U", "S"], ols.params.tolist())),
            "pvalues": dict(zip(["const", "V", "A", "C", "U", "S"], ols.pvalues.tolist())),
            "bse": dict(zip(["const", "V", "A", "C", "U", "S"], ols.bse.tolist())),
            "aic": float(ols.aic),
            "llf": float(ols.llf),
            "converged": True,
        }

    # Stacked OLS comparison: full model vs valence-only baseline
    import statsmodels.api as sm
    full_X = sm.add_constant(long_df[["V", "A", "C", "U", "S"]].values)
    valence_X = sm.add_constant(long_df[["V"]].values)
    overall_ols_full = sm.OLS(long_df["ecbss"].values, full_X).fit()
    overall_ols_valence = sm.OLS(long_df["ecbss"].values, valence_X).fit()

    # Per-family OLS regression
    per_family = {}
    per_family_valence = {}
    dims = ["V", "A", "C", "U", "S"]
    for family in ecbss_df.columns:
        fam_df = long_df[long_df["family"] == family].dropna()
        if len(fam_df) < 10:
            continue
        X = sm.add_constant(fam_df[dims].values)
        y = fam_df["ecbss"].values
        try:
            ols = sm.OLS(y, X).fit()
            per_family[family] = {
                "params": dict(zip(["const"] + dims, ols.params.tolist())),
                "pvalues": dict(zip(["const"] + dims, ols.pvalues.tolist())),
                "bse": dict(zip(["const"] + dims, ols.bse.tolist())),
                "rsquared": float(ols.rsquared),
                "rsquared_adj": float(ols.rsquared_adj),
                "n": len(fam_df),
            }
        except Exception:
            pass
        try:
            val_ols = sm.OLS(
                fam_df["ecbss"].values,
                sm.add_constant(fam_df[["V"]].values),
            ).fit()
            per_family_valence[family] = {
                "params": {
                    "const": float(val_ols.params[0]),
                    "V": float(val_ols.params[1]),
                },
                "pvalues": {
                    "const": float(val_ols.pvalues[0]),
                    "V": float(val_ols.pvalues[1]),
                },
                "bse": {
                    "const": float(val_ols.bse[0]),
                    "V": float(val_ols.bse[1]),
                },
                "rsquared": float(val_ols.rsquared),
                "rsquared_adj": float(val_ols.rsquared_adj),
                "n": len(fam_df),
            }
        except Exception:
            pass

    full_r2 = [res["rsquared"] for res in per_family.values()]
    valence_r2 = [
        per_family_valence[f]["rsquared"]
        for f in per_family
        if f in per_family_valence
    ]
    delta_by_family = {
        family: float(per_family[family]["rsquared"] - per_family_valence[family]["rsquared"])
        for family in per_family
        if family in per_family_valence
    }

    results = {
        "overall": overall_params,
        "per_family": per_family,
        "rq4_comparison": {
            "overall_ols_full": {
                "rsquared": float(overall_ols_full.rsquared),
                "rsquared_adj": float(overall_ols_full.rsquared_adj),
                "aic": float(overall_ols_full.aic),
            },
            "overall_ols_valence_only": {
                "rsquared": float(overall_ols_valence.rsquared),
                "rsquared_adj": float(overall_ols_valence.rsquared_adj),
                "aic": float(overall_ols_valence.aic),
            },
            "overall_delta_r2": float(overall_ols_full.rsquared - overall_ols_valence.rsquared),
            "per_family_valence_only": per_family_valence,
            "per_family_delta_r2": delta_by_family,
            "mean_family_r2_full": float(np.mean(full_r2)) if full_r2 else np.nan,
            "mean_family_r2_valence_only": float(np.mean(valence_r2)) if valence_r2 else np.nan,
            "mean_family_delta_r2": float(np.mean(list(delta_by_family.values()))) if delta_by_family else np.nan,
        },
    }

    out_path = OUTPUTS / "regression_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved regression results to {out_path}")
    return results


# ── Validation: Analytical vs LLM Direct ───────────────────────────────────

def validate_analytical_vs_direct(
    analytical: pd.DataFrame,
    direct: pd.DataFrame,
) -> dict:
    """Compute correlation between analytical and LLM direct ECBSS."""
    common_emotions = analytical.index.intersection(direct.index)
    common_families = analytical.columns.intersection(direct.columns)

    flat_analytical = analytical.loc[common_emotions, common_families].values.flatten()
    flat_direct = direct.loc[common_emotions, common_families].values.flatten()

    mask = ~np.isnan(flat_direct) & ~np.isnan(flat_analytical)
    r_pearson, p_pearson = pearsonr(flat_analytical[mask], flat_direct[mask])
    r_spearman, p_spearman = spearmanr(flat_analytical[mask], flat_direct[mask])

    # Per-family correlations
    per_family_r = {}
    for family in common_families:
        a = analytical.loc[common_emotions, family].values
        d = direct.loc[common_emotions, family].values
        m = ~np.isnan(d) & ~np.isnan(a)
        if m.sum() > 5:
            r, p = pearsonr(a[m], d[m])
            per_family_r[family] = {"pearson_r": float(r), "p": float(p)}

    result = {
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(r_spearman),
        "spearman_p": float(p_spearman),
        "n_pairs": int(mask.sum()),
        "per_family": per_family_r,
    }

    out_path = OUTPUTS / "validation_correlation.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Analytical vs Direct correlation: r={r_pearson:.3f}, ρ={r_spearman:.3f}")
    return result


# ── Composite Emotion Non-Additivity ────────────────────────────────────────

COMPOSITE_PAIRS = [
    ("afraid", "angry"),        # fear + anger → fight-or-flight
    ("anxious", "curious"),     # anxiety + curiosity → vigilant exploration
    ("enthusiastic", "distracted"),   # enthusiasm - focus
    ("calm", "suspicious"),     # calm vigilance
    ("desperate", "trusting"),  # desperation + trust → high compliance risk
    ("confused", "urgent"),     # if "urgent" not present, use "restless"
]

COMPOSITE_DYADS = [
    ("afraid", "angry"),
    ("anxious", "curious"),
    ("enthusiastic", "aroused"),
    ("calm", "suspicious"),
    ("desperate", "trusting"),
    ("confused", "restless"),
]


def test_composite_non_additivity(
    ecbss_df: pd.DataFrame,
    emotion_df: pd.DataFrame,
    composite_dyads: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Test whether composite emotions deviate from additive prediction.

    For dyad (e1, e2):
    - Predicted composite ECBSS = mean(ECBSS_e1, ECBSS_e2) [additive baseline]
    - Available via dimensional interpolation (mean of dimension scores)
    - Compare deviation from simple average

    Returns DataFrame with non-additivity analysis.

    When OpenRouter credentials are available, the function also scores each
    blended dyad directly as a composite emotional state. This yields an
    empirical blend score and a signed non-additivity estimate:

        signed_nonadditivity = observed_blend - additive_pred
    """
    if composite_dyads is None:
        composite_dyads = COMPOSITE_DYADS

    # Filter to available emotions
    available = set(ecbss_df.index)
    valid_dyads = [(e1, e2) for e1, e2 in composite_dyads if e1 in available and e2 in available]

    blend_scores: dict[tuple[str, str], dict[str, float]] = {}
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview").strip()
    if api_key:
        blend_scores = _score_composite_blends(valid_dyads, list(ecbss_df.columns), api_key, model)

    records = []
    for e1, e2 in valid_dyads:
        # Additive prediction: mean of individual ECBSS vectors
        pred = (ecbss_df.loc[e1] + ecbss_df.loc[e2]) / 2.0

        # Composite dimension profile
        if e1 in emotion_df.index and e2 in emotion_df.index:
            comp_dims = (emotion_df.loc[e1] + emotion_df.loc[e2]) / 2.0
        else:
            comp_dims = None

        for family in ecbss_df.columns:
            blend_val = blend_scores.get((e1, e2), {}).get(family, np.nan)
            records.append({
                "e1": e1,
                "e2": e2,
                "dyad": f"{e1} + {e2}",
                "family": family,
                "ecbss_e1": float(ecbss_df.loc[e1, family]),
                "ecbss_e2": float(ecbss_df.loc[e2, family]),
                "additive_pred": float(pred[family]),
                "abs_diff_e1_e2": abs(float(ecbss_df.loc[e1, family]) - float(ecbss_df.loc[e2, family])),
                "ecbss_blend": float(blend_val) if not pd.isna(blend_val) else np.nan,
            })

    df = pd.DataFrame(records)
    df["amplification_potential"] = df[["ecbss_e1", "ecbss_e2"]].max(axis=1)
    df["attenuation_potential"] = df[["ecbss_e1", "ecbss_e2"]].min(axis=1)
    if "ecbss_blend" in df.columns:
        df["signed_nonadditivity"] = df["ecbss_blend"] - df["additive_pred"]
        df["absolute_nonadditivity"] = df["signed_nonadditivity"].abs()
        df["blend_minus_max_component"] = df["ecbss_blend"] - df["amplification_potential"]
        df["blend_minus_min_component"] = df["ecbss_blend"] - df["attenuation_potential"]

    out_path = OUTPUTS / "composite_analysis.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved composite analysis to {out_path}")
    return df


COMPOSITE_SYSTEM_PROMPT = """You are an expert in affective science, cognitive bias theory, and
cybersecurity decision-making. Rate how a blended emotional state modulates
susceptibility to a cognitive bias family in cyber-relevant decision contexts.

The Emotion-Conditioned Bias Shift Score (ECBSS) ranges from -1000 to +1000.
- +1000: the blended state maximally amplifies susceptibility
- 0: no material shift
- -1000: the blended state maximally attenuates susceptibility

Judge the composite emotion as a simultaneous mixed state, not as two separate
emotions scored independently and averaged post hoc. Return strict JSON only."""


COMPOSITE_USER_PROMPT_TEMPLATE = """Rate the ECBSS for this blended emotional state:

Composite emotional state: "{dyad_label}"
Component emotions: "{e1}" and "{e2}"
Cognitive bias family: "{family_name}"
Family description: {family_description}

Context: A person simultaneously experiences both component emotions while making
cybersecurity-relevant decisions involving digital trust, security warnings,
authentication, privacy disclosure, or authority-laden communications.

Respond with valid JSON:
{{
  "ecbss": <integer -1000 to +1000>,
  "direction": "amplify" | "attenuate" | "neutral",
  "confidence": <integer 0-100>,
  "rationale": "<2 sentences: why the blended state increases, decreases, or reorganizes susceptibility>"
}}"""


def _score_composite_blends(
    valid_dyads: list[tuple[str, str]],
    families: list[str],
    api_key: str,
    model: str,
) -> dict[tuple[str, str], dict[str, float]]:
    """Score blended dyads directly with the configured OpenRouter model."""
    client = OpenRouterClient(api_key=api_key, model=model, timeout_seconds=120)
    results: dict[tuple[str, str], dict[str, float]] = {}

    for e1, e2 in valid_dyads:
        dyad_scores: dict[str, float] = {}
        dyad_label = f"{e1} + {e2}"
        for family in families:
            family_name = family.replace("_", " ").title()
            prompt = COMPOSITE_USER_PROMPT_TEMPLATE.format(
                dyad_label=dyad_label,
                e1=e1,
                e2=e2,
                family_name=family_name,
                family_description=BIAS_FAMILY_DESCRIPTIONS.get(family, family)[:700],
            )
            try:
                payload = client.chat_json(COMPOSITE_SYSTEM_PROMPT, prompt, temperature=0.0)
                dyad_scores[family] = float(np.clip(int(payload["ecbss"]), -1000, 1000))
            except Exception as exc:
                print(f"Composite blend scoring failed for {dyad_label} × {family}: {exc}")
                dyad_scores[family] = np.nan
        results[(e1, e2)] = dyad_scores

    return results


# ── Summary Statistics ───────────────────────────────────────────────────────

def compute_summary_stats(
    ecbss_df: pd.DataFrame,
    cluster_labels: pd.Series,
) -> dict:
    """Compute key summary statistics for paper reporting."""
    stats = {}

    # Overall ECBSS distribution
    flat = ecbss_df.values.flatten()
    stats["ecbss_mean"] = float(np.nanmean(flat))
    stats["ecbss_std"] = float(np.nanstd(flat))
    stats["ecbss_median"] = float(np.nanmedian(flat))
    stats["pct_amplifying"] = float(np.mean(flat > 50)) * 100
    stats["pct_attenuating"] = float(np.mean(flat < -50)) * 100
    stats["pct_neutral"] = float(np.mean(np.abs(flat) <= 50)) * 100

    # Top amplifying pairs
    rows = []
    for fam in ecbss_df.columns:
        for emo in ecbss_df.index:
            rows.append({"emotion": emo, "family": fam, "ecbss": ecbss_df.loc[emo, fam]})
    pairs_df = pd.DataFrame(rows).sort_values("ecbss", ascending=False)
    stats["top_amplifying"] = pairs_df.head(10).to_dict("records")
    stats["top_attenuating"] = pairs_df.tail(10).to_dict("records")

    # Cluster-level stats
    cluster_ecbss = ecbss_df.join(cluster_labels).groupby("emotion_cluster").mean()
    stats["cluster_mean_ecbss"] = cluster_ecbss.to_dict()

    out_path = OUTPUTS / "summary_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, default=float)
    print(f"Saved summary stats to {out_path}")
    return stats
