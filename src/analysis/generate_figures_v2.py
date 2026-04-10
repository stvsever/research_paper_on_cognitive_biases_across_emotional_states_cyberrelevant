"""Generate all Nature-style figures using visualizations_v2.py."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUTPUTS = ROOT / "src/review_stages/analysis_outputs"

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from analysis.data_loader import load_taxonomy, load_emotion_scores
from analysis.statistical_analysis import cluster_emotions, run_umap, compute_cluster_ecbss
from analysis.network_analysis import build_bipartite_network, compute_network_metrics


def load_or_compute_umap(emotion_df, force=False):
    path = OUTPUTS / "umap_coords.npy"
    if path.exists() and not force:
        coords = np.load(path)
        print("UMAP: loaded from cache")
    else:
        print("UMAP: computing...")
        coords = run_umap(emotion_df)
        np.save(path, coords)
        print("UMAP: done")
    return coords


def main():
    print("Loading data...")
    _, taxonomy_df = load_taxonomy()
    emotion_df = load_emotion_scores()

    ecbss_df = pd.read_csv(OUTPUTS / "ecbss_direct.csv", index_col="emotion")
    analytical_df = pd.read_csv(OUTPUTS / "ecbss_analytical.csv", index_col="emotion") \
        if (OUTPUTS / "ecbss_analytical.csv").exists() else None

    # Align
    common = emotion_df.index.intersection(ecbss_df.index)
    emotion_df = emotion_df.loc[common]
    ecbss_df = ecbss_df.loc[common]
    if analytical_df is not None:
        analytical_df = analytical_df.loc[common] if all(i in analytical_df.index for i in common) else None

    # Cluster labels
    cluster_labels, _ = cluster_emotions(emotion_df)
    print(f"Emotions: {len(common)}, Clusters: {cluster_labels.nunique()}")

    # UMAP coords
    umap_coords = load_or_compute_umap(emotion_df, force=False)

    # Cluster ECBSS
    cluster_ecbss = compute_cluster_ecbss(ecbss_df, cluster_labels)

    # Load regression results
    with open(OUTPUTS / "regression_results.json") as f:
        regression_results = json.load(f)

    # Load bias profiles
    with open(OUTPUTS / "bias_sensitivity_profiles.json") as f:
        bias_profiles = json.load(f)

    # Load advanced analysis results
    bootstrap_cis = None
    perm_results = None
    cohen_d_results = None
    pca_results = None
    variance_results = None

    if (OUTPUTS / "bootstrap_cluster_means.json").exists():
        with open(OUTPUTS / "bootstrap_cluster_means.json") as f:
            bdata = json.load(f)
        # Reshape: {cluster_id_str: {family: {mean, ci_lower, ci_upper}}}
        # → {(cluster_id_int, family): (ci_lower, ci_upper)}  (tuple-key format expected by figS7)
        bootstrap_cis = {}
        for cid_str, fam_dict in bdata.items():
            cid = int(cid_str)
            for fam, vals in fam_dict.items():
                if vals.get("ci_lower") is not None:
                    bootstrap_cis[(cid, fam)] = (vals["ci_lower"], vals["ci_upper"])
                    bootstrap_cis[(cid, fam, "mean")] = vals.get("mean")
        bootstrap_cis = bootstrap_cis if bootstrap_cis else None
        print(f"Bootstrap CIs: {len(bootstrap_cis or {})} cells")

    if (OUTPUTS / "permutation_test_regression.json").exists():
        with open(OUTPUTS / "permutation_test_regression.json") as f:
            perm_results = json.load(f)
        print(f"Permutation results loaded")

    if (OUTPUTS / "pca_with_loadings.json").exists():
        with open(OUTPUTS / "pca_with_loadings.json") as f:
            pca_results = json.load(f)
        print(f"PCA results loaded")

    pca_coords = np.load(OUTPUTS / "pca_coords.npy") if (OUTPUTS / "pca_coords.npy").exists() else None

    if (OUTPUTS / "variance_decomposition.json").exists():
        with open(OUTPUTS / "variance_decomposition.json") as f:
            variance_results = json.load(f)
        print(f"Variance decomposition loaded")

    cohen_d_p = OUTPUTS / f"cohen_d_social_influence_authority_affiliation_and_identity_bias.json"
    # Try short name
    for p in OUTPUTS.glob("cohen_d_*.json"):
        with open(p) as f:
            cohen_d_results = json.load(f)
        print(f"Cohen's d: d={cohen_d_results.get('cohen_d', '?'):.3f}")
        break

    # Load composite analysis
    composite_df = None
    if (OUTPUTS / "composite_analysis.csv").exists():
        composite_df = pd.read_csv(OUTPUTS / "composite_analysis.csv")
        print(f"Composite analysis: {len(composite_df)} rows")

    # Generate all figures
    from analysis.visualizations_v2 import run_all_figures
    run_all_figures(
        taxonomy_df=taxonomy_df,
        emotion_df=emotion_df,
        umap_coords=umap_coords,
        cluster_labels=cluster_labels,
        cluster_ecbss=cluster_ecbss,
        ecbss_df=ecbss_df,
        bias_profiles=bias_profiles,
        regression_results=regression_results,
        permutation_results=perm_results,
        bootstrap_cis=bootstrap_cis,
        cohen_d_results=cohen_d_results,
        pca_results=pca_results,
        pca_coords=pca_coords,
        analytical_df=analytical_df,
        direct_df=ecbss_df,
        composite_df=composite_df,
        variance_results=variance_results,
    )

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
