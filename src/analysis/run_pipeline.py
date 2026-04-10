"""Full analysis pipeline orchestrator.

Runs all stages in order, with caching at each step.
Usage: python3 src/analysis/run_pipeline.py [--force-all] [--force-llm]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np
import pandas as pd

OUTPUTS = ROOT / "src/review_stages/analysis_outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def run_pipeline(force_all: bool = False, force_llm: bool = False) -> None:
    print("=" * 70)
    print("COGNITIVE BIAS VULNERABILITY ANALYSIS PIPELINE")
    print("=" * 70)

    # ─── Stage 1: Load Data ──────────────────────────────────────────────
    print("\n[1/9] Loading data assets...")
    from analysis.data_loader import load_taxonomy, load_emotion_scores, load_emotion_lexicon

    _, taxonomy_df = load_taxonomy()
    emotion_df = load_emotion_scores()
    lexicon = load_emotion_lexicon()
    print(f"  Taxonomy: {len(taxonomy_df)} leaf biases, "
          f"{taxonomy_df['cluster'].nunique()} clusters, "
          f"{taxonomy_df['family'].nunique()} families")
    print(f"  Emotion dimension scores: {len(emotion_df)} emotions")
    print(f"  Emotion lexicon: {len(lexicon)} terms")

    # ─── Stage 2: Bias Family Sensitivity Profiling ──────────────────────
    print("\n[2/9] Bias family dimensional sensitivity profiling (LLM)...")
    from analysis.bias_profiler import run_bias_profiling
    bias_profiles = run_bias_profiling(force=force_all or force_llm)
    print(f"  Profiled {len(bias_profiles)} bias families")

    # ─── Stage 3: Analytical ECBSS Matrix ───────────────────────────────
    print("\n[3/9] Computing analytical ECBSS matrix...")
    from analysis.ecbss_scorer import compute_analytical_ecbss, save_analytical_ecbss, ANALYTICAL_OUTPUT

    if ANALYTICAL_OUTPUT.exists() and not force_all:
        print(f"  Loading from {ANALYTICAL_OUTPUT}")
        analytical_ecbss = pd.read_csv(ANALYTICAL_OUTPUT, index_col="emotion")
    else:
        analytical_ecbss = compute_analytical_ecbss(emotion_df, bias_profiles)
        save_analytical_ecbss(analytical_ecbss)
    print(f"  Analytical ECBSS: {analytical_ecbss.shape} (emotions × families)")

    # ─── Stage 4: LLM Direct ECBSS Scoring ──────────────────────────────
    print("\n[4/9] LLM direct ECBSS scoring (parallel)...")
    from analysis.ecbss_scorer import run_llm_direct_scoring, DIRECT_OUTPUT

    emotions_to_score = emotion_df.index.tolist()
    family_keys = list(bias_profiles.keys())

    direct_ecbss = run_llm_direct_scoring(
        emotions=emotions_to_score,
        family_keys=family_keys,
        max_workers=50,
        force=force_all or force_llm,
    )
    print(f"  LLM direct ECBSS: {direct_ecbss.shape} (emotions × families)")

    # ─── Stage 5: Statistical Analysis ──────────────────────────────────
    print("\n[5/9] Statistical analysis...")
    from analysis.statistical_analysis import (
        cluster_emotions,
        get_emotion_cluster_profiles,
        hierarchical_clustering_emotions,
        hierarchical_clustering_biases,
        run_pca,
        run_umap,
        compute_cluster_ecbss,
        run_mixed_effects_regression,
        validate_analytical_vs_direct,
        test_composite_non_additivity,
        compute_summary_stats,
    )

    # Use LLM direct as primary ECBSS
    primary_ecbss = direct_ecbss.copy()

    # Align with emotion_df
    common_emotions = emotion_df.index.intersection(primary_ecbss.index)
    primary_ecbss = primary_ecbss.loc[common_emotions]
    emotion_df_aligned = emotion_df.loc[common_emotions]
    analytical_aligned = analytical_ecbss.loc[analytical_ecbss.index.intersection(primary_ecbss.index)]

    print(f"  Common emotions (in both dim scores + ECBSS): {len(common_emotions)}")

    # Emotion clustering
    cluster_labels, cluster_centers = cluster_emotions(emotion_df_aligned)
    print(f"  Emotion clusters: {cluster_labels.nunique()} clusters")

    # Cluster dimensional profiles
    cluster_profiles = get_emotion_cluster_profiles(emotion_df_aligned, cluster_labels)
    cluster_profiles.to_csv(OUTPUTS / "cluster_profiles.csv")
    print(f"  Cluster profiles saved")

    # Cluster-level ECBSS
    cluster_ecbss = compute_cluster_ecbss(primary_ecbss, cluster_labels)
    cluster_ecbss.to_csv(OUTPUTS / "cluster_ecbss.csv")
    print(f"  Cluster ECBSS matrix: {cluster_ecbss.shape}")

    # Hierarchical clustering
    link_emotions = hierarchical_clustering_emotions(emotion_df_aligned)
    link_biases = hierarchical_clustering_biases(primary_ecbss)
    np.save(OUTPUTS / "linkage_emotions.npy", link_emotions)
    np.save(OUTPUTS / "linkage_biases.npy", link_biases)
    print(f"  Hierarchical clustering computed")

    # PCA
    pca_coords, pca_model = run_pca(emotion_df_aligned)
    np.save(OUTPUTS / "pca_coords.npy", pca_coords)
    pca_ev = pca_model.explained_variance_ratio_
    print(f"  PCA: PC1={pca_ev[0]*100:.1f}%, PC2={pca_ev[1]*100:.1f}%")

    # UMAP
    print("  Computing UMAP embedding...")
    umap_path = OUTPUTS / "umap_coords.npy"
    if umap_path.exists() and not force_all:
        umap_coords = np.load(umap_path)
        print(f"  Loaded UMAP from {umap_path}")
    else:
        umap_coords = run_umap(emotion_df_aligned)
        np.save(umap_path, umap_coords)
    print(f"  UMAP embedding: {umap_coords.shape}")

    # Save cluster labels
    cluster_labels.to_csv(OUTPUTS / "emotion_cluster_labels.csv")

    # Regression
    print("  Running mixed-effects regression...")
    regression_results = run_mixed_effects_regression(primary_ecbss, emotion_df_aligned)

    # Validation
    print("  Validating analytical vs LLM direct...")
    validation = validate_analytical_vs_direct(analytical_aligned, primary_ecbss)
    print(f"  Validation: r={validation['pearson_r']:.3f}, ρ={validation['spearman_r']:.3f}")

    # Composite emotions
    print("  Testing composite emotion non-additivity...")
    composite_df = test_composite_non_additivity(primary_ecbss, emotion_df_aligned)

    # Summary stats
    summary = compute_summary_stats(primary_ecbss, cluster_labels)
    print(f"  Summary: {summary['pct_amplifying']:.1f}% amplifying, "
          f"{summary['pct_attenuating']:.1f}% attenuating, "
          f"{summary['pct_neutral']:.1f}% neutral pairs")

    # ─── Stage 6: Network Analysis ───────────────────────────────────────
    print("\n[6/9] Network analysis...")
    from analysis.network_analysis import (
        build_bipartite_network,
        compute_network_metrics,
        compute_emotion_emotion_similarity,
        compute_bias_bias_similarity,
    )

    G = build_bipartite_network(cluster_ecbss, ecbss_threshold=80)
    net_metrics = compute_network_metrics(G)

    with open(OUTPUTS / "network_metrics.json", "w") as f:
        json.dump(net_metrics, f, indent=2, default=float)

    sim_emotions = compute_emotion_emotion_similarity(primary_ecbss)
    sim_biases = compute_bias_bias_similarity(primary_ecbss)
    sim_emotions.to_csv(OUTPUTS / "emotion_similarity.csv")
    sim_biases.to_csv(OUTPUTS / "bias_similarity.csv")

    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Density: {net_metrics['density']:.3f}")
    print(f"  Amplifying: {net_metrics.get('n_amplifying_edges', 0)} edges, "
          f"Attenuating: {net_metrics.get('n_attenuating_edges', 0)} edges")

    # ─── Stage 7: Generate Figures ───────────────────────────────────────
    print("\n[7/9] Generating figures...")
    from analysis.visualizations import (
        fig1_taxonomy_overview,
        fig2_emotion_landscape,
        fig3_ecbss_heatmap,
        fig4_dimensional_profiles,
        fig5_network,
        fig6_regression,
        fig7_hypothesis_evaluation,
        figS1_full_ecbss_heatmap,
        figS2_validation_scatter,
        figS3_composite_emotions,
        figS4_cluster_profiles,
    )

    _, taxonomy_df_fresh = load_taxonomy()

    print("  Figure 1: Taxonomy overview...")
    fig1_taxonomy_overview(taxonomy_df_fresh)

    print("  Figure 2: Emotion landscape (UMAP)...")
    fig2_emotion_landscape(emotion_df_aligned, umap_coords, cluster_labels)

    print("  Figure 3: ECBSS heatmap...")
    fig3_ecbss_heatmap(cluster_ecbss, primary_ecbss, cluster_labels)

    print("  Figure 4: Dimensional profiles...")
    fig4_dimensional_profiles(bias_profiles)

    print("  Figure 5: Network...")
    fig5_network(cluster_ecbss, cluster_labels, taxonomy_df_fresh)

    print("  Figure 6: Regression results...")
    fig6_regression(regression_results)

    print("  Figure 7: Hypothesis evaluation...")
    fig7_hypothesis_evaluation(cluster_ecbss, primary_ecbss, cluster_labels)

    print("  Figure S1: Full ECBSS heatmap...")
    figS1_full_ecbss_heatmap(primary_ecbss, cluster_labels)

    print("  Figure S2: Validation scatter...")
    figS2_validation_scatter(analytical_aligned, primary_ecbss)

    print("  Figure S3: Composite emotions...")
    figS3_composite_emotions(composite_df)

    print("  Figure S4: Cluster profiles...")
    figS4_cluster_profiles(emotion_df_aligned, cluster_labels)

    print("  All figures generated!")

    # ─── Stage 8: Generate Tables ────────────────────────────────────────
    print("\n[8/9] Generating tables...")
    generate_tables(taxonomy_df_fresh, primary_ecbss, cluster_ecbss, cluster_labels,
                    regression_results, summary, validation, bias_profiles)

    # ─── Stage 9: Save Results JSON for LaTeX ────────────────────────────
    print("\n[9/9] Saving results summary for LaTeX...")

    results_for_paper = {
        "n_families": int(taxonomy_df_fresh["family"].nunique()),
        "n_clusters": int(taxonomy_df_fresh["cluster"].nunique()),
        "n_leaf_biases": int(len(taxonomy_df_fresh)),
        "n_emotions_scored": int(len(common_emotions)),
        "n_ecbss_pairs": int(len(common_emotions) * len(family_keys)),
        "pct_amplifying": float(summary["pct_amplifying"]),
        "pct_attenuating": float(summary["pct_attenuating"]),
        "pct_neutral": float(summary["pct_neutral"]),
        "validation_pearson_r": float(validation["pearson_r"]),
        "validation_spearman_r": float(validation["spearman_r"]),
        "n_emotion_clusters": int(cluster_labels.nunique()),
        "top_amplifying": summary["top_amplifying"][:5],
        "top_attenuating": summary["top_attenuating"][:5],
        "network_density": float(net_metrics["density"]),
        "n_network_edges": int(G.number_of_edges()),
        "overall_regression": {
            "params": regression_results["overall"].get("params", {}),
            "pvalues": regression_results["overall"].get("pvalues", {}),
        },
    }

    with open(OUTPUTS / "results_for_paper.json", "w") as f:
        json.dump(results_for_paper, f, indent=2, default=float)
    print(f"  Results saved to {OUTPUTS}/results_for_paper.json")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Figures saved to: {ROOT / 'paper/figures'}")
    print(f"  Tables saved to:  {ROOT / 'paper/tables'}")
    print(f"  Results JSON:     {OUTPUTS / 'results_for_paper.json'}")


def generate_tables(
    taxonomy_df: pd.DataFrame,
    ecbss_df: pd.DataFrame,
    cluster_ecbss: pd.DataFrame,
    cluster_labels: pd.Series,
    regression_results: dict,
    summary: dict,
    validation: dict,
    bias_profiles: dict,
) -> None:
    """Generate LaTeX-ready tables."""
    tables_dir = ROOT / "paper/assets/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    from analysis.data_loader import get_family_short_labels
    FAMILY_SHORT_LABELS = get_family_short_labels()

    # Table 1: Taxonomy summary
    family_stats = taxonomy_df.groupby("family").agg(
        N_clusters=("cluster", "nunique"),
        N_biases=("leaf_bias", "count"),
    ).reset_index()
    family_stats["Family"] = family_stats["family"].map(
        lambda x: FAMILY_SHORT_LABELS.get(x, x[:25]).replace("\n", " ")
    )
    family_stats = family_stats[["Family", "N_clusters", "N_biases"]].sort_values("N_biases", ascending=False)

    with open(tables_dir / "table1_taxonomy.tex", "w") as f:
        f.write("% Table 1: Taxonomy Summary\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Bias Family & $N_{\\text{clusters}}$ & $N_{\\text{biases}}$ \\\\\n")
        f.write("\\midrule\n")
        total_clusters = 0
        total_biases = 0
        for _, row in family_stats.iterrows():
            f.write(f"{row['Family']} & {row['N_clusters']} & {row['N_biases']} \\\\\n")
            total_clusters += row['N_clusters']
            total_biases += row['N_biases']
        f.write("\\midrule\n")
        f.write(f"\\textit{{Total}} & {total_clusters} & {total_biases} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    # Table 2: Top amplifying/attenuating pairs
    rows = []
    for fam in ecbss_df.columns:
        for emo in ecbss_df.index:
            rows.append({"emotion": emo, "family": fam, "ecbss": ecbss_df.loc[emo, fam]})
    pairs_df = pd.DataFrame(rows)

    top_amp = pairs_df.nlargest(10, "ecbss")
    top_att = pairs_df.nsmallest(10, "ecbss")

    with open(tables_dir / "table2_top_pairs.tex", "w") as f:
        f.write("% Table 2: Top Amplifying and Attenuating Emotion-Bias Pairs\n")
        f.write("\\begin{tabular}{llr}\n")
        f.write("\\toprule\n")
        f.write("Emotion & Bias Family & ECBSS \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{3}{l}{\\textit{Top amplifying (ECBSS > 0)}} \\\\\n")
        for _, row in top_amp.iterrows():
            fam_short = FAMILY_SHORT_LABELS.get(row["family"], row["family"][:20]).replace("\n", " ")
            f.write(f"{row['emotion'].capitalize()} & {fam_short} & +{row['ecbss']:.0f} \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{3}{l}{\\textit{Top attenuating (ECBSS < 0)}} \\\\\n")
        for _, row in top_att.iterrows():
            fam_short = FAMILY_SHORT_LABELS.get(row["family"], row["family"][:20]).replace("\n", " ")
            f.write(f"{row['emotion'].capitalize()} & {fam_short} & {row['ecbss']:.0f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    # Table 3: Regression summary
    per_family = regression_results.get("per_family", {})
    dims = ["V", "A", "C", "U", "S"]

    with open(tables_dir / "table3_regression.tex", "w") as f:
        f.write("% Table 3: Regression Results\n")
        f.write("\\begin{tabular}{l" + "r" * len(dims) + "r}\n")
        f.write("\\toprule\n")
        f.write("Family & " + " & ".join(f"$\\beta_{{\\text{{{d}}}}}$" for d in dims) + " & $R^2$ \\\\\n")
        f.write("\\midrule\n")
        for fam in [f for f in FAMILY_SHORT_LABELS if f in per_family]:
            res = per_family[fam]
            params = res.get("params", {})
            pvals = res.get("pvalues", {})
            r2 = res.get("rsquared", 0)
            fam_short = FAMILY_SHORT_LABELS.get(fam, fam[:15]).replace("\n", " ")
            betas = []
            for d in dims:
                b = params.get(d, 0)
                p = pvals.get(d, 1)
                sig = "^{***}" if p < 0.001 else ("^{**}" if p < 0.01 else ("^{*}" if p < 0.05 else ""))
                betas.append(f"${b:.2f}{sig}$")
            f.write(f"{fam_short} & " + " & ".join(betas) + f" & {r2:.3f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    # Table 4: Bias family sensitivity profiles
    with open(tables_dir / "table4_bias_sensitivity.tex", "w") as f:
        f.write("% Table 4: Bias Family Dimensional Sensitivity Profiles\n")
        f.write("\\begin{tabular}{l" + "r" * 5 + "}\n")
        f.write("\\toprule\n")
        f.write("Family & V & A & C & U & S \\\\\n")
        f.write("\\midrule\n")
        for fam_key in FAMILY_SHORT_LABELS:
            if fam_key not in bias_profiles:
                continue
            profile = bias_profiles[fam_key]
            fam_short = FAMILY_SHORT_LABELS[fam_key].replace("\n", " ")
            vals = [f"{profile.get(d, 0):+.0f}" for d in dims]
            f.write(f"{fam_short} & " + " & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    # Table S1: Cluster ECBSS
    with open(tables_dir / "tableS1_cluster_ecbss.tex", "w") as f:
        families_in_cluster = [f for f in FAMILY_SHORT_LABELS if f in cluster_ecbss.columns]
        short_fams = [FAMILY_SHORT_LABELS.get(f, f[:10]).replace("\n", " ") for f in families_in_cluster]
        f.write("% Table S1: Cluster-Level Mean ECBSS\n")
        f.write("\\begin{tabular}{l" + "r" * len(families_in_cluster) + "}\n")
        f.write("\\toprule\n")
        f.write("Cluster & " + " & ".join(short_fams) + " \\\\\n")
        f.write("\\midrule\n")
        for cid in sorted(cluster_ecbss.index):
            cname = {0: "Hi-Ar.-Neg.", 1: "Lo-Val.-Wth.", 2: "Calm Pos.",
                     3: "Hi-Ar.-Pos.", 4: "Soc.Threat", 5: "Hostile"}.get(cid, f"C{cid}")
            vals = [f"{cluster_ecbss.loc[cid, f]:+.0f}" if f in cluster_ecbss.columns else "—"
                    for f in families_in_cluster]
            f.write(f"{cname} & " + " & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"  Tables saved to {tables_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full analysis pipeline")
    parser.add_argument("--force-all", action="store_true", help="Force re-run all stages")
    parser.add_argument("--force-llm", action="store_true", help="Force re-run LLM scoring only")
    args = parser.parse_args()

    run_pipeline(force_all=args.force_all, force_llm=args.force_llm)
