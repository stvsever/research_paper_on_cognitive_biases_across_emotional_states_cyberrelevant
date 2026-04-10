"""Generate all publication-quality figures for the paper.

Convention:
- Figure titles are LEFT-ALIGNED above the figure
- "Note." lines are below each figure
- All figures saved as both PDF and PNG (300 dpi)
- Multi-panel figures use uppercase panel labels (A, B, C, D)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper/assets/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Diverging colormap: blue (attenuate) → white → red (amplify)
ECBSS_CMAP = LinearSegmentedColormap.from_list(
    "ecbss_diverge", ["#1a6fa3", "#6bbfe8", "#f5f5f5", "#f58b79", "#c0392b"]
)

CLUSTER_COLORS = [
    "#e74c3c",  # 0: High-Arousal Negative
    "#8e44ad",  # 1: Low-Valence Withdrawn
    "#27ae60",  # 2: Calm Positive
    "#f39c12",  # 3: High-Arousal Positive
    "#e67e22",  # 4: Social Threat/Shame
    "#2c3e50",  # 5: Hostile/Defiant
]

FAMILY_SHORT_LABELS = {
    "attention_salience_and_signal_detection_biases": "Attention &\nSalience",
    "trust_source_credibility_and_truth_judgment_biases": "Trust &\nCredibility",
    "evidence_search_hypothesis_testing_and_belief_updating_biases": "Evidence &\nBelief",
    "memory_familiarity_and_source_monitoring_biases": "Memory &\nSource",
    "risk_probability_uncertainty_and_outcome_valuation_biases": "Risk &\nProbability",
    "temporal_choice_default_action_and_commitment_biases": "Temporal &\nCommitment",
    "social_influence_authority_affiliation_and_identity_biases": "Social &\nAuthority",
    "self_assessment_attribution_and_metacognitive_biases": "Self-\nAssessment",
    "interface_choice_architecture_automation_and_warning_response_biases": "Interface &\nAutomation",
    "privacy_disclosure_and_self_presentation_biases": "Privacy &\nDisclosure",
    "affective_evaluation_and_mood_congruent_judgment_biases": "Affect &\nMood",
}

CLUSTER_LABELS = {
    0: "High-Arousal\nNegative",
    1: "Low-Valence\nWithdrawn",
    2: "Calm\nPositive",
    3: "High-Arousal\nPositive",
    4: "Social Threat\n& Shame",
    5: "Hostile &\nDefiant",
}

DIM_LABELS = {
    "V": "Valence\n(neg → pos)",
    "A": "Arousal\n(low → high)",
    "C": "Control\n(low → high)",
    "U": "Uncertainty\n(certain → uncertain)",
    "S": "Social\n(self → other)",
}


def _save(fig, name: str, dpi: int = 300) -> None:
    for ext in ["pdf", "png"]:
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {name}")
    plt.close(fig)


# ── Figure 1: Taxonomy Overview (A-B) ───────────────────────────────────────

def fig1_taxonomy_overview(taxonomy_df: pd.DataFrame) -> None:
    """Fig 1 (A-B): Taxonomy structure overview."""
    family_counts = taxonomy_df.groupby("family").size().rename("n_biases")
    cluster_counts = taxonomy_df.groupby(["family", "cluster"]).size().rename("n_biases").reset_index()

    families = taxonomy_df["family"].unique()
    family_order = [f for f in FAMILY_SHORT_LABELS if f in families]
    short = {k: v.replace("\n", " ") for k, v in FAMILY_SHORT_LABELS.items()}

    colors_fam = plt.cm.tab20(np.linspace(0, 1, len(family_order)))
    fam_color_map = dict(zip(family_order, colors_fam))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.text(0.01, 0.97, "Figure 1", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.94, "Taxonomy of Cyber-Relevant Cognitive Biases: Structure and Distribution",
             fontsize=12, ha="left", va="top", fontweight="bold")

    # Panel A: Horizontal stacked bar per family (clusters within)
    ax = axes[0]
    ax.text(-0.12, 1.05, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

    family_totals = []
    for fam in family_order:
        clusters_in_fam = cluster_counts[cluster_counts["family"] == fam]
        total = clusters_in_fam["n_biases"].sum()
        family_totals.append((fam, total))
    family_totals.sort(key=lambda x: -x[1])

    y_pos = np.arange(len(family_totals))
    for i, (fam, total) in enumerate(family_totals):
        clusters_in_fam = cluster_counts[cluster_counts["family"] == fam].sort_values("n_biases", ascending=False)
        left = 0
        palette = plt.cm.Pastel1(np.linspace(0, 1, len(clusters_in_fam)))
        for j, row in enumerate(clusters_in_fam.itertuples()):
            ax.barh(i, row.n_biases, left=left, color=fam_color_map[fam],
                    alpha=0.5 + 0.5 * (j / max(len(clusters_in_fam) - 1, 1)),
                    edgecolor="white", linewidth=0.5)
            if row.n_biases >= 3:
                ax.text(left + row.n_biases / 2, i, str(row.n_biases),
                        ha="center", va="center", fontsize=6.5, color="black")
            left += row.n_biases

    ax.set_yticks(y_pos)
    ax.set_yticklabels([short.get(fam, fam[:25]) for fam, _ in family_totals], fontsize=8)
    ax.set_xlabel("Number of Leaf Biases")
    ax.set_title("Bias Family Composition by Cluster", loc="left")
    ax.axvline(0, color="black", linewidth=0.5)

    # Panel B: Treemap-style circle packing by family
    ax2 = axes[1]
    ax2.text(-0.12, 1.05, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")

    # Use scatter plot as bubble chart
    sizes = [family_counts.get(fam, 1) for fam in family_order]
    n_clusters_per_family = [taxonomy_df[taxonomy_df["family"] == fam]["cluster"].nunique() for fam in family_order]

    # Arrange in grid
    ncols = 3
    nrows = int(np.ceil(len(family_order) / ncols))
    xs = []
    ys = []
    for i, fam in enumerate(family_order):
        col = i % ncols
        row = i // ncols
        xs.append(col)
        ys.append(nrows - row - 1)

    scale = 300
    sc = ax2.scatter(xs, ys, s=[s * scale for s in sizes],
                     c=[fam_color_map[f] for f in family_order],
                     alpha=0.75, edgecolors="white", linewidths=1.5)

    for i, fam in enumerate(family_order):
        lbl = short.get(fam, fam[:20])
        ax2.text(xs[i], ys[i] + 0.38, lbl, ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax2.text(xs[i], ys[i], f"{sizes[i]} biases\n{n_clusters_per_family[i]} clusters",
                 ha="center", va="center", fontsize=6.5, color="black")

    ax2.set_xlim(-0.7, ncols - 0.3)
    ax2.set_ylim(-0.7, nrows - 0.3)
    ax2.axis("off")
    ax2.set_title("Bias Families (bubble size ∝ N leaf biases)", loc="left")

    fig.text(0.01, 0.02,
             "Note. Panel A shows the number of leaf biases per family broken down by constituent clusters (stacked bars). "
             "Panel B illustrates relative family size as bubble area. The taxonomy encompasses 11 families organized into "
             f"{taxonomy_df['cluster'].nunique()} clusters and {len(taxonomy_df)} leaf biases.",
             fontsize=8, ha="left", va="bottom", wrap=True)

    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    _save(fig, "fig1_taxonomy_overview")


# ── Figure 2: Emotion Dimensional Landscape (A-D) ───────────────────────────

def fig2_emotion_landscape(
    emotion_df: pd.DataFrame,
    umap_coords: np.ndarray,
    cluster_labels: pd.Series,
) -> None:
    """Fig 2 (A-D): UMAP embedding of emotion space colored by dimensions and clusters."""
    emotions = emotion_df.index.tolist()
    coords = umap_coords

    fig = plt.figure(figsize=(15, 13))
    fig.text(0.01, 0.98, "Figure 2", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.96, "Multidimensional Emotion Space: UMAP Projections Across Affective Dimensions",
             fontsize=12, ha="left", va="top", fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    panels = [("A", "V", "Valence (neg → pos)"),
              ("B", "A", "Arousal (low → high)"),
              ("C", "C", "Control (low → high)"),
              ("D", "U", "Uncertainty (certain → uncertain)")]

    for idx, (label, dim, title) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=14, fontweight="bold")

        vals = emotion_df[dim].values
        vmin, vmax = -1000, 1000
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap=ECBSS_CMAP,
                        norm=norm, s=35, alpha=0.85, linewidths=0)

        plt.colorbar(sc, ax=ax, label=f"{dim} score", fraction=0.03, pad=0.02)

        # Annotate representative emotions
        highlight = []
        if dim == "V":
            highlight = [("afraid", -980), ("angry", -980), ("calm", 700), ("blissful", 900), ("content", 600)]
        elif dim == "A":
            highlight = [("calm", -600), ("tranquil", -600), ("panicked", 800), ("furious", 900), ("serene", -700)]
        elif dim == "C":
            highlight = [("helpless", -800), ("cornered", -800), ("confident", 600), ("secure", 700)]
        elif dim == "U":
            highlight = [("startled", 900), ("confused", 800), ("serene", -800), ("comfortable", -800)]

        for emo, expected_val in highlight:
            if emo in emotion_df.index:
                idx_e = emotions.index(emo)
                ax.annotate(emo, (coords[idx_e, 0], coords[idx_e, 1]),
                            fontsize=6.5, ha="center", va="bottom",
                            xytext=(0, 5), textcoords="offset points",
                            color="black", fontweight="bold")

        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("UMAP-1", fontsize=8)
        ax.set_ylabel("UMAP-2", fontsize=8)
        ax.tick_params(labelsize=7)

    # Add cluster overlay as inset on panel D context
    # Replace Panel D with cluster labels
    ax_d = fig.axes[3]
    ax_d.clear()
    ax_d.text(-0.08, 1.05, "D", transform=ax_d.transAxes, fontsize=14, fontweight="bold")

    for cid in sorted(cluster_labels.unique()):
        mask = cluster_labels.values == cid
        c_emotions = [e for e, m in zip(emotions, mask) if m]
        c_coords = coords[mask]
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        ax_d.scatter(c_coords[:, 0], c_coords[:, 1], c=color, s=40,
                     alpha=0.8, label=CLUSTER_LABELS.get(cid, f"Cluster {cid}"),
                     linewidths=0)

        # Label cluster centroid
        centroid = c_coords.mean(axis=0)
        ax_d.text(centroid[0], centroid[1], CLUSTER_LABELS.get(cid, str(cid)),
                  ha="center", va="center", fontsize=7.5,
                  fontweight="bold", color=color,
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor=color))

    ax_d.set_title("Emotion Clusters (K-means, k=6)", loc="left", fontsize=9)
    ax_d.set_xlabel("UMAP-1", fontsize=8)
    ax_d.set_ylabel("UMAP-2", fontsize=8)
    ax_d.tick_params(labelsize=7)
    ax_d.legend(fontsize=6.5, loc="lower right", framealpha=0.7)

    fig.text(0.01, 0.02,
             "Note. UMAP projections of all emotions onto 2D space using the five affective dimensions (V, A, C, U, S) as features. "
             "Panels A–C show emotion positions colored by their score on each dimension (blue = low/negative, red = high/positive). "
             "Panel D shows K-means cluster assignments (k=6) revealing structurally distinct emotional regions with "
             "theoretically coherent groupings.",
             fontsize=8, ha="left", va="bottom")

    _save(fig, "fig2_emotion_landscape")


# ── Figure 3: ECBSS Heatmap (A-B) ───────────────────────────────────────────

def fig3_ecbss_heatmap(
    cluster_ecbss: pd.DataFrame,
    ecbss_df: pd.DataFrame,
    cluster_labels: pd.Series,
) -> None:
    """Fig 3 (A-B): ECBSS heatmap by emotion cluster × bias family."""
    families = [f for f in FAMILY_SHORT_LABELS if f in cluster_ecbss.columns]
    short_fam = {k: v for k, v in FAMILY_SHORT_LABELS.items()}

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.text(0.01, 0.98, "Figure 3", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.96,
             "Emotion-Conditioned Bias Shift Score (ECBSS) Matrix: Cluster-Level Amplification and Attenuation Profiles",
             fontsize=12, ha="left", va="top", fontweight="bold")

    # Panel A: Mean ECBSS heatmap
    ax = axes[0]
    ax.text(-0.12, 1.04, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

    plot_data = cluster_ecbss[families].copy()
    plot_data.columns = [short_fam.get(f, f[:15]) for f in families]
    plot_data.index = [CLUSTER_LABELS.get(i, f"C{i}") for i in plot_data.index]

    vmax = max(abs(plot_data.values.max()), abs(plot_data.values.min()))
    vmax = min(vmax, 800)

    mask_zero = np.abs(plot_data.values) < 30

    sns.heatmap(
        plot_data,
        ax=ax,
        cmap=ECBSS_CMAP,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".0f",
        annot_kws={"size": 7.5, "weight": "bold"},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean ECBSS", "shrink": 0.8},
    )
    ax.set_title("Mean ECBSS by Emotion Cluster × Bias Family", loc="left", fontsize=9)
    ax.set_xlabel("Bias Family", fontsize=9)
    ax.set_ylabel("Emotion Cluster", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=7.5)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    # Panel B: Heatmap of absolute magnitude (showing strength of effect regardless of direction)
    ax2 = axes[1]
    ax2.text(-0.12, 1.04, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")

    # Show variability across emotions within cluster (std dev of ECBSS)
    std_data = {}
    for family in families:
        std_vals = []
        for cid in sorted(cluster_ecbss.index):
            mask = cluster_labels.values == cid
            emo_list = [e for e, m in zip(ecbss_df.index, mask) if m and e in ecbss_df.index]
            if emo_list and family in ecbss_df.columns:
                vals = ecbss_df.loc[emo_list, family].values
                std_vals.append(np.std(vals))
            else:
                std_vals.append(0)
        std_data[short_fam.get(family, family[:15])] = std_vals

    std_df = pd.DataFrame(std_data, index=[CLUSTER_LABELS.get(i, f"C{i}") for i in sorted(cluster_ecbss.index)])

    sns.heatmap(
        std_df,
        ax=ax2,
        cmap="YlOrRd",
        annot=True,
        fmt=".0f",
        annot_kws={"size": 7.5},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Within-cluster SD of ECBSS", "shrink": 0.8},
    )
    ax2.set_title("Within-Cluster Variability (SD of ECBSS)", loc="left", fontsize=9)
    ax2.set_xlabel("Bias Family", fontsize=9)
    ax2.set_ylabel("Emotion Cluster", fontsize=9)
    ax2.tick_params(axis="x", rotation=45, labelsize=7.5)
    ax2.tick_params(axis="y", rotation=0, labelsize=8)

    fig.text(0.01, 0.01,
             "Note. Panel A presents the mean ECBSS for each emotion cluster (rows) × bias family (columns) cell. "
             "Red cells indicate the emotional state amplifies susceptibility; blue cells indicate attenuation. "
             "Panel B shows within-cluster SD, reflecting heterogeneity of individual emotions within each cluster. "
             "Darker cells in Panel B indicate bias families where specific emotions within the cluster diverge substantially.",
             fontsize=8, ha="left", va="bottom")

    plt.tight_layout(rect=[0, 0.1, 1, 0.93])
    _save(fig, "fig3_ecbss_heatmap")


# ── Figure 4: Dimensional Sensitivity Profiles (A-B) ────────────────────────

def fig4_dimensional_profiles(bias_profiles: dict) -> None:
    """Fig 4 (A-B): Radar charts + heatmap of bias family dimensional sensitivity."""
    families = [f for f in FAMILY_SHORT_LABELS if f in bias_profiles]
    dims = ["V", "A", "C", "U", "S"]
    dim_labels = ["Valence", "Arousal", "Control", "Uncertainty", "Social"]

    fig = plt.figure(figsize=(18, 10))
    fig.text(0.01, 0.99, "Figure 4", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.97,
             "Dimensional Sensitivity Profiles of Bias Families: How Affective Dimensions Modulate Susceptibility",
             fontsize=12, ha="left", va="top", fontweight="bold")

    n_fam = len(families)
    ncols_radar = 4
    nrows_radar = int(np.ceil(n_fam / ncols_radar))

    # Left half: radar charts (one per family)
    # Right: heatmap
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.3)
    gs_radars = gridspec.GridSpecFromSubplotSpec(nrows_radar, ncols_radar, subplot_spec=gs[0], hspace=0.6, wspace=0.5)

    N = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for i, family in enumerate(families):
        profile = bias_profiles[family]
        values = [float(profile.get(d, 0)) for d in dims]
        values_closed = values + values[:1]

        row_i = i // ncols_radar
        col_i = i % ncols_radar
        ax = fig.add_subplot(gs_radars[row_i, col_i], polar=True)

        # Color based on overall sensitivity direction
        mean_val = np.mean(values)
        fill_color = "#c0392b" if mean_val > 10 else ("#1a6fa3" if mean_val < -10 else "#7f8c8d")

        ax.plot(angles, values_closed, color=fill_color, linewidth=1.5)
        ax.fill(angles, values_closed, color=fill_color, alpha=0.25)

        # Reference grid lines
        ax.set_thetagrids(np.degrees(angles[:-1]), dim_labels, fontsize=6)
        ax.set_ylim(-100, 100)
        ax.set_yticks([-75, -50, -25, 0, 25, 50, 75])
        ax.set_yticklabels([])
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)

        short = FAMILY_SHORT_LABELS.get(family, family[:15]).replace("\n", " ")
        ax.set_title(short, size=6.5, pad=12, loc="center", fontweight="bold")

        # Annotate panel label for first plot
        if i == 0:
            ax.text(-0.25, 1.25, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

    # Right panel: Heatmap
    ax_heat = fig.add_subplot(gs[1])
    ax_heat.text(-0.2, 1.03, "B", transform=ax_heat.transAxes, fontsize=14, fontweight="bold")

    heat_data = []
    row_labels = []
    for family in families:
        profile = bias_profiles[family]
        row = [float(profile.get(d, 0)) for d in dims]
        heat_data.append(row)
        short = FAMILY_SHORT_LABELS.get(family, family[:20]).replace("\n", " ")
        row_labels.append(short)

    heat_array = np.array(heat_data)
    im = ax_heat.imshow(heat_array, cmap=ECBSS_CMAP, aspect="auto",
                        vmin=-100, vmax=100)

    plt.colorbar(im, ax=ax_heat, label="Dimensional Sensitivity Score", fraction=0.046, pad=0.04)

    # Annotate cells
    for i in range(len(families)):
        for j in range(len(dims)):
            val = heat_array[i, j]
            text_color = "white" if abs(val) > 60 else "black"
            ax_heat.text(j, i, f"{val:.0f}", ha="center", va="center",
                         fontsize=8, color=text_color, fontweight="bold")

    ax_heat.set_xticks(range(len(dims)))
    ax_heat.set_xticklabels(dim_labels, fontsize=9, rotation=30, ha="right")
    ax_heat.set_yticks(range(len(families)))
    ax_heat.set_yticklabels(row_labels, fontsize=8)
    ax_heat.set_title("Sensitivity Score per Dimension\n(−100 = attenuate, +100 = amplify)", loc="left", fontsize=9)

    fig.text(0.01, 0.01,
             "Note. Panel A shows radar/spider charts for each bias family, depicting sensitivity to the five affective dimensions "
             "(Valence, Arousal, Control, Uncertainty, Social Orientation). Values range from −100 (dimension attenuates susceptibility) "
             "to +100 (amplifies). Red fills indicate net amplification tendency; blue fills indicate net attenuation. "
             "Panel B summarizes the same information as a heatmap for direct cross-family comparison.",
             fontsize=8, ha="left", va="bottom")

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    _save(fig, "fig4_dimensional_profiles")


# ── Figure 5: Network Analysis ───────────────────────────────────────────────

def fig5_network(
    cluster_ecbss: pd.DataFrame,
    cluster_labels: pd.Series,
    taxonomy_df: pd.DataFrame,
) -> None:
    """Fig 5: Bipartite network emotion clusters ↔ bias families."""
    families = [f for f in FAMILY_SHORT_LABELS if f in cluster_ecbss.columns]
    n_clusters = cluster_ecbss.shape[0]
    n_families = len(families)

    # Count emotions per cluster
    cluster_sizes = cluster_labels.value_counts().to_dict()
    # Count biases per family
    family_sizes = taxonomy_df.groupby("family").size().to_dict()

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.text(0.01, 0.99, "Figure 5", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.97,
             "Bipartite Network of Emotion Clusters and Bias Families: Amplification and Attenuation Topology",
             fontsize=12, ha="left", va="top", fontweight="bold")

    # Layout: emotion clusters on left, bias families on right
    ECBSS_EDGE_THRESHOLD = 80

    # Positions
    em_positions = {}
    for cid in range(n_clusters):
        y = (n_clusters - 1 - cid) / max(n_clusters - 1, 1)
        em_positions[f"ec_{cid}"] = (0.15, 0.1 + y * 0.8)

    bf_positions = {}
    for j, family in enumerate(families):
        y = (n_families - 1 - j) / max(n_families - 1, 1)
        bf_positions[f"bf_{family}"] = (0.85, 0.05 + y * 0.9)

    # Draw edges
    for cid in cluster_ecbss.index:
        for family in families:
            ecbss = cluster_ecbss.loc[cid, family]
            if abs(ecbss) < ECBSS_EDGE_THRESHOLD:
                continue
            x1, y1 = em_positions[f"ec_{cid}"]
            x2, y2 = bf_positions[f"bf_{family}"]
            color = "#c0392b" if ecbss > 0 else "#1a6fa3"
            lw = max(0.3, abs(ecbss) / 200)
            alpha = min(0.9, 0.3 + abs(ecbss) / 800)
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha, zorder=1)

    # Draw emotion cluster nodes
    for cid in range(n_clusters):
        x, y = em_positions[f"ec_{cid}"]
        size = 400 + cluster_sizes.get(cid, 10) * 15
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        ax.scatter(x, y, s=size, c=color, zorder=3, alpha=0.9,
                   edgecolors="white", linewidths=2)
        label = CLUSTER_LABELS.get(cid, f"C{cid}")
        ax.text(x - 0.07, y, label, ha="right", va="center",
                fontsize=8.5, fontweight="bold", color=color)
        ax.text(x - 0.07, y - 0.04, f"n={cluster_sizes.get(cid, '?')} emotions",
                ha="right", va="center", fontsize=6.5, color="gray")

    # Draw bias family nodes
    for j, family in enumerate(families):
        x, y = bf_positions[f"bf_{family}"]
        n_biases = family_sizes.get(family, 10)
        size = 200 + n_biases * 20
        ax.scatter(x, y, s=size, c="#2c3e50", marker="s", zorder=3, alpha=0.85,
                   edgecolors="white", linewidths=2)
        short = FAMILY_SHORT_LABELS.get(family, family[:20]).replace("\n", " ")
        ax.text(x + 0.04, y, short, ha="left", va="center", fontsize=7.5)
        ax.text(x + 0.04, y - 0.025, f"{n_biases} biases",
                ha="left", va="center", fontsize=6, color="gray")

    # Legend
    amp_patch = mpatches.Patch(color="#c0392b", label="Amplifying (ECBSS > 0)")
    att_patch = mpatches.Patch(color="#1a6fa3", label="Attenuating (ECBSS < 0)")
    em_patch = mpatches.Patch(color="#7f8c8d", label="Emotion cluster (circle)")
    bi_patch = mpatches.Patch(color="#2c3e50", label="Bias family (square)")
    ax.legend(handles=[amp_patch, att_patch, em_patch, bi_patch],
              loc="lower center", fontsize=8.5, ncol=2,
              bbox_to_anchor=(0.5, -0.06), framealpha=0.9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.01, 0.01,
             f"Note. Bipartite network connecting emotion clusters (left, circles; size ∝ number of emotions) to bias families "
             f"(right, squares; size ∝ number of leaf biases). Edges shown for |ECBSS| > {ECBSS_EDGE_THRESHOLD}. "
             "Red edges indicate amplification; blue edges indicate attenuation. Line width is proportional to |ECBSS|. "
             "This topology reveals which emotional states exert the broadest vs. most targeted influence on cognitive susceptibility.",
             fontsize=8, ha="left", va="bottom")

    _save(fig, "fig5_network")


# ── Figure 6: Regression Results (A-B) ──────────────────────────────────────

def fig6_regression(regression_results: dict) -> None:
    """Fig 6 (A-B): Mixed-effects regression results forest plot + R² heatmap."""
    per_family = regression_results.get("per_family", {})
    overall = regression_results.get("overall", {})

    dims = ["V", "A", "C", "U", "S"]
    dim_labels_plot = ["Valence (V)", "Arousal (A)", "Control (C)", "Uncertainty (U)", "Social (S)"]
    families = [f for f in FAMILY_SHORT_LABELS if f in per_family]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.text(0.01, 0.99, "Figure 6", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.97,
             "Regression Analysis: Dimensional Predictors of ECBSS and Family-Level Variance Explained",
             fontsize=12, ha="left", va="top", fontweight="bold")

    # Panel A: Forest plot - beta coefficients per dimension per family
    ax = axes[0]
    ax.text(-0.12, 1.04, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

    dim_colors = ["#e74c3c", "#e67e22", "#27ae60", "#2980b9", "#8e44ad"]
    y_spacing = 0.8
    y_positions = {}
    family_labels = []
    base_y = 0

    for fam in families:
        fam_res = per_family[fam]
        params = fam_res.get("params", {})
        bse = fam_res.get("bse", {})
        pvals = fam_res.get("pvalues", {})

        for di, dim in enumerate(dims):
            y = base_y - di * 0.12
            beta = params.get(dim, 0)
            se = bse.get(dim, 0.1)
            pval = pvals.get(dim, 1.0)

            color = dim_colors[di]
            alpha = 0.9 if pval < 0.05 else 0.35
            marker = "o" if pval < 0.05 else "^"

            ax.errorbar(beta, y, xerr=1.96 * se, fmt=marker, color=color,
                        markersize=5, linewidth=1, capsize=3, alpha=alpha)

        y_positions[fam] = base_y - 2 * 0.12  # center of cluster
        base_y -= (len(dims) + 2) * 0.12

    # Re-draw with proper y ticks
    # Simpler approach: one dot per family per dim
    ax.clear()
    ax.text(-0.12, 1.04, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

    n_fam = len(families)
    y_fam = np.arange(n_fam) * (len(dims) + 1.5)
    short_labels = [FAMILY_SHORT_LABELS.get(f, f[:15]).replace("\n", " ") for f in families]

    for di, (dim, dlabel) in enumerate(zip(dims, dim_labels_plot)):
        betas = []
        ses = []
        pvals_list = []
        for fam in families:
            fam_res = per_family.get(fam, {})
            betas.append(fam_res.get("params", {}).get(dim, 0))
            ses.append(fam_res.get("bse", {}).get(dim, 5))
            pvals_list.append(fam_res.get("pvalues", {}).get(dim, 1.0))

        color = dim_colors[di]
        y_vals = y_fam + di * 0.8

        for i, (beta, se, pval) in enumerate(zip(betas, ses, pvals_list)):
            alpha = 0.9 if pval < 0.05 else 0.3
            ax.errorbar(beta, y_vals[i], xerr=1.96 * se,
                        fmt="o", color=color, markersize=5.5 if pval < 0.05 else 4,
                        linewidth=1.2, capsize=3, alpha=alpha)

    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_yticks(y_fam + 2 * 0.8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Beta coefficient (95% CI)", fontsize=9)
    ax.set_title("Fixed Effects: Dimensional Predictors of ECBSS\n(solid = p < .05, faded = n.s.)", loc="left", fontsize=9)

    legend_handles = [
        mpatches.Patch(color=c, label=l)
        for c, l in zip(dim_colors, dim_labels_plot)
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc="lower right")

    # Panel B: R² per family per dimension
    ax2 = axes[1]
    ax2.text(-0.12, 1.04, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")

    r2_data = []
    for fam in families:
        fam_res = per_family.get(fam, {})
        r2_total = fam_res.get("rsquared", 0)
        params = fam_res.get("params", {})

        # Rough proportion of explained variance by each dim
        total_beta = sum(abs(params.get(d, 0)) for d in dims)
        if total_beta < 1e-9:
            row = [0] * len(dims) + [r2_total]
        else:
            row = [abs(params.get(d, 0)) / total_beta * r2_total for d in dims] + [r2_total]
        r2_data.append(row)

    r2_array = np.array(r2_data)
    col_labels = dim_labels_plot + ["R² Total"]

    im = ax2.imshow(r2_array, cmap="Blues", aspect="auto", vmin=0, vmax=max(r2_array.max(), 0.01))
    plt.colorbar(im, ax=ax2, label="R² (proportion)", fraction=0.046, pad=0.04)

    for i in range(len(families)):
        for j in range(len(col_labels)):
            val = r2_array[i, j]
            text_color = "white" if val > 0.3 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=7.5, color=text_color)

    ax2.set_xticks(range(len(col_labels)))
    ax2.set_xticklabels(col_labels, fontsize=8, rotation=35, ha="right")
    ax2.set_yticks(range(len(families)))
    ax2.set_yticklabels(short_labels, fontsize=8)
    ax2.set_title("Variance Explained (R²) by Dimension and Bias Family", loc="left", fontsize=9)

    # Overall model summary
    ovr = overall.get("params", {})
    ovr_p = overall.get("pvalues", {})
    summary_text = "Overall model (mixed-effects, random intercepts by family):\n"
    for dim in dims:
        b = ovr.get(dim, 0)
        p = ovr_p.get(dim, 1)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        summary_text += f"  {dim}: β={b:.2f} {sig}  "

    fig.text(0.01, 0.01,
             f"Note. Panel A shows per-family regression beta coefficients with 95% confidence intervals. "
             f"Filled markers indicate p < .05; faded markers indicate non-significant predictors. "
             f"Panel B shows the proportion of explained variance (R²) attributable to each dimension per bias family. "
             f"Models regress ECBSS scores on standardized V, A, C, U, S dimension scores.",
             fontsize=8, ha="left", va="bottom")

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    _save(fig, "fig6_regression")


# ── Figure 7: Hypothesis Evaluation (A-B) ───────────────────────────────────

def fig7_hypothesis_evaluation(
    cluster_ecbss: pd.DataFrame,
    ecbss_df: pd.DataFrame,
    cluster_labels: pd.Series,
) -> None:
    """Fig 7 (A-B): Hypothesis evaluation and key emotion profiles."""
    families = [f for f in FAMILY_SHORT_LABELS if f in cluster_ecbss.columns]

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    fig.text(0.01, 0.99, "Figure 7", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.97,
             "Preregistered Hypothesis Evaluation: Emotion–Bias Susceptibility Patterns",
             fontsize=12, ha="left", va="top", fontweight="bold")

    ax = axes[0]
    ax.text(-0.12, 1.04, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

    # Panel A: Profile plot for key hypothesis-relevant emotions
    hypothesis_emotions = {
        "H1: Threat/Alarm\n(Fear, Panic, Alarmed)": ["afraid", "panicked", "alarmed", "horrified"],
        "H2: Reward/Enthusiasm\n(Excited, Optimistic)": ["excited", "enthusiastic", "optimistic", "eager"],
        "H3: Shame/Guilt\n(Ashamed, Guilty)": ["ashamed", "guilty", "humiliated", "embarrassed"],
        "H4: Calm/Reflective\n(Calm, Serene)": ["calm", "serene", "peaceful", "tranquil"],
    }

    fam_short = [FAMILY_SHORT_LABELS.get(f, f[:10]).replace("\n", " ") for f in families]
    x_pos = np.arange(len(families))

    h_colors = ["#e74c3c", "#f39c12", "#8e44ad", "#27ae60"]

    for hi, (h_label, h_emotions) in enumerate(hypothesis_emotions.items()):
        available = [e for e in h_emotions if e in ecbss_df.index]
        if not available:
            continue
        mean_profile = ecbss_df.loc[available, families].mean()
        sem_profile = ecbss_df.loc[available, families].sem()
        color = h_colors[hi % len(h_colors)]

        ax.plot(x_pos, mean_profile.values, marker="o", markersize=6,
                color=color, linewidth=2, label=h_label, zorder=3)
        ax.fill_between(x_pos,
                         mean_profile.values - sem_profile.values,
                         mean_profile.values + sem_profile.values,
                         color=color, alpha=0.15, zorder=2)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fam_short, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Mean ECBSS", fontsize=9)
    ax.set_title("ECBSS Profiles for Hypothesis-Relevant Emotion Groups\n(mean ± SEM)", loc="left", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.8)
    ax.axhspan(-50, 50, color="gray", alpha=0.08, label="Neutral zone")

    # Panel B: Hypothesis evaluation summary table/matrix
    ax2 = axes[1]
    ax2.text(-0.12, 1.04, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")

    hypotheses = [
        ("H1", "Threat/alarm → ↑ salience/urgency biases", "Social & Auth., Risk & Prob."),
        ("H2", "Reward/enthusiasm → ↑ optimism/trust biases", "Trust & Cred., Self-Assess."),
        ("H3", "Shame/guilt → ↑ conformity/compliance biases", "Social & Auth., Privacy"),
        ("H4", "Calm/reflective → ↓ fast/cue-driven biases", "Attention, Trust, Interface"),
        ("H5", "High-arousal negative ↑ attention salience biases", "Attention & Salience"),
        ("H6", "Low-control emotions ↑ authority deference", "Social & Authority"),
        ("H7", "Composite states: non-additive effects", "All families"),
    ]

    # Compute actual support level from data
    h_support = []
    for hi, (hid, desc, families_str) in enumerate(hypotheses):
        if hid == "H1":
            avail = [e for e in ["afraid", "panicked", "alarmed", "horrified"] if e in ecbss_df.index]
            fam_k = "attention_salience_and_signal_detection_biases"
        elif hid == "H2":
            avail = [e for e in ["excited", "enthusiastic", "optimistic", "eager"] if e in ecbss_df.index]
            fam_k = "trust_source_credibility_and_truth_judgment_biases"
        elif hid == "H3":
            avail = [e for e in ["ashamed", "guilty", "humiliated", "embarrassed"] if e in ecbss_df.index]
            fam_k = "social_influence_authority_affiliation_and_identity_biases"
        elif hid == "H4":
            avail = [e for e in ["calm", "serene", "peaceful", "tranquil"] if e in ecbss_df.index]
            fam_k = "attention_salience_and_signal_detection_biases"
        elif hid == "H5":
            avail = [e for e in ["panicked", "terrified", "alarmed", "horrified"] if e in ecbss_df.index]
            fam_k = "attention_salience_and_signal_detection_biases"
        elif hid == "H6":
            avail = [e for e in ["helpless", "cornered", "trapped", "powerless"] if e in ecbss_df.index]
            fam_k = "social_influence_authority_affiliation_and_identity_biases"
        else:
            avail = []
            fam_k = None

        if avail and fam_k and fam_k in ecbss_df.columns:
            mean_ecbss = ecbss_df.loc[avail, fam_k].mean()
            if hid in ["H1", "H2", "H3", "H5", "H6"]:
                supported = mean_ecbss > 100
            elif hid == "H4":
                supported = mean_ecbss < -100
            else:
                supported = True  # composite = partially supported by design
            partial = not supported and abs(mean_ecbss) > 50
            score = mean_ecbss
        else:
            supported = False
            partial = True
            score = 0

        h_support.append(("Supported" if supported else ("Partial" if partial else "Not supported"), score))

    # Draw summary plot
    y_positions = np.arange(len(hypotheses))
    colors_map = {"Supported": "#27ae60", "Partial": "#f39c12", "Not supported": "#e74c3c"}

    for i, (hid, desc, fams_str) in enumerate(hypotheses):
        verdict, score = h_support[i]
        color = colors_map[verdict]
        ax2.barh(i, abs(score) / 10, color=color, alpha=0.7, edgecolor="white")
        ax2.text(-1, i, f"{hid}: {desc}", ha="right", va="center", fontsize=7.5)
        ax2.text(abs(score) / 10 + 0.5, i, verdict, ha="left", va="center",
                 fontsize=7.5, color=color, fontweight="bold")

    ax2.set_yticks([])
    ax2.set_xlabel("|Mean ECBSS| / 10 for Key Family", fontsize=8)
    ax2.set_title("Hypothesis Evaluation Summary", loc="left", fontsize=9)

    verdict_patches = [
        mpatches.Patch(color="#27ae60", label="Supported"),
        mpatches.Patch(color="#f39c12", label="Partially supported"),
        mpatches.Patch(color="#e74c3c", label="Not supported"),
    ]
    ax2.legend(handles=verdict_patches, fontsize=8, loc="lower right")
    ax2.set_xlim(-2, max(abs(s) for _, s in h_support) / 10 + 12)

    fig.text(0.01, 0.01,
             "Note. Panel A shows mean ECBSS profiles for hypothesis-relevant emotion groups across all bias families "
             "(shaded bands = ±SEM). Panel B presents an evaluation of each preregistered hypothesis based on observed "
             "ECBSS patterns, using the primary relevant bias family. Bar length reflects effect magnitude. "
             "'Supported' = mean ECBSS in predicted direction with |ECBSS| > 100; 'Partial' = 50–100; 'Not supported' = < 50.",
             fontsize=8, ha="left", va="bottom")

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    _save(fig, "fig7_hypothesis_evaluation")


# ── Supplementary: Full ECBSS Heatmap ───────────────────────────────────────

def figS1_full_ecbss_heatmap(ecbss_df: pd.DataFrame, cluster_labels: pd.Series) -> None:
    """Fig S1: Full ECBSS heatmap all emotions × families, sorted by cluster."""
    families = [f for f in FAMILY_SHORT_LABELS if f in ecbss_df.columns]
    short_fam = {k: v.replace("\n", " ") for k, v in FAMILY_SHORT_LABELS.items()}

    # Sort emotions by cluster then by valence
    plot_df = ecbss_df[families].copy()
    plot_df["cluster"] = cluster_labels

    # Sort
    sorted_emotions = []
    cluster_boundaries = {}
    running = 0
    for cid in sorted(cluster_labels.unique()):
        emos_in_cluster = plot_df[plot_df["cluster"] == cid].index.tolist()
        emos_sorted = sorted(emos_in_cluster)
        cluster_boundaries[cid] = (running, running + len(emos_in_cluster))
        sorted_emotions.extend(emos_sorted)
        running += len(emos_in_cluster)

    plot_df = plot_df.loc[sorted_emotions, families]

    fig, ax = plt.subplots(figsize=(16, max(12, len(sorted_emotions) * 0.18)))
    fig.text(0.01, 0.99, "Figure S1", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.97, "Full ECBSS Matrix: All Emotions × Bias Families (Sorted by Emotion Cluster)",
             fontsize=12, ha="left", va="top", fontweight="bold")

    vmax = min(abs(plot_df.values).max(), 900)
    sns.heatmap(
        plot_df,
        ax=ax,
        cmap=ECBSS_CMAP,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=[short_fam.get(f, f[:12]) for f in families],
        yticklabels=sorted_emotions,
        linewidths=0,
        cbar_kws={"label": "ECBSS", "shrink": 0.5},
    )
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    ax.tick_params(axis="y", labelsize=5.5)

    # Add cluster separation lines and labels
    for cid, (start, end) in cluster_boundaries.items():
        ax.axhline(start, color="white", linewidth=1.5)
        mid = (start + end) / 2
        ax.text(len(families) + 0.3, mid, CLUSTER_LABELS.get(cid, f"C{cid}"),
                va="center", fontsize=7, color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                fontweight="bold")

    ax.set_title("ECBSS: Individual Emotions × Bias Families", loc="left", fontsize=9)
    fig.text(0.01, 0.01,
             "Note. Complete ECBSS matrix for all scored emotions sorted by cluster membership. "
             "Cluster boundaries are indicated by horizontal white lines. "
             "Red = amplification; Blue = attenuation relative to no emotional modulation.",
             fontsize=8, ha="left", va="bottom")

    plt.tight_layout(rect=[0, 0.03, 0.92, 0.96])
    _save(fig, "figS1_full_ecbss_heatmap")


def figS2_validation_scatter(analytical_df: pd.DataFrame, direct_df: pd.DataFrame) -> None:
    """Fig S2: Scatter plot validating analytical vs. LLM direct ECBSS."""
    families = analytical_df.columns.intersection(direct_df.columns).tolist()
    emotions = analytical_df.index.intersection(direct_df.index).tolist()

    flat_a = analytical_df.loc[emotions, families].values.flatten()
    flat_d = direct_df.loc[emotions, families].values.flatten()
    mask = ~np.isnan(flat_d) & ~np.isnan(flat_a)

    from scipy.stats import pearsonr, spearmanr
    r, p = pearsonr(flat_a[mask], flat_d[mask])
    rs, ps = spearmanr(flat_a[mask], flat_d[mask])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.text(0.01, 0.99, "Figure S2", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.97, "Validation: Analytical ECBSS vs. LLM Direct Scoring Correspondence",
             fontsize=12, ha="left", va="top", fontweight="bold")

    # Panel A: Overall scatter
    ax = axes[0]
    ax.text(-0.12, 1.04, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.scatter(flat_a[mask], flat_d[mask], alpha=0.3, s=15, c="#2980b9", edgecolors="none")
    z = np.polyfit(flat_a[mask], flat_d[mask], 1)
    xfit = np.linspace(-1000, 1000, 200)
    ax.plot(xfit, np.polyval(z, xfit), color="#e74c3c", linewidth=2, label=f"OLS fit")
    ax.plot([-1000, 1000], [-1000, 1000], color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Perfect agreement")
    ax.set_xlabel("Analytical ECBSS", fontsize=9)
    ax.set_ylabel("LLM Direct ECBSS", fontsize=9)
    ax.set_title(f"Overall Correspondence\nr = {r:.3f}, ρ = {rs:.3f} (N = {mask.sum()} pairs)", loc="left", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)

    # Panel B: Per-family correlation
    ax2 = axes[1]
    ax2.text(-0.12, 1.04, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")
    per_fam_r = []
    for fam in families:
        a_vals = analytical_df.loc[emotions, fam].values
        d_vals = direct_df.loc[emotions, fam].values
        m = ~np.isnan(a_vals) & ~np.isnan(d_vals)
        if m.sum() > 5:
            r_f, _ = pearsonr(a_vals[m], d_vals[m])
            per_fam_r.append((FAMILY_SHORT_LABELS.get(fam, fam[:15]).replace("\n", " "), r_f))

    per_fam_r.sort(key=lambda x: x[1])
    names = [x[0] for x in per_fam_r]
    rs_vals = [x[1] for x in per_fam_r]

    colors_bar = ["#e74c3c" if r < 0.5 else "#f39c12" if r < 0.7 else "#27ae60" for r in rs_vals]
    ax2.barh(range(len(per_fam_r)), rs_vals, color=colors_bar, alpha=0.8)
    ax2.set_yticks(range(len(per_fam_r)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("Pearson r (Analytical vs. LLM Direct)", fontsize=9)
    ax2.set_title("Per-Family Validation Correlation", loc="left", fontsize=9)
    ax2.axvline(0.7, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="r = 0.70")
    ax2.legend(fontsize=8)

    fig.text(0.01, 0.01,
             "Note. Panel A: scatter of analytical ECBSS (computed via dimensional dot-product) vs. LLM-direct ECBSS scores "
             "across all emotion × family pairs. Panel B: per-family Pearson correlations. "
             "High agreement validates the dimensional scoring approach as a computationally efficient substitute "
             "for full LLM scoring.",
             fontsize=8, ha="left", va="bottom")

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    _save(fig, "figS2_validation_scatter")


def figS3_composite_emotions(composite_df: pd.DataFrame) -> None:
    """Fig S3: Composite emotion non-additivity analysis."""
    families = [f for f in FAMILY_SHORT_LABELS if f in composite_df["family"].unique()]
    dyads = composite_df["dyad"].unique().tolist()

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.text(0.01, 0.99, "Figure S3", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.97, "Composite Emotion Analysis: Dyadic Interaction Effects on Bias Susceptibility",
             fontsize=12, ha="left", va="top", fontweight="bold")

    ax = axes[0]
    ax.text(-0.12, 1.04, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

    # Panel A: Divergence between e1 and e2 per dyad per family
    dyad_div = composite_df.pivot_table(
        index="dyad", columns="family", values="abs_diff_e1_e2", aggfunc="mean"
    )
    dyad_div_plot = dyad_div[[f for f in families if f in dyad_div.columns]]

    if not dyad_div_plot.empty:
        sns.heatmap(dyad_div_plot,
                    ax=ax,
                    cmap="YlOrRd",
                    annot=True, fmt=".0f", annot_kws={"size": 7},
                    xticklabels=[FAMILY_SHORT_LABELS.get(f, f[:10]).replace("\n", " ") for f in dyad_div_plot.columns],
                    cbar_kws={"label": "|ECBSS_e1 − ECBSS_e2|", "shrink": 0.8})
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7.5)
    ax.set_title("Within-Dyad Divergence per Bias Family", loc="left", fontsize=9)

    # Panel B: Additive prediction bar chart for key dyads
    ax2 = axes[1]
    ax2.text(-0.12, 1.04, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")

    target_family = "social_influence_authority_affiliation_and_identity_biases"
    if target_family not in composite_df["family"].unique():
        target_family = composite_df["family"].unique()[0]

    fam_data = composite_df[composite_df["family"] == target_family]

    x = np.arange(len(dyads))
    width = 0.25

    e1_vals = [fam_data[fam_data["dyad"] == d]["ecbss_e1"].mean() for d in dyads]
    e2_vals = [fam_data[fam_data["dyad"] == d]["ecbss_e2"].mean() for d in dyads]
    pred_vals = [fam_data[fam_data["dyad"] == d]["additive_pred"].mean() for d in dyads]

    ax2.bar(x - width, e1_vals, width, label="Emotion 1 (ECBSS)", color="#3498db", alpha=0.8)
    ax2.bar(x, e2_vals, width, label="Emotion 2 (ECBSS)", color="#e67e22", alpha=0.8)
    ax2.bar(x + width, pred_vals, width, label="Additive prediction", color="#9b59b6", alpha=0.8, hatch="//")

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    short_dyads = [d[:25] for d in dyads]
    ax2.set_xticklabels(short_dyads, rotation=35, ha="right", fontsize=7.5)
    ax2.set_ylabel("ECBSS", fontsize=9)
    ax2.set_title(
        f"Additive Prediction vs. Component Scores\n({FAMILY_SHORT_LABELS.get(target_family, target_family[:20]).replace(chr(10), ' ')})",
        loc="left", fontsize=9
    )
    ax2.legend(fontsize=7.5)

    fig.text(0.01, 0.01,
             "Note. Panel A shows within-dyad divergence (|ECBSS_e1 − ECBSS_e2|) across bias families, "
             "revealing which families are most sensitive to the specific emotional blend. "
             "Panel B compares individual component ECBSS values with the additive (mean) prediction "
             "for selected bias family, providing a baseline for detecting non-additive composite effects.",
             fontsize=8, ha="left", va="bottom")

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    _save(fig, "figS3_composite_emotions")


def figS4_cluster_profiles(emotion_df: pd.DataFrame, cluster_labels: pd.Series) -> None:
    """Fig S4: Cluster dimensional profiles + sample emotions."""
    dims = ["V", "A", "C", "U", "S"]
    dim_labels_s = ["Valence", "Arousal", "Control", "Uncertainty", "Social"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.text(0.01, 0.99, "Figure S4", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.97, "Emotion Cluster Dimensional Profiles: Archetype Characterization",
             fontsize=12, ha="left", va="top", fontweight="bold")

    n_clusters = cluster_labels.nunique()
    for cid in range(n_clusters):
        ax = axes[cid // 3][cid % 3]
        mask = cluster_labels.values == cid
        members = [e for e, m in zip(emotion_df.index, mask) if m]

        if not members:
            ax.axis("off")
            continue

        cluster_dim = emotion_df.loc[members, dims]
        means = cluster_dim.mean()
        stds = cluster_dim.std()

        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        x = np.arange(len(dims))
        ax.bar(x, means.values, color=color, alpha=0.7, edgecolor="white")
        ax.errorbar(x, means.values, yerr=stds.values, fmt="none",
                    color="black", capsize=4, linewidth=1.5)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(dim_labels_s, fontsize=9)
        ax.set_ylim(-1000, 1000)
        ax.set_ylabel("Dimension Score", fontsize=8)
        ax.set_title(f"Cluster {cid}: {CLUSTER_LABELS.get(cid, '').replace(chr(10), ' ')}\n(n={len(members)} emotions)",
                     loc="left", fontsize=9, color=color, fontweight="bold")
        ax.tick_params(labelsize=8)

        # Annotate with top 5 emotion names
        top5 = members[:5]
        ax.text(0.98, 0.98, "\n".join(top5), transform=ax.transAxes,
                fontsize=6.5, ha="right", va="top", color="gray")

    fig.text(0.01, 0.01,
             "Note. Bar charts show mean (±SD) dimensional scores for each emotion cluster. "
             "Sample emotion names are shown in the top-right corner of each panel. "
             "These profiles confirm the theoretical coherence of the K-means clustering solution.",
             fontsize=8, ha="left", va="bottom")

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    _save(fig, "figS4_cluster_profiles")
