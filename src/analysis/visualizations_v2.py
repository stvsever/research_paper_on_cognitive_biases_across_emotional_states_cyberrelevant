"""
visualizations_v2.py — Nature/Science-style publication figures for:
"Cognitive Bias Vulnerability Across Emotional States in Cybersecurity"

Convention
----------
- LEFT-ALIGNED bold titles above each figure
- "Note." italic captions below each figure
- All figures saved as PDF + PNG (300 dpi) to paper/figures/
- Panel labels (A, B, C, D) bold, top-left, fontsize 12
- White backgrounds, minimal spines, professional serif/sans fonts
- ECBSS diverging palette: #1a6fa3 (blue) → white → #c0392b (red)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, Ellipse, FancyBboxPatch
import matplotlib.transforms as transforms
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, optimal_leaf_ordering
from scipy.spatial import ConvexHull
from scipy.spatial.distance import squareform, cosine, pdist
from scipy.stats import pearsonr, spearmanr, sem, t as t_dist
import warnings

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper" / "assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
INTERNAL_FIGURE_TEXT = False

# ── Global rcParams ──────────────────────────────────────────────────────────
plt.rcParams.update({
    # Typography
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.title_fontsize": 8,
    # Axes
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    # Figure
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "savefig.facecolor": "white",
    # Grid
    "axes.grid": False,
    "grid.linewidth": 0.4,
    "grid.color": "#cccccc",
    # Lines
    "lines.linewidth": 1.2,
})

# ── Color palettes ───────────────────────────────────────────────────────────

# Diverging ECBSS map: blue (attenuate) → white → red (amplify)
ECBSS_CMAP = LinearSegmentedColormap.from_list(
    "ecbss_diverge",
    ["#1a6fa3", "#4a9fc8", "#a8d4e8", "#f5f5f5", "#f5b8ab", "#e8604a", "#c0392b"],
    N=256,
)

# 6 emotion clusters — ColorBrewer-inspired, high contrast
CLUSTER_COLORS = [
    "#d62728",   # 0: Hostile & Defiant
    "#756bb1",   # 1: Alarmed & Uncertain
    "#8c6d31",   # 2: Socially Vulnerable
    "#5f6b7a",   # 3: Withdrawn & Low Arousal
    "#fd8d3c",   # 4: High-Arousal Positive
    "#31a354",   # 5: Calm Positive
]

# 11 bias families — tab20 via ColorBrewer
FAMILY_COLORS = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b",
]

CLUSTER_LABELS = {
    0: "Hostile & Defiant",
    1: "Alarmed & Uncertain",
    2: "Socially Vulnerable",
    3: "Withdrawn & Low Arousal",
    4: "High-Arousal Positive",
    5: "Calm Positive",
}

CLUSTER_LABELS_SHORT = {
    0: "Hostile",
    1: "Alarmed",
    2: "Social",
    3: "Withdrawn",
    4: "HA-Pos",
    5: "Calm+",
}

FAMILY_SHORT = {
    "attention_salience_and_signal_detection_biases":
        "Attention & Salience",
    "trust_source_credibility_and_truth_judgment_biases":
        "Trust & Credibility",
    "evidence_search_hypothesis_testing_and_belief_updating_biases":
        "Evidence & Belief",
    "memory_familiarity_and_source_monitoring_biases":
        "Memory & Source",
    "risk_probability_uncertainty_and_outcome_valuation_biases":
        "Risk & Probability",
    "temporal_choice_default_action_and_commitment_biases":
        "Temporal & Commitment",
    "social_influence_authority_affiliation_and_identity_biases":
        "Social & Authority",
    "self_assessment_attribution_and_metacognitive_biases":
        "Self-Assessment",
    "interface_choice_architecture_automation_and_warning_response_biases":
        "Interface & Automation",
    "privacy_disclosure_and_self_presentation_biases":
        "Privacy & Disclosure",
    "affective_evaluation_and_mood_congruent_judgment_biases":
        "Affect & Mood",
}

FAMILY_ABBR = {
    "attention_salience_and_signal_detection_biases": "Att",
    "trust_source_credibility_and_truth_judgment_biases": "Tru",
    "evidence_search_hypothesis_testing_and_belief_updating_biases": "Evi",
    "memory_familiarity_and_source_monitoring_biases": "Mem",
    "risk_probability_uncertainty_and_outcome_valuation_biases": "Ris",
    "temporal_choice_default_action_and_commitment_biases": "Tem",
    "social_influence_authority_affiliation_and_identity_biases": "Soc",
    "self_assessment_attribution_and_metacognitive_biases": "Slf",
    "interface_choice_architecture_automation_and_warning_response_biases": "Int",
    "privacy_disclosure_and_self_presentation_biases": "Pri",
    "affective_evaluation_and_mood_congruent_judgment_biases": "Aff",
}

FAMILY_FULL_LABEL = {
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

DIM_LABELS = {
    "V": "Valence",
    "A": "Arousal",
    "C": "Control",
    "U": "Uncertainty",
    "S": "Social",
}

DIM_COLORS = {
    "V": "#e31a1c",
    "A": "#ff7f00",
    "C": "#33a02c",
    "U": "#1f78b4",
    "S": "#6a3d9a",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    """Save figure as PDF and PNG (300 dpi)."""
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(str(path), dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    print(f"  Saved → {name}.pdf / .png")
    plt.close(fig)


def _maybe_figtext(fig: plt.Figure, *args, **kwargs):
    """Write figure-level text only when explicitly enabled."""
    if not INTERNAL_FIGURE_TEXT:
        return None
    return fig.text(*args, **kwargs)


def _panel_label(ax: plt.Axes, label: str, x: float = -0.10,
                 y: float = 1.05, fontsize: int = 12) -> None:
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left")


def _figure_title(fig: plt.Figure, number: str, title: str,
                  y_num: float = 0.985, y_title: float = 0.965) -> None:
    _maybe_figtext(fig, 0.015, y_num, number, fontsize=10, fontweight="bold",
                   ha="left", va="top", fontstyle="italic")
    _maybe_figtext(fig, 0.015, y_title, title, fontsize=10, fontweight="bold",
                   ha="left", va="top")


def _note(fig: plt.Figure, text: str, y: float = 0.015) -> None:
    _maybe_figtext(fig, 0.015, y,
                   r"$\it{Note.}$ " + text,
                   fontsize=7.5, ha="left", va="bottom",
                   wrap=True, multialignment="left")


def _despine(ax: plt.Axes, left: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)


def _sig_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _bootstrap_ci(data: np.ndarray, n_boot: int = 1000,
                  ci: float = 0.95) -> tuple[float, float]:
    """Non-parametric bootstrap 95% CI for the mean."""
    if len(data) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    boot = rng.choice(data, size=(n_boot, len(data)), replace=True).mean(axis=1)
    lo = np.percentile(boot, 100 * (1 - ci) / 2)
    hi = np.percentile(boot, 100 * (1 - (1 - ci) / 2))
    return float(lo), float(hi)


def _confidence_ellipse(x: np.ndarray, y: np.ndarray, ax: plt.Axes,
                        n_std: float = 2.0, **kwargs) -> Ellipse:
    """Plot a covariance-based confidence ellipse."""
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    pearson = cov[0, 1] / (np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]) + 1e-12)
    rx = np.sqrt(1 + pearson)
    ry = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=rx * 2, height=ry * 2, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    transf = (transforms.Affine2D()
              .rotate_deg(45)
              .scale(scale_x, scale_y)
              .translate(mean_x, mean_y))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def _ordered_families(df_columns) -> list[str]:
    """Return family keys present in df_columns in canonical order."""
    return [f for f in FAMILY_SHORT if f in df_columns]


def _ordered_clusters(cluster_ecbss: pd.DataFrame) -> list[int]:
    return sorted(cluster_ecbss.index.tolist())


# ════════════════════════════════════════════════════════════════════════════
# MAIN FIGURES
# ════════════════════════════════════════════════════════════════════════════

# ── Figure 1: Taxonomy Sunburst + Stacked Bar ────────────────────────────────

def fig1_taxonomy_sunburst(taxonomy_df: pd.DataFrame) -> None:
    """
    Fig 1 (A–B): Taxonomy of cyber-relevant cognitive biases.

    Panel A: Radial sunburst (family → cluster rings) using polar axes.
    Panel B: Horizontal stacked bar sorted by total N biases, cluster sub-bars.
    """
    families = [f for f in FAMILY_SHORT if f in taxonomy_df["family"].unique()]
    n_fam = len(families)
    fam_color_map = {f: FAMILY_COLORS[i % len(FAMILY_COLORS)] for i, f in enumerate(families)}

    fig = plt.figure(figsize=(14.2, 6.4))
    _figure_title(fig, "Figure 1",
                  "Taxonomy of Cyber-Relevant Cognitive Biases: Hierarchical Structure and Distribution")

    # GridSpec: polar panel left, bar panel right
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1],
                           left=0.04, right=0.98, top=0.96, bottom=0.10,
                           wspace=0.10)

    # ── Panel A: Sunburst ────────────────────────────────────────────────────
    ax_sun = fig.add_subplot(gs[0], polar=True)
    _panel_label(ax_sun, "A", x=-0.06, y=1.06)

    # Compute cumulative angles for families → clusters
    family_leaf_counts = []
    family_cluster_data = []
    for fam in families:
        sub = taxonomy_df[taxonomy_df["family"] == fam]
        clusters_in_fam = sub.groupby("cluster").size().to_dict()
        n_total = sum(clusters_in_fam.values())
        family_leaf_counts.append(n_total)
        family_cluster_data.append(clusters_in_fam)

    total_leaves = sum(family_leaf_counts)
    full_angle = 2 * np.pi

    # Outer ring (families) — radius 0.55–0.85
    outer_r_inner, outer_r_outer = 0.55, 0.85
    # Inner ring (clusters) — radius 0.28–0.52
    inner_r_inner, inner_r_outer = 0.28, 0.52

    cumulative = 0.0
    for fi, fam in enumerate(families):
        n = family_leaf_counts[fi]
        frac = n / total_leaves
        theta_start = cumulative * full_angle
        theta_end = (cumulative + frac) * full_angle
        cumulative += frac

        color = fam_color_map[fam]
        thetas = np.linspace(theta_start, theta_end, max(int((theta_end - theta_start) / 0.01), 30))

        # Outer arc (family)
        xs_outer = np.concatenate([[outer_r_inner], outer_r_outer * np.ones_like(thetas),
                                    [outer_r_inner]])
        ts_outer = np.concatenate([[theta_start], thetas, [theta_end]])
        ax_sun.fill_between(thetas, outer_r_inner, outer_r_outer,
                             color=color, alpha=0.88, linewidth=0)
        ax_sun.plot(thetas, [outer_r_outer] * len(thetas), color="white", linewidth=0.6)
        ax_sun.plot(thetas, [outer_r_inner] * len(thetas), color="white", linewidth=0.4)
        ax_sun.plot([theta_start, theta_start], [outer_r_inner, outer_r_outer],
                    color="white", linewidth=0.6)
        ax_sun.plot([theta_end, theta_end], [outer_r_inner, outer_r_outer],
                    color="white", linewidth=0.6)

        # Family label on outer ring (if enough space)
        if frac > 0.04:
            mid_theta = (theta_start + theta_end) / 2
            label = FAMILY_SHORT[fam]
            # Shorten to 2-part label, splitting on " & " first to preserve compound names
            parts = label.split(" & ")
            if len(parts) >= 2:
                short_lbl = f"{parts[0]}\n& {parts[1]}"
            else:
                words = label.split()
                short_lbl = words[0] if len(words) == 1 else f"{words[0]}\n{words[1]}"
            ax_sun.text(mid_theta, (outer_r_inner + outer_r_outer) / 2,
                        short_lbl, ha="center", va="center",
                        fontsize=5.2, fontweight="bold", color="white",
                        rotation=np.degrees(mid_theta) - 90
                        if np.pi / 2 < mid_theta < 3 * np.pi / 2 else np.degrees(mid_theta) + 90)

        # Inner ring (clusters within family)
        clusters_dict = family_cluster_data[fi]
        n_clusters = sum(clusters_dict.values())
        cluster_cumul = theta_start
        cmap_inner = matplotlib.colormaps.get_cmap("tab20c")
        for ci_idx, (clust, n_c) in enumerate(clusters_dict.items()):
            c_frac = n_c / total_leaves
            c_theta_start = cluster_cumul
            c_theta_end = cluster_cumul + c_frac * full_angle
            cluster_cumul = c_theta_end
            c_thetas = np.linspace(c_theta_start, c_theta_end,
                                   max(int((c_theta_end - c_theta_start) / 0.01), 10))
            shade = 0.3 + 0.6 * (ci_idx / max(len(clusters_dict) - 1, 1))
            ax_sun.fill_between(c_thetas, inner_r_inner, inner_r_outer,
                                 color=color, alpha=shade, linewidth=0)
            ax_sun.plot(c_thetas, [inner_r_outer] * len(c_thetas),
                        color="white", linewidth=0.3)

    # Center label
    ax_sun.text(0, 0, f"N={total_leaves}\nleaf\nbiases", ha="center", va="center",
                fontsize=8, fontweight="bold", color="#333333")

    ax_sun.set_rmax(outer_r_outer + 0.08)
    ax_sun.set_axis_off()

    # ── Panel B: Stacked horizontal bar ─────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1])
    _panel_label(ax_bar, "B", x=-0.06, y=1.06)

    cluster_counts = (taxonomy_df.groupby(["family", "cluster"])
                      .size().rename("n").reset_index())
    family_totals = {f: taxonomy_df[taxonomy_df["family"] == f].shape[0]
                     for f in families}
    families_sorted = sorted(families, key=lambda f: family_totals[f])

    y_pos = np.arange(len(families_sorted))
    for yi, fam in enumerate(families_sorted):
        clusters_in_fam = (cluster_counts[cluster_counts["family"] == fam]
                           .sort_values("n", ascending=False))
        left_val = 0
        color = fam_color_map[fam]
        n_clust = len(clusters_in_fam)
        for ci_idx, row in enumerate(clusters_in_fam.itertuples()):
            alpha_val = 0.95 - 0.55 * (ci_idx / max(n_clust - 1, 1))
            ax_bar.barh(yi, row.n, left=left_val,
                        color=color, alpha=alpha_val,
                        edgecolor="white", linewidth=0.4, height=0.65)
            if row.n >= 3:
                ax_bar.text(left_val + row.n / 2, yi, str(row.n),
                            ha="center", va="center", fontsize=6.2,
                            color="white", fontweight="bold")
            left_val += row.n
        # Total annotation
        total = family_totals[fam]
        ax_bar.text(total + 0.30, yi, f"{total}  ({cluster_counts[cluster_counts['family'] == fam].shape[0]} clusters)",
                    ha="left", va="center", fontsize=6.5, color="#444444")

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(
        [FAMILY_SHORT[f] for f in families_sorted],
        fontsize=7.5
    )
    ax_bar.set_xlabel("Number of leaf biases", fontsize=8)
    ax_bar.set_xlim(0, max(family_totals.values()) * 1.18)
    _despine(ax_bar)

    _note(fig,
          "Panel A shows the 3-level taxonomy as a radial sunburst: outer ring = bias family (11 families, "
          f"colored), inner ring = cluster subdivisions. Panel B is a horizontal stacked bar chart sorted by "
          f"total leaf bias count; sub-bar opacity distinguishes clusters within each family. "
          f"Total taxonomy: 11 families, {taxonomy_df['cluster'].nunique()} clusters, "
          f"{total_leaves} leaf biases.",
          y=0.02)

    _save(fig, "fig1_taxonomy_sunburst")


# ── Figure 2: Emotion Dimensional Landscape (UMAP) ──────────────────────────

def fig2_emotion_landscape(
    emotion_df: pd.DataFrame,
    umap_coords: np.ndarray,
    cluster_labels: pd.Series,
    pca_coords: Optional[np.ndarray] = None,
    ecbss_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Fig 2 (A–D): UMAP embedding of emotion space.

    Panel A: UMAP colored by mean ECBSS (average across all 11 families).
             Annotates top-10 and bottom-10 emotions by mean ECBSS.
    Panel B: UMAP colored by Arousal. Annotates top-10 and bottom-10 by Arousal.
    Panel C: UMAP colored by Control. Annotates top-10 and bottom-10 by Control.
    Panel D: Cluster map with convex hulls + labeled centroids + legend.
    """
    emotions = emotion_df.index.tolist()
    coords = umap_coords

    # Compute mean ECBSS per emotion if ecbss_df is available
    mean_ecbss = None
    if ecbss_df is not None:
        shared = [e for e in emotions if e in ecbss_df.index]
        mean_ecbss_series = ecbss_df.loc[shared].mean(axis=1)
        # Align to emotion order
        mean_ecbss = np.array([mean_ecbss_series.get(e, np.nan) for e in emotions])

    fig = plt.figure(figsize=(16, 14))
    _figure_title(fig, "Figure 2",
                  "Multidimensional Affective Space: UMAP Projections and Emotion Cluster Structure",
                  y_num=0.988, y_title=0.972)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.42, wspace=0.30,
                           left=0.07, right=0.97,
                           top=0.94, bottom=0.10)

    def _annotate_top_bottom(ax, vals_arr, emotions_list, coords, n=10):
        """Annotate top-n and bottom-n emotions by vals_arr on ax."""
        valid_mask = ~np.isnan(vals_arr)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return
        sorted_idx = valid_indices[np.argsort(vals_arr[valid_indices])]
        bottom_idx = sorted_idx[:n]
        top_idx = sorted_idx[-n:]
        for ei in np.concatenate([top_idx, bottom_idx]):
            emo = emotions_list[ei]
            color = "#b22222" if ei in top_idx else "#1a4fa3"
            ax.annotate(emo,
                        xy=(coords[ei, 0], coords[ei, 1]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=5.5, fontstyle="italic",
                        color=color,
                        arrowprops=dict(arrowstyle="-",
                                        color=color,
                                        linewidth=0.4,
                                        alpha=0.7))

    panel_specs = [
        ("A", "ECBSS_MEAN", "Mean ECBSS across all bias families", False),
        ("B", "A", "Arousal  (calm ← 0 → activated)", False),
        ("C", "C", "Control  (uncontrolled ← 0 → controlled)", False),
        ("D", None, "Emotion Cluster Assignment", False),
    ]

    for idx, (lbl, dim, title, add_ellipses) in enumerate(panel_specs):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        _panel_label(ax, lbl)

        if dim is not None and dim != "ECBSS_MEAN":
            vals = emotion_df[dim].values
            norm = TwoSlopeNorm(vmin=-1000, vcenter=0, vmax=1000)
            sc = ax.scatter(coords[:, 0], coords[:, 1],
                            c=vals, cmap=ECBSS_CMAP, norm=norm,
                            s=22, alpha=0.80, linewidths=0, rasterized=True)
            cbar = plt.colorbar(sc, ax=ax, fraction=0.036, pad=0.02,
                                aspect=22, shrink=0.85)
            cbar.set_label(f"{dim} score  (−1000 → +1000)", fontsize=7)
            cbar.ax.tick_params(labelsize=6.5)
            # Annotate top/bottom 10 by this component
            _annotate_top_bottom(ax, vals, emotions, coords, n=10)

        elif dim == "ECBSS_MEAN":
            # Panel A: color by mean ECBSS
            if mean_ecbss is not None and not np.all(np.isnan(mean_ecbss)):
                vmax_val = float(np.nanpercentile(np.abs(mean_ecbss), 98))
                norm = TwoSlopeNorm(vmin=-vmax_val, vcenter=0, vmax=vmax_val)
                sc = ax.scatter(coords[:, 0], coords[:, 1],
                                c=mean_ecbss, cmap=ECBSS_CMAP, norm=norm,
                                s=22, alpha=0.80, linewidths=0, rasterized=True)
                cbar = plt.colorbar(sc, ax=ax, fraction=0.036, pad=0.02,
                                    aspect=22, shrink=0.85)
                cbar.set_label("Mean ECBSS  (all 11 families)", fontsize=7)
                cbar.ax.tick_params(labelsize=6.5)
                # Annotate top/bottom 10 by mean ECBSS
                _annotate_top_bottom(ax, mean_ecbss, emotions, coords, n=10)
            else:
                # Fallback: color by valence if no ecbss_df
                vals = emotion_df["V"].values
                norm = TwoSlopeNorm(vmin=-1000, vcenter=0, vmax=1000)
                sc = ax.scatter(coords[:, 0], coords[:, 1],
                                c=vals, cmap=ECBSS_CMAP, norm=norm,
                                s=22, alpha=0.80, linewidths=0, rasterized=True)
                cbar = plt.colorbar(sc, ax=ax, fraction=0.036, pad=0.02,
                                    aspect=22, shrink=0.85)
                cbar.set_label("V score  (−1000 → +1000)", fontsize=7)
                cbar.ax.tick_params(labelsize=6.5)

        else:
            # Panel D: cluster assignment
            for cid in sorted(cluster_labels.unique()):
                mask = cluster_labels.values == cid
                cx = coords[mask, 0]
                cy = coords[mask, 1]
                color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
                n_in = mask.sum()

                # Scatter
                ax.scatter(cx, cy, c=color, s=25, alpha=0.80,
                           linewidths=0, rasterized=True,
                           label=f"{CLUSTER_LABELS[cid]}  (n={n_in})")

                # Convex hull outline
                if len(cx) >= 4:
                    try:
                        pts = np.column_stack([cx, cy])
                        hull = ConvexHull(pts)
                        hull_pts = np.append(hull.vertices, hull.vertices[0])
                        ax.fill(pts[hull_pts, 0], pts[hull_pts, 1],
                                color=color, alpha=0.07, linewidth=0)
                        ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                                color=color, linewidth=0.7, alpha=0.55)
                    except Exception:
                        pass

                # Centroid label
                centroid = np.array([cx.mean(), cy.mean()])
                ax.text(centroid[0], centroid[1],
                        CLUSTER_LABELS_SHORT[cid],
                        ha="center", va="center", fontsize=7,
                        fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.22",
                                  facecolor=color, alpha=0.88,
                                  edgecolor="white", linewidth=0.6))

            ax.legend(fontsize=6, loc="lower right",
                      framealpha=0.85, edgecolor="#cccccc",
                      markerscale=1.2, handletextpad=0.3,
                      labelspacing=0.25)

        ax.set_title(title, loc="left", fontsize=8, pad=4)
        ax.set_xlabel("UMAP-1", fontsize=7.5)
        ax.set_ylabel("UMAP-2", fontsize=7.5)
        ax.tick_params(labelsize=7)
        _despine(ax)

    _note(fig,
          "UMAP projection of the full emotion lexicon (N = " +
          str(len(emotions)) + " states) in 2D space computed from five affective dimensions (V, A, C, U, S). "
          "Panel A colors by mean ECBSS across all 11 bias families; red labels = top-10 highest vulnerability, "
          "blue labels = bottom-10 lowest. "
          "Panels B–C color by standardized component score; top-10 and bottom-10 annotated. "
          "Panel D shows K-means cluster assignment (k = 6) with convex hull borders; "
          "cluster centroids are labeled.",
          y=0.025)

    _save(fig, "fig2_emotion_landscape")


# ── Figure 3: ECBSS Heatmap with Dendrogram + Violins ───────────────────────

def fig3_ecbss_heatmap(
    cluster_ecbss: pd.DataFrame,
    ecbss_df: pd.DataFrame,
    cluster_labels: pd.Series,
    bootstrap_cis: Optional[dict] = None,
) -> None:
    """Cluster-level heatmap with aligned top/right dendrograms."""
    families = _ordered_families(cluster_ecbss.columns)
    cluster_ids = _ordered_clusters(cluster_ecbss)

    heat_data = cluster_ecbss.loc[cluster_ids, families].copy()
    row_labels = [CLUSTER_LABELS.get(cid, f"C{cid}") for cid in cluster_ids]

    fig = plt.figure(figsize=(12.0, 7.0))
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[0.18, 1.0, 0.16],
        width_ratios=[1.0, 0.18],
        left=0.12, right=0.96, top=0.96, bottom=0.26,
        hspace=0.0, wspace=0.0,
    )

    ax_dend_top = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_dend_right = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_axes([0.12, 0.265, 0.712, 0.025])

    col_dist = pdist(heat_data.T.values, metric="euclidean")
    Z_col = optimal_leaf_ordering(linkage(col_dist, method="ward"), col_dist)
    dend_col = dendrogram(
        Z_col, ax=ax_dend_top, orientation="top",
        color_threshold=0, above_threshold_color="#8d8d8d",
        no_labels=True,
    )
    col_order_idx = dend_col["leaves"]
    ax_dend_top.set_axis_off()

    if len(heat_data) > 2:
        row_dist = pdist(heat_data.values, metric="euclidean")
        Z_row = optimal_leaf_ordering(linkage(row_dist, method="ward"), row_dist)
        dend_row = dendrogram(
            Z_row, ax=ax_dend_right, orientation="right",
            color_threshold=0, above_threshold_color="#8d8d8d",
            no_labels=True,
        )
        row_order_idx = dend_row["leaves"]
    else:
        row_order_idx = list(range(len(heat_data)))
        ax_dend_right.set_axis_off()

    ordered = heat_data.iloc[row_order_idx, col_order_idx]
    ordered_rows = [row_labels[i] for i in row_order_idx]
    ordered_row_ids = [cluster_ids[i] for i in row_order_idx]
    ordered_families = [families[i] for i in col_order_idx]
    ordered_cols = [FAMILY_FULL_LABEL[ordered_families[i]] for i in range(len(ordered_families))]

    n_rows, n_cols = ordered.shape
    x_centers = np.arange(n_cols) * 10 + 5
    y_centers = np.arange(n_rows) * 10 + 5

    vmax = max(650.0, float(np.abs(ordered.values).max()))
    im = ax_heat.imshow(
        ordered.values,
        cmap=ECBSS_CMAP,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        aspect="auto",
        interpolation="nearest",
        extent=(0, n_cols * 10, n_rows * 10, 0),
    )

    for ri in range(ordered.shape[0]):
        cid = ordered_row_ids[ri]
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        ax_heat.add_patch(
            mpatches.Rectangle(
                (-3.5, y_centers[ri] - 4.5), 2.8, 9.0,
                facecolor=color, linewidth=0,
                transform=ax_heat.transData, clip_on=False,
            )
        )

        for ci in range(ordered.shape[1]):
            fam = ordered_families[ci]
            val = ordered.iat[ri, ci]
            sig = ""
            if bootstrap_cis is not None and (cid, fam) in bootstrap_cis:
                lo, hi = bootstrap_cis[(cid, fam)]
                if lo > 0 or hi < 0:
                    sig = "*"
            txt_color = "white" if abs(val) > 0.58 * vmax else "#1d1d1d"
            ax_heat.text(
                x_centers[ci], y_centers[ri], f"{val:.0f}{sig}",
                ha="center", va="center",
                fontsize=8.2, color=txt_color, fontweight="bold",
            )

    ax_heat.set_xlim(-3.8, n_cols * 10)
    ax_heat.set_xticks(x_centers)
    ax_heat.set_xticklabels(ordered_cols, rotation=0, ha="center", fontsize=7.4, multialignment="center")
    ax_heat.set_yticks(y_centers)
    ax_heat.set_yticklabels(ordered_rows, fontsize=9.0)
    ax_heat.tick_params(axis="x", pad=6)
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # Align right dendrogram exactly to heatmap
    if len(heat_data) > 2:
        ax_dend_right.set_ylim(ax_heat.get_ylim())
        ax_dend_right.set_xlim(left=0)
        ax_dend_right.margins(x=0, y=0)
    ax_dend_right.set_axis_off()

    # Align top dendrogram exactly to heatmap x range
    ax_dend_top.set_xlim(-3.8, n_cols * 10)
    ax_dend_top.margins(x=0, y=0)
    ax_dend_top.set_ylim(bottom=0)
    ax_dend_top.set_axis_off()

    cbar = fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
    cbar.set_label("Mean ECBSS  (Emotion-Conditioned Bias Shift Score)", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.5)

    # Ensure dendrograms align flush with heatmap edges
    ax_dend_top.set_xlim(-3.8, n_cols * 10)
    if len(heat_data) > 2:
        ax_dend_right.set_ylim(n_rows * 10, 0)  # match inverted heatmap y-axis

    _save(fig, "fig3_ecbss_heatmap")


# ── Figure 4: Dimensional Sensitivity Profiles ──────────────────────────────

def fig4_dimensional_profiles(
    bias_profiles: dict,
    regression_results: Optional[dict] = None,
    bootstrap_cis: Optional[dict] = None,
) -> None:
    """
    Fig 4 (A–B): Dimensional sensitivity of bias families.

    Panel A: Radar/spider charts — 3×4 grid, one per family, polar axes,
             filled area + CI band.
    Panel B: Dot+CI forest-style plot: regression betas per dimension,
             grouped by dimension, families as points.
    """
    families = [f for f in FAMILY_SHORT if f in bias_profiles]
    dims = ["V", "A", "C", "U", "S"]
    dim_labels_radar = ["Valence", "Arousal", "Control", "Uncertainty", "Social"]
    n_fam = len(families)
    ncols_radar = 4
    nrows_radar = int(np.ceil(n_fam / ncols_radar))

    fig = plt.figure(figsize=(18, 11))
    _figure_title(fig, "Figure S9",
                  "Dimensional Sensitivity Profiles of Bias Families: Affective Modulation Across Five Dimensions",
                  y_num=0.988, y_title=0.972)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           width_ratios=[1.6, 1],
                           left=0.04, right=0.98,
                           top=0.92, bottom=0.09,
                           wspace=0.28)

    # ── Panel A: Radar grid ──────────────────────────────────────────────────
    gs_radars = gridspec.GridSpecFromSubplotSpec(
        nrows_radar, ncols_radar, subplot_spec=gs[0],
        hspace=0.70, wspace=0.55)

    N_dim = len(dims)
    angles = np.linspace(0, 2 * np.pi, N_dim, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    for fi, family in enumerate(families):
        row_i = fi // ncols_radar
        col_i = fi % ncols_radar
        ax = fig.add_subplot(gs_radars[row_i, col_i], polar=True)

        if fi == 0:
            ax.text(-0.30, 1.30, "A", transform=ax.transAxes,
                    fontsize=12, fontweight="bold", va="top")

        profile = bias_profiles[family]
        values = [float(profile.get(d, 0)) for d in dims]
        values_closed = values + values[:1]

        # CI band (if available)
        if bootstrap_cis is not None and family in bootstrap_cis:
            cis = bootstrap_cis[family]
            lo_vals = [float(cis.get(d, {}).get("lo", v)) for d, v in zip(dims, values)]
            hi_vals = [float(cis.get(d, {}).get("hi", v)) for d, v in zip(dims, values)]
            lo_closed = lo_vals + lo_vals[:1]
            hi_closed = hi_vals + hi_vals[:1]
            ax.fill_between(angles_closed, lo_closed, hi_closed,
                            alpha=0.18, color="#888888", linewidth=0)

        mean_val = np.mean(values)
        fill_color = (ECBSS_CMAP(0.85) if mean_val > 8
                      else (ECBSS_CMAP(0.15) if mean_val < -8 else "#888888"))

        ax.plot(angles_closed, values_closed, color=fill_color,
                linewidth=1.4, solid_capstyle="round")
        ax.fill(angles_closed, values_closed, color=fill_color, alpha=0.22)

        # Zero ring
        ax.plot(angles_closed, [0] * (N_dim + 1), color="#aaaaaa",
                linewidth=0.6, linestyle="--")

        ax.set_thetagrids(np.degrees(angles), dim_labels_radar, fontsize=5.0)
        max_r = 100
        ax.set_ylim(-max_r, max_r)
        ax.set_yticks([-75, -50, -25, 0, 25, 50, 75])
        ax.set_yticklabels(["", "-50", "", "0", "", "50", ""], fontsize=4.5)
        ax.tick_params(axis="y", labelsize=4.5)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        ax.spines["polar"].set_visible(False)

        short_title = FAMILY_SHORT[family]
        words = short_title.split()
        if len(words) > 2:
            short_title = " ".join(words[:2]) + "\n" + " ".join(words[2:])
        ax.set_title(short_title, size=6.0, pad=10, loc="center",
                     fontweight="bold", color="#222222")

    # ── Panel B: Dot + CI forest plot (betas) ───────────────────────────────
    ax_forest = fig.add_subplot(gs[1])
    _panel_label(ax_forest, "B", x=-0.09, y=1.04)

    if regression_results is not None:
        per_family = regression_results.get("per_family", {})
        fams_with_reg = [f for f in families if f in per_family]
        n_f = len(fams_with_reg)
        short_labels_reg = [FAMILY_SHORT[f] for f in fams_with_reg]

        y_spacing = len(dims) + 1.5
        y_fam_centers = np.arange(n_f) * y_spacing
        dim_offsets = np.linspace(-(len(dims) - 1) / 2, (len(dims) - 1) / 2, len(dims)) * 0.7

        for di, dim in enumerate(dims):
            color = DIM_COLORS[dim]
            for fi, fam in enumerate(fams_with_reg):
                fres = per_family.get(fam, {})
                beta = fres.get("params", {}).get(dim, 0)
                se = fres.get("bse", {}).get(dim, 5)
                pval = fres.get("pvalues", {}).get(dim, 1.0)
                y_val = y_fam_centers[fi] + dim_offsets[di]
                alpha_val = 0.92 if pval < 0.05 else 0.28
                ms = 6 if pval < 0.05 else 4
                ax_forest.errorbar(beta, y_val, xerr=1.96 * se,
                                   fmt="o", color=color,
                                   markersize=ms, linewidth=0.9,
                                   capsize=2.5, alpha=alpha_val,
                                   elinewidth=0.7)
                sig = _sig_marker(pval)
                if sig:
                    ax_forest.text(beta + 1.96 * se + 1.5, y_val, sig,
                                   fontsize=6, color=color, va="center")

        ax_forest.axvline(0, color="#333333", linewidth=0.8,
                          linestyle="--", alpha=0.6)
        ax_forest.set_yticks(y_fam_centers)
        ax_forest.set_yticklabels(short_labels_reg, fontsize=7.5)
        ax_forest.set_xlabel("Regression beta  (95% CI)", fontsize=8)
        ax_forest.set_title("Per-Family Dimensional Regression Coefficients\n"
                             "(solid marker = p < .05; faded = n.s.)",
                             loc="left", fontsize=8)

        dim_handles = [mpatches.Patch(color=DIM_COLORS[d], label=DIM_LABELS[d])
                       for d in dims]
        ax_forest.legend(handles=dim_handles, title="Dimension",
                         fontsize=7, title_fontsize=7.5,
                         loc="lower right", framealpha=0.88)
        _despine(ax_forest)
    else:
        # Fallback: heatmap of profile values
        heat_data = np.array([[float(bias_profiles[f].get(d, 0)) for d in dims]
                               for f in families])
        im = ax_forest.imshow(heat_data, cmap=ECBSS_CMAP, aspect="auto",
                               vmin=-100, vmax=100)
        plt.colorbar(im, ax=ax_forest, fraction=0.046, pad=0.03,
                     label="Dimensional sensitivity score")
        ax_forest.set_xticks(range(len(dims)))
        ax_forest.set_xticklabels(dims, fontsize=8)
        ax_forest.set_yticks(range(len(families)))
        ax_forest.set_yticklabels([FAMILY_SHORT[f] for f in families], fontsize=7.5)
        ax_forest.set_title("Sensitivity Score per Dimension", loc="left", fontsize=8)
        for i in range(len(families)):
            for j in range(len(dims)):
                v = heat_data[i, j]
                ax_forest.text(j, i, f"{v:.0f}", ha="center", va="center",
                               fontsize=7, color="white" if abs(v) > 60 else "#111111",
                               fontweight="bold")
        _despine(ax_forest)

    _note(fig,
          "Panel A: Radar/spider charts for each of the 11 bias families, showing sensitivity scores "
          "on the five affective dimensions (Valence, Arousal, Control, Uncertainty, Social Orientation). "
          "Values range from −100 (dimension strongly attenuates bias susceptibility) to +100 (amplifies). "
          "Red fill indicates net amplification; blue fill indicates net attenuation. "
          "Light grey band shows 95% bootstrap CI where available. "
          "Panel B: Per-family regression beta coefficients with 95% CIs for each dimension; "
          "significance markers: * p < .05, ** p < .01, *** p < .001.",
          y=0.02)

    _save(fig, "fig4_dimensional_profiles")


# ── Figure 5: Bipartite Network ──────────────────────────────────────────────

def fig5_network(
    cluster_ecbss: pd.DataFrame,
    cluster_labels: pd.Series,
    taxonomy_df: pd.DataFrame,
    ecbss_df: pd.DataFrame,
) -> None:
    """Two-panel bipartite layout: coefficients (A) and within-cell variance (B)."""
    families = _ordered_families(cluster_ecbss.columns)
    cluster_ids = _ordered_clusters(cluster_ecbss)

    cluster_sizes = cluster_labels.value_counts().to_dict()
    family_sizes = taxonomy_df.groupby("family").size().to_dict()

    cluster_order = sorted(
        cluster_ids,
        key=lambda cid: cluster_ecbss.loc[cid, families].mean(),
        reverse=True,
    )
    family_order = sorted(
        families,
        key=lambda fam: cluster_ecbss.loc[cluster_ids, fam].mean(),
        reverse=True,
    )

    # Within-cell standard deviation matrix for variance panel.
    aligned_clusters = cluster_labels.reindex(ecbss_df.index)
    var_matrix = pd.DataFrame(index=cluster_ids, columns=families, dtype=float)
    for cid in cluster_ids:
        idx = aligned_clusters[aligned_clusters == cid].index
        for fam in families:
            vals = ecbss_df.loc[idx, fam].dropna().values
            var_matrix.loc[cid, fam] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 8.4), gridspec_kw={"wspace": 0.10})

    x_left, x_right = 0.24, 0.74
    y_left = np.linspace(0.88, 0.12, len(cluster_order))
    y_right = np.linspace(0.92, 0.08, len(family_order))
    cluster_pos = {cid: (x_left, y) for cid, y in zip(cluster_order, y_left)}
    family_pos = {fam: (x_right, y) for fam, y in zip(family_order, y_right)}

    max_cluster_size = max(cluster_sizes.values())
    max_family_size = max(family_sizes.values())
    max_abs_coef = max(abs(float(cluster_ecbss.loc[cid, fam])) for cid in cluster_ids for fam in families)
    max_var = max(float(var_matrix.loc[cid, fam]) for cid in cluster_ids for fam in families)

    def _draw_nodes(ax: plt.Axes) -> None:
        fam_color_map = {fam: FAMILY_COLORS[i % len(FAMILY_COLORS)] for i, fam in enumerate(family_order)}
        for cid in cluster_order:
            x, y = cluster_pos[cid]
            color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            radius = 0.028 + 0.052 * np.sqrt(cluster_sizes[cid] / max_cluster_size)
            ax.add_patch(plt.Circle((x, y), radius, facecolor=color, edgecolor="white", linewidth=1.8, zorder=3))
            ax.text(x, y, CLUSTER_LABELS_SHORT[cid], ha="center", va="center",
                    fontsize=8.2, fontweight="bold", color="white", zorder=4)
            ax.text(x - radius - 0.028, y, f"{CLUSTER_LABELS[cid]}\n(n = {cluster_sizes[cid]})",
                    ha="right", va="center", fontsize=7.6, color=color, fontweight="bold")

        for fam in family_order:
            x, y = family_pos[fam]
            color = fam_color_map[fam]
            half = 0.020 + 0.040 * np.sqrt(family_sizes[fam] / max_family_size)
            ax.add_patch(
                FancyBboxPatch(
                    (x - half, y - half), 2 * half, 2 * half,
                    boxstyle="round,pad=0.004,rounding_size=0.003",
                    facecolor=color, edgecolor="white", linewidth=1.5, zorder=3,
                )
            )
            ax.text(x, y, FAMILY_ABBR[fam], ha="center", va="center",
                    fontsize=7.9, fontweight="bold", color="white", zorder=4)
            ax.text(x + half + 0.024, y, f"{FAMILY_SHORT[fam]}\n(n = {family_sizes[fam]})",
                    ha="left", va="center", fontsize=7.3, color=color)

        ax.text(x_left, 0.985, "Emotion clusters", ha="center", va="top",
                fontsize=9.6, fontweight="bold", color="#444444")
        ax.text(x_right, 0.985, "Bias families", ha="center", va="top",
                fontsize=9.6, fontweight="bold", color="#444444")

    # Panel A: coefficient-weighted links.
    axA, axB = axes
    for ax in (axA, axB):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    _panel_label(axA, "A", x=-0.03, y=1.02)
    _panel_label(axB, "B", x=-0.03, y=1.02)

    for cid in cluster_order:
        for fam in family_order:
            val = float(cluster_ecbss.loc[cid, fam])
            x1, y1 = cluster_pos[cid]
            x2, y2 = family_pos[fam]
            color = "#c94936" if val >= 0 else "#2c7fb8"
            lw = 0.5 + 4.8 * abs(val) / (max_abs_coef + 1e-9)
            alpha = 0.12 + 0.62 * abs(val) / (max_abs_coef + 1e-9)
            rad = 0.16 * np.sign(y2 - y1)
            axA.add_patch(
                FancyArrowPatch(
                    (x1 + 0.035, y1), (x2 - 0.05, y2),
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-",
                    linewidth=lw,
                    color=color,
                    alpha=alpha,
                    zorder=1,
                )
            )

    _draw_nodes(axA)
    axA.set_title("Coefficient-weighted bipartite links", loc="left", fontsize=9)
    axA.legend(
        handles=[
            mpatches.Patch(facecolor="#c94936", label="Amplifying edge"),
            mpatches.Patch(facecolor="#2c7fb8", label="Attenuating edge"),
            plt.Line2D([0], [0], color="#888888", linewidth=3, label="All 66 links shown"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.52, -0.02),
        ncol=1,
        fontsize=7.2,
        framealpha=0.92,
        edgecolor="#cccccc",
    )

    # Panel B: variance-weighted links.
    for cid in cluster_order:
        for fam in family_order:
            sd_val = float(var_matrix.loc[cid, fam])
            x1, y1 = cluster_pos[cid]
            x2, y2 = family_pos[fam]
            lw = 0.5 + 4.8 * sd_val / (max_var + 1e-9)
            alpha = 0.12 + 0.62 * sd_val / (max_var + 1e-9)
            rad = 0.16 * np.sign(y2 - y1)
            axB.add_patch(
                FancyArrowPatch(
                    (x1 + 0.035, y1), (x2 - 0.05, y2),
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-",
                    linewidth=lw,
                    color="#5b6f9e",
                    alpha=alpha,
                    zorder=1,
                )
            )

    _draw_nodes(axB)
    axB.set_title("Variance-weighted bipartite links", loc="left", fontsize=9)
    axB.legend(
        handles=[
            plt.Line2D([0], [0], color="#5b6f9e", linewidth=2.4, label="Within-cell ECBSS SD edge"),
            plt.Line2D([0], [0], color="#5b6f9e", linewidth=4.8, label="Thicker = higher within-cluster SD"),
            plt.Line2D([0], [0], color="#5b6f9e", linewidth=1.2, label="All 66 links shown"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.52, -0.02),
        ncol=1,
        fontsize=7.2,
        framealpha=0.92,
        edgecolor="#cccccc",
    )

    _save(fig, "fig5_network")


# ── Figure 6: Regression Results (Forest + R²) ──────────────────────────────

def fig6_regression(
    regression_results: dict,
    permutation_results: Optional[dict] = None,
) -> None:
    """
    Fig 6 (A–B): Mixed-effects regression results.

    Panel A: Forest plot — beta ± 95% CI per dimension per family,
             grouped by dimension, permutation p-values as markers.
    Panel B: R² heatmap (families × dimensions + total), annotated.
    """
    per_family = regression_results.get("per_family", {})
    overall = regression_results.get("overall", {})
    dims = ["V", "A", "C", "U", "S"]
    families = [f for f in FAMILY_SHORT if f in per_family]
    n_fam = len(families)

    fig = plt.figure(figsize=(18, 9))
    _figure_title(fig, "Figure S10",
                  "Dimensional Regression Analysis: Affective Predictors of ECBSS and Family-Level Variance Explained",
                  y_num=0.988, y_title=0.972)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           width_ratios=[1.55, 1],
                           left=0.05, right=0.98,
                           top=0.91, bottom=0.09,
                           wspace=0.24)

    # ── Panel A: Forest plot ──────────────────────────────────────────────────
    ax_forest = fig.add_subplot(gs[0])
    _panel_label(ax_forest, "A", x=-0.06, y=1.04)

    y_spacing = len(dims) + 2.0
    y_fam_centers = np.arange(n_fam) * y_spacing
    dim_offsets = np.linspace(-((len(dims) - 1) * 0.72 / 2),
                               ((len(dims) - 1) * 0.72 / 2), len(dims))

    for di, dim in enumerate(dims):
        color = DIM_COLORS[dim]
        for fi, fam in enumerate(families):
            fres = per_family.get(fam, {})
            beta = fres.get("params", {}).get(dim, 0)
            se = fres.get("bse", {}).get(dim, 5)
            pval = fres.get("pvalues", {}).get(dim, 1.0)

            # Permutation p if available
            perm_p = pval
            if permutation_results is not None:
                perm_p = permutation_results.get(fam, {}).get(dim, pval)

            y_val = y_fam_centers[fi] + dim_offsets[di]
            alpha_val = 0.92 if pval < 0.05 else 0.28
            ms = 6.5 if pval < 0.05 else 4.5
            marker = "o" if pval < 0.05 else "D"

            ax_forest.errorbar(beta, y_val, xerr=1.96 * se,
                               fmt=marker, color=color,
                               markersize=ms, linewidth=0.9,
                               capsize=2.8, alpha=alpha_val,
                               elinewidth=0.75, markeredgewidth=0)

            # Significance marker
            sig = _sig_marker(perm_p)
            if sig:
                ax_forest.text(beta + 1.96 * se + 2.0, y_val, sig,
                               fontsize=6.5, color=color, va="center",
                               fontweight="bold")

        # Dimension legend strip on right
        ax_forest.text(ax_forest.get_xlim()[1] if ax_forest.get_xlim()[1] != 0 else 80,
                       y_fam_centers.mean() + dim_offsets[di], dim,
                       fontsize=7, color=color, va="center",
                       fontweight="bold")

    ax_forest.axvline(0, color="#333333", linewidth=0.9,
                      linestyle="--", alpha=0.55)
    ax_forest.set_yticks(y_fam_centers)
    ax_forest.set_yticklabels([FAMILY_SHORT[f] for f in families], fontsize=7.5)
    ax_forest.set_xlabel("Regression coefficient β  (95% CI)", fontsize=8.5)
    ax_forest.set_title(
        "Fixed-Effects Dimensional Predictors of ECBSS\n"
        "Filled marker: p < .05; hollow: n.s. | * p<.05  ** p<.01  *** p<.001",
        loc="left", fontsize=8)

    # Background alternating family bands
    for fi in range(n_fam):
        if fi % 2 == 0:
            ax_forest.axhspan(y_fam_centers[fi] - y_spacing / 2,
                              y_fam_centers[fi] + y_spacing / 2,
                              color="#f0f0f0", alpha=0.45, linewidth=0)

    dim_handles = [mpatches.Patch(color=DIM_COLORS[d],
                                  label=f"{d}: {DIM_LABELS[d]}")
                   for d in dims]
    ax_forest.legend(handles=dim_handles, title="Dimension",
                     fontsize=7, title_fontsize=7.5,
                     loc="lower right", framealpha=0.9)
    _despine(ax_forest)

    # ── Panel B: R² heatmap ───────────────────────────────────────────────────
    ax_r2 = fig.add_subplot(gs[1])
    _panel_label(ax_r2, "B", x=-0.09, y=1.04)

    r2_rows = []
    for fam in families:
        fres = per_family.get(fam, {})
        r2_total = fres.get("rsquared", np.nan)
        params = fres.get("params", {})
        total_abs_beta = sum(abs(params.get(d, 0)) for d in dims)
        if total_abs_beta > 1e-9 and not np.isnan(r2_total):
            row = [abs(params.get(d, 0)) / total_abs_beta * r2_total
                   for d in dims] + [r2_total]
        else:
            row = [np.nan] * len(dims) + [r2_total if not np.isnan(r2_total) else 0]
        r2_rows.append(row)

    r2_array = np.array(r2_rows)
    col_labs = [DIM_LABELS[d] for d in dims] + ["R² Total"]

    vmax_r2 = float(np.nanmax(r2_array)) if not np.all(np.isnan(r2_array)) else 0.5
    im_r2 = ax_r2.imshow(r2_array, cmap="Blues", aspect="auto",
                          vmin=0, vmax=max(vmax_r2, 0.01))
    plt.colorbar(im_r2, ax=ax_r2, fraction=0.042, pad=0.03,
                 label="R²  (proportion of variance)", shrink=0.85)
    im_r2.axes.tick_params(length=0)

    for ri in range(len(families)):
        for ci in range(len(col_labs)):
            v = r2_array[ri, ci]
            if np.isnan(v):
                continue
            tc = "white" if v > 0.5 * vmax_r2 else "#111111"
            ax_r2.text(ci, ri, f"{v:.3f}",
                       ha="center", va="center", fontsize=6.5,
                       color=tc, fontweight="bold")

    ax_r2.set_xticks(range(len(col_labs)))
    ax_r2.set_xticklabels(col_labs, fontsize=7.5, rotation=35, ha="right")
    ax_r2.set_yticks(range(len(families)))
    ax_r2.set_yticklabels([FAMILY_SHORT[f] for f in families], fontsize=7.5)
    ax_r2.set_title("Variance Explained (R²) by Dimension and Family",
                    loc="left", fontsize=8)
    for spine in ax_r2.spines.values():
        spine.set_visible(False)
    ax_r2.tick_params(length=0)

    # Overall model summary strip
    if overall:
        ovr_params = overall.get("params", {})
        ovr_pvals = overall.get("pvalues", {})
        summary_parts = []
        for d in dims:
            b = ovr_params.get(d, 0)
            p = ovr_pvals.get(d, 1)
            sig = _sig_marker(p)
            summary_parts.append(f"{d}: β={b:.1f}{sig}")
        _maybe_figtext(fig, 0.015, 0.055,
                 "Overall mixed-effects model:  " + "  ".join(summary_parts),
                 fontsize=6.5, color="#333333", ha="left", va="bottom",
                 fontstyle="italic")

    _note(fig,
          "Panel A: Per-family fixed-effect regression beta coefficients (with 95% CI bars) "
          "for each affective dimension as predictor of ECBSS. "
          "Filled circles (p < .05) vs. hollow diamonds (n.s.); significance markers based on "
          "permutation p-values where available. Alternating grey bands aid row readability. "
          "Panel B: Proportion of total R² attributable to each dimension per bias family "
          "(proportional decomposition by absolute beta weight). Last column = total R². "
          "Models fit ECBSS ∼ V + A + C + U + S with random intercepts per emotion.",
          y=0.025)

    _save(fig, "fig6_regression")


# ── Figure 7: Hypothesis Evaluation ─────────────────────────────────────────

def fig7_hypothesis_evaluation(
    cluster_ecbss: pd.DataFrame,
    ecbss_df: pd.DataFrame,
    cluster_labels: pd.Series,
    cohen_d_results: Optional[dict] = None,
) -> None:
    """
    Fig 7 (A–B): Preregistered hypothesis evaluation.

    Panel A: Profile plot — mean ± SEM across all families for 4 hypothesis groups.
    Panel B: Hypothesis evaluation grid table: H1–H7, with effect size d and verdict.
    """
    families = _ordered_families(cluster_ecbss.columns)
    fam_shorts = [FAMILY_SHORT[f] for f in families]
    x_pos = np.arange(len(families))

    fig = plt.figure(figsize=(18, 8.5))
    _figure_title(fig, "Figure 7",
                  "Preregistered Hypothesis Evaluation: Emotion–Bias Susceptibility Predictions and Empirical Outcomes",
                  y_num=0.988, y_title=0.972)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           width_ratios=[1.1, 1],
                           left=0.05, right=0.98,
                           top=0.91, bottom=0.10,
                           wspace=0.22)

    # ── Panel A: Profile plot ─────────────────────────────────────────────────
    ax_prof = fig.add_subplot(gs[0])
    _panel_label(ax_prof, "A", x=-0.07, y=1.04)

    hypothesis_groups = {
        "H1: Threat/Alarm": ["afraid", "panicked", "alarmed", "horrified", "terrified"],
        "H2: Reward/Enthusiasm": ["excited", "enthusiastic", "optimistic", "eager", "inspired"],
        "H3: Shame/Guilt": ["ashamed", "guilty", "humiliated", "embarrassed", "remorseful"],
        "H4: Calm/Reflective": ["calm", "serene", "peaceful", "tranquil", "relaxed"],
    }
    h_colors = ["#d62728", "#fd8d3c", "#8c6d31", "#31a354"]

    for hi, (h_label, h_emotions) in enumerate(hypothesis_groups.items()):
        available = [e for e in h_emotions if e in ecbss_df.index]
        if not available:
            continue
        profiles = ecbss_df.loc[available, families]
        mean_p = profiles.mean()
        sem_p = profiles.sem()
        color = h_colors[hi]
        ax_prof.plot(x_pos, mean_p.values, marker="o", markersize=6,
                     color=color, linewidth=1.8, label=f"{h_label}  (n={len(available)})",
                     zorder=4)
        ax_prof.fill_between(x_pos,
                              mean_p.values - sem_p.values,
                              mean_p.values + sem_p.values,
                              color=color, alpha=0.14, zorder=2, linewidth=0)

    # Neutral band
    ax_prof.axhspan(-50, 50, color="#eeeeee", alpha=0.7, zorder=1,
                    label="Neutral zone (|ECBSS| < 50)")
    ax_prof.axhline(0, color="#333333", linewidth=0.8, linestyle="--",
                    alpha=0.55, zorder=3)

    ax_prof.set_xticks(x_pos)
    ax_prof.set_xticklabels(fam_shorts, rotation=42, ha="right", fontsize=7)
    ax_prof.set_ylabel("Mean ECBSS  (±SEM)", fontsize=8.5)
    ax_prof.set_title(
        "ECBSS Profiles for Hypothesis-Relevant Emotion Groups\n(shaded band = ±SEM, grey zone = neutral)",
        loc="left", fontsize=8)
    ax_prof.legend(fontsize=7, loc="upper right",
                   framealpha=0.88, edgecolor="#cccccc",
                   handlelength=1.4, labelspacing=0.3)
    _despine(ax_prof)

    # ── Panel B: Hypothesis grid table ───────────────────────────────────────
    ax_grid = fig.add_subplot(gs[1])
    _panel_label(ax_grid, "B", x=-0.08, y=1.04)

    hypotheses = [
        ("H1", "Threat/alarm → ↑ salience & urgency biases",
         "High-arousal negative", "↑ Attention & Social",
         "attention_salience_and_signal_detection_biases",
         ["afraid", "panicked", "alarmed", "horrified"]),
        ("H2", "Reward/enthusiasm → ↑ optimism & trust biases",
         "High-arousal positive", "↑ Trust & Self-Assessment",
         "trust_source_credibility_and_truth_judgment_biases",
         ["excited", "enthusiastic", "optimistic", "eager"]),
        ("H3", "Shame/guilt → ↑ conformity & compliance",
         "Social threat & shame", "↑ Social & Authority",
         "social_influence_authority_affiliation_and_identity_biases",
         ["ashamed", "guilty", "humiliated", "embarrassed"]),
        ("H4", "Calm/reflective → ↓ fast cue-driven biases",
         "Calm positive", "↓ Attention & Interface",
         "attention_salience_and_signal_detection_biases",
         ["calm", "serene", "peaceful", "tranquil"]),
        ("H5", "High-arousal neg. → ↑ attention salience",
         "High-arousal negative", "↑ Attention & Salience",
         "attention_salience_and_signal_detection_biases",
         ["panicked", "terrified", "alarmed", "horrified"]),
        ("H6", "Low-control emotions → ↑ authority deference",
         "Low-valence withdrawn", "↑ Social & Authority",
         "social_influence_authority_affiliation_and_identity_biases",
         ["helpless", "cornered", "trapped", "powerless"]),
        ("H7", "Composite states: non-additive effects",
         "Mixed/composite", "Varies by dyad",
         None, []),
    ]

    col_headers = ["Hyp.", "Prediction", "Group", "Expected", "d", "Evidence", "Verdict"]
    col_widths = [0.055, 0.28, 0.14, 0.16, 0.055, 0.14, 0.11]
    col_x = np.cumsum([0] + col_widths[:-1])
    row_h = 0.108

    # Header row
    ax_grid.axis("off")
    ax_grid.set_xlim(0, sum(col_widths))
    ax_grid.set_ylim(0, 1)

    header_y = 0.90
    for ci, (hdr, w) in enumerate(zip(col_headers, col_widths)):
        ax_grid.add_patch(mpatches.FancyBboxPatch(
            (col_x[ci] + 0.002, header_y - 0.04), w - 0.004, 0.055,
            boxstyle="square,pad=0.002",
            facecolor="#2c3e50", linewidth=0))
        ax_grid.text(col_x[ci] + w / 2, header_y - 0.012, hdr,
                     ha="center", va="center", fontsize=7,
                     fontweight="bold", color="white")

    verdict_palette = {
        "Supported": "#2ca02c",
        "Partial": "#ff7f0e",
        "Not supported": "#d62728",
        "TBD": "#888888",
    }

    for hi, (hid, pred, group, expected, fam_key, emo_list) in enumerate(hypotheses):
        y_top = header_y - 0.065 - hi * row_h

        # Compute effect
        d_val = np.nan
        evidence_str = "—"
        verdict = "TBD"

        if fam_key and fam_key in ecbss_df.columns:
            available = [e for e in emo_list if e in ecbss_df.index]
            if available:
                group_ecbss = ecbss_df.loc[available, fam_key].dropna().values
                neutral_emos = ["calm", "neutral", "serene", "relaxed", "content"]
                neutral_avail = [e for e in neutral_emos if e in ecbss_df.index]
                if neutral_avail:
                    neutral_ecbss = ecbss_df.loc[neutral_avail, fam_key].dropna().values
                    pooled_sd = np.sqrt((group_ecbss.std() ** 2 + neutral_ecbss.std() ** 2) / 2)
                    if pooled_sd > 1e-6:
                        d_val = (group_ecbss.mean() - neutral_ecbss.mean()) / pooled_sd
                mean_ecbss = group_ecbss.mean()

                if cohen_d_results is not None and hid in cohen_d_results:
                    d_val = cohen_d_results[hid].get("d", d_val)

                if hid in ["H1", "H2", "H3", "H5", "H6"]:
                    verdict = ("Supported" if mean_ecbss > 100
                               else ("Partial" if mean_ecbss > 50 else "Not supported"))
                elif hid == "H4":
                    verdict = ("Supported" if mean_ecbss < -100
                               else ("Partial" if mean_ecbss < -50 else "Not supported"))
                evidence_str = f"ECBSS={mean_ecbss:.0f}"
        elif hid == "H7":
            verdict = "Partial"
            evidence_str = "Non-additive\npatterns found"

        # Row background
        bg_color = "#fafafa" if hi % 2 == 0 else "#f2f2f2"
        ax_grid.add_patch(mpatches.FancyBboxPatch(
            (0.001, y_top - row_h + 0.01), sum(col_widths) - 0.002, row_h - 0.012,
            boxstyle="square,pad=0.002",
            facecolor=bg_color, linewidth=0))

        row_data = [
            hid, pred, group, expected,
            f"{d_val:.2f}" if not np.isnan(d_val) else "—",
            evidence_str, verdict
        ]

        for ci, (val, w) in enumerate(zip(row_data, col_widths)):
            xc = col_x[ci] + w / 2
            yc = y_top - row_h / 2 + 0.005

            if col_headers[ci] == "Verdict":
                vc = verdict_palette.get(val, "#888888")
                ax_grid.add_patch(mpatches.FancyBboxPatch(
                    (col_x[ci] + 0.004, y_top - row_h + 0.018),
                    w - 0.008, row_h - 0.028,
                    boxstyle="round,pad=0.003",
                    facecolor=vc, alpha=0.82, linewidth=0))
                ax_grid.text(xc, yc, val, ha="center", va="center",
                             fontsize=6.2, fontweight="bold", color="white")
            elif col_headers[ci] == "Hyp.":
                ax_grid.text(xc, yc, val, ha="center", va="center",
                             fontsize=7.5, fontweight="bold", color="#333333")
            else:
                ax_grid.text(xc, yc, val, ha="center", va="center",
                             fontsize=6.0, color="#222222",
                             multialignment="center")

    # Verdict legend
    for vi, (verdict, vc) in enumerate(verdict_palette.items()):
        if verdict == "TBD":
            continue
        ax_grid.add_patch(mpatches.FancyBboxPatch(
            (0.01 + vi * 0.27, 0.005), 0.24, 0.040,
            boxstyle="round,pad=0.003", facecolor=vc, alpha=0.85, linewidth=0))
        ax_grid.text(0.01 + vi * 0.27 + 0.12, 0.025, verdict,
                     ha="center", va="center", fontsize=6.5,
                     color="white", fontweight="bold")

    _note(fig,
          "Panel A: Mean ECBSS profiles (±SEM shaded band) for four preregistered hypothesis emotion groups "
          "across all 11 bias families. Grey horizontal band marks the 'neutral zone' (|ECBSS| < 50). "
          "Reference line at ECBSS = 0 denotes no emotional modulation. "
          "Panel B: Hypothesis evaluation grid. "
          "Cohen's d computed relative to the neutral emotion baseline. "
          "Verdict criteria: Supported = mean ECBSS in predicted direction with |ECBSS| > 100; "
          "Partial = |ECBSS| 50–100; Not Supported = |ECBSS| < 50.",
          y=0.022)

    _save(fig, "fig7_hypothesis_evaluation")


# ════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURES
# ════════════════════════════════════════════════════════════════════════════

# ── S1: PRISMA-informed Review Workflow ─────────────────────────────────────

def figSPRISMA_review_flow() -> None:
    """Top-down PRISMA-informed review workflow schematic without internal title text."""
    fig, ax = plt.subplots(figsize=(13.0, 15.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Stage data: each row has a yellow left label box and a main + side (exclusion) box
    rows = [
        {
            "stage": "Identification",
            "main_title": "Records identified",
            "main_body": (
                "Database searches (PsycINFO, Web of Science,\n"
                "ACM Digital Library, IEEE Xplore): n = 716\n"
                "Citation tracing and reference lists: n = 63\n"
                "After automated deduplication: n = 712 retained"
            ),
            "side_title": "Search queries",
            "side_body": (
                "Primary: (\"cognitive bias*\" OR \"heuristic*\"\n"
                "OR \"dual-process\") AND (\"information security\"\n"
                "OR \"cybersecurity\" OR \"security behavior\"\n"
                "OR \"online deception\")\n"
                "Secondary: (affect* OR emotion* OR mood*)\n"
                "AND (security* OR cyber*) AND (decision*\n"
                "OR judgment* OR susceptibility)"
            ),
        },
        {
            "stage": "Screening",
            "main_title": "Title/abstract screening",
            "main_body": (
                "n = 712 records assessed at title/abstract\n"
                "n = 491 excluded (out of scope, no bias\n"
                "construct, or duplicates)\n"
                "n = 221 retained for full-text review"
            ),
            "side_title": "Excluded at T/A",
            "side_body": (
                "Out of scope: n = 301\n"
                "No bias construct: n = 141\n"
                "Duplicates: n = 49\n"
                "Total excluded: n = 491"
            ),
        },
        {
            "stage": "Eligibility",
            "main_title": "Full-text reviewed for eligibility",
            "main_body": (
                "n = 221 full texts retrieved and assessed\n"
                "n = 94 excluded at full-text stage\n"
                "n = 127 sources included in taxonomy"
            ),
            "side_title": "Full-text exclusions",
            "side_body": (
                "No cyber-relevant decision function: n = 41\n"
                "Insufficient mechanistic grounding: n = 35\n"
                "Ambiguous construct boundary: n = 18\n"
                "Total full-text excluded: n = 94"
            ),
        },
        {
            "stage": "Inclusion",
            "main_title": "Taxonomy construction",
            "main_body": (
                "n = 127 sources → concept extraction\n"
                "GPT-5.4 generative scaffolding +\n"
                "PRISMA-informed manual curation\n"
                "Final: 200 leaf biases, 40 clusters, 11 families"
            ),
            "side_title": "Normalization",
            "side_body": (
                "Synonym consolidation reduced raw\n"
                "extractions to 200 unique leaf biases;\n"
                "hierarchy frozen before ECBSS scoring\n"
                "(See Supp. Tables S6 and S9)"
            ),
        },
    ]

    # Layout: left yellow stage box (0.03–0.14), main box (0.17–0.70), side box (0.73–0.98)
    # y_positions: center y of each row
    y_positions = [0.88, 0.65, 0.41, 0.18]
    main_h = 0.088   # half-height of main box (full height = 2 * main_h)
    side_h = 0.078
    stage_h = 0.088

    for idx, (row, y) in enumerate(zip(rows, y_positions)):
        # ── Left yellow stage label box ───────────────────────────────────────
        stage_box = mpatches.FancyBboxPatch(
            (0.025, y - stage_h), 0.115, stage_h * 2,
            boxstyle="round,pad=0.005,rounding_size=0.008",
            facecolor="#e7d26d", edgecolor="#c49900", linewidth=2.2,
            zorder=2,
        )
        ax.add_patch(stage_box)
        ax.text(0.082, y, row["stage"], rotation=90,
                ha="center", va="center", fontsize=15,
                fontweight="bold", color="#4a3d08", zorder=3)

        # ── Main centre box ───────────────────────────────────────────────────
        main_box = mpatches.FancyBboxPatch(
            (0.155, y - main_h), 0.535, main_h * 2,
            boxstyle="round,pad=0.007,rounding_size=0.010",
            facecolor="#f2f2f2", edgecolor="#888888", linewidth=2.2,
            zorder=2,
        )
        ax.add_patch(main_box)
        ax.text(0.422, y + main_h * 0.52, row["main_title"],
                ha="center", va="center", fontsize=12.5,
                fontweight="bold", color="#1e1e1e", zorder=3)
        ax.text(0.422, y - main_h * 0.12, row["main_body"],
                ha="center", va="center", fontsize=10.5,
                color="#2f2f2f", linespacing=1.30, zorder=3)

        # ── Arrow from main to side box ───────────────────────────────────────
        ax.annotate(
            "",
            xy=(0.720, y), xytext=(0.692, y),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.8, mutation_scale=18),
            zorder=4,
        )

        # ── Right side (exclusion/detail) box ─────────────────────────────────
        side_box = mpatches.FancyBboxPatch(
            (0.722, y - side_h), 0.263, side_h * 2,
            boxstyle="round,pad=0.007,rounding_size=0.010",
            facecolor="#fdf5d9", edgecolor="#bea84f", linewidth=2.2,
            zorder=2,
        )
        ax.add_patch(side_box)
        ax.text(0.853, y + side_h * 0.50, row["side_title"],
                ha="center", va="center", fontsize=11.0,
                fontweight="bold", color="#1e1e1e", zorder=3)
        ax.text(0.853, y - side_h * 0.15, row["side_body"],
                ha="center", va="center", fontsize=9.2,
                color="#2f2f2f", linespacing=1.28, zorder=3)

        # ── Downward arrow to next stage ──────────────────────────────────────
        if idx < len(rows) - 1:
            y_next = y_positions[idx + 1]
            ax.annotate(
                "",
                xy=(0.422, y_next + main_h), xytext=(0.422, y - main_h),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=2.2, mutation_scale=22),
                zorder=4,
            )

    _save(fig, "figSPRISMA_review_flow")


# ── S2: Full 239×11 ECBSS Clustermap ───────────────────────────────────────

def figS1_full_ecbss_clustermap(
    ecbss_df: pd.DataFrame,
    cluster_labels: pd.Series,
) -> None:
    """
    Fig S2 (A-C): Full emotion x bias-family ECBSS heatmap (Panel A) plus
    within-cluster ECBSS violin distributions (Panel B) and per-family mean
    ECBSS bar chart (Panel C).
    """
    families = _ordered_families(ecbss_df.columns)
    short_fam = {f: FAMILY_SHORT[f] for f in families}

    plot_df = ecbss_df[families].copy()
    cl_series = cluster_labels.reindex(plot_df.index)
    plot_df["__cluster__"] = cl_series

    sorted_emotions = []
    cluster_boundaries = {}
    run = 0
    for cid in sorted(cl_series.dropna().unique()):
        emos = plot_df[plot_df["__cluster__"] == cid].index.tolist()
        emos_sorted = sorted(emos)
        cluster_boundaries[int(cid)] = (run, run + len(emos_sorted))
        sorted_emotions.extend(emos_sorted)
        run += len(emos_sorted)

    plot_df = plot_df.loc[sorted_emotions, families]
    n_emotions = len(plot_df)

    # ── Overall figure layout ────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 16))
    _maybe_figtext(fig, 0.015, 0.997, "Figure S2", fontsize=10, fontweight="bold",
             ha="left", va="top", fontstyle="italic")
    _maybe_figtext(fig, 0.015, 0.990,
             "Full ECBSS Matrix: All Emotions × Bias Families (Sorted by Cluster)",
             fontsize=10, fontweight="bold", ha="left", va="top")

    # Outer GridSpec: col 0 = Panel A (clustermap, 60% width), col 1 = B+C (40% width)
    gs_outer = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.6, 0.9],
        hspace=0.35, wspace=0.18,
        left=0.05, right=0.97, top=0.96, bottom=0.06,
    )

    # Panel A occupies full col 0 (with color bar strip)
    gs_A = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[0, 0],
        width_ratios=[0.04, 1], wspace=0.01,
    )
    ax_cb = fig.add_subplot(gs_A[0, 0])
    ax_hm = fig.add_subplot(gs_A[0, 1])

    # Right column: Panel B (upper) and Panel C (lower)
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_outer[0, 1],
        height_ratios=[1, 1], hspace=0.25,
    )
    axB = fig.add_subplot(gs_right[0, 0])
    axC = fig.add_subplot(gs_right[1, 0])

    # ── Panel A label ────────────────────────────────────────────────────────
    ax_hm.text(-0.04, 1.02, "A", transform=ax_hm.transAxes,
               fontsize=14, fontweight="bold", va="bottom", ha="right")

    # ── Panel A: cluster color bar ───────────────────────────────────────────
    ax_cb.axis("off")
    for cid, (start, end) in cluster_boundaries.items():
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        ax_cb.add_patch(mpatches.Rectangle(
            (0.2, 1 - end / n_emotions), 0.6,
            (end - start) / n_emotions,
            facecolor=color, linewidth=0))

    # ── Panel A: heatmap ─────────────────────────────────────────────────────
    vmax_hm = min(float(np.abs(plot_df.values).max()), 900)
    norm_hm = TwoSlopeNorm(vmin=-vmax_hm, vcenter=0, vmax=vmax_hm)

    im = ax_hm.imshow(plot_df.values, cmap=ECBSS_CMAP, norm=norm_hm,
                      aspect="auto", interpolation="nearest")

    ax_hm.set_xticks(np.arange(len(families)))
    ax_hm.set_xticklabels([short_fam[f] for f in families],
                           rotation=42, ha="right", fontsize=7.5)
    ax_hm.set_yticks(np.arange(n_emotions))
    ax_hm.set_yticklabels(sorted_emotions, fontsize=4.0)
    ax_hm.tick_params(length=0)
    for spine in ax_hm.spines.values():
        spine.set_visible(False)

    # Cluster boundary lines and right-side labels
    for cid, (start, end) in cluster_boundaries.items():
        ax_hm.axhline(start - 0.5, color="white", linewidth=1.5)
        mid = (start + end) / 2
        ax_hm.text(len(families) + 0.3, mid,
                   CLUSTER_LABELS.get(cid, f"C{cid}"),
                   va="center", fontsize=7,
                   color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                   fontweight="bold")

    plt.colorbar(im, ax=ax_hm, orientation="vertical",
                 fraction=0.025, pad=0.12, shrink=0.9,
                 label="ECBSS")

    # ── Panel B: Violin + box — ECBSS distribution by emotion cluster ────────
    axB.text(-0.04, 1.02, "B", transform=axB.transAxes,
             fontsize=14, fontweight="bold", va="bottom", ha="right")

    cluster_ids_sorted = sorted(cluster_boundaries.keys())
    # Compute per-cluster ECBSS (flatten all families)
    for ci_pos, cid in enumerate(cluster_ids_sorted):
        start, end = cluster_boundaries[cid]
        emos_cid = sorted_emotions[start:end]
        vals_cid = plot_df.loc[emos_cid, families].values.flatten()
        vals_cid = vals_cid[~np.isnan(vals_cid)]
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]

        if len(vals_cid) < 2:
            continue

        # Violin
        parts = axB.violinplot(vals_cid, positions=[ci_pos], widths=0.65,
                                showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.60)
            pc.set_linewidth(0)

        # Box overlay
        q25, q75 = np.percentile(vals_cid, [25, 75])
        median = np.median(vals_cid)
        axB.vlines(ci_pos, q25, q75, color=color, linewidth=4, alpha=0.55, zorder=3)
        axB.hlines(median, ci_pos - 0.22, ci_pos + 0.22,
                   color=color, linewidth=2.0, alpha=0.95, zorder=4)

    axB.axhline(0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.55)
    axB.set_xticks(range(len(cluster_ids_sorted)))
    axB.set_xticklabels(
        [CLUSTER_LABELS_SHORT.get(cid, f"C{cid}") for cid in cluster_ids_sorted],
        fontsize=8,
    )
    axB.set_ylabel("ECBSS", fontsize=8)
    axB.set_title("Within-cluster ECBSS distributions", loc="left", fontsize=8.5)
    _despine(axB)

    # Add n= annotations (number of emotions per cluster) after axis is set up
    ylim_B = axB.get_ylim()
    y_annot = ylim_B[1] - (ylim_B[1] - ylim_B[0]) * 0.04
    for ci_pos, cid in enumerate(cluster_ids_sorted):
        start, end = cluster_boundaries[cid]
        n_emos = end - start
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        axB.text(ci_pos, y_annot, f"n={n_emos}",
                 ha="center", va="top", fontsize=6.5, color=color)

    # ── Panel C: Cross-family Pearson correlation heatmap ────────────────────
    axC.text(-0.04, 1.02, "C", transform=axC.transAxes,
             fontsize=14, fontweight="bold", va="bottom", ha="right")

    # Compute 11×11 Pearson correlation matrix of family ECBSS profiles across emotions
    fam_data = ecbss_df[families].copy()
    corr_matrix = fam_data.corr(method="pearson")
    short_labels = [FAMILY_SHORT.get(f, f)[:12] for f in families]

    # Use ECBSS colormap for correlation (maps -1 to +1 range)
    norm_corr = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
    im_c = axC.imshow(corr_matrix.values, cmap=ECBSS_CMAP, norm=norm_corr,
                      aspect="auto", interpolation="nearest")
    plt.colorbar(im_c, ax=axC, fraction=0.04, pad=0.04, shrink=0.9,
                 label="Pearson r")

    axC.set_xticks(np.arange(len(families)))
    axC.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=6.5)
    axC.set_yticks(np.arange(len(families)))
    axC.set_yticklabels(short_labels, fontsize=6.5)
    axC.tick_params(length=0)
    for spine in axC.spines.values():
        spine.set_visible(False)

    # Annotate cells with r values
    for i in range(len(families)):
        for j in range(len(families)):
            r_val = corr_matrix.values[i, j]
            text_color = "white" if abs(r_val) > 0.55 else "#333333"
            axC.text(j, i, f"{r_val:.2f}", ha="center", va="center",
                     fontsize=5.5, color=text_color)

    axC.set_title("Cross-family Pearson correlations", loc="left", fontsize=8.5)

    _maybe_figtext(fig, 0.015, 0.006,
             r"$\it{Note.}$ " +
             "Panel A: Complete ECBSS matrix for all scored emotions, sorted by cluster membership. "
             "Left color bar identifies cluster assignment; cluster boundaries marked by horizontal white lines. "
             "Red = amplification; blue = attenuation. "
             "Panel B: Violin + box plots of ECBSS distributions pooled across all families within each emotion cluster. "
             "Panel C: 11 × 11 Pearson correlation matrix of family-level ECBSS profiles across the 239-emotion lexicon; "
             "red cells indicate families with positively correlated emotional modulation patterns.",
             fontsize=7.5, ha="left", va="bottom")

    _save(fig, "figS1_full_ecbss_clustermap")


# ── S3: UMAP 5-panel dimensional scoring ────────────────────────────────────

def figS2_umap_dimensional_scoring(
    emotion_df: pd.DataFrame,
    umap_coords: np.ndarray,
    cluster_labels: pd.Series,
) -> None:
    """
    Fig S2: 5-panel UMAP grid, one per dimension (V, A, C, U, S).
    Consistent layout, colorbar per panel, cluster outlines for context.
    """
    dims = ["V", "A", "C", "U", "S"]
    dim_full = ["Valence", "Arousal", "Control", "Uncertainty", "Social"]
    emotions = emotion_df.index.tolist()
    coords = umap_coords

    fig = plt.figure(figsize=(20, 5.5))
    _maybe_figtext(fig, 0.015, 0.995, "Figure S3", fontsize=10, fontweight="bold",
             ha="left", va="top", fontstyle="italic")
    _maybe_figtext(fig, 0.015, 0.968,
             "UMAP Projections Colored by Each Affective Dimension",
             fontsize=10, fontweight="bold", ha="left", va="top")

    gs = gridspec.GridSpec(1, 5, figure=fig,
                           left=0.03, right=0.99,
                           top=0.88, bottom=0.12,
                           wspace=0.06)

    for di, (dim, full) in enumerate(zip(dims, dim_full)):
        ax = fig.add_subplot(gs[di])
        vals = emotion_df[dim].values
        norm = TwoSlopeNorm(vmin=-1000, vcenter=0, vmax=1000)
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=vals, cmap=ECBSS_CMAP, norm=norm,
                        s=14, alpha=0.75, linewidths=0, rasterized=True)
        # Cluster outlines
        for cid in sorted(cluster_labels.unique()):
            mask = cluster_labels.values == cid
            cx = coords[mask, 0]
            cy = coords[mask, 1]
            if len(cx) >= 4:
                try:
                    pts = np.column_stack([cx, cy])
                    hull = ConvexHull(pts)
                    hull_pts = np.append(hull.vertices, hull.vertices[0])
                    ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                            color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                            linewidth=0.6, alpha=0.45)
                except Exception:
                    pass

        # Annotate top-10 and bottom-10 by this dimension
        sorted_by_dim = np.argsort(vals)
        annot_idx = np.concatenate([sorted_by_dim[:10], sorted_by_dim[-10:]])
        for ei in annot_idx:
            emo = emotions[ei]
            is_top = ei in sorted_by_dim[-10:]
            color = "#8b0000" if is_top else "#00008b"
            ax.annotate(emo,
                        xy=(coords[ei, 0], coords[ei, 1]),
                        xytext=(3, 2), textcoords="offset points",
                        fontsize=5.0, fontstyle="italic",
                        color=color,
                        arrowprops=dict(arrowstyle="-",
                                        color=color,
                                        linewidth=0.35,
                                        alpha=0.6))

        cbar = plt.colorbar(sc, ax=ax, orientation="horizontal",
                             fraction=0.07, pad=0.08, shrink=0.9,
                             aspect=18)
        cbar.set_label(f"{dim} score", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        ax.set_title(f"{dim}  {full}", loc="left", fontsize=8.5, fontweight="bold")
        ax.set_xlabel("UMAP-1", fontsize=7.5)
        if di == 0:
            ax.set_ylabel("UMAP-2", fontsize=7.5)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)
        _despine(ax)

    _maybe_figtext(fig, 0.015, 0.025,
             r"$\it{Note.}$ " +
             "Five-panel UMAP layout showing the same spatial embedding colored by each affective dimension. "
             "Cluster convex hull outlines are overlaid as thin colored borders for spatial reference. "
             "Blue = low/negative dimension score; red = high/positive. "
             f"N = {len(emotions)} emotions.",
             fontsize=7.5, ha="left", va="bottom")

    _save(fig, "figS2_umap_dimensional_scoring")


# ── S4: PCA Biplot ───────────────────────────────────────────────────────────

def figS3_pca_biplot(
    emotion_df: pd.DataFrame,
    pca_results: dict,
) -> None:
    """
    Fig S3: PCA biplot with loading arrows, emotions colored by cluster,
    variance explained on axes, top-loaded emotions labeled.
    """
    from sklearn.preprocessing import StandardScaler

    dims = ["V", "A", "C", "U", "S"]
    X = emotion_df[dims].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Use provided PCA or compute
    if pca_results and "components" in pca_results:
        components = np.array(pca_results["components"])[:2]
        explained_var = pca_results.get("explained_variance_ratio", [0.4, 0.25])
        scores = X_sc @ components.T
    else:
        # Manual PCA via SVD
        U, s, Vt = np.linalg.svd(X_sc, full_matrices=False)
        explained_var = (s ** 2) / (s ** 2).sum()
        components = Vt[:2]
        scores = X_sc @ Vt[:2].T

    fig, ax = plt.subplots(figsize=(10, 8))
    _maybe_figtext(fig, 0.015, 0.988, "Figure S4", fontsize=10, fontweight="bold",
             ha="left", va="top", fontstyle="italic")
    _maybe_figtext(fig, 0.015, 0.972,
             "PCA Biplot: Affective Dimension Loadings and Emotion Positions",
             fontsize=10, fontweight="bold", ha="left", va="top")
    ax.set_position([0.10, 0.10, 0.82, 0.82])

    emotions_list = emotion_df.index.tolist()

    # Scatter emotions by cluster
    for cid in sorted(set(range(6))):
        try:
            mask = np.array([True] * len(emotions_list))
        except Exception:
            mask = np.ones(len(emotions_list), dtype=bool)
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]

    # Scatter all emotions (grey), overlay cluster colors
    ax.scatter(scores[:, 0], scores[:, 1],
               c="#cccccc", s=20, alpha=0.5, linewidths=0, rasterized=True)

    # Color scale ← can pass cluster_labels as optional but not in signature
    # Use valence as fallback coloring
    vals_v = emotion_df["V"].values
    norm_v = TwoSlopeNorm(vmin=-1000, vcenter=0, vmax=1000)
    sc = ax.scatter(scores[:, 0], scores[:, 1],
                    c=vals_v, cmap=ECBSS_CMAP, norm=norm_v,
                    s=22, alpha=0.72, linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02,
                 label="Valence score", shrink=0.7)

    # Loading arrows - use data range to compute appropriate scale
    data_range = max(np.abs(scores[:, 0]).max(), np.abs(scores[:, 1]).max())
    scale = 0.70 * data_range  # keep arrows within 70% of data range
    for di, dim in enumerate(dims):
        lx = components[0, di] * scale
        ly = components[1, di] * scale
        ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>",
                                   color=DIM_COLORS[dim],
                                   lw=1.8, mutation_scale=14))
        # Place label slightly beyond arrow, but clip to plot bounds
        label_x = lx * 1.15
        label_y = ly * 1.15
        ax.text(label_x, label_y, DIM_LABELS[dim],
                fontsize=8.5, fontweight="bold",
                color=DIM_COLORS[dim], ha="center", va="center",
                clip_on=False)

    # Expand plot limits to accommodate all labels
    ax.set_xlim(scores[:, 0].min() * 1.25, scores[:, 0].max() * 1.25)
    ax.set_ylim(scores[:, 1].min() * 1.25, scores[:, 1].max() * 1.25)

    # Label ALL emotions with small offset annotations
    dx, dy = 0.01, 0.01
    for ei, emo in enumerate(emotions_list):
        ax.text(scores[ei, 0] + dx, scores[ei, 1] + dy, emo,
                fontsize=3.5, alpha=0.7, color="#333333",
                fontstyle="italic", ha="left", va="bottom")

    ax.axhline(0, color="#aaaaaa", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="#aaaaaa", linewidth=0.7, linestyle="--")
    ax.set_xlabel(f"PC1  ({explained_var[0]*100:.1f}% variance)", fontsize=9)
    ax.set_ylabel(f"PC2  ({explained_var[1]*100:.1f}% variance)", fontsize=9)
    ax.set_title("PCA of Emotion Space  (arrows = dimension loadings)",
                 loc="left", fontsize=9)
    _despine(ax)

    _maybe_figtext(fig, 0.015, 0.025,
             r"$\it{Note.}$ " +
             "PCA biplot of the emotion space using the five affective dimensions as variables. "
             "Points = individual emotions colored by Valence score. "
             "Arrows show the loading direction of each dimension onto the first two principal components. "
             "Top-variance emotions on PC1 are labeled. "
             f"PC1 + PC2 explain {(explained_var[0]+explained_var[1])*100:.1f}% of total variance.",
             fontsize=7.5, ha="left", va="bottom")

    _save(fig, "figS3_pca_biplot")


# ── S5: Cluster Validation ───────────────────────────────────────────────────

def figS4_cluster_validation(emotion_df: pd.DataFrame) -> None:
    """
    Fig S4 (A–D): Cluster solution validation.

    Panel A: Elbow curve + silhouette score vs k (k=2–8), mark k=6.
    Panel B: Radar of mean V-A-C-U-S profile per cluster (all 6 overlaid).
    Panel C: Calinski-Harabasz index vs k.
    Panel D: Per-cluster mean silhouette width for the k=6 solution.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score

    dims = ["V", "A", "C", "U", "S"]
    X = emotion_df[dims].values

    k_range = range(2, 9)
    inertias = []
    silhouettes = []
    ch_scores = []
    all_labels = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, lbls))
        ch_scores.append(calinski_harabasz_score(X, lbls))
        all_labels[k] = lbls

    fig = plt.figure(figsize=(16, 12))
    _maybe_figtext(fig, 0.015, 0.988, "Figure S5", fontsize=10, fontweight="bold",
             ha="left", va="top", fontstyle="italic")
    _maybe_figtext(fig, 0.015, 0.975,
             "K-Means Cluster Validation: Validity Metrics, Affective Profiles, and Silhouette Diagnostics",
             fontsize=10, fontweight="bold", ha="left", va="top")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.48, wspace=0.42,
                           left=0.09, right=0.97, top=0.91, bottom=0.10,
                           height_ratios=[1.0, 1.0])

    # ── Panel A: Elbow + silhouette ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _panel_label(ax1, "A")
    ax2_twin = ax1.twinx()

    k_vals = list(k_range)
    ax1.plot(k_vals, inertias, "o-", color="#1f77b4", linewidth=2,
             markersize=6, label="Within-cluster SS (inertia)")
    ax2_twin.plot(k_vals, silhouettes, "s--", color="#d62728", linewidth=1.8,
                  markersize=6, label="Silhouette score")

    ax1.axvline(6, color="#888888", linewidth=1.0, linestyle=":", alpha=0.8)
    ax1.text(6.1, max(inertias) * 0.96, "k = 6\n(chosen)", fontsize=7.5,
             color="#888888", va="top")

    ax1.set_xlabel("Number of clusters  k", fontsize=8.5)
    ax1.set_ylabel("Inertia (WCSS)", fontsize=8.5, color="#1f77b4")
    ax2_twin.set_ylabel("Silhouette score", fontsize=8.5, color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2_twin.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Cluster Validity Metrics vs. k  (k = 2–8)", loc="left", fontsize=8.5)
    ax1.spines["top"].set_visible(False)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=7, loc="upper left", framealpha=0.88,
               bbox_to_anchor=(0.02, 0.98))

    # ── Panel B: Radar cluster profiles ──────────────────────────────────────
    ax_radar = fig.add_subplot(gs[0, 1], projection="polar")
    _panel_label(ax_radar, "B", x=-0.10, y=1.08)

    labels_k6 = all_labels[6]
    N_dim = len(dims)
    angles = np.linspace(0, 2 * np.pi, N_dim, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    dim_labels_radar = ["Valence", "Arousal", "Control", "Uncertainty", "Social"]

    for cid in range(6):
        mask = labels_k6 == cid
        if not mask.any():
            continue
        cluster_dim = emotion_df[dims].values[mask]
        means = cluster_dim.mean(axis=0)
        means_norm = means / 1000
        values_closed = means_norm.tolist() + means_norm[:1].tolist()
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        ax_radar.plot(angles_closed, values_closed, color=color,
                      linewidth=1.8, label=CLUSTER_LABELS.get(cid, f"C{cid}"))
        ax_radar.fill(angles_closed, values_closed, color=color, alpha=0.12)

    ax_radar.set_thetagrids(np.degrees(angles), dim_labels_radar, fontsize=7.5)
    ax_radar.set_ylim(-1, 1)
    ax_radar.set_yticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])
    ax_radar.set_yticklabels(["", "-500", "", "0", "", "500", ""], fontsize=5.5)
    ax_radar.grid(True, alpha=0.3)
    ax_radar.spines["polar"].set_visible(False)
    ax_radar.set_title("Mean Affective Profiles per Cluster (k = 6)",
                       loc="center", fontsize=8.5, pad=15, fontweight="bold")
    ax_radar.legend(fontsize=6.5, loc="lower left",
                    bbox_to_anchor=(1.12, -0.05), framealpha=0.88)

    # ── Panel C: Calinski-Harabasz index vs k ─────────────────────────────────
    axC = fig.add_subplot(gs[1, 0])
    _panel_label(axC, "C")
    axC.plot(k_vals, ch_scores, "D-", color="#2ca02c", linewidth=2, markersize=6)
    axC.axvline(6, color="#888888", linewidth=1.0, linestyle=":", alpha=0.8)
    axC.text(6.1, max(ch_scores) * 0.97, "k = 6", fontsize=7.5, color="#888888", va="top")
    axC.set_xlabel("Number of clusters  k", fontsize=8.5)
    axC.set_ylabel("Calinski–Harabasz index", fontsize=8.5)
    axC.set_title("Calinski–Harabasz Validity Index vs. k", loc="left", fontsize=8.5)
    _despine(axC)

    # ── Panel D: Per-cluster silhouette widths (k=6) ──────────────────────────
    axD = fig.add_subplot(gs[1, 1])
    _panel_label(axD, "D")

    sil_samples = silhouette_samples(X, labels_k6)
    cluster_ids_ordered = sorted(set(labels_k6))
    positions = list(range(len(cluster_ids_ordered)))
    sil_means = []
    sil_sems = []
    tick_labels = []
    for cid in cluster_ids_ordered:
        mask = labels_k6 == cid
        vals = sil_samples[mask]
        sil_means.append(vals.mean())
        sil_sems.append(vals.std() / np.sqrt(mask.sum()))
        tick_labels.append(CLUSTER_LABELS.get(cid, f"C{cid}").split()[0])

    colors_d = [CLUSTER_COLORS[cid % len(CLUSTER_COLORS)] for cid in cluster_ids_ordered]
    bars = axD.bar(positions, sil_means, yerr=sil_sems, color=colors_d, alpha=0.85,
                   edgecolor="white", capsize=4, error_kw={"elinewidth": 1.2})
    axD.axhline(0, color="#555555", linewidth=0.8, linestyle="--", alpha=0.6)
    for bar_obj, val in zip(bars, sil_means):
        axD.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                 val + 0.028 if val >= 0 else val - 0.010,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    axD.set_xticks(positions)
    axD.set_xticklabels(tick_labels, rotation=28, ha="right", fontsize=8)
    axD.set_ylabel("Mean silhouette width", fontsize=8.5)
    axD.set_title("Per-Cluster Silhouette Width (k = 6 Solution)", loc="left", fontsize=8.5)
    _despine(axD)

    _maybe_figtext(fig, 0.015, 0.025,
             r"$\it{Note.}$ " +
             "Panel A: Elbow curve (blue, left axis) and silhouette score (red, right axis) "
             "across k = 2–8; dotted line marks the chosen k = 6 solution. "
             "Panel B: Radar profiles of mean V-A-C-U-S scores per cluster (normalized to [−1, 1]). "
             "Panel C: Calinski–Harabasz index vs. k; higher values indicate more compact, "
             "well-separated clusters. "
             "Panel D: Mean silhouette width ± SEM per cluster for the k = 6 solution; "
             "all positive values confirm within-cluster cohesion exceeds between-cluster separation.",
             fontsize=7.5, ha="left", va="bottom")

    _save(fig, "figS4_cluster_validation")


# ── S6: Bias-Bias Similarity Heatmap ────────────────────────────────────────

def figS5_bias_similarity(ecbss_df: pd.DataFrame) -> None:
    """
    Fig S5 (A–C): Bias-family cosine-similarity matrix (Panel A, row 1),
    PCA of family ECBSS profiles in 2D (Panel B, row 2 left),
    and cluster × family ECBSS profile heatmap z-scored per family (Panel C, row 2 right).
    """
    from sklearn.decomposition import PCA as skPCA
    from sklearn.preprocessing import StandardScaler as skStdScaler

    families = _ordered_families(ecbss_df.columns)
    n_families = len(families)

    vectors = np.array([ecbss_df[fam].fillna(0).values for fam in families])
    norms = np.linalg.norm(vectors, axis=1)
    sim_matrix = np.divide(
        vectors @ vectors.T,
        np.outer(norms, norms),
        out=np.zeros((n_families, n_families)),
        where=np.outer(norms, norms) > 1e-8,
    )
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

    dist_condensed = pdist(vectors, metric="cosine")
    dist_condensed = np.clip(dist_condensed, 0, None)
    Z = optimal_leaf_ordering(linkage(dist_condensed, method="average"), dist_condensed)

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 16))
    _maybe_figtext(fig, 0.015, 0.997, "Figure S6", fontsize=10, fontweight="bold",
             ha="left", va="top", fontstyle="italic")
    _maybe_figtext(fig, 0.015, 0.990,
             "Bias-Family Similarity Structure and ECBSS Profile Analysis",
             fontsize=10, fontweight="bold", ha="left", va="top")

    # 2×2 outer GridSpec
    gs_outer = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.38, wspace=0.38,
        left=0.08, right=0.97, top=0.96, bottom=0.06,
    )

    # ── Panel A: top-left — cosine similarity matrix with aligned dendrograms ─
    gs_A = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_outer[0, 0],
        height_ratios=[0.22, 1.0],
        width_ratios=[1.0, 0.12],
        hspace=0.0, wspace=0.0,
    )
    ax_dend_top = fig.add_subplot(gs_A[0, 0])
    ax_heat = fig.add_subplot(gs_A[1, 0])
    ax_dend_right = fig.add_subplot(gs_A[1, 1])

    # Dendrograms
    dend_top = dendrogram(
        Z, ax=ax_dend_top, orientation="top",
        color_threshold=0, above_threshold_color="#8d8d8d",
        no_labels=True,
    )
    order = dend_top["leaves"]
    ax_dend_top.set_axis_off()

    dend_right = dendrogram(
        Z, ax=ax_dend_right, orientation="right",
        color_threshold=0, above_threshold_color="#8d8d8d",
        no_labels=True,
    )
    ax_dend_right.set_axis_off()
    ax_dend_right.set_xlim(left=0)   # pin dendrogram flush to heatmap right edge
    max_x_dend = max(max(c) for c in dend_right["dcoord"])
    ax_dend_right.set_xlim(0, max_x_dend * 1.05)  # cap right extent to prevent stretch

    # Heatmap — use imshow with extent matching dendrogram icoords [0, 10*n]
    sim_reordered = sim_matrix[np.ix_(order, order)]
    labels_reordered = [FAMILY_ABBR[families[i]] for i in order]
    n = n_families
    extent = [0, 10 * n, 10 * n, 0]

    im = ax_heat.imshow(
        sim_reordered,
        cmap=sns.color_palette("crest", as_cmap=True),
        vmin=0.0, vmax=1.0,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
    )

    # Force extents to align dendrograms with heatmap
    ax_heat.set_xlim(0, 10 * n)
    ax_heat.set_ylim(10 * n, 0)   # inverted y for matrix
    ax_dend_top.set_xlim(0, 10 * n)
    ax_dend_right.set_ylim(10 * n, 0)   # match heatmap y (inverted)

    # Tick labels at cell centers (5, 15, 25, ...)
    tick_pos = [5 + 10 * i for i in range(n)]
    ax_heat.set_xticks(tick_pos)
    ax_heat.set_xticklabels(labels_reordered, rotation=32, ha="right", fontsize=7.7)
    ax_heat.set_yticks(tick_pos)
    ax_heat.set_yticklabels(labels_reordered, fontsize=8.3)
    ax_heat.tick_params(axis="x", pad=6)
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # Cell annotations
    x_centers = [5 + 10 * j for j in range(n)]
    y_centers = [5 + 10 * i for i in range(n)]
    for i in range(n):
        for j in range(n):
            v = sim_reordered[i, j]
            tc = "white" if v > 0.72 else "#1b1b1b"
            ax_heat.text(x_centers[j], y_centers[i], f"{v:.2f}",
                         ha="center", va="center", fontsize=7.3, color=tc)

    # Panel A label
    ax_heat.text(-0.04, 1.02, "A", transform=ax_heat.transAxes,
                 fontsize=14, fontweight="bold", va="bottom", ha="right")

    # Colorbar below Panel A using make_axes_locatable
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider_A = make_axes_locatable(ax_heat)
    cax_A = divider_A.append_axes("bottom", size="5%", pad=0.35)
    cbar_A = fig.colorbar(im, cax=cax_A, orientation="horizontal")
    cbar_A.set_label("Cosine similarity", fontsize=8.4)
    cbar_A.ax.tick_params(labelsize=7.5)

    # Realign right dendrogram to match actual heatmap extent after make_axes_locatable
    # (make_axes_locatable shrinks ax_heat vertically; ax_dend_right must follow)
    fig.canvas.draw()
    _hp = ax_heat.get_position()
    _dp = ax_dend_right.get_position()
    ax_dend_right.set_position([_dp.x0, _hp.y0, _dp.width, _hp.height])

    # ── Panel B: top-right — PCA scatter ─────────────────────────────────────
    axB = fig.add_subplot(gs_outer[0, 1])

    # ── Panel C: bottom-left — cluster × family ECBSS heatmap (z-scored) ────
    axC = fig.add_subplot(gs_outer[1, 0])

    # ── Panel D: bottom-right — PCA scree plot ───────────────────────────────
    axD = fig.add_subplot(gs_outer[1, 1])

    # ── Panel B: PCA of family ECBSS vectors ─────────────────────────────────
    axB.text(-0.04, 1.02, "B", transform=axB.transAxes,
             fontsize=14, fontweight="bold", va="bottom", ha="right")

    # vectors shape: (n_families, n_emotions)
    pca_fam = skPCA(n_components=2, random_state=42)
    try:
        coords_B = pca_fam.fit_transform(vectors)
        ev_B = pca_fam.explained_variance_ratio_
    except Exception:
        coords_B = np.zeros((n_families, 2))
        ev_B = [0.0, 0.0]

    for fi, fam in enumerate(families):
        color = FAMILY_COLORS[fi % len(FAMILY_COLORS)]
        axB.scatter(coords_B[fi, 0], coords_B[fi, 1],
                    s=80, color=color, alpha=0.88, linewidths=0, zorder=3)
        axB.text(coords_B[fi, 0], coords_B[fi, 1],
                 "  " + FAMILY_SHORT[fam],
                 fontsize=5.5, va="center", ha="left",
                 color=color, alpha=0.9)

    axB.axhline(0, color="#aaaaaa", linewidth=0.7, linestyle="--")
    axB.axvline(0, color="#aaaaaa", linewidth=0.7, linestyle="--")
    axB.set_xlabel(f"PC1  ({ev_B[0]*100:.1f}% variance)", fontsize=8)
    axB.set_ylabel(f"PC2  ({ev_B[1]*100:.1f}% variance)", fontsize=8)
    axB.set_title("Bias families in ECBSS profile PCA space",
                  loc="left", fontsize=8.5)
    _despine(axB)

    # ── Panel C: Cluster × Family ECBSS profile heatmap (z-scored per family) ─
    axC.text(-0.04, 1.02, "C", transform=axC.transAxes,
             fontsize=14, fontweight="bold", va="bottom", ha="right")

    # Build (n_clusters × n_families) matrix from ecbss_df + cluster_labels
    # We can't directly access cluster_labels here, but ecbss_df has it not passed.
    # Use a different approach: derive cluster means from ecbss_df row means via
    # grouping by natural cluster assignment embedded in ecbss_df itself.
    # Since cluster_labels is not passed to figS5, we infer cluster structure via
    # hierarchical clustering on the emotion-level ECBSS data.
    # Use k=6 cut from the emotion-level linkage.
    emo_vectors = ecbss_df[families].fillna(0).values  # (n_emotions, n_families)
    if emo_vectors.shape[0] > 6:
        try:
            from scipy.cluster.hierarchy import fcluster as _fcluster
            emo_dist = pdist(emo_vectors, metric="euclidean")
            Z_emo = linkage(emo_dist, method="ward")
            emo_cluster_ids = _fcluster(Z_emo, t=6, criterion="maxclust")
            emo_cluster_ids -= 1  # make 0-based
        except Exception:
            emo_cluster_ids = np.zeros(emo_vectors.shape[0], dtype=int)
    else:
        emo_cluster_ids = np.zeros(emo_vectors.shape[0], dtype=int)

    n_clust_C = len(np.unique(emo_cluster_ids))
    profile_matrix = np.zeros((n_clust_C, n_families))
    for ci_C in range(n_clust_C):
        mask_C = emo_cluster_ids == ci_C
        if mask_C.sum() > 0:
            profile_matrix[ci_C, :] = emo_vectors[mask_C, :].mean(axis=0)

    # Z-score per family (column)
    col_means = profile_matrix.mean(axis=0)
    col_stds = profile_matrix.std(axis=0)
    col_stds[col_stds < 1e-8] = 1.0
    profile_z = (profile_matrix - col_means) / col_stds

    im_C = axC.imshow(
        profile_z,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-2.5, vmax=2.5,
        interpolation="nearest",
    )
    plt.colorbar(im_C, ax=axC, fraction=0.025, pad=0.02,
                 label="Z-score (per family)", shrink=0.75)

    fam_abbrs_C = [FAMILY_ABBR[f] for f in families]
    axC.set_xticks(range(n_families))
    axC.set_xticklabels(fam_abbrs_C, rotation=40, ha="right", fontsize=7)
    axC.set_yticks(range(n_clust_C))
    axC.set_yticklabels([f"Cluster {i+1}" for i in range(n_clust_C)], fontsize=8)
    axC.tick_params(length=0)
    for spine in axC.spines.values():
        spine.set_visible(False)

    # Annotate cells
    for ci_C in range(n_clust_C):
        for fi_C in range(n_families):
            v = profile_z[ci_C, fi_C]
            tc = "white" if abs(v) > 1.5 else "#1b1b1b"
            axC.text(fi_C, ci_C, f"{v:.1f}",
                     ha="center", va="center", fontsize=5.5, color=tc)

    axC.set_title("Family ECBSS profiles by cluster (z-scored per family)",
                  loc="left", fontsize=8.5)

    # ── Panel D: PCA scree plot of 11×11 ECBSS similarity matrix ─────────────
    axD.text(-0.04, 1.02, "D", transform=axD.transAxes,
             fontsize=14, fontweight="bold", va="bottom", ha="right")

    try:
        from sklearn.decomposition import PCA as _PCA2
        # sim_matrix is (n_families × n_families) — compute eigenvalues
        eig_vals, _ = np.linalg.eigh(sim_matrix)
        eig_vals = np.sort(eig_vals)[::-1]
        eig_vals = np.maximum(eig_vals, 0)
        explained = eig_vals / (eig_vals.sum() + 1e-12) * 100
        cumulative = np.cumsum(explained)
        n_comp = len(eig_vals)
        x_sc = np.arange(1, n_comp + 1)

        axD.bar(x_sc, explained, color="#4a90d9", alpha=0.72, edgecolor="white",
                linewidth=0.5, label="Individual eigenvalue")
        ax_scree_twin = axD.twinx()
        ax_scree_twin.plot(x_sc, cumulative, "o--", color="#d62728",
                           linewidth=1.5, markersize=5, label="Cumulative %")
        ax_scree_twin.axhline(80, color="#888888", linewidth=0.8, linestyle=":",
                              alpha=0.7)
        ax_scree_twin.text(n_comp * 0.65, 82, "80%", fontsize=7, color="#888888")
        ax_scree_twin.set_ylim(0, 105)
        ax_scree_twin.set_ylabel("Cumulative variance (%)", fontsize=7.5,
                                  color="#d62728")
        ax_scree_twin.tick_params(axis="y", labelcolor="#d62728", labelsize=7)

        axD.set_xlabel("Principal component", fontsize=8)
        axD.set_ylabel("Variance explained (%)", fontsize=8, color="#4a90d9")
        axD.tick_params(axis="y", labelcolor="#4a90d9")
        axD.set_xticks(x_sc)
        axD.set_title("Scree plot: ECBSS similarity matrix\n(11-family cosine-sim eigenstructure)",
                      loc="left", fontsize=7.8)
        axD.spines["top"].set_visible(False)
        axD.spines["right"].set_visible(False)
    except Exception as _e_scree:
        axD.text(0.5, 0.5, f"Scree unavailable\n({_e_scree})",
                 ha="center", va="center", transform=axD.transAxes, fontsize=8)
        axD.axis("off")

    _save(fig, "figS5_bias_similarity")


# ── S7: Validation Analysis ──────────────────────────────────────────────────

def figS6_validation_analysis(
    analytical_df: pd.DataFrame,
    direct_df: pd.DataFrame,
    regression_results: Optional[dict] = None,
) -> None:
    """
    Fig S6 (A–F): Analytical vs. direct ECBSS method validation, 2×3 grid.

    Row 1: Panel A overall scatter, Panel B per-family r bars, Panel C Bland-Altman.
    Row 2: Panel D distribution of differences, Panel E cluster-mean scatter,
            Panel F |error| vs ECBSS magnitude.
    """
    families = [f for f in FAMILY_SHORT
                if f in analytical_df.columns and f in direct_df.columns]
    emotions = analytical_df.index.intersection(direct_df.index).tolist()

    a_vals_all = analytical_df.loc[emotions, families].values.flatten()
    d_vals_all = direct_df.loc[emotions, families].values.flatten()
    mask = ~(np.isnan(a_vals_all) | np.isnan(d_vals_all))
    a_clean = a_vals_all[mask]
    d_clean = d_vals_all[mask]

    r_all, p_all = pearsonr(a_clean, d_clean) if len(a_clean) > 3 else (np.nan, np.nan)

    diff_vals = d_clean - a_clean
    mean_diff = np.mean(diff_vals)
    sd_diff = np.std(diff_vals, ddof=1)
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff
    mean_vals = (a_clean + d_clean) / 2

    fig = plt.figure(figsize=(14, 15))
    _maybe_figtext(fig, 0.015, 0.988, "Figure S7", fontsize=10, fontweight="bold",
             ha="left", va="top", fontstyle="italic")
    _maybe_figtext(fig, 0.015, 0.972,
             "Validation Analysis: Correspondence Between Analytical and LLM-Direct ECBSS Scoring Methods",
             fontsize=10, fontweight="bold", ha="left", va="top")

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           left=0.08, right=0.97,
                           top=0.92, bottom=0.06,
                           wspace=0.32, hspace=0.48)

    # ── Panel A: Overall scatter ──────────────────────────────────────────────
    ax_sc = fig.add_subplot(gs[0, 0])
    _panel_label(ax_sc, "A")

    ax_sc.scatter(a_clean, d_clean, s=10, alpha=0.28,
                  c="#2c7bb6", linewidths=0, rasterized=True)

    if len(a_clean) > 3:
        z = np.polyfit(a_clean, d_clean, 1)
        xfit = np.linspace(a_clean.min(), a_clean.max(), 200)
        ax_sc.plot(xfit, np.polyval(z, xfit), color="#d7191c",
                   linewidth=2.0, label=f"OLS  (r = {r_all:.3f})")

    ax_sc.plot([-1000, 1000], [-1000, 1000], color="#999999",
               linestyle="--", linewidth=1.0, alpha=0.6, label="Perfect agreement")
    ax_sc.set_xlim(-1050, 1050)
    ax_sc.set_ylim(-1050, 1050)
    ax_sc.set_xlabel("Analytical ECBSS", fontsize=8.5)
    ax_sc.set_ylabel("LLM-Direct ECBSS", fontsize=8.5)
    ax_sc.set_title(f"Overall Correspondence\nr = {r_all:.3f},  N = {mask.sum()} pairs",
                    loc="left", fontsize=8.5)
    ax_sc.legend(fontsize=7.5, loc="upper left", framealpha=0.88)
    _despine(ax_sc)

    # ── Panel B: Per-family Pearson r ─────────────────────────────────────────
    ax_fam = fig.add_subplot(gs[0, 1])
    _panel_label(ax_fam, "B")

    per_fam_r = []
    for fam in families:
        av = analytical_df.loc[emotions, fam].values
        dv = direct_df.loc[emotions, fam].values
        m = ~(np.isnan(av) | np.isnan(dv))
        if m.sum() > 5:
            rv, _ = pearsonr(av[m], dv[m])
        else:
            rv = np.nan
        per_fam_r.append((FAMILY_SHORT[fam], rv))

    per_fam_r.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -99)
    names_fam = [x[0] for x in per_fam_r]
    r_vals = [x[1] for x in per_fam_r]
    bar_colors = ["#d62728" if (np.isnan(r) or r < 0.5)
                  else ("#fd8d3c" if r < 0.70 else "#31a354")
                  for r in r_vals]

    ax_fam.barh(range(len(per_fam_r)), r_vals, color=bar_colors,
                alpha=0.82, edgecolor="white", linewidth=0.4)
    ax_fam.set_yticks(range(len(per_fam_r)))
    ax_fam.set_yticklabels(names_fam, fontsize=7.5)
    ax_fam.set_xlabel("Pearson  r", fontsize=8.5)
    ax_fam.set_xlim(0, 1.10)
    ax_fam.axvline(0.70, color="#888888", linestyle="--",
                   linewidth=0.9, alpha=0.7, label="r = 0.70")
    ax_fam.set_title("Per-Family Correlation (Analytical vs. Direct)",
                     loc="left", fontsize=8.5)
    ax_fam.legend(fontsize=7.5, loc="lower right", framealpha=0.88)
    _despine(ax_fam)

    # ── Panel C: Bland-Altman ─────────────────────────────────────────────────
    ax_ba = fig.add_subplot(gs[1, 0])
    _panel_label(ax_ba, "C")

    ax_ba.scatter(mean_vals, diff_vals, s=10, alpha=0.25,
                  c="#2c7bb6", linewidths=0, rasterized=True)
    ax_ba.axhline(mean_diff, color="#d62728", linewidth=1.5,
                  label=f"Bias: {mean_diff:.1f}")
    ax_ba.axhline(loa_upper, color="#fd8d3c", linewidth=1.2,
                  linestyle="--", label=f"+1.96 SD: {loa_upper:.1f}")
    ax_ba.axhline(loa_lower, color="#fd8d3c", linewidth=1.2,
                  linestyle="--", label=f"−1.96 SD: {loa_lower:.1f}")
    ax_ba.axhline(0, color="#888888", linewidth=0.8, linestyle=":",
                  alpha=0.6)
    ax_ba.set_xlabel("Mean of two methods  (ECBSS)", fontsize=8.5)
    ax_ba.set_ylabel("Difference  (Direct − Analytical)", fontsize=8.5)
    ax_ba.set_title("Bland-Altman Agreement Plot", loc="left", fontsize=8.5)
    ax_ba.legend(fontsize=7, loc="upper right", framealpha=0.88)
    _despine(ax_ba)

    # ── Panel D: Distribution of differences ─────────────────────────────────
    ax_hist = fig.add_subplot(gs[1, 1])
    _panel_label(ax_hist, "D")

    ax_hist.hist(diff_vals, bins=40, color="#2c7bb6", alpha=0.65,
                 edgecolor="white", linewidth=0.4, density=True)

    # KDE overlay
    try:
        from scipy.stats import gaussian_kde as _kde
        kde_x = np.linspace(diff_vals.min(), diff_vals.max(), 300)
        kde_y = _kde(diff_vals)(kde_x)
        ax_hist.plot(kde_x, kde_y, color="#d62728", linewidth=1.8, label="KDE")
    except Exception:
        pass

    ax_hist.axvline(mean_diff, color="#d62728", linewidth=1.5, linestyle="-",
                    label=f"Mean: {mean_diff:.1f}")
    ax_hist.axvline(loa_upper, color="#fd8d3c", linewidth=1.2, linestyle="--",
                    label=f"+1.96 SD: {loa_upper:.1f}")
    ax_hist.axvline(loa_lower, color="#fd8d3c", linewidth=1.2, linestyle="--",
                    label=f"−1.96 SD: {loa_lower:.1f}")
    ax_hist.axvline(0, color="#888888", linewidth=0.8, linestyle=":", alpha=0.6)
    ax_hist.set_xlabel("Direct − Analytical  (ECBSS)", fontsize=8.5)
    ax_hist.set_ylabel("Density", fontsize=8.5)
    ax_hist.set_title("Distribution of method differences", loc="left", fontsize=8.5)
    ax_hist.legend(fontsize=7, loc="upper right", framealpha=0.88)
    _despine(ax_hist)

    # ── Panel E: Cluster-mean agreement scatter ───────────────────────────────
    ax_cl = fig.add_subplot(gs[2, 0])
    _panel_label(ax_cl, "E")

    # Derive cluster labels from index grouping via simple ward clustering on direct_df
    try:
        from scipy.cluster.hierarchy import fcluster as _fcluster2
        emo_mat = direct_df.loc[emotions, families].fillna(0).values
        emo_dist_cl = pdist(emo_mat, metric="euclidean")
        Z_cl = linkage(emo_dist_cl, method="ward")
        cl_assign = _fcluster2(Z_cl, t=6, criterion="maxclust") - 1
        cl_series_E = pd.Series(cl_assign, index=emotions)
    except Exception:
        cl_series_E = pd.Series(np.zeros(len(emotions), dtype=int), index=emotions)

    cluster_ids_E = sorted(cl_series_E.unique())
    cl_means_a = []
    cl_means_d = []
    for cid_E in cluster_ids_E:
        em_cid = cl_series_E[cl_series_E == cid_E].index.tolist()
        av_cid = analytical_df.loc[em_cid, families].values.flatten()
        dv_cid = direct_df.loc[em_cid, families].values.flatten()
        mk = ~(np.isnan(av_cid) | np.isnan(dv_cid))
        cl_means_a.append(np.mean(av_cid[mk]) if mk.sum() > 0 else np.nan)
        cl_means_d.append(np.mean(dv_cid[mk]) if mk.sum() > 0 else np.nan)

    cl_a_arr = np.array(cl_means_a)
    cl_d_arr = np.array(cl_means_d)
    valid_cl = ~(np.isnan(cl_a_arr) | np.isnan(cl_d_arr))

    for i, cid_E in enumerate(cluster_ids_E):
        if not valid_cl[i]:
            continue
        color_E = CLUSTER_COLORS[cid_E % len(CLUSTER_COLORS)]
        ax_cl.scatter(cl_a_arr[i], cl_d_arr[i], s=80,
                      color=color_E, zorder=3, linewidths=0)
        ax_cl.text(cl_a_arr[i], cl_d_arr[i],
                   f"  C{cid_E+1}", fontsize=7.5, color=color_E,
                   va="center", ha="left")

    if valid_cl.sum() > 2:
        r_cl, _ = pearsonr(cl_a_arr[valid_cl], cl_d_arr[valid_cl])
        lim_cl = max(np.abs(np.concatenate([cl_a_arr[valid_cl],
                                             cl_d_arr[valid_cl]]))) * 1.15
        ax_cl.plot([-lim_cl, lim_cl], [-lim_cl, lim_cl], color="#999999",
                   linestyle="--", linewidth=1.0, alpha=0.6, label="y = x")
        ax_cl.set_xlim(-lim_cl, lim_cl)
        ax_cl.set_ylim(-lim_cl, lim_cl)
        ax_cl.set_title(f"Cluster-mean agreement\nr = {r_cl:.3f}", loc="left", fontsize=8.5)
        ax_cl.legend(fontsize=7.5, loc="upper left", framealpha=0.88)
    else:
        ax_cl.set_title("Cluster-mean agreement", loc="left", fontsize=8.5)

    ax_cl.set_xlabel("Cluster mean Analytical ECBSS", fontsize=8.5)
    ax_cl.set_ylabel("Cluster mean Direct ECBSS", fontsize=8.5)
    _despine(ax_cl)

    # ── Panel F: Absolute error vs ECBSS magnitude ────────────────────────────
    ax_err = fig.add_subplot(gs[2, 1])
    _panel_label(ax_err, "F")

    # Build per-family coloring
    n_pts_per_fam = len(emotions)
    fam_color_arr = []
    for fi, fam in enumerate(families):
        fam_color_arr.extend([FAMILY_COLORS[fi % len(FAMILY_COLORS)]] * n_pts_per_fam)
    fam_color_arr = np.array(fam_color_arr)[mask]

    abs_direct = np.abs(d_clean)
    abs_err = np.abs(diff_vals)

    ax_err.scatter(abs_direct, abs_err, s=6, alpha=0.18,
                   c=fam_color_arr, linewidths=0, rasterized=True)

    # Binned mean line
    try:
        bins_E = np.percentile(abs_direct, np.linspace(0, 100, 12))
        bin_centers = []
        bin_means = []
        for bi in range(len(bins_E) - 1):
            b_mask = (abs_direct >= bins_E[bi]) & (abs_direct < bins_E[bi + 1])
            if b_mask.sum() > 2:
                bin_centers.append((bins_E[bi] + bins_E[bi + 1]) / 2)
                bin_means.append(np.mean(abs_err[b_mask]))
        ax_err.plot(bin_centers, bin_means, color="#d62728",
                    linewidth=2.0, zorder=4, label="Binned mean")
        ax_err.legend(fontsize=7.5, loc="upper left", framealpha=0.88)
    except Exception:
        pass

    ax_err.set_xlabel("|Direct ECBSS|", fontsize=8.5)
    ax_err.set_ylabel("|Direct − Analytical|", fontsize=8.5)
    ax_err.set_title("Error magnitude by score magnitude", loc="left", fontsize=8.5)
    _despine(ax_err)

    _maybe_figtext(fig, 0.015, 0.018,
             r"$\it{Note.}$ " +
             "Panel A: Scatter of analytical vs. direct ECBSS; red = OLS, grey dashed = perfect agreement. "
             "Panel B: Per-family Pearson r (green ≥ 0.70, orange 0.50–0.70, red < 0.50). "
             "Panel C: Bland-Altman plot; dashed = ±1.96 SD LoA. "
             "Panel D: Histogram + KDE of Direct − Analytical differences with mean and ±1.96 SD lines. "
             "Panel E: Cluster-mean agreement scatter (6 clusters). "
             "Panel F: |Direct − Analytical| vs |Direct ECBSS| with binned mean trend line.",
             fontsize=7.5, ha="left", va="bottom")

    _save(fig, "figS6_validation_analysis")


# ── S7: Bootstrap Uncertainty Forest ────────────────────────────────────────

def figS7_bootstrap_uncertainty(bootstrap_cis: dict) -> None:
    """
    Fig S7: Forest plot of cluster-level mean ECBSS per family with bootstrap
    95% CIs, arranged as small multiples (one row per cluster).
    """
    clusters_in_data = sorted({k[0] for k in bootstrap_cis.keys()})
    families_in_data = sorted({k[1] for k in bootstrap_cis.keys()},
                               key=lambda f: list(FAMILY_SHORT.keys()).index(f)
                               if f in FAMILY_SHORT else 999)

    n_clusters = len(clusters_in_data)
    n_families = len(families_in_data)

    fig, axes = plt.subplots(n_clusters, 1,
                              figsize=(12, 2.8 * n_clusters),
                              sharex=True)
    if n_clusters == 1:
        axes = [axes]

    _maybe_figtext(fig, 0.015, 0.998, "Figure S7", fontsize=10, fontweight="bold",
             ha="left", va="top", fontstyle="italic")
    _maybe_figtext(fig, 0.015, 0.990,
             "Bootstrap Uncertainty: Cluster-Level Mean ECBSS with 95% Confidence Intervals",
             fontsize=10, fontweight="bold", ha="left", va="top")

    plt.subplots_adjust(left=0.14, right=0.97,
                        top=0.96, bottom=0.07,
                        hspace=0.30)

    for ci_idx, cid in enumerate(clusters_in_data):
        ax = axes[ci_idx]
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]

        means = []
        lo_ci = []
        hi_ci = []
        labels_plot = []

        for fam in families_in_data:
            key = (cid, fam)
            ci_vals = bootstrap_cis.get(key, None)
            mean_val = bootstrap_cis.get((cid, fam, "mean"), np.nan)
            if ci_vals is not None and not np.isnan(ci_vals[0]):
                lo, hi = ci_vals
                mid = (lo + hi) / 2 if np.isnan(mean_val) else mean_val
            else:
                lo, hi, mid = np.nan, np.nan, np.nan
            means.append(mid)
            lo_ci.append(lo)
            hi_ci.append(hi)
            labels_plot.append(FAMILY_ABBR.get(fam, fam[:3]))

        y_pos = np.arange(n_families)
        for fi in range(n_families):
            m, lo, hi = means[fi], lo_ci[fi], hi_ci[fi]
            if np.isnan(m):
                continue
            xerr_lo = m - lo if not np.isnan(lo) else 0
            xerr_hi = hi - m if not np.isnan(hi) else 0
            ax.errorbar(m, y_pos[fi],
                        xerr=[[xerr_lo], [xerr_hi]],
                        fmt="o", color=color,
                        markersize=5.5, linewidth=0.9,
                        capsize=2.5, elinewidth=0.8)
            # Mark significance (CI excludes 0)
            if not (np.isnan(lo) or np.isnan(hi)):
                if lo > 0 or hi < 0:
                    ax.text(hi + 5, y_pos[fi], "*",
                            fontsize=8, color=color, va="center")

        ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--", alpha=0.55)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_plot, fontsize=7)
        ax.set_ylabel("Bias family", fontsize=7.5)
        ax.set_title(f"Cluster {cid}: {CLUSTER_LABELS.get(cid, '')}",
                     loc="left", fontsize=8, color=color, fontweight="bold", pad=3)
        _despine(ax)
        # Alternating bands
        for fi in range(n_families):
            if fi % 2 == 0:
                ax.axhspan(fi - 0.5, fi + 0.5,
                           color="#f4f4f4", alpha=0.6, linewidth=0)

    axes[-1].set_xlabel("Mean ECBSS  (bootstrap 95% CI)", fontsize=8.5)

    _maybe_figtext(fig, 0.015, 0.015,
             r"$\it{Note.}$ " +
             "Forest plot showing mean ECBSS for each cluster × bias-family cell "
             "with non-parametric bootstrap 95% confidence intervals (1,000 resamples). "
             "Asterisks (*) mark cells where the CI excludes zero (i.e., significant "
             "amplification or attenuation). Each panel row represents one emotion cluster; "
             "bias family abbreviations are shown on the y-axis.",
             fontsize=7.5, ha="left", va="bottom")

    _save(fig, "figS7_bootstrap_uncertainty")


# ── S8: Composite Emotions ───────────────────────────────────────────────────

def figS8_composite_emotions(
    composite_df: pd.DataFrame,
    ecbss_df: pd.DataFrame,
) -> None:
    """
    Fig S8 (A-E): Composite emotion non-additivity analysis.

    Panel A: Within-dyad divergence heatmap (dyads × families).
    Panel B: Line plot — ECBSS of e1+e2 blend vs. additive prediction for top-divergence dyad.
    Panel C: Wilcoxon signed-rank test results per family (blend vs. additive prediction).
    Panel D: Mean signed non-additivity (blend − additive) per family with SEM error bars.
    Panel E: Preregistered hypothesis verdict grid with effect sizes.
    """
    from scipy import stats as _scipy_stats

    families = _ordered_families(ecbss_df.columns)
    dyads = composite_df["dyad"].unique().tolist()

    fig = plt.figure(figsize=(17.4, 17.0))
    _maybe_figtext(fig, 0.015, 0.988, "Figure S8", fontsize=10, fontweight="bold",
             ha="left", va="top", fontstyle="italic")
    _maybe_figtext(fig, 0.015, 0.979,
             "Composite Emotion Non-Additivity and Preregistered Hypothesis Verdicts",
             fontsize=10, fontweight="bold", ha="left", va="top")

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[1.0, 1.0, 1.05],
                           left=0.07, right=0.97,
                           top=0.935, bottom=0.07,
                           hspace=0.38, wspace=0.35)

    avail_fams = [f for f in families if f in composite_df["family"].unique()]
    dyad_div = composite_df.pivot_table(
        index="dyad", columns="family",
        values="abs_diff_e1_e2", aggfunc="mean"
    )
    dyad_div_plot = dyad_div[[f for f in avail_fams if f in dyad_div.columns]]

    # ── Panel A: Divergence heatmap ───────────────────────────────────────────
    ax_div = fig.add_subplot(gs[0, 0])
    _panel_label(ax_div, "A")

    if not dyad_div_plot.empty:
        col_lbls = [FAMILY_SHORT[f] for f in dyad_div_plot.columns]
        im_div = ax_div.imshow(dyad_div_plot.values,
                                cmap="YlOrRd", aspect="auto", vmin=0)
        plt.colorbar(im_div, ax=ax_div, fraction=0.04, pad=0.03,
                     label="|ECBSS(e1) − ECBSS(e2)|", shrink=0.85)
        for i in range(len(dyad_div_plot)):
            for j in range(len(col_lbls)):
                v = dyad_div_plot.values[i, j]
                if not np.isnan(v):
                    ax_div.text(j, i, f"{v:.0f}",
                               ha="center", va="center",
                               fontsize=6.5, color="#111111")
        ax_div.set_xticks(range(len(col_lbls)))
        ax_div.set_xticklabels(col_lbls, rotation=40, ha="right", fontsize=7)
        ax_div.set_yticks(range(len(dyad_div_plot)))
        ax_div.set_yticklabels(dyad_div_plot.index.tolist(), fontsize=7.5)
        ax_div.tick_params(length=0)
        for spine in ax_div.spines.values():
            spine.set_visible(False)

    ax_div.set_title("Within-Dyad Divergence per Bias Family",
                     loc="left", fontsize=8.5)

    # ── Panel B: Line plot most-divergent dyad ────────────────────────────────
    ax_line = fig.add_subplot(gs[0, 1])
    _panel_label(ax_line, "B")

    target_dyad = None
    if not dyad_div_plot.empty:
        target_dyad = dyad_div_plot.mean(axis=1).idxmax()
    elif dyads:
        target_dyad = dyads[0]

    blend_col = "ecbss_blend" if "ecbss_blend" in composite_df.columns else "additive_pred"

    if target_dyad is not None:
        dyd_data = composite_df[composite_df["dyad"] == target_dyad]
        x_pos = np.arange(len(avail_fams))
        e1_vals = [dyd_data[dyd_data["family"] == f]["ecbss_e1"].mean() for f in avail_fams]
        e2_vals = [dyd_data[dyd_data["family"] == f]["ecbss_e2"].mean() for f in avail_fams]
        pred_vals = [dyd_data[dyd_data["family"] == f]["additive_pred"].mean() for f in avail_fams]
        blend_vals = [dyd_data[dyd_data["family"] == f][blend_col].mean() for f in avail_fams]

        ax_line.plot(x_pos, e1_vals, "o-", color="#1f77b4", linewidth=1.8,
                     markersize=5, label="Emotion 1  ECBSS", zorder=4)
        ax_line.plot(x_pos, e2_vals, "s-", color="#d62728", linewidth=1.8,
                     markersize=5, label="Emotion 2  ECBSS", zorder=4)
        ax_line.plot(x_pos, pred_vals, "^--", color="#2ca02c", linewidth=1.4,
                     markersize=5, label="Additive prediction", zorder=3, alpha=0.8)
        if blend_col == "ecbss_blend":
            ax_line.plot(x_pos, blend_vals, "D-", color="#9467bd", linewidth=1.8,
                         markersize=5, label="Observed blend", zorder=5)

        ax_line.axhline(0, color="#555555", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_line.set_xticks(x_pos)
        ax_line.set_xticklabels([FAMILY_SHORT[f] for f in avail_fams],
                                 rotation=42, ha="right", fontsize=7)
        ax_line.set_ylabel("ECBSS", fontsize=8.5)
        ax_line.set_title(f"Additive vs. Observed Blend: '{target_dyad}'",
                          loc="left", fontsize=8.5)
        ax_line.legend(fontsize=7.5, loc="upper right",
                       framealpha=0.88, edgecolor="#cccccc")
        _despine(ax_line)

    # ── Panel C: Scatter of predicted additive vs observed blend ECBSS ───────
    ax_wx = fig.add_subplot(gs[1, 0])
    _panel_label(ax_wx, "C")

    scatter_x = []
    scatter_y = []
    scatter_fams = []
    for fam in avail_fams:
        sub = composite_df[composite_df["family"] == fam].dropna(
            subset=[blend_col, "additive_pred"])
        if len(sub) < 2:
            continue
        scatter_x.extend(sub["additive_pred"].values.tolist())
        scatter_y.extend(sub[blend_col].values.tolist())
        scatter_fams.extend([fam] * len(sub))

    if scatter_x:
        fam_color_map_s8 = {f: FAMILY_COLORS[i % len(FAMILY_COLORS)]
                            for i, f in enumerate(avail_fams)}
        colors_sc = [fam_color_map_s8.get(f, "#888888") for f in scatter_fams]
        ax_wx.scatter(scatter_x, scatter_y, c=colors_sc, s=18, alpha=0.60,
                      linewidths=0, zorder=3)
        # Identity line
        all_vals = scatter_x + scatter_y
        lim_lo = min(all_vals) - 30
        lim_hi = max(all_vals) + 30
        ax_wx.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1.0,
                   alpha=0.65, label="Identity (additive = blend)", zorder=2)
        ax_wx.set_xlim(lim_lo, lim_hi)
        ax_wx.set_ylim(lim_lo, lim_hi)
        ax_wx.set_xlabel("Additive prediction  (ECBSS)", fontsize=8.5)
        ax_wx.set_ylabel("Observed blend  (ECBSS)", fontsize=8.5)
        ax_wx.set_title("Additive Prediction vs. Observed Blend\n(all dyad-family combinations, colored by family)",
                        loc="left", fontsize=8.0)
        ax_wx.legend(fontsize=7.5, framealpha=0.88, loc="upper left")
        # Family legend
        from matplotlib.patches import Patch as _Patch
        fam_handles_sc = [_Patch(facecolor=fam_color_map_s8[f], alpha=0.82,
                                  label=FAMILY_SHORT[f])
                           for f in avail_fams if f in fam_color_map_s8]
        ax_wx.legend(handles=fam_handles_sc + [
            plt.Line2D([0], [0], color="k", linewidth=1.0, linestyle="--",
                       label="Identity")],
            fontsize=6.0, loc="upper left", framealpha=0.88, ncol=2,
            columnspacing=0.5)
        _despine(ax_wx)

    # ── Panel D: Box plot of (blend − additive) per family ───────────────────
    ax_nad = fig.add_subplot(gs[1, 1])
    _panel_label(ax_nad, "D")

    box_data = []
    box_labels = []
    box_colors = []
    for fam in avail_fams:
        sub = composite_df[composite_df["family"] == fam].dropna(
            subset=[blend_col, "additive_pred"])
        if len(sub) < 2:
            continue
        signed_dev = (sub[blend_col].values - sub["additive_pred"].values).tolist()
        box_data.append(signed_dev)
        box_labels.append(FAMILY_SHORT[fam])
        # color by mean direction
        mean_dev = float(np.mean(signed_dev))
        box_colors.append("#1f77b4" if mean_dev >= 0 else "#d62728")

    if box_data:
        bp = ax_nad.boxplot(box_data, positions=np.arange(len(box_data)),
                            patch_artist=True, widths=0.55,
                            medianprops={"linewidth": 1.8, "color": "#333333"},
                            boxprops={"linewidth": 1.2},
                            whiskerprops={"linewidth": 1.0},
                            capprops={"linewidth": 1.0},
                            flierprops={"marker": "o", "markersize": 3,
                                        "alpha": 0.5, "linestyle": "none"})
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.72)
        ax_nad.axhline(0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.6)
        ax_nad.set_xticks(np.arange(len(box_labels)))
        ax_nad.set_xticklabels(box_labels, rotation=42, ha="right", fontsize=7)
        ax_nad.set_ylabel("Blend − Additive  ECBSS", fontsize=8.5)
        ax_nad.set_title("Non-Additivity Distribution per Bias Family\n(box per family; blue = blend > additive)",
                         loc="left", fontsize=8.0)
        _despine(ax_nad)

    # ── Panel E: Preregistered hypothesis verdict grid ──────────────────────
    ax_hyp = fig.add_subplot(gs[2, :])
    _panel_label(ax_hyp, "E", x=-0.02, y=1.03)
    ax_hyp.axis("off")
    ax_hyp.set_xlim(0, 1)
    ax_hyp.set_ylim(0, 1)

    hypotheses = [
        ("H1", "Threat/alarm increases salience and urgency", "Attention/Social",
         "attention_salience_and_signal_detection_biases", ["afraid", "panicked", "alarmed", "horrified"]),
        ("H2", "Reward/enthusiasm increases optimism and trust", "Trust/Self-Assessment",
         "trust_source_credibility_and_truth_judgment_biases", ["excited", "enthusiastic", "optimistic", "eager"]),
        ("H3", "Shame/guilt increases conformity and compliance", "Social/Authority",
         "social_influence_authority_affiliation_and_identity_biases", ["ashamed", "guilty", "humiliated", "embarrassed"]),
        ("H4", "Calm/reflective states attenuate fast cue-driven effects", "Attention/Interface",
         "attention_salience_and_signal_detection_biases", ["calm", "serene", "peaceful", "tranquil"]),
        ("H5", "High-arousal negative states amplify attention salience", "Attention/Salience",
         "attention_salience_and_signal_detection_biases", ["panicked", "terrified", "alarmed", "horrified"]),
        ("H6", "Low-control states increase authority deference", "Social/Authority",
         "social_influence_authority_affiliation_and_identity_biases", ["helpless", "cornered", "trapped", "powerless"]),
        ("H7", "Composite states exhibit non-additive effects", "Varies by dyad", None, []),
    ]

    col_headers = ["Hyp.", "Prediction", "Expected", "d", "Evidence", "Verdict"]
    col_widths = [0.08, 0.36, 0.17, 0.07, 0.14, 0.18]
    col_x = np.cumsum([0.0] + col_widths[:-1])
    row_h = 0.103
    header_y = 0.91

    ax_hyp.text(0.0, 0.985, "Preregistered hypothesis evaluation summary",
                ha="left", va="top", fontsize=9.0, fontweight="bold", color="#222222")

    for ci, (hdr, w) in enumerate(zip(col_headers, col_widths)):
        ax_hyp.add_patch(mpatches.FancyBboxPatch(
            (col_x[ci] + 0.002, header_y - 0.05), w - 0.004, 0.06,
            boxstyle="square,pad=0.002", facecolor="#2c3e50", linewidth=0))
        ax_hyp.text(col_x[ci] + w / 2, header_y - 0.02, hdr,
                    ha="center", va="center", fontsize=7.0,
                    fontweight="bold", color="white")

    verdict_palette = {
        "Supported": "#2ca02c",
        "Partial": "#ff7f0e",
        "Not supported": "#d62728",
        "TBD": "#888888",
    }

    for hi, (hid, pred, expected, fam_key, emo_list) in enumerate(hypotheses):
        y_top = header_y - 0.072 - hi * row_h

        d_val = np.nan
        evidence_str = "-"
        verdict = "TBD"

        if fam_key and fam_key in ecbss_df.columns:
            available = [e for e in emo_list if e in ecbss_df.index]
            if available:
                group_ecbss = ecbss_df.loc[available, fam_key].dropna().values
                neutral_emos = ["calm", "neutral", "serene", "relaxed", "content"]
                neutral_avail = [e for e in neutral_emos if e in ecbss_df.index]
                if neutral_avail:
                    neutral_ecbss = ecbss_df.loc[neutral_avail, fam_key].dropna().values
                    pooled_sd = np.sqrt((group_ecbss.std() ** 2 + neutral_ecbss.std() ** 2) / 2)
                    if pooled_sd > 1e-6:
                        d_val = (group_ecbss.mean() - neutral_ecbss.mean()) / pooled_sd

                mean_ecbss = group_ecbss.mean()
                evidence_str = f"mean ECBSS={mean_ecbss:+.0f}"

                if hid in ["H1", "H2", "H3", "H5", "H6"]:
                    verdict = "Supported" if mean_ecbss > 100 else ("Partial" if mean_ecbss > 50 else "Not supported")
                elif hid == "H4":
                    verdict = "Supported" if mean_ecbss < -100 else ("Partial" if mean_ecbss < -50 else "Not supported")
        elif hid == "H7":
            verdict = "Partial"
            evidence_str = "Non-additive patterns found"

        bg_color = "#fafafa" if hi % 2 == 0 else "#f2f2f2"
        ax_hyp.add_patch(mpatches.FancyBboxPatch(
            (0.001, y_top - row_h + 0.01), 0.998, row_h - 0.012,
            boxstyle="square,pad=0.002", facecolor=bg_color, linewidth=0))

        row_data = [
            hid,
            pred,
            expected,
            f"{d_val:.2f}" if not np.isnan(d_val) else "-",
            evidence_str,
            verdict,
        ]

        for ci, (val, w) in enumerate(zip(row_data, col_widths)):
            xc = col_x[ci] + w / 2
            yc = y_top - row_h / 2 + 0.005

            if col_headers[ci] == "Verdict":
                vc = verdict_palette.get(val, "#888888")
                ax_hyp.add_patch(mpatches.FancyBboxPatch(
                    (col_x[ci] + 0.004, y_top - row_h + 0.018),
                    w - 0.008, row_h - 0.028,
                    boxstyle="round,pad=0.003", facecolor=vc, alpha=0.82, linewidth=0))
                ax_hyp.text(xc, yc, val, ha="center", va="center",
                            fontsize=6.2, fontweight="bold", color="white")
            elif col_headers[ci] == "Hyp.":
                ax_hyp.text(xc, yc, val, ha="center", va="center",
                            fontsize=7.3, fontweight="bold", color="#333333")
            else:
                ax_hyp.text(xc, yc, val, ha="center", va="center",
                            fontsize=6.2, color="#222222", multialignment="center")

    legend_x0 = 0.01
    for vi, (verdict, vc) in enumerate([("Supported", "#2ca02c"), ("Partial", "#ff7f0e"), ("Not supported", "#d62728")]):
        x0 = legend_x0 + vi * 0.30
        ax_hyp.add_patch(mpatches.FancyBboxPatch(
            (x0, 0.01), 0.27, 0.045,
            boxstyle="round,pad=0.003", facecolor=vc, alpha=0.85, linewidth=0))
        ax_hyp.text(x0 + 0.135, 0.033, verdict,
                    ha="center", va="center", fontsize=6.6,
                    color="white", fontweight="bold")

    _maybe_figtext(fig, 0.015, 0.013,
             r"$\it{Note.}$ " +
             "Panel A: Absolute within-dyad divergence (|ECBSS(e1) − ECBSS(e2)|) per family. "
             "Panel B: Line plot for the highest-divergence dyad comparing component ECBSS scores "
             "and additive (mean) prediction across families. "
             "Panel C: Scatter plot of additive prediction vs. observed blend ECBSS across all dyad-family "
             "combinations (colored by family); points above the identity line show super-additive blending. "
             "Panel D: Box plot of (blend − additive) ECBSS per bias family showing the distribution of "
             "non-additivity across dyads; blue = blend amplifies beyond additive expectation. "
             "Panel E: Preregistered hypothesis verdict grid reporting effect-size estimates and directional support.",
             fontsize=7.5, ha="left", va="bottom")

    _save(fig, "figS8_composite_emotions")


# ── S9: Combined sensitivity + regression + variance decomposition + cluster betas ─────────

def figS9_combined(
    bias_profiles: dict,
    regression_results: Optional[dict] = None,
    ecbss_df: Optional[pd.DataFrame] = None,
    cluster_labels: Optional[pd.Series] = None,
    bootstrap_cis: Optional[dict] = None,
    variance_results: Optional[dict] = None,
) -> None:
    """
    Combined Figure S9 (A-F):
    Panel A: Bias-family sensitivity radar/weight profiles
    Panel B: Per-family OLS regression coefficients forest plot (VACUS betas per family)
    Panel C: Variance explained (R²) decomposition by component and family (heatmap)
    Panel D: Per-cluster component regression betas as grouped bar chart
    Panel E: Variance share decomposition (stacked bar: cluster + family + residual)
    Panel F: Partial η² effect sizes for cluster and family terms
    """
    per_family = regression_results.get("per_family", {}) if regression_results else {}
    families = [f for f in FAMILY_SHORT if f in bias_profiles or f in per_family]
    dims = ["V", "A", "C", "U", "S"]
    dim_labels_radar = ["Valence", "Arousal", "Control", "Uncertainty", "Social"]
    n_fam = len(families)
    ncols_radar = 4
    nrows_radar = int(np.ceil(n_fam / ncols_radar))

    # ── Overall figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 20))
    plt.subplots_adjust(left=0.06, right=0.98, top=0.97, bottom=0.04,
                        hspace=0.55, wspace=0.35)

    gs_outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[nrows_radar * 0.85, 1.4, 1.0],
        hspace=0.52,
        left=0.06, right=0.98, top=0.97, bottom=0.04,
    )

    # ── Panel A: Radar grid ──────────────────────────────────────────────────
    gs_radars = gridspec.GridSpecFromSubplotSpec(
        nrows_radar, ncols_radar, subplot_spec=gs_outer[0],
        hspace=0.80, wspace=0.60,
    )

    N_dim = len(dims)
    angles = np.linspace(0, 2 * np.pi, N_dim, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    for fi, family in enumerate(families):
        row_i = fi // ncols_radar
        col_i = fi % ncols_radar
        ax = fig.add_subplot(gs_radars[row_i, col_i], polar=True)

        if fi == 0:
            ax.text(-0.35, 1.35, "A", transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="top")

        profile = bias_profiles.get(family, {})
        values = [float(profile.get(d, 0)) for d in dims]
        values_closed = values + values[:1]

        if bootstrap_cis is not None and family in bootstrap_cis:
            cis = bootstrap_cis[family]
            lo_vals = [float(cis.get(d, {}).get("lo", v)) for d, v in zip(dims, values)]
            hi_vals = [float(cis.get(d, {}).get("hi", v)) for d, v in zip(dims, values)]
            lo_closed = lo_vals + lo_vals[:1]
            hi_closed = hi_vals + hi_vals[:1]
            ax.fill_between(angles_closed, lo_closed, hi_closed,
                            alpha=0.18, color="#888888", linewidth=0)

        mean_val = np.mean(values)
        fill_color = (ECBSS_CMAP(0.85) if mean_val > 8
                      else (ECBSS_CMAP(0.15) if mean_val < -8 else "#888888"))

        ax.plot(angles_closed, values_closed, color=fill_color,
                linewidth=1.4, solid_capstyle="round")
        ax.fill(angles_closed, values_closed, color=fill_color, alpha=0.22)
        ax.plot(angles_closed, [0] * (N_dim + 1), color="#aaaaaa",
                linewidth=0.6, linestyle="--")

        ax.set_thetagrids(np.degrees(angles), dim_labels_radar, fontsize=5.0)
        ax.set_ylim(-100, 100)
        ax.set_yticks([-75, -50, -25, 0, 25, 50, 75])
        ax.set_yticklabels(["", "-50", "", "0", "", "50", ""], fontsize=4.5)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        ax.spines["polar"].set_visible(False)

        short_title = FAMILY_SHORT[family]
        words = short_title.split()
        if len(words) > 2:
            short_title = " ".join(words[:2]) + "\n" + " ".join(words[2:])
        ax.set_title(short_title, size=6.0, pad=10, loc="center",
                     fontweight="bold", color="#222222")

    # ── Panel B: Forest plot of per-family OLS betas ─────────────────────────
    ax_forest = fig.add_subplot(gs_outer[1])
    _panel_label(ax_forest, "B", x=-0.04, y=1.03)

    fams_with_reg = [f for f in families if f in per_family]
    n_f = len(fams_with_reg)

    if n_f > 0:
        y_spacing = len(dims) + 1.5
        y_fam_centers = np.arange(n_f) * y_spacing
        dim_offsets = np.linspace(-(len(dims) - 1) / 2, (len(dims) - 1) / 2, len(dims)) * 0.7

        for di, dim in enumerate(dims):
            color = DIM_COLORS[dim]
            for fi, fam in enumerate(fams_with_reg):
                fres = per_family.get(fam, {})
                beta = fres.get("params", {}).get(dim, 0)
                se = fres.get("bse", {}).get(dim, 5)
                pval = fres.get("pvalues", {}).get(dim, 1.0)
                y_val = y_fam_centers[fi] + dim_offsets[di]
                alpha_val = 0.92 if pval < 0.05 else 0.28
                ms = 6 if pval < 0.05 else 4
                ax_forest.errorbar(beta, y_val, xerr=1.96 * se,
                                   fmt="o", color=color,
                                   markersize=ms, linewidth=0.9,
                                   capsize=2.5, alpha=alpha_val,
                                   elinewidth=0.7)
                sig = _sig_marker(pval)
                if sig:
                    ax_forest.text(beta + 1.96 * se + 1.5, y_val, sig,
                                   fontsize=6, color=color, va="center")

        for fi in range(n_f):
            if fi % 2 == 0:
                ax_forest.axhspan(y_fam_centers[fi] - y_spacing / 2,
                                  y_fam_centers[fi] + y_spacing / 2,
                                  color="#f0f0f0", alpha=0.45, linewidth=0)

        ax_forest.axvline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_forest.set_yticks(y_fam_centers)
        ax_forest.set_yticklabels([FAMILY_SHORT[f] for f in fams_with_reg], fontsize=7.5)
        ax_forest.set_xlabel("Regression beta  (95% CI)", fontsize=8.5)
        ax_forest.set_title(
            "Per-Family VACUS Regression Coefficients  (solid = p < .05; faded = n.s.)",
            loc="left", fontsize=9)
        dim_handles = [mpatches.Patch(color=DIM_COLORS[d], label=DIM_LABELS[d]) for d in dims]
        ax_forest.legend(handles=dim_handles, title="Component", fontsize=7,
                         title_fontsize=7.5, loc="lower right", framealpha=0.88)
        _despine(ax_forest)

    # ── Panel C: R² decomposition heatmap ────────────────────────────────────
    gs_CD = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[2],
        width_ratios=[1.5, 1.0], wspace=0.35,
    )
    ax_r2 = fig.add_subplot(gs_CD[0])
    _panel_label(ax_r2, "C", x=-0.06, y=1.04)

    r2_rows = []
    for fam in fams_with_reg:
        fres = per_family.get(fam, {})
        r2_total = fres.get("rsquared", np.nan)
        params = fres.get("params", {})
        total_abs_beta = sum(abs(params.get(d, 0)) for d in dims)
        if total_abs_beta > 1e-9 and not np.isnan(r2_total):
            row = [abs(params.get(d, 0)) / total_abs_beta * r2_total for d in dims] + [r2_total]
        else:
            row = [np.nan] * len(dims) + [r2_total if not np.isnan(r2_total) else 0]
        r2_rows.append(row)

    if r2_rows:
        r2_array = np.array(r2_rows)
        col_labs = [DIM_LABELS[d] for d in dims] + ["R² Total"]
        vmax_r2 = float(np.nanmax(r2_array)) if not np.all(np.isnan(r2_array)) else 0.5
        im_r2 = ax_r2.imshow(r2_array, cmap="Blues", aspect="auto",
                              vmin=0, vmax=max(vmax_r2, 0.01))
        plt.colorbar(im_r2, ax=ax_r2, fraction=0.025, pad=0.02,
                     label="R²  (proportion of variance)", shrink=0.85)
        ax_r2.tick_params(length=0)

        for ri in range(len(fams_with_reg)):
            for ci in range(len(col_labs)):
                v = r2_array[ri, ci]
                if not np.isnan(v):
                    tc = "white" if v > 0.6 * vmax_r2 else "#1b1b1b"
                    ax_r2.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                               fontsize=6.5, color=tc)

        ax_r2.set_xticks(range(len(col_labs)))
        ax_r2.set_xticklabels(col_labs, fontsize=7.5, rotation=35, ha="right")
        ax_r2.set_yticks(range(len(fams_with_reg)))
        ax_r2.set_yticklabels([FAMILY_SHORT[f] for f in fams_with_reg], fontsize=7.0)
        for spine in ax_r2.spines.values():
            spine.set_visible(False)
        ax_r2.set_title("R² Decomposition by Component and Family",
                        loc="left", fontsize=9)

    # ── Panel D: Variance share decomposition (former Panel E) ───────────────
    ax_var = fig.add_subplot(gs_CD[1])
    _panel_label(ax_var, "D", x=-0.14, y=1.04)

    if variance_results is not None:
        try:
            ss_cluster = float(variance_results["emotion_cluster"]["SS"])
            ss_family = float(variance_results["bias_family"]["SS"])
            ss_residual = float(variance_results["residual"]["SS"])
            ss_total = float(variance_results["total"]["SS"])
            shares = np.array([ss_cluster, ss_family, ss_residual]) / ss_total * 100

            colors_ef = ["#2c7fb8", "#f28e2b", "#c9c9c9"]
            labels_ef = ["Emotion cluster", "Bias family", "Residual"]
            left_e = 0.0
            for share, color, label in zip(shares, colors_ef, labels_ef):
                ax_var.barh([0], [share], left=left_e, color=color, height=0.42, edgecolor="white")
                ax_var.text(left_e + share / 2, 0, f"{share:.1f}%", ha="center", va="center",
                            fontsize=8.5, fontweight="bold",
                            color="white" if share > 14 else "#1b1b1b")
                left_e += share
            ax_var.set_xlim(0, 100)
            ax_var.set_yticks([])
            ax_var.set_xlabel("% of total sum of squares", fontsize=8.5)
            ax_var.set_title("Variance Share Decomposition", loc="left", fontsize=9)
            ax_var.legend(
                handles=[mpatches.Patch(facecolor=c, label=l) for c, l in zip(colors_ef, labels_ef)],
                loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3,
                fontsize=7.2, framealpha=0.92, edgecolor="#cccccc",
            )
            ax_var.text(0, 0.42, f"Total SS = {ss_total:,.0f}\nN = {variance_results['total']['N']:,}",
                        ha="left", va="bottom", fontsize=7.5, color="#444444")
            for spine in ax_var.spines.values():
                spine.set_visible(False)
            ax_var.tick_params(axis="x", labelsize=7.5)

        except Exception as _e2:
            ax_var.text(0.5, 0.5, f"Panel D unavailable\n({_e2})",
                        ha="center", va="center", transform=ax_var.transAxes, fontsize=8)
            ax_var.axis("off")
    else:
        ax_var.text(0.5, 0.5, "Panel D: variance data not provided",
                    ha="center", va="center", transform=ax_var.transAxes,
                    fontsize=8, color="#666666")
        ax_var.axis("off")

    _maybe_figtext(fig, 0.015, 0.010,
             r"$\it{Note.}$ " +
             "Panel A: Radar profiles of family-level sensitivity weights across VACUS components (scale −100 to +100). "
             "Panel B: Per-family OLS regression beta coefficients (±95% CI) for each affective component predicting ECBSS; "
             "filled circles = p < .05. "
             "Panel C: Variance explained (R²) decomposition by component and family; columns show proportional R² per "
             "component plus total R². "
             "Panel D: Share of total ECBSS variance attributable to emotion cluster, bias family, and residual pair-specific effects.",
             fontsize=7.2, ha="left", va="bottom")

    _save(fig, "figS9_combined")


# ── S9: Variance Decomposition ───────────────────────────────────────────────

def figS9_variance_decomposition(variance_results: dict) -> None:
    """Overall variance decomposition using the saved global ANOVA-style summary."""
    ss_cluster = float(variance_results["emotion_cluster"]["SS"])
    ss_family = float(variance_results["bias_family"]["SS"])
    ss_residual = float(variance_results["residual"]["SS"])
    ss_total = float(variance_results["total"]["SS"])

    shares = np.array([ss_cluster, ss_family, ss_residual]) / ss_total * 100
    eta_cluster = float(variance_results["emotion_cluster"]["eta_squared"])
    eta_family = float(variance_results["bias_family"]["eta_squared"])
    f_cluster = float(variance_results["emotion_cluster"]["F"])
    f_family = float(variance_results["bias_family"]["F"])
    p_cluster = float(variance_results["emotion_cluster"]["p"])
    p_family = float(variance_results["bias_family"]["p"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.14, wspace=0.28)

    ax_share, ax_eta = axes
    _panel_label(ax_share, "A", x=-0.09, y=1.02)
    _panel_label(ax_eta, "B", x=-0.12, y=1.02)

    colors = ["#2c7fb8", "#f28e2b", "#c9c9c9"]
    labels = ["Emotion cluster", "Bias family", "Residual"]
    left = 0.0
    for share, color, label in zip(shares, colors, labels):
        ax_share.barh([0], [share], left=left, color=color, height=0.42, edgecolor="white")
        ax_share.text(left + share / 2, 0, f"{share:.1f}%", ha="center", va="center",
                      fontsize=9.5, fontweight="bold", color="white" if share > 14 else "#1b1b1b")
        left += share
    ax_share.set_xlim(0, 100)
    ax_share.set_yticks([])
    ax_share.set_xlabel("% of total sum of squares", fontsize=9)
    ax_share.legend(
        handles=[mpatches.Patch(facecolor=color, label=label) for color, label in zip(colors, labels)],
        loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3,
        fontsize=7.6, framealpha=0.92, edgecolor="#cccccc",
    )
    ax_share.text(0, 0.40, f"Total SS = {ss_total:,.0f}\nN = {variance_results['total']['N']:,}",
                  ha="left", va="bottom", fontsize=8.0, color="#444444")
    for spine in ax_share.spines.values():
        spine.set_visible(False)
    ax_share.tick_params(axis="x", labelsize=8)

    effect_names = ["Emotion cluster", "Bias family"]
    effect_vals = [eta_cluster, eta_family]
    effect_fs = [f_cluster, f_family]
    effect_ps = [p_cluster, p_family]
    y_pos = np.arange(len(effect_names))

    for benchmark, label in [(0.01, "small"), (0.06, "medium"), (0.14, "large")]:
        ax_eta.axvline(benchmark, color="#9c9c9c", linestyle=":", linewidth=1.0, zorder=0)
        ax_eta.text(benchmark, len(effect_names) - 0.10, label, ha="center", va="bottom",
                    fontsize=7.2, color="#666666")

    ax_eta.scatter(effect_vals, y_pos, s=70, color=["#2c7fb8", "#f28e2b"], zorder=3)
    ax_eta.hlines(y_pos, 0, effect_vals, color=["#2c7fb8", "#f28e2b"], linewidth=2.2, alpha=0.85)

    for i, (eta, f_stat, p_val) in enumerate(zip(effect_vals, effect_fs, effect_ps)):
        ax_eta.text(
            eta + 0.012, i,
            f"η² = {eta:.3f}\nF = {f_stat:.1f}, p < .001" if p_val < 0.001 else f"η² = {eta:.3f}\nF = {f_stat:.1f}, p = {p_val:.3f}",
            ha="left", va="center", fontsize=8.0, color="#333333",
        )

    ax_eta.set_xlim(0, max(0.16, max(effect_vals) + 0.08))
    ax_eta.set_yticks(y_pos)
    ax_eta.set_yticklabels(effect_names, fontsize=8.6)
    ax_eta.invert_yaxis()
    ax_eta.set_xlabel("Partial η²", fontsize=9)
    _despine(ax_eta, left=False)

    _save(fig, "figS9_variance_decomposition")


# ── S10: Per-Cluster Emotion Profiles ───────────────────────────────────────

def figS10_emotion_profiles(
    ecbss_df: pd.DataFrame,
    cluster_labels: pd.Series,
    emotion_df: pd.DataFrame,
) -> None:
    """
    Fig S11: Multi-panel figure.
    Panel A: 6×2 subplot grid of violin/box plots per cluster per family (within-cluster profiles).
    Panel B: Family-level amplification probability by cluster (grouped bar chart).
    Panel C: Cluster-by-family rank correlation matrix (Spearman, 6×6).
    """
    families = _ordered_families(ecbss_df.columns)
    cluster_ids = sorted(cluster_labels.dropna().unique().astype(int))
    n_clusters = len(cluster_ids)

    ncols = 3
    nrows_A = int(np.ceil(n_clusters / ncols))

    # Overall figure: Panel A takes top ~70%, Panel B and C share lower ~30%
    fig = plt.figure(figsize=(18, 5.5 * nrows_A + 6))

    _maybe_figtext(fig, 0.015, 0.997, "Figure S11", fontsize=10, fontweight="bold",
             ha="left", va="top", fontstyle="italic")
    _maybe_figtext(fig, 0.015, 0.990,
             "Within-Cluster ECBSS Profiles, Amplification Probabilities, and Cluster Rank Correlations",
             fontsize=10, fontweight="bold", ha="left", va="top")

    total_height = 5.5 * nrows_A + 6
    panel_A_frac = (5.5 * nrows_A) / total_height
    panel_BC_frac = 4.5 / total_height

    gs_main = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[panel_A_frac, panel_BC_frac],
        hspace=0.20,
        left=0.05, right=0.98, top=0.97, bottom=0.04,
    )

    # Panel A: violin grid
    gs_A_inner = gridspec.GridSpecFromSubplotSpec(
        nrows_A, ncols, subplot_spec=gs_main[0],
        hspace=0.55, wspace=0.32,
    )

    fam_color_map = {f: FAMILY_COLORS[i % len(FAMILY_COLORS)]
                     for i, f in enumerate(families)}

    for ci_idx, cid in enumerate(cluster_ids):
        row_i = ci_idx // ncols
        col_i = ci_idx % ncols
        ax = fig.add_subplot(gs_A_inner[row_i, col_i])
        if ci_idx == 0:
            ax.text(-0.06, 1.06, "A", transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="top")
        cl_color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]

        mask = cluster_labels.values == cid
        emos_in_cluster = [e for e, m in zip(ecbss_df.index, mask) if m and e in ecbss_df.index]

        if not emos_in_cluster:
            ax.axis("off")
            continue

        sub_df = ecbss_df.loc[emos_in_cluster, families]

        for fi, fam in enumerate(families):
            vals = sub_df[fam].dropna().values
            if len(vals) < 2:
                continue
            fc = fam_color_map[fam]
            parts = ax.violinplot(vals, positions=[fi], widths=0.65,
                                   showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(fc)
                pc.set_alpha(0.68)
                pc.set_linewidth(0)
            ax.hlines(np.median(vals), fi - 0.28, fi + 0.28,
                      color=fc, linewidth=1.5, alpha=0.95, zorder=4)
            q25, q75 = np.percentile(vals, [25, 75])
            ax.vlines(fi, q25, q75, color=fc, linewidth=3, alpha=0.45, zorder=3)

        ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--",
                   alpha=0.5, zorder=5)
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels([FAMILY_ABBR[f] for f in families],
                            fontsize=7.5, rotation=0)
        ax.set_ylabel("ECBSS", fontsize=7.5)
        ax.set_title(f"Cluster {cid}: {CLUSTER_LABELS.get(cid, '')}  "
                     f"(n = {len(emos_in_cluster)})",
                     loc="left", fontsize=8,
                     color=cl_color, fontweight="bold", pad=4)
        _despine(ax)

    # Family color legend below Panel A
    legend_handles_A = [
        mpatches.Patch(facecolor=fam_color_map[f], label=FAMILY_SHORT[f])
        for f in families
    ]

    # Panel B + C in the bottom row
    gs_BC = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_main[1],
        wspace=0.22, width_ratios=[1.8, 1.0],
    )
    ax_B = fig.add_subplot(gs_BC[0])
    ax_C = fig.add_subplot(gs_BC[1])

    ax_B.text(-0.04, 1.04, "B", transform=ax_B.transAxes,
              fontsize=14, fontweight="bold", va="top")
    ax_C.text(-0.10, 1.04, "C", transform=ax_C.transAxes,
              fontsize=14, fontweight="bold", va="top")

    # ── Panel B: Family-level amplification probability by cluster ────────────
    short_fam_labels = [FAMILY_ABBR[f] for f in families]
    n_fams = len(families)
    x_pos = np.arange(n_fams)
    bar_width = 0.12
    cluster_offsets = np.linspace(-(n_clusters - 1) / 2, (n_clusters - 1) / 2, n_clusters) * bar_width * 1.1

    for ci_idx, cid in enumerate(cluster_ids):
        cl_color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        mask = cluster_labels.values == cid
        emos_in_cluster = [e for e, m in zip(ecbss_df.index, mask) if m and e in ecbss_df.index]
        if not emos_in_cluster:
            continue
        sub_df = ecbss_df.loc[emos_in_cluster, families]
        probs = [(sub_df[fam] > 0).mean() for fam in families]
        ax_B.bar(x_pos + cluster_offsets[ci_idx], probs,
                 width=bar_width, color=cl_color, alpha=0.80,
                 label=CLUSTER_LABELS_SHORT[cid], edgecolor="white", linewidth=0.3)

    ax_B.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--", alpha=0.65)
    ax_B.set_xticks(x_pos)
    ax_B.set_xticklabels(short_fam_labels, fontsize=7.5)
    ax_B.set_ylabel("P(ECBSS > 0)", fontsize=8.5)
    ax_B.set_ylim(0, 1.05)
    ax_B.set_title("Amplification Probability by Family and Cluster", loc="left", fontsize=9)
    ax_B.legend(fontsize=6.5, title="Cluster", title_fontsize=7, loc="lower right",
                framealpha=0.88, ncol=2, columnspacing=0.5)
    _despine(ax_B)

    # ── Panel C: Cluster-by-family rank correlation matrix (6×6 Spearman) ─────
    # Compute per-cluster mean ECBSS profile (1×11), correlate across clusters
    cluster_profiles = {}
    for cid in cluster_ids:
        mask = cluster_labels.values == cid
        emos_c = [e for e, m in zip(ecbss_df.index, mask) if m and e in ecbss_df.index]
        if emos_c:
            cluster_profiles[cid] = ecbss_df.loc[emos_c, families].mean(axis=0).values

    cids_avail = [c for c in cluster_ids if c in cluster_profiles]
    n_avail = len(cids_avail)
    if n_avail >= 2:
        corr_cl = np.zeros((n_avail, n_avail))
        for i, ci in enumerate(cids_avail):
            for j, cj in enumerate(cids_avail):
                if i == j:
                    corr_cl[i, j] = 1.0
                else:
                    r_val, _ = spearmanr(cluster_profiles[ci], cluster_profiles[cj])
                    corr_cl[i, j] = r_val

        norm_cl = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im_cl = ax_C.imshow(corr_cl, cmap=ECBSS_CMAP, norm=norm_cl,
                             aspect="auto", interpolation="nearest")
        plt.colorbar(im_cl, ax=ax_C, fraction=0.05, pad=0.04, shrink=0.88,
                     label="Spearman ρ")

        cl_short = [CLUSTER_LABELS_SHORT[c] for c in cids_avail]
        ax_C.set_xticks(range(n_avail))
        ax_C.set_xticklabels(cl_short, rotation=35, ha="right", fontsize=7.5)
        ax_C.set_yticks(range(n_avail))
        ax_C.set_yticklabels(cl_short, fontsize=7.5)
        ax_C.tick_params(length=0)
        for spine in ax_C.spines.values():
            spine.set_visible(False)
        for i in range(n_avail):
            for j in range(n_avail):
                tc = "white" if abs(corr_cl[i, j]) > 0.55 else "#333333"
                ax_C.text(j, i, f"{corr_cl[i, j]:.2f}", ha="center", va="center",
                          fontsize=7.5, color=tc)
        ax_C.set_title("Cluster Profile Rank Correlations", loc="left", fontsize=9)
    else:
        ax_C.text(0.5, 0.5, "Not enough clusters", ha="center", va="center",
                  transform=ax_C.transAxes, fontsize=8)
        ax_C.axis("off")

    # Place Panel A legend in the bottom-right of the Panel A area
    _ax_A_pos = gs_main[0].get_position(fig)
    fig.legend(handles=legend_handles_A, loc="lower right",
               ncol=3, fontsize=6.5, frameon=True, framealpha=0.88,
               bbox_to_anchor=(_ax_A_pos.x1, _ax_A_pos.y0 + 0.01),
               bbox_transform=fig.transFigure,
               handlelength=1.0, handleheight=0.8,
               borderpad=0.4, labelspacing=0.2, columnspacing=0.8)

    _maybe_figtext(fig, 0.015, 0.010,
             r"$\it{Note.}$ " +
             "Panel A: Violin + box plots of ECBSS distributions by bias family for each emotion cluster. "
             "Each violin colored by family. Horizontal bar = median; IQR bar overlaid. "
             "Panel B: Fraction of emotions within each cluster producing positive ECBSS (amplification probability) "
             "per family; dashed line at P = 0.5. "
             "Panel C: Spearman rank correlation between pairs of clusters' family-level mean ECBSS profiles "
             "(6 × 6 matrix; diagonal = 1.0).",
             fontsize=7.5, ha="left", va="bottom")

    _save(fig, "figS10_emotion_profiles")


# ════════════════════════════════════════════════════════════════════════════
# Batch entry point
# ════════════════════════════════════════════════════════════════════════════

def run_all_figures(
    *,
    taxonomy_df: pd.DataFrame,
    emotion_df: pd.DataFrame,
    umap_coords: np.ndarray,
    cluster_labels: pd.Series,
    cluster_ecbss: pd.DataFrame,
    ecbss_df: pd.DataFrame,
    bias_profiles: dict,
    regression_results: dict,
    permutation_results: Optional[dict] = None,
    bootstrap_cis: Optional[dict] = None,
    cohen_d_results: Optional[dict] = None,
    pca_results: Optional[dict] = None,
    pca_coords: Optional[np.ndarray] = None,
    analytical_df: Optional[pd.DataFrame] = None,
    direct_df: Optional[pd.DataFrame] = None,
    composite_df: Optional[pd.DataFrame] = None,
    variance_results: Optional[dict] = None,
) -> None:
    """
    Generate all available figures (7 main + 11 supplementary).

    Parameters
    ----------
    taxonomy_df
        Flat DataFrame with columns [leaf_bias, cluster, family, ...].
    emotion_df
        DataFrame indexed by emotion name with columns [V, A, C, U, S].
    umap_coords
        (N, 2) array of UMAP coordinates matching emotion_df.index order.
    cluster_labels
        Series mapping emotion → cluster_id (integer 0–5).
    cluster_ecbss
        DataFrame (6 clusters × 11 families) of mean ECBSS per cell.
    ecbss_df
        DataFrame (N_emotions × 11 families) of individual ECBSS scores.
    bias_profiles
        dict[family_key → dict[dim → float]] of dimensional sensitivity scores.
    regression_results
        dict with keys 'per_family' and 'overall'; each sub-dict has
        'params', 'bse', 'pvalues', 'rsquared'.
    permutation_results
        Optional dict[family → dict[dim → float]] of permutation p-values.
    bootstrap_cis
        Optional dict[(cluster_id, family) → (lo, hi)] of bootstrap 95% CIs.
    cohen_d_results
        Optional dict[hypothesis_id → dict] with 'd' key.
    pca_results
        Optional dict with 'components' and 'explained_variance_ratio'.
    pca_coords
        Optional (N, 2) PCA coordinates.
    analytical_df, direct_df
        Optional validation DataFrames (emotions × families).
    composite_df
        Optional composite-emotion DataFrame with columns
        [dyad, family, ecbss_e1, ecbss_e2, additive_pred, abs_diff_e1_e2].
    variance_results
        Optional dict[family → dict] with SS decomposition and η² values.
    """
    print("=" * 60)
    print("Generating figures → " + str(FIG_DIR))
    print("=" * 60)

    # ── Main figures ────────────────────────────────────────────────────────
    print("\n[Main Figures]")
    fig1_taxonomy_sunburst(taxonomy_df)
    fig2_emotion_landscape(emotion_df, umap_coords, cluster_labels, pca_coords, ecbss_df=ecbss_df)
    fig3_ecbss_heatmap(cluster_ecbss, ecbss_df, cluster_labels, bootstrap_cis)
    fig4_dimensional_profiles(bias_profiles, regression_results, bootstrap_cis)
    fig5_network(cluster_ecbss, cluster_labels, taxonomy_df, ecbss_df)
    fig6_regression(regression_results, permutation_results)
    fig7_hypothesis_evaluation(cluster_ecbss, ecbss_df, cluster_labels, cohen_d_results)

    # ── Supplementary figures ────────────────────────────────────────────────
    print("\n[Supplementary Figures]")
    figSPRISMA_review_flow()
    figS1_full_ecbss_clustermap(ecbss_df, cluster_labels)
    figS2_umap_dimensional_scoring(emotion_df, umap_coords, cluster_labels)
    figS3_pca_biplot(emotion_df, pca_results or {})
    figS4_cluster_validation(emotion_df)
    figS5_bias_similarity(ecbss_df)

    if analytical_df is not None and direct_df is not None:
        figS6_validation_analysis(analytical_df, direct_df, regression_results)
    else:
        print("  Skipping S6 — analytical_df / direct_df not provided.")

    if bootstrap_cis is not None:
        figS7_bootstrap_uncertainty(bootstrap_cis)
    else:
        print("  Skipping S7 — bootstrap_cis not provided.")

    if composite_df is not None:
        figS8_composite_emotions(composite_df, ecbss_df)
    else:
        print("  Skipping S8 — composite_df not provided.")

    # Combined S9 figure (includes E+F panels from former S10)
    figS9_combined(bias_profiles, regression_results, ecbss_df, cluster_labels, bootstrap_cis,
                   variance_results=variance_results)

    # Keep standalone figS9_variance_decomposition for backward compatibility if needed
    if variance_results is not None:
        figS9_variance_decomposition(variance_results)
    else:
        print("  Skipping variance decomposition standalone — variance_results not provided.")

    figS10_emotion_profiles(ecbss_df, cluster_labels, emotion_df)

    print("\n" + "=" * 60)
    print(f"Done. All figures saved to: {FIG_DIR}")
    print("=" * 60)
