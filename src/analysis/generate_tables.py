"""Generate LaTeX tables used by the manuscript and supplementary materials."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis.data_loader import load_emotion_scores, load_taxonomy, get_family_short_labels

OUTPUTS = ROOT / "src/review_stages/analysis_outputs"
TABLES = ROOT / "paper/assets/tables"
TABLES.mkdir(parents=True, exist_ok=True)

SHORT = get_family_short_labels()
CLUSTER_LABELS = {
    0: "Hostile & Defiant",
    1: "Alarmed & Uncertain",
    2: "Socially Vulnerable",
    3: "Withdrawn & Low Arousal",
    4: "High-Arousal Positive",
    5: "Calm Positive",
}
CLUSTER_SHORT = {
    0: "Hostile",
    1: "Alarmed",
    2: "Social",
    3: "Withdrawn",
    4: "HA-Pos",
    5: "Calm+",
}
DIMS = ["V", "A", "C", "U", "S"]


def esc(value: object) -> str:
    """Escape a value for LaTeX."""
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_table(name: str, content: str) -> None:
    (TABLES / name).write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"wrote {name}")


def cluster_representatives(emotion_df: pd.DataFrame, cluster_labels: pd.Series, n: int = 6) -> dict[int, str]:
    """Return centroid-nearest exemplar emotions for each cluster."""
    z = pd.DataFrame(
        StandardScaler().fit_transform(emotion_df[DIMS]),
        index=emotion_df.index,
        columns=DIMS,
    )
    joined = z.join(cluster_labels.rename("emotion_cluster")).dropna()
    centroids = joined.groupby("emotion_cluster")[DIMS].mean()

    reps: dict[int, str] = {}
    for cid in sorted(centroids.index.astype(int)):
        sub = joined[joined["emotion_cluster"] == cid][DIMS]
        dist = ((sub - centroids.loc[cid]) ** 2).sum(axis=1).sort_values()
        reps[int(cid)] = ", ".join(esc(e) for e in dist.index[:n])
    return reps


def make_cluster_matrix(cluster_ecbss: pd.DataFrame, families: list[str]) -> str:
    lines = [
        r"{\renewcommand{\tabcolsep}{10pt}",
        r"\begin{tabular}{l" + "r" * len(families) + "r}",
        r"\toprule",
        "Cluster & " + " & ".join(esc(SHORT[f]) for f in families) + r" & \textit{Mean} \\",
        r"\midrule",
    ]
    for cid in sorted(cluster_ecbss.index.astype(int)):
        vals = [f"{cluster_ecbss.loc[cid, fam]:+.0f}" for fam in families]
        row_mean = cluster_ecbss.loc[cid, families].mean()
        lines.append(f"{esc(CLUSTER_SHORT[cid])} & " + " & ".join(vals) + f" & \\textit{{{row_mean:+.0f}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", "}"]
    return "\n".join(lines)


def make_longtable(
    name: str,
    caption: str,
    label: str,
    colspec: str,
    header: str,
    body_lines: list[str],
    n_cols: int,
) -> None:
    content = "\n".join(
        [
            rf"\begin{{longtable}}{{{colspec}}}",
            rf"\caption{{{caption}}}\label{{{label}}}\\",
            r"\toprule",
            header,
            r"\midrule",
            r"\endfirsthead",
            rf"\multicolumn{{{n_cols}}}{{@{{}}l}}{{\textbf{{Table \thetable\ (continued)}}}}\\",
            rf"\multicolumn{{{n_cols}}}{{@{{}}p{{\textwidth}}@{{}}}}{{{caption}}}\\",
            r"\toprule",
            header,
            r"\midrule",
            r"\endhead",
            r"\midrule",
            rf"\multicolumn{{{n_cols}}}{{r}}{{\textit{{Continued on next page}}}}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
            *body_lines,
            r"\end{longtable}",
        ]
    )
    write_table(name, content)


def main() -> None:
    _, taxonomy_df = load_taxonomy()
    emotion_df = load_emotion_scores()
    ecbss_df = pd.read_csv(OUTPUTS / "ecbss_direct.csv", index_col="emotion")
    regression_results = json.loads((OUTPUTS / "regression_results.json").read_text())
    bias_profiles = json.loads((OUTPUTS / "bias_sensitivity_profiles.json").read_text())
    cluster_ecbss = pd.read_csv(OUTPUTS / "cluster_ecbss.csv", index_col="emotion_cluster")
    cluster_labels = pd.read_csv(OUTPUTS / "emotion_cluster_labels.csv").set_index("emotion")["emotion_cluster"]

    # Table 1 / Supplementary Table S1: taxonomy summary
    family_stats = (
        taxonomy_df.groupby("family")
        .agg(N_clusters=("cluster", "nunique"), N_biases=("leaf_bias", "count"))
        .reset_index()
    )
    family_mean_ecbss = ecbss_df.mean().rename("Mean_ECBSS")
    family_stats = family_stats.merge(
        family_mean_ecbss.reset_index().rename(columns={"index": "family"}),
        on="family", how="left"
    )
    family_stats["Family"] = family_stats["family"].map(lambda x: esc(SHORT.get(x, x)))
    family_stats = family_stats.sort_values("Mean_ECBSS", ascending=False)

    lines = [
        r"\begin{tabularx}{\linewidth}{Xccr}",
        r"\toprule",
        r"Bias family & $n_{\mathrm{clusters}}$ & $n_{\mathrm{biases}}$ & Mean ECBSS \\",
        r"\midrule",
    ]
    for _, row in family_stats.iterrows():
        mean_ecbss = f"{row['Mean_ECBSS']:+.0f}" if pd.notna(row.get('Mean_ECBSS')) else "---"
        lines.append(f"{row['Family']} & {int(row['N_clusters'])} & {int(row['N_biases'])} & {mean_ecbss} \\\\")
    lines += [
        r"\midrule",
        rf"\textit{{Total/Mean}} & {int(family_stats['N_clusters'].sum())} & {int(family_stats['N_biases'].sum())} & {family_stats['Mean_ECBSS'].mean():+.0f} \\",
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    write_table("table1_taxonomy.tex", "\n".join(lines))

    # Table 2 / Supplementary Table S2: top amplifying + attenuating pairs
    pairs_df = (
        ecbss_df.reset_index()
        .melt(id_vars="emotion", var_name="family", value_name="ecbss")
        .sort_values("ecbss", ascending=False)
    )
    top_amp = pairs_df.head(8)
    top_att = pairs_df.tail(8).sort_values("ecbss")

    # Add cluster assignment for each emotion
    emotion_to_cluster = cluster_labels.to_dict()

    lines = [
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\centering\arraybackslash}p{1.4cm}>{\centering\arraybackslash}p{1.2cm}@{}}",
        r"\toprule",
        r"Emotion & Bias family & ECBSS & Cluster \\",
        r"\midrule",
        r"\multicolumn{4}{@{}l}{Highest amplifying pairs} \\[2pt]",
    ]
    for _, row in top_amp.iterrows():
        em = str(row['emotion'])
        cl = emotion_to_cluster.get(em, "?")
        cl_label = CLUSTER_SHORT.get(int(cl), str(cl)) if cl != "?" else "?"
        lines.append(
            f"{esc(em.title())} & {esc(SHORT[row['family']])} & $+${row['ecbss']:.0f} & {cl_label} \\\\"
        )
    lines += [r"\midrule", r"\multicolumn{4}{@{}l}{Strongest attenuating pairs} \\[2pt]"]
    for _, row in top_att.iterrows():
        em = str(row['emotion'])
        cl = emotion_to_cluster.get(em, "?")
        cl_label = CLUSTER_SHORT.get(int(cl), str(cl)) if cl != "?" else "?"
        lines.append(
            f"{esc(em.title())} & {esc(SHORT[row['family']])} & ${row['ecbss']:.0f}$ & {cl_label} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write_table("table2_top_pairs.tex", "\n".join(lines))

    # Table 3 / Supplementary Table S3: per-family regression coefficients
    per_family = regression_results["per_family"]
    ordered_families = [f for f in SHORT if f in per_family]
    lines = [
        r"\begin{tabularx}{\linewidth}{X" + "r" * (len(DIMS) + 1) + "}",
        r"\toprule",
        "Bias family & " + " & ".join(f"$\\beta_{{{d}}}$" for d in DIMS) + r" & $R^2$ \\",
        r"\midrule",
    ]
    for fam in ordered_families:
        res = per_family[fam]
        coeffs = []
        for dim in DIMS:
            beta = float(res["params"].get(dim, 0))
            p = float(res["pvalues"].get(dim, 1))
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            coeffs.append(f"${beta:.1f}{'^{' + sig + '}' if sig else ''}$")
        lines.append(
            f"{esc(SHORT[fam])} & " + " & ".join(coeffs) + f" & {float(res['rsquared']):.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write_table("table3_regression.tex", "\n".join(lines))

    # Table 4 / Supplementary Table S8: dimensional sensitivity profiles
    lines = [
        r"\begin{tabularx}{\linewidth}{X" + "r" * len(DIMS) + "}",
        r"\toprule",
        r"Bias family & Valence (V) & Arousal (A) & Control (C) & Uncertainty (U) & Social Orientation (S) \\",
        r"\midrule",
    ]
    for fam in [f for f in SHORT if f in bias_profiles]:
        vals = [f"{float(bias_profiles[fam].get(dim, 0)):+.0f}" for dim in DIMS]
        lines.append(f"{esc(SHORT[fam])} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write_table("table4_bias_sensitivity.tex", "\n".join(lines))
    write_table("tableS9_bias_sensitivity.tex", "\n".join(lines))

    # Supplementary Table S4a / S4b: cluster x family matrix
    matrix_families = [f for f in SHORT if f in cluster_ecbss.columns]
    split = (len(matrix_families) + 1) // 2
    first_half = matrix_families[:split]
    second_half = matrix_families[split:]
    matrix_text_a = make_cluster_matrix(cluster_ecbss, first_half)
    matrix_text_b = make_cluster_matrix(cluster_ecbss, second_half)
    write_table("tableS4a_cluster_ecbss.tex", matrix_text_a)
    write_table("tableS4b_cluster_ecbss.tex", matrix_text_b)
    # Backward-compatible filenames used by older drafts
    write_table("tableS1a_cluster_ecbss.tex", matrix_text_a)
    write_table("tableS1b_cluster_ecbss.tex", matrix_text_b)

    # Supplementary Table S5: cluster profiles and exemplar emotions
    cluster_summary = (
        emotion_df.join(cluster_labels.rename("emotion_cluster"))
        .groupby("emotion_cluster")[DIMS]
        .mean()
        .round(1)
    )
    cluster_counts = cluster_labels.value_counts().sort_index()
    representatives = cluster_representatives(emotion_df, cluster_labels)
    lines = [
        r"\begin{tabularx}{\textwidth}{lcrrrrr>{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"Cluster & $n$ & V & A & C & U & S & Representative emotions \\",
        r"\midrule",
    ]
    for cid in sorted(cluster_summary.index.astype(int)):
        row = cluster_summary.loc[cid]
        lines.append(
            f"{esc(CLUSTER_LABELS[cid])} & {int(cluster_counts.loc[cid])} & "
            f"{row['V']:.0f} & {row['A']:.0f} & {row['C']:.0f} & {row['U']:.0f} & {row['S']:.0f} & "
            f"{representatives[cid]} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write_table("tableS5_cluster_profiles.tex", "\n".join(lines))

    # Supplementary Table S6: grouped taxonomy inventory by family
    inventory_df = (
        taxonomy_df.groupby(["family", "cluster"])
        .agg(N_biases=("leaf_bias", "count"))
        .reset_index()
        .sort_values(["family", "cluster"])
    )
    body_lines = []
    for fam in [f for f in SHORT if f in inventory_df["family"].unique()]:
        fam_df = inventory_df[inventory_df["family"] == fam].copy()
        fam_total = int(fam_df["N_biases"].sum())
        body_lines.append(
            rf"\addlinespace[2pt]\multicolumn{{2}}{{@{{}}l}}{{\textbf{{{esc(SHORT[fam])}}} "
            rf"($n_{{\mathrm{{clusters}}}}={len(fam_df)}$, $n_{{\mathrm{{biases}}}}={fam_total}$)}} \\"
        )
        for row in fam_df.itertuples(index=False):
            cluster_name = esc(str(row.cluster).replace("_", " ").title())
            body_lines.append(f"{cluster_name} & {int(row.N_biases)} \\\\")
    make_longtable(
        name="tableS6_taxonomy_inventory.tex",
        caption="Taxonomy clusters grouped by bias family",
        label="tab:s6-taxonomy",
        colspec=r"p{0.74\textwidth}r",
        header=r"Taxonomy cluster & $n_{\mathrm{biases}}$ \\",
        body_lines=body_lines,
        n_cols=2,
    )

    # Supplementary Table S7: grouped emotion lexicon with full V-A-C-U-S scores
    lexicon_df = (
        emotion_df.join(cluster_labels.rename("emotion_cluster"))
        .reset_index()
        .rename(columns={"index": "emotion"})
        .sort_values(["emotion_cluster", "emotion"])
        .reset_index(drop=True)
    )
    body_lines = []
    for cid in sorted(lexicon_df["emotion_cluster"].astype(int).unique()):
        sub = lexicon_df[lexicon_df["emotion_cluster"] == cid]
        body_lines.append(
            rf"\addlinespace[2pt]\multicolumn{{6}}{{@{{}}l}}{{\textbf{{{esc(CLUSTER_LABELS[cid])}}} "
            rf"($n={len(sub)}$)}} \\"
        )
        for row in sub.itertuples(index=False):
            body_lines.append(
                f"{esc(str(row.emotion).title())} & "
                f"{int(row.V):+d} & {int(row.A):+d} & {int(row.C):+d} & {int(row.U):+d} & {int(row.S):+d} \\\\"
            )
    make_longtable(
        name="tableS7_emotion_lexicon.tex",
        caption="Emotion states included in the final lexicon and their assigned emotion cluster, with detailed V-A-C-U-S appraisal scores",
        label="tab:s7-emotions",
        colspec=r"@{}p{0.30\textwidth}rrrrr",
        header=r"Emotion & V & A & C & U & S \\",
        body_lines=body_lines,
        n_cols=6,
    )
    make_longtable(
        name="tableS8_emotion_lexicon.tex",
        caption="Emotion states in the final lexicon with affective component scores (V = Valence, A = Arousal, C = Control/Coping, U = Uncertainty, S = Social Orientation)",
        label="tab:s8-emotions",
        colspec=r"@{}p{0.30\textwidth}rrrrr",
        header=r"Emotion & Valence (V) & Arousal (A) & Control (C) & Uncertainty (U) & Social Orientation (S) \\",
        body_lines=body_lines,
        n_cols=6,
    )

    # Supplementary Table S10: full hierarchical leaf inventory
    tax_counts = taxonomy_df.groupby(["family", "cluster"]).size().rename("cluster_n").reset_index()
    fam_counts = taxonomy_df.groupby("family").size().rename("family_n").reset_index()
    s10_df = taxonomy_df.merge(tax_counts, on=["family", "cluster"], how="left")
    s10_df = s10_df.merge(fam_counts, on="family", how="left")
    s10_df = s10_df.sort_values(["family", "cluster", "leaf_label"]) 

    body_s10: list[str] = []
    for fam in [f for f in SHORT if f in s10_df["family"].unique()]:
        fam_sub = s10_df[s10_df["family"] == fam]
        body_s10.append(
            rf"\addlinespace[2pt]\multicolumn{{5}}{{@{{}}l}}{{\textbf{{{esc(SHORT[fam])}}} ($n_{{\mathrm{{leaves}}}}={len(fam_sub)}$)}} \\"
        )
        for row in fam_sub.itertuples(index=False):
            body_s10.append(
                f"{esc(str(row.cluster_label))} & {esc(str(row.leaf_label))} & {int(row.cluster_n)} & {int(row.family_n)} & {esc(str(row.leaf_bias))} \\\\"
            )

    make_longtable(
        name="tableS10_taxonomy_leaf_inventory.tex",
        caption="Full hierarchical cognitive-bias inventory at leaf level",
        label="tab:s10-taxonomy",
        colspec=r"@{}p{0.26\textwidth}p{0.26\textwidth}rrp{0.26\textwidth}@{}",
        header=r"Cluster & Leaf bias label & Cluster $n$ & Family $n$ & Canonical key \\",
        body_lines=body_s10,
        n_cols=5,
    )

    print("All tables generated successfully.")


if __name__ == "__main__":
    main()
