"""Fix wide tables for LaTeX layout."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

OUTPUTS = ROOT / "src/review_stages/analysis_outputs"
TABLES = ROOT / "paper/assets/tables"

from analysis.data_loader import get_family_short_labels

SHORT = get_family_short_labels()
cluster_ecbss = pd.read_csv(OUTPUTS / "cluster_ecbss.csv", index_col="emotion_cluster")
regression_results = json.load(open(OUTPUTS / "regression_results.json"))
DIMS = ["V", "A", "C", "U", "S"]


def esc(s):
    return s.replace("&", r"\&").replace("\n", " ")


def nl():
    return " \\\\"


# ── Split Table S1 into two halves ────────────────────────────────────────────
families_ordered = [f for f in SHORT if f in cluster_ecbss.columns]
cnames = {
    0: "Hi-Ar.-Neg.", 1: "Lo-Val.-Wth.", 2: "Calm Pos.",
    3: "Hi-Ar.-Pos.", 4: "Soc.Threat", 5: "Hostile",
}
half = len(families_ordered) // 2 + 1


def make_half(families):
    short_fams = [esc(SHORT.get(f, f[:12])) for f in families]
    lines = []
    lines.append(r"\begin{tabular}{@{}l" + "r" * len(families) + "@{}}")
    lines.append(r"\toprule")
    lines.append("Cluster & " + " & ".join(short_fams) + nl())
    lines.append(r"\midrule")
    for cid in sorted(cluster_ecbss.index):
        cname = cnames.get(int(cid), f"C{cid}")
        vals = [
            f"{cluster_ecbss.loc[cid, f]:+.0f}" if f in cluster_ecbss.columns else "--"
            for f in families
        ]
        lines.append(cname + " & " + " & ".join(vals) + nl())
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


(TABLES / "tableS1a_cluster_ecbss.tex").write_text(make_half(families_ordered[:half]))
(TABLES / "tableS1b_cluster_ecbss.tex").write_text(make_half(families_ordered[half:]))
print("Split cluster ECBSS tables (S1a, S1b)")

# ── Fix Table S1 to use the split version ─────────────────────────────────────
# Also rewrite the old tableS1 to point to the two halves
(TABLES / "tableS1_cluster_ecbss.tex").write_text(make_half(families_ordered[:6]))
print("Fixed tableS1 to 6 columns")

# ── Fix Table 3 (regression) ──────────────────────────────────────────────────
per_family = regression_results.get("per_family", {})
lines = []
lines.append(r"\begin{tabular}{@{}l" + "r" * (len(DIMS) + 1) + "@{}}")
lines.append(r"\toprule")
lines.append("Family & " + " & ".join(f"$\\beta_{{{d}}}$" for d in DIMS) + r" & $R^2$" + nl())
lines.append(r"\midrule")

for fam_key in [f for f in SHORT if f in per_family]:
    res = per_family[fam_key]
    params = res.get("params", {})
    pvals = res.get("pvalues", {})
    r2 = res.get("rsquared", 0)
    fam_short = esc(SHORT.get(fam_key, fam_key[:15]))
    betas = []
    for d in DIMS:
        b = params.get(d, 0)
        p = pvals.get(d, 1)
        if p < 0.001:
            sig = "^{***}"
        elif p < 0.01:
            sig = "^{**}"
        elif p < 0.05:
            sig = "^{*}"
        else:
            sig = ""
        cell = f"${b:.0f}{sig}$"
        betas.append(cell)
    lines.append(fam_short + " & " + " & ".join(betas) + f" & {r2:.2f}" + nl())

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
(TABLES / "table3_regression.tex").write_text("\n".join(lines) + "\n")
print("Fixed table3_regression")
print("All fixes applied")
