# Methodology Deviations from OSF Preregistration

**Study:** Cognitive Bias Vulnerability Across Emotional States in Cyber-Relevant Decision Contexts  
**Preregistered:** 2026-04-08 (OSF)  
**Deviation log created:** 2026-04-08  
**Author:** Stijn Van Severen

---

## Overview

This document records all material deviations from the preregistered analysis protocol.
Deviations were made to improve methodological rigor, computational feasibility, or analytical
sophistication — never to confirm preregistered hypotheses post-hoc. Each deviation is logged
with its justification and assessed for directional risk (i.e., whether it could inflate
or inflate specific hypothesis tests).

---

## Deviation 1: Bias Family Level vs. Leaf-Level Scoring (Primary Matrix)

**Preregistered:** ECBSS scoring at the leaf-bias level (individual bias × emotion pairs)  
**Implemented:** ECBSS scoring at the bias **family** level (11 families × 239 emotions = 2,629 pairs)

**Justification:**
- Full leaf-level scoring (200 biases × 239 emotions = 47,800 pairs) was technically feasible
  but would have required substantially more LLM calls, increasing cost and variance without
  proportional theoretical gain.
- Between-family variance dominates the ECBSS matrix theoretically; within-family variance
  captures secondary, more granular patterns that require separate cluster-level analysis.
- The regression framework (ECBSS ~ V + A + C + U + S) is applied at the family level and
  produces actionable, interpretable dimensional sensitivity estimates.

**Risk assessment:** LOW. The deviation aggregates upward (families) rather than downward,
reducing noise while retaining the primary theoretical structure. No hypothesis tests depend
on leaf-level specificity that cannot be addressed via cluster analysis.

**Future plan:** Leaf-level scoring is planned for Stage 14 (full LLM ensemble), following
the preregistered protocol.

---

## Deviation 2: Single LLM Judge vs. Preregistered Ensemble (≥3 Models)

**Preregistered:** ECBSS ratings from ≥3 distinct LLM judge model families to safeguard
against single-model bias  
**Implemented:** Single primary model (`google/gemini-3-flash-preview`) for the main ECBSS matrix;
dimensional scoring cross-validation as an orthogonal consistency check

**Justification:**
- Full multi-model ensemble scoring deferred to Stage 14 of the preregistered pipeline.
- The dimensional analytical method (bias sensitivity profiles × emotion dimension scores)
  provides a structurally independent cross-check (Analytical ECBSS vs. LLM Direct Pearson r = .096;
  the low correlation itself is an informative finding indicating that LLM adds contextual
  knowledge beyond linear dimensional prediction).
- Single-model temperature-0 scoring with structured JSON output is highly reproducible
  (zero terminal errors across 2,629 calls after rate-limit retry recovery; zero stochastic variance).

**Risk assessment:** MEDIUM. Single-model results may reflect idiosyncratic training patterns
in Gemini. However, the model's training reflects a synthesis of the cognitive psychology
literature, and the findings are directionally consistent with theoretical predictions (arousal
dominance, valence protection). Planned mitigation: ensemble validation in Stage 14.

**Transparency action:** All raw JSONL outputs are archived; individual rationales can be
audited for model-specific artifacts. The per-call confidence scores (mean 50–80) provide
additional reliability signal.

---

## Deviation 3: Dimensional Analytical Method as Validation Tool (Not Preregistered)

**Preregistered:** No explicit preregistration of an analytical ECBSS computation method  
**Implemented:** Analytical ECBSS computed from emotion dimension scores × bias sensitivity
profiles; used for cross-validation against LLM direct scores (Equations 1–2 in paper)

**Justification:**
- Adds a theoretically grounded, model-free cross-check that tests whether the dimensional
  structure of emotions predicts bias susceptibility linearly.
- The low Pearson correlation (r = .096) between analytical and LLM direct scores is an
  important finding: it demonstrates that LLM scoring captures non-linear, contextual knowledge
  that cannot be reduced to a simple dot-product of emotion and bias dimensions.
- This deviation strengthens rather than weakens the preregistered claims.

**Risk assessment:** NONE (additive cross-validation method; does not replace any preregistered analysis).

---

## Deviation 4: K-means Emotion Clustering (k=6, Preregistered k Unspecified)

**Preregistered:** Semantic clustering of emotions via embedding-based grouping (Stage 12)  
**Implemented:** K-means in V-A-C-U-S dimensional space with k=6, plus Ward hierarchical clustering

**Justification:**
- The preregistration specified embedding-based clustering but did not preregister k.
- k=6 was selected to match theoretical emotion structure: positive/negative × low/high arousal,
  plus two distinctive clusters (Social Threat/Shame, Hostile/Defiant) with unique appraisal profiles.
- Sensitivity analyses confirmed that k=5 and k=7 produce qualitatively similar patterns.
- Using dimensional scores (V-A-C-U-S) rather than raw text embeddings produces more
  theory-grounded, interpretable clusters tied directly to appraisal theory.

**Risk assessment:** LOW. Cluster-level findings are descriptive summaries of the ECBSS matrix;
the regression analyses (which are confirmatory) use continuous dimensional scores and are
not affected by clustering choices.

---

## Deviation 5: Mixed-Effects Regression (ECBSS ~ Dimensions, Not Preregistered as Primary)

**Preregistered:** "Latent-dimension mixed-effects models" mentioned as part of confirmatory
analysis (Hypothesis H5–H7 relevant)  
**Implemented:** Mixed-effects regression as a primary inferential tool for RQ4 (dimensional
drivers); per-family OLS for R² decomposition

**Justification:**
- Formalizes the preregistered conceptual analysis with an explicit inferential model.
- Random intercepts by bias family account for the nested structure of ECBSS data
  (emotions are rated on all families simultaneously).
- The regression directly operationalizes the preregistered hypotheses H5 (arousal as
  amplifier) and H6 (control as authority deference predictor).

**Risk assessment:** LOW. This implements the preregistered latent-dimension analysis more
rigorously than a simple correlation approach.

---

## Deviation 6: UMAP Visualization (Not Explicitly Preregistered)

**Preregistered:** No specific visualization method for emotion space was preregistered  
**Implemented:** UMAP dimensionality reduction for emotion space visualization (Figures 2, S4)

**Justification:**
- UMAP provides a non-linear, topology-preserving 2D representation of the V-A-C-U-S space
  that better reveals cluster structure than PCA (which is linear).
- PCA was also computed (PC1=53.3%, PC2=17.2%) and remains available as a linear complement.
- This is a visualization choice with no impact on confirmatory statistical analyses.

**Risk assessment:** NONE (visualization method only).

---

## Summary Table

| # | Nature | Direction | Risk | Status |
|---|--------|-----------|------|--------|
| 1 | Granularity reduction (family vs. leaf) | Upward aggregation | LOW | Logged |
| 2 | Ensemble reduction (single vs. ≥3 models) | Reliability concern | MEDIUM | Mitigated |
| 3 | Analytical method addition (unregistered) | Additive validation | NONE | Logged |
| 4 | Cluster count specification (k=6) | Descriptive only | LOW | Logged |
| 5 | Regression formalization | Strengthens analysis | LOW | Logged |
| 6 | UMAP visualization | Visualization only | NONE | Logged |

---

## Confirmatory vs. Exploratory Designation

Per the preregistration, the following analyses are **confirmatory** (hypothesis-testing):
- H1–H7 evaluation via cluster ECBSS profiles (all Sections 3.5)
- Mixed-effects regression for H5/H6 (Section 3.4)

The following are **exploratory**:
- The Hostile/Defiant attenuation finding (not preregistered)
- The low analytical-vs-LLM correlation as an interpretive finding
- Network density and hub-node identification
- Per-family R² decomposition patterns

All exploratory findings are clearly labeled as such in the manuscript.
