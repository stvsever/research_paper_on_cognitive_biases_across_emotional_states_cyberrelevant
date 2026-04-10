"""Advanced statistical analysis module for the cognitive bias vulnerability study.

Provides factor analysis, canonical correlation, bootstrap CIs, silhouette validation,
permutation tests, variance decomposition, Cohen's d, full PCA, and ICC computations.

All results are persisted to:
  src/review_stages/analysis_outputs/

Usage
-----
    from analysis.advanced_analysis import run_all_advanced
    results = run_all_advanced()

Or call individual functions directly after loading data via the project's
``data_loader`` and ``statistical_analysis`` modules.

Python compatibility: 3.9+
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import svd
from scipy.stats import f as f_dist
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "src/review_stages/analysis_outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with rows containing any NaN removed."""
    return df.dropna()


def _numeric_values(df: pd.DataFrame) -> np.ndarray:
    """Return float64 numpy array from a DataFrame, coercing types."""
    return df.values.astype(np.float64)


def _save_json(data: dict, filename: str) -> None:
    """Persist *data* as pretty-printed JSON to the shared outputs folder."""
    path = OUTPUTS / filename
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=float)


def _normalise_dims(emotion_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise V, A, C, U, S columns from the ±1000 scale to unit scale."""
    cols = [c for c in ["V", "A", "C", "U", "S"] if c in emotion_df.columns]
    out = emotion_df[cols].copy().astype(float)
    out = out / 1000.0
    return out


# ---------------------------------------------------------------------------
# 1. Factor Analysis of ECBSS matrix
# ---------------------------------------------------------------------------

def run_factor_analysis(
    ecbss_df: pd.DataFrame,
    n_factors: int = 3,
) -> dict:
    """Factor analysis of the ECBSS matrix to identify latent bias dimensions.

    Uses ``sklearn.decomposition.FactorAnalysis`` (maximum-likelihood via EM).
    Rows are emotions; columns are bias families.  The analysis is performed
    on the transposed matrix so that *factors* describe variance across
    *emotions* for each bias family (i.e. latent emotional loading patterns).

    Parameters
    ----------
    ecbss_df:
        DataFrame of shape (N_emotions × N_families) with ECBSS values.
    n_factors:
        Number of latent factors to extract (default 3).

    Returns
    -------
    dict with keys:
        ``loadings``        – dict[family -> list[float]] of factor loadings
        ``variance_explained`` – list[float] per factor
        ``communalities``   – dict[family -> float]
        ``n_factors``       – int
        ``n_obs``           – int (number of emotions used)
        ``families``        – list[str]
    """
    clean = _drop_na_rows(ecbss_df.T)  # (N_families × N_emotions) after drop
    if clean.shape[0] < n_factors:
        warnings.warn(
            f"run_factor_analysis: only {clean.shape[0]} complete observations; "
            f"reducing n_factors from {n_factors} to {clean.shape[0] - 1}."
        )
        n_factors = max(1, clean.shape[0] - 1)

    families = clean.index.tolist()
    X = StandardScaler().fit_transform(_numeric_values(clean))

    fa = FactorAnalysis(n_components=n_factors, random_state=42, max_iter=1000)
    fa.fit(X)

    # Loadings: (N_families × n_factors)
    loadings_arr = fa.components_.T  # sklearn stores (n_factors × n_features)
    # components_ is (n_factors × n_families) → transpose → (n_families × n_factors)
    loadings_arr = fa.components_.T  # shape: (N_families, n_factors)

    loadings_dict: dict[str, list[float]] = {
        fam: loadings_arr[i].tolist()
        for i, fam in enumerate(families)
    }

    # Variance explained per factor: sum of squared loadings / n_variables
    ss_loadings = np.sum(loadings_arr ** 2, axis=0)  # (n_factors,)
    variance_explained = (ss_loadings / len(families)).tolist()

    # Communalities: sum of squared loadings across factors per variable
    communalities = {
        fam: float(np.sum(loadings_arr[i] ** 2))
        for i, fam in enumerate(families)
    }

    result: dict = {
        "n_factors": n_factors,
        "n_obs": int(clean.shape[1]),
        "families": families,
        "loadings": loadings_dict,
        "variance_explained": variance_explained,
        "communalities": communalities,
    }

    _save_json(result, "factor_analysis.json")
    return result


# ---------------------------------------------------------------------------
# 2. Canonical Correlation Analysis
# ---------------------------------------------------------------------------

def run_canonical_correlation(
    emotion_df: pd.DataFrame,
    ecbss_df: pd.DataFrame,
) -> dict:
    """Canonical correlation between V-A-C-U-S dimensions and ECBSS profiles.

    Implemented directly via SVD of the cross-covariance matrix following
    Hotelling (1936).  Both data sets are standardised before decomposition.

    Parameters
    ----------
    emotion_df:
        DataFrame with emotion dimensional scores; must contain columns
        V, A, C, U, S (values in any scale; internally normalised).
    ecbss_df:
        DataFrame (N_emotions × N_families) with ECBSS values.

    Returns
    -------
    dict with keys:
        ``canonical_correlations``  – list[float] of r_c values (descending)
        ``n_pairs``                 – int
        ``x_coefficients``          – list[list[float]], shape (n_dims × n_cc)
        ``y_coefficients``          – list[list[float]], shape (n_fam × n_cc)
        ``x_canonical_variates``    – dict[dim -> list[float]], the canonical scores
        ``y_canonical_variates``    – dict[family -> list[float]]
        ``wilks_lambda``            – list[float], Wilks' Lambda per root
        ``chi2``                    – list[float]
        ``df``                      – list[int]
        ``p_values``                – list[float]
    """
    dim_cols = [c for c in ["V", "A", "C", "U", "S"] if c in emotion_df.columns]
    common_idx = emotion_df.index.intersection(ecbss_df.index)
    if len(common_idx) == 0:
        raise ValueError("run_canonical_correlation: no common emotion indices.")

    X_raw = emotion_df.loc[common_idx, dim_cols].copy().astype(float)
    Y_raw = ecbss_df.loc[common_idx].copy().astype(float)

    # Drop rows with any NaN in either set
    combined = pd.concat([X_raw, Y_raw], axis=1).dropna()
    X_raw = combined.iloc[:, : len(dim_cols)]
    Y_raw = combined.iloc[:, len(dim_cols):]

    n, p = X_raw.shape
    q = Y_raw.shape[1]
    k = min(p, q)

    X = StandardScaler().fit_transform(X_raw.values.astype(float))
    Y = StandardScaler().fit_transform(Y_raw.values.astype(float))

    # Cross-covariance matrix
    Sxy = (X.T @ Y) / (n - 1)       # (p × q)
    Sxx = (X.T @ X) / (n - 1)       # (p × p)
    Syy = (Y.T @ Y) / (n - 1)       # (q × q)

    # Regularise to avoid singular matrices
    eps = 1e-8
    Sxx_inv_sqrt = np.linalg.pinv(
        np.linalg.cholesky(Sxx + eps * np.eye(p))
    )
    Syy_inv_sqrt = np.linalg.pinv(
        np.linalg.cholesky(Syy + eps * np.eye(q))
    )

    M = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt  # (p × q)

    U_svd, canonical_corrs, Vt_svd = svd(M, full_matrices=False)
    canonical_corrs = np.clip(canonical_corrs[:k], 0.0, 1.0)

    # Canonical coefficients (back to original standardised space)
    A = Sxx_inv_sqrt @ U_svd[:, :k]   # X-side  (p × k)
    B = Syy_inv_sqrt @ Vt_svd[:k].T   # Y-side  (q × k)

    # Canonical variates
    U_var = X @ A   # (n × k)
    V_var = Y @ B   # (n × k)

    # Wilks' Lambda and approximate chi-square tests (Bartlett approximation)
    wilks_lambda, chi2_vals, df_vals, p_vals = [], [], [], []
    for i in range(k):
        lam = float(np.prod(1.0 - canonical_corrs[i:] ** 2))
        wilks_lambda.append(lam)
        chi_sq = float(-(n - 1 - 0.5 * (p + q + 1)) * np.log(max(lam, 1e-300)))
        df_val = int((p - i) * (q - i))
        chi2_vals.append(chi_sq)
        df_vals.append(df_val)
        p_vals.append(float(stats.chi2.sf(chi_sq, df_val)))

    result: dict = {
        "n_pairs": int(n),
        "n_x_vars": int(p),
        "n_y_vars": int(q),
        "n_canonical_pairs": int(k),
        "canonical_correlations": canonical_corrs.tolist(),
        "x_coefficients": A.tolist(),
        "y_coefficients": B.tolist(),
        "x_canonical_variates": {
            dim_cols[i]: U_var[:, j].tolist()
            for j in range(k)
            for i in range(p)
            if j == i and i < p  # diagonal: each dim's first-pair variate
        },
        "y_canonical_variates": {
            fam: V_var[:, j].tolist()
            for j, fam in enumerate(Y_raw.columns[:k])
        },
        "wilks_lambda": wilks_lambda,
        "chi2": chi2_vals,
        "df": df_vals,
        "p_values": p_vals,
        "dim_names": dim_cols,
        "family_names": Y_raw.columns.tolist(),
    }

    _save_json(result, "canonical_correlation.json")
    return result


# ---------------------------------------------------------------------------
# 3. Bootstrap Cluster Means
# ---------------------------------------------------------------------------

def bootstrap_cluster_means(
    ecbss_df: pd.DataFrame,
    cluster_labels: pd.Series,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap 95 % confidence intervals on cluster-level mean ECBSS per family.

    For each (emotion cluster, bias family) cell, resamples the emotions
    belonging to that cluster with replacement and computes the mean ECBSS,
    repeating *n_bootstrap* times to build a sampling distribution.

    Parameters
    ----------
    ecbss_df:
        DataFrame (N_emotions × N_families).
    cluster_labels:
        Series mapping emotion → cluster id, indexed by emotion name.
    n_bootstrap:
        Number of bootstrap replications (default 1000).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict: cluster_id (str) -> family (str) -> {mean, ci_lower, ci_upper, n}
    """
    rng = np.random.default_rng(seed)
    clean = _drop_na_rows(ecbss_df)
    aligned_labels = cluster_labels.loc[cluster_labels.index.intersection(clean.index)]
    clean = clean.loc[aligned_labels.index]

    cluster_ids = sorted(aligned_labels.unique())
    families = clean.columns.tolist()

    results: dict = {}
    for cid in cluster_ids:
        mask = aligned_labels == cid
        group = clean.loc[mask]
        n = len(group)
        if n == 0:
            continue
        cid_key = str(cid)
        results[cid_key] = {}
        for fam in families:
            vals = group[fam].dropna().values
            if len(vals) == 0:
                results[cid_key][fam] = {"mean": None, "ci_lower": None, "ci_upper": None, "n": 0}
                continue
            boot_means = np.array([
                rng.choice(vals, size=len(vals), replace=True).mean()
                for _ in range(n_bootstrap)
            ])
            results[cid_key][fam] = {
                "mean": float(np.mean(vals)),
                "ci_lower": float(np.percentile(boot_means, 2.5)),
                "ci_upper": float(np.percentile(boot_means, 97.5)),
                "n": int(len(vals)),
            }

    _save_json(results, "bootstrap_cluster_means.json")
    return results


# ---------------------------------------------------------------------------
# 4. Silhouette Analysis
# ---------------------------------------------------------------------------

def compute_silhouette_analysis(
    emotion_df: pd.DataFrame,
    k_range: range = range(2, 9),
) -> dict:
    """Compute silhouette scores for k = 2 … 8 to validate the k = 6 choice.

    Scales the V, A, C, U, S features then runs K-Means with each k
    (n_init = 20, random_state = 42) and records the silhouette score.

    Parameters
    ----------
    emotion_df:
        DataFrame with V, A, C, U, S columns.
    k_range:
        Range of k values to evaluate (default range(2, 9)).

    Returns
    -------
    dict with keys:
        ``scores``      – dict[k (str) -> float]
        ``optimal_k``   – int (k achieving highest silhouette score)
        ``scores_list`` – list[dict] ordered by k, for easy serialisation
    """
    dim_cols = [c for c in ["V", "A", "C", "U", "S"] if c in emotion_df.columns]
    clean = emotion_df[dim_cols].dropna()
    X = StandardScaler().fit_transform(clean.values.astype(float))

    scores: dict[str, float] = {}
    for k in k_range:
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[str(k)] = float(score)

    if not scores:
        raise ValueError("compute_silhouette_analysis: no valid k values produced scores.")

    optimal_k = int(max(scores, key=lambda k: scores[k]))
    scores_list = [{"k": int(k), "silhouette_score": v} for k, v in sorted(scores.items(), key=lambda x: int(x[0]))]

    result: dict = {
        "scores": scores,
        "optimal_k": optimal_k,
        "scores_list": scores_list,
        "k_range": [int(k) for k in k_range],
    }

    _save_json(result, "silhouette_analysis.json")
    return result


# ---------------------------------------------------------------------------
# 5. Permutation Test for Regression Betas
# ---------------------------------------------------------------------------

def run_permutation_test_regression(
    ecbss_df: pd.DataFrame,
    emotion_df: pd.DataFrame,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test for regression betas: ECBSS ~ V + A + C + U + S.

    The ECBSS vector (stacked across families) is shuffled relative to the
    dimensional predictors, the OLS beta is recomputed, and the empirical
    p-value is the fraction of permuted betas that meet or exceed the
    observed beta in absolute value.

    Parameters
    ----------
    ecbss_df:
        DataFrame (N_emotions × N_families).
    emotion_df:
        DataFrame with V, A, C, U, S columns.
    n_permutations:
        Number of permutation draws (default 1000).
    seed:
        Random seed.

    Returns
    -------
    dict with keys:
        ``observed_betas``  – dict[dim -> float]
        ``p_values``        – dict[dim -> float]
        ``null_distributions`` – dict[dim -> list[float]] (the permuted betas)
        ``n_permutations``  – int
        ``n_obs``           – int
    """
    rng = np.random.default_rng(seed)
    dim_cols = [c for c in ["V", "A", "C", "U", "S"] if c in emotion_df.columns]

    common_idx = emotion_df.index.intersection(ecbss_df.index)
    X_raw = emotion_df.loc[common_idx, dim_cols].copy().astype(float)
    Y_raw = ecbss_df.loc[common_idx].copy().astype(float)

    combined = pd.concat([X_raw, Y_raw], axis=1).dropna()
    X_raw = combined.iloc[:, :len(dim_cols)].values
    Y_raw = combined.iloc[:, len(dim_cols):].values

    n_obs = len(X_raw)
    n_fam = Y_raw.shape[1]

    # Stack: each row = one (emotion, family) observation
    X_stack = np.tile(X_raw, (n_fam, 1))             # (n_obs*n_fam, n_dims)
    Y_stack = Y_raw.T.reshape(-1)                      # (n_obs*n_fam,)

    # Normalise predictors for numerical stability
    X_scale = X_stack / 1000.0

    # OLS via normal equations: beta = (X'X)^{-1} X'y
    X_design = np.column_stack([np.ones(len(X_scale)), X_scale])  # add intercept
    XtX_inv = np.linalg.pinv(X_design.T @ X_design)
    observed_betas_arr = XtX_inv @ X_design.T @ Y_stack          # (n_dims+1,)

    observed_betas = {dim_cols[i]: float(observed_betas_arr[i + 1]) for i in range(len(dim_cols))}

    # Permutation null distributions
    null_dists: dict[str, list[float]] = {d: [] for d in dim_cols}

    for _ in range(n_permutations):
        perm_idx = rng.permutation(len(Y_stack))
        Y_perm = Y_stack[perm_idx]
        betas_perm = XtX_inv @ X_design.T @ Y_perm
        for i, d in enumerate(dim_cols):
            null_dists[d].append(float(betas_perm[i + 1]))

    p_values: dict[str, float] = {}
    for d in dim_cols:
        obs_abs = abs(observed_betas[d])
        null_arr = np.array(null_dists[d])
        p_values[d] = float(np.mean(np.abs(null_arr) >= obs_abs))

    result: dict = {
        "observed_betas": observed_betas,
        "p_values": p_values,
        "null_distributions": null_dists,
        "n_permutations": n_permutations,
        "n_obs": n_obs,
        "dim_cols": dim_cols,
    }

    _save_json(result, "permutation_test_regression.json")
    return result


# ---------------------------------------------------------------------------
# 6. Variance Decomposition
# ---------------------------------------------------------------------------

def compute_variance_decomposition(
    ecbss_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    cluster_labels: pd.Series,
) -> dict:
    """ANOVA-style variance decomposition of ECBSS.

    Decomposes total ECBSS variance into:
    - Between-emotion-clusters  (emotion cluster factor)
    - Between-bias-families     (bias family factor)
    - Residual

    Uses Type-I (sequential) sum-of-squares analogous to a two-way ANOVA
    without interaction (because interaction cannot be estimated when each
    cell contains a single ECBSS value).

    Parameters
    ----------
    ecbss_df:
        DataFrame (N_emotions × N_families).
    taxonomy_df:
        DataFrame with a ``family`` column (used to validate family names).
    cluster_labels:
        Series mapping emotion name → cluster id.

    Returns
    -------
    dict with keys (one entry per factor plus ``residual`` and ``total``):
        Each factor entry: {SS, df, MS, F, p, eta_squared}
    """
    clean = _drop_na_rows(ecbss_df)
    aligned_labels = cluster_labels.loc[cluster_labels.index.intersection(clean.index)]
    clean = clean.loc[aligned_labels.index]

    families = clean.columns.tolist()
    emotions = clean.index.tolist()
    n_emo = len(emotions)
    n_fam = len(families)
    N = n_emo * n_fam

    # Build long-format arrays
    y = clean.values.flatten().astype(float)          # (N,)
    cluster_idx = np.array([aligned_labels[e] for e in emotions])
    cluster_vec = np.repeat(cluster_idx, n_fam)       # (N,) — emotion cluster per obs
    family_vec = np.tile(np.arange(n_fam), n_emo)     # (N,) — family index per obs

    grand_mean = y.mean()

    # --- Between-cluster SS ---
    cluster_means = np.array([y[cluster_vec == c].mean() for c in np.unique(cluster_vec)])
    cluster_counts = np.array([np.sum(cluster_vec == c) for c in np.unique(cluster_vec)])
    ss_cluster = float(np.sum(cluster_counts * (cluster_means - grand_mean) ** 2))
    df_cluster = int(len(np.unique(cluster_vec)) - 1)

    # --- Between-family SS ---
    family_means = np.array([y[family_vec == f].mean() for f in range(n_fam)])
    family_counts = np.full(n_fam, n_emo)
    ss_family = float(np.sum(family_counts * (family_means - grand_mean) ** 2))
    df_family = int(n_fam - 1)

    # --- Total SS ---
    ss_total = float(np.sum((y - grand_mean) ** 2))
    df_total = N - 1

    # --- Residual SS ---
    ss_residual = ss_total - ss_cluster - ss_family
    df_residual = df_total - df_cluster - df_family
    df_residual = max(df_residual, 1)

    ms_residual = ss_residual / df_residual

    # F-statistics and p-values
    ms_cluster = ss_cluster / df_cluster if df_cluster > 0 else 0.0
    ms_family = ss_family / df_family if df_family > 0 else 0.0

    f_cluster = float(ms_cluster / ms_residual) if ms_residual > 0 else 0.0
    f_family = float(ms_family / ms_residual) if ms_residual > 0 else 0.0

    p_cluster = float(f_dist.sf(f_cluster, df_cluster, df_residual))
    p_family = float(f_dist.sf(f_family, df_family, df_residual))

    # Eta-squared (partial)
    eta2_cluster = float(ss_cluster / (ss_cluster + ss_residual)) if (ss_cluster + ss_residual) > 0 else 0.0
    eta2_family = float(ss_family / (ss_family + ss_residual)) if (ss_family + ss_residual) > 0 else 0.0

    result: dict = {
        "emotion_cluster": {
            "SS": ss_cluster,
            "df": df_cluster,
            "MS": float(ms_cluster),
            "F": f_cluster,
            "p": p_cluster,
            "eta_squared": eta2_cluster,
        },
        "bias_family": {
            "SS": ss_family,
            "df": df_family,
            "MS": float(ms_family),
            "F": f_family,
            "p": p_family,
            "eta_squared": eta2_family,
        },
        "residual": {
            "SS": float(ss_residual),
            "df": int(df_residual),
            "MS": float(ms_residual),
        },
        "total": {
            "SS": float(ss_total),
            "df": int(df_total),
            "N": N,
        },
        "grand_mean": float(grand_mean),
        "n_emotions": int(n_emo),
        "n_families": int(n_fam),
    }

    _save_json(result, "variance_decomposition.json")
    return result


# ---------------------------------------------------------------------------
# 7. Cohen's d Effect Size
# ---------------------------------------------------------------------------

def cohen_d_hypothesis(
    ecbss_df: pd.DataFrame,
    emotions_group: list[str],
    reference_emotions: list[str],
    family_key: str,
) -> dict:
    """Cohen's d effect size for a directional hypothesis.

    Compares the mean ECBSS of *emotions_group* against *reference_emotions*
    for the given *family_key*.  Uses the pooled standard deviation.

    Parameters
    ----------
    ecbss_df:
        DataFrame (N_emotions × N_families).
    emotions_group:
        Emotions of theoretical interest (e.g. high-arousal negative).
    reference_emotions:
        Baseline emotions for comparison.
    family_key:
        Column name in *ecbss_df* to analyse.

    Returns
    -------
    dict with keys:
        ``cohen_d``, ``mean_group``, ``mean_reference``,
        ``std_group``, ``std_reference``, ``pooled_std``,
        ``n_group``, ``n_reference``,
        ``t_statistic``, ``t_p_value`` (Welch's t-test),
        ``interpretation`` – "small" / "medium" / "large" / "very large"
    """
    if family_key not in ecbss_df.columns:
        raise KeyError(f"cohen_d_hypothesis: '{family_key}' not found in ecbss_df.")

    col = ecbss_df[family_key].dropna()

    avail_group = [e for e in emotions_group if e in col.index]
    avail_ref = [e for e in reference_emotions if e in col.index]

    if not avail_group or not avail_ref:
        raise ValueError(
            "cohen_d_hypothesis: insufficient overlapping emotions in ecbss_df index."
        )

    a = col.loc[avail_group].values.astype(float)
    b = col.loc[avail_ref].values.astype(float)

    mean_a, mean_b = a.mean(), b.mean()
    std_a = a.std(ddof=1) if len(a) > 1 else 0.0
    std_b = b.std(ddof=1) if len(b) > 1 else 0.0

    # Pooled SD (Hedges convention)
    pooled_std = np.sqrt(((len(a) - 1) * std_a ** 2 + (len(b) - 1) * std_b ** 2) /
                         (len(a) + len(b) - 2)) if (len(a) + len(b) > 2) else 1.0
    pooled_std = max(pooled_std, 1e-9)

    d = float((mean_a - mean_b) / pooled_std)

    # Welch's t-test
    t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)

    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    elif abs_d < 1.2:
        interpretation = "large"
    else:
        interpretation = "very large"

    result: dict = {
        "family_key": family_key,
        "cohen_d": d,
        "mean_group": float(mean_a),
        "mean_reference": float(mean_b),
        "std_group": float(std_a),
        "std_reference": float(std_b),
        "pooled_std": float(pooled_std),
        "n_group": int(len(a)),
        "n_reference": int(len(b)),
        "t_statistic": float(t_stat),
        "t_p_value": float(t_p),
        "interpretation": interpretation,
        "emotions_group": avail_group,
        "reference_emotions": avail_ref,
    }

    # Save with a name that includes the family key (sanitised)
    safe_key = family_key.replace("/", "_")[:60]
    _save_json(result, f"cohen_d_{safe_key}.json")
    return result


# ---------------------------------------------------------------------------
# 8. Full PCA with Loadings
# ---------------------------------------------------------------------------

def run_pca_with_loadings(
    emotion_df: pd.DataFrame,
) -> dict:
    """Full PCA of the V-A-C-U-S space with loadings and biplot coordinates.

    Performs PCA on all five affective dimensions, retaining all components.
    Scales the data to unit variance before decomposition.

    Parameters
    ----------
    emotion_df:
        DataFrame with V, A, C, U, S columns (any number of emotion rows).

    Returns
    -------
    dict with keys:
        ``coords_2d``               – list[list[float]], shape (N_emotions, 2)
        ``loadings``                – dict[dim -> list[float]], per-component loading
        ``explained_variance_ratio`` – list[float], one per component
        ``cumulative_variance``     – list[float]
        ``scores``                  – list[list[float]], shape (N_emotions, n_components)
        ``biplot_scale``            – float (sqrt of first eigenvalue, for biplot arrows)
        ``emotions``                – list[str]
        ``dim_names``               – list[str]
        ``n_components``            – int
    """
    dim_cols = [c for c in ["V", "A", "C", "U", "S"] if c in emotion_df.columns]
    clean = emotion_df[dim_cols].dropna()
    emotions = clean.index.tolist()

    X = StandardScaler().fit_transform(clean.values.astype(float))

    n_components = len(dim_cols)
    pca = PCA(n_components=n_components, random_state=42)
    scores_arr = pca.fit_transform(X)                # (N_emotions, n_components)

    # Loadings: components_ is (n_components, n_features); transpose for per-dim view
    # Each row of components_ is a principal axis; column = feature loading
    loadings_dict = {
        dim: pca.components_[:, i].tolist()          # list length = n_components
        for i, dim in enumerate(dim_cols)
    }

    evr = pca.explained_variance_ratio_.tolist()
    cumvar = np.cumsum(pca.explained_variance_ratio_).tolist()
    biplot_scale = float(np.sqrt(pca.explained_variance_[0]))

    result: dict = {
        "n_components": n_components,
        "n_emotions": len(emotions),
        "dim_names": dim_cols,
        "emotions": emotions,
        "coords_2d": scores_arr[:, :2].tolist(),
        "scores": scores_arr.tolist(),
        "loadings": loadings_dict,
        "explained_variance_ratio": evr,
        "cumulative_variance": cumvar,
        "biplot_scale": biplot_scale,
        "components": pca.components_.tolist(),      # (n_components, n_features) raw matrix
    }

    _save_json(result, "pca_with_loadings.json")

    # Also save 2-D coordinates as CSV (convenient for plotting)
    coords_df = pd.DataFrame(
        scores_arr[:, :2],
        index=emotions,
        columns=["PC1", "PC2"],
    )
    coords_df.index.name = "emotion"
    coords_df.to_csv(OUTPUTS / "pca_with_loadings_coords.csv")

    return result


# ---------------------------------------------------------------------------
# 9. Intraclass Correlation
# ---------------------------------------------------------------------------

def compute_icc_across_models(
    ecbss_path_primary: str,
    ecbss_path_analytical: str,
) -> dict:
    """Intraclass correlation (ICC) between LLM-direct and analytical ECBSS matrices.

    Implements ICC(2,1) and ICC(3,1) as defined by Shrout & Fleiss (1979):
    - ICC(2,1): Two-way random effects, single measures, absolute agreement
    - ICC(3,1): Two-way mixed effects, single measures, consistency

    The two matrices are aligned on common (emotion, family) cells before
    computing.  Missing values are excluded pairwise.

    Parameters
    ----------
    ecbss_path_primary:
        File path to the primary (LLM-direct) ECBSS CSV.
    ecbss_path_analytical:
        File path to the analytical ECBSS CSV.

    Returns
    -------
    dict with keys:
        ``icc2``            – float, ICC(2,1) absolute agreement
        ``icc3``            – float, ICC(3,1) consistency
        ``icc2_ci_lower``   – float, 95% CI lower bound (F-distribution)
        ``icc2_ci_upper``   – float, 95% CI upper bound
        ``icc3_ci_lower``   – float
        ``icc3_ci_upper``   – float
        ``n_raters``        – int (always 2 here)
        ``n_subjects``      – int (number of aligned cells)
        ``interpretation``  – str
        ``f_value``         – float
        ``df1``             – int
        ``df2``             – int
    """
    p1 = Path(ecbss_path_primary)
    p2 = Path(ecbss_path_analytical)

    if not p1.exists():
        raise FileNotFoundError(f"compute_icc_across_models: primary path not found: {p1}")
    if not p2.exists():
        raise FileNotFoundError(f"compute_icc_across_models: analytical path not found: {p2}")

    df1 = pd.read_csv(p1, index_col=0)
    df2 = pd.read_csv(p2, index_col=0)

    common_idx = df1.index.intersection(df2.index)
    common_cols = df1.columns.intersection(df2.columns)

    if len(common_idx) == 0 or len(common_cols) == 0:
        raise ValueError("compute_icc_across_models: no overlapping rows/columns.")

    Y1 = df1.loc[common_idx, common_cols].values.flatten().astype(float)
    Y2 = df2.loc[common_idx, common_cols].values.flatten().astype(float)

    # Remove pairwise NaN
    mask = ~(np.isnan(Y1) | np.isnan(Y2))
    Y1 = Y1[mask]
    Y2 = Y2[mask]

    n = len(Y1)          # number of subjects (aligned cells)
    k = 2                # number of raters

    # Assemble (n × k) rating matrix
    ratings = np.column_stack([Y1, Y2])   # (n, 2)
    grand_mean = ratings.mean()
    row_means = ratings.mean(axis=1)      # subject means
    col_means = ratings.mean(axis=0)      # rater means

    # ANOVA sums of squares (two-way, balanced)
    ss_total = float(np.sum((ratings - grand_mean) ** 2))
    ss_rows = float(k * np.sum((row_means - grand_mean) ** 2))
    ss_cols = float(n * np.sum((col_means - grand_mean) ** 2))
    ss_error = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_cols = k - 1
    df_error = (n - 1) * (k - 1)
    df_error = max(df_error, 1)

    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols if df_cols > 0 else 0.0
    ms_error = ss_error / df_error

    # ICC(2,1) — absolute agreement (random raters)
    icc2 = float((ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n))

    # ICC(3,1) — consistency (fixed raters)
    icc3 = float((ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error))

    # 95% CI via F-distribution (Shrout & Fleiss, 1979, equation 13)
    alpha = 0.05
    f_lower = ms_rows / ms_error / f_dist.ppf(1 - alpha / 2, df_rows, df_error)
    f_upper = ms_rows / ms_error / f_dist.ppf(alpha / 2, df_rows, df_error)

    # ICC(2,1) CI
    ci2_lower = float((f_lower - 1) / (f_lower + k - 1 + k * (ms_cols - ms_error) / (n * ms_error)))
    ci2_upper = float((f_upper - 1) / (f_upper + k - 1 + k * (ms_cols - ms_error) / (n * ms_error)))

    # ICC(3,1) CI
    ci3_lower = float((f_lower - 1) / (f_lower + k - 1))
    ci3_upper = float((f_upper - 1) / (f_upper + k - 1))

    # Interpretation thresholds (Koo & Mae, 2016)
    icc_ref = icc2
    if icc_ref < 0.5:
        interpretation = "poor"
    elif icc_ref < 0.75:
        interpretation = "moderate"
    elif icc_ref < 0.9:
        interpretation = "good"
    else:
        interpretation = "excellent"

    f_value = float(ms_rows / ms_error) if ms_error > 0 else 0.0

    result: dict = {
        "icc2": icc2,
        "icc3": icc3,
        "icc2_ci_lower": ci2_lower,
        "icc2_ci_upper": ci2_upper,
        "icc3_ci_lower": ci3_lower,
        "icc3_ci_upper": ci3_upper,
        "n_raters": k,
        "n_subjects": int(n),
        "interpretation": interpretation,
        "f_value": f_value,
        "df1": int(df_rows),
        "df2": int(df_error),
        "ms_rows": float(ms_rows),
        "ms_error": float(ms_error),
        "ms_cols": float(ms_cols),
        "primary_path": str(p1),
        "analytical_path": str(p2),
        "n_common_emotions": int(len(common_idx)),
        "n_common_families": int(len(common_cols)),
    }

    _save_json(result, "icc_across_models.json")
    return result


# ---------------------------------------------------------------------------
# 10. Orchestrator: run_all_advanced
# ---------------------------------------------------------------------------

def run_all_advanced(force: bool = False) -> dict:
    """Run all advanced analyses and return a combined results dict.

    Each sub-analysis is skipped (and its cached JSON loaded) if the output
    file already exists *and* ``force=False``.

    Parameters
    ----------
    force:
        When True, all sub-analyses are re-run regardless of existing outputs.

    Returns
    -------
    dict keyed by analysis name, each value is the sub-analysis result dict.
    """
    import sys
    sys.path.insert(0, str(ROOT / "src"))

    # ── Load shared data assets ─────────────────────────────────────────────
    from analysis.data_loader import load_taxonomy, load_emotion_scores
    from analysis.statistical_analysis import cluster_emotions

    _, taxonomy_df = load_taxonomy()
    emotion_df = load_emotion_scores()

    ecbss_direct_path = OUTPUTS / "ecbss_direct.csv"
    ecbss_analytical_path = OUTPUTS / "ecbss_analytical.csv"

    if not ecbss_direct_path.exists():
        raise FileNotFoundError(
            f"run_all_advanced: primary ECBSS not found at {ecbss_direct_path}. "
            "Run run_pipeline.py first."
        )

    ecbss_df = pd.read_csv(ecbss_direct_path, index_col="emotion")

    # Align to emotions present in both dimension scores and ECBSS matrix
    common_idx = emotion_df.index.intersection(ecbss_df.index)
    ecbss_df = ecbss_df.loc[common_idx]
    emotion_df_aligned = emotion_df.loc[common_idx]

    cluster_labels, _ = cluster_emotions(emotion_df_aligned)

    combined: dict = {}

    # ── 1. Factor Analysis ─────────────────────────────────────────────────
    cache_fa = OUTPUTS / "factor_analysis.json"
    if not force and cache_fa.exists():
        with open(cache_fa) as fh:
            combined["factor_analysis"] = json.load(fh)
        print("factor_analysis: loaded from cache")
    else:
        print("Running factor_analysis …")
        combined["factor_analysis"] = run_factor_analysis(ecbss_df, n_factors=3)

    # ── 2. Canonical Correlation ───────────────────────────────────────────
    cache_cca = OUTPUTS / "canonical_correlation.json"
    if not force and cache_cca.exists():
        with open(cache_cca) as fh:
            combined["canonical_correlation"] = json.load(fh)
        print("canonical_correlation: loaded from cache")
    else:
        print("Running canonical_correlation …")
        combined["canonical_correlation"] = run_canonical_correlation(emotion_df_aligned, ecbss_df)

    # ── 3. Bootstrap Cluster Means ─────────────────────────────────────────
    cache_boot = OUTPUTS / "bootstrap_cluster_means.json"
    if not force and cache_boot.exists():
        with open(cache_boot) as fh:
            combined["bootstrap_cluster_means"] = json.load(fh)
        print("bootstrap_cluster_means: loaded from cache")
    else:
        print("Running bootstrap_cluster_means …")
        combined["bootstrap_cluster_means"] = bootstrap_cluster_means(
            ecbss_df, cluster_labels, n_bootstrap=1000, seed=42
        )

    # ── 4. Silhouette Analysis ─────────────────────────────────────────────
    cache_sil = OUTPUTS / "silhouette_analysis.json"
    if not force and cache_sil.exists():
        with open(cache_sil) as fh:
            combined["silhouette_analysis"] = json.load(fh)
        print("silhouette_analysis: loaded from cache")
    else:
        print("Running silhouette_analysis …")
        combined["silhouette_analysis"] = compute_silhouette_analysis(
            emotion_df_aligned, k_range=range(2, 9)
        )

    # ── 5. Permutation Test ────────────────────────────────────────────────
    cache_perm = OUTPUTS / "permutation_test_regression.json"
    if not force and cache_perm.exists():
        with open(cache_perm) as fh:
            combined["permutation_test_regression"] = json.load(fh)
        print("permutation_test_regression: loaded from cache")
    else:
        print("Running permutation_test_regression …")
        combined["permutation_test_regression"] = run_permutation_test_regression(
            ecbss_df, emotion_df_aligned, n_permutations=1000, seed=42
        )

    # ── 6. Variance Decomposition ──────────────────────────────────────────
    cache_var = OUTPUTS / "variance_decomposition.json"
    if not force and cache_var.exists():
        with open(cache_var) as fh:
            combined["variance_decomposition"] = json.load(fh)
        print("variance_decomposition: loaded from cache")
    else:
        print("Running variance_decomposition …")
        combined["variance_decomposition"] = compute_variance_decomposition(
            ecbss_df, taxonomy_df, cluster_labels
        )

    # ── 7. Cohen's d (exemplar hypothesis) ────────────────────────────────
    # Hypothesis: High-arousal negative emotions amplify social-influence biases
    social_family = "social_influence_authority_affiliation_and_identity_biases"
    avail_cols = ecbss_df.columns.tolist()

    if social_family in avail_cols:
        cache_d = OUTPUTS / f"cohen_d_{social_family[:60]}.json"
        if not force and cache_d.exists():
            with open(cache_d) as fh:
                combined["cohen_d"] = json.load(fh)
            print("cohen_d: loaded from cache")
        else:
            print("Running cohen_d …")
            # High-arousal negative emotions (cluster 0 representative)
            hi_ar_neg = [e for e in ["afraid", "panicked", "alarmed", "terrified", "horrified"]
                         if e in ecbss_df.index]
            # Calm positive emotions (cluster 2 representative)
            calm_pos = [e for e in ["calm", "serene", "content", "peaceful", "relaxed"]
                        if e in ecbss_df.index]
            if hi_ar_neg and calm_pos:
                combined["cohen_d"] = cohen_d_hypothesis(
                    ecbss_df, hi_ar_neg, calm_pos, social_family
                )
            else:
                # Fallback: use all emotions in each cluster
                hi_ar_neg = cluster_labels[cluster_labels == 0].index.tolist()
                calm_pos = cluster_labels[cluster_labels == 2].index.tolist()
                combined["cohen_d"] = cohen_d_hypothesis(
                    ecbss_df, hi_ar_neg, calm_pos, social_family
                )
    else:
        print(f"cohen_d: skipped (family '{social_family}' not in ecbss_df columns)")
        combined["cohen_d"] = {}

    # ── 8. PCA with Loadings ───────────────────────────────────────────────
    cache_pca = OUTPUTS / "pca_with_loadings.json"
    if not force and cache_pca.exists():
        with open(cache_pca) as fh:
            combined["pca_with_loadings"] = json.load(fh)
        print("pca_with_loadings: loaded from cache")
    else:
        print("Running pca_with_loadings …")
        combined["pca_with_loadings"] = run_pca_with_loadings(emotion_df_aligned)

    # ── 9. ICC ─────────────────────────────────────────────────────────────
    cache_icc = OUTPUTS / "icc_across_models.json"
    if not force and cache_icc.exists():
        with open(cache_icc) as fh:
            combined["icc_across_models"] = json.load(fh)
        print("icc_across_models: loaded from cache")
    elif ecbss_analytical_path.exists():
        print("Running icc_across_models …")
        combined["icc_across_models"] = compute_icc_across_models(
            str(ecbss_direct_path),
            str(ecbss_analytical_path),
        )
    else:
        print("icc_across_models: skipped (analytical ECBSS not found)")
        combined["icc_across_models"] = {}

    # ── Save combined summary ──────────────────────────────────────────────
    summary: dict = {}
    for key, val in combined.items():
        # Store only scalar top-level fields for the summary (avoid huge lists)
        scalars = {k: v for k, v in val.items()
                   if isinstance(v, (int, float, str, bool, type(None)))} if isinstance(val, dict) else {}
        summary[key] = scalars

    _save_json(summary, "advanced_analysis_summary.json")
    print(f"\nAll advanced analyses complete. Outputs in {OUTPUTS}")
    return combined
