"""Feature engineering for the LightGBM meta-learner.

Builds a (user, item) feature matrix by merging CF model scores with LLM
yes_prob scores. Scores are rank-normalised within each user's candidate set
so the meta-learner sees relative rather than absolute values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CF_COLS = ["pop", "knn", "ease", "bpr", "lgcn", "dcn", "neumf", "sasrec"]
LLM_COL = "llm_yes_prob"
ALL_FEAT_COLS = CF_COLS + [LLM_COL]
LLM_DEFAULT = 0.5   # neutral fill when LLM scores are unavailable


def _rank_normalise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace raw scores with per-user rank percentile in [0, 1].

    Within each user's candidate set, the highest-scoring item gets 1.0,
    the lowest gets 0.0. Ties are averaged.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
            continue
        df[col] = (
            df.groupby("user_id")[col]
            .rank(method="average", pct=True)
            .astype(np.float32)
        )
    return df


def build_feature_matrix(
    cf_scores: pd.DataFrame,
    llm_features: pd.DataFrame | None = None,
    normalise: bool = True,
) -> pd.DataFrame:
    """Merge CF scores with LLM features and optionally rank-normalise.

    Args:
        cf_scores:    DataFrame with columns [user_id, item_id, label, pop, knn, …]
        llm_features: DataFrame with columns [user_id, item_id, yes_prob].
                      Pass None to skip (all LLM scores default to 0.5).
        normalise:    If True, replace scores with per-user rank percentiles.

    Returns:
        DataFrame with columns [user_id, item_id, label, pop, knn, ease,
        bpr, lgcn, dcn, neumf, llm_yes_prob].
    """
    df = cf_scores.copy()

    # Merge LLM yes_prob — left join so all CF rows are kept
    if llm_features is not None and not llm_features.empty:
        llm = llm_features[["user_id", "item_id", "yes_prob"]].rename(
            columns={"yes_prob": LLM_COL}
        )
        df = df.merge(llm, on=["user_id", "item_id"], how="left")
    else:
        df[LLM_COL] = np.nan

    df[LLM_COL] = df[LLM_COL].fillna(LLM_DEFAULT).astype(np.float32)

    # Ensure all CF columns exist (fill missing models with 0)
    for col in CF_COLS:
        if col not in df.columns:
            df[col] = 0.0

    if normalise:
        df = _rank_normalise(df, CF_COLS + [LLM_COL])

    return df


def split_Xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays ready for LightGBM training."""
    X = df[ALL_FEAT_COLS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    return X, y
