"""Enhanced feature engineering for the meta-learner.

Adds user-level, item-level, and interaction-level features on top of
the raw CF model scores, giving LambdaRank more signal to rank with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cf_pipeline.features import CF_COLS, LLM_COL, LLM_DEFAULT

# All feature columns for the enhanced meta-learner
ENHANCED_FEAT_COLS = CF_COLS + [
    LLM_COL,
    # User-level
    "user_n_interactions",      # log1p of user's training history length
    "user_avg_item_pop",        # mean popularity of items in user's history
    # Item-level
    "item_n_interactions",      # log1p of item's global interaction count
    "item_pop_rank_pct",        # item's popularity percentile [0,1]
]


def build_stats(train: pd.DataFrame, items: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-user and per-item statistics from the training set.

    Returns:
        user_stats: DataFrame indexed by user_id with columns [user_n_interactions, user_avg_item_pop]
        item_stats: DataFrame indexed by item_id with columns [item_n_interactions, item_pop_rank_pct]
    """
    # Item interaction counts
    item_counts = train.groupby("item_id").size().reset_index(name="item_n_interactions_raw")
    n_items = len(item_counts)
    item_counts["item_n_interactions"] = np.log1p(item_counts["item_n_interactions_raw"]).astype(np.float32)
    item_counts["item_pop_rank_pct"] = (
        item_counts["item_n_interactions_raw"].rank(pct=True).astype(np.float32)
    )
    item_stats = item_counts.set_index("item_id")[["item_n_interactions", "item_pop_rank_pct"]]

    # User interaction counts + avg popularity of items they've interacted with
    merged = train.merge(item_counts[["item_id", "item_n_interactions_raw"]], on="item_id", how="left")
    user_agg = merged.groupby("user_id").agg(
        user_n_interactions_raw=("item_id", "count"),
        user_avg_item_pop=("item_n_interactions_raw", "mean"),
    ).reset_index()
    user_agg["user_n_interactions"] = np.log1p(user_agg["user_n_interactions_raw"]).astype(np.float32)
    user_agg["user_avg_item_pop"] = np.log1p(user_agg["user_avg_item_pop"]).astype(np.float32)
    user_stats = user_agg.set_index("user_id")[["user_n_interactions", "user_avg_item_pop"]]

    return user_stats, item_stats


def _rank_normalise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
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


def build_enhanced_feature_matrix(
    cf_scores: pd.DataFrame,
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    llm_features: pd.DataFrame | None = None,
    normalise: bool = True,
) -> pd.DataFrame:
    """Merge CF scores, stat features, and LLM features into one matrix.

    Args:
        cf_scores:   DataFrame [user_id, item_id, label, pop, knn, ease, ...]
        user_stats:  From build_stats() — per-user features
        item_stats:  From build_stats() — per-item features
        llm_features: [user_id, item_id, yes_prob] or None
        normalise:   Rank-normalise CF scores per user

    Returns:
        DataFrame with all ENHANCED_FEAT_COLS + [user_id, item_id, label]
    """
    df = cf_scores.copy()

    # Merge user stats
    df = df.merge(user_stats.reset_index(), on="user_id", how="left")
    # Merge item stats
    df = df.merge(item_stats.reset_index(), on="item_id", how="left")

    # Fill unseen users/items with medians
    for col in ["user_n_interactions", "user_avg_item_pop"]:
        df[col] = df[col].fillna(df[col].median()).astype(np.float32)
    for col in ["item_n_interactions", "item_pop_rank_pct"]:
        df[col] = df[col].fillna(0.0).astype(np.float32)

    # LLM yes_prob
    if llm_features is not None and not llm_features.empty:
        llm = llm_features[["user_id", "item_id", "yes_prob"]].rename(
            columns={"yes_prob": LLM_COL}
        )
        df = df.merge(llm, on=["user_id", "item_id"], how="left")
    else:
        df[LLM_COL] = np.nan
    df[LLM_COL] = df[LLM_COL].fillna(LLM_DEFAULT).astype(np.float32)

    # Ensure all CF cols exist
    for col in CF_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Rank-normalise CF scores (not the stat features — they're already comparable)
    if normalise:
        df = _rank_normalise(df, CF_COLS + [LLM_COL])

    return df


def split_Xy_grouped(df: pd.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (X, y, groups) for LambdaRank training.

    groups: list of per-user candidate counts (e.g. [100, 100, ...])
    """
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    groups = df.groupby("user_id").size().tolist()
    return X, y, groups
