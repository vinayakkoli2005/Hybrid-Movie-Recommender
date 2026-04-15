import pandas as pd


def binarize_ratings(df: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    """Drop ratings below threshold; keep only (user_id, item_id, timestamp).

    Args:
        df: DataFrame with columns user_id, item_id, rating, timestamp.
        threshold: Minimum rating to keep (inclusive). Default 4.

    Returns:
        DataFrame with columns [user_id, item_id, timestamp], rating dropped.
    """
    pos = df[df["rating"] >= threshold].copy()
    return pos[["user_id", "item_id", "timestamp"]].reset_index(drop=True)
