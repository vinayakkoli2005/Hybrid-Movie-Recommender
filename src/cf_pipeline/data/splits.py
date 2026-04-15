import pandas as pd


def leave_one_out_split(
    df: pd.DataFrame,
    min_interactions: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split binarized interactions into train / val / test per user.

    Protocol (standard NCF leave-one-out):
      - For each eligible user (≥ min_interactions interactions):
          test  = the single latest interaction (by timestamp)
          val   = the second-latest interaction
          train = all remaining interactions
      - Users with fewer than min_interactions are dropped entirely.

    Tie-breaking: when two interactions share the same timestamp, item_id is
    used as a secondary sort key so the split is fully deterministic.

    Args:
        df: Binarized interactions with columns [user_id, item_id, timestamp].
        min_interactions: Minimum number of interactions a user must have to
            be included (default 3 — the minimum for a valid 3-way split).

    Returns:
        (train, val, test) — three DataFrames with the same columns as df,
        each with a 0-based integer index.
    """
    # Sort deterministically: primary timestamp DESC, secondary item_id ASC
    # (item_id tiebreaker ensures identical results across repeated calls)
    df = df.sort_values(["user_id", "timestamp", "item_id"],
                        ascending=[True, True, True]).copy()

    # Drop users with too few interactions
    counts = df.groupby("user_id").size()
    eligible = counts[counts >= min_interactions].index
    df = df[df["user_id"].isin(eligible)].copy()

    # Rank interactions per user: rank 1 = latest (highest timestamp)
    # method="first" breaks ties by position in the sorted array (item_id order)
    df["_rank"] = (
        df.groupby("user_id")["timestamp"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    test  = df[df["_rank"] == 1].drop(columns="_rank").reset_index(drop=True)
    val   = df[df["_rank"] == 2].drop(columns="_rank").reset_index(drop=True)
    train = df[df["_rank"] >= 3].drop(columns="_rank").reset_index(drop=True)

    return train, val, test
