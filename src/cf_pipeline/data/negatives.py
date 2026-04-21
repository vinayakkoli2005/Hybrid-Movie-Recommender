from __future__ import annotations

import numpy as np
import pandas as pd


def sample_negatives(
    eval_set: pd.DataFrame,
    train: pd.DataFrame,
    all_item_ids: list[int],
    n_neg: int = 99,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample n_neg negative items per (user, positive) row in eval_set.

    Negatives are guaranteed to be:
      - Disjoint from the user's full train history.
      - Not equal to the held-out positive item.
      - Sampled WITHOUT replacement from all_item_ids.
      - Deterministic for a given seed (frozen).

    Eval_set is sorted by (user_id, item_id) before iteration so the RNG
    sequence — and therefore the negatives — are identical regardless of
    the row order passed in.

    Args:
        eval_set:     DataFrame with columns [user_id, item_id, ...].
                      One row per (user, positive-item) pair.
        train:        DataFrame with columns [user_id, item_id, ...].
                      All interactions used for training.
        all_item_ids: Pool of candidate item IDs to sample from.
                      Should be the set of items that survived the TMDB join.
        n_neg:        Number of negatives per user. Default 99 (NCF protocol).
        seed:         RNG seed for reproducibility. Default 42.

    Returns:
        DataFrame with columns:
          user_id   (int)  — the user
          positive  (int)  — the held-out positive item
          negatives (list) — list of n_neg negative item IDs (Python ints)
    """
    # Sort eval_set for deterministic RNG sequence regardless of input order
    eval_set = eval_set.sort_values(["user_id", "item_id"]).reset_index(drop=True)

    rng = np.random.default_rng(seed)

    # Build per-user history dict once (O(n) — avoids repeated filtering)
    user_history: dict[int, set[int]] = (
        train.groupby("user_id")["item_id"].apply(set).to_dict()
    )

    all_items_arr = np.asarray(all_item_ids, dtype=np.int32)

    rows = []
    for _, r in eval_set.iterrows():
        u   = int(r["user_id"])
        pos = int(r["item_id"])

        # Forbidden = everything the user has already seen + the positive itself
        forbidden = user_history.get(u, set()) | {pos}

        # Remove forbidden items from the candidate pool
        candidates = all_items_arr[~np.isin(all_items_arr, list(forbidden))]

        if len(candidates) < n_neg:
            raise ValueError(
                f"User {u}: only {len(candidates)} candidates available "
                f"but n_neg={n_neg} requested. "
                "Reduce n_neg or expand the item pool."
            )

        chosen = rng.choice(candidates, size=n_neg, replace=False).tolist()
        rows.append({"user_id": u, "positive": pos, "negatives": chosen})

    return pd.DataFrame(rows)
