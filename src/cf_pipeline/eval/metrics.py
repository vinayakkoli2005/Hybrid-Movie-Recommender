"""Vectorized ranking metrics for the NCF leave-one-out protocol.

Convention throughout:
  - scores is a (n_users, n_candidates) numpy array.
  - Column 0 is ALWAYS the positive item for every user.
  - Remaining columns (1..n_candidates-1) are the negatives.
  - Higher score = model predicts this item is more relevant.
"""
import numpy as np


def _rank_of_positive(scores: np.ndarray) -> np.ndarray:
    """Return the 1-indexed rank of column-0 (the positive) for each row.

    Tie-breaking is conservative: items with the SAME score as the positive
    are treated as ranked ABOVE it. This prevents inflating metrics when the
    model outputs identical scores for many items.

    Args:
        scores: (n_users, n_candidates) float array.

    Returns:
        ranks: (n_users,) int array, 1-indexed.
    """
    pos_score = scores[:, 0:1]                    # (n, 1)  — broadcast-ready
    rank = (scores > pos_score).sum(axis=1) + 1   # items strictly better + 1
    return rank                                    # (n,) int


def hit_rate_at_k(scores: np.ndarray, k: int) -> float:
    """Fraction of users whose positive item is ranked in the top-K.

    HR@K = (# users where rank(positive) ≤ K) / total_users
    """
    rank = _rank_of_positive(scores)
    return float((rank <= k).mean())


def ndcg_at_k(scores: np.ndarray, k: int) -> float:
    """Mean Normalised Discounted Cumulative Gain at K.

    Under the single-positive protocol the ideal DCG is always 1.0,
    so NDCG@K = DCG@K = 1/log2(rank+1) when rank ≤ K, else 0.
    """
    rank = _rank_of_positive(scores)
    in_topk = rank <= k
    gains = np.where(in_topk, 1.0 / np.log2(rank + 1), 0.0)
    return float(gains.mean())


def all_metrics(
    scores: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    """Compute HR@K and NDCG@K for every K in ks.

    Returns:
        Dict with keys like 'HR@1', 'NDCG@1', 'HR@5', 'NDCG@5', …
    """
    out: dict[str, float] = {}
    for k in ks:
        out[f"HR@{k}"]   = hit_rate_at_k(scores, k)
        out[f"NDCG@{k}"] = ndcg_at_k(scores, k)
    return out
