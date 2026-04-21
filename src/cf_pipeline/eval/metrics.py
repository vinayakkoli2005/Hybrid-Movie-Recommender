from __future__ import annotations

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


def map_at_k(scores: np.ndarray, k: int) -> float:
    """Mean Average Precision at K.

    With a single positive per user: AP@K = 1/rank if rank <= K else 0.
    """
    rank = _rank_of_positive(scores)
    ap = np.where(rank <= k, 1.0 / rank, 0.0)
    return float(ap.mean())


def mar_at_k(scores: np.ndarray, k: int) -> float:
    """Mean Average Recall at K.

    With a single positive per user: Recall@K = HR@K (identical).
    Included for completeness.
    """
    return hit_rate_at_k(scores, k)


def novelty_at_k(
    scores: np.ndarray,
    item_ids: np.ndarray,
    item_popularity: dict[int, int],
    n_train: int,
    k: int,
) -> float:
    """Mean self-information novelty of top-K recommendations.

    novelty = mean over users of mean over top-K items of -log2(pop(i)/n_train)
    Higher = more novel (less popular items recommended).

    Args:
        scores:          (n_users, n_candidates) score matrix
        item_ids:        (n_users, n_candidates) item ID matrix (col-0 = positive)
        item_popularity: dict mapping item_id -> interaction count in training
        n_train:         total training interactions
        k:               cutoff
    """
    top_k_idx = np.argsort(-scores, axis=1)[:, :k]   # (n_users, k)
    novelties = []
    for row in range(scores.shape[0]):
        row_nov = []
        for idx in top_k_idx[row]:
            iid = int(item_ids[row, idx])
            pop = item_popularity.get(iid, 1)
            row_nov.append(-np.log2(pop / n_train))
        novelties.append(np.mean(row_nov))
    return float(np.mean(novelties))


def all_metrics(
    scores: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10, 20),
    item_ids: np.ndarray | None = None,
    item_popularity: dict[int, int] | None = None,
    n_train: int | None = None,
) -> dict[str, float]:
    """Compute HR, NDCG, MAP, MAR, and Novelty for every K in ks.

    Pass item_ids, item_popularity, and n_train to also compute Novelty.
    """
    out: dict[str, float] = {}
    for k in ks:
        out[f"HR@{k}"]   = hit_rate_at_k(scores, k)
        out[f"NDCG@{k}"] = ndcg_at_k(scores, k)
        out[f"MAP@{k}"]  = map_at_k(scores, k)
        out[f"MAR@{k}"]  = mar_at_k(scores, k)
        if item_ids is not None and item_popularity is not None and n_train is not None:
            out[f"Novelty@{k}"] = novelty_at_k(scores, item_ids, item_popularity, n_train, k)
    return out
