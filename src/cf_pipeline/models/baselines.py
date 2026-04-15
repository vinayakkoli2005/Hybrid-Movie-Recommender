"""Non-personalised and neighbourhood-based baseline rankers.

Task 15 — PopularityRanker
Task 16 — ItemKNNRanker

All classes extend BaseRanker and are callable through the eval harness via
model.score(user_ids, item_ids) → (n_users, n_candidates) score matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from cf_pipeline.models.base import BaseRanker


class PopularityRanker(BaseRanker):
    """Global popularity baseline — user-agnostic, item-frequency scoring.

    Assigns each item a score equal to its interaction count in the training
    set.  Every user receives the same ranking; no personalisation.

    Expected HR@10 on ML-1M (NCF 99-neg protocol): ≈ 0.57
    Reference: He et al., NeuMF (WWW 2017), Table 2.

    Usage:
        ranker = PopularityRanker()
        ranker.fit(train_df)          # train_df has column 'item_id'
        scores = ranker.score(user_ids, item_ids)
    """

    def fit(self, train: pd.DataFrame) -> "PopularityRanker":
        """Count interactions per item in the training set.

        Args:
            train: DataFrame with at least an 'item_id' column.

        Returns:
            self (for chaining).
        """
        counts = train["item_id"].value_counts()
        self._pop: dict[int, int] = counts.to_dict()
        return self

    # ------------------------------------------------------------------
    # BaseRanker interface
    # ------------------------------------------------------------------

    def score(
        self,
        user_ids: np.ndarray,   # (n_users,)  — ignored (non-personalised)
        item_ids: np.ndarray,   # (n_users, n_candidates)
    ) -> np.ndarray:            # (n_users, n_candidates)  higher = more popular
        """Return popularity counts as scores.

        Items not seen during training receive score 0.

        Args:
            user_ids: (n_users,) array — ignored by this model.
            item_ids: (n_users, n_candidates) integer array.

        Returns:
            Float score matrix of the same shape.
        """
        if not hasattr(self, "_pop"):
            raise RuntimeError("Call fit() before score().")

        scores = np.vectorize(lambda iid: self._pop.get(int(iid), 0))(item_ids)
        return scores.astype(np.float32)


# =============================================================================
# Task 16 — ItemKNN (cosine with shrinkage on user-item co-occurrence matrix)
# =============================================================================

class ItemKNNRanker(BaseRanker):
    """Item-based K-Nearest Neighbour collaborative filtering.

    Scores a candidate item *i* for user *u* by summing the shrinkage-cosine
    similarities between *i* and every item in *u*'s training history:

        score(u, i) = Σ_{h ∈ H_u}  sim(i, h)

    where

        sim(i, j) = dot(i, j) / (‖i‖ · ‖j‖ + shrinkage)

    and dot products / norms are computed over the binary user-item vectors.

    Top-k pruning zeros out all but the k largest neighbours per item,
    preventing noise from distant items and reducing memory.

    Expected HR@10 on ML-1M (NCF 99-neg protocol): ≈ 0.50–0.55.
    Reference: Deshpande & Karypis, "Item-Based Top-N Recommendation
               Algorithms", TOIS 2004.

    Args:
        k_neighbors: Number of nearest neighbours to retain per item.
        shrinkage:   Denominator regulariser — prevents inflating similarity
                     for items with very few co-occurrences.
    """

    def __init__(self, k_neighbors: int = 20, shrinkage: float = 10.0) -> None:
        self.k = k_neighbors
        self.shrinkage = shrinkage
        self._user_to_idx: dict[int, int] = {}
        self._item_to_idx: dict[int, int] = {}
        self._item_sim: np.ndarray | None = None      # (n_items, n_items) float32
        self._user_items: csr_matrix | None = None    # (n_users, n_items) binary

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train: pd.DataFrame) -> "ItemKNNRanker":
        """Build the item–item similarity matrix from training interactions.

        Steps
        -----
        1. Build a binary (n_users × n_items) CSR matrix from `train`.
        2. Compute shrinkage-cosine item–item similarity.
        3. Zero-out the diagonal (an item is not its own neighbour).
        4. Prune to top-k neighbours per item.

        Args:
            train: DataFrame with columns ['user_id', 'item_id'].

        Returns:
            self (for chaining).
        """
        users = sorted(train["user_id"].unique())
        items = sorted(train["item_id"].unique())
        self._user_to_idx = {u: i for i, u in enumerate(users)}
        self._item_to_idx = {it: j for j, it in enumerate(items)}

        row_idx = train["user_id"].map(self._user_to_idx).to_numpy(dtype=np.int32)
        col_idx = train["item_id"].map(self._item_to_idx).to_numpy(dtype=np.int32)
        data    = np.ones(len(train), dtype=np.float32)

        n_u, n_i = len(users), len(items)
        self._user_items = csr_matrix(
            (data, (row_idx, col_idx)), shape=(n_u, n_i)
        )

        # --- shrinkage-cosine similarity -----------------------------------
        # item_user: (n_items, n_users) — each row is a binary user vector
        item_user = self._user_items.T.tocsr()

        # Raw dot products: S[i,j] = number of shared users between items i,j
        S = (item_user @ item_user.T).toarray().astype(np.float64)

        # Per-item L2 norms: norm[i] = sqrt(S[i,i]) = sqrt(#users of item i)
        norms = np.sqrt(np.diag(S))                          # (n_items,)

        # Denominator matrix: norm_i * norm_j + shrinkage
        D = np.outer(norms, norms) + self.shrinkage           # (n_items, n_items)

        sim = (S / D).astype(np.float32)

        # Self-similarity is uninformative → zero it out
        np.fill_diagonal(sim, 0.0)

        # Top-k pruning: keep only the k largest neighbours per item row.
        # np.argpartition is O(n) vs O(n log n) for full sort.
        if self.k < n_i - 1:
            # indices of the (n_i - k) SMALLEST similarities per row
            prune_idx = np.argpartition(sim, n_i - self.k, axis=1)[:, : n_i - self.k]
            rows = np.arange(n_i)[:, None].repeat(n_i - self.k, axis=1)
            sim[rows, prune_idx] = 0.0

        self._item_sim = sim
        return self

    # ------------------------------------------------------------------
    # BaseRanker interface
    # ------------------------------------------------------------------

    def score(
        self,
        user_ids: np.ndarray,   # (n_users,)
        item_ids: np.ndarray,   # (n_users, n_candidates)  col-0 = positive
    ) -> np.ndarray:            # (n_users, n_candidates)  higher = better
        """Personalised scoring via summed item–item similarity to user history.

        For each user *u* and candidate item *i*, the score is:

            score(u, i) = item_sim[i, :] @ user_history_binary[u, :]

        The inner product is computed as a single matrix-vector multiplication
        per user, eliminating all Python loops over candidates.

        Args:
            user_ids: (n_users,) integer user IDs.
            item_ids: (n_users, n_candidates) integer item ID matrix.

        Returns:
            Float32 score matrix of shape (n_users, n_candidates).
            Unknown users and unseen items both score 0.
        """
        if self._item_sim is None or self._user_items is None:
            raise RuntimeError("Call fit() before score().")

        n_users, n_cands = item_ids.shape
        out = np.zeros((n_users, n_cands), dtype=np.float32)

        for r, u in enumerate(user_ids):
            uidx = self._user_to_idx.get(int(u))
            if uidx is None:
                continue  # cold-start user → zero scores

            # Dense binary history vector for this user: (n_items,)
            user_history = self._user_items[uidx].toarray().ravel()

            # Map candidate item IDs → internal indices (-1 = unseen)
            iidxs = np.array(
                [self._item_to_idx.get(int(iid), -1) for iid in item_ids[r]],
                dtype=np.int32,
            )
            mask = iidxs >= 0

            if not mask.any():
                continue

            # Vectorised: (n_valid_cands, n_items) @ (n_items,) → (n_valid_cands,)
            out[r, mask] = self._item_sim[iidxs[mask]] @ user_history

        return out
