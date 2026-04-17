from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from cf_pipeline.models.base import BaseRanker


class EASERRanker(BaseRanker):
    def __init__(self, reg_lambda: float = 500.0) -> None:
        self.reg_lambda = reg_lambda
        self._user_to_idx: dict[int, int] = {}
        self._item_to_idx: dict[int, int] = {}
        self._X: csr_matrix | None = None
        self._B: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> "EASERRanker":
        users = sorted(train["user_id"].unique())
        items = sorted(train["item_id"].unique())
        self._user_to_idx = {int(user_id): idx for idx, user_id in enumerate(users)}
        self._item_to_idx = {int(item_id): idx for idx, item_id in enumerate(items)}

        rows = train["user_id"].map(self._user_to_idx).to_numpy(dtype=np.int32)
        cols = train["item_id"].map(self._item_to_idx).to_numpy(dtype=np.int32)
        data = np.ones(len(train), dtype=np.float32)
        n_users, n_items = len(users), len(items)

        self._X = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        G = (self._X.T @ self._X).toarray().astype(np.float64)
        diag_idx = np.diag_indices_from(G)
        G[diag_idx] += self.reg_lambda

        P = np.linalg.inv(G)
        B = -P / np.diag(P)[None, :]
        B[diag_idx] = 0.0
        self._B = B.astype(np.float32)
        return self

    def score(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
    ) -> np.ndarray:
        if self._X is None or self._B is None:
            raise RuntimeError("Call fit() before score().")

        n_users, n_candidates = item_ids.shape
        out = np.zeros((n_users, n_candidates), dtype=np.float32)

        for row, user_id in enumerate(user_ids):
            user_idx = self._user_to_idx.get(int(user_id))
            if user_idx is None:
                continue

            user_vec = self._X[user_idx]
            user_scores = np.asarray(user_vec @ self._B).ravel()

            candidate_idx = np.array(
                [self._item_to_idx.get(int(item_id), -1) for item_id in item_ids[row]],
                dtype=np.int32,
            )
            mask = candidate_idx >= 0
            if not mask.any():
                continue

            out[row, mask] = user_scores[candidate_idx[mask]].astype(np.float32)

        return out
