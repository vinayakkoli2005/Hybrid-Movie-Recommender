from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix, csr_matrix

from cf_pipeline.models.base import BaseRanker


def _build_norm_adj(
    n_u: int,
    n_i: int,
    u_arr: np.ndarray,
    i_arr: np.ndarray,
) -> torch.sparse.Tensor:
    n = n_u + n_i
    rows = np.concatenate([u_arr, i_arr + n_u])
    cols = np.concatenate([i_arr + n_u, u_arr])
    data = np.ones(len(rows), dtype=np.float32)

    adj = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg_inv_sqrt = 1.0 / np.sqrt(np.where(deg == 0, 1.0, deg))
    degree = csr_matrix((deg_inv_sqrt, (np.arange(n), np.arange(n))), shape=(n, n))
    norm = (degree @ adj @ degree).tocoo()

    indices = torch.from_numpy(np.vstack([norm.row, norm.col])).long()
    values = torch.from_numpy(norm.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


class _LightGCN(nn.Module):
    def __init__(self, n_u: int, n_i: int, emb_dim: int, n_layers: int) -> None:
        super().__init__()
        self.n_u = n_u
        self.n_i = n_i
        self.n_layers = n_layers
        self.emb = nn.Embedding(n_u + n_i, emb_dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self, adj: torch.sparse.Tensor) -> torch.Tensor:
        x = self.emb.weight
        out = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            out.append(x)
        return torch.stack(out, dim=0).mean(dim=0)


class LightGCNRanker(BaseRanker):
    def __init__(
        self,
        emb_dim: int = 64,
        n_layers: int = 3,
        n_epochs: int = 200,
        batch_size: int = 8192,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self._model: _LightGCN | None = None
        self._adj: torch.sparse.Tensor | None = None
        self._user_to_idx: dict[int, int] = {}
        self._item_to_idx: dict[int, int] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._final_emb: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> "LightGCNRanker":
        users = sorted(train["user_id"].unique())
        items = sorted(train["item_id"].unique())
        self._user_to_idx = {int(user_id): idx for idx, user_id in enumerate(users)}
        self._item_to_idx = {int(item_id): idx for idx, item_id in enumerate(items)}

        user_idx = train["user_id"].map(self._user_to_idx).to_numpy(dtype=np.int64)
        item_idx = train["item_id"].map(self._item_to_idx).to_numpy(dtype=np.int64)
        n_users, n_items = len(users), len(items)

        self._adj = _build_norm_adj(n_users, n_items, user_idx, item_idx).to(self._device)
        self._model = _LightGCN(
            n_u=n_users,
            n_i=n_items,
            emb_dim=self.emb_dim,
            n_layers=self.n_layers,
        ).to(self._device)
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        rng = np.random.default_rng(42)

        self._model.train()
        for _ in range(self.n_epochs):
            perm = rng.permutation(len(user_idx))

            for start in range(0, len(user_idx), self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                users_batch = torch.from_numpy(user_idx[batch_idx]).long().to(self._device)
                pos_batch = torch.from_numpy(item_idx[batch_idx]).long().to(self._device)
                neg_batch = torch.from_numpy(
                    rng.integers(0, n_items, size=len(batch_idx), dtype=np.int64)
                ).long().to(self._device)

                emb = self._model.propagate(self._adj)
                user_emb = emb[users_batch]
                pos_emb = emb[n_users + pos_batch]
                neg_emb = emb[n_users + neg_batch]
                pos_scores = (user_emb * pos_emb).sum(-1)
                neg_scores = (user_emb * neg_emb).sum(-1)
                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            self._model.eval()
            self._final_emb = self._model.propagate(self._adj).cpu().numpy()

        return self

    def score(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
    ) -> np.ndarray:
        if self._final_emb is None:
            raise RuntimeError("Call fit() before score().")

        n_train_users = len(self._user_to_idx)
        n_users, n_candidates = item_ids.shape
        out = np.zeros((n_users, n_candidates), dtype=np.float32)

        for row, user_id in enumerate(user_ids):
            user_idx = self._user_to_idx.get(int(user_id))
            if user_idx is None:
                continue

            candidate_idx = np.array(
                [self._item_to_idx.get(int(item_id), -1) for item_id in item_ids[row]],
                dtype=np.int64,
            )
            mask = candidate_idx >= 0
            if not mask.any():
                continue

            user_emb = self._final_emb[user_idx]
            item_emb = self._final_emb[n_train_users + candidate_idx[mask]]
            out[row, mask] = (item_emb @ user_emb).astype(np.float32)

        return out
