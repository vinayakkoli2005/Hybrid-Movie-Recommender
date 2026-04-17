from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cf_pipeline.models.base import BaseRanker


class _CrossLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(dim, dim)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.W(xl) + xl


class _DCNv2(nn.Module):
    def __init__(
        self,
        n_u: int,
        n_i: int,
        emb_dim: int,
        cross_layers: int,
        deep: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_u + 1, emb_dim)
        self.item_emb = nn.Embedding(n_i + 1, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        in_dim = 2 * emb_dim
        self.crosses = nn.ModuleList([_CrossLayer(in_dim) for _ in range(cross_layers)])

        layers: list[nn.Module] = []
        prev = in_dim
        for hidden_dim in deep:
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = hidden_dim

        self.deep = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim + prev, 1)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        x0 = torch.cat([self.user_emb(user_idx), self.item_emb(item_idx)], dim=-1)
        xl = x0
        for layer in self.crosses:
            xl = layer(x0, xl)
        deep_out = self.deep(x0)
        return self.head(torch.cat([xl, deep_out], dim=-1)).squeeze(-1)


class DCNRanker(BaseRanker):
    def __init__(
        self,
        emb_dim: int = 64,
        cross_layers: int = 3,
        deep: tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
        n_epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 4096,
    ) -> None:
        self.emb_dim = emb_dim
        self.cross_layers = cross_layers
        self.deep = tuple(deep)
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self._model: _DCNv2 | None = None
        self._user_to_idx: dict[int, int] = {}
        self._item_to_idx: dict[int, int] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, train: pd.DataFrame) -> "DCNRanker":
        users = sorted(train["user_id"].unique())
        items = sorted(train["item_id"].unique())
        self._user_to_idx = {int(user_id): idx for idx, user_id in enumerate(users)}
        self._item_to_idx = {int(item_id): idx for idx, item_id in enumerate(items)}

        user_idx = train["user_id"].map(self._user_to_idx).to_numpy(dtype=np.int64)
        item_idx = train["item_id"].map(self._item_to_idx).to_numpy(dtype=np.int64)
        n_users, n_items = len(users), len(items)

        self._model = _DCNv2(
            n_u=n_users,
            n_i=n_items,
            emb_dim=self.emb_dim,
            cross_layers=self.cross_layers,
            deep=self.deep,
            dropout=self.dropout,
        ).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
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

                pos_logits = self._model(users_batch, pos_batch)
                neg_logits = self._model(users_batch, neg_batch)
                logits = torch.cat([pos_logits, neg_logits])
                labels = torch.cat([
                    torch.ones_like(pos_logits),
                    torch.zeros_like(neg_logits),
                ])
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    @torch.no_grad()
    def _forward_batch(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before score().")

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

            users_t = torch.full(
                (int(mask.sum()),),
                user_idx,
                dtype=torch.long,
                device=self._device,
            )
            items_t = torch.from_numpy(candidate_idx[mask]).long().to(self._device)
            scores = torch.sigmoid(self._model(users_t, items_t)).cpu().numpy()
            out[row, mask] = scores.astype(np.float32)

        return out

    def score(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before score().")

        self._model.eval()
        return self._forward_batch(user_ids, item_ids)

    def score_with_uncertainty(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        n_mc: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("Call fit() before score_with_uncertainty().")

        self._model.train()
        passes = [self._forward_batch(user_ids, item_ids) for _ in range(n_mc)]
        stack = np.stack(passes, axis=0)
        return stack.mean(axis=0).astype(np.float32), stack.var(axis=0).astype(np.float32)
