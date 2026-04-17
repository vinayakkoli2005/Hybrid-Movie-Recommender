from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cf_pipeline.models.base import BaseRanker


class _BPRModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, emb_dim)
        self.item_emb = nn.Embedding(n_items + 1, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score_pair(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(user_idx) * self.item_emb(item_idx)).sum(-1)


class BPRMFRanker(BaseRanker):
    def __init__(
        self,
        emb_dim: int = 64,
        n_epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 1024,
        weight_decay: float = 1e-5,
    ) -> None:
        self.emb_dim = emb_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self._model: _BPRModel | None = None
        self._user_to_idx: dict[int, int] = {}
        self._item_to_idx: dict[int, int] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, train: pd.DataFrame) -> "BPRMFRanker":
        users = sorted(train["user_id"].unique())
        items = sorted(train["item_id"].unique())
        self._user_to_idx = {int(user_id): idx for idx, user_id in enumerate(users)}
        self._item_to_idx = {int(item_id): idx for idx, item_id in enumerate(items)}

        user_idx = train["user_id"].map(self._user_to_idx).to_numpy(dtype=np.int64)
        item_idx = train["item_id"].map(self._item_to_idx).to_numpy(dtype=np.int64)
        n_users, n_items = len(users), len(items)

        self._model = _BPRModel(n_users, n_items, self.emb_dim).to(self._device)
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

                pos_scores = self._model.score_pair(users_batch, pos_batch)
                neg_scores = self._model.score_pair(users_batch, neg_batch)
                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    @torch.no_grad()
    def score(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before score().")

        self._model.eval()
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
            scores = self._model.score_pair(users_t, items_t).cpu().numpy()
            out[row, mask] = scores.astype(np.float32)

        return out
