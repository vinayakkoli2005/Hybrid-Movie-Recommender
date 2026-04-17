from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cf_pipeline.models.base import BaseRanker


class _NeuMF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int,
        mlp_layers: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.gmf_user_emb = nn.Embedding(n_users + 1, emb_dim)
        self.gmf_item_emb = nn.Embedding(n_items + 1, emb_dim)
        self.mlp_user_emb = nn.Embedding(n_users + 1, emb_dim)
        self.mlp_item_emb = nn.Embedding(n_items + 1, emb_dim)

        layers: list[nn.Module] = []
        in_dim = 2 * emb_dim
        for hidden_dim in mlp_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(emb_dim + in_dim, 1)

        for emb in (
            self.gmf_user_emb,
            self.gmf_item_emb,
            self.mlp_user_emb,
            self.mlp_item_emb,
        ):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        gmf = self.gmf_user_emb(user_idx) * self.gmf_item_emb(item_idx)
        mlp_in = torch.cat(
            [self.mlp_user_emb(user_idx), self.mlp_item_emb(item_idx)],
            dim=-1,
        )
        mlp_out = self.mlp(mlp_in)
        return self.out(torch.cat([gmf, mlp_out], dim=-1)).squeeze(-1)


class NeuMFRanker(BaseRanker):
    def __init__(
        self,
        emb_dim: int = 32,
        mlp_layers: tuple[int, ...] = (64, 32, 16),
        n_epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 4096,
    ) -> None:
        self.emb_dim = emb_dim
        self.mlp_layers = tuple(mlp_layers)
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self._model: _NeuMF | None = None
        self._user_to_idx: dict[int, int] = {}
        self._item_to_idx: dict[int, int] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, train: pd.DataFrame) -> "NeuMFRanker":
        users = sorted(train["user_id"].unique())
        items = sorted(train["item_id"].unique())
        self._user_to_idx = {int(user_id): idx for idx, user_id in enumerate(users)}
        self._item_to_idx = {int(item_id): idx for idx, item_id in enumerate(items)}

        user_idx = train["user_id"].map(self._user_to_idx).to_numpy(dtype=np.int64)
        item_idx = train["item_id"].map(self._item_to_idx).to_numpy(dtype=np.int64)
        n_users, n_items = len(users), len(items)

        self._model = _NeuMF(
            n_users=n_users,
            n_items=n_items,
            emb_dim=self.emb_dim,
            mlp_layers=self.mlp_layers,
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
            scores = torch.sigmoid(self._model(users_t, items_t)).cpu().numpy()
            out[row, mask] = scores.astype(np.float32)

        return out
