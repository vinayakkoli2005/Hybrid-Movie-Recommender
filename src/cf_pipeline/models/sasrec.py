"""SASRec: Self-Attentive Sequential Recommendation.

Wang-Cheng Kang & Julian McAuley, ICDM 2018.
Standard implementation with causal self-attention + binary cross-entropy loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cf_pipeline.models.base import BaseRanker


class _SASRecModel(nn.Module):
    def __init__(
        self,
        n_items: int,
        hidden: int,
        max_len: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_len, hidden)
        self.emb_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(hidden)
        self.hidden = hidden
        self.max_len = max_len

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """seqs: (B, L) item indices, 0 = pad."""
        B, L = seqs.shape
        pos = torch.arange(L, device=seqs.device).unsqueeze(0)
        x = self.item_emb(seqs) + self.pos_emb(pos)
        x = self.emb_drop(x)

        # Causal mask: position i cannot attend to j > i
        causal = torch.triu(
            torch.full((L, L), float("-inf"), device=seqs.device), diagonal=1
        )
        # No padding mask — avoids NaN when full rows are masked out in softmax
        x = self.transformer(x, mask=causal)
        x = self.out_norm(x)
        return x  # (B, L, H)

    def predict(self, seqs: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """seqs: (B, L), items: (B, C) → scores (B, C)."""
        h = self.forward(seqs)[:, -1, :]          # (B, H) last position
        e = self.item_emb(items)                   # (B, C, H)
        return (h.unsqueeze(1) * e).sum(-1)        # (B, C)


class SASRecRanker(BaseRanker):
    def __init__(
        self,
        hidden: int = 64,
        max_len: int = 200,
        n_heads: int = 1,
        n_layers: int = 2,
        dropout: float = 0.5,
        n_epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        n_neg: int = 1,
        device: str = "cuda:0",
    ) -> None:
        self.hidden = hidden
        self.max_len = max_len
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_neg = n_neg
        self.device = device
        self._model: _SASRecModel | None = None
        self._item_to_idx: dict[int, int] = {}
        self._idx_to_item: dict[int, int] = {}
        self._user_seqs: dict[int, list[int]] = {}

    def _build_seqs(self, train: pd.DataFrame) -> dict[int, list[int]]:
        seqs: dict[int, list[int]] = {}
        for uid, grp in train.sort_values("timestamp").groupby("user_id") if "timestamp" in train.columns \
                else train.groupby("user_id"):
            seqs[int(uid)] = [self._item_to_idx[int(i)] for i in grp["item_id"] if int(i) in self._item_to_idx]
        return seqs

    def _pad(self, seq: list[int]) -> list[int]:
        if len(seq) >= self.max_len:
            return seq[-self.max_len:]
        return [0] * (self.max_len - len(seq)) + seq

    def fit(self, train: pd.DataFrame) -> "SASRecRanker":
        items = sorted(train["item_id"].unique())
        # idx 0 reserved for padding; real items start at 1
        self._item_to_idx = {int(item): idx + 1 for idx, item in enumerate(items)}
        self._idx_to_item = {v: k for k, v in self._item_to_idx.items()}
        n_items = len(items)

        self._model = _SASRecModel(
            n_items, self.hidden, self.max_len, self.n_heads, self.n_layers, self.dropout
        ).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        self._user_seqs = self._build_seqs(train)
        all_item_idx = np.arange(1, n_items + 1, dtype=np.int64)
        rng = np.random.default_rng(42)

        # Build training samples: (seq, pos_item) pairs
        samples: list[tuple[list[int], int]] = []
        for uid, seq in self._user_seqs.items():
            for t in range(1, len(seq)):
                samples.append((seq[:t], seq[t]))

        for epoch in range(self.n_epochs):
            self._model.train()
            rng.shuffle(samples)  # type: ignore[arg-type]
            total_loss = 0.0
            n_batches = 0

            for start in range(0, len(samples), self.batch_size):
                batch = samples[start:start + self.batch_size]
                seqs_t = torch.tensor(
                    [self._pad(s) for s, _ in batch], dtype=torch.long, device=self.device
                )
                pos_t = torch.tensor([p for _, p in batch], dtype=torch.long, device=self.device)
                # Sample negatives
                neg_idx = rng.integers(1, n_items + 1, size=len(batch), dtype=np.int64)
                neg_t = torch.from_numpy(neg_idx).to(self.device)

                items_t = torch.stack([pos_t, neg_t], dim=1)  # (B, 2)
                scores = self._model.predict(seqs_t, items_t)  # (B, 2)
                labels = torch.zeros(len(batch), dtype=torch.long, device=self.device)  # class 0 = positive
                loss = nn.functional.cross_entropy(scores, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                optimizer.step()
                lv = loss.item()
                if not np.isnan(lv):
                    total_loss += lv
                    n_batches += 1

            if (epoch + 1) % 20 == 0:
                print(f"  epoch {epoch+1}/{self.n_epochs}  loss={total_loss/max(n_batches,1):.4f}")

        return self

    @torch.no_grad()
    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        self._model.eval()
        n_users, n_cands = item_ids.shape
        out = np.zeros((n_users, n_cands), dtype=np.float32)

        for row, uid in enumerate(user_ids):
            seq = self._user_seqs.get(int(uid), [])
            padded = self._pad(seq)
            seq_t = torch.tensor([padded], dtype=torch.long, device=self.device)

            cand_idx = np.array(
                [self._item_to_idx.get(int(i), 0) for i in item_ids[row]], dtype=np.int64
            )
            mask = cand_idx > 0
            if not mask.any():
                continue

            items_t = torch.tensor(cand_idx[mask][None], dtype=torch.long, device=self.device)
            scores = self._model.predict(seq_t, items_t)[0].cpu().numpy()
            out[row, mask] = scores.astype(np.float32)

        return out
