from __future__ import annotations

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss


def _item_text(row: pd.Series) -> str:
    parts = [str(row.get("title", "")), str(row.get("genres", "")), str(row.get("overview", ""))]
    return " | ".join([p for p in parts if p and p != "nan"])


class DenseItemIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._index = None
        self._ids: np.ndarray | None = None

    def build(self, items: pd.DataFrame) -> "DenseItemIndex":
        texts = items.apply(_item_text, axis=1).tolist()
        embs = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        self._ids = items["item_id"].to_numpy()
        d = embs.shape[1]
        self._index = faiss.IndexFlatIP(d)
        self._index.add(embs.astype(np.float32))
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        q = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        scores, idxs = self._index.search(q, k)
        return [(int(self._ids[i]), float(s)) for i, s in zip(idxs[0], scores[0])]

    def search_by_id(self, query_item_id: int, k: int = 5) -> list[tuple[int, float]]:
        pos = int(np.where(self._ids == query_item_id)[0][0])
        v = self._index.reconstruct(pos).reshape(1, -1)
        scores, idxs = self._index.search(v, k + 1)
        out = [
            (int(self._ids[i]), float(s))
            for i, s in zip(idxs[0], scores[0])
            if int(self._ids[i]) != query_item_id
        ]
        return out[:k]


class BM25ItemIndex:
    def __init__(self):
        self._bm25 = None
        self._ids = None

    def build(self, items: pd.DataFrame) -> "BM25ItemIndex":
        texts = items.apply(_item_text, axis=1).tolist()
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._ids = items["item_id"].to_numpy()
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(query.lower().split())
        top_idx = np.argsort(-scores)[:k]
        return [(int(self._ids[i]), float(scores[i])) for i in top_idx]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]], k: int = 5, c: int = 60
) -> list[tuple[int, float]]:
    """RRF: score(i) = sum over lists of 1/(c + rank_in_list)."""
    fused: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, (iid, _) in enumerate(lst, start=1):
            fused[iid] = fused.get(iid, 0.0) + 1.0 / (c + rank)
    return sorted(fused.items(), key=lambda x: -x[1])[:k]


_HYDE_TEMPLATE = """Given a user who has liked these movies:
{history_block}

Write 1-2 sentences describing the kind of movie they would enjoy next. Be concrete: mention themes, tone, and genre.
Then check whether this candidate fits: {candidate_title} ({candidate_genres}). Plot: {candidate_overview}

Brief reasoning + ideal-movie description:"""


def build_hyde_query_prompt(history: list[dict], candidate: dict) -> str:
    lines = [f"- {h.get('title', '?')} ({h.get('genres', '?')})" for h in history[:10]]
    history_block = "\n".join(lines) if lines else "(no history)"
    return _HYDE_TEMPLATE.format(
        history_block=history_block,
        candidate_title=candidate.get("title", "?"),
        candidate_genres=candidate.get("genres", "?"),
        candidate_overview=candidate.get("overview", ""),
    )
