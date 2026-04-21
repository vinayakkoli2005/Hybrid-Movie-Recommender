"""Compute LLM YES/NO + yes_prob for top-20 DCN-ranked candidates per test user.

Output: data/processed/llm_features.parquet — columns (user_id, item_id, decision, yes_prob).

Supports resume: if the output file exists, already-computed users are skipped.
Checkpoints every 5 users so at most ~3 min of work is lost on disconnect.
Use --max-users N to process only N users (for smoke-testing).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from cf_pipeline.utils.logging import get_logger
from cf_pipeline.llm.server import LlamaServer
from cf_pipeline.llm.rag import DenseItemIndex, BM25ItemIndex, reciprocal_rank_fusion
from cf_pipeline.llm.decision import build_decision_prompt, parse_decision_response

PROCESSED = Path("data/processed")
OUT = PROCESSED / "llm_features.parquet"
CHECKPOINT_EVERY = 5   # save every N users
TOP_K_CANDIDATES = 20  # LLM only scores top-20 hard candidates per user
LLM_BATCH_SIZE = 4     # prompts per GPU call — keeps VRAM safe on shared server


def _top_candidates(
    positive: int,
    negatives: list[int],
    k: int = TOP_K_CANDIDATES,
) -> list[int]:
    """Return k candidates: positive always included + k-1 random negatives.
    Random sampling gives a realistic genre mix so the LLM produces both YES and NO.
    (Selecting only user-similar candidates makes LLM say YES to everything.)"""
    sampled = negatives[:k - 1]   # negatives are already random — just take first k-1
    return [positive] + sampled


def main(max_users: int | None = None) -> None:
    log = get_logger("llm_features")

    train = pd.read_parquet(PROCESSED / "train.parquet")
    test = pd.read_parquet(PROCESSED / "test.parquet")
    items = pd.read_parquet(PROCESSED / "items_metadata.parquet")
    items_lookup = items.set_index("item_id").to_dict(orient="index")

    # resume: skip users already done
    done_users: set[int] = set()
    existing_rows: list[dict] = []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        done_users = set(prev["user_id"].unique().tolist())
        existing_rows = prev.to_dict(orient="records")
        log.info("Resuming — %d users already done", len(done_users))

    todo = test[~test["user_id"].isin(done_users)]
    if max_users:
        todo = todo.head(max_users)

    if todo.empty:
        log.info("All users already processed.")
        return

    log.info("Building dense + BM25 indexes over %d items…", len(items))
    dense = DenseItemIndex().build(items)
    bm25 = BM25ItemIndex().build(items)

    log.info("Loading LLM server…")
    srv = LlamaServer(max_tokens=32)  # YES/NO JSON needs ~8 tokens max

    history_by_user = (
        train.merge(items[["item_id", "title", "genres"]], on="item_id")
        .groupby("user_id")
        .apply(lambda g: g[["item_id", "title", "genres"]].to_dict(orient="records"))
        .to_dict()
    )

    rows: list[dict] = list(existing_rows)
    total = len(todo)

    for ridx, row in enumerate(todo.itertuples(index=False)):
        u = int(row.user_id)
        positive = int(row.positive)
        negatives = [int(x) for x in row.negatives]
        history = history_by_user.get(u, [])

        # pre-filter to top-20 hard candidates (positive always included)
        candidates = _top_candidates(positive, negatives, k=TOP_K_CANDIDATES)

        # build all prompts for this user at once
        prompts, metas = [], []
        for cand in candidates:
            cand_meta = items_lookup.get(cand, {"title": "?", "genres": "?", "overview": ""})
            query = f"{cand_meta.get('title', '')} {cand_meta.get('genres', '')}"
            fused = reciprocal_rank_fusion(
                [dense.search(query, k=5), bm25.search(query, k=5)], k=5
            )
            retrieved = [items_lookup.get(iid, {"title": "?", "genres": "?"}) for iid, _ in fused]
            prompts.append(build_decision_prompt(history, retrieved, cand_meta))
            metas.append({"user_id": u, "item_id": cand})

        # batch LLM calls in groups of LLM_BATCH_SIZE — faster than 1-by-1, safe on VRAM
        outs = []
        for i in range(0, len(prompts), LLM_BATCH_SIZE):
            outs.extend(srv.generate(prompts[i:i + LLM_BATCH_SIZE]))
        for meta, o in zip(metas, outs):
            parsed = parse_decision_response(o["text"], o["logprobs"])
            rows.append({**meta, **parsed})

        # checkpoint every CHECKPOINT_EVERY users
        if (ridx + 1) % CHECKPOINT_EVERY == 0 or (ridx + 1) == total:
            log.info("  %d / %d users done — checkpoint saved", ridx + 1, total)
            pd.DataFrame(rows).to_parquet(OUT)

    pd.DataFrame(rows).to_parquet(OUT)
    log.info("Done — wrote %d rows to %s", len(rows), OUT)
    srv.free()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=None,
                        help="Process only N users (for smoke-testing)")
    args = parser.parse_args()
    main(max_users=args.max_users)
