"""Train SASRec and add its scores to cf_scores_val/test.parquet."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch

from cf_pipeline.eval.metrics import all_metrics
from cf_pipeline.eval.protocol import build_candidate_matrix
from cf_pipeline.models.sasrec import SASRecRanker
from cf_pipeline.utils.logging import get_logger

PROCESSED = Path("data/processed")
DEVICE    = "cuda:0"   # GPU 0 is free (28 GB)


def main() -> None:
    log = get_logger("train_sasrec")

    train   = pd.read_parquet(PROCESSED / "train.parquet")
    val     = pd.read_parquet(PROCESSED / "val.parquet")
    test    = pd.read_parquet(PROCESSED / "test.parquet")
    cf_val  = pd.read_parquet(PROCESSED / "cf_scores_val.parquet")
    cf_test = pd.read_parquet(PROCESSED / "cf_scores_test.parquet")

    val_u,  val_i  = build_candidate_matrix(val)
    test_u, test_i = build_candidate_matrix(test)

    log.info("Training SASRec on GPU %s…", DEVICE)
    log.info("  train interactions: %d  |  users: %d  |  items: %d",
             len(train), train.user_id.nunique(), train.item_id.nunique())

    model = SASRecRanker(
        hidden=64,
        max_len=200,
        n_heads=1,
        n_layers=2,
        dropout=0.5,
        n_epochs=200,
        batch_size=256,
        lr=1e-3,
        device=DEVICE,
    )

    t0 = time.time()
    model.fit(train)
    log.info("Training done in %.1f min", (time.time() - t0) / 60)

    log.info("Scoring val set…")
    val_scores  = model.score(val_u,  val_i)
    log.info("Scoring test set…")
    test_scores = model.score(test_u, test_i)

    # Quick eval on val
    from cf_pipeline.eval.metrics import ndcg_at_k, hit_rate_at_k
    log.info("Val  HR@10=%.4f  NDCG@10=%.4f",
             hit_rate_at_k(val_scores, 10), ndcg_at_k(val_scores, 10))
    log.info("Test HR@10=%.4f  NDCG@10=%.4f",
             hit_rate_at_k(test_scores, 10), ndcg_at_k(test_scores, 10))

    # Append sasrec column to cf_scores parquets
    cf_val["sasrec"]  = val_scores.ravel().astype(np.float32)
    cf_test["sasrec"] = test_scores.ravel().astype(np.float32)

    cf_val.to_parquet(PROCESSED / "cf_scores_val.parquet",  index=False)
    cf_test.to_parquet(PROCESSED / "cf_scores_test.parquet", index=False)
    log.info("SASRec scores added to cf_scores_val/test.parquet")
    log.info("Columns now: %s", cf_val.columns.tolist())


if __name__ == "__main__":
    main()
