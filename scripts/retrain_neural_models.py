"""Retrain neural CF models with proper epochs and score val+test.

Merges new neural scores into existing cf_scores_val/test parquets,
replacing the old lgcn/dcn/neumf columns.

Run on free GPU:
  CUDA_VISIBLE_DEVICES=1 python3.8 scripts/retrain_neural_models.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from cf_pipeline.eval.protocol import build_candidate_matrix
from cf_pipeline.models.dcn import DCNRanker
from cf_pipeline.models.lightgcn import LightGCNRanker
from cf_pipeline.models.neumf import NeuMFRanker
from cf_pipeline.utils.logging import get_logger

PROCESSED = Path("data/processed")


def _fit_score(name, model, train, val_u, val_i, test_u, test_i, log):
    t0 = time.time()
    log.info("Fitting %s…", name)
    model.fit(train)
    log.info("  %s fit in %.0fs — scoring…", name, time.time() - t0)
    val_s  = model.score(val_u,  val_i)
    test_s = model.score(test_u, test_i)
    log.info("  %s total %.0fs", name, time.time() - t0)
    return val_s, test_s


def main() -> None:
    log = get_logger("retrain_neural")

    train = pd.read_parquet(PROCESSED / "train.parquet")
    val   = pd.read_parquet(PROCESSED / "val.parquet")
    test  = pd.read_parquet(PROCESSED / "test.parquet")

    val_u,  val_i  = build_candidate_matrix(val)
    test_u, test_i = build_candidate_matrix(test)

    n_val,  n_cval  = val_i.shape
    n_test, n_ctest = test_i.shape

    # Load existing score parquets to update
    cf_val  = pd.read_parquet(PROCESSED / "cf_scores_val.parquet")
    cf_test = pd.read_parquet(PROCESSED / "cf_scores_test.parquet")

    specs = [
        # Full 200 epochs for LightGCN (its default, we used only 20 before)
        ("lgcn",  LightGCNRanker(emb_dim=64,  n_layers=3, n_epochs=200, batch_size=8192, lr=1e-3)),
        # Deeper DCN with more epochs
        ("dcn",   DCNRanker(emb_dim=128, cross_layers=3, deep=(256, 128, 64),
                            dropout=0.3, n_epochs=100, lr=5e-4, batch_size=4096)),
        # NeuMF with bigger embeddings + more epochs
        ("neumf", NeuMFRanker(emb_dim=64, mlp_layers=(256, 128, 64),
                              n_epochs=100, lr=5e-4, batch_size=4096)),
    ]

    for name, model in specs:
        val_s, test_s = _fit_score(name, model, train, val_u, val_i, test_u, test_i, log)
        # Overwrite existing column in-place
        cf_val[name]  = val_s.ravel().astype(np.float32)
        cf_test[name] = test_s.ravel().astype(np.float32)
        # Checkpoint after each model
        cf_val.to_parquet(PROCESSED / "cf_scores_val.parquet",   index=False)
        cf_test.to_parquet(PROCESSED / "cf_scores_test.parquet", index=False)
        log.info("  %s scores saved.", name)

    log.info("All neural models retrained and scores updated.")


if __name__ == "__main__":
    main()
