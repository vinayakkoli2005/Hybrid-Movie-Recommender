"""Fit all CF models on train, score val + test candidate sets, save to disk.

Output:
  data/processed/cf_scores_val.parquet   — used to train the meta-learner
  data/processed/cf_scores_test.parquet  — used to evaluate the full pipeline

Columns: user_id, item_id, label, pop, knn, bpr, ease, lgcn, dcn, neumf

Run (uses GPU 1 so LoRA on GPU 0 is undisturbed):
  CUDA_VISIBLE_DEVICES=1 python3.8 scripts/dump_scores.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from cf_pipeline.eval.protocol import build_candidate_matrix
from cf_pipeline.models.baselines import ItemKNNRanker, PopularityRanker
from cf_pipeline.models.bpr_mf import BPRMFRanker
from cf_pipeline.models.dcn import DCNRanker
from cf_pipeline.models.ease import EASERRanker
from cf_pipeline.models.lightgcn import LightGCNRanker
from cf_pipeline.models.neumf import NeuMFRanker
from cf_pipeline.utils.logging import get_logger

PROCESSED = Path("data/processed")


def _candidates_to_df(
    eval_df: pd.DataFrame,
    scores_dict: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Flatten (n_users, n_cands) score arrays into long-form rows."""
    user_ids = eval_df["user_id"].to_numpy()
    pos = eval_df["positive"].to_numpy()
    negs = np.stack(eval_df["negatives"].tolist())           # (n, 99)
    item_ids = np.concatenate([pos.reshape(-1, 1), negs], axis=1)  # (n, 100)

    n_users, n_cands = item_ids.shape
    labels = np.zeros((n_users, n_cands), dtype=np.int8)
    labels[:, 0] = 1   # col-0 is always the positive

    rows = []
    uid_rep = np.repeat(user_ids, n_cands)
    iid_flat = item_ids.ravel()
    lbl_flat = labels.ravel()

    df = pd.DataFrame({"user_id": uid_rep, "item_id": iid_flat, "label": lbl_flat})
    for name, score_mat in scores_dict.items():
        df[name] = score_mat.ravel().astype(np.float32)

    return df


def _fit_and_score(name, model, train, user_ids, item_ids, log):
    t0 = time.time()
    log.info("Fitting %s…", name)
    model.fit(train)
    log.info("  fit done (%.0fs) — scoring…", time.time() - t0)
    scores = model.score(user_ids, item_ids)
    log.info("  %s done (%.0fs total)", name, time.time() - t0)
    return scores


def main() -> None:
    log = get_logger("dump_scores")

    train = pd.read_parquet(PROCESSED / "train.parquet")
    val   = pd.read_parquet(PROCESSED / "val.parquet")
    test  = pd.read_parquet(PROCESSED / "test.parquet")

    val_users,  val_items  = build_candidate_matrix(val)
    test_users, test_items = build_candidate_matrix(test)

    # ---------------------------------------------------------------
    # Models to fit — quick ones first, neural last
    # ---------------------------------------------------------------
    model_specs = [
        ("pop",   PopularityRanker()),
        ("knn",   ItemKNNRanker(k_neighbors=20, shrinkage=10.0)),
        ("ease",  EASERRanker(reg_lambda=500.0)),
        ("bpr",   BPRMFRanker(emb_dim=64, n_epochs=20)),
        ("lgcn",  LightGCNRanker(emb_dim=64, n_layers=3, n_epochs=20, batch_size=8192)),
        ("dcn",   DCNRanker(emb_dim=64, cross_layers=3, n_epochs=20)),
        ("neumf", NeuMFRanker(emb_dim=64, n_epochs=20)),
    ]

    val_scores:  dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}

    for name, model in model_specs:
        try:
            val_scores[name]  = _fit_and_score(name, model, train, val_users,  val_items,  log)
            test_scores[name] = model.score(test_users, test_items)   # model already fitted
        except Exception as exc:
            log.warning("  %s FAILED — filling with zeros. Error: %s", name, exc)
            val_scores[name]  = np.zeros((len(val_users),  val_items.shape[1]),  dtype=np.float32)
            test_scores[name] = np.zeros((len(test_users), test_items.shape[1]), dtype=np.float32)

    log.info("Assembling and saving score dataframes…")
    val_df  = _candidates_to_df(val,  val_scores)
    test_df = _candidates_to_df(test, test_scores)

    val_df.to_parquet(PROCESSED / "cf_scores_val.parquet",   index=False)
    test_df.to_parquet(PROCESSED / "cf_scores_test.parquet", index=False)

    log.info(
        "Done — val: %d rows, test: %d rows",
        len(val_df), len(test_df),
    )


if __name__ == "__main__":
    main()
