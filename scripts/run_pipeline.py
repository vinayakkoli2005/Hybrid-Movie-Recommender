"""End-to-end hybrid pipeline evaluation.

Loads:
  checkpoints/meta_lgbm.pkl             — trained meta-learner
  data/processed/cf_scores_test.parquet — CF scores for test candidates
  data/processed/llm_features.parquet   — LLM yes_prob (0.5 default if missing)

Scores every test (user, candidate) pair through the meta-learner, computes
HR@K and NDCG@K, and writes results/hybrid_pipeline.json.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from cf_pipeline.eval.metrics import all_metrics
from cf_pipeline.features import ALL_FEAT_COLS, build_feature_matrix
from cf_pipeline.utils.logging import get_logger

PROCESSED   = Path("data/processed")
CHECKPOINTS = Path("checkpoints")
RESULTS     = Path("results")


def _scores_to_matrix(df: pd.DataFrame, score_col: str) -> np.ndarray:
    """Reshape long-form scores back to (n_users, n_cands) with col-0 = positive."""
    # Each user has exactly n_cands rows; positive label = col-0 by construction
    n_cands = df.groupby("user_id").size().iloc[0]
    user_ids = df["user_id"].unique()
    n_users  = len(user_ids)

    # Rows are already in order (user×candidates) from dump_scores.py
    scores_flat = df[score_col].to_numpy(dtype=np.float32)
    return scores_flat.reshape(n_users, n_cands)


def main() -> None:
    log = get_logger("run_pipeline")
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ---- Load meta-learner --------------------------------------------------
    model_path = CHECKPOINTS / "meta_lgbm.pkl"
    if not model_path.exists():
        log.error("No meta-learner found at %s — run train_meta_learner.py first.", model_path)
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    log.info("Meta-learner loaded from %s", model_path)

    # ---- Load test CF scores -----------------------------------------------
    cf_test = pd.read_parquet(PROCESSED / "cf_scores_test.parquet")

    llm_feats = None
    llm_path  = PROCESSED / "llm_features.parquet"
    if llm_path.exists():
        llm_feats = pd.read_parquet(llm_path)

    # ---- Build feature matrix & score --------------------------------------
    log.info("Building test feature matrix…")
    df_test = build_feature_matrix(cf_test, llm_feats, normalise=True)
    X_test  = df_test[ALL_FEAT_COLS].to_numpy(dtype=np.float32)
    preds   = model.predict(X_test).astype(np.float32)
    df_test["hybrid_score"] = preds

    # ---- Reshape to (n_users, n_cands) and compute metrics -----------------
    score_mat = _scores_to_matrix(df_test, "hybrid_score")
    metrics   = all_metrics(score_mat, ks=(1, 5, 10, 20))

    log.info("=== Hybrid Pipeline Results ===")
    for k in (1, 5, 10, 20):
        log.info("  HR@%-2d = %.4f   NDCG@%-2d = %.4f", k, metrics[f"HR@{k}"], k, metrics[f"NDCG@{k}"])

    # ---- Save --------------------------------------------------------------
    payload = {"experiment": "hybrid_pipeline", "metrics": metrics}
    out_path = RESULTS / "hybrid_pipeline.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Results saved → %s", out_path)

    # ---- Individual CF model baselines for comparison ----------------------
    log.info("=== Individual CF Baselines (on test set) ===")
    from cf_pipeline.features import CF_COLS
    for col in CF_COLS:
        if col not in df_test.columns:
            continue
        mat = _scores_to_matrix(df_test, col)
        m   = all_metrics(mat, ks=(10,))
        log.info("  %-6s  HR@10=%.4f  NDCG@10=%.4f", col, m["HR@10"], m["NDCG@10"])


if __name__ == "__main__":
    main()
