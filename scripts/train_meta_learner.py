"""Train LightGBM meta-learner on CF + LLM features from the val split.

Reads:
  data/processed/cf_scores_val.parquet   — CF scores for val candidates
  data/processed/llm_features.parquet    — LLM yes_prob (0.5 default for unknowns)

Writes:
  checkpoints/meta_lgbm.pkl              — trained LightGBM model
  results/meta_learner_val_auc.txt       — validation AUC for monitoring

Re-run this script after Task 29 (LoRA LLM features) to add the LLM signal.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cf_pipeline.features import ALL_FEAT_COLS, build_feature_matrix, split_Xy
from cf_pipeline.utils.logging import get_logger

PROCESSED   = Path("data/processed")
CHECKPOINTS = Path("checkpoints")
RESULTS     = Path("results")


def main() -> None:
    log = get_logger("train_meta_learner")

    # ---- Load inputs -------------------------------------------------------
    log.info("Loading CF scores…")
    cf_val = pd.read_parquet(PROCESSED / "cf_scores_val.parquet")

    llm_feats = None
    llm_path  = PROCESSED / "llm_features.parquet"
    if llm_path.exists():
        llm_feats = pd.read_parquet(llm_path)
        log.info("LLM features loaded: %d rows", len(llm_feats))
    else:
        log.info("No LLM features found — llm_yes_prob will default to 0.5")

    # ---- Build feature matrix ----------------------------------------------
    log.info("Building feature matrix…")
    df_val = build_feature_matrix(cf_val, llm_feats, normalise=True)
    X_val, y_val = split_Xy(df_val)
    log.info(
        "Feature matrix: %d rows, %d features  (pos=%d, neg=%d)",
        len(y_val), X_val.shape[1], int(y_val.sum()), int((y_val == 0).sum()),
    )

    # 80 / 20 split within val for early stopping
    n = len(X_val)
    split = int(0.8 * n)
    X_tr, y_tr = X_val[:split], y_val[:split]
    X_ev, y_ev = X_val[split:], y_val[split:]

    # ---- Train LightGBM ----------------------------------------------------
    log.info("Training LightGBM…")
    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=ALL_FEAT_COLS)
    eval_data  = lgb.Dataset(X_ev, label=y_ev, reference=train_data)

    params = {
        "objective":       "binary",
        "metric":          "auc",
        "learning_rate":   0.05,
        "num_leaves":      63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":    5,
        "verbose":         -1,
    }

    callbacks = [lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[eval_data],
        callbacks=callbacks,
    )

    # ---- Evaluate on hold-out -----------------------------------------------
    preds_ev = model.predict(X_ev)
    auc = roc_auc_score(y_ev, preds_ev)
    log.info("Val AUC: %.4f", auc)

    # ---- Feature importance -------------------------------------------------
    importance = dict(zip(ALL_FEAT_COLS, model.feature_importance("gain")))
    log.info("Feature importance (gain):")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        log.info("  %-15s %.1f", feat, imp)

    # ---- Save ---------------------------------------------------------------
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    model_path = CHECKPOINTS / "meta_lgbm.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info("Model saved → %s", model_path)

    (RESULTS / "meta_learner_val_auc.txt").write_text(f"val_auc={auc:.4f}\n")
    log.info("Done.")


if __name__ == "__main__":
    main()
