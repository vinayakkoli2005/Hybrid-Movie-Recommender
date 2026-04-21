"""Train final LambdaRank model with best Optuna params and evaluate on test set."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd

from cf_pipeline.eval.metrics import all_metrics
from cf_pipeline.eval.protocol import build_candidate_matrix
from cf_pipeline.features_enhanced import (
    ENHANCED_FEAT_COLS,
    build_enhanced_feature_matrix,
    build_stats,
    split_Xy_grouped,
)
from cf_pipeline.utils.logging import get_logger

PROCESSED   = Path("data/processed")
RESULTS     = Path("results")
CHECKPOINTS = Path("checkpoints")


def main() -> None:
    log = get_logger("train_final_model")
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    # Load best params
    best = json.loads((RESULTS / "best_params.json").read_text())
    params = best["best_params"]
    log.info("Best val NDCG@10=%.4f (trial %d)", best["best_val_ndcg"], best["best_trial"])
    log.info("Params: %s", params)

    # Load data
    log.info("Loading data…")
    train   = pd.read_parquet(PROCESSED / "train.parquet")
    test_df = pd.read_parquet(PROCESSED / "test.parquet")
    cf_val  = pd.read_parquet(PROCESSED / "cf_scores_val.parquet")
    cf_test = pd.read_parquet(PROCESSED / "cf_scores_test.parquet")

    llm_feats = None
    if (PROCESSED / "llm_features.parquet").exists():
        llm_feats = pd.read_parquet(PROCESSED / "llm_features.parquet")
        log.info("LLM features loaded — %d rows, yes_prob mean=%.3f",
                 len(llm_feats), llm_feats["yes_prob"].mean())

    user_stats, item_stats = build_stats(train)
    df_val  = build_enhanced_feature_matrix(cf_val,  user_stats, item_stats, llm_feats)
    df_test = build_enhanced_feature_matrix(cf_test, user_stats, item_stats, llm_feats)

    # Train on full val set (90/10 split for early stopping)
    user_ids = df_val["user_id"].unique()
    np.random.seed(42)
    np.random.shuffle(user_ids)
    sp = int(0.9 * len(user_ids))
    sp_tr = set(user_ids[:sp])
    sp_ev = set(user_ids[sp:])

    df_tr = df_val[df_val["user_id"].isin(sp_tr)].reset_index(drop=True)
    df_ev = df_val[df_val["user_id"].isin(sp_ev)].reset_index(drop=True)

    X_tr, y_tr, g_tr = split_Xy_grouped(df_tr, ENHANCED_FEAT_COLS)
    X_ev, y_ev, g_ev = split_Xy_grouped(df_ev, ENHANCED_FEAT_COLS)
    X_te, _,  _      = split_Xy_grouped(df_test, ENHANCED_FEAT_COLS)

    base = {
        "objective":    "lambdarank",
        "metric":       "ndcg",
        "eval_at":      [5, 10],
        "verbose":      -1,
        "label_gain":   [0, 1],
        "device":       "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id":   0,
    }
    base.update(params)

    log.info("Training final model on full val set…")
    tr_ds = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=ENHANCED_FEAT_COLS)
    ev_ds = lgb.Dataset(X_ev, label=y_ev, group=g_ev, reference=tr_ds)
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(base, tr_ds, num_boost_round=2000,
                      valid_sets=[ev_ds], callbacks=callbacks)

    # Evaluate on test
    log.info("Evaluating on test set…")
    preds = model.predict(X_te).astype(np.float32)
    n_cands = df_test.groupby("user_id").size().iloc[0]
    n_users = df_test["user_id"].nunique()
    mat = preds.reshape(n_users, n_cands)

    # Build item_ids matrix and popularity dict for novelty
    _, test_item_ids = build_candidate_matrix(test_df)
    item_popularity = train.groupby("item_id").size().to_dict()
    n_train = len(train)

    metrics = all_metrics(
        mat, ks=(1, 5, 10, 20),
        item_ids=test_item_ids,
        item_popularity=item_popularity,
        n_train=n_train,
    )

    log.info("=== Final Model — Test Results ===")
    for k in (1, 5, 10, 20):
        log.info("  HR@%-2d = %.4f   NDCG@%-2d = %.4f   MAP@%-2d = %.4f   MAR@%-2d = %.4f   Novelty@%-2d = %.4f",
                 k, metrics[f"HR@{k}"],
                 k, metrics[f"NDCG@{k}"],
                 k, metrics[f"MAP@{k}"],
                 k, metrics[f"MAR@{k}"],
                 k, metrics[f"Novelty@{k}"])

    # Save model
    model_path = CHECKPOINTS / "meta_lgbm_tuned.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info("Model saved → %s", model_path)

    # Save results
    payload = {
        "experiment": "tuned_lambdarank_pipeline",
        "best_val_ndcg": round(best["best_val_ndcg"], 4),
        "best_params": params,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
    }
    (RESULTS / "tuned_pipeline.json").write_text(json.dumps(payload, indent=2))
    log.info("Results saved → results/tuned_pipeline.json")

    # Feature importance
    importance = dict(zip(ENHANCED_FEAT_COLS, model.feature_importance("gain")))
    log.info("Feature importance (top 5):")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
        log.info("  %-25s %.1f", feat, imp)


if __name__ == "__main__":
    main()
