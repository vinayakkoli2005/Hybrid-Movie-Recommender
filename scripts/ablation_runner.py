"""Ablation study: drop one feature at a time, measure NDCG@10 impact.

For each feature in the meta-learner, we retrain LightGBM without it and
evaluate on the test set. The delta vs. the full model shows each feature's
marginal contribution.

Output: results/ablation.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd

from cf_pipeline.eval.metrics import ndcg_at_k, hit_rate_at_k
from cf_pipeline.features import ALL_FEAT_COLS, build_feature_matrix
from cf_pipeline.utils.logging import get_logger

PROCESSED   = Path("data/processed")
RESULTS     = Path("results")


def _train_lgbm(X_tr, y_tr, X_ev, y_ev, feat_names: list[str]) -> lgb.Booster:
    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names)
    eval_data  = lgb.Dataset(X_ev, label=y_ev, reference=train_data)
    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.05, "num_leaves": 63,
        "min_child_samples": 20, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1,
    }
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    return lgb.train(params, train_data, num_boost_round=500,
                     valid_sets=[eval_data], callbacks=callbacks)


def _eval_model(model, df_test: pd.DataFrame, feat_cols: list[str]) -> dict[str, float]:
    X = df_test[feat_cols].to_numpy(dtype=np.float32)
    preds = model.predict(X).astype(np.float32)
    df_test = df_test.copy()
    df_test["score"] = preds

    n_cands = df_test.groupby("user_id").size().iloc[0]
    n_users = df_test["user_id"].nunique()
    score_mat = preds.reshape(n_users, n_cands)
    return {
        "HR@10":   round(hit_rate_at_k(score_mat, 10), 4),
        "NDCG@10": round(ndcg_at_k(score_mat,   10), 4),
        "HR@5":    round(hit_rate_at_k(score_mat,  5), 4),
        "NDCG@5":  round(ndcg_at_k(score_mat,    5), 4),
    }


def main() -> None:
    log = get_logger("ablation_runner")
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----------------------------------------------------------
    log.info("Loading CF scores…")
    cf_val  = pd.read_parquet(PROCESSED / "cf_scores_val.parquet")
    cf_test = pd.read_parquet(PROCESSED / "cf_scores_test.parquet")

    llm_feats = None
    llm_path  = PROCESSED / "llm_features.parquet"
    if llm_path.exists():
        llm_feats = pd.read_parquet(llm_path)

    df_val  = build_feature_matrix(cf_val,  llm_feats, normalise=True)
    df_test = build_feature_matrix(cf_test, llm_feats, normalise=True)

    n = len(df_val)
    split = int(0.8 * n)
    X_tr, y_tr = df_val[ALL_FEAT_COLS].to_numpy()[:split],  df_val["label"].to_numpy()[:split]
    X_ev, y_ev = df_val[ALL_FEAT_COLS].to_numpy()[split:],  df_val["label"].to_numpy()[split:]

    results: dict[str, dict] = {}

    # ---- Full model (baseline for delta) ------------------------------------
    log.info("Training FULL model…")
    full_model = _train_lgbm(X_tr, y_tr, X_ev, y_ev, ALL_FEAT_COLS)
    results["full"] = _eval_model(full_model, df_test, ALL_FEAT_COLS)
    log.info("  full → NDCG@10=%.4f", results["full"]["NDCG@10"])

    # ---- Ablations: drop one feature at a time ------------------------------
    for drop_feat in ALL_FEAT_COLS:
        feat_subset = [f for f in ALL_FEAT_COLS if f != drop_feat]
        log.info("Ablation: drop %-15s (%d features left)…", drop_feat, len(feat_subset))

        Xtr_ab = df_val[feat_subset].to_numpy()[:split]
        Xev_ab = df_val[feat_subset].to_numpy()[split:]

        model = _train_lgbm(Xtr_ab, y_tr, Xev_ab, y_ev, feat_subset)
        metrics = _eval_model(model, df_test, feat_subset)

        delta_ndcg = round(metrics["NDCG@10"] - results["full"]["NDCG@10"], 4)
        delta_hr   = round(metrics["HR@10"]   - results["full"]["HR@10"],   4)
        results[f"drop_{drop_feat}"] = {**metrics, "delta_NDCG@10": delta_ndcg, "delta_HR@10": delta_hr}
        log.info(
            "  drop %-14s → NDCG@10=%.4f  Δ=%.4f",
            drop_feat, metrics["NDCG@10"], delta_ndcg,
        )

    # ---- Save ---------------------------------------------------------------
    out = RESULTS / "ablation.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("Saved → %s", out)

    # ---- Pretty summary -----------------------------------------------------
    log.info("\n=== Ablation Summary (sorted by NDCG@10 drop) ===")
    log.info("  %-18s  NDCG@10  Δ NDCG@10", "dropped feature")
    log.info("  %-18s  %.4f   %s", "none (full)", results["full"]["NDCG@10"], "—")
    drops = [
        (k.replace("drop_", ""), v)
        for k, v in results.items() if k.startswith("drop_")
    ]
    for feat, v in sorted(drops, key=lambda x: x[1]["delta_NDCG@10"]):
        log.info("  %-18s  %.4f   %+.4f", feat, v["NDCG@10"], v["delta_NDCG@10"])


if __name__ == "__main__":
    main()
