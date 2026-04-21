"""LambdaRank meta-learner with Optuna hyperparameter search.

Uses:
  - LambdaRank objective (directly optimises NDCG@10)
  - Enhanced feature engineering (user/item stats + CF scores)
  - Optuna (300 trials) to find best LightGBM hyperparameters

Reads:
  data/processed/cf_scores_val.parquet
  data/processed/cf_scores_test.parquet

Writes:
  checkpoints/meta_lgbm_tuned.pkl   — best model
  results/tuned_pipeline.json       — test-set HR/NDCG

Re-run after retrain_neural_models.py finishes for even better scores.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from cf_pipeline.eval.metrics import all_metrics
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
N_TRIALS    = 80


def _score_matrix(df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    n_cands = df.groupby("user_id").size().iloc[0]
    n_users = df["user_id"].nunique()
    return preds.reshape(n_users, n_cands)


def _train_lambdarank(params: dict, X_tr, y_tr, groups_tr, X_ev, y_ev, groups_ev):
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
    tr_ds = lgb.Dataset(X_tr, label=y_tr, group=groups_tr,
                        feature_name=ENHANCED_FEAT_COLS)
    ev_ds = lgb.Dataset(X_ev, label=y_ev, group=groups_ev,
                        reference=tr_ds)
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(base, tr_ds, num_boost_round=1000,
                      valid_sets=[ev_ds], callbacks=callbacks)
    return model


def main() -> None:
    log = get_logger("tune_meta_learner")
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    log.info("Loading data…")
    train   = pd.read_parquet(PROCESSED / "train.parquet")
    cf_val  = pd.read_parquet(PROCESSED / "cf_scores_val.parquet")
    cf_test = pd.read_parquet(PROCESSED / "cf_scores_test.parquet")

    llm_feats = None
    if (PROCESSED / "llm_features.parquet").exists():
        llm_feats = pd.read_parquet(PROCESSED / "llm_features.parquet")

    log.info("Building user/item stats…")
    user_stats, item_stats = build_stats(train)

    log.info("Building feature matrices…")
    df_val  = build_enhanced_feature_matrix(cf_val,  user_stats, item_stats, llm_feats)
    df_test = build_enhanced_feature_matrix(cf_test, user_stats, item_stats, llm_feats)

    log.info("Feature columns (%d): %s", len(ENHANCED_FEAT_COLS), ENHANCED_FEAT_COLS)

    # 80/20 split within val for Optuna
    user_ids = df_val["user_id"].unique()
    np.random.seed(42)
    np.random.shuffle(user_ids)
    split_idx = int(0.8 * len(user_ids))
    train_uids = set(user_ids[:split_idx])
    eval_uids  = set(user_ids[split_idx:])

    df_tr = df_val[df_val["user_id"].isin(train_uids)].reset_index(drop=True)
    df_ev = df_val[df_val["user_id"].isin(eval_uids)].reset_index(drop=True)

    X_tr, y_tr, g_tr = split_Xy_grouped(df_tr, ENHANCED_FEAT_COLS)
    X_ev, y_ev, g_ev = split_Xy_grouped(df_ev, ENHANCED_FEAT_COLS)
    X_te, _, _       = split_Xy_grouped(df_test, ENHANCED_FEAT_COLS)

    log.info("Train users: %d (%d rows)  |  Eval users: %d (%d rows)",
             len(train_uids), len(df_tr), len(eval_uids), len(df_ev))

    # ── Optuna objective ─────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate":     trial.suggest_float("learning_rate",     0.01,  0.3,   log=True),
            "num_leaves":        trial.suggest_int(  "num_leaves",        31,    511),
            "min_child_samples": trial.suggest_int(  "min_child_samples", 5,     100),
            "feature_fraction":  trial.suggest_float("feature_fraction",  0.4,   1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction",  0.4,   1.0),
            "bagging_freq":      trial.suggest_int(  "bagging_freq",      1,     10),
            "lambda_l1":         trial.suggest_float("lambda_l1",         1e-8,  10.0, log=True),
            "lambda_l2":         trial.suggest_float("lambda_l2",         1e-8,  10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0,   1.0),
            "max_depth":         trial.suggest_int(  "max_depth",         3,     12),
        }
        try:
            model = _train_lambdarank(params, X_tr, y_tr, g_tr, X_ev, y_ev, g_ev)
            preds = model.predict(X_ev).astype(np.float32)
            mat   = _score_matrix(df_ev, preds)
            from cf_pipeline.eval.metrics import ndcg_at_k
            return ndcg_at_k(mat, 10)
        except Exception:
            return 0.0

    # ── Run Optuna ───────────────────────────────────────────────────────────
    log.info("Running Optuna (%d trials)…", N_TRIALS)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # SQLite storage — survives interrupts, allows resume
    storage_path = CHECKPOINTS / "optuna_study.db"
    storage = f"sqlite:///{storage_path}"
    study = optuna.create_study(
        study_name="lambdarank_tuning",
        direction="maximize",
        sampler=TPESampler(seed=42),
        storage=storage,
        load_if_exists=True,  # resume from previous run if interrupted
    )

    # Warm-start only if starting fresh (no completed trials yet)
    if len(study.trials) == 0:
        study.enqueue_trial({
            "learning_rate": 0.05, "num_leaves": 127, "min_child_samples": 20,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "lambda_l1": 0.1, "lambda_l2": 0.1, "min_gain_to_split": 0.0, "max_depth": 7,
        })
    else:
        log.info("Resuming study — %d trials already done, best=%.4f",
                 len(study.trials), study.best_value)

    best_params_path = RESULTS / "best_params.json"

    def _log_callback(study, trial):
        if trial.number % 20 == 0 or trial.value == study.best_value:
            log.info("  Trial %3d | NDCG@10=%.4f | best=%.4f",
                     trial.number, trial.value, study.best_value)
        # Save best params to disk every time a new best is found
        if trial.value == study.best_value:
            best_params_path.write_text(json.dumps({
                "best_val_ndcg": round(study.best_value, 4),
                "best_trial":    study.best_trial.number,
                "best_params":   study.best_params,
            }, indent=2))

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[_log_callback], n_jobs=1)

    log.info("Best NDCG@10 on val: %.4f", study.best_value)
    log.info("Best params: %s", study.best_params)

    # ── Train final model with best params on full val set ───────────────────
    log.info("Training final model on full val set…")
    X_full, y_full, g_full = split_Xy_grouped(df_val, ENHANCED_FEAT_COLS)
    # Use 90/10 split for early stopping in final training
    n_full = df_val["user_id"].nunique()
    sp = int(0.9 * n_full)
    sp_uids_tr = set(user_ids[:sp])
    sp_uids_ev = set(user_ids[sp:])
    df_sp_tr = df_val[df_val["user_id"].isin(sp_uids_tr)].reset_index(drop=True)
    df_sp_ev = df_val[df_val["user_id"].isin(sp_uids_ev)].reset_index(drop=True)
    Xf_tr, yf_tr, gf_tr = split_Xy_grouped(df_sp_tr, ENHANCED_FEAT_COLS)
    Xf_ev, yf_ev, gf_ev = split_Xy_grouped(df_sp_ev, ENHANCED_FEAT_COLS)

    final_model = _train_lambdarank(study.best_params, Xf_tr, yf_tr, gf_tr, Xf_ev, yf_ev, gf_ev)

    # ── Evaluate on test ─────────────────────────────────────────────────────
    log.info("Evaluating on test set…")
    preds_test = final_model.predict(X_te).astype(np.float32)
    mat_test   = _score_matrix(df_test, preds_test)
    metrics    = all_metrics(mat_test, ks=(1, 5, 10, 20))

    log.info("=== Tuned LambdaRank Pipeline ===")
    for k in (1, 5, 10, 20):
        log.info("  HR@%-2d = %.4f   NDCG@%-2d = %.4f",
                 k, metrics[f"HR@{k}"], k, metrics[f"NDCG@{k}"])

    # ── Save ─────────────────────────────────────────────────────────────────
    model_path = CHECKPOINTS / "meta_lgbm_tuned.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    log.info("Model saved → %s", model_path)

    payload = {
        "experiment": "tuned_lambdarank_pipeline",
        "best_val_ndcg": round(study.best_value, 4),
        "best_params": study.best_params,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
    }
    (RESULTS / "tuned_pipeline.json").write_text(json.dumps(payload, indent=2))
    log.info("Results saved → results/tuned_pipeline.json")

    # Feature importance
    importance = dict(zip(ENHANCED_FEAT_COLS, final_model.feature_importance("gain")))
    log.info("Feature importance (top 5):")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
        log.info("  %-25s %.1f", feat, imp)


if __name__ == "__main__":
    main()
