"""Evaluate the hybrid pipeline and CF baselines on cold users.

Cold users are defined two ways:
  (a) bottom-quartile: users with < 25 training interactions (~2,749 users)
  (b) profiled:        the 19 users in cold_start_profiles.json

Outputs:
  results/cold_user_table.json   — HR/NDCG for each model × user group
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
from cf_pipeline.features import ALL_FEAT_COLS, CF_COLS, build_feature_matrix
from cf_pipeline.features_enhanced import (
    ENHANCED_FEAT_COLS,
    build_enhanced_feature_matrix,
    build_stats,
)
from cf_pipeline.utils.logging import get_logger

PROCESSED   = Path("data/processed")
RESULTS     = Path("results")
CHECKPOINTS = Path("checkpoints")

COLD_THRESHOLD = 25   # interactions in training → "cold" user


def _score_matrix_for_users(df: pd.DataFrame, score_col: str, user_ids: list[int]) -> np.ndarray:
    sub = df[df["user_id"].isin(user_ids)].copy()
    if sub.empty:
        return np.empty((0, 0))
    n_cands = sub.groupby("user_id").size().iloc[0]
    return sub[score_col].to_numpy(dtype=np.float32).reshape(-1, n_cands)


def _eval_group(df_test: pd.DataFrame, user_ids: list[int]) -> dict[str, dict]:
    """Return HR@10 / NDCG@10 for each model for the given user subset."""
    metrics: dict[str, dict] = {}

    # CF baselines
    for col in CF_COLS:
        if col not in df_test.columns:
            continue
        mat = _score_matrix_for_users(df_test, col, user_ids)
        if mat.shape[0] == 0:
            continue
        m = all_metrics(mat, ks=(5, 10, 20))
        metrics[col] = {k: round(v, 4) for k, v in m.items()}

    # Hybrid pipeline (original binary LightGBM)
    if "hybrid_score" in df_test.columns:
        mat = _score_matrix_for_users(df_test, "hybrid_score", user_ids)
        if mat.shape[0] > 0:
            m = all_metrics(mat, ks=(5, 10, 20))
            metrics["hybrid"] = {k: round(v, 4) for k, v in m.items()}

    # Hybrid pipeline (tuned LambdaRank)
    if "hybrid_tuned_score" in df_test.columns:
        mat = _score_matrix_for_users(df_test, "hybrid_tuned_score", user_ids)
        if mat.shape[0] > 0:
            m = all_metrics(mat, ks=(5, 10, 20))
            metrics["hybrid_tuned"] = {k: round(v, 4) for k, v in m.items()}

    return metrics


def main() -> None:
    log = get_logger("cold_user_eval")
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----------------------------------------------------------
    train   = pd.read_parquet(PROCESSED / "train.parquet")
    cf_test = pd.read_parquet(PROCESSED / "cf_scores_test.parquet")

    llm_feats = None
    llm_path  = PROCESSED / "llm_features.parquet"
    if llm_path.exists():
        llm_feats = pd.read_parquet(llm_path)

    df_test_basic = build_feature_matrix(cf_test, llm_feats, normalise=True)

    # Score with original binary meta-learner
    model_path = CHECKPOINTS / "meta_lgbm.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            meta = pickle.load(f)
        X = df_test_basic[ALL_FEAT_COLS].to_numpy(dtype=np.float32)
        df_test_basic["hybrid_score"] = meta.predict(X).astype(np.float32)
    else:
        log.warning("No meta-learner found — hybrid scores will be skipped.")

    # Score with tuned LambdaRank meta-learner (enhanced features)
    user_stats, item_stats = build_stats(train)
    df_test_enh = build_enhanced_feature_matrix(cf_test, user_stats, item_stats, llm_feats)
    tuned_path = CHECKPOINTS / "meta_lgbm_tuned.pkl"
    if tuned_path.exists():
        with open(tuned_path, "rb") as f:
            meta_tuned = pickle.load(f)
        X_enh = df_test_enh[ENHANCED_FEAT_COLS].to_numpy(dtype=np.float32)
        df_test_basic = df_test_basic.merge(
            df_test_enh[["user_id", "item_id"]].assign(
                hybrid_tuned_score=meta_tuned.predict(X_enh).astype(np.float32)
            ),
            on=["user_id", "item_id"], how="left",
        )
    else:
        log.warning("No tuned meta-learner found — tuned hybrid scores will be skipped.")

    df_test = df_test_basic

    # ---- Define cold user groups -------------------------------------------
    interactions = train.groupby("user_id").size().reset_index(name="n")
    test_users   = set(cf_test["user_id"].unique())

    # Group (a): bottom-quartile by interaction count
    cold_bq = set(
        interactions[interactions["n"] < COLD_THRESHOLD]["user_id"].tolist()
    ) & test_users

    # Group (b): profiled cold users from build_cold_start_profiles.py
    profiles_path = PROCESSED / "cold_start_profiles.json"
    cold_profiled: set[int] = set()
    if profiles_path.exists():
        profiles = json.loads(profiles_path.read_text())
        cold_profiled = {int(uid) for uid in profiles.keys()} & test_users

    # All test users (full benchmark)
    all_test = list(test_users)

    log.info("User groups — all: %d, cold_bq (<25 int): %d, cold_profiled: %d",
             len(all_test), len(cold_bq), len(cold_profiled))

    # ---- Evaluate -----------------------------------------------------------
    groups = {
        "all_users":       all_test,
        "cold_bottom_quartile": list(cold_bq),
        "cold_profiled":   list(cold_profiled),
    }

    output: dict[str, dict] = {}
    for group_name, uids in groups.items():
        if not uids:
            log.info("Skipping %s — no users in test set.", group_name)
            continue
        log.info("Evaluating group '%s' (%d users)…", group_name, len(uids))
        output[group_name] = _eval_group(df_test, uids)

        # Log summary
        for model, m in output[group_name].items():
            log.info("  %-10s  HR@10=%.4f  NDCG@10=%.4f", model, m.get("HR@10", 0), m.get("NDCG@10", 0))

    # ---- Save ---------------------------------------------------------------
    out_path = RESULTS / "cold_user_table.json"
    out_path.write_text(json.dumps(output, indent=2))
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
