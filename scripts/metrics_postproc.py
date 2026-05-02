"""Post-process metrics and reproduce key PDF figures:
- Diversity vs NDCG (avg across users for K in [1,5,10,20])
- Novelty histogram for K=10 (per-user novelty distribution)

Saves PNGs to `results/figures/`.

Run: python scripts/metrics_postproc.py
"""
from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    train = pd.read_parquet(PROCESSED / "train.parquet")
    test = pd.read_parquet(PROCESSED / "test.parquet")
    cf_val = pd.read_parquet(PROCESSED / "cf_scores_val.parquet")
    cf_test = pd.read_parquet(PROCESSED / "cf_scores_test.parquet")
    llm_feats = None
    if (PROCESSED / "llm_features.parquet").exists():
        llm_feats = pd.read_parquet(PROCESSED / "llm_features.parquet")
    cf_all = pd.concat([cf_val, cf_test], ignore_index=True).drop_duplicates(subset=["user_id", "item_id"])
    return train, test, cf_all, llm_feats


def build_predictions(cf_all: pd.DataFrame, train: pd.DataFrame, llm_feats: pd.DataFrame | None):
    # Import local feature builder (same logic as app.py)
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from cf_pipeline.features_enhanced import (
        ENHANCED_FEAT_COLS,
        build_enhanced_feature_matrix,
        build_stats,
    )

    user_stats, item_stats = build_stats(train)
    df_all = build_enhanced_feature_matrix(cf_all, user_stats, item_stats, llm_feats)

    # Load model
    model_path = CHECKPOINTS / "meta_lgbm_tuned.pkl"
    if not model_path.exists():
        model_path = CHECKPOINTS / "meta_lgbm.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X = df_all[ENHANCED_FEAT_COLS].to_numpy(dtype=np.float32)
    df_all["pred_score"] = model.predict(X).astype(np.float32)
    df_all["rank"] = (
        df_all.groupby("user_id")["pred_score"].rank(ascending=False, method="average").astype(float)
    )
    df_all["n_cands"] = df_all.groupby("user_id")["rank"].transform("count")
    df_all["rank"] = df_all["rank"].round().astype(int)

    return df_all


def per_user_topk(df_all: pd.DataFrame, k: int) -> Dict[int, list[int]]:
    topk = df_all[df_all["rank"] <= k].copy()
    out: Dict[int, list[int]] = {}
    for uid, g in topk.groupby("user_id"):
        out[int(uid)] = [int(x) for x in g.sort_values("rank")["item_id"].tolist()]
    return out


def compute_diversity(topk: Dict[int, list[int]], items_lookup: Dict[int, dict]) -> Dict[int, int]:
    divers = {}
    for uid, lst in topk.items():
        genres = set()
        for iid in lst:
            for g in items_lookup.get(int(iid), {}).get("genres", "").split("|"):
                g = g.strip()
                if g:
                    genres.add(g)
        divers[uid] = len(genres)
    return divers


def compute_novelty(topk: Dict[int, list[int]], item_popularity: Dict[int, int], n_train: int) -> Dict[int, float]:
    nov = {}
    for uid, lst in topk.items():
        vals = []
        for iid in lst:
            pop = item_popularity.get(int(iid), 1)
            vals.append(-math.log2(pop / max(1, n_train)))
        nov[uid] = float(np.mean(vals)) if vals else 0.0
    return nov


def compute_ndcg_per_user(topk: Dict[int, list[int]], test_map: Dict[int, int], k: int) -> Dict[int, float]:
    ndcg = {}
    for uid, lst in topk.items():
        true = test_map.get(uid)
        if true is None:
            ndcg[uid] = 0.0
            continue
        try:
            rank = lst.index(true) + 1
        except ValueError:
            rank = None
        if rank is None or rank > k:
            ndcg[uid] = 0.0
        else:
            ndcg[uid] = 1.0 / math.log2(rank + 1)
    return ndcg


def compute_personalization(rec_lists: Iterable[list[int]], sample_size: int = 400) -> float:
    recs = list(rec_lists)
    if len(recs) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(recs), size=sample_size, replace=False)
        recs = [recs[i] for i in idx]
    sets = [set(r) for r in recs]
    dissimilarities = []
    for i in range(len(sets)):
        a = sets[i]
        for j in range(i + 1, len(sets)):
            b = sets[j]
            union = len(a | b)
            if union == 0:
                continue
            jaccard = len(a & b) / union
            dissimilarities.append(1.0 - jaccard)
    return float(np.mean(dissimilarities)) if dissimilarities else 0.0


def plot_diversity_vs_ndcg(k_list, avg_diversity, avg_ndcg, out_path: Path):
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(k_list, avg_diversity, marker="o", color="#ff79c6", label="Diversity")
    ax1.set_xlabel("K (top-K)")
    ax1.set_ylabel("Avg Diversity (unique genres)", color="#ff79c6")
    ax1.tick_params(axis="y", labelcolor="#ff79c6")

    ax2 = ax1.twinx()
    ax2.plot(k_list, avg_ndcg, marker="s", color="#8be9fd", label="NDCG")
    ax2.set_ylabel("Avg NDCG", color="#8be9fd")
    ax2.tick_params(axis="y", labelcolor="#8be9fd")

    fig.tight_layout()
    safe_mkdir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_novelty_hist(nov_vals, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(list(nov_vals), bins=40, color="#bd93f9", edgecolor="#222"
            )
    ax.set_xlabel("Novelty (per-user avg top-K)")
    ax.set_ylabel("Users")
    fig.tight_layout()
    safe_mkdir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    train, test, cf_all, llm_feats = load_data()
    df_all = build_predictions(cf_all, train, llm_feats)

    items = pd.read_parquet(PROCESSED / "items_metadata.parquet")
    items_lookup = items.set_index("item_id").to_dict(orient="index")

    # Build test map (single positive per user). Support different test parquet schemas.
    if "item_id" in test.columns:
        pos_col = "item_id"
    elif "positive" in test.columns:
        pos_col = "positive"
    else:
        raise RuntimeError("Unable to find positive item column in test.parquet (expected 'item_id' or 'positive')")
    test_map = dict(zip(test["user_id"].astype(int).tolist(), test[pos_col].astype(int).tolist()))

    item_pop = train.groupby("item_id").size().to_dict()
    n_train = len(train)

    ks = [1, 5, 10, 20]
    avg_diversity = []
    avg_ndcg = []

    all_rec_lists = []
    for k in ks:
        topk = per_user_topk(df_all, k)
        divers = compute_diversity(topk, items_lookup)
        avg_diversity.append(float(np.mean(list(divers.values()))) if divers else 0.0)

        ndcgs = compute_ndcg_per_user(topk, test_map, k)
        avg_ndcg.append(float(np.mean(list(ndcgs.values()))) if ndcgs else 0.0)

        if k == 10:
            nov = compute_novelty(topk, item_pop, n_train)
            plot_novelty_hist(list(nov.values()), RESULTS / "figures" / "novelty_hist_k10.png")

        all_rec_lists.extend([lst for lst in topk.values()])

    personalization = compute_personalization(all_rec_lists)
    coverage = len({iid for lst in all_rec_lists for iid in lst}) / max(1, items["item_id"].nunique())

    # Save summary
    safe_mkdir(RESULTS)
    summary = {
        "ks": ks,
        "avg_diversity": avg_diversity,
        "avg_ndcg": avg_ndcg,
        "personalization": personalization,
        "coverage": coverage,
    }
    # Also compute frozen-negative protocol metrics (match tuned_pipeline.json)
    # Build per-user score rows using eval_negatives.json
    import json
    neg_path = Path(PROCESSED / "eval_negatives.json")
    frozen_metrics = {}
    if neg_path.exists():
        neg_data = json.loads(neg_path.read_text())
        # support both dict with 'test' list or mapping where key->entry
        test_entries = neg_data.get("test") if isinstance(neg_data, dict) and "test" in neg_data else neg_data
        rows = []
        items_rows = []
        for entry in test_entries:
            uid = int(entry["user_id"]) if isinstance(entry, dict) else None
            if uid is None:
                continue
            pos = int(entry.get("positive", entry.get("pos", None)))
            cand_list = [pos] + entry.get("negatives", [])
            items_rows.append(cand_list)
            # fetch scores
            scores_for_uid = []
            for iid in cand_list:
                sel = df_all[(df_all["user_id"] == uid) & (df_all["item_id"] == iid)]
                if not sel.empty:
                    scores_for_uid.append(float(sel["pred_score"].iloc[0]))
                else:
                    scores_for_uid.append(-1e6)
            rows.append(scores_for_uid)
        if rows:
            from cf_pipeline.eval.metrics import all_metrics

            scores_arr = np.array(rows)
            items_arr = np.array(items_rows)
            frozen_metrics = all_metrics(
                scores_arr,
                ks=tuple(ks),
                item_ids=items_arr,
                item_popularity=train.groupby("item_id").size().to_dict(),
                n_train=len(train),
            )
    summary["frozen_metrics"] = frozen_metrics
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    (RESULTS / "figures" / "diversity_vs_ndcg.png").unlink(missing_ok=True)
    plot_diversity_vs_ndcg(ks, avg_diversity, avg_ndcg, RESULTS / "figures" / "diversity_vs_ndcg.png")

    # Save summary JSON/text
    import json

    (RESULTS / "figures" / "summary.json").write_text(json.dumps(summary, indent=2))

    print("Figures written to:", RESULTS / "figures")


if __name__ == "__main__":
    main()
