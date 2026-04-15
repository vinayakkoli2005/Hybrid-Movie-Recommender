"""One-shot data preparation pipeline.

Runs the full sequence:
  load raw data → binarize → join TMDB → leave-one-out split → freeze negatives

Writes to data/processed/:
  train.parquet          — training interactions (user_id, item_id, timestamp)
  val.parquet            — val set   (user_id, positive, negatives)
  test.parquet           — test set  (user_id, positive, negatives)
  items_metadata.parquet — item catalogue with TMDB fields
  eval_negatives.json    — frozen JSON snapshot for reproducibility audits

Run from the project root:
    python scripts/prepare_data.py
"""
import json
from pathlib import Path

from cf_pipeline.data.binarize import binarize_ratings
from cf_pipeline.data.join_tmdb import join_movies_with_tmdb
from cf_pipeline.data.loaders import (
    load_links,
    load_ml1m_movies,
    load_ml1m_ratings,
    load_tmdb_metadata,
)
from cf_pipeline.data.negatives import sample_negatives
from cf_pipeline.data.splits import leave_one_out_split
from cf_pipeline.utils.logging import get_logger
from cf_pipeline.utils.seeds import set_global_seed

RAW  = Path("data/raw")
OUT  = Path("data/processed")
SEED = 42


def main() -> None:
    log = get_logger("prepare_data")
    set_global_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw ML-1M ────────────────────────────────────────────────────
    log.info("Loading ML-1M ratings…")
    ratings = load_ml1m_ratings(RAW / "ml-1m")
    log.info(
        f"  {len(ratings):,} raw ratings | "
        f"{ratings['user_id'].nunique():,} users | "
        f"{ratings['item_id'].nunique():,} items"
    )

    # ── 2. Binarize (keep rating >= 4) ───────────────────────────────────────
    log.info("Binarizing (rating >= 4)…")
    pos = binarize_ratings(ratings, threshold=4)
    log.info(
        f"  {len(pos):,} positive interactions "
        f"({100 * len(pos) / len(ratings):.1f}% of raw)"
    )

    # ── 3. Load movies + links + TMDB; inner-join ────────────────────────────
    log.info("Loading movies, links, TMDB metadata…")
    movies     = load_ml1m_movies(RAW / "ml-1m")
    links      = load_links(RAW / "ml-1m")
    tmdb       = load_tmdb_metadata(RAW / "tmdb")
    items_meta = join_movies_with_tmdb(movies, links, tmdb)
    log.info(
        f"  {len(items_meta):,} / {len(movies):,} items matched with TMDB "
        f"({100 * len(items_meta) / len(movies):.1f}%)"
    )

    # ── 4. Drop interactions for items without TMDB metadata ─────────────────
    valid_items = set(items_meta["item_id"])
    before      = len(pos)
    pos         = pos[pos["item_id"].isin(valid_items)].reset_index(drop=True)
    log.info(
        f"  Filtered interactions: {before:,} → {len(pos):,} "
        f"({before - len(pos):,} dropped, items not in TMDB)"
    )
    log.info(
        f"  Users after filter: {pos['user_id'].nunique():,}"
    )

    # ── 5. Leave-one-out split ───────────────────────────────────────────────
    log.info("Leave-one-out split (min_interactions=3)…")
    train, val_pos, test_pos = leave_one_out_split(pos, min_interactions=3)
    log.info(f"  train={len(train):,} rows | {train['user_id'].nunique():,} users")
    log.info(f"  val  ={len(val_pos):,} rows  | {val_pos['user_id'].nunique():,} users")
    log.info(f"  test ={len(test_pos):,} rows  | {test_pos['user_id'].nunique():,} users")

    # ── 6. Freeze 99 negatives per val + test user ───────────────────────────
    all_items = sorted(valid_items)
    log.info(f"Sampling 99 negatives per val  user (seed={SEED + 1})…")
    val_with_neg  = sample_negatives(val_pos,  train, all_items, n_neg=99, seed=SEED + 1)
    log.info(f"Sampling 99 negatives per test user (seed={SEED})…")
    test_with_neg = sample_negatives(test_pos, train, all_items, n_neg=99, seed=SEED)
    log.info("  Negatives sampled.")

    # ── 7. Write parquet files ───────────────────────────────────────────────
    log.info(f"Writing output to {OUT}/…")
    train.to_parquet(OUT / "train.parquet", index=False)
    val_with_neg.to_parquet(OUT / "val.parquet", index=False)
    test_with_neg.to_parquet(OUT / "test.parquet", index=False)
    items_meta.to_parquet(OUT / "items_metadata.parquet", index=False)
    log.info("  train.parquet       written")
    log.info("  val.parquet         written")
    log.info("  test.parquet        written")
    log.info("  items_metadata.parquet written")

    # ── 8. Write frozen JSON snapshot for audit / reproducibility ────────────
    log.info("Writing eval_negatives.json snapshot…")
    snapshot = {
        "seed_test": SEED,
        "seed_val":  SEED + 1,
        "n_neg":     99,
        "test": test_with_neg.to_dict(orient="records"),
        "val":  val_with_neg.to_dict(orient="records"),
    }
    with open(OUT / "eval_negatives.json", "w") as f:
        json.dump(snapshot, f)
    log.info("  eval_negatives.json written")

    # ── 9. Final summary ─────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("prepare_data.py  COMPLETE")
    log.info("=" * 55)
    log.info(f"  Items with metadata : {len(items_meta):,}")
    log.info(f"  Train interactions  : {len(train):,}")
    log.info(f"  Val users           : {val_with_neg['user_id'].nunique():,}  (each: 1 pos + 99 neg)")
    log.info(f"  Test users          : {test_with_neg['user_id'].nunique():,}  (each: 1 pos + 99 neg)")
    log.info(f"  Output dir          : {OUT.resolve()}")


if __name__ == "__main__":
    main()
